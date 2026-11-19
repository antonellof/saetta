// Qwen3.8-Flash-Next forward pass, one token per step.
// The whole token is one serial Metal command buffer: quantized matvecs
// (src/kernels/qmatvec.metal) and the small ops (src/kernels/decode.metal) run on the
// GPU in order. The host writes only what it knows before encoding: the embedding row,
// the token's n-gram rows (PLE) and the RoPE angles in double precision. Past 2,048
// tokens the QSA block selection runs on the host, which flushes the command buffer
// once per attention layer. Formulas follow
// vendor/transformers/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py and are
// checked against a numpy reference by `saetta verify`.
//
// Public API used by main, verify and the server: Model (load, step, prefill,
// prefill_with, verify_chunk, spec_greedy, spec_verify, draft_mtp), Session (create,
// conversations, restore points, prefix points), model_argmax and the profiler
// (model_prof_*). The other types are the model's internals, visible because Model holds them.
#ifndef SAETTA_MODEL_H
#define SAETTA_MODEL_H

#include "base.h"
#include "config.h"
#include "gpu.h"
#include "metal/shim.h"
#include "safetensors.h"
#include "weights.h"

// Width of the hyper-connection streams: hc_count x hidden_size (10240).
#define HCW (CONFIG_HC_COUNT * CONFIG_HIDDEN_SIZE)
// Prefill chunk: tokens per pass.
#define CH 4096
// Most tokens a speculative verification chunk carries (the drafted ones plus the last).
#define KMAX 8
// Prefix points: the recurrent state every PT_EVERY tokens of a prompt, at most PT_MAX.
#define PT_EVERY 4096
#define PT_MAX 8

// n-gram heads of the PLE layer (16), and chunks of at most MODEL_SMALL tokens use the
// decode-style kernels.
#define MODEL_NG_HEADS ((CONFIG_NGRAM_SIZE - 1) * CONFIG_HEADS_PER_NGRAM)
#define MODEL_SMALL 8

// ---- internals ---------------------------------------------------------------------------

// An f32 parameter (norm weight, conv weight, ...) in the GPU parameter buffer: float offset
// off, n values at s.
typedef struct {
    size_t off;
    float *s;
    size_t n;
} Vec;

typedef struct {
    sx_buffer *buf;
    size_t used;
} Params;

typedef struct {
    Vec norm;
    Linear down, up;
    bool has_inject;
    Linear inject;
} Residual;

typedef struct {
    size_t slot;
    Linear qkv, z, a, b, out;
    Vec conv; // [CONV][KCONV]
    Vec a_log, dt_bias;
    Vec norm; // [DV], plain weight
} Gdn;

typedef struct {
    size_t slot;
    Linear q, k, v, o;
    Vec q_norm, k_norm;
    Linear ix;
    Vec ix_qn, ix_kn;
} Attn;

typedef struct {
    // experts of this layer (the router's rows): 512 in the full model, 256 in the pruned
    // Coder trunk, 512 in its MTP head; buffers are sized for num_experts (the most)
    uint32_t ne;
    Linear router, shared_gate, sg, su, sd, eg, eu, ed;
} Moe;

typedef struct {
    Linear key, value;
    Vec norm_key, norm_query, norm_conv;
    Vec conv; // [HCW][4]
    int64_t mult[CONFIG_NGRAM_SIZE];
    int64_t offsets[MODEL_NG_HEADS];
    int64_t vocab[MODEL_NG_HEADS];
    TensorInfo shard_w[128], shard_s[128], shard_b[128];
} Ple;

typedef struct {
    Residual attn_hc, mlp_hc;
    Gdn *gdn;   // NULL unless a GDN layer
    Attn *attn; // NULL unless an attention layer
    Moe moe;
    Ple *ple;   // NULL unless the layer carries the n-gram embedding
} Layer;

// The MTP draft head (wiring as in sushi src/qwen4/qwen4_forward.zig:1220-1308, MIT): row r
// pairs the trunk's last-layer streams at position r (before the mixer) with the embedding of
// token r+1 and runs at position r+1:
//   X_k = fc_hidden(rms_10240(H) * (1 + n_h))_k + fc_embedding(rms(embed(tok)) * (1 + n_e)), k < 4,
// then one QSA + MoE layer with its own KV slot, its own mixer and the trunk's lm_head. The
// pack keeps its dense matrices in BF16; they are requantized at load to 4-bit, group 32
// (routers 8-bit).
typedef struct {
    Linear fc_e, fc_h;
    Vec n_e, n_h;
    Layer layer;
    Residual mixer;
} Mtp;

// The templated pipelines: field name, kernel name (in this order; the profiler reports them so).
#define MODEL_PIPES(X)                                                                          \
    X(q4, "qmatvec_4_32") X(q8, "qmatvec_8_64") X(qmv4_l8, "qmv4_l8") X(qmv4_l256, "qmv4_l256")    \
    X(qmv_multi_l8, "qmv_multi_l8") X(qmv_multi_l32, "qmv_multi_l32")                            \
    X(qmv_multi_l256, "qmv_multi_l256") X(qmv_gate_up, "qmv_gate_up") X(qmv_up_mix, "qmv_up_mix") \
    X(qmv_gate_up_sh, "qmv_gate_up_sh") X(qmv8_router_sg, "qmv8_router_sg")                      \
    X(gdn_conv_norm, "gdn_conv_norm") X(attn_prep_dec, "attn_prep_dec") X(mv_bf16, "mv_bf16")      \
    X(ix_score_sg, "ix_score_sg") X(ix_select_rx, "ix_select_rx") X(qmv_down_sh, "qmv_down_sh")    \
    X(add_streams, "add_streams") X(bcast_streams, "bcast_streams") X(hc_add_norm, "hc_add_norm")  \
    X(qe4, "qmatvec_expert_4_32") X(rmsnorm, "rmsnorm") X(silu_scale, "silu_scale")                \
    X(hc_mix, "hc_mix") X(hc_add, "hc_add") X(gdn_conv, "gdn_conv") X(gdn_qknorm, "gdn_qknorm")    \
    X(gdn_step, "gdn_step") X(gdn_gnorm, "gdn_gnorm") X(rope, "rope") X(copy_f32, "copy_f32")      \
    X(attn_decode, "attn_decode") X(moe_topk, "moe_topk") X(swiglu, "swiglu")                      \
    X(moe_combine, "moe_combine") X(ple_gate, "ple_gate") X(ple_conv, "ple_conv")                  \
    X(copy_rows, "copy_rows") X(ix_block, "ix_block") X(ix_score, "ix_score")                      \
    X(ix_select, "ix_select") X(ix_score_t, "ix_score_t") X(ix_select_t, "ix_select_t")            \
    X(moe_group, "moe_group") X(gather_rows, "gather_rows") X(gather_rows_h, "gather_rows_h")      \
    X(swiglu_h, "swiglu_h") X(moe_combine_g, "moe_combine_g") X(mmg4, "qmm_grouped_4_32")         \
    X(attn_split, "attn_split") X(attn_split_g, "attn_split_g") X(attn_split_gh, "attn_split_gh")  \
    X(copy_rows_h, "copy_rows_h") X(attn_merge, "attn_merge") X(mm4, "qmm_4_32") X(mm8, "qmm_8_64")

typedef struct {
#define MODEL_PIPE_FIELD(f, k) sx_pipeline *f;
    MODEL_PIPES(MODEL_PIPE_FIELD)
#undef MODEL_PIPE_FIELD
} Pipes;

// Prefill matmuls on the neural accelerators (half inputs): src/kernels/qmm_mpp.metal.
// qsa: QSA attention for prompt chunks on the accelerators (f16 KV), NULL if it does not compile.
// dq / mmh: dense prompt matmuls as unpack-to-f16 then an f16 GEMM (SAETTA_DENSE_FUSED=1 keeps
// the fused tile-dequant kernel); NULL if they do not compile.
typedef struct {
    sx_pipeline *dense, *grouped, *to_half, *qsa, *dq, *mmh, *dq_bf16;
} Mpp;

// Run-time (bits, group) kernels (src/qgen.h) for tensors outside the 4/32 and 8/64 fast
// paths (the Coder pack mixes 4/5/6/8 bits and groups 32/64/128); compiled only when needed.
typedef struct {
    sx_pipeline *mv, *expert;
    sx_pipeline *multi[3]; // lanes 8, 32, 256
    sx_pipeline *gate_up, *gate_up_sh, *down_sh, *up_mix;
    sx_pipeline *rows1[MODEL_SMALL + 1];
    sx_pipeline *rows[MODEL_SMALL + 1];
    sx_pipeline *qmm, *qmm_grouped;
    sx_pipeline *dq, *qmmh, *qmmh_g; // NULL when they do not compile
} Gen;

typedef struct Model Model;
typedef struct Session Session;

// Encoder for one token: dispatches go into one serial command buffer.
typedef struct {
    Model *m;
    sx_cmd *cmd;
    bool failed;
    // B.xh holds the half copy of this (buffer, offset, count); any dispatch other than an
    // accelerator matmul clears it, since it may have rewritten the source.
    bool xh_set;
    sx_buffer *xh_buf;
    size_t xh_off, xh_n;
} Enc;

// A verification chunk encoded during the last draft step.
typedef struct {
    Enc e;
    size_t T, pos;
    const Session *s;
} PendVerify;

struct Model {
    Gen *gen; // NULL unless the pack mixes quantizations
    const Pack *pack;
    Gpu *g;
    Pipes p;
    Mpp *mpp; // NULL when the accelerator kernels do not compile
    // qmv_rows kernels per chunk size 1..SMALL: [0] 4-bit group 32, [1] 8-bit group 64; rows1_p
    // one row per simdgroup, rows_p 4 rows per simdgroup sharing the inputs
    sx_pipeline *rows_p[2][MODEL_SMALL + 1];
    sx_pipeline *rows1_p[2][MODEL_SMALL + 1];
    // Prefill 4-bit matmuls on the accelerators in half precision (8-bit router and gates stay
    // f32). Off with SAETTA_PREFILL_F32=1 (exact f32 path) or when the kernels do not compile.
    bool fast_prefill;
    Layer *layers; // num_hidden_layers
    Residual mixer;
    Linear embed, lm_head;
    Params params;
    sx_buffer *scratch, *bscratch, *logits;
    sx_buffer *vlogits; // [KMAX][vocab], verification chunks
    // MTP head (NULL when the pack has none or SAETTA_MTP=0).
    Mtp *mtp;
    sx_buffer *mtp_hid;    // [CH + 1][HCW] trunk streams before the mixer, rows paired with the next tokens
    sx_buffer *mtp_chain;  // [HCW] the head's own stream, fed to the next draft step
    sx_buffer *mtp_logits; // [vocab]
    // Prefill inputs of a chunk, prepared on the host (by a thread for the next chunk while
    // the GPU runs the current one): [CH][H] embeddings, then [CH][ple_embed_dim] n-gram rows.
    sx_buffer *stage[2];
    // A verification chunk encoded while the MTP head's last draft step ran on the GPU
    // (draft_mtp), committed by verify_chunk once the drafts are known; any other call drops it.
    bool has_pend;
    PendVerify pend;
};

// Per-conversation restore point.
typedef struct {
    sx_buffer *gdn, *conv, *ple;
    uint32_t prev[2];
    size_t since_eos;
    VecU32 tokens;
} Ckpt;

// A prefix point; absent while gdn is NULL.
typedef struct {
    sx_buffer *gdn, *conv, *ple;
    uint32_t prev[2];
    size_t since_eos;
} Point;

// The per-conversation part of a session: KV cache, indexer keys, pooled block keys and the
// restore points. Every request restores a restore point or resets, so the recurrent state,
// speculation slots and scratch buffers are shared by all conversations, and a server keeps
// several conversations by swapping these in (session_swap_conv).
typedef struct {
    size_t cap;
    sx_buffer *kv, *ixk, *bk;
    Ckpt ck[2];
    Point pts[PT_MAX];
    VecU32 ptoks;
} Conv;

#define SESSION_NO_ROLL ((size_t)-1)

// Per-session state, in GPU buffers.
struct Session {
    size_t pos;
    // Capacity of the active conversation's buffers (tokens; their row stride); grows up to
    // max_cap (SAETTA_CTX), which sizes the shared RoPE table and score scratch.
    size_t cap, max_cap;
    // KV cache in f16 (default) or f32 (exact path: SAETTA_KV_F32=1, and verify's exact checks).
    bool kv_half;
    sx_buffer *gdn_state; // [n_gdn][NV][DV][DK] (column-major per head)
    sx_buffer *gdn_conv;  // [n_gdn][3][CONV]
    sx_buffer *kv;        // [n_attn + 1][2][cap][NKV*HD], f16 or f32; slot n_attn is the MTP head
    sx_buffer *ixk;       // [n_attn][cap][IXD], raw indexer keys
    sx_buffer *ple_hist;  // [9][HCW]
    sx_buffer *rope;      // [cap][ROT/2] (cos, sin), from double precision
    sx_buffer *bk;        // [n_attn][cap/BLOCK][IXD], pooled indexer keys
    sx_buffer *scores;    // [AB][cap/BLOCK] (row 0 in decode)
    uint32_t prev[2];
    size_t since_eos;
    // When set, model_step copies the residual streams after each layer here
    // ([layers][HCW], trace_len floats); used by verify.
    float *trace;
    size_t trace_len;
    sx_buffer *trace_buf;
    // n-gram row ids of the current chunk, and of the next one (computed on a thread).
    uint64_t (*ple_rows)[MODEL_NG_HEADS];
    uint64_t (*ple_rows_b)[MODEL_NG_HEADS];
    bool ids_ready;
    size_t stage_cur; // Model.stage buffer of the current chunk
    // Speculative verification: when spec is set, chunks also save the recurrent state after
    // each token (slot t), so a rejected draft rolls back without a replay.
    bool spec;
    sx_buffer *sp_gdn;  // [KMAX][n_gdn][NV][DV][DK]
    sx_buffer *sp_conv; // [KMAX][n_gdn][3][CONV]
    sx_buffer *sp_ple;  // [KMAX][9][HCW]
    // After a partly accepted verify chunk: the slot holding the live recurrent state (the
    // next pass reads its start state there); SESSION_NO_ROLL when it is in gdn_state /
    // gdn_conv / ple_hist.
    size_t roll;
    // Restore points: [0] one token before a prompt's end, [1] after a reply was generated (so
    // the next turn, whose prompt repeats the reply, re-reads only what is new: sushi's
    // decode-end point). session_restore takes the longest that is a prefix of the new prompt.
    Ckpt ck[2];
    size_t ck_pick;
    // MTP: mtp_hid row 0 holds the trunk stream before pos (mtp_ok); mtp_n verified rows are
    // pending, paired with mtp_tok and then the next token (model_draft_mtp).
    bool mtp_ok;
    size_t mtp_n;
    uint32_t mtp_tok[KMAX];
    // Prefix points: the recurrent state every PT_EVERY tokens of the prompt prefix ptoks
    // (point k at position (k+1) * PT_EVERY), so another conversation that shares the prefix
    // starts there after copying the KV rows.
    Point pts[PT_MAX];
    VecU32 ptoks;
    // The last verification chunk: its start position and length; its per-token states stay
    // in the speculation slots until the next chunk (session_roll_to).
    size_t chunk_pos0, chunk_t;
};

// ---- Model -------------------------------------------------------------------------------

// Compiles the kernels, maps the weights (norms and conv weights converted to f32 in a GPU
// buffer, the MTP head requantized), allocates the scratch buffers. Errors: "CompileFailed",
// "AllocFailed", "MissingTensor", "UnexpectedDType", "TooManyExperts", "ParamsFull",
// "RequantBufferFull", pack_linear errors.
Err model_load(Arena *a, const Pack *pack, Gpu *g, Model *out);

// Runs one token at position s->pos; *logits gets the logits ([vocab], valid until the next
// call). Errors: "ContextFull", "GpuFailed".
Err model_step(Model *m, Session *s, uint32_t token, float **logits);

// Runs the prompt tokens in chunks of CH and gives the logits of the last one.
// trace (verify only, one chunk): residual streams [layers][T][HCW] after each layer.
Err model_prefill(Model *m, Session *s, const uint32_t *tokens, size_t n, sx_buffer *trace, float **logits);

// Hook of model_prefill_with: chunk_done(ctx, s->pos) runs after each chunk (the GPU is done
// with it); when aligned(ctx) is true the chunks of the first PT_MAX * PT_EVERY positions end
// on multiples of PT_EVERY (prefix points) when the prompt starts on one. Either may be NULL.
typedef struct {
    void *ctx;
    bool (*aligned)(void *ctx);
    Err (*chunk_done)(void *ctx, size_t pos);
} PrefillHook;

// model_prefill with a hook. One call keeps the n-gram preparation of the next chunk
// overlapped with the GPU: separate calls would each pay it (about 200 ms per call).
Err model_prefill_with(Model *m, Session *s, const uint32_t *tokens, size_t n, sx_buffer *trace,
                       const PrefillHook *hook, float **logits);

// Runs toks (1 to KMAX) as one chunk; *logits gets the logits of every position,
// [n][vocab] (speculative verification; valid until the next call).
Err model_verify_chunk(Model *m, Session *s, const uint32_t *toks, size_t n, float **logits);

typedef struct {
    size_t accepted;
    uint32_t next;
} SpecResult;

// Speculative greedy step. x is the next token to consume, drafts the nd guessed tokens after
// it (1 to KMAX-1). Runs them as one chunk, accepts the longest prefix of drafts that greedy
// decoding agrees with, rolls the recurrent state back to just after the accepted tokens
// (from the per-token slots the chunk saved), and returns how many drafts were accepted and
// the greedy token after them.
Err model_spec_greedy(Model *m, Session *s, uint32_t x, const uint32_t *drafts, size_t nd, SpecResult *out);

// Token rule of model_spec_verify: the token at a verified position from its logits row
// ([vocab]), given the draft there (has_draft false after the last draft).
typedef uint32_t (*SpecPick)(void *ctx, const float *row, bool has_draft, uint32_t draft);

// Speculative step with a token rule: a draft is accepted while the rule returns it. Greedy is
// argmax; sampling accepts a draft with its target probability and otherwise draws from the
// rest (speculative sampling: the output follows the target distribution exactly, since a
// draft is one token with probability 1).
Err model_spec_verify(Model *m, Session *s, uint32_t x, const uint32_t *drafts, size_t nd, SpecPick pick,
                      void *ctx, SpecResult *out);

// Drafts up to k tokens after x (the next token to consume, at s->pos) with the MTP head into
// out (room for k). First the pending verified rows (mtp_hid rows 0..mtp_n, paired with
// mtp_tok and then x) run at positions s->pos - mtp_n + 1 .. s->pos, rewriting the head's KV
// there with the trunk's true streams; the last row gives draft 1. Each further draft feeds
// the head's own stream and the previous draft. *nd gets the number of drafts (0 without a head).
Err model_draft_mtp(Model *m, Session *s, uint32_t x, size_t k, uint32_t *out, size_t *nd);

// Index of the first largest value of v[0..n) (as a scalar `x > best` scan), vectorized.
uint32_t model_argmax(const float *v, size_t n);

// SAETTA_PROFILE=1: every dispatch in its own command buffer, GPU time summed per kernel.
void model_prof_init(void);
// Drops what was counted so far (e.g. the prompt, before measuring decode alone).
void model_prof_reset(void);
// Prints the table to stderr (per token over `tokens`), when on.
void model_prof_report(size_t tokens);

// ---- Session -----------------------------------------------------------------------------

// Context capacity from $SAETTA_CTX (tokens, default 32768); KV f16 unless SAETTA_KV_F32.
// Errors: "AllocFailed", "InvalidCharacter" / "Overflow" (bad SAETTA_CTX).
Err session_create(Gpu *g, Session **out);
Err session_create_kv(Gpu *g, bool kv_half, Session **out);
// A session whose conversation buffers hold cap tokens, growable (session_grow) up to
// max_cap; the RoPE table and the score scratch are sized for max_cap.
Err session_create_cap(Gpu *g, bool kv_half, size_t cap, size_t max_cap, Session **out);

size_t session_kv_elem(const Session *s); // bytes per KV element: 2 (f16) or 4
// Bytes of the conversation buffers (KV, indexer keys, block keys) at capacity cap.
size_t session_kv_bytes(const Session *s, size_t cap);
// Reallocates the active conversation's buffers at capacity cap (between requests), keeping
// its rows; the old buffers are released. cap >= s->cap, <= max_cap, a multiple of 4.
Err session_grow(Session *s, Gpu *g, size_t cap);

// Saves the recurrent state after the tokens consumed so far (which are toks[0..n), n ==
// s->pos) in restore point 0 (prompt), or slot (0: prompt, 1: end of a reply).
Err session_save_checkpoint(Session *s, const uint32_t *toks, size_t n);
Err session_save_checkpoint_at(Session *s, size_t slot, const uint32_t *toks, size_t n);

// Tokens of prompt covered by a restore point (0 if none is a prefix), leaving at least one
// token to run so the request gets its logits; the longest one wins and is the one
// session_restore uses.
size_t session_reusable(Session *s, const uint32_t *prompt, size_t n);

typedef struct {
    size_t n, pick;
} Reusable;
// session_reusable for the restore points of a conversation, active or not.
Reusable session_reusable_in(const Ckpt ck[2], const uint32_t *prompt, size_t n);

// Bytes of one more conversation at capacity cap.
size_t session_conv_bytes(const Session *s, size_t cap);
// Allocates the buffers of another conversation at capacity cap, empty. Errors: "AllocFailed".
Err session_new_conv(const Session *s, Gpu *g, size_t cap, Conv *out);
// Swaps the session's conversation with cv (between requests only).
void session_swap_conv(Session *s, Conv *cv);

// Moves the live state back to just after token p - 1 when that token was consumed by the
// last verification chunk (a reply that ended with accepted drafts still queued): the state
// is that chunk's slot, and the n-gram history is recomputed from toks (the consumed tokens,
// at least p). Returns false when p is outside that chunk.
bool session_roll_to(Session *s, size_t p, const uint32_t *toks);

// Keeps the prefix points within the first n tokens.
void session_keep_points(Session *s, size_t n);
// Saves prefix point k (at position (k+1) * PT_EVERY == s->pos), allocating it on first use;
// the prefix tokens are toks[0..n) (n == s->pos).
Err session_save_point(Session *s, Gpu *g, const uint32_t *toks, size_t n);
// The prefix point usable for prompt: its position (a multiple of PT_EVERY, shorter than the
// prompt), 0 if none.
size_t session_shared_point(const uint32_t *ptoks, size_t nptoks, const Point pts[PT_MAX], const uint32_t *prompt,
                            size_t n);
// Starts at this conversation's own prefix point at position n (its KV rows below n are in place).
void session_start_at_own_point(Session *s, size_t n);
// Before rows from position n on are rewritten: drops the restore points and prefix points
// that reach past n (their state would no longer match the KV cache).
void session_invalidate_from(Session *s, size_t n);
// Starts this (reset) session at position n of a prefix it shares with src: copies the KV,
// indexer-key and block-key rows below n and the recurrent state of src's prefix point there,
// and adopts src's points up to n by copying them.
Err session_start_from(Session *s, Gpu *g, const Conv *src, size_t n);
// Back to the restore point session_reusable picked.
void session_restore(Session *s);
// The n-gram token history after consuming token.
void session_push_history(Session *s, uint32_t token);
// Back to an empty context, keeping the allocations.
void session_reset(Session *s);

#endif
