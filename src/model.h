// Forward pass, one token per step, for the models in this binary: Qwen3.8-Flash-Next
// (src/model.c, internals src/model_flash.h, constants src/config_flash.h) and Qwen3.8-27B
// (src/dense.c, src/dense.h, src/config_27b.h). Each is compiled against its own
// compile-time constants and exports a ModelArch (src/model_arch.h); src/model_api.c picks
// one by the pack's model_type and forwards these calls to it. Flash-Next below:
// The whole token is one serial Metal command buffer: quantized matvecs
// (src/kernels/qmatvec.metal) and the small ops (src/kernels/decode.metal) run on the
// GPU in order. The host writes only what it knows before encoding: the embedding row,
// the token's n-gram rows (PLE) and the RoPE angles in double precision. Past 2,048
// tokens the QSA block selection runs on the host, which flushes the command buffer
// once per attention layer. Formulas follow the transformers qwen4_exp model and are checked
// against a numpy reference by `saetta verify`.
//
// Public API used by main, verify and the server: Model (load, step, prefill,
// prefill_with, verify_chunk, spec_greedy, spec_verify, draft_mtp), Session (create,
// conversations, restore points, prefix points), model_argmax and the profiler
// (model_prof_*). Model is opaque here; Session and its parts have one layout for both models.
#ifndef SAETTA_MODEL_H
#define SAETTA_MODEL_H

#include "base.h"
#include "chat.h"
#include "ctxlimits.h"
#include "gpu.h"
#include "metal/shim.h"
#include "safetensors.h"
#include "weights.h"

// Prefill chunk: tokens per pass.
#define CH 4096
// Most tokens a speculative verification chunk carries (the drafted ones plus the last).
#define KMAX 8
// Prefix points: the recurrent state every PT_EVERY tokens of a prompt, at most PT_MAX.
#define PT_EVERY 4096
#define PT_MAX 8

// Chunks of at most MODEL_SMALL tokens use the decode-style kernels.
#define MODEL_SMALL 8

#define SESSION_NO_ROLL ((size_t)-1)
// n-gram heads of Flash-Next's PLE layer (Session.ple_rows rows).
#define SESSION_NG_HEADS 16

typedef struct Model Model;
typedef struct Session Session;
typedef struct ModelArch ModelArch;

// What the rest of the engine needs to know about the loaded model.
typedef struct {
    const char *model_type; // config.json text_config.model_type
    const char *id;         // served model id (OpenAI "model")
    const char *name;       // display name
    size_t layers, width, vocab, hidden; // width: residual stream floats per token (verify)
    const uint32_t *eos;
    size_t n_eos;
    size_t max_positions; // config.json max_position_embeddings: the largest context
    // what the model implements beyond a plain forward pass: speculative decoding (prompt
    // lookup, verification chunks) and serving (tokenizer, chat template, sessions)
    bool no_spec, no_serve;
    ChatFormat chat; // the chat template the model was trained with
} ModelInfo;

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

// Per-session state, in GPU buffers. One layout for both models (Flash-Next uses every field;
// 27B leaves the indexer and PLE fields unset).
struct Session {
    const ModelArch *arch; // the model that created it (session_* dispatch on it)
    size_t pos;
    // Capacity of the active conversation's buffers (tokens; their row stride); grows up to
    // max_cap (SAETTA_CTX), which sizes the shared RoPE table and score scratch.
    size_t cap, max_cap;
    // KV cache in f16 (default) or f32 (exact path: SAETTA_KV_F32=1, and verify's exact checks).
    bool kv_half;
    sx_buffer *gdn_state; // [n_gdn][NV][DV][DK] (column-major per head)
    sx_buffer *gdn_conv;  // [n_gdn][3][CONV]
    sx_buffer *kv;        // [n_attn + 1][2][cap][NKV*HD], f16 or f32; slot n_attn is the MTP head
                          // (27B: only when its head is loaded)
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
    uint64_t (*ple_rows)[SESSION_NG_HEADS];
    uint64_t (*ple_rows_b)[SESSION_NG_HEADS];
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
// The model is chosen by the pack's model_type ("UnknownModelType" if neither); *out is
// allocated in the arena.
Err model_load(Arena *a, const Pack *pack, Gpu *g, Model **out);

// The loaded model's facts; for a pack before loading, model_info_for (NULL if unknown).
const ModelInfo *model_info(const Model *m);
const ModelInfo *model_info_for(Arena *a, const Pack *pack);
// Same from the pack folder alone (its config.json), without opening the weights.
const ModelInfo *model_info_dir(Arena *a, const char *dir);
// `saetta info`: every linear of the pack against the shapes the model expects, problems
// printed to out; returns their count (the pack's model must be known).
size_t model_check_pack(Arena *a, const Pack *pack, FILE *out);
// Whether the model drafts with an MTP head (model_draft_mtp gives drafts).
bool model_has_mtp(const Model *m);
// Prompt matmuls on the accelerators in half precision (verify turns them off for its exact
// checks); model_fast_prefill_available: whether the kernels compiled.
bool model_fast_prefill(const Model *m);
bool model_fast_prefill_available(const Model *m);
void model_set_fast_prefill(Model *m, bool on);

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

// LoRA adapters applied at run time (27B build; the Flash-Next build returns
// "LoraUnsupported"). model_lora_load reads a PEFT adapter directory (adapter_config.json,
// adapter_model.safetensors) as adapter `name`. Errors: "LoraUnsupported", "TooManyAdapters",
// "BadAdapterConfig", "LoraTargetUnsupported", "LoraShapeMismatch", "LoraUnmatchedTensor",
// "MapFailed", safetensors_parse errors, "AllocFailed".
Err model_lora_load(Arena *a, Model *m, const char *name, const char *dir);
// Index of the adapter called name, or -1.
int model_lora_find(const Model *m, const char *name);
// The adapter the following passes apply: an index from model_lora_find, or -1 for none.
void model_lora_select(Model *m, int idx);
// The selected adapter (-1: none), the count and the names of the loaded ones.
int model_lora_selected(const Model *m);
size_t model_lora_count(const Model *m);
const char *model_lora_name(const Model *m, size_t i);

// Index of the first largest value of v[0..n) (as a scalar `x > best` scan), vectorized.
uint32_t model_argmax(const float *v, size_t n);

// SAETTA_PROFILE=1: every dispatch in its own command buffer, GPU time summed per kernel.
void model_prof_init(void);
// Drops what was counted so far (e.g. the prompt, before measuring decode alone).
void model_prof_reset(void);
// Prints the table to stderr (per token over `tokens`), when on.
void model_prof_report(size_t tokens);

// ---- Session -----------------------------------------------------------------------------

// A context capacity (tokens) from the text of $SAETTA_CTX or --ctx (NULL: the default,
// SESSION_CTX_DEFAULT), rounded down to a multiple of SESSION_CTX_ALIGN (src/ctxlimits.h).
// Errors: "InvalidCharacter" / "Overflow" (not a number), "ContextTooSmall" (below
// SESSION_CTX_MIN), "ContextTooLarge" (above max, the model's max_position_embeddings).
// NotEnoughMemory (after a message) when the pack does not fit this Mac's memory (fit.h); OK for
// packs that stream their experts and for an empty pack. gpu_init calls it before touching the pack.
Err model_fit_check(Arena *a, const Pack *pack);

Err session_ctx_parse(const char *text, size_t max, size_t *out);

// Context capacity from $SAETTA_CTX (session_ctx_parse); KV f16 unless SAETTA_KV_F32.
// Errors: "AllocFailed", session_ctx_parse's (bad SAETTA_CTX).
Err session_create(const Model *m, Gpu *g, Session **out);
Err session_create_kv(const Model *m, Gpu *g, bool kv_half, Session **out);
// A session whose conversation buffers hold cap tokens, growable (session_grow) up to
// max_cap; the RoPE table and the score scratch are sized for max_cap.
Err session_create_cap(const Model *m, Gpu *g, bool kv_half, size_t cap, size_t max_cap, Session **out);

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
// Empties held conversation cv: its buffers reallocated at capacity cap, its restore points and
// prefix tokens dropped. Errors: "AllocFailed" (cv unchanged).
Err session_conv_shrink(const Session *s, Gpu *g, Conv *cv, size_t cap);
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
