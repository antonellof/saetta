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
#include "state_store.h"
#include "weights.h"

// Prefill chunk: tokens per pass.
#define CH 4096
// Most tokens a speculative verification chunk carries (the drafted ones plus the last).
#define KMAX 8
// Checkpoints at message boundaries (prefix points): the recurrent state at chosen positions of a
// prompt, at most PT_MAX per conversation: each request's start, and the positions where the
// prompt's prefill ends a chunk for one (session_point_cuts picks them from the positions of the
// <|im_start|> id and every PT_EVERY tokens, at least PT_GAP apart and from the end); past PT_MAX
// the older ones are thinned to a geometric spacing (session_point_slot), the PT_RECENT most recent
// and the oldest kept.
#define PT_EVERY 4096
#define PT_MAX 16
#define PT_GAP 1024
#define PT_RECENT 4
// Checkpoints an inactive conversation keeps (session_conv_trim_points: its most recent, its last
// request's start, its oldest), so every conversation's checkpoints fit in what the context fit
// reserves (session_points_reserve).
#define PT_HELD 3

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
    // the MTP drafter's policy in the decode loops (main.c, server.c): with mtp_no_lookup it
    // drafts only when prompt lookup offers nothing (else when lookup offers under 3 tokens), at
    // most mtp_kmax tokens (0: KMAX - 1), from the controller's starting costs depth_ms (ms of a
    // plain step and of rounds of 1..4 drafts; NULL: DEPTH_INIT's, measured on Qwen)
    bool mtp_no_lookup;
    uint8_t mtp_kmax;
    const float *depth_ms;
    ChatFormat chat; // the chat template the model was trained with
} ModelInfo;

// Per-conversation restore point. next: the token after tokens (its MTP row at tokens.len pairs
// it), when has_next (the server sets it after a save; a save clears it): a prompt then reuses the
// point only when its token there is next, as for a checkpoint (session_shared_point).
typedef struct {
    sx_buffer *gdn, *conv, *ple;
    uint32_t prev[2];
    size_t since_eos;
    VecU32 tokens;
    uint32_t next;
    bool has_next;
} Ckpt;

// A checkpoint (prefix point) at position pos; its buffers are absent while gdn is NULL. valid: saved
// for the conversation's current prefix tokens (ptoks, which covers pos); cleared whenever anything
// below pos is read again or replaced (session_keep_points), so a buffer left by an earlier
// occupant of the slot is never used. The slots are in no order.
typedef struct {
    sx_buffer *gdn, *conv, *ple;
    uint32_t prev[2];
    size_t since_eos;
    size_t pos;
    bool valid;
    bool start; // saved at a request's start (the end of the history it continued)
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
    // Checkpoints (prefix points) on the prompt prefix ptoks: a later prompt that leaves this
    // conversation's history inside it (a retry, an edit, a fork) restarts at the nearest one, and
    // another conversation that shares the prefix starts there after copying the KV rows.
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
// with it); a chunk ends at each of the absolute positions cuts[0..ncuts) (increasing; the
// checkpoints), and the tokens between two cuts are read in the fewest chunks of at most CH
// tokens, of equal size (prefill_chunk_len). chunk_done may be NULL, cuts too (ncuts 0).
typedef struct {
    void *ctx;
    const size_t *cuts;
    size_t ncuts;
    Err (*chunk_done)(void *ctx, size_t pos);
} PrefillHook;

// The length of the chunk that starts at offset i of a prefill of n tokens from position pos0
// (hook may be NULL: equal chunks over the whole prompt).
size_t prefill_chunk_len(const PrefillHook *hook, size_t pos0, size_t n, size_t i);

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

// Keeps the prefix points within the first n tokens (the others are no longer valid).
void session_keep_points(Session *s, size_t n);
// Saves a checkpoint at position n == s->pos (the prefix tokens are toks[0..n), and toks[n] the
// next one, which its MTP row pairs: ptoks keeps n + 1 tokens) in the slot
// session_point_slot picks: allocated on first use when may_alloc (the caller's memory check),
// else an empty slot's or a thinned point's buffers; nothing when none is left.
Err session_save_point(Session *s, Gpu *g, const uint32_t *toks, size_t n, bool may_alloc);
// The slot for a new checkpoint at pos (past every valid one): a valid point already at pos; else
// a slot with buffers and no valid point; else, with may_alloc, a slot without buffers; else, when
// some point is valid, the one thinning drops (the oldest and the PT_RECENT most recent kept; of
// the others, the one whose neighbours' gap is smallest against its distance from pos, which
// leaves a geometric spacing); -1 when there is none.
int session_point_slot(const Point pts[PT_MAX], size_t pos, bool may_alloc);
// The slot of the valid point at pos, -1 if none.
int session_point_at(const Point pts[PT_MAX], size_t pos);
// Bytes of one checkpoint's recurrent buffers (118 MB Flash-Next, 157 MB 27B).
size_t session_point_bytes(const Session *s);
// Bytes the checkpoints of convs conversations may hold: PT_MAX for the active one, PT_HELD for
// each of the others (the context fit reserves them).
uint64_t session_points_reserve(const Session *s, size_t convs);
// Keeps keep valid checkpoints of held conversation cv and releases every other slot's buffers
// (valid or not; its restore points stay), choosing in this order: the most recent (a retry), the
// most recent saved at a request's start (where its last request's history ends: another caller
// sharing that history starts there), the oldest (the system prompt's end, which subagents share),
// then the most recent others. Returns the slots released.
size_t session_conv_trim_points(Conv *cv, size_t keep);
// The checkpoint usable for prompt: the highest valid point at a position pos < the prompt's common
// prefix with ptoks (a valid point's state and every KV row below it are this conversation's for
// those tokens, so the points below it need not be valid; the token at pos matches too, the one
// its MTP row pairs): its position (shorter than the prompt), 0 if none.
size_t session_shared_point(const uint32_t *ptoks, size_t nptoks, const Point pts[PT_MAX], const uint32_t *prompt,
                            size_t n);

// The checkpoint positions of a prefill from pos0 of the prompt ids[0..n) (which reads it to n - 1,
// where the prompt's restore point is saved), increasing, into out (room for cap): the candidates
// are the message starts (positions of im_start; none when it is UINT32_MAX) and the multiples of
// PT_EVERY in (pos0, n - 1); one is taken when it is at least PT_GAP after the previous one taken
// (pos0 at first) and PT_GAP before n - 1, so no checkpoint makes a chunk shorter than PT_GAP (the
// server keeps one at each request's start, which needs no cut). Returns the count. Depends on ids
// and pos0 alone, so a prompt read from the same start is cut the same way in any process.
size_t session_point_cuts(const uint32_t *ids, size_t n, size_t pos0, uint32_t im_start, size_t *out, size_t cap);
// Starts at this conversation's own checkpoint at position n (its KV rows below n are in place).
void session_start_at_own_point(Session *s, size_t n);
// Before rows from position n on are rewritten: drops the restore points and prefix points
// that reach past n (their state would no longer match the KV cache).
void session_invalidate_from(Session *s, size_t n);
// Starts this (reset) session at position n of a prefix it shares with src: copies the KV,
// indexer-key and block-key rows below n (and the MTP slot's rows at n, which its head reads after
// a start: the rows a state holds, session_state_layout) and the recurrent state of src's
// checkpoint there,
// and adopts src's valid points up to n by copying them (at most alloc_max of them into new buffers:
// the caller's memory check; an allocation that fails leaves none of them valid and returns its
// error). Its own restore points are dropped (they
// were for the rows replaced): a restore point holds only while every row below its length is
// this conversation's computation for its tokens; the other paths that replace rows below one
// drop it as well (session_invalidate_from past its length, session_conv_shrink, a restore
// from the state store, a LoRA switch, which invalidates from 0).
Err session_start_from(Session *s, Gpu *g, const Conv *src, size_t n, size_t alloc_max);
// Back to the restore point session_reusable picked.
void session_restore(Session *s);
// The n-gram token history after consuming token.
void session_push_history(Session *s, uint32_t token);
// Back to an empty context, keeping the allocations.
void session_reset(Session *s);
// The active conversation's row buffers (KV; Flash-Next also its indexer and pooled block keys)
// and restore point slot's recurrent buffers, as a state layout for the state store
// (src/state_store.h), for a state of n tokens: the rec entries also hold the MTP head's rows at
// position n (its K and V rows; Flash-Next also its indexer-key row and the pooled block holding
// n). Flash-Next's first prompt chunk after a restore (mtp_ok false) skips the head's row n and
// then reads it: saved with the state, it leaves a restored conversation's head as the resident
// one's. The 27B's first chunk after a restore rewrites row n and keeps row n - 1 (in the state's
// rows below n), so there the saved row n is only overwritten. slot: 0 and 1 the restore points
// ck[0] and ck[1]; 2 + k the checkpoint pts[k] (at n == pts[k].pos). False when the model keeps no such
// state (DeepSeek-V4-Flash, Inkling 2 Mini).
bool session_state_layout(const Session *s, size_t slot, size_t n, StLayout *out);

#endif
