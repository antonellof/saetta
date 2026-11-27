// HTTP server: OpenAI POST /v1/chat/completions and Anthropic POST /v1/messages (JSON or SSE,
// one generation path; src/anthropic.h maps the request), POST /v1/messages/count_tokens,
// GET /v1/models, GET /health. One request at a time, one session reused across requests.
// Streams never go quiet for long: a long prompt is read in slices with a keep-alive between
// them, and a keep-alive goes out during decode when nothing could be sent for a while (a
// tool call is held back until it is complete); Claude Code aborts a silent stream.
// Streamed and non-streamed answers carry the same bytes: content and reasoning are
// emitted only once no later token can change them (trailing whitespace and partial
// tags are held back), and tool calls are parsed from the whole answer at the end.
#ifndef SAETTA_SERVER_H
#define SAETTA_SERVER_H

#include "base.h"
#include "json.h"
#include "qgen.h"
#include "spec.h"

// Loads the pack in dir and serves on 127.0.0.1:port until the process ends. Errors:
// "AlreadyRunning" (another server holds the lock; message printed), load errors, socket
// errors.
// loras: nloras "name=path" LoRA adapters (--lora-modules); path a folder, a download name
// (27b-pi) or a Hugging Face id under the models folder (download_lora_dir). Each is checked
// against the base model (download_lora_check, then model_lora_load): "LoraDoesNotFit" or the
// load's error, the reason printed. A request whose model is an adapter's name runs with it.
// kwargs_default: NULL or a JSON object of chat template kwargs (enable_thinking, ...) used when
// a request does not set them (vLLM's --default-chat-template-kwargs).
Err server_serve(Arena *arena, const char *dir, uint16_t port, const char *const *loras, size_t nloras,
                 const char *kwargs_default);

// ---- in process (Saetta Code): the server's engine and request path without a socket --------

typedef struct Engine Engine;
// Loads the pack in dir as server_serve does (model, its LoRA adapters, tokenizer, conversations
// sized to this Mac's memory), without a socket, the instance lock or the stats thread. loras,
// nloras and kwargs_default as for server_serve.
Err server_engine_open(Arena *arena, const char *dir, const char *const *loras, size_t nloras, const char *kwargs_default,
                       Engine **out);
// The model id the engine answers as, and its context in tokens.
const char *server_engine_model(const Engine *e);
size_t server_engine_context(const Engine *e);
// The engine's live statistics (stats.h; Saetta Code's /monitor reads them with stats_json).
struct Stats *server_engine_stats(Engine *e);
// Conversation state on the SSD for this engine (src/state_store.h), under root (Saetta Code:
// ~/.saetta/state): a request whose prompt extends a saved state reads it instead of the
// history. version: the engine's (states of another are not read). Errors: "StateUnsupported"
// (the model keeps no such state: Inkling 2 Mini, DeepSeek-V4-Flash), "StateDirFailed",
// "StateWriterFailed" (no writer thread; the engine runs without the store).
Err server_engine_state_enable(Engine *e, const char *root, const char *version);
// Saves, in the background, the state of the conversation the calling thread's last request ran
// in (Saetta Code: at the end of each agent run), unless another request has used that
// conversation since or it runs with a LoRA adapter. It waits for the engine's turn (other threads'
// requests); nothing runs on the token path.
void server_engine_state_save(Engine *e);
// Waits for a save in progress (before the process exits).
void server_engine_state_flush(Engine *e);
// For tests: conversation i's speculation controller (one per conversation); false for a
// conversation not open.
bool server_engine_controller(const Engine *e, size_t i, Gate *gate, Depth *depth);
// For tests: checkpoint slots holding buffers over every conversation (*active: the active one's),
// and the bytes the context fit reserved for checkpoints.
size_t server_engine_point_slots(const Engine *e, size_t *active, uint64_t *reserved);
// One POST /v1/chat/completions with this body, on the caller's thread: the response body
// bytes (the SSE stream, or the JSON answer or error) go to sink as they are produced, with the
// response status; sink returns nonzero to stop (the request ends as when a client leaves), and
// is also called with p NULL, n 0 to ask whether the caller is gone. Errors: harness failures
// only (a failed request is an error answer through sink).
Err server_engine_chat(Engine *e, Str body, int (*sink)(void *ctx, int status, const void *p, size_t n), void *ctx);
// server_engine_chat for an agent session (Saetta Code's session id; NULL or "": none): a prompt
// that no restore point continues restarts in place, at a checkpoint of the conversation this
// thread's last request of the same session used; a request of another session (or none) gets
// another conversation, copying rows, and never overwrites that one.
Err server_engine_chat_session(Engine *e, Str body, const char *session,
                               int (*sink)(void *ctx, int status, const void *p, size_t n), void *ctx);
// Prefill ahead of the request (Saetta Code in process): body is the next request of agent session
// session as far as it is known, its tool message with tool_call_id partial_id still growing (a
// running tool's output so far). The tokens of the prompt that no continuation of that message
// changes, less a margin, are read into the conversation this session's last request ran in, after
// its last reply (nothing when another request used that conversation since, when the prompt does
// not continue that reply, with a LoRA adapter, or when memory would have to be taken from another
// conversation), in chunks, checkpoints at message starts as the request would keep them. It
// yields: before each chunk it stops when another request waits for the engine's turn or *stop is
// set. The request then continues where the read stopped when its prompt starts with the tokens
// read (and from where it would otherwise when not): what it generates is the same. *read: the
// tokens read. Errors: "AheadUnsupported" (Inkling 2 Mini and other chat formats).
Err server_engine_ahead(Engine *e, Str body, const char *session, const char *partial_id, volatile int *stop, size_t *read);
// Speculation fed by the agent (Saetta Code in process): text the agent session has seen, keyed (a
// file's content, by its path; fed again under the same key, the new text replaces the old). Its
// JSON-escaped form (the form an edit's oldText and newText take inside the model's tool call) is
// tokenized on the caller's thread and kept for that session, bounded (a text's first 65,536 tokens,
// 131,072 per session, the oldest texts dropped first, 4 sessions, the least recently fed dropped);
// prompt lookup then also drafts from it in that session's requests. Drafts are verified as every
// draft is: a greedy request keeps only the model's own argmax tokens, a sampled one keeps the
// sampling distribution (its tokens depend on the drafts, as with any speculation). Errors:
// "FeedUnsupported" (Inkling 2 Mini, DeepSeek-V4-Flash), the tokenizer's. Off with SAETTA_FEED=0.
Err server_engine_feed(Engine *e, const char *session, Str key, Str text);
// Whether server_engine_feed keeps anything (the model drafts from fed texts, SAETTA_FEED not 0).
bool server_engine_feeds(const Engine *e);
// Drops the session's fed texts.
void server_engine_feed_clear(Engine *e, const char *session);
// For tests: the session's fed tokens (*texts: its texts), and *bytes: the memory every feed holds.
size_t server_engine_feed_tokens(Engine *e, const char *session, size_t *texts, size_t *bytes);
// For tests: the verified passes of the last request whose drafts came from fed texts.
size_t server_engine_feed_drafts(const Engine *e);
// The last request's tokens the tool-call grammar replaced (SAETTA_GRAMMAR=1; for tests): every pick it
// changed, also in a verification round past where a stop string or max_tokens then cut the reply.
size_t server_engine_grammar_replaced(const Engine *e);
// One HTTP connection on fd (accepted on a socket of the caller's) answered as server_serve
// answers it: the same endpoints and request path. The request is read first (the head and body
// deadlines; the caller sets the socket's send and receive timeouts: no connection guard here),
// then it waits for the engine's turn (requests run one at a time, in arrival order, with
// server_engine_chat's), sending a streamed request keep-alives meanwhile. Calls run on threads
// of their own. The caller closes fd. A shared connection has 5 s for its head; authed(ctx) (may be
// NULL) is called once its token is checked (code_share.c counts the connections not yet checked).
// GET /saetta/whoami?n=<nonce> is answered without the token: HMAC-SHA256(token, nonce), in hex,
// so a client can tell the holder from another program on its port before it sends the token.
Err server_engine_connection(Engine *e, int fd, void (*authed)(void *ctx), void *ctx);
// The token server_engine_connection requires (Authorization: Bearer <token>; a request with an
// Origin header is refused too): 401 / 403 before the body is read. NULL or "": none.
void server_engine_set_token(Engine *e, const char *token);

// The error object a failed request is answered with (also the data of a stream's error
// event): 4xx is the client's (invalid_request_error; Anthropic request_too_large for 413),
// 5xx the server's (OpenAI server_error, Anthropic api_error).
void server_error_json(Buf *o, bool anthropic, int status, Str msg);

// ---- sampling, exposed for the unit tests --------------------------------------------------

typedef struct {
    float temperature, top_p;
    size_t top_k;
} Sampling;

// The numbers of an OpenAI-shaped request: max_completion_tokens or max_tokens (default 4096;
// an integer from 1 to SESSION_CTX_MAX), temperature (1; 0 to 1e6), top_p (0.95; 0 to 1),
// top_k (20; below 1 means 256, the most kept), seed (any integer an i64 holds). Errors:
// "BadMaxTokens", "BadTemperature", "BadTopP", "BadTopK", "BadSeed".
typedef struct {
    size_t max_tokens;
    Sampling sp;
    bool has_seed;
    uint64_t seed;
} ReqNumbers;
Err server_request_numbers(const JValue *r, ReqNumbers *out);

// The k largest of logits[0..n) into cand[0..k), largest first (ties: earlier index first);
// returns how many (min(k, n)).
size_t server_top_k(const float *logits, size_t n, size_t k, uint32_t *cand);

// Speculative sampling (a SpecPick for model_spec_verify, ctx a SampleChooser): a draft is one
// token with probability 1, so it is accepted with its target probability, and on rejection
// the token is drawn from the target with the draft removed; the output then follows the
// sampling distribution exactly. Rows have n values.
typedef struct {
    QgenPrng *rng;
    Sampling sp;
    uint32_t *cand; // room for 256
    size_t n;
} SampleChooser;
uint32_t server_sample_pick(void *ctx, const float *row, bool has_draft, uint32_t draft);

#endif
