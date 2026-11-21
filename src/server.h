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

// Loads the pack in dir and serves on 127.0.0.1:port until the process ends. Errors:
// "AlreadyRunning" (another server holds the lock; message printed), load errors, socket
// errors.
// loras: nloras "name=path" LoRA adapters (--lora-modules); path a folder or a Hugging Face id
// under the models folder. A request whose model is an adapter's name runs with it.
// kwargs_default: NULL or a JSON object of chat template kwargs (enable_thinking, ...) used when
// a request does not set them (vLLM's --default-chat-template-kwargs).
Err server_serve(Arena *arena, const char *dir, uint16_t port, const char *const *loras, size_t nloras,
                 const char *kwargs_default);

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
