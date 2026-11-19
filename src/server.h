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
#include "qgen.h"

// Loads the pack in dir and serves on 127.0.0.1:port until the process ends. Errors:
// "AlreadyRunning" (another server holds the lock; message printed), load errors, socket
// errors.
Err server_serve(Arena *arena, const char *dir, uint16_t port);

// ---- sampling, exposed for the unit tests --------------------------------------------------

typedef struct {
    float temperature, top_p;
    size_t top_k;
} Sampling;

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
