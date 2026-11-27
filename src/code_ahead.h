// Prefill ahead of the request in Saetta Code (in process): while a tool runs, the next request as
// far as it is known (the history, the round's finished results, the running tool's output so far)
// is handed to a thread of its own, which has the engine read it (server.h server_engine_ahead), so
// the request finds those tokens already read. The latest snapshot wins; a read in flight stops at
// its next chunk when the request comes (the engine's turn) or on code_ahead_cancel.
#ifndef SAETTA_CODE_AHEAD_H
#define SAETTA_CODE_AHEAD_H

#include "base.h"

typedef struct Engine Engine;
typedef struct CodeAhead CodeAhead;

// A reader for engine e (its thread starts at the first post); NULL when e is NULL or SAETTA_AHEAD=0.
CodeAhead *code_ahead_new(Engine *e);
// The next request's body as far as it is known, for agent session session, its tool message with
// tool_call_id partial_id still growing (copied; replaces a snapshot not yet read).
void code_ahead_post(CodeAhead *ah, Str body, const char *session, const char *partial_id);
// The request goes now (or the run ends): a snapshot not yet read is dropped, and a read in flight
// stops at its next chunk. Does not wait.
void code_ahead_cancel(CodeAhead *ah);
// Tokens read ahead so far (all posts), for the log.
size_t code_ahead_total(CodeAhead *ah);
// Stops the thread (waits for a read in flight to stop) and frees the reader.
void code_ahead_free(CodeAhead *ah);

#endif
