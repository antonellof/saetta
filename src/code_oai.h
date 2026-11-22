// Saetta Code OpenAI client internals (code_oai.c), exposed for src/test/code_oai.c: the SSE
// stream parser without a socket, pi's JSON.parse, and the seam for the system prompt text.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at
// v1.5.2, b4afddd: packages/ai api/openai-completions.ts, utils/streaming-json.ts,
// utils/json-parse.ts).
#ifndef SAETTA_CODE_OAI_H
#define SAETTA_CODE_OAI_H

#include "code.h"

// The prompt text of the system message; code_system_text unless a test replaces it.
extern Str (*code_oai_system_text)(Arena *a, const JValue *system);

// One streamed answer. The partial message and the events handed to the sink live in the
// arena; an event and its partial are valid during the callback only (the partial keeps
// changing). Events go out as pi's consumer would see them: the first event of a network
// read at once, the others of that read once it has been parsed (pi's agent loop copies the
// partial when it gets to an event, after the stream has run ahead).
typedef struct CodeOaiStream CodeOaiStream;
CodeOaiStream *code_oai_stream_new(Arena *a, const CodeModel *m, const CodeEnv *env, const CodeSink *sink);
// The response headers arrived with a 2xx status: pi's "start" event.
void code_oai_stream_start(CodeOaiStream *s);
// One read of the response body. Returns false once the stream has failed (stop reading).
bool code_oai_stream_feed(CodeOaiStream *s, const void *p, size_t n);
// The body ended (error NULL) or the request failed with error, pi's errorMessage for it.
// Returns the final assistant message, after the "done" or "error" event.
JValue *code_oai_stream_end(CodeOaiStream *s, const char *error);

// The errorMessage pi records for an HTTP error status with this response body.
char *code_oai_http_error(Arena *a, int status, Str body);

// JSON.parse as V8 does it: duplicate keys (the last wins, in the first one's place), integer
// keys first, numbers as JavaScript formats them (J_NUMBER_STRING). On a syntax error returns
// NULL and sets *err to V8's message.
JValue *code_oai_js_parse(Arena *a, Str text, char **err);
// JSON.stringify of a tree from code_oai_js_parse or json_parse.
void code_oai_js_write(Buf *out, const JValue *v);

// pi's StreamingJsonParser for tool-call arguments, fed one delta at a time; the value is the
// partial arguments object ({} until there is one), valid until the next feed.
typedef struct CodeArgsParser CodeArgsParser;
CodeArgsParser *code_args_parser_new(void);
void code_args_parser_feed(CodeArgsParser *p, Str text);
const JValue *code_args_parser_value(CodeArgsParser *p);
void code_args_parser_free(CodeArgsParser *p);

#endif
