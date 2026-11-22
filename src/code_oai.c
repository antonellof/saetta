// Saetta Code OpenAI chat completions client (see code.h). Port of pi-ai api/openai-completions.ts,
// transform-messages.ts.
#include "code.h"

Err code_oai_body(Arena *a, const CodeModel *m, const JValue *system, const JValue *messages, const JValue *tools, Buf *out) {
    return "NotImplemented";
}

Err code_oai_complete(Arena *a, const CodeModel *m, const CodeEnv *env, Str body, const CodeSink *sink, JValue **assistant) {
    return "NotImplemented";
}
