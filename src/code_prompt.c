// Saetta Code system prompt (see code.h). Port of pi core/system-prompt.ts, resource-loader.ts, skills.ts.
#include "code.h"

Err code_system_message(Arena *a, const CodeEnv *env, const char *const *tools, size_t ntools, JValue **out) {
    return "NotImplemented";
}

Str code_system_text(Arena *a, const JValue *system) { return S(""); }
