// Saetta Code tools (see code.h). Port of pi core/tools/*.ts.
#include "code.h"

JValue *code_tool_schemas(Arena *a, const char *const *tools, size_t ntools) { return json_new(a, J_ARRAY); }

Err code_tool_run(Arena *a, const CodeEnv *env, const JValue *call, JValue **result, JValue **raw) { return "NotImplemented"; }
