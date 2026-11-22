// Saetta Code sessions (see code.h). Port of pi core/session-manager.ts.
#include "code.h"

Err code_session_open(Arena *a, const CodeEnv *env, bool cont, bool no_session, CodeSession **out) { return "NotImplemented"; }

const JValue *code_session_messages(const CodeSession *s) { return NULL; }

Err code_session_append(CodeSession *s, const char *type, JValue *fields) { return "NotImplemented"; }
