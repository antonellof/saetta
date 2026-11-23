// Subagents (`saetta code --subagents`, or "subagents": true in settings.json): a "subagent" tool
// the model calls to hand a self-contained task to a child agent. Children run the agent loop
// (request, tool calls, results, until an answer) on threads of their own, up to
// SUBAGENT_PARALLEL at once; each child's request goes to the model as its own conversation and
// starts with the parent's: the same system prompt, tool declarations and messages up to the
// call, then the task as a user message. A model server that reuses prompt prefixes (saetta's,
// src/server.c) then reads only the child's own part. The child's final text is the tool's
// result; its whole transcript is in the result's details (so the session file, /resume and
// /export keep it), and its progress goes out as tool_execution_update events while it runs.
// Off by default: pi has no built-in subagents, and with it off the system prompt and the tools
// sent are pi's byte for byte.
// Saetta's own feature, shaped after Claude Code's Task tool (description, prompt) and pi's
// example subagent extension (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi
// v1.5.2, b4afddd: packages/coding-agent/examples/extensions/subagent/index.ts), whose views
// code_subagent_ui.c ports.
#ifndef SAETTA_CODE_SUBAGENT_H
#define SAETTA_CODE_SUBAGENT_H

#include "code.h"

#define SUBAGENT_PARALLEL 3
#define SUBAGENT_TOOL "subagent"

void code_subagent_enable(bool on);
bool code_subagent_enabled(void);
// The tool's OpenAI declaration ({"type":"function","function":{...}}).
JValue *code_subagent_schema(Arena *a);

// What the children of one assistant message share, taken from the parent when its tool calls
// run. Nothing here is written while children run; each child works on its own copies.
typedef struct {
    const CodeModel *model;
    const CodeEnv *env;       // cwd, abort (Esc stops the children too), session facts
    const JValue *system;     // the parent's current system message
    const JValue *tools;      // the OpenAI tools the parent's request declared (subagent among them)
    const JValue *prefix;     // the parent's conversation before the calling assistant message
    const char *const *names; // the tools a child may run: the parent's, subagent excepted
    size_t nnames;
    // A progress event (tool_execution_update) to print or draw: called from the children's
    // threads, one at a time, while the parent's thread waits in code_subagent_wait.
    void (*emit)(void *ctx, JValue *event);
    void *emit_ctx;
} SubagentParent;

typedef struct SubagentBatch SubagentBatch;
SubagentBatch *code_subagent_begin(const SubagentParent *p);
// One subagent call of the message: *msg and *raw are set by code_subagent_wait (pi's toolResult
// message, and the result as tool_execution_end prints it).
void code_subagent_add(SubagentBatch *b, Arena *a, const JValue *call, JValue **msg, JValue **raw);
// Runs the calls added (at most SUBAGENT_PARALLEL at a time) and waits for all of them; the
// results are built in each call's arena. Frees the batch.
Err code_subagent_wait(SubagentBatch *b);

#ifdef SAETTA_CODE_TUI_H
// The tool view of a subagent call (code_ui.c's tool box): the call, then the child's progress or
// result (pi's example extension, single mode). result is {"content","details"}.
void code_subagent_ui_render(const JValue *args, const JValue *result, bool has_result, bool partial, bool is_error,
                             bool expanded, int width, Lines *out);
#endif

#endif
