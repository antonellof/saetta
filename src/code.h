// Saetta Code: a minimal coding agent in the saetta binary (`saetta code`), a port of pi.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at
// v1.5.2, b4afddd: packages coding-agent, ai, agent).
//
// Every module works on pi's own message objects, as JSON trees (JValue), in the shapes pi
// stores in its session files (session version 3):
//   system      {"role":"system","content":"","sections":{...},...}
//   user        {"role":"user","content":[{"type":"text","text":...}],"timestamp":N}
//   assistant   {"role":"assistant","content":[{"type":"thinking",...},{"type":"text",...},
//                {"type":"toolCall","id","name","arguments":{...}}],"api","provider","model",
//                "usage":{...},"stopReason","timestamp","responseId","rawStopReason",...}
//   toolResult  {"role":"toolResult","toolCallId","toolName","content":[...],"isError",
//                "durationMs","timestamp"}
// Key order is insertion order and json_write writes what JSON.stringify writes, so a request
// body or a session line is byte-identical to pi's when the trees are built in pi's order.
#ifndef SAETTA_CODE_H
#define SAETTA_CODE_H

#include "base.h"
#include "json.h"

// The model as pi's models.json describes it (the "openai-completions" api only), with the
// provider's compat flags and the session's thinking level.
typedef struct {
    const char *provider;  // the provider key in models.json ("saetta")
    const char *id;        // the model id sent as "model"
    const char *name;
    const char *base_url;  // "http://127.0.0.1:8080/v1"
    const char *api_key;
    bool reasoning;
    uint64_t context_window, max_tokens;
    // compat (pi-ai's OpenAICompletionsCompat): the fields saetta's config sets
    bool supports_developer_role, supports_store, supports_reasoning_effort;
    const char *max_tokens_field; // "max_tokens" or "max_completion_tokens"
    const char *thinking_format;  // "qwen-chat-template", ...
    const JValue *compat;         // the whole compat object, for the fields not listed
    const char *thinking;         // "off", "minimal", "low", "medium", "high", "xhigh"
} CodeModel;

// Where the agent runs.
typedef struct {
    const char *cwd;       // process working directory, as pi's process.cwd()
    const char *home;      // $HOME
    const char *agent_dir; // $PI_CODING_AGENT_DIR, else ~/.pi/agent
    double now_ms;         // wall clock in ms since the epoch (pi's Date.now()), for timestamps
    const char *pi_docs; // the pi-coding-agent package folder for the <docs> section, or NULL to omit it
    const char *system_prompt;                // --system-prompt (text, or a file to read), or NULL
    const char *const *append_system_prompt;  // each --append-system-prompt (text or file), in order
    size_t nappend_system_prompt;
} CodeEnv;

// ---- system prompt (code_prompt.c; pi: core/system-prompt.js, resource-loader.js, skills.js)

// pi's system message for these tools (names in pi's order: read, bash, edit, write, ...):
// {"role":"system","content":"","sections":{...}}, the sections of pi's
// buildSystemPromptSections (preamble, tools, rules, docs, addendum, project_context, skills,
// cwd) from the context files (AGENTS.md / CLAUDE.md lookup), skills, SYSTEM.md and
// APPEND_SYSTEM.md, with pi's project trust in print mode. The agent loop adds "timestamp"
// and "toolsAdded".
Err code_system_message(Arena *a, const CodeEnv *env, const char *const *tools, size_t ntools, JValue **out);
// The prompt text the model gets, from a system message (pi-ai getSystemMessageText).
Str code_system_text(Arena *a, const JValue *system);

// ---- tools (code_tools.c; pi: core/tools/*.js)

// The OpenAI "tools" entries for these tool names, in order ({"type":"function","function":...}).
JValue *code_tool_schemas(Arena *a, const char *const *tools, size_t ntools);
// Runs one tool call ({"type":"toolCall","id","name","arguments"}) in env->cwd and returns
// pi's toolResult message. A failing tool is a result with "isError":true, not an Err; Err
// is for the harness failing (out of memory, no such tool name is a result too).
// *raw (when raw is not NULL) is the tool's own return value as pi's tool_execution_end event
// carries it: {"content":[...],"details":...,"structuredContent":...}.
Err code_tool_run(Arena *a, const CodeEnv *env, const JValue *call, JValue **result, JValue **raw);

// ---- OpenAI chat completions client (code_oai.c; pi-ai: api/openai-completions.js,
// transform-messages.js)

// Receives pi's streaming events while an answer arrives, each an object such as
// {"type":"thinking_delta","contentIndex":0,"delta":"..."} with the partial assistant message
// (pi-ai's AssistantMessageEvent), for json mode and the TUI.
typedef struct {
    void (*on_event)(void *ctx, const JValue *event, const JValue *partial);
    void *ctx;
} CodeSink;

// The request body pi sends for this context: system message, conversation messages (pi
// shapes), tools (code_tool_schemas output, may be NULL).
Err code_oai_body(Arena *a, const CodeModel *m, const JValue *system, const JValue *messages, const JValue *tools, Buf *out);
// POSTs body to m->base_url/chat/completions, streams the answer through sink (may be NULL)
// and returns pi's assistant message. A transport or HTTP error is an assistant message with
// "stopReason":"error" and "errorMessage", as pi records it; Err only for harness failures.
Err code_oai_complete(Arena *a, const CodeModel *m, const CodeEnv *env, Str body, const CodeSink *sink, JValue **assistant);

// ---- sessions (code_session.c; pi: core/session-manager.js)

typedef struct CodeSession CodeSession;
// Opens a new session file for env->cwd under agent_dir/sessions, or (cont) the most recent
// one there; path NULL with no_session keeps it in memory.
Err code_session_open(Arena *a, const CodeEnv *env, bool cont, bool no_session, CodeSession **out);
// The messages on the current branch, in order (pi buildSessionContext).
const JValue *code_session_messages(const CodeSession *s);
// Appends an entry ("message", "model_change", "thinking_level_change", ...): type, id,
// parentId, timestamp, then the members of fields in order.
Err code_session_append(CodeSession *s, const char *type, JValue *fields);
// The header entry ({"type":"session","version":3,"id",...}).
const JValue *code_session_header(const CodeSession *s);
// The field of the last entry of this type on the current branch, or NULL.
const JValue *code_session_last(const CodeSession *s, const char *type, const char *field);
// True when the session has no entry yet besides its header.
bool code_session_is_new(const CodeSession *s);

// ---- `saetta code` (code_main.c)

// args: the arguments after "code". Returns the process exit code in *code.
Err code_run(const char *const *args, size_t n, int *code);

#endif
