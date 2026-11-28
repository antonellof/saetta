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
typedef struct CodeModel {
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
    // The in-process engine (server.h server_engine_open), or NULL to call base_url over HTTP.
    // The request body and the answer are the same either way.
    struct Engine *engine;
    // A check before each request (code_share.c, a model another Saetta Code serves): 1 send it,
    // 2 busy (the answer is a retryable 503 error), 0 the server is not the one joined (the answer
    // is a "Connection error" and nothing, the token least of all, is sent). NULL: none.
    int (*verify)(const struct CodeModel *m);
    // A model another Saetta Code serves: requests go over this Unix socket (src/unix_http.h), not
    // to base_url. NULL: none.
    const char *sock;
    // Requests ask saetta's engine (engine or sock set; never sent elsewhere) for a checkpoint
    // just before their generation prompt ("saetta_fork_point", src/server.c): the agent's own
    // requests while it offers subagents, whose prompts are its prompt less that, plus their task.
    bool fork_point;
    // Fields added to the request body, for saetta's engine only (engine or sock set): Saetta
    // Code's own switches ("thinking_token_budget", SAETTA_EFFORT=auto). NULL: none.
    const JValue *extra;
} CodeModel;

// saetta's version (main.c sets it), for Saetta Code's header.
extern const char *code_version;

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
    // The session facts pi's bash tool exports as PI_SESSION_ID, PI_SESSION_FILE, PI_PROVIDER,
    // PI_MODEL and PI_REASONING_LEVEL (bash.ts resolveSpawnContext); NULL leaves one unset.
    const char *session_id, *session_file, *provider, *model, *thinking;
    // Set (by a signal handler) to abort the request in flight: the client stops reading and
    // records the answer as pi does, stopReason "aborted"; a tool call not yet run returns
    // "Operation aborted" and a running bash command has its process group killed ("Command
    // aborted"). NULL when nothing aborts.
    volatile int *abort;
    // A running tool's partial result (pi's onUpdate: the bash tool's output so far, at most every
    // 100 ms), or NULL. The value is valid during the call only.
    void (*on_partial)(void *ctx, const JValue *partial);
    void *partial_ctx;
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
// pi's toolResult message: the whole per-call path of pi's agent loop (unknown tool,
// prepareArguments, argument validation and coercion, execution, thrown errors). A failing
// tool is a result with "isError":true, not an Err; Err is for the harness failing (no such
// tool name is a result too). raw (may be NULL) gets the tool's own return value as json
// mode's tool_execution_end prints it ({"content","details","structuredContent",...}).
Err code_tool_run(Arena *a, const CodeEnv *env, const JValue *call, JValue **result, JValue **raw);
// pi's truncateModelOutput (its subagent example): output cut to 2000 lines and 50 KB with a notice.
Str code_tool_truncate_model_output(Arena *a, Str output);
// The validation code_tool_run applies (prepareToolCall: an arguments parse error, then the
// schema with pi's coercion), for a tool that runs elsewhere (code_subagent.c): the arguments to
// run with, or NULL and *err, the text of the call's error result.
JValue *code_tool_validate(Arena *a, const JValue *call, const JValue *params, const char **err);
// pi's killTrackedDetachedChildren (utils/shell.ts): SIGKILLs the process group of every bash
// command still running (pi tracks a group from spawn until its wait ends). Async-signal-safe,
// for exit and the SIGTERM / SIGHUP handlers.
void code_tools_kill_detached(void);
// The user's own shell command (rpc mode's bash: pi's createLocalShellOperations().exec, no
// timeout): run in env->cwd in its own process group, stdin /dev/null, each chunk of stdout and
// stderr to on_data as it arrives; *abort_flag nonzero kills the group (*aborted). Returns the exit
// code (128 + the signal); *thrown when it could not run.
int code_tools_exec(Arena *a, const CodeEnv *env, const char *command, void (*on_data)(void *ctx, const void *p, size_t n), void *ctx,
                    volatile int *abort_flag, bool *aborted, const char **thrown);
// truncateTail with the defaults (2000 lines, 50 KB): the tail kept, *truncated.
Str code_tools_truncate_tail(Str s, bool *truncated);
// The file a read (read true: with its macOS spellings that exist, as resolveReadPath) or an edit
// or write of path p names, absolute.
char *code_tools_path(Arena *a, const CodeEnv *env, Str p, bool read);

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

// The speed of model requests, for the interactive footer and /session (Saetta Code's own, not
// pi's): generated tokens and the time spent generating them (prompt reading excluded), and
// the prompt tokens read (the reused ones excluded) with their time. The in-process engine
// gives its own numbers (the request log's "N generated in M ms"); another model's come from
// the stream: usage's completion_tokens (else the content chunks) over the time from the
// first streamed token to the end, and no prompt numbers (prompt_ms 0).
typedef struct {
    double gen_tokens, gen_ms, prompt_tokens, prompt_ms;
} CodeGenStats;
// The numbers of the last code_oai_complete on this thread; false (and *g zero) when it measured
// nothing: an error or aborted answer, or no token generated.
bool code_oai_last_gen(CodeGenStats *g);
// Adds one request's numbers to *sum (a turn's requests: total tokens over total time).
void code_gen_add(CodeGenStats *sum, const CodeGenStats *g);
// "17.9 tok/s" for tokens over ms (one decimal below 100, an integer from 100), appended to b;
// nothing when either is not positive. Returns whether it wrote.
bool code_fmt_tps(Buf *b, double tokens, double ms);

// ---- sessions (code_session.c; pi: core/session-manager.js)

typedef struct CodeSession CodeSession;
// Opens a new session file for env->cwd under agent_dir/sessions, or (cont) the most recent
// one there; path NULL with no_session keeps it in memory.
Err code_session_open(Arena *a, const CodeEnv *env, bool cont, bool no_session, CodeSession **out);
// That session file (--session).
Err code_session_open_path(Arena *a, const char *path, CodeSession **out);
// Whether the session has been written to its file (pi writes it once there is a conversation).
bool code_session_written(const CodeSession *s);
// The messages on the current branch, in order (pi buildSessionContext).
const JValue *code_session_messages(const CodeSession *s);
// Appends an entry ("message", "model_change", "thinking_level_change", ...): type, id,
// parentId, timestamp, then the members of fields in order.
Err code_session_append(CodeSession *s, const char *type, JValue *fields);
// The header entry ({"type":"session","version":3,"id",...}).
const JValue *code_session_header(const CodeSession *s);
// The field of the last entry of this type on the current branch, or NULL.
const JValue *code_session_last(const CodeSession *s, const char *type, const char *field);
// The session file (written once the session holds a message), NULL when not persisted.
const char *code_session_path(const CodeSession *s);
// Every entry, the header first (the whole tree, for /tree).
const JValue *code_session_entries(const CodeSession *s);
// Moves the current leaf (pi's branch()): the next entry is appended under it. NULL: the root.
void code_session_set_leaf(CodeSession *s, const char *id);
// The id of the last entry appended (the current leaf), NULL before the first.
const char *code_session_leaf(const CodeSession *s);
// The last entry appended (the header before any).
const JValue *code_session_entry_last(const CodeSession *s);
// True when the session has no entry yet besides its header.
bool code_session_is_new(const CodeSession *s);
// An entry made elsewhere (a run's journal), appended as it is (id, parentId, timestamp kept).
Err code_session_append_raw(CodeSession *s, JValue *entry);
// The write-ahead hook of a run's journal (code_durable.c): called before (after false) and after
// (after true) each entry is written to a session file. NULL: none.
extern void (*code_session_wal)(const CodeSession *s, const JValue *entry, bool after);

// s with each maximal ill-formed UTF-8 subpart replaced by U+FFFD (TextDecoder's replacement),
// s itself when it is well formed.
Str code_utf8_clean(Arena *a, Str s);

// ---- `saetta code` (code_main.c)

// args: the arguments after "code". Returns the process exit code in *code.
Err code_run(const char *const *args, size_t n, int *code);

#endif
