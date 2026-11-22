// Saetta Code's MCP support (code_mcp.c, code_mcp_ui.c): pi's built-in MCP integration, the same
// mcp.json files and format, the same tools offered to the model, the same /mcp manager.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at v1.5.2,
// b4afddd): packages/mcp/src (client.ts, config.ts, naming.ts, input-schema.ts, output-limit.ts,
// protocol/, transports/stdio.ts, transports/lines.ts, transports/streamable-http.ts) and
// packages/coding-agent/src/extensions/mcp (config.ts, index.ts, runtime.ts, tools.ts,
// resources.ts, ui.ts, log.ts), core/mcp-servers.ts, core/resolve-config-value.ts.
//
// The servers of mcp.json (agent_dir/mcp.json, and <cwd>/.pi/mcp.json for a trusted project)
// connect in the background when the session starts, each on its own thread. Their tools are
// registered as mcp__<server>__<tool> and declared to the model from the next prompt on; the
// first prompt waits up to 10 s for servers with direct tools (pi's waitForDirectServers). A
// tool call runs on the caller's thread and blocks until the server answers, the request times
// out (the server's "timeout", 60 s by default, reset by progress notifications) or env->abort
// is set. Every function here is thread safe.
#ifndef SAETTA_CODE_MCP_H
#define SAETTA_CODE_MCP_H

#include "code.h"

// ---- the session (index.ts session_start / session_shutdown) --------------------------------------

// Loads the configuration and starts connecting the enabled servers. enabled false: --no-mcp (no
// server connects, no MCP tool exists). allow (may be NULL): the --tools entries (names or
// patterns with *), which decide which MCP tools are declared as pi's allowlist does: given
// without an mcp__ entry, or empty (--no-tools), it declares none.
void code_mcp_start(const CodeEnv *env, bool enabled, const char *const *allow, size_t nallow, bool allow_given);
// Closes every connection (stdin closed, the server's process group sent SIGTERM after 0.5 s and
// SIGKILL 2 s later) and waits for the threads. Safe to call twice.
void code_mcp_shutdown(void);
// SIGTERM to every server process group still running (async-signal-safe; pi's exit hook).
void code_mcp_kill_signal_safe(void);

// pi's project trust for env->cwd (code_prompt.c), which decides whether <cwd>/.pi/mcp.json is read.
bool code_project_trusted(Arena *a, const CodeEnv *env);

// ---- the model's tools (tools.ts, index.ts registerTools) -----------------------------------------

// Before a prompt (before_agent_start): the first one waits for the servers with declared tools,
// at most 10 s (then a notice says they are still connecting).
void code_mcp_before_prompt(void);
// The MCP tool names declared to the model now, in registration order (in the arena).
size_t code_mcp_declared(Arena *a, const char ***names);
// The OpenAI tools entry of an MCP tool ({"type":"function","function":{name, description,
// parameters}}), or NULL when name is not a registered MCP tool.
JValue *code_mcp_tool_schema(Arena *a, Str name);
// An MCP tool's parameters (its normalized input schema), NULL when name is not one.
JValue *code_mcp_tool_parameters(Arena *a, Str name);
// Calls the tool with these (validated) arguments: the model-facing content, details
// ({server, tool, fullOutputPath?}), the structured result for json mode (the CallToolResult
// without _meta) and isError; or *thrown when the call failed (the message pi's agent loop
// records). Progress notifications go to env->on_partial.
void code_mcp_tool_call(Arena *a, const CodeEnv *env, Str name, const JValue *args, JValue **content, JValue **details,
                        JValue **structured, bool *is_error, const char **thrown);

// ---- the interactive mode: code_mcp_ui.h ---------------------------------------------------------

// The UI thread's wake-up, called (from any thread) when something /mcp shows changed or a notice
// is waiting.
void code_mcp_set_waker(void (*wake)(void));
// On every wake-up of the UI thread: hands each waiting notice to notify (kind 0 info, 1
// warning, 2 error: showStatus, showWarning, showError); true when the screen should be drawn
// again.
bool code_mcp_ui_tick(void (*notify)(int kind, const char *msg));

// ---- the manager's data (index.ts serversMenu, serverMenu, showTools, chooseExposure) ---------------

typedef struct {
    char *value, *label, *description;
} McpMenuItem;
typedef struct {
    char *title, *details, *error, *empty, *selected, *confirm_label, *cancel_label;
    McpMenuItem *items;
    size_t n;
} McpMenu;
void code_mcp_menu_free(McpMenu *m);
// The open view: its menu is rebuilt on every change of what it shows, keeping its selection
// (the item the user moved to, else the menu's own, else the first), as pi's subscribed menu does.
typedef enum { MCP_VIEW_NONE, MCP_VIEW_SERVERS, MCP_VIEW_SERVER, MCP_VIEW_TOOLS, MCP_VIEW_EXPOSURE, MCP_VIEW_SIGNIN } McpView;
void code_mcp_view_set(McpView kind, const char *server);
McpView code_mcp_view(char **server); // *server malloc'd (may be NULL)
// The open view's menu; returns the selected item's index.
size_t code_mcp_view_menu(McpMenu *m);
// onSelectionChange
void code_mcp_view_select(const char *value);
// runAction for an item of the server menu ("reconnect", "enable", "disable", "enable-project",
// "disable-project", "signin", "signout"; the background ones return at once and report through
// the menu; "signin" opens the sign-in screen, MCP_VIEW_SIGNIN).
void code_mcp_run_action(const char *name, const char *action);
// The exposure menu's choice (chooseExposure, setExposure).
void code_mcp_choose_exposure(const char *name, const char *exposure);
// The label of an MCP tool's view, "server/tool" (malloc'd), or NULL.
char *code_mcp_tool_label(const char *name);
// A counter that changes whenever something the manager shows changes.
uint64_t code_mcp_generation(void);
// The subcommands (/mcp reconnect|login|logout [server]); notices go out through code_mcp_ui_tick.
// Returns the servers to choose from (malloc'd, *npick of them) when the user must pick one
// (pickServer's ctx.ui.select), else NULL.
char **code_mcp_subcommand(const char *args, size_t *npick);
// A notice for the UI (kind as in code_mcp_ui_tick).
void code_mcp_notify(int kind, const char *msg);

// ---- OAuth sign-in (index.ts signInWithUi; code_mcp_oauth.c) -----------------------------------------

// The sign-in screen (MCP_VIEW_SIGNIN, from the server menu's "Sign in" or /mcp login): its title,
// and either a status message (ui.status) or the authorization URL to show with the paste prompt
// (ui.redirectUrl). Each malloc'd; false when no sign-in runs.
bool code_mcp_signin_screen(char **title, char **message, char **url);
// The cancel key on it: the sign-in stops ("Sign-in cancelled.").
void code_mcp_signin_cancel(void);
// A redirect URL pasted on the redirect screen (empty: ignored).
void code_mcp_signin_submit(const char *url);

// turn_start: servers waiting for a sign-in reconnect when their stored tokens changed since
// (a sign-in in another process, pi's `pi mcp login`). True when one did (the tools to declare
// may have changed).
bool code_mcp_turn_start(void);

// The session moved to another folder (/resume, /import): pi's session_shutdown then session_start.
// Every connection closes, global servers' too; the configuration is read again for env->cwd (its
// .pi/mcp.json only when that folder is trusted, code_project_trusted) and the enabled servers
// connect again, with the same --no-mcp and --tools choices.
void code_mcp_restart(const CodeEnv *env);

#endif
