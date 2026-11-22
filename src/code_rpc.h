// Saetta Code's rpc mode (`saetta code --mode rpc`, code_rpc.c): pi's headless protocol, JSON
// lines on stdin (commands) and stdout (responses and the session's events).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi v1.5.2,
// b4afddd): packages/coding-agent/src/modes/rpc/rpc-mode.ts, rpc-types.ts, jsonl.ts.
//
// Threads: the main thread reads stdin and answers each command at once (pi handles commands
// while a prompt runs); the agent thread runs prompts and compactions. Both hold one lock while
// they touch the run's state (the session, the model, the run's arena); the agent thread lets it
// go only around its blocking calls (the model's answer, a tool, a retry's wait), during which it
// allocates in an arena of its own (code_main.c). The rpc layer never allocates in the run's
// arena without the lock.
#ifndef SAETTA_CODE_RPC_H
#define SAETTA_CODE_RPC_H

#include "code.h"
#include "code_tui.h"

// What the rpc mode needs from the agent (code_main.c), besides the interactive mode's host calls.
// Each is called with the lock held and the agent thread idle, unless it says otherwise.
typedef struct {
    const CodeUiOps *ui;
    CodeModel *(*model)(CodeUiHost *h);
    CodeEnv *(*env)(CodeUiHost *h);
    Arena *(*arena)(CodeUiHost *h); // the run's arena
    // One prompt to its end on the agent thread (AgentSession.prompt after its preflight).
    Err (*prompt)(CodeUiHost *h, const char *text);
    // The thinking level set; a thinking_level_change entry when it differs from before (before:
    // the level to compare with, NULL for the current one).
    void (*set_thinking)(CodeUiHost *h, const char *level, const char *before);
    // AgentSessionRuntime.newSession: a fresh session for the folder, parentSession in its header.
    Err (*new_session)(CodeUiHost *h, const char *parent);
    // A session made current from a file (switchSession); a missing file is a new session there.
    const char *(*switch_session)(CodeUiHost *h, const char *path);
    // settings retry.enabled for this process (setAutoRetryEnabled), and abortRetry: a retry's
    // wait cancelled (any thread).
    void (*set_auto_retry)(CodeUiHost *h, bool enabled);
    void (*abort_retry)(CodeUiHost *h);
} CodeRpcOps;

// Runs the protocol until stdin ends (exit code 0).
Err code_rpc_run(const CodeRpcOps *ops, CodeUiHost *h);

// The agent's side, from code_main.c:
// an event of the session (emit), written as one line;
void code_rpc_event(const JValue *ev);
// the next queued message for the run (runLoop's getSteeringMessages / getFollowUpMessages), NULL
// when none: more false for the first of a turn, true for each further one (steering or follow-up
// mode "all"); malloc'd;
char *code_rpc_take_queued(bool steering, bool more);
bool code_rpc_has_queued(void);
// the prompt's run is about to start (pi's preflightResult("started")): the response is written;
void code_rpc_prompt_started(void);
// the lock let go around a blocking call and taken again (no-ops outside the rpc mode).
void code_rpc_unlock(void);
void code_rpc_lock(void);

#endif
