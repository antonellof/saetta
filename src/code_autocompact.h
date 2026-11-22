// Saetta Code's automatic compaction (code_autocompact.c): the threshold check after a run and
// before each further turn, the context overflow recovery (compact, then retry the turn once), and
// the compaction settings. The summary itself is code_sessions.c's (the one /compact uses).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi v1.5.2,
// b4afddd): packages/coding-agent/src/core/agent-session.ts (_checkCompaction,
// _runAutoCompaction, _compactBeforeNextAssistantResponse, _omitRecoveryAttempt,
// _findLastAssistantMessage), core/settings-manager.ts (getCompactionSettings,
// setCompactionEnabled), core/compaction/compaction.ts (shouldCompact, estimateContextTokens,
// calculateContextTokens), packages/ai/src/utils/overflow.ts (isContextOverflow,
// isRecoverableLength).
#ifndef SAETTA_CODE_AUTOCOMPACT_H
#define SAETTA_CODE_AUTOCOMPACT_H

#include "code.h"

// settings-manager.ts getCompactionSettings: compaction.enabled, reserveTokens, keepRecentTokens,
// the model's compaction.modelOverrides["provider/id"] first, pi's defaults (true, 16384, 20000).
typedef struct {
    bool enabled;
    int64_t reserve_tokens, keep_recent_tokens;
} CodeCompactionSettings;
CodeCompactionSettings code_compaction_settings(const JValue *settings, const char *provider, const char *id);
// setCompactionEnabled: settings.json's compaction.enabled written (/settings' Auto-compact).
void code_compaction_set_enabled(const char *agent_dir, bool enabled);

// isContextOverflow (window 0: the error patterns only) and isRecoverableLength.
bool code_context_overflow(const JValue *assistant, uint64_t context_window);
bool code_recoverable_length(const JValue *assistant, uint64_t desired_max_output);

// The agent's side of an automatic compaction.
typedef struct {
    Arena *a;
    CodeSession **session; // the current session (a /resume or /fork replaces it)
    const CodeModel *model;
    const CodeEnv *env;    // env->abort cancels the summary (pi's abortCompaction)
    void (*emit)(void *ctx, JValue *event); // the session event, as pi's _emit
    bool (*has_queued)(void *ctx);          // agent.hasQueuedMessages()
    void *ctx;
    bool overflow_attempted; // pi's _overflowRecoveryAttempted
} CodeAutoCompact;

// _checkCompaction after a run (skip_aborted) or before a prompt (!skip_aborted) for the assistant
// message of entry entry_id (NULL: look it up): whether the run goes on (agent.continue(), for
// the overflow retry or queued messages).
bool code_autocompact_check(CodeAutoCompact *c, const JValue *assistant, const char *entry_id, bool skip_aborted);
// _prepareRun's check: the last assistant message of the context, if any.
void code_autocompact_before_prompt(CodeAutoCompact *c);
// _compactBeforeNextAssistantResponse: the threshold check before a further turn of the run.
void code_autocompact_before_turn(CodeAutoCompact *c);
// What resets _overflowRecoveryAttempted: a user message starting, an assistant message that
// ended neither in an error nor at the length limit.
void code_autocompact_observe(CodeAutoCompact *c, const JValue *message);

#endif
