// Saetta Code's session commands, the part that works on session files (code_sessions.c): the
// session list of /resume, /fork and /clone, the session name of /name, the statistics of /session,
// and the manual compaction of /compact. Their screens are in code_sessions_ui.c.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi v1.5.2,
// b4afddd): packages/coding-agent/src/core/session-manager.ts (SessionInfo, buildSessionInfo, list,
// listAll, buildSessionPath, buildContextEntries, sessionEntryToContextMessages,
// projectContextEntry, buildSessionProjection, createBranchedSession, appendSessionInfo,
// getSessionName, appendCompaction), core/agent-session-runtime.ts (fork),
// core/agent-session.ts (getUserMessagesForForking, getSessionStats, getContextUsage, compact),
// core/compaction/compaction.ts and utils.ts, core/messages.ts (convertToLlm).
#ifndef SAETTA_CODE_SESSIONS_H
#define SAETTA_CODE_SESSIONS_H

#include "code.h"

// ---- additions to code_session.c --------------------------------------------------------------

char *code_session_uuid(Arena *a, double now_ms);           // uuidv7
char *code_session_iso(Arena *a, double now_ms);            // new Date(ms).toISOString()
char *code_session_new_entry_id(Arena *a, const JValue *entries); // generateId
// A session of these entries (the header first) kept at path (NULL: in memory); the file is
// written now when they hold a conversation, else at the first message.
Err code_session_from_entries(Arena *a, const char *path, JValue *entries, CodeSession **out);
Arena *code_session_arena(const CodeSession *s);
// code_session_append with the entry's time given (ms since the epoch).
Err code_session_append_at(CodeSession *s, const char *type, JValue *fields, double now_ms);

// ---- from code_main.c ------------------------------------------------------------------------------

// getCurrentSystemMessage: every system message of messages replayed into one, or NULL.
JValue *code_current_system(Arena *a, const JValue *messages);

// ---- the session list (SessionInfo) -------------------------------------------------------------

typedef struct {
    const char *path, *id, *cwd;
    const char *name;   // NULL when unnamed
    const char *parent; // the header's parentSession, or NULL
    double created, modified; // ms since the epoch
    size_t message_count;
    const char *first_message, *all_text;
} CodeSessionInfo;

// Called as the sessions load (SessionListProgress): partial (sorted by activity) only at pi's
// publish points, else NULL.
typedef void (*CodeSessionProgress)(void *ctx, size_t loaded, size_t total, const CodeSessionInfo *partial, size_t npartial);
// SessionManager.list for one session folder, and listAll for every folder under
// agent_dir/sessions: sorted by activity, the newest first.
size_t code_sessions_list(Arena *a, const char *dir, CodeSessionProgress cb, void *ctx, CodeSessionInfo **out);
size_t code_sessions_list_all(Arena *a, const char *agent_dir, CodeSessionProgress cb, void *ctx, CodeSessionInfo **out);
// getDefaultSessionDir: agent_dir/sessions/--<cwd>--
char *code_sessions_dir(Arena *a, const char *agent_dir, const char *cwd);

// ---- names ---------------------------------------------------------------------------------------

// getSessionName: the latest session_info entry's name, trimmed; NULL when none or empty.
const char *code_sessions_name(Arena *a, const JValue *entries);
// appendSessionInfo: the name with line breaks as spaces, trimmed.
Err code_sessions_set_name(CodeSession *s, const char *name);
// SessionManager.open(path).appendSessionInfo(name) (the /resume selector's rename).
Err code_sessions_rename_file(const char *path, const char *name);

// ---- the projection (buildSessionProjection) ------------------------------------------------------

// buildSessionPath: the entries from the root to leaf (none for a NULL leaf), header skipped.
JValue *code_sessions_branch(Arena *a, const JValue *entries, const char *leaf);
// buildContextEntries: the latest compaction, the entries it kept, then the ones after it.
JValue *code_sessions_context_entries(Arena *a, const JValue *entries, const char *leaf);
// sessionEntryToContextMessages: pi's messages for one entry (roles compactionSummary,
// branchSummary and custom kept as pi has them), appended to out.
void code_sessions_entry_messages(Arena *a, const JValue *entry, JValue *out);
// estimateTokens (chars / 4, rounded up) of one of those messages.
int64_t code_sessions_estimate(const JValue *message);

// ---- /fork, /clone ------------------------------------------------------------------------------------

// getUserMessagesForForking: every user message with text, in file order ({"entryId","text"}).
JValue *code_sessions_fork_messages(Arena *a, const JValue *entries);
// AgentSessionRuntime.fork: the session for a fork at entry_id ("before": up to the user
// message's parent, *selected its text; "at": up to the entry itself, for /clone). NULL on
// success, else pi's error text.
const char *code_sessions_fork(Arena *a, CodeSession *cur, const char *cwd, const char *entry_id, bool at, CodeSession **out, char **selected);

// SessionManager.forkFrom: a copy of the session file at path (a new id, parentSession the source)
// for cwd in dir; *out its file.
const char *code_sessions_fork_from(Arena *a, const char *path, const char *cwd, const char *dir, char **out);

// ---- /session ----------------------------------------------------------------------------------------

typedef struct {
    size_t user, assistant, tool_calls, tool_results, total;
    double input, output, cache_read, cache_write, cost;
    double hit_rate; // the latest assistant's cache hit rate, < 0 when none (the footer's CH)
} CodeSessionStats;
// getSessionStats over every entry (and the footer's usage totals).
void code_sessions_stats(const JValue *entries, CodeSessionStats *st);
// getContextUsage's unknown case: the branch's latest compaction has no answer with usage after it.
bool code_sessions_context_unknown(Arena *a, const JValue *entries, const char *leaf);

// ---- /compact (AgentSession.compact, manual) ------------------------------------------------------

// Prepares the compaction of the session's branch, asks m for the summary (env->abort cancels)
// and appends the compaction entry (*entry). NULL on success; "aborted"; else the error text that
// follows "Compaction failed: ".
const char *code_sessions_compact(Arena *a, CodeSession *s, const CodeModel *m, const CodeEnv *env, const char *instructions,
                                  int64_t keep_recent_tokens, const JValue **entry);
// The same, with started(ctx) called once the preparation succeeded (before the summary request):
// "Already compacted" and "Nothing to compact (session too small)" come before it (pi's
// prepareCompaction returning undefined).
const char *code_sessions_compact_cb(Arena *a, CodeSession *s, const CodeModel *m, const CodeEnv *env, const char *instructions,
                                     int64_t keep_recent_tokens, const JValue **entry, void (*started)(void *ctx), void *ctx);
// buildSessionProjection(...).messages of the branch ending at leaf (pi's roles).
JValue *code_sessions_projection(Arena *a, const JValue *entries, const char *leaf);
// estimateProjectedContextTokens of the branch ending at leaf.
double code_sessions_projected_tokens(Arena *a, const JValue *entries, const char *leaf);

#endif
