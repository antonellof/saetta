// Durable runs (Saetta Code's own; pi 1.5.2's coding agent has none): a journal of the run going
// on, written ahead of what it describes, beside the session file (<session>.jsonl.run), so a run
// cut off by a crash, a kill or a closed terminal can be resumed. The concepts are pi Durable's
// (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi v1.5.2, b4afddd:
// packages/durable: the effect sandwich, the replay policy per tool, the interrupted result, the
// restart limit), the code is Saetta's.
//
// The session file stays pi's, byte for byte: the journal is a second file pi never lists (it
// reads only *.jsonl). Records are JSON lines, each one write(2) on an O_APPEND descriptor, closed
// by a CRC32C; reading stops at the first torn, unparsable or out-of-sequence line and cuts it
// off. The process holds flock(LOCK_EX) on the journal while its run goes on, so a journal that
// can be locked was left by a process that is gone.
#ifndef SAETTA_CODE_DURABLE_H
#define SAETTA_CODE_DURABLE_H

#include "code.h"

// A run interrupted this many times in a row (no new entry between) is not resumed again.
#define DURABLE_RESTART_LIMIT 3

// ---- the journal of the run going on (all no-ops without a file-backed session) -----------------

// Reads $SAETTA_DURABLE_CRASH and $SAETTA_DURABLE_SYNC and installs the session hook. Idempotent.
void code_durable_init(void);
// A run starts on s: the journal is created (an old, unlocked one replaced) and the run record
// written (the prompt, the leaf, the entries when the file does not exist yet).
void code_durable_begin(CodeSession *s, const char *mode, const char *prompt);
// The run settled (answered, aborted, error): the journal is removed.
void code_durable_end(void);
bool code_durable_active(void);
// The interactive and rpc queues: a message typed while the run goes on, and one taken into it.
void code_durable_queue(bool steer, const char *text);
void code_durable_take(const char *text);
// Before a model request built on the session up to leaf.
void code_durable_request(const char *leaf, Str body);
// A tool call about to run (safe: rerun after a crash), its output so far (at most once a second),
// its result.
void code_durable_intent(const JValue *call, bool safe);
void code_durable_output(const JValue *call, const JValue *partial);
void code_durable_result(const JValue *call, const JValue *msg);
// read, grep, find and ls: run again after a crash. Every other tool is never run again silently.
bool code_durable_safe(Str tool_name);
// The text of an interrupted call's error result (pi Durable's message, with the reason), with the
// output journalled before the crash.
const char *code_durable_interrupted_text(Arena *a, Str tool_name, Str output);

// ---- recovery ---------------------------------------------------------------------------------------

typedef enum {
    DC_HAVE,        // its result is in the session already
    DC_RECORDED,    // it finished: its journalled result is appended, it is not run again
    DC_NOT_STARTED, // no intent: it never started, it runs as the round would have
    DC_SAFE,        // started, read-only: run again
    DC_ASK,         // started, may have had effects: run again only when the user says so
} DurCallState;

typedef struct {
    const JValue *call; // the toolCall block
    const char *id, *name;
    DurCallState state;
    JValue *result;   // DC_RECORDED: the toolResult message
    Str output;       // what it printed before the crash (DC_SAFE, DC_ASK)
    const char *hint; // write and edit: whether the change looks applied, or NULL
    bool rerun;       // DC_ASK: the user's decision
} DurCall;

typedef struct {
    char *journal, *session; // the paths
    int fd;                  // the journal, locked
    double started_ms;       // when the run started
    int64_t interruptions;   // in a row, this one included
    bool limit;              // interruptions reached DURABLE_RESTART_LIMIT
    const char *prompt;      // the prompt, when nothing of the run reached the session
    const char *name;        // the session's name (session_info) or its first user message
    // the round left open: the calls of the session's last assistant message
    DurCall *calls;
    size_t ncalls;
    bool waiting_model; // the session ends in a user message or tool results: a request was due
    // queued messages not delivered
    const char **queued;
    bool *queued_steer;
    size_t nqueued;
    JValue *records; // the journal's valid records
    size_t nvalid;   // their bytes
} DurRecovery;

// The most recent interrupted run in the folder dir (prefer: a session file whose run is chosen
// before the others), locked; NULL when none. Journals of finished runs, of sessions that no
// longer exist and of sessions that moved on since (pi or another Saetta Code appended) are
// removed on the way; live ones (locked by their process) are left alone.
DurRecovery *code_durable_find(Arena *a, const char *dir, const char *prefer);
// The session file brought up to the journal (entries journalled and not written appended, the
// file created when the crash came before its first write), then opened (*out) and its open round
// read into rec->calls.
Err code_durable_reconcile(Arena *a, DurRecovery *rec, CodeSession **out);
// The resumed run journals into rec's file, which keeps its records; the queue mirror is rec's.
void code_durable_adopt(DurRecovery *rec, CodeSession *s);
// The journal removed (Discard), or let go for another time (NULL-safe).
void code_durable_drop(DurRecovery *rec);
void code_durable_release(DurRecovery *rec);
// What the run was doing when it stopped ("It was running: bash  npm test"), from the journal
// alone, and rec->name filled in; for the recovery prompt, before anything is written.
const char *code_durable_running(Arena *a, DurRecovery *rec);
// "4 minutes", "less than a minute", ... since ms.
const char *code_durable_ago(Arena *a, double ms);
// "bash  npm test", "edit  src/a.c", for the recovery prompt.
const char *code_durable_call_summary(Arena *a, const JValue *call);

// For the tests: the journal's path for a session file, and the crash hook re-read.
char *code_durable_path(Arena *a, const char *session_path);
void code_durable_test_reset(void);
// For the tests: the journal let go as a dead process leaves it (closed, not removed), and CRC32C.
void code_durable_test_abandon(void);
uint32_t code_durable_crc32c(const void *p, size_t n);

#endif
