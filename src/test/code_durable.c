// Tests for src/code_durable.c: the run journal (records, CRC, torn and damaged lines), the lock
// that tells a live run from a dead one, and the reconciliation of a session file with the journal
// of its interrupted run (entries the file lost, a file never written, a session that moved on).
#include "../code_durable.h"
#include "../test.h"

#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

static char *tmp_dir(Arena *a) {
    const char *t = getenv("TMPDIR");
    char *d = arena_printf(a, "%s/saetta-durable-XXXXXX", t && *t ? t : "/tmp");
    return mkdtemp(d);
}

static CodeEnv env_in(Arena *a, const char *d) {
    CodeEnv env = {.cwd = d, .home = d, .agent_dir = path_join(a, d, "agent"), .now_ms = 1795700000000.0};
    return env;
}

static JValue *text_message(Arena *a, const char *role, const char *text) {
    JValue *m = json_new(a, J_OBJECT);
    json_put(a, m, S("role"), json_string(a, str(role)));
    JValue *c = json_new(a, J_ARRAY), *b = json_new(a, J_OBJECT);
    json_put(a, b, S("type"), json_string(a, S("text")));
    json_put(a, b, S("text"), json_string(a, str(text)));
    json_push(a, c, b);
    json_put(a, m, S("content"), c);
    json_put(a, m, S("timestamp"), json_int(a, 1795700000000));
    return m;
}

static Err add(CodeSession *s, Arena *a, JValue *msg) {
    JValue *f = json_new(a, J_OBJECT);
    json_put(a, f, S("message"), msg);
    return code_session_append(s, "message", f);
}

static Str slurp(Arena *a, const char *path) {
    char *text;
    size_t len;
    if (read_file(path, 1 << 24, &text, &len)) return S("");
    char *c = arena_strndup(a, text, len);
    free(text);
    return str_n(c, len);
}

static const char *dir_of(Arena *a, const char *path) { return arena_strndup(a, path, (size_t)(strrchr(path, '/') - path)); }

// A session with a model change, then a run that appended its user message and died.
static CodeSession *interrupted(Arena *a, const char *d, bool written_before) {
    CodeEnv env = env_in(a, d);
    CodeSession *s;
    if (code_session_open(a, &env, false, false, &s)) return NULL;
    JValue *f = json_new(a, J_OBJECT);
    json_put(a, f, S("provider"), json_string(a, S("saetta")));
    if (code_session_append(s, "model_change", f)) return NULL;
    if (written_before) {
        if (add(s, a, text_message(a, "user", "first"))) return NULL;
        if (add(s, a, text_message(a, "assistant", "one"))) return NULL;
    }
    code_durable_begin(s, "print", "second");
    if (add(s, a, text_message(a, "user", "second"))) return NULL;
    code_durable_test_abandon();
    return s;
}

TEST(code_durable_crc32c_vector) { EXPECT_EQ(code_durable_crc32c("123456789", 9), 0xE3069283u); }

TEST(code_durable_live_run_not_recovered) {
    Arena a = {0};
    char *d = tmp_dir(&a);
    code_durable_init();
    CodeEnv env = env_in(&a, d);
    CodeSession *s;
    EXPECT_OK(code_session_open(&a, &env, false, false, &s));
    code_durable_begin(s, "print", "hi");
    EXPECT_OK(add(s, &a, text_message(&a, "user", "hi")));
    EXPECT(code_durable_active());
    // the run holds its journal's lock: another look finds nothing to resume
    EXPECT(!code_durable_find(&a, dir_of(&a, code_session_path(s)), NULL));
    EXPECT(file_exists(code_durable_path(&a, code_session_path(s))));
    code_durable_end();
    EXPECT(!code_durable_active());
    EXPECT(!file_exists(code_durable_path(&a, code_session_path(s))));
    arena_reset(&a);
}

TEST(code_durable_recover_after_user_message) {
    Arena a = {0};
    char *d = tmp_dir(&a);
    code_durable_init();
    CodeSession *s = interrupted(&a, d, true);
    EXPECT(s);
    const char *sp = code_session_path(s);
    DurRecovery *rec = code_durable_find(&a, dir_of(&a, sp), NULL);
    EXPECT(rec);
    EXPECT_STR(str(rec->session), str(sp));
    CodeSession *r;
    EXPECT_OK(code_durable_reconcile(&a, rec, &r));
    EXPECT(r);
    EXPECT(rec->waiting_model);
    EXPECT_EQ(rec->ncalls, 0);
    EXPECT_EQ(rec->interruptions, 1);
    EXPECT(!rec->prompt);
    EXPECT_STR(str(rec->name), S("first"));
    // the file is untouched: nothing was missing
    EXPECT_EQ(code_session_messages(r)->arr.len, 3);
    code_durable_drop(rec);
    EXPECT(!file_exists(code_durable_path(&a, sp)));
    arena_reset(&a);
}

TEST(code_durable_file_lost_its_tail) {
    Arena a = {0};
    char *d = tmp_dir(&a);
    code_durable_init();
    CodeSession *s = interrupted(&a, d, true);
    EXPECT(s);
    const char *sp = code_session_path(s);
    Str whole = slurp(&a, sp);
    // a power loss took the last line (and half of it stays)
    ptrdiff_t cut = (ptrdiff_t)whole.len - 2;
    while (cut > 0 && whole.ptr[cut - 1] != '\n') cut--;
    EXPECT(cut > 0);
    EXPECT_OK(write_file(sp, whole.ptr, (size_t)cut + 10));
    DurRecovery *rec = code_durable_find(&a, dir_of(&a, sp), NULL);
    EXPECT(rec); // a torn line is no entry: the file is behind the journal, not moved on
    CodeSession *r;
    EXPECT_OK(code_durable_reconcile(&a, rec, &r));
    // the journal's copy of the user message is back, after a repaired line break
    const JValue *m = code_session_messages(r);
    EXPECT_EQ(m->arr.len, 3);
    EXPECT_STR(json_strs(json_get(json_get(m->arr.items[2], "content")->arr.items[0], "text")), S("second"));
    code_durable_drop(rec);
    arena_reset(&a);
}

TEST(code_durable_file_never_written) {
    Arena a = {0};
    char *d = tmp_dir(&a);
    code_durable_init();
    CodeSession *s = interrupted(&a, d, false);
    EXPECT(s);
    const char *sp = code_session_path(s);
    Str whole = slurp(&a, sp);
    // the crash came before the file's first write
    unlink(sp);
    DurRecovery *rec = code_durable_find(&a, dir_of(&a, sp), NULL);
    EXPECT(rec);
    CodeSession *r;
    EXPECT_OK(code_durable_reconcile(&a, rec, &r));
    EXPECT(r);
    EXPECT_STR(slurp(&a, sp), whole); // made again from the journal, byte for byte
    code_durable_drop(rec);
    arena_reset(&a);
}

TEST(code_durable_stale_and_finished_dropped) {
    Arena a = {0};
    char *d = tmp_dir(&a);
    code_durable_init();
    CodeSession *s = interrupted(&a, d, true);
    EXPECT(s);
    const char *sp = code_session_path(s);
    // pi resumed the session meanwhile and appended to it
    CodeSession *other;
    EXPECT_OK(code_session_open_path(&a, sp, &other));
    EXPECT_OK(add(other, &a, text_message(&a, "user", "from pi")));
    EXPECT(!code_durable_find(&a, dir_of(&a, sp), NULL));
    EXPECT(!file_exists(code_durable_path(&a, sp)));
    // a journal whose run ended (the end record written, the removal cut off)
    code_durable_begin(other, "print", "x");
    EXPECT_OK(add(other, &a, text_message(&a, "user", "x")));
    char *jp = code_durable_path(&a, sp);
    code_durable_end();
    EXPECT(!file_exists(jp));
    // a session that is gone, written before its run: its journal goes
    CodeSession *s2 = interrupted(&a, tmp_dir(&a), true);
    EXPECT(s2);
    unlink(code_session_path(s2));
    EXPECT(!code_durable_find(&a, dir_of(&a, code_session_path(s2)), NULL));
    EXPECT(!file_exists(code_durable_path(&a, code_session_path(s2))));
    arena_reset(&a);
}

TEST(code_durable_damaged_records_cut) {
    Arena a = {0};
    char *d = tmp_dir(&a);
    code_durable_init();
    CodeSession *s = interrupted(&a, d, true);
    EXPECT(s);
    const char *sp = code_session_path(s);
    char *jp = code_durable_path(&a, sp);
    Str j = slurp(&a, jp);
    // three records (run, entry, ...): flip a byte inside the second one's text
    ptrdiff_t nl = str_find_char(j, '\n', 0);
    EXPECT(nl > 0);
    char *bad = arena_strndup(&a, j.ptr, j.len);
    ptrdiff_t at = str_find(str_n(bad, j.len), S("second"), (size_t)nl);
    EXPECT(at > 0);
    bad[at] = 'S';
    EXPECT_OK(write_file(jp, bad, j.len));
    // only the run record holds; the user message in the file is then no entry of the run: the
    // session looks moved on, and the journal goes
    EXPECT(!code_durable_find(&a, dir_of(&a, sp), NULL));
    EXPECT(!file_exists(jp));
    // a torn last record is cut, the rest kept
    char *torn = arena_printf(&a, "%.*s{\"t\":\"request\",\"se", (int)j.len, j.ptr);
    EXPECT_OK(write_file(jp, torn, strlen(torn)));
    DurRecovery *rec = code_durable_find(&a, dir_of(&a, sp), NULL);
    EXPECT(rec);
    EXPECT_EQ(rec->records->arr.len, 2);
    EXPECT_EQ(rec->nvalid, j.len);
    CodeSession *r;
    EXPECT_OK(code_durable_reconcile(&a, rec, &r));
    code_durable_adopt(rec, r); // the torn line is cut before the run goes on
    code_durable_test_abandon();
    Str after = slurp(&a, jp);
    EXPECT(str_starts(after, j));
    EXPECT(after.len > j.len && after.ptr[j.len] == '{' && str_find(after, S("\"t\":\"recovered\",\"seq\":2"), 0) > 0);
    code_durable_drop(rec);
    arena_reset(&a);
}

TEST(code_durable_restart_limit_and_queue) {
    Arena a = {0};
    char *d = tmp_dir(&a);
    code_durable_init();
    CodeSession *s = interrupted(&a, d, true);
    EXPECT(s);
    const char *sp = code_session_path(s);
    for (int k = 1; k <= DURABLE_RESTART_LIMIT; k++) {
        DurRecovery *rec = code_durable_find(&a, dir_of(&a, sp), NULL);
        EXPECT(rec);
        CodeSession *r;
        EXPECT_OK(code_durable_reconcile(&a, rec, &r));
        EXPECT_EQ(rec->interruptions, k);
        EXPECT(rec->limit == (k >= DURABLE_RESTART_LIMIT));
        if (k == 1) {
            EXPECT_EQ(rec->nqueued, 0);
            code_durable_adopt(rec, r);
            // a steer queued in the resumed run, then the process dies again
            code_durable_queue(true, "also this");
            code_durable_test_abandon();
        } else {
            EXPECT_EQ(rec->nqueued, 1);
            EXPECT_STR(str(rec->queued[0]), S("also this"));
            EXPECT(rec->queued_steer[0]);
            code_durable_adopt(rec, r);
            code_durable_test_abandon();
        }
    }
    // a queued message taken and journalled as an entry is not offered again; progress resets the count
    DurRecovery *rec = code_durable_find(&a, dir_of(&a, sp), NULL);
    EXPECT(rec);
    CodeSession *r;
    EXPECT_OK(code_durable_reconcile(&a, rec, &r));
    code_durable_adopt(rec, r);
    code_durable_take("also this");
    EXPECT_OK(add(r, &a, text_message(&a, "user", "also this")));
    code_durable_test_abandon();
    rec = code_durable_find(&a, dir_of(&a, sp), NULL);
    EXPECT(rec);
    EXPECT_OK(code_durable_reconcile(&a, rec, &r));
    EXPECT_EQ(rec->nqueued, 0);
    EXPECT_EQ(rec->interruptions, 1);
    code_durable_drop(rec);
    arena_reset(&a);
}

TEST(code_durable_texts) {
    Arena a = {0};
    EXPECT_STR(str(code_durable_interrupted_text(&a, S("bash"), S(""))),
               S("Tool bash was interrupted and may have partially run: Saetta Code stopped while it ran, and it was not run again."));
    JValue *call;
    EXPECT_OK(json_parse(&a, S("{\"type\":\"toolCall\",\"id\":\"c1\",\"name\":\"bash\",\"arguments\":{\"command\":\"npm test\\nnext\"}}"), true, &call));
    EXPECT_STR(str(code_durable_call_summary(&a, call)), S("bash  npm test"));
    EXPECT(code_durable_safe(S("read")) && code_durable_safe(S("ls")) && !code_durable_safe(S("bash")) && !code_durable_safe(S("edit")));
    double now = (double)(wall_ns() / 1000000);
    EXPECT_STR(str(code_durable_ago(&a, now - 270000)), S("4 minutes"));
    EXPECT_STR(str(code_durable_ago(&a, now - 5000)), S("less than a minute"));
    arena_reset(&a);
}
