// Durable runs (see code_durable.h): the journal, its reading, and the reconciliation of a session
// file with the journal of its interrupted run.
// Concepts adapted from pi Durable (MIT, Copyright (c) 2025 Mario Zechner;
// github.com/earendil-works/pi v1.5.2, b4afddd: packages/durable/src/harness/tool.ts (the intent
// committed before execute, rerun only for replay "safe", the interrupted result's message),
// harness/agent.ts (the restart limit), storage/jsonl/storage.ts (a torn last line cut off at open)).
#include "code_durable.h"
#include "hash.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

enum { JOURNAL_FORMAT = 1, OUTPUT_CAP = 50 * 1024 };

// ---- CRC32C (Castagnoli), bitwise table ---------------------------------------------------------------

static uint32_t g_crc_table[256];

static void crc_init(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = c & 1 ? (c >> 1) ^ 0x82F63B78u : c >> 1;
        g_crc_table[i] = c;
    }
}

static uint32_t crc32c(const void *p, size_t n) {
    if (!g_crc_table[1]) crc_init();
    const uint8_t *b = p;
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) c = g_crc_table[(c ^ b[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

// ---- state ------------------------------------------------------------------------------------------

typedef struct {
    char *text;
    bool steer, taken, delivered; // taken: a take record; delivered: an entry followed it
} QItem;

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static bool g_inited;
static int g_fd = -1;
static char *g_path;
static const CodeSession *g_sess;
static int64_t g_seq;
static VEC(QItem) g_queue;
static double g_last_output_ms;
static char *g_output_call;
// the crash hook (tests): kill -9 right after the n-th record of this type ("appended": after a
// session entry reached the file)
static char g_crash_type[32];
static long g_crash_at, g_crash_count;
// none, barrier (F_BARRIERFSYNC) or full (F_FULLFSYNC) after the records that precede an effect
static enum { SYNC_NONE, SYNC_BARRIER, SYNC_FULL } g_sync = SYNC_BARRIER;

static void crash_point(const char *type) {
    if (!g_crash_type[0] || strcmp(g_crash_type, type)) return;
    if (++g_crash_count == g_crash_at) kill(getpid(), SIGKILL);
}

static void wal_hook(const CodeSession *s, const JValue *entry, bool after);

void code_durable_test_reset(void) {
    g_crash_type[0] = 0;
    g_crash_at = g_crash_count = 0;
    const char *c = getenv("SAETTA_DURABLE_CRASH");
    if (c && *c) {
        const char *colon = strchr(c, ':');
        size_t n = colon ? (size_t)(colon - c) : strlen(c);
        if (n < sizeof g_crash_type) {
            memcpy(g_crash_type, c, n);
            g_crash_type[n] = 0;
            g_crash_at = colon ? atol(colon + 1) : 1;
            if (g_crash_at < 1) g_crash_at = 1;
        }
    }
    const char *s = getenv("SAETTA_DURABLE_SYNC");
    g_sync = !s || !*s || !strcmp(s, "barrier") ? SYNC_BARRIER : !strcmp(s, "full") ? SYNC_FULL : SYNC_NONE;
}

void code_durable_init(void) {
    pthread_mutex_lock(&g_lock);
    if (!g_inited) {
        g_inited = true;
        code_durable_test_reset();
        code_session_wal = wal_hook;
    }
    pthread_mutex_unlock(&g_lock);
}

char *code_durable_path(Arena *a, const char *session_path) { return arena_printf(a, "%s.run", session_path); }

bool code_durable_active(void) {
    pthread_mutex_lock(&g_lock);
    bool on = g_fd >= 0;
    pthread_mutex_unlock(&g_lock);
    return on;
}

// ---- writing ------------------------------------------------------------------------------------------

static void sync_fd(int fd) {
    if (g_sync == SYNC_BARRIER) {
        if (fcntl(fd, F_BARRIERFSYNC) != 0) fsync(fd);
    } else if (g_sync == SYNC_FULL) {
        if (fcntl(fd, F_FULLFSYNC) != 0) fsync(fd);
    }
}

// One record: {"t":type,"seq":N, the members of rec..., "crc":"<8 hex>"} and a newline, in one write.
// Called with g_lock held.
static void put_l(const char *type, const JValue *rec, bool sync) {
    if (g_fd < 0) return;
    Arena t = {0};
    JValue *o = json_new(&t, J_OBJECT);
    json_put(&t, o, S("t"), json_string(&t, str(type)));
    json_put(&t, o, S("seq"), json_int(&t, g_seq));
    for (size_t i = 0; rec && i < rec->obj.len; i++) json_put(&t, o, rec->obj.items[i].key, rec->obj.items[i].val);
    Buf b = {0};
    json_write(&b, o);
    uint32_t crc = crc32c(b.ptr, b.len);
    b.len--; // the closing brace
    buf_printf(&b, ",\"crc\":\"%08x\"}\n", crc);
    const char *p = b.ptr;
    size_t n = b.len;
    while (n > 0) {
        ssize_t w = write(g_fd, p, n);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) break; // a journal that cannot be written loses its recovery, never the run
        p += w;
        n -= (size_t)w;
    }
    g_seq++;
    if (sync) sync_fd(g_fd);
    buf_free(&b);
    arena_reset(&t);
    crash_point(type);
}

static void put(const char *type, const JValue *rec, bool sync) {
    pthread_mutex_lock(&g_lock);
    put_l(type, rec, sync);
    pthread_mutex_unlock(&g_lock);
}

static void queue_clear_l(void) {
    for (size_t i = 0; i < g_queue.len; i++) free(g_queue.ptr[i].text);
    g_queue.len = 0;
}

void code_durable_begin(CodeSession *s, const char *mode, const char *prompt) {
    const char *sp = s ? code_session_path(s) : NULL;
    if (!g_inited || !sp) return;
    pthread_mutex_lock(&g_lock);
    if (g_fd >= 0) { // a run inside a run never happens; keep the outer one
        pthread_mutex_unlock(&g_lock);
        return;
    }
    Arena t = {0};
    char *path = code_durable_path(&t, sp);
    int fd = open(path, O_RDWR | O_CREAT | O_APPEND | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) != 0) { // another process runs this session
        close(fd);
        fd = -1;
    }
    if (fd >= 0) {
        if (ftruncate(fd, 0) != 0) { // an old journal the user passed on
            close(fd);
            fd = -1;
        }
    }
    if (fd < 0) {
        arena_reset(&t);
        pthread_mutex_unlock(&g_lock);
        return;
    }
    g_fd = fd;
    g_path = strdup(path);
    g_sess = s;
    g_seq = 0;
    queue_clear_l();
    free(g_output_call);
    g_output_call = NULL;
    JValue *r = json_new(&t, J_OBJECT);
    json_put(&t, r, S("v"), json_int(&t, JOURNAL_FORMAT));
    json_put(&t, r, S("saetta"), json_string(&t, str(code_version ? code_version : "")));
    json_put(&t, r, S("pid"), json_int(&t, (int64_t)getpid()));
    json_put(&t, r, S("time"), json_int(&t, (int64_t)(wall_ns() / 1000000)));
    json_put(&t, r, S("mode"), json_string(&t, str(mode)));
    const char *leaf = code_session_leaf(s);
    json_put(&t, r, S("leaf"), leaf ? json_string(&t, str(leaf)) : json_new(&t, J_NULL));
    json_put(&t, r, S("prompt"), prompt ? json_string(&t, str(prompt)) : json_new(&t, J_NULL));
    // a session not written yet: its entries so far, so the file can be made from the journal
    if (!code_session_written(s)) json_put(&t, r, S("pre"), (JValue *)code_session_entries(s));
    put_l("run", r, true);
    arena_reset(&t);
    pthread_mutex_unlock(&g_lock);
}

static void close_l(bool remove) {
    if (g_fd < 0) return;
    if (remove && g_path) unlink(g_path);
    close(g_fd); // the lock goes with it
    g_fd = -1;
    free(g_path);
    g_path = NULL;
    g_sess = NULL;
    queue_clear_l();
}

void code_durable_test_abandon(void) {
    pthread_mutex_lock(&g_lock);
    close_l(false);
    pthread_mutex_unlock(&g_lock);
}

uint32_t code_durable_crc32c(const void *p, size_t n) { return crc32c(p, n); }

void code_durable_end(void) {
    pthread_mutex_lock(&g_lock);
    if (g_fd >= 0) {
        put_l("end", NULL, false);
        close_l(true);
    }
    pthread_mutex_unlock(&g_lock);
}

void code_durable_queue(bool steer, const char *text) {
    if (!text) return;
    pthread_mutex_lock(&g_lock);
    if (g_fd >= 0) {
        vec_push(&g_queue, ((QItem){strdup(text), steer, false, false}));
        Arena t = {0};
        JValue *r = json_new(&t, J_OBJECT);
        json_put(&t, r, S("kind"), json_string(&t, steer ? S("steer") : S("followUp")));
        json_put(&t, r, S("text"), json_string(&t, str(text)));
        put_l("queue", r, true);
        arena_reset(&t);
    }
    pthread_mutex_unlock(&g_lock);
}

void code_durable_take(const char *text) {
    if (!text) return;
    pthread_mutex_lock(&g_lock);
    if (g_fd >= 0) {
        for (size_t i = 0; i < g_queue.len; i++)
            if (!g_queue.ptr[i].taken && !strcmp(g_queue.ptr[i].text, text)) {
                g_queue.ptr[i].taken = true;
                break;
            }
        Arena t = {0};
        JValue *r = json_new(&t, J_OBJECT);
        json_put(&t, r, S("text"), json_string(&t, str(text)));
        put_l("take", r, false);
        arena_reset(&t);
    }
    pthread_mutex_unlock(&g_lock);
}

static void hex(char *out, const uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; i++) snprintf(out + 2 * i, 3, "%02x", p[i]);
}

void code_durable_request(const char *leaf, Str body) {
    if (!code_durable_active()) return;
    uint8_t h[SHA256_LEN];
    sha256(body.ptr, body.len, h);
    char hx[2 * SHA256_LEN + 1];
    hex(hx, h, SHA256_LEN);
    Arena t = {0};
    JValue *r = json_new(&t, J_OBJECT);
    json_put(&t, r, S("leaf"), leaf ? json_string(&t, str(leaf)) : json_new(&t, J_NULL));
    json_put(&t, r, S("body"), json_string(&t, str(hx)));
    put("request", r, false);
    arena_reset(&t);
}

void code_durable_intent(const JValue *call, bool safe) {
    if (!code_durable_active()) return;
    Arena t = {0};
    JValue *r = json_new(&t, J_OBJECT);
    json_put(&t, r, S("id"), (JValue *)json_get(call, "id"));
    json_put(&t, r, S("name"), (JValue *)json_get(call, "name"));
    json_put(&t, r, S("arguments"), (JValue *)json_get(call, "arguments"));
    json_put(&t, r, S("policy"), json_string(&t, safe ? S("safe") : S("ask")));
    // an effect may follow: its intent reaches the disk first (only the never-re-run tools)
    put("intent", r, !safe);
    arena_reset(&t);
}

// The text blocks of a result's content, joined.
static void content_text(const JValue *content, Buf *out) {
    for (size_t i = 0; content && json_is(content, J_ARRAY) && i < content->arr.len; i++) {
        const JValue *b = content->arr.items[i];
        if (!str_eqz(json_strs(json_get(b, "type")), "text")) continue;
        buf_addstr(out, json_strs(json_get(b, "text")));
    }
}

void code_durable_output(const JValue *call, const JValue *partial) {
    if (!code_durable_active() || !call) return;
    const JValue *pc = json_get(partial, "content");
    if (!json_is(pc, J_ARRAY) || !pc->arr.len) return; // nothing printed yet: not worth a record
    double now = (double)(now_ns() / 1000000);
    const char *id = json_str(json_get(call, "id"));
    pthread_mutex_lock(&g_lock);
    bool same = g_output_call && id && !strcmp(g_output_call, id);
    bool due = !same || now - g_last_output_ms >= 1000;
    if (due) {
        free(g_output_call);
        g_output_call = id ? strdup(id) : NULL;
        g_last_output_ms = now;
    }
    pthread_mutex_unlock(&g_lock);
    if (!due) return;
    Buf b = {0};
    content_text(json_get(partial, "content"), &b);
    Str tail = buf_str(&b);
    if (tail.len > OUTPUT_CAP) {
        size_t from = tail.len - OUTPUT_CAP;
        while (from < tail.len && ((unsigned char)tail.ptr[from] & 0xC0) == 0x80) from++; // a whole UTF-8 character
        tail = str_sub(tail, from, tail.len);
    }
    Arena t = {0};
    JValue *r = json_new(&t, J_OBJECT);
    json_put(&t, r, S("id"), (JValue *)json_get(call, "id"));
    json_put(&t, r, S("text"), json_string(&t, tail));
    put("output", r, false);
    arena_reset(&t);
    buf_free(&b);
}

void code_durable_result(const JValue *call, const JValue *msg) {
    if (!code_durable_active() || !msg) return;
    Arena t = {0};
    JValue *r = json_new(&t, J_OBJECT);
    json_put(&t, r, S("id"), (JValue *)json_get(call, "id"));
    json_put(&t, r, S("message"), (JValue *)msg);
    put("result", r, false);
    arena_reset(&t);
}

// Before an entry reaches the session file: the whole entry, so recovery can write it if the file
// lost it or never got it. After: the "appended" crash point.
static void wal_hook(const CodeSession *s, const JValue *entry, bool after) {
    pthread_mutex_lock(&g_lock);
    bool mine = g_fd >= 0 && g_sess == s;
    if (mine && !after) {
        Arena t = {0};
        JValue *r = json_new(&t, J_OBJECT);
        json_put(&t, r, S("entry"), (JValue *)entry);
        put_l("entry", r, false);
        arena_reset(&t);
        // a queued message taken into the run is delivered once its entry is journalled
        for (size_t i = 0; i < g_queue.len; i++)
            if (g_queue.ptr[i].taken) g_queue.ptr[i].delivered = true;
    }
    pthread_mutex_unlock(&g_lock);
    if (mine && after) crash_point("appended");
}

bool code_durable_safe(Str name) {
    return str_eqz(name, "read") || str_eqz(name, "grep") || str_eqz(name, "find") || str_eqz(name, "ls");
}

const char *code_durable_interrupted_text(Arena *a, Str name, Str output) {
    const char *head = arena_printf(a,
                                    "Tool %.*s was interrupted and may have partially run: Saetta Code stopped while it ran, and it "
                                    "was not run again.",
                                    (int)name.len, name.ptr);
    if (!output.len) return head;
    return arena_printf(a, "%s\n\nOutput before the interruption:\n%.*s", head, (int)output.len, output.ptr);
}

// ---- reading --------------------------------------------------------------------------------------------

// The record of one line, or NULL when its CRC, JSON or shape fails.
static JValue *parse_record(Arena *a, Str line) {
    static const char tag[] = ",\"crc\":\"";
    size_t tl = sizeof tag - 1;
    if (line.len < tl + 10 || line.ptr[line.len - 1] != '}' || line.ptr[line.len - 2] != '"') return NULL;
    size_t at = line.len - tl - 10;
    if (memcmp(line.ptr + at, tag, tl)) return NULL;
    char hx[9];
    memcpy(hx, line.ptr + at + tl, 8);
    hx[8] = 0;
    char *end;
    unsigned long want = strtoul(hx, &end, 16);
    if (*end) return NULL;
    Buf body = {0};
    buf_add(&body, line.ptr, at);
    buf_addc(&body, '}');
    bool ok = crc32c(body.ptr, body.len) == (uint32_t)want;
    JValue *v = NULL;
    // numbers kept as written: an entry is written back byte for byte
    if (ok && (json_parse(a, buf_str(&body), false, &v) || !json_is(v, J_OBJECT) || !json_str(json_get(v, "t")))) v = NULL;
    buf_free(&body);
    return ok ? v : NULL;
}

static int64_t num(const JValue *v) {
    if (!v) return -1;
    if (v->kind == J_INTEGER) return v->i;
    if (v->kind == J_NUMBER_STRING) {
        int64_t x;
        return parse_i64(v->s, &x) ? -1 : x;
    }
    return -1;
}

// The valid records of a journal (in order) and the length they cover.
static JValue *read_records(Arena *a, int fd, size_t *valid) {
    JValue *recs = json_new(a, J_ARRAY);
    *valid = 0;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) return recs;
    size_t n = (size_t)st.st_size;
    char *text = arena_alloc(a, n + 1);
    size_t got = 0;
    while (got < n) {
        ssize_t r = pread(fd, text + got, n - got, (off_t)got);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) break;
        got += (size_t)r;
    }
    Str rest = str_n(text, got);
    size_t off = 0;
    int64_t seq = 0;
    while (off < rest.len) {
        ptrdiff_t nl = str_find_char(rest, '\n', off);
        if (nl < 0) break; // torn
        JValue *rec = parse_record(a, str_sub(rest, off, (size_t)nl));
        if (!rec || num(json_get(rec, "seq")) != seq) break;
        if (seq == 0 && !str_eqz(json_strs(json_get(rec, "t")), "run")) break;
        json_push(a, recs, rec);
        seq++;
        off = (size_t)nl + 1;
        *valid = off;
    }
    return recs;
}

// ---- the session file, as lines -------------------------------------------------------------------------

typedef struct {
    VEC(const char *) ids; // per line (NULL for the header and lines that do not parse)
    bool exists;
    size_t len;
    bool ends_nl;
} SessFile;

static void read_session(Arena *a, const char *path, SessFile *f) {
    *f = (SessFile){0};
    char *text;
    size_t len;
    if (read_file(path, (size_t)1 << 30, &text, &len)) return;
    f->exists = true;
    f->len = len;
    f->ends_nl = len == 0 || text[len - 1] == '\n';
    Str rest = str_n(text, len);
    size_t off = 0;
    while (off < rest.len) {
        ptrdiff_t nl = str_find_char(rest, '\n', off);
        Str line = str_sub(rest, off, nl < 0 ? rest.len : (size_t)nl);
        off = nl < 0 ? rest.len : (size_t)nl + 1;
        if (!str_trim(line).len) continue;
        JValue *e = NULL;
        const char *id = NULL;
        if (!json_parse(a, line, false, &e) && json_is(e, J_OBJECT) && !str_eqz(json_strs(json_get(e, "type")), "session"))
            id = json_str(json_get(e, "id"));
        if (id) vec_push(&f->ids, arena_strdup(a, id)); // the header and lines that do not parse are not entries (pi skips them)
    }
    free(text);
}

// The run's journalled entries (first occurrence of each id), in order.
static void journal_entries(Arena *a, const JValue *recs, VEC(const JValue *) *out) {
    for (size_t i = 0; i < recs->arr.len; i++) {
        const JValue *r = recs->arr.items[i];
        if (!str_eqz(json_strs(json_get(r, "t")), "entry")) continue;
        const JValue *e = json_get(r, "entry");
        const char *id = json_str(json_get(e, "id"));
        if (!id) continue;
        bool dup = false;
        for (size_t k = 0; k < out->len && !dup; k++) dup = !strcmp(json_str(json_get(out->ptr[k], "id")), id);
        if (!dup) vec_push(out, e);
    }
    (void)a;
}

// What the session file holds after the run's leaf, and whether it is a prefix of the journal's
// entries: -1 stale (the session moved on without this run), else how many it holds.
static ptrdiff_t session_progress(const SessFile *f, const char *leaf, const JValue *const *ents, size_t nents) {
    size_t start = 0;
    if (leaf) {
        bool found = false;
        for (size_t i = 0; i < f->ids.len; i++)
            if (f->ids.ptr[i] && !strcmp(f->ids.ptr[i], leaf)) {
                start = i + 1;
                found = true;
            }
        if (!found) return -1;
    }
    size_t after = f->ids.len - start;
    if (after > nents) return -1;
    for (size_t k = 0; k < after; k++) {
        const char *id = f->ids.ptr[start + k];
        if (!id || strcmp(id, json_str(json_get(ents[k], "id")))) return -1;
    }
    return (ptrdiff_t)after;
}

static const JValue *run_record(const JValue *recs) { return recs->arr.len ? recs->arr.items[0] : NULL; }

static bool has_type(const JValue *recs, const char *t) {
    for (size_t i = 0; i < recs->arr.len; i++)
        if (str_eqz(json_strs(json_get(recs->arr.items[i], "t")), t)) return true;
    return false;
}

// ---- finding an interrupted run --------------------------------------------------------------------------

static bool ends_with(const char *s, const char *suf) {
    size_t n = strlen(s), m = strlen(suf);
    return n >= m && !strcmp(s + n - m, suf);
}

// A journal checked under its lock: NULL (and the file removed when it is of no use) unless it holds
// an interrupted run of a session that is still where the run left it.
static DurRecovery *check_journal(Arena *a, const char *jpath) {
    int fd = open(jpath, O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return NULL;
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) { // live
        close(fd);
        return NULL;
    }
    size_t valid;
    JValue *recs = read_records(a, fd, &valid);
    const JValue *run = run_record(recs);
    char *spath = arena_strndup(a, jpath, strlen(jpath) - 4); // without ".run"
    bool drop = !run || has_type(recs, "end") || num(json_get(run, "v")) != JOURNAL_FORMAT;
    SessFile f;
    if (!drop) {
        read_session(a, spath, &f);
        if (!f.exists && !json_is(json_get(run, "pre"), J_ARRAY)) drop = true; // the session is gone
        else if (f.exists) {
            VEC(const JValue *) ents = {0};
            journal_entries(a, recs, (void *)&ents);
            drop = session_progress(&f, json_str(json_get(run, "leaf")), ents.ptr, ents.len) < 0;
            vec_free(&ents);
        }
    }
    if (drop) {
        unlink(jpath);
        close(fd);
        return NULL;
    }
    DurRecovery *rec = arena_alloc(a, sizeof *rec);
    rec->journal = arena_strdup(a, jpath);
    rec->session = spath;
    rec->fd = fd;
    rec->records = recs;
    rec->nvalid = valid;
    rec->started_ms = (double)num(json_get(run, "time"));
    return rec;
}

DurRecovery *code_durable_find(Arena *a, const char *dir, const char *prefer) {
    DurRecovery *best = NULL;
    if (prefer) {
        char *jp = code_durable_path(a, prefer);
        if (file_exists(jp)) best = check_journal(a, jp);
        if (best) return best;
    }
    DIR *d = opendir(dir);
    if (!d) return NULL;
    struct dirent *de;
    while ((de = readdir(d))) {
        if (!ends_with(de->d_name, ".jsonl.run")) continue;
        DurRecovery *r = check_journal(a, path_join(a, dir, de->d_name));
        if (!r) continue;
        if (!best || r->started_ms > best->started_ms) {
            if (best) close(best->fd);
            best = r;
        } else close(r->fd);
    }
    closedir(d);
    return best;
}

void code_durable_release(DurRecovery *rec) {
    if (!rec || rec->fd < 0) return;
    close(rec->fd);
    rec->fd = -1;
}

void code_durable_drop(DurRecovery *rec) {
    if (!rec || rec->fd < 0) return;
    unlink(rec->journal);
    code_durable_release(rec);
}

// ---- reconciliation ---------------------------------------------------------------------------------------

static Err append_lines(const char *path, Str text, bool create) {
    int fd = open(path, O_WRONLY | O_APPEND | O_CLOEXEC | (create ? O_CREAT | O_EXCL : 0), 0644);
    if (fd < 0) return err_errno(errno);
    const char *p = text.ptr;
    size_t n = text.len;
    while (n > 0) {
        ssize_t w = write(fd, p, n);
        if (w < 0 && errno == EINTR) continue;
        if (w < 0) {
            close(fd);
            return err_errno(errno);
        }
        p += w;
        n -= (size_t)w;
    }
    close(fd);
    return OK;
}

static const char *display_name(Arena *a, CodeSession *s) {
    const JValue *name = code_session_last(s, "session_info", "name");
    if (json_str(name) && *json_str(name)) return json_str(name);
    const JValue *m = code_session_messages(s);
    for (size_t i = 0; i < m->arr.len; i++) {
        const JValue *msg = m->arr.items[i];
        if (!str_eqz(json_strs(json_get(msg, "role")), "user")) continue;
        Buf b = {0};
        content_text(json_get(msg, "content"), &b);
        Str t = str_trim(buf_str(&b));
        ptrdiff_t nl = str_find_char(t, '\n', 0);
        if (nl >= 0) t = str_sub(t, 0, (size_t)nl);
        char *r = t.len > 60 ? arena_printf(a, "%.*s...", 57, t.ptr) : str_z(a, t);
        buf_free(&b);
        return r;
    }
    return "untitled";
}

static const char *path_of(Arena *a, const char *cwd, const char *p) {
    if (!p) return NULL;
    if (p[0] == '@') p++;
    if (p[0] == '/') return p;
    return path_join(a, cwd, p);
}

// Whether write or edit looks applied, for the question (a hint: the user decides).
static const char *change_hint(Arena *a, const char *cwd, const JValue *call) {
    Str name = json_strs(json_get(call, "name"));
    const JValue *args = json_get(call, "arguments");
    const char *path = path_of(a, cwd, json_str(json_get(args, "path")));
    if (!path || (!str_eqz(name, "write") && !str_eqz(name, "edit"))) return NULL;
    char *text;
    size_t len;
    bool have = !read_file(path, (size_t)1 << 28, &text, &len);
    if (str_eqz(name, "write")) {
        Str want = json_strs(json_get(args, "content"));
        if (!have) return "The file does not exist: the write did not happen.";
        bool same = len == want.len && !memcmp(text, want.ptr, len);
        free(text);
        return same ? "The file already holds the content it was writing." : "The file does not hold that content.";
    }
    if (!have) return NULL;
    Str file = str_n(text, len);
    const JValue *edits = json_get(args, "edits");
    size_t n = 0, old_found = 0, new_found = 0;
    for (size_t i = 0; json_is(edits, J_ARRAY) && i < edits->arr.len; i++) {
        Str o = json_strs(json_get(edits->arr.items[i], "oldText")), nw = json_strs(json_get(edits->arr.items[i], "newText"));
        n++;
        if (o.len && str_find(file, o, 0) >= 0) old_found++;
        if (nw.len && str_find(file, nw, 0) >= 0) new_found++;
    }
    free(text);
    if (!n) return NULL;
    if (old_found == n) return "Its old text is still in the file: the edit was not applied.";
    if (new_found == n) return "Its new text is in the file: the edit looks applied.";
    return "The file matches neither the old text nor the new text of every edit.";
}

Err code_durable_reconcile(Arena *a, DurRecovery *rec, CodeSession **out) {
    const JValue *recs = rec->records;
    const JValue *run = run_record(recs);
    VEC(const JValue *) ents = {0};
    journal_entries(a, recs, (void *)&ents);
    SessFile f;
    read_session(a, rec->session, &f);
    Buf add = {0};
    bool create = false;
    size_t have = 0;
    if (!f.exists) {
        // the crash came before the file's first write: the entries before the run, then the run's
        const JValue *pre = json_get(run, "pre");
        for (size_t i = 0; json_is(pre, J_ARRAY) && i < pre->arr.len; i++) {
            json_write(&add, pre->arr.items[i]);
            buf_addc(&add, '\n');
        }
        create = true;
    } else {
        ptrdiff_t p = session_progress(&f, json_str(json_get(run, "leaf")), ents.ptr, ents.len);
        if (p < 0) {
            vec_free(&ents);
            return "SessionMovedOn";
        }
        have = (size_t)p;
        if (!f.ends_nl) buf_addc(&add, '\n'); // pi's repair of a torn last line
    }
    for (size_t k = have; k < ents.len; k++) {
        json_write(&add, ents.ptr[k]);
        buf_addc(&add, '\n');
    }
    size_t nents = ents.len;
    vec_free(&ents);
    Err e = OK;
    if (add.len && (!create || nents > 0)) e = append_lines(rec->session, buf_str(&add), create);
    buf_free(&add);
    if (e) return e;
    // nothing of the run reached the file (or the journal): the prompt runs again
    if (nents == 0) rec->prompt = json_str(json_get(run, "prompt"));
    if (create && nents == 0) {
        // the session never reached the disk: a new session takes the prompt
        *out = NULL;
    } else
        TRY(code_session_open_path(a, rec->session, out));
    // interruptions in a row: the recoveries since the run's last entry, and this one
    size_t last_entry = 0;
    for (size_t i = 0; i < recs->arr.len; i++)
        if (str_eqz(json_strs(json_get(recs->arr.items[i], "t")), "entry")) last_entry = i;
    int64_t again = 0;
    for (size_t i = last_entry; i < recs->arr.len; i++)
        if (str_eqz(json_strs(json_get(recs->arr.items[i], "t")), "recovered")) again++;
    rec->interruptions = again + 1;
    rec->limit = rec->interruptions >= DURABLE_RESTART_LIMIT;
    // the queued messages not delivered: a take counts once an entry follows it
    VEC(const char *) qt = {0};
    VEC(bool) qs = {0};
    VEC(bool) qtaken = {0};
    for (size_t i = 0; i < recs->arr.len; i++) {
        const JValue *r = recs->arr.items[i];
        Str t = json_strs(json_get(r, "t"));
        if (str_eqz(t, "queue")) {
            vec_push(&qt, json_str(json_get(r, "text")));
            vec_push(&qs, str_eqz(json_strs(json_get(r, "kind")), "steer"));
            vec_push(&qtaken, false);
        } else if (str_eqz(t, "take")) {
            bool delivered = false;
            for (size_t k = i + 1; k < recs->arr.len && !delivered; k++)
                delivered = str_eqz(json_strs(json_get(recs->arr.items[k], "t")), "entry");
            if (!delivered) continue;
            const char *text = json_str(json_get(r, "text"));
            for (size_t k = 0; k < qt.len; k++)
                if (!qtaken.ptr[k] && text && qt.ptr[k] && !strcmp(qt.ptr[k], text)) {
                    qtaken.ptr[k] = true;
                    break;
                }
        }
    }
    rec->queued = arena_alloc(a, (qt.len + 1) * sizeof *rec->queued);
    rec->queued_steer = arena_alloc(a, (qt.len + 1) * sizeof *rec->queued_steer);
    for (size_t k = 0; k < qt.len; k++)
        if (!qtaken.ptr[k] && qt.ptr[k]) {
            rec->queued[rec->nqueued] = qt.ptr[k];
            rec->queued_steer[rec->nqueued++] = qs.ptr[k];
        }
    vec_free(&qt);
    vec_free(&qs);
    vec_free(&qtaken);
    if (!*out) return OK;
    rec->name = display_name(a, *out);
    // the open round: the last assistant message's calls without a result after it
    const JValue *msgs = code_session_messages(*out);
    ptrdiff_t ai = -1;
    for (size_t i = msgs->arr.len; i-- > 0;) {
        Str role = json_strs(json_get(msgs->arr.items[i], "role"));
        if (str_eqz(role, "assistant")) {
            ai = (ptrdiff_t)i;
            break;
        }
        if (!str_eqz(role, "toolResult")) break; // a user message (or other) after it: no open round
    }
    const JValue *lastm = msgs->arr.len ? msgs->arr.items[msgs->arr.len - 1] : NULL;
    Str last_role = json_strs(json_get(lastm, "role"));
    if (ai >= 0) {
        const JValue *am = msgs->arr.items[ai];
        Str stop = json_strs(json_get(am, "stopReason"));
        const JValue *content = json_get(am, "content");
        size_t ncalls = 0;
        for (size_t i = 0; content && i < content->arr.len; i++)
            if (str_eqz(json_strs(json_get(content->arr.items[i], "type")), "toolCall")) ncalls++;
        if (ncalls && !str_eqz(stop, "error") && !str_eqz(stop, "aborted")) {
            rec->calls = arena_alloc(a, ncalls * sizeof *rec->calls);
            const char *cwd = json_str(json_get(code_session_header(*out), "cwd"));
            for (size_t i = 0; i < content->arr.len; i++) {
                const JValue *c = content->arr.items[i];
                if (!str_eqz(json_strs(json_get(c, "type")), "toolCall")) continue;
                DurCall *dc = &rec->calls[rec->ncalls++];
                dc->call = c;
                dc->id = json_str(json_get(c, "id"));
                dc->name = json_str(json_get(c, "name"));
                bool in_session = false;
                for (size_t k = (size_t)ai + 1; k < msgs->arr.len && !in_session; k++) {
                    const char *tid = json_str(json_get(msgs->arr.items[k], "toolCallId"));
                    in_session = dc->id && tid && !strcmp(dc->id, tid);
                }
                if (in_session) {
                    dc->state = DC_HAVE;
                    continue;
                }
                const JValue *intent = NULL, *result = NULL, *output = NULL;
                for (size_t k = 0; k < recs->arr.len; k++) {
                    const JValue *r = recs->arr.items[k];
                    if (!dc->id || !str_eqz(json_strs(json_get(r, "id")), dc->id)) continue;
                    Str t = json_strs(json_get(r, "t"));
                    if (str_eqz(t, "intent")) intent = r;
                    else if (str_eqz(t, "result")) result = r;
                    else if (str_eqz(t, "output")) output = r;
                }
                if (result) {
                    dc->state = DC_RECORDED;
                    dc->result = (JValue *)json_get(result, "message");
                } else if (!intent) dc->state = DC_NOT_STARTED;
                else {
                    dc->state = str_eqz(json_strs(json_get(intent, "policy")), "safe") && code_durable_safe(str(dc->name)) ? DC_SAFE : DC_ASK;
                    dc->output = json_strs(json_get(output, "text"));
                    if (dc->state == DC_ASK) dc->hint = change_hint(a, cwd ? cwd : "/", c);
                }
            }
        }
    }
    bool open_round = false;
    for (size_t i = 0; i < rec->ncalls; i++) open_round |= rec->calls[i].state != DC_HAVE;
    rec->waiting_model = !open_round && (str_eqz(last_role, "user") || str_eqz(last_role, "toolResult"));
    return OK;
}

void code_durable_adopt(DurRecovery *rec, CodeSession *s) {
    pthread_mutex_lock(&g_lock);
    close_l(false);
    if (rec->fd >= 0) {
        // the records after the last valid one go (a torn line), then the run goes on in this file
        int fl = fcntl(rec->fd, F_GETFL);
        if (ftruncate(rec->fd, (off_t)rec->nvalid) == 0 && fl >= 0 && fcntl(rec->fd, F_SETFL, fl | O_APPEND) == 0) {
            g_fd = rec->fd;
            rec->fd = -1;
            g_path = strdup(rec->journal);
            g_sess = s;
            g_seq = (int64_t)rec->records->arr.len;
            for (size_t i = 0; i < rec->nqueued; i++) vec_push(&g_queue, ((QItem){strdup(rec->queued[i]), rec->queued_steer[i], false, false}));
            Arena t = {0};
            JValue *r = json_new(&t, J_OBJECT);
            json_put(&t, r, S("interruptions"), json_int(&t, rec->interruptions));
            json_put(&t, r, S("pid"), json_int(&t, (int64_t)getpid()));
            JValue *dec = json_new(&t, J_ARRAY);
            for (size_t i = 0; i < rec->ncalls; i++) {
                const DurCall *c = &rec->calls[i];
                static const char *const names[] = {"have", "recorded", "not_started", "safe", "ask"};
                JValue *d = json_new(&t, J_OBJECT);
                json_put(&t, d, S("id"), json_string(&t, str(c->id)));
                json_put(&t, d, S("state"), json_string(&t, str(names[c->state])));
                json_put(&t, d, S("rerun"), json_bool(&t, c->rerun));
                json_push(&t, dec, d);
            }
            json_put(&t, r, S("calls"), dec);
            put_l("recovered", r, true);
            arena_reset(&t);
        }
    }
    pthread_mutex_unlock(&g_lock);
}

const char *code_durable_running(Arena *a, DurRecovery *rec) {
    const JValue *recs = rec->records;
    const JValue *open = NULL;
    for (size_t i = 0; i < recs->arr.len; i++) {
        const JValue *r = recs->arr.items[i];
        Str t = json_strs(json_get(r, "t"));
        if (str_eqz(t, "intent")) open = r;
        else if (str_eqz(t, "result") && open && str_eqz(json_strs(json_get(r, "id")), json_str(json_get(open, "id")))) open = NULL;
    }
    if (!rec->name) {
        CodeSession *s;
        rec->name = file_exists(rec->session) && !code_session_open_path(a, rec->session, &s) ? display_name(a, s) : "new session";
    }
    if (open) return arena_printf(a, "It was running: %s", code_durable_call_summary(a, open));
    const JValue *last = recs->arr.len ? recs->arr.items[recs->arr.len - 1] : NULL;
    if (str_eqz(json_strs(json_get(last, "t")), "request")) return "It was waiting for the model's answer.";
    return "It stopped between two steps.";
}

// ---- texts for the prompt ---------------------------------------------------------------------------------

const char *code_durable_ago(Arena *a, double ms) {
    double now = (double)(wall_ns() / 1000000);
    int64_t s = (int64_t)((now - ms) / 1000);
    if (s < 60) return "less than a minute";
    int64_t m = s / 60;
    if (m < 60) return arena_printf(a, "%lld minute%s", (long long)m, m == 1 ? "" : "s");
    int64_t h = m / 60;
    if (h < 48) return arena_printf(a, "%lld hour%s", (long long)h, h == 1 ? "" : "s");
    int64_t d = h / 24;
    return arena_printf(a, "%lld days", (long long)d);
}

const char *code_durable_call_summary(Arena *a, const JValue *call) {
    Str name = json_strs(json_get(call, "name"));
    const JValue *args = json_get(call, "arguments");
    Str what = json_strs(json_get(args, str_eqz(name, "bash") ? "command" : "path"));
    if (!what.len) what = json_strs(json_get(args, "pattern"));
    if (!what.len) what = json_strs(json_get(args, "description"));
    ptrdiff_t nl = str_find_char(what, '\n', 0);
    if (nl >= 0) what = str_sub(what, 0, (size_t)nl);
    if (what.len > 80) return arena_printf(a, "%.*s  %.*s...", (int)name.len, name.ptr, 77, what.ptr);
    return what.len ? arena_printf(a, "%.*s  %.*s", (int)name.len, name.ptr, (int)what.len, what.ptr) : str_z(a, name);
}
