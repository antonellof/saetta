// Saetta Code sessions (see code.h): pi's session files, version 3, one JSON entry per line.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi v1.5.2,
// packages/coding-agent/src/core/session-manager.ts).
//
// The file is created only once the session holds a user or assistant message (pi's
// _hasConversation); before that, entries stay in memory. Entries form a tree by parentId; the
// current branch is the path from the root to the last entry appended (the leaf).
#include "code.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

struct CodeSession {
    Arena *a;
    char *path; // NULL when not persisted
    bool flushed;
    JValue *entries;  // every entry, the header first
    JValue *messages; // the messages of the current branch, rebuilt on demand
    const char *leaf; // id of the last entry, NULL before the first
    const char *id;
};

// ---- ids and times ------------------------------------------------------------------------------

static void random_bytes(uint8_t *p, size_t n) { arc4random_buf(p, n); }

// randomUUID().slice(0, 8): 8 lowercase hex digits.
static char *short_id(Arena *a, const JValue *entries) {
    for (int tries = 0; tries < 100; tries++) {
        uint8_t b[4];
        random_bytes(b, sizeof b);
        char *id = arena_printf(a, "%02x%02x%02x%02x", b[0], b[1], b[2], b[3]);
        bool used = false;
        for (size_t i = 0; i < entries->arr.len && !used; i++)
            used = str_eqz(json_strs(json_get(entries->arr.items[i], "id")), id);
        if (!used) return id;
    }
    return arena_printf(a, "%08x", arc4random());
}

// uuidv7: 48-bit Unix ms, version 7, variant 10, the rest random.
static char *uuid_v7(Arena *a, double now_ms) {
    uint8_t b[16];
    random_bytes(b, sizeof b);
    uint64_t ms = (uint64_t)now_ms;
    for (int i = 0; i < 6; i++) b[i] = (uint8_t)(ms >> (40 - 8 * i));
    b[6] = (uint8_t)(0x70 | (b[6] & 0x0F));
    b[8] = (uint8_t)(0x80 | (b[8] & 0x3F));
    return arena_printf(a, "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", b[0], b[1], b[2], b[3], b[4], b[5],
                        b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
}

// new Date(ms).toISOString(): 2026-11-22T08:37:11.713Z
static char *iso_time(Arena *a, double now_ms) {
    time_t s = (time_t)(now_ms / 1000);
    int ms = (int)((uint64_t)now_ms % 1000);
    struct tm t;
    gmtime_r(&s, &t);
    return arena_printf(a, "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min,
                        t.tm_sec, ms);
}

static double wall_ms(void) { return (double)(wall_ns() / 1000000); }

// ---- files ---------------------------------------------------------------------------------------

// agent_dir/sessions/--<cwd with / and : as ->--
static char *session_dir(Arena *a, const CodeEnv *env) {
    const char *c = env->cwd;
    while (*c == '/') c++;
    char *safe = arena_printf(a, "--%s--", c);
    for (char *p = safe + 2; *p; p++)
        if (*p == '/' || *p == '\\' || *p == ':') *p = '-';
    return path_join(a, path_join(a, env->agent_dir, "sessions"), safe);
}

static void write_line(Buf *b, const JValue *v) {
    json_write(b, v);
    buf_addc(b, '\n');
}

static Err append_text(const char *path, Str text, bool create_excl) {
    int fd = open(path, O_WRONLY | O_APPEND | O_CLOEXEC | (create_excl ? O_CREAT | O_EXCL : 0), 0644);
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

static bool has_conversation(const JValue *entries) {
    for (size_t i = 0; i < entries->arr.len; i++) {
        const JValue *e = entries->arr.items[i];
        if (!str_eqz(json_strs(json_get(e, "type")), "message")) continue;
        Str role = json_strs(json_get(json_get(e, "message"), "role"));
        if (str_eqz(role, "user") || str_eqz(role, "assistant")) return true;
    }
    return false;
}

static Err persist(CodeSession *s, const JValue *entry) {
    if (!s->path) return OK;
    Buf b = {0};
    Err e = OK;
    if (!s->flushed) {
        if (!has_conversation(s->entries)) return OK;
        for (size_t i = 0; i < s->entries->arr.len; i++) write_line(&b, s->entries->arr.items[i]);
        e = append_text(s->path, buf_str(&b), true);
        if (!e) s->flushed = true;
    } else {
        write_line(&b, entry);
        e = append_text(s->path, buf_str(&b), false);
    }
    buf_free(&b);
    return e;
}

// ---- the most recent session for a cwd (findMostRecentSession) ---------------------------------

static char *most_recent(Arena *a, const char *dir, const char *cwd) {
    DIR *d = opendir(dir);
    if (!d) return NULL;
    char *best = NULL;
    double best_mtime = -1;
    struct dirent *de;
    while ((de = readdir(d))) {
        size_t n = strlen(de->d_name);
        if (n < 6 || strcmp(de->d_name + n - 6, ".jsonl") != 0) continue;
        char *p = path_join(a, dir, de->d_name);
        struct stat st;
        if (stat(p, &st) != 0) continue;
        double mt = (double)st.st_mtimespec.tv_sec * 1e3 + (double)st.st_mtimespec.tv_nsec / 1e6;
        if (mt <= best_mtime) continue;
        // the header's cwd must match
        char *text;
        size_t len;
        if (read_file(p, (size_t)1 << 20, &text, &len)) continue;
        Str s = str_n(text, len);
        ssize_t nl = str_find(s, S("\n"), 0);
        JValue *h = NULL;
        bool ok = !json_parse(a, nl >= 0 ? str_sub(s, 0, (size_t)nl) : s, true, &h) && h &&
                  str_eqz(json_strs(json_get(h, "type")), "session") && str_eqz(json_strs(json_get(h, "cwd")), cwd);
        free(text);
        if (!ok) continue;
        best = p;
        best_mtime = mt;
    }
    closedir(d);
    return best;
}

static Err load(CodeSession *s, const char *path) {
    char *text;
    size_t len;
    TRY(read_file(path, (size_t)1 << 30, &text, &len));
    Str rest = str_n(text, len);
    while (rest.len > 0) {
        ssize_t nl = str_find(rest, S("\n"), 0);
        Str line = nl >= 0 ? str_sub(rest, 0, (size_t)nl) : rest;
        rest = nl >= 0 ? str_sub(rest, (size_t)nl + 1, rest.len) : S("");
        if (str_trim(line).len == 0) continue;
        JValue *e;
        if (json_parse(s->a, line, true, &e) || !e || e->kind != J_OBJECT) continue; // pi skips bad lines
        json_push(s->a, s->entries, e);
        if (str_eqz(json_strs(json_get(e, "type")), "session")) s->id = json_str(json_get(e, "id"));
        else s->leaf = json_str(json_get(e, "id"));
    }
    free(text);
    s->path = arena_strdup(s->a, path);
    s->flushed = true;
    return OK;
}

// ---- public -------------------------------------------------------------------------------------

Err code_session_open(Arena *a, const CodeEnv *env, bool cont, bool no_session, CodeSession **out) {
    CodeSession *s = arena_alloc(a, sizeof *s);
    s->a = a;
    s->entries = json_new(a, J_ARRAY);
    char *dir = session_dir(a, env);
    if (cont && !no_session) {
        char *p = most_recent(a, dir, env->cwd);
        if (p) {
            TRY(load(s, p));
            *out = s;
            return OK;
        }
    }
    double now = env->now_ms > 0 ? env->now_ms : wall_ms();
    s->id = uuid_v7(a, now);
    char *ts = iso_time(a, now);
    JValue *h = json_new(a, J_OBJECT);
    json_put(a, h, S("type"), json_string(a, S("session")));
    json_put(a, h, S("version"), json_int(a, 3));
    json_put(a, h, S("id"), json_string(a, str(s->id)));
    json_put(a, h, S("timestamp"), json_string(a, str(ts)));
    json_put(a, h, S("cwd"), json_string(a, str(env->cwd)));
    json_push(a, s->entries, h);
    if (!no_session) {
        TRY(mkdir_p(dir));
        char *fts = arena_strdup(a, ts);
        for (char *p = fts; *p; p++)
            if (*p == ':' || *p == '.') *p = '-';
        s->path = path_join(a, dir, arena_printf(a, "%s_%s.jsonl", fts, s->id));
    }
    *out = s;
    return OK;
}

const JValue *code_session_header(const CodeSession *s) { return s->entries->arr.items[0]; }

// The path from the root to the leaf, as entries.
static void branch(const CodeSession *s, JValue *out) {
    size_t n = 0;
    const char *id = s->leaf;
    // walk up from the leaf, then reverse
    VEC(const JValue *) up = {0};
    while (id) {
        const JValue *found = NULL;
        for (size_t i = s->entries->arr.len; i-- > 0;) {
            const JValue *e = s->entries->arr.items[i];
            if (str_eqz(json_strs(json_get(e, "id")), id)) {
                found = e;
                break;
            }
        }
        if (!found) break;
        vec_push(&up, found);
        const JValue *p = json_get(found, "parentId");
        id = json_is(p, J_STRING) ? p->s.ptr : NULL;
        if (++n > s->entries->arr.len) break; // a cycle
    }
    for (size_t i = up.len; i-- > 0;) json_push(s->a, out, (JValue *)up.ptr[i]);
    vec_free(&up);
}

const JValue *code_session_messages(const CodeSession *s) {
    CodeSession *m = (CodeSession *)s;
    JValue *path = json_new(s->a, J_ARRAY);
    branch(s, path);
    m->messages = json_new(s->a, J_ARRAY);
    for (size_t i = 0; i < path->arr.len; i++) {
        const JValue *e = path->arr.items[i];
        if (str_eqz(json_strs(json_get(e, "type")), "message")) json_push(s->a, m->messages, (JValue *)json_get(e, "message"));
    }
    return m->messages;
}

// The last value of an entry field on the current branch (thinking level, model).
const JValue *code_session_last(const CodeSession *s, const char *type, const char *field) {
    JValue *path = json_new(s->a, J_ARRAY);
    branch(s, path);
    for (size_t i = path->arr.len; i-- > 0;) {
        const JValue *e = path->arr.items[i];
        if (str_eqz(json_strs(json_get(e, "type")), type)) return json_get(e, field);
    }
    return NULL;
}

bool code_session_is_new(const CodeSession *s) { return s->entries->arr.len == 1; }

Err code_session_append(CodeSession *s, const char *type, JValue *fields) {
    Arena *a = s->a;
    JValue *e = json_new(a, J_OBJECT);
    json_put(a, e, S("type"), json_string(a, str(type)));
    const char *id = short_id(a, s->entries);
    json_put(a, e, S("id"), json_string(a, str(id)));
    json_put(a, e, S("parentId"), s->leaf ? json_string(a, str(s->leaf)) : json_new(a, J_NULL));
    json_put(a, e, S("timestamp"), json_string(a, str(iso_time(a, wall_ms()))));
    if (fields)
        for (size_t i = 0; i < fields->obj.len; i++) json_put(a, e, fields->obj.items[i].key, fields->obj.items[i].val);
    json_push(a, s->entries, e);
    const char *prev = s->leaf;
    s->leaf = id;
    Err err = persist(s, e);
    if (err) { // keep memory in line with the file
        s->entries->arr.len--;
        s->leaf = prev;
    }
    return err;
}
