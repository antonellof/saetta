// Saetta Code sessions (see code.h): pi's session files, version 3, one JSON entry per line.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi v1.5.2,
// b4afddd: packages/coding-agent/src/core/session-manager.ts (the file format, buildContextEntries,
// sessionEntryToContextMessages, projectContextEntry, findMostRecentSession), core/messages.ts
// (convertToLlm for summaries, custom and bash messages)).
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

// The file's first line, without reading the rest; NULL when longer than max or unreadable.
static char *first_line(Arena *a, const char *path, size_t max) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    Buf b = {0};
    int c;
    while ((c = fgetc(f)) != EOF && c != '\n') {
        if (b.len >= max) {
            fclose(f);
            buf_free(&b);
            return NULL;
        }
        buf_addc(&b, (char)c);
    }
    fclose(f);
    char *r = arena_strndup(a, b.ptr ? b.ptr : "", b.len);
    buf_free(&b);
    return r;
}

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
        // the header's cwd must match; only the first line is read (pi's header scan, at most 1 MiB)
        char *line = first_line(a, p, (size_t)1 << 20);
        JValue *h = NULL;
        bool ok = line && !json_parse(a, str(line), true, &h) && h && str_eqz(json_strs(json_get(h, "type")), "session") &&
                  str_eqz(json_strs(json_get(h, "cwd")), cwd);
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

// ---- UTF-8 ---------------------------------------------------------------------------------------

// The length of the well-formed sequence at p (1 to 4), or 0 with *bad the length of its
// maximal ill-formed subpart (Unicode 3.9, as TextDecoder replaces).
static size_t u8_len(const unsigned char *p, size_t n, size_t *bad) {
    unsigned char c = p[0];
    *bad = 1;
    if (c < 0x80) return 1;
    size_t need;
    unsigned char lo = 0x80, hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF) need = 1;
    else if (c >= 0xE0 && c <= 0xEF) {
        need = 2;
        if (c == 0xE0) lo = 0xA0;
        if (c == 0xED) hi = 0x9F;
    } else if (c >= 0xF0 && c <= 0xF4) {
        need = 3;
        if (c == 0xF0) lo = 0x90;
        if (c == 0xF4) hi = 0x8F;
    } else return 0;
    for (size_t k = 1; k <= need; k++) {
        if (k >= n) return 0;
        unsigned char x = p[k];
        if (x < (k == 1 ? lo : 0x80) || x > (k == 1 ? hi : 0xBF)) return 0;
        *bad = k + 1;
    }
    return need + 1;
}

Str code_utf8_clean(Arena *a, Str s) {
    const unsigned char *p = (const unsigned char *)s.ptr;
    size_t i = 0, bad;
    while (i < s.len) {
        size_t l = u8_len(p + i, s.len - i, &bad);
        if (!l) break;
        i += l;
    }
    if (i == s.len) return s;
    Buf b = {0};
    buf_add(&b, s.ptr, i);
    while (i < s.len) {
        size_t l = u8_len(p + i, s.len - i, &bad);
        if (l) {
            buf_add(&b, s.ptr + i, l);
            i += l;
        } else {
            buf_adds(&b, "\xef\xbf\xbd");
            i += bad > s.len - i ? s.len - i : bad;
        }
    }
    Str r = {arena_strndup(a, b.ptr, b.len), b.len};
    buf_free(&b);
    return r;
}

// ---- the context (buildSessionContext: buildContextEntries, sessionEntryToContextMessages and
// convertToLlm for the roles the client does not know) -----------------------------------------

// new Date(iso).getTime() for pi's own timestamps (2026-11-22T08:37:11.713Z)
static int64_t iso_ms(Str t) {
    struct tm tm = {0};
    int ms = 0;
    char buf[40];
    size_t n = t.len < sizeof buf - 1 ? t.len : sizeof buf - 1;
    memcpy(buf, t.ptr, n);
    buf[n] = 0;
    if (sscanf(buf, "%4d-%2d-%2dT%2d:%2d:%2d.%3dZ", &tm.tm_year, &tm.tm_mon, &tm.tm_mday, &tm.tm_hour, &tm.tm_min, &tm.tm_sec, &ms) < 6) return 0;
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;
    return (int64_t)timegm(&tm) * 1000 + ms;
}

static JValue *text_user(Arena *a, const char *text, int64_t ts) {
    JValue *m = json_new(a, J_OBJECT);
    json_put(a, m, S("role"), json_string(a, S("user")));
    JValue *c = json_new(a, J_ARRAY), *b = json_new(a, J_OBJECT);
    json_put(a, b, S("type"), json_string(a, S("text")));
    json_put(a, b, S("text"), json_string(a, str(text)));
    json_push(a, c, b);
    json_put(a, m, S("content"), c);
    json_put(a, m, S("timestamp"), json_int(a, ts));
    return m;
}

// messages.ts bashExecutionToText
static char *bash_text(Arena *a, const JValue *m) {
    Buf b = {0};
    Str cmd = json_strs(json_get(m, "command")), out = json_strs(json_get(m, "output"));
    buf_printf(&b, "Ran `%.*s`\n", (int)cmd.len, cmd.ptr);
    if (out.len) buf_printf(&b, "```\n%.*s\n```", (int)out.len, out.ptr);
    else buf_adds(&b, "(no output)");
    const JValue *ec = json_get(m, "exitCode");
    if (json_is(json_get(m, "cancelled"), J_BOOL) && json_get(m, "cancelled")->b) buf_adds(&b, "\n\n(command cancelled)");
    else if (json_is(ec, J_INTEGER) && ec->i != 0) buf_printf(&b, "\n\nCommand exited with code %lld", (long long)ec->i);
    Str full = json_strs(json_get(m, "fullOutputPath"));
    if (json_is(json_get(m, "truncated"), J_BOOL) && json_get(m, "truncated")->b && full.len)
        buf_printf(&b, "\n\n[Output truncated. Full output: %.*s]", (int)full.len, full.ptr);
    char *r = arena_strndup(a, b.ptr ? b.ptr : "", b.len);
    buf_free(&b);
    return r;
}

// One entry's model-visible messages.
static void entry_messages(Arena *a, const JValue *e, const JValue *edit, JValue *out) {
    Str type = json_strs(json_get(e, "type"));
    int64_t ts = iso_ms(json_strs(json_get(e, "timestamp")));
    JValue *m = NULL;
    if (str_eqz(type, "message")) {
        m = (JValue *)json_get(e, "message");
        if (!m) return;
        Str role = json_strs(json_get(m, "role"));
        const JValue *c = json_get(m, "content");
        if (!c || c->kind == J_NULL) { // old or hand-edited files
            JValue *copy = json_new(a, J_OBJECT);
            for (size_t i = 0; i < m->obj.len; i++) json_put(a, copy, m->obj.items[i].key, m->obj.items[i].val);
            json_put(a, copy, S("content"), str_eqz(role, "system") ? json_string(a, S("")) : json_new(a, J_ARRAY));
            m = copy;
        }
        if (str_eqz(role, "bashExecution")) {
            if (json_is(json_get(m, "excludeFromContext"), J_BOOL) && json_get(m, "excludeFromContext")->b) return;
            const JValue *t = json_get(m, "timestamp");
            m = text_user(a, bash_text(a, m), json_is(t, J_INTEGER) ? t->i : ts);
        } else if (str_eqz(role, "custom")) {
            const JValue *cc = json_get(m, "content");
            const JValue *t = json_get(m, "timestamp");
            JValue *u = text_user(a, "", json_is(t, J_INTEGER) ? t->i : ts);
            if (json_is(cc, J_STRING)) json_put(a, json_get(u, "content")->arr.items[0], S("text"), (JValue *)cc);
            else json_put(a, u, S("content"), (JValue *)cc);
            m = u;
        }
    } else if (str_eqz(type, "custom_message")) {
        const JValue *cc = json_get(e, "content");
        JValue *u = text_user(a, "", ts);
        if (json_is(cc, J_STRING)) json_put(a, json_get(u, "content")->arr.items[0], S("text"), (JValue *)cc);
        else if (cc) json_put(a, u, S("content"), (JValue *)cc);
        else json_put(a, u, S("content"), json_new(a, J_ARRAY));
        m = u;
    } else if (str_eqz(type, "branch_summary") && json_strs(json_get(e, "summary")).len) {
        Str sum = json_strs(json_get(e, "summary"));
        m = text_user(a, arena_printf(a, "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n%.*s</summary>", (int)sum.len, sum.ptr), ts);
    } else if (str_eqz(type, "compaction")) {
        const JValue *sys = json_get(e, "systemMessage");
        if (sys) json_push(a, out, (JValue *)sys);
        Str sum = json_strs(json_get(e, "summary"));
        m = text_user(a, arena_printf(a, "The conversation history before this point was compacted into the following summary:\n\n<summary>\n%.*s\n</summary>", (int)sum.len, sum.ptr), ts);
    }
    if (!m) return;
    if (edit) { // context_edit: replace or drop the content
        const JValue *rep = json_get(edit, "replacement");
        if (!rep || rep->kind == J_NULL) return;
        Str role = json_strs(json_get(m, "role"));
        if (str_eqz(role, "user") || str_eqz(role, "assistant") || str_eqz(role, "toolResult")) {
            JValue *copy = json_new(a, J_OBJECT);
            for (size_t i = 0; i < m->obj.len; i++) json_put(a, copy, m->obj.items[i].key, m->obj.items[i].val);
            const JValue *rc = json_get(rep, "content");
            if (json_is(rc, J_STRING) && !str_eqz(role, "user")) {
                JValue *arr = json_new(a, J_ARRAY), *b = json_new(a, J_OBJECT);
                json_put(a, b, S("type"), json_string(a, S("text")));
                json_put(a, b, S("text"), (JValue *)rc);
                json_push(a, arr, b);
                json_put(a, copy, S("content"), arr);
            } else json_put(a, copy, S("content"), (JValue *)rc);
            m = copy;
        }
    }
    json_push(a, out, m);
}

const JValue *code_session_messages(const CodeSession *s) {
    CodeSession *m = (CodeSession *)s;
    Arena *a = s->a;
    JValue *path = json_new(a, J_ARRAY);
    branch(s, path);
    // the latest compaction: itself, the kept entries before it (but their system messages),
    // then everything after it
    ptrdiff_t ci = -1;
    for (size_t i = 0; i < path->arr.len; i++)
        if (str_eqz(json_strs(json_get(path->arr.items[i], "type")), "compaction")) ci = (ptrdiff_t)i;
    JValue *ctx = path;
    if (ci >= 0) {
        ctx = json_new(a, J_ARRAY);
        const JValue *comp = path->arr.items[ci];
        json_push(a, ctx, (JValue *)comp);
        Str first = json_strs(json_get(comp, "firstKeptEntryId"));
        bool kept = false;
        for (ptrdiff_t i = 0; i < ci; i++) {
            const JValue *e = path->arr.items[i];
            if (str_eq(json_strs(json_get(e, "id")), first)) kept = true;
            bool sys = str_eqz(json_strs(json_get(e, "type")), "message") && str_eqz(json_strs(json_get(json_get(e, "message"), "role")), "system");
            if (kept && !sys) json_push(a, ctx, (JValue *)e);
        }
        for (size_t i = (size_t)ci + 1; i < path->arr.len; i++) json_push(a, ctx, path->arr.items[i]);
    }
    m->messages = json_new(a, J_ARRAY);
    for (size_t i = 0; i < ctx->arr.len; i++) {
        const JValue *e = ctx->arr.items[i];
        // an older compaction kept inside the newest range contributes nothing
        if (i > 0 && str_eqz(json_strs(json_get(e, "type")), "compaction")) continue;
        // the latest context_edit targeting this entry
        const JValue *edit = NULL;
        Str id = json_strs(json_get(e, "id"));
        for (size_t k = 0; k < ctx->arr.len; k++) {
            const JValue *x = ctx->arr.items[k];
            if (str_eqz(json_strs(json_get(x, "type")), "context_edit") && str_eq(json_strs(json_get(x, "targetId")), id)) edit = x;
        }
        entry_messages(a, e, edit, m->messages);
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

const char *code_session_path(const CodeSession *s) { return s->path; }

const JValue *code_session_entries(const CodeSession *s) { return s->entries; }

void code_session_set_leaf(CodeSession *s, const char *id) { s->leaf = id ? arena_strdup(s->a, id) : NULL; }

const char *code_session_leaf(const CodeSession *s) { return s->leaf; }

const JValue *code_session_entry_last(const CodeSession *s) { return s->entries->arr.items[s->entries->arr.len - 1]; }

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
