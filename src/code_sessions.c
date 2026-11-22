// Saetta Code's session commands on session files (see code_sessions.h).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi v1.5.2,
// b4afddd): packages/coding-agent/src/core/session-manager.ts, core/agent-session-runtime.ts,
// core/agent-session.ts, core/compaction/compaction.ts and utils.ts, core/messages.ts,
// core/usage-totals.ts. Each function names the pi function it ports.
#include "code_sessions.h"

#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

// ---- helpers -------------------------------------------------------------------------------------

static double wall_ms(void) { return (double)(wall_ns() / 1000000); }

static bool jtrue(const JValue *v) { return json_is(v, J_BOOL) && v->b; }

static double jnum(const JValue *v) {
    if (json_is(v, J_INTEGER)) return (double)v->i;
    if (json_is(v, J_FLOAT)) return v->f;
    if (json_is(v, J_NUMBER_STRING)) return strtod(v->s.ptr, NULL);
    return 0;
}

static bool jhas_num(const JValue *v) { return json_is(v, J_INTEGER) || json_is(v, J_FLOAT) || json_is(v, J_NUMBER_STRING); }

// A number as JSON.stringify writes it (an integer when integral).
static JValue *jnumber(Arena *a, double d) {
    if (d == floor(d) && fabs(d) < 9e15) return json_int(a, (int64_t)d);
    JValue *v = json_new(a, J_FLOAT);
    v->f = d;
    return v;
}

static Str type_of(const JValue *e) { return json_strs(json_get(e, "type")); }

static Str role_of(const JValue *m) { return json_strs(json_get(m, "role")); }

// JavaScript's string length: UTF-16 code units.
static size_t js_len(Str s) {
    size_t n = 0;
    for (size_t i = 0; i < s.len; i++) {
        unsigned char c = (unsigned char)s.ptr[i];
        if ((c & 0xC0) == 0x80) continue;
        n += c >= 0xF0 ? 2 : 1;
    }
    return n;
}

// new Date(iso).getTime(), NAN when it does not parse.
static double iso_ms(Str t) {
    char buf[40];
    size_t n = t.len < sizeof buf - 1 ? t.len : sizeof buf - 1;
    memcpy(buf, t.ptr, n);
    buf[n] = 0;
    struct tm tm = {0};
    int ms = 0;
    int got = sscanf(buf, "%4d-%2d-%2dT%2d:%2d:%2d.%3dZ", &tm.tm_year, &tm.tm_mon, &tm.tm_mday, &tm.tm_hour, &tm.tm_min, &tm.tm_sec, &ms);
    if (got < 6) return NAN;
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;
    return (double)timegm(&tm) * 1000 + ms;
}

static JValue *text_block(Arena *a, Str text) {
    JValue *b = json_new(a, J_OBJECT);
    json_put(a, b, S("type"), json_string(a, S("text")));
    json_put(a, b, S("text"), json_string(a, text));
    return b;
}

// contentText(content, separator): the text blocks joined.
static void content_text(Buf *b, const JValue *content, const char *sep) {
    if (json_is(content, J_STRING)) {
        buf_addstr(b, content->s);
        return;
    }
    bool first = true;
    for (size_t i = 0; json_is(content, J_ARRAY) && i < content->arr.len; i++) {
        const JValue *blk = content->arr.items[i];
        if (!str_eqz(type_of(blk), "text")) continue;
        if (!first) buf_adds(b, sep);
        buf_addstr(b, json_strs(json_get(blk, "text")));
        first = false;
    }
}

static JValue *shallow(Arena *a, const JValue *o) {
    JValue *c = json_new(a, J_OBJECT);
    for (size_t i = 0; json_is(o, J_OBJECT) && i < o->obj.len; i++) json_put(a, c, o->obj.items[i].key, o->obj.items[i].val);
    return c;
}

// ---- the session folder (getDefaultSessionDir) ------------------------------------------------------

char *code_sessions_dir(Arena *a, const char *agent_dir, const char *cwd) {
    const char *c = cwd;
    if (*c == '/' || *c == '\\') c++;
    char *safe = arena_printf(a, "--%s--", c);
    for (char *p = safe + 2; p[2]; p++)
        if (*p == '/' || *p == '\\' || *p == ':') *p = '-';
    return path_join(a, path_join(a, agent_dir, "sessions"), safe);
}

// ---- the session list (buildSessionInfo, listSessionsFromDir, list, listAll) ---------------------

// extractTextContent: a message's text blocks joined with spaces.
static void message_text(Buf *b, const JValue *m) { content_text(b, json_get(m, "content"), " "); }

// buildSessionInfo: NULL when the file is not a session.
static bool session_info(Arena *a, const char *path, const struct stat *st, CodeSessionInfo *out) {
    char *text;
    size_t len;
    if (read_file(path, (size_t)1 << 31, &text, &len)) return false;
    const JValue *header = NULL;
    const char *name = NULL;
    size_t count = 0;
    double last = NAN;
    Buf all = {0}, first = {0};
    bool have_first = false, have_all = false;
    Str rest = str_n(text, len);
    bool ok = true;
    while (rest.len > 0 && ok) {
        ptrdiff_t nl = str_find(rest, S("\n"), 0);
        Str line = nl >= 0 ? str_sub(rest, 0, (size_t)nl) : rest;
        rest = nl >= 0 ? str_sub(rest, (size_t)nl + 1, rest.len) : S("");
        if (line.len && line.ptr[line.len - 1] == '\r') line.len--; // crlfDelay
        if (!str_trim(line).len) continue;
        JValue *e;
        if (json_parse(a, line, true, &e) || !json_is(e, J_OBJECT)) continue;
        if (!header) {
            if (!str_eqz(type_of(e), "session")) ok = false;
            header = e;
            continue;
        }
        if (str_eqz(type_of(e), "session_info")) {
            Str n = str_trim(json_strs(json_get(e, "name")));
            name = n.len ? str_z(a, n) : NULL;
        }
        if (!str_eqz(type_of(e), "message")) continue;
        count++;
        const JValue *m = json_get(e, "message");
        Str role = role_of(m);
        bool content = json_get(m, "content") != NULL;
        if (!json_is(json_get(m, "role"), J_STRING) || !content) continue;
        if (!str_eqz(role, "user") && !str_eqz(role, "assistant")) continue;
        // getMessageActivityTime
        const JValue *ts = json_get(m, "timestamp");
        double t = jhas_num(ts) ? jnum(ts) : iso_ms(json_strs(json_get(e, "timestamp")));
        if (!isnan(t)) last = isnan(last) ? fmax(0, t) : fmax(last, t);
        Buf tb = {0};
        message_text(&tb, m);
        if (tb.len) {
            if (have_all) buf_addc(&all, ' ');
            buf_addstr(&all, buf_str(&tb));
            have_all = true;
            if (!have_first && str_eqz(role, "user")) {
                buf_addstr(&first, buf_str(&tb));
                have_first = true;
            }
        }
        buf_free(&tb);
    }
    free(text);
    if (!ok || !header) {
        buf_free(&all);
        buf_free(&first);
        return false;
    }
    *out = (CodeSessionInfo){0};
    out->path = arena_strdup(a, path);
    out->id = json_str(json_get(header, "id")) ? json_str(json_get(header, "id")) : "";
    out->cwd = json_str(json_get(header, "cwd")) ? json_str(json_get(header, "cwd")) : "";
    out->name = name;
    out->parent = json_str(json_get(header, "parentSession"));
    double ht = json_is(json_get(header, "timestamp"), J_STRING) ? iso_ms(json_strs(json_get(header, "timestamp"))) : NAN;
    out->created = ht;
    double mtime = (double)st->st_mtimespec.tv_sec * 1e3 + floor((double)st->st_mtimespec.tv_nsec / 1e6);
    out->modified = !isnan(last) && last > 0 ? last : !isnan(ht) ? ht : mtime;
    out->message_count = count;
    out->first_message = have_first && first.len ? arena_strndup(a, first.ptr, first.len) : "(no messages)";
    out->all_text = arena_strndup(a, all.ptr ? all.ptr : "", all.len);
    buf_free(&all);
    buf_free(&first);
    return true;
}

// sortSessionInfos: by modified, the newest first (a stable sort, as Array.prototype.sort is).
static void sort_infos(CodeSessionInfo *v, size_t n) {
    for (size_t i = 1; i < n; i++) {
        CodeSessionInfo x = v[i];
        size_t j = i;
        while (j > 0 && v[j - 1].modified < x.modified) {
            v[j] = v[j - 1];
            j--;
        }
        v[j] = x;
    }
}

typedef struct {
    char *path;
    struct stat st;
    bool have_st;
} Candidate;

static int name_desc(const void *x, const void *y) { return strcmp(*(char *const *)y, *(char *const *)x); }

// The infos of these files in order, with pi's progress calls: publish says which loads publish
// the partial list.
static size_t load_infos(Arena *a, Candidate *files, size_t n, bool all_scope, CodeSessionProgress cb, void *ctx, CodeSessionInfo **out) {
    CodeSessionInfo *res = arena_alloc(a, (n + 1) * sizeof *res);
    CodeSessionInfo *partial = arena_alloc(a, (n + 1) * sizeof *partial);
    size_t nres = 0, loaded = 0;
    for (size_t i = 0; i < n; i++) {
        struct stat st;
        bool have = files[i].have_st || stat(files[i].path, &st) == 0;
        if (files[i].have_st) st = files[i].st;
        CodeSessionInfo info;
        bool got = have && session_info(a, files[i].path, &st, &info);
        loaded++;
        if (got) res[nres++] = info;
        if (!cb) continue;
        bool publish = all_scope ? (i == 0 || loaded % 100 == 0 || loaded == n) : (loaded == 1 || loaded % 10 == 0 || loaded == n);
        if (publish) {
            memcpy(partial, res, nres * sizeof *res);
            sort_infos(partial, nres);
            cb(ctx, loaded, n, partial, nres);
        } else cb(ctx, loaded, n, NULL, 0);
    }
    sort_infos(res, nres);
    *out = res;
    return nres;
}

size_t code_sessions_list(Arena *a, const char *dir, CodeSessionProgress cb, void *ctx, CodeSessionInfo **out) {
    *out = NULL;
    DIR *d = opendir(dir);
    if (!d) return 0;
    VEC(char *) names = {0};
    struct dirent *de;
    while ((de = readdir(d))) {
        size_t n = strlen(de->d_name);
        if (n >= 6 && !strcmp(de->d_name + n - 6, ".jsonl")) vec_push(&names, arena_strdup(a, de->d_name));
    }
    closedir(d);
    qsort(names.ptr, names.len, sizeof *names.ptr, name_desc);
    Candidate *files = arena_alloc(a, (names.len + 1) * sizeof *files);
    for (size_t i = 0; i < names.len; i++) files[i] = (Candidate){.path = path_join(a, dir, names.ptr[i])};
    size_t n = load_infos(a, files, names.len, false, cb, ctx, out);
    vec_free(&names);
    return n;
}

static int cand_cmp(const void *x, const void *y) {
    const Candidate *p = x, *q = y;
    double mp = p->have_st ? (double)p->st.st_mtimespec.tv_sec * 1e3 + (double)p->st.st_mtimespec.tv_nsec / 1e6 : -INFINITY;
    double mq = q->have_st ? (double)q->st.st_mtimespec.tv_sec * 1e3 + (double)q->st.st_mtimespec.tv_nsec / 1e6 : -INFINITY;
    if (mq > mp) return 1;
    if (mq < mp) return -1;
    const char *bp = strrchr(p->path, '/'), *bq = strrchr(q->path, '/');
    return strcmp(bq ? bq + 1 : q->path, bp ? bp + 1 : p->path);
}

size_t code_sessions_list_all(Arena *a, const char *agent_dir, CodeSessionProgress cb, void *ctx, CodeSessionInfo **out) {
    *out = NULL;
    char *root = path_join(a, agent_dir, "sessions");
    DIR *d = opendir(root);
    if (!d) return 0;
    VEC(Candidate) files = {0};
    struct dirent *de;
    while ((de = readdir(d))) {
        if (de->d_name[0] == '.' && (!de->d_name[1] || (de->d_name[1] == '.' && !de->d_name[2]))) continue;
        if (de->d_type != DT_DIR && de->d_type != DT_LNK) continue;
        char *dir = path_join(a, root, de->d_name);
        DIR *sd = opendir(dir);
        if (!sd) continue;
        struct dirent *fe;
        while ((fe = readdir(sd))) {
            size_t n = strlen(fe->d_name);
            if (n < 6 || strcmp(fe->d_name + n - 6, ".jsonl")) continue;
            Candidate c = {.path = path_join(a, dir, fe->d_name)};
            c.have_st = stat(c.path, &c.st) == 0;
            vec_push(&files, c);
        }
        closedir(sd);
    }
    closedir(d);
    qsort(files.ptr, files.len, sizeof *files.ptr, cand_cmp);
    size_t n = load_infos(a, files.ptr, files.len, true, cb, ctx, out);
    vec_free(&files);
    return n;
}

// ---- names (getSessionName, appendSessionInfo) ------------------------------------------------------

const char *code_sessions_name(Arena *a, const JValue *entries) {
    // Walk entries in reverse to find the latest session_info entry; an empty name clears it.
    for (size_t i = entries->arr.len; i-- > 0;) {
        const JValue *e = entries->arr.items[i];
        if (!str_eqz(type_of(e), "session_info")) continue;
        Str n = str_trim(json_strs(json_get(e, "name")));
        return n.len ? str_z(a, n) : NULL;
    }
    return NULL;
}

// name.replace(/[\r\n]+/g, " ").trim()
static char *sanitize_name(Arena *a, const char *name) {
    Buf b = {0};
    for (const char *p = name; *p;) {
        if (*p == '\r' || *p == '\n') {
            while (*p == '\r' || *p == '\n') p++;
            buf_addc(&b, ' ');
        } else buf_addc(&b, *p++);
    }
    Str t = str_trim(buf_str(&b));
    char *r = arena_strndup(a, t.ptr, t.len);
    buf_free(&b);
    return r;
}

static Err append_info(CodeSession *s, Arena *a, const char *name) {
    JValue *f = json_new(a, J_OBJECT);
    json_put(a, f, S("name"), json_string(a, str(sanitize_name(a, name))));
    return code_session_append(s, "session_info", f);
}

Err code_sessions_set_name(CodeSession *s, const char *name) { return append_info(s, code_session_arena(s), name); }

Err code_sessions_rename_file(const char *path, const char *name) {
    Arena a = {0};
    CodeSession *s;
    Err e = code_session_open_path(&a, path, &s);
    if (!e) e = append_info(s, &a, name);
    arena_reset(&a);
    return e;
}

// ---- the projection (buildSessionPath, buildContextEntries, sessionEntryToContextMessages,
// projectContextEntry, buildSessionProjection) -------------------------------------------------------

static const JValue *by_id(const JValue *entries, Str id) {
    for (size_t i = entries->arr.len; i-- > 0;) {
        const JValue *e = entries->arr.items[i];
        if (str_eqz(type_of(e), "session")) continue;
        if (str_eq(json_strs(json_get(e, "id")), id)) return e;
    }
    return NULL;
}

JValue *code_sessions_branch(Arena *a, const JValue *entries, const char *leaf) {
    JValue *path = json_new(a, J_ARRAY);
    if (!leaf) return path;
    const JValue *cur = by_id(entries, str(leaf));
    if (!cur) // leaf ??= the last entry
        for (size_t i = entries->arr.len; i-- > 0 && !cur;)
            if (!str_eqz(type_of(entries->arr.items[i]), "session")) cur = entries->arr.items[i];
    VEC(const JValue *) up = {0};
    while (cur && up.len <= entries->arr.len) {
        vec_push(&up, cur);
        const JValue *p = json_get(cur, "parentId");
        cur = json_is(p, J_STRING) ? by_id(entries, p->s) : NULL;
    }
    for (size_t i = up.len; i-- > 0;) json_push(a, path, (JValue *)up.ptr[i]);
    vec_free(&up);
    return path;
}

static bool is_system_entry(const JValue *e) {
    return str_eqz(type_of(e), "message") && str_eqz(role_of(json_get(e, "message")), "system");
}

// buildContextEntries over a path (root first).
static JValue *context_of_path(Arena *a, const JValue *path) {
    ptrdiff_t ci = -1;
    for (size_t i = 0; i < path->arr.len; i++)
        if (str_eqz(type_of(path->arr.items[i]), "compaction")) ci = (ptrdiff_t)i;
    if (ci < 0) return (JValue *)path;
    JValue *ctx = json_new(a, J_ARRAY);
    const JValue *comp = path->arr.items[ci];
    json_push(a, ctx, (JValue *)comp);
    Str first = json_strs(json_get(comp, "firstKeptEntryId"));
    bool found = false;
    for (ptrdiff_t i = 0; i < ci; i++) {
        const JValue *e = path->arr.items[i];
        if (str_eq(json_strs(json_get(e, "id")), first)) found = true;
        if (found && !is_system_entry(e)) json_push(a, ctx, (JValue *)e);
    }
    for (size_t i = (size_t)ci + 1; i < path->arr.len; i++) json_push(a, ctx, path->arr.items[i]);
    return ctx;
}

JValue *code_sessions_context_entries(Arena *a, const JValue *entries, const char *leaf) {
    return context_of_path(a, code_sessions_branch(a, entries, leaf));
}

void code_sessions_entry_messages(Arena *a, const JValue *e, JValue *out) {
    Str type = type_of(e);
    double ts = iso_ms(json_strs(json_get(e, "timestamp")));
    JValue *tsv = isnan(ts) ? json_new(a, J_NULL) : jnumber(a, ts);
    if (str_eqz(type, "message")) {
        const JValue *m = json_get(e, "message");
        if (!m) return;
        Str role = role_of(m);
        const JValue *c = json_get(m, "content");
        if (!c || c->kind == J_NULL) {
            if (str_eqz(role, "system")) {
                JValue *copy = shallow(a, m);
                json_put(a, copy, S("content"), json_string(a, S("")));
                json_push(a, out, copy);
                return;
            }
            if (str_eqz(role, "user") || str_eqz(role, "assistant") || str_eqz(role, "toolResult")) {
                JValue *copy = shallow(a, m);
                json_put(a, copy, S("content"), json_new(a, J_ARRAY));
                json_push(a, out, copy);
                return;
            }
        }
        json_push(a, out, (JValue *)m);
    } else if (str_eqz(type, "custom_message")) {
        JValue *m = json_new(a, J_OBJECT);
        json_put(a, m, S("role"), json_string(a, S("custom")));
        json_put(a, m, S("customType"), json_get(e, "customType") ? json_get(e, "customType") : json_new(a, J_NULL));
        json_put(a, m, S("content"), json_get(e, "content") ? json_get(e, "content") : json_new(a, J_ARRAY));
        json_put(a, m, S("display"), json_get(e, "display") ? json_get(e, "display") : json_bool(a, false));
        if (json_get(e, "details")) json_put(a, m, S("details"), json_get(e, "details"));
        json_put(a, m, S("timestamp"), tsv);
        json_push(a, out, m);
    } else if (str_eqz(type, "branch_summary") && json_strs(json_get(e, "summary")).len) {
        JValue *m = json_new(a, J_OBJECT);
        json_put(a, m, S("role"), json_string(a, S("branchSummary")));
        json_put(a, m, S("summary"), json_get(e, "summary"));
        json_put(a, m, S("fromId"), json_get(e, "fromId") ? json_get(e, "fromId") : json_new(a, J_NULL));
        json_put(a, m, S("timestamp"), tsv);
        json_push(a, out, m);
    } else if (str_eqz(type, "compaction")) {
        const JValue *sys = json_get(e, "systemMessage");
        if (sys) json_push(a, out, (JValue *)sys);
        JValue *m = json_new(a, J_OBJECT);
        json_put(a, m, S("role"), json_string(a, S("compactionSummary")));
        json_put(a, m, S("summary"), json_get(e, "summary") ? json_get(e, "summary") : json_string(a, S("")));
        json_put(a, m, S("tokensBefore"), json_get(e, "tokensBefore") ? json_get(e, "tokensBefore") : json_int(a, 0));
        json_put(a, m, S("timestamp"), tsv);
        json_push(a, out, m);
    }
}

// projectContextEntry
static void project_entry(Arena *a, const JValue *e, const JValue *edit, JValue *out) {
    JValue *msgs = json_new(a, J_ARRAY);
    code_sessions_entry_messages(a, e, msgs);
    if (!edit) {
        for (size_t i = 0; i < msgs->arr.len; i++) json_push(a, out, msgs->arr.items[i]);
        return;
    }
    const JValue *rep = json_get(edit, "replacement");
    if (!rep || rep->kind == J_NULL) return;
    for (size_t i = 0; i < msgs->arr.len; i++) {
        JValue *m = msgs->arr.items[i];
        Str role = role_of(m);
        if (!str_eqz(role, "user") && !str_eqz(role, "assistant") && !str_eqz(role, "toolResult") && !str_eqz(role, "custom")) {
            json_push(a, out, m);
            continue;
        }
        JValue *copy = shallow(a, m);
        const JValue *rc = json_get(rep, "content");
        if ((str_eqz(role, "assistant") || str_eqz(role, "toolResult")) && json_is(rc, J_STRING)) {
            JValue *arr = json_new(a, J_ARRAY);
            json_push(a, arr, text_block(a, rc->s));
            json_put(a, copy, S("content"), arr);
        } else json_put(a, copy, S("content"), rc ? (JValue *)rc : json_new(a, J_NULL));
        json_push(a, out, copy);
    }
}

typedef struct {
    const JValue *src;
    JValue *msgs;
} PEntry;
typedef VEC(PEntry) Projection;

// buildSessionProjection over a path (root first): each context entry with its messages.
static void project(Arena *a, const JValue *path, Projection *p, JValue *all) {
    JValue *ctx = context_of_path(a, path);
    for (size_t i = 0; i < ctx->arr.len; i++) {
        const JValue *e = ctx->arr.items[i];
        const JValue *edit = NULL;
        Str id = json_strs(json_get(e, "id"));
        for (size_t k = 0; k < ctx->arr.len; k++) {
            const JValue *x = ctx->arr.items[k];
            if (str_eqz(type_of(x), "context_edit") && str_eq(json_strs(json_get(x, "targetId")), id)) edit = x;
        }
        PEntry pe = {e, json_new(a, J_ARRAY)};
        if (!(str_eqz(type_of(e), "compaction") && i > 0)) project_entry(a, e, edit, pe.msgs);
        vec_push(p, pe);
        for (size_t k = 0; all && k < pe.msgs->arr.len; k++) json_push(a, all, pe.msgs->arr.items[k]);
    }
}

// ---- estimates (compaction.ts) -----------------------------------------------------------------------

// estimateTextAndImageContentChars
static int64_t content_chars(const JValue *c) {
    if (json_is(c, J_STRING)) return (int64_t)js_len(c->s);
    int64_t n = 0;
    for (size_t i = 0; json_is(c, J_ARRAY) && i < c->arr.len; i++) {
        const JValue *b = c->arr.items[i];
        if (str_eqz(type_of(b), "text") && json_strs(json_get(b, "text")).len) n += (int64_t)js_len(json_strs(json_get(b, "text")));
        else if (str_eqz(type_of(b), "image")) n += 4800;
    }
    return n;
}

static int64_t ceil_div4(int64_t c) { return (c + 3) / 4; }

int64_t code_sessions_estimate(const JValue *m) {
    Str role = role_of(m);
    int64_t chars = 0;
    if (str_eqz(role, "system")) {
        chars = content_chars(json_get(m, "content"));
        const JValue *sec = json_get(m, "sections");
        for (size_t i = 0; json_is(sec, J_OBJECT) && i < sec->obj.len; i++)
            if (json_is(sec->obj.items[i].val, J_STRING)) chars += (int64_t)js_len(sec->obj.items[i].val->s);
        const JValue *ta = json_get(m, "toolsAdded");
        if (ta) {
            Buf b = {0};
            json_write(&b, ta);
            chars += (int64_t)js_len(buf_str(&b));
            buf_free(&b);
        }
        return ceil_div4(chars);
    }
    if (str_eqz(role, "user") || str_eqz(role, "custom") || str_eqz(role, "toolResult")) return ceil_div4(content_chars(json_get(m, "content")));
    if (str_eqz(role, "assistant")) {
        const JValue *c = json_get(m, "content");
        for (size_t i = 0; json_is(c, J_ARRAY) && i < c->arr.len; i++) {
            const JValue *b = c->arr.items[i];
            Str t = type_of(b);
            if (str_eqz(t, "text")) chars += (int64_t)js_len(json_strs(json_get(b, "text")));
            else if (str_eqz(t, "thinking")) chars += (int64_t)js_len(json_strs(json_get(b, "thinking")));
            else if (str_eqz(t, "toolCall")) {
                Buf j = {0};
                if (json_get(b, "arguments")) json_write(&j, json_get(b, "arguments"));
                chars += (int64_t)(js_len(json_strs(json_get(b, "name"))) + js_len(buf_str(&j)));
                buf_free(&j);
            }
        }
        return ceil_div4(chars);
    }
    if (str_eqz(role, "bashExecution"))
        return ceil_div4((int64_t)(js_len(json_strs(json_get(m, "command"))) + js_len(json_strs(json_get(m, "output")))));
    if (str_eqz(role, "branchSummary") || str_eqz(role, "compactionSummary")) return ceil_div4((int64_t)js_len(json_strs(json_get(m, "summary"))));
    return 0;
}

// calculateContextTokens
static double context_tokens(const JValue *u) {
    double t = jnum(json_get(u, "totalTokens"));
    return t ? t : jnum(json_get(u, "input")) + jnum(json_get(u, "output")) + jnum(json_get(u, "cacheRead")) + jnum(json_get(u, "cacheWrite"));
}

// getAssistantUsage
static const JValue *assistant_usage(const JValue *m) {
    if (!str_eqz(role_of(m), "assistant") || !json_get(m, "usage")) return NULL;
    Str stop = json_strs(json_get(m, "stopReason"));
    if (str_eqz(stop, "aborted") || str_eqz(stop, "error")) return NULL;
    const JValue *u = json_get(m, "usage");
    return json_is(u, J_OBJECT) && context_tokens(u) > 0 ? u : NULL;
}

// estimateProjectedContextTokens (estimateContextTokens first).
static double projected_tokens(Arena *a, const Projection *p, const JValue *msgs, const JValue *branch) {
    ptrdiff_t ui = -1;
    for (size_t i = msgs->arr.len; i-- > 0 && ui < 0;)
        if (assistant_usage(msgs->arr.items[i])) ui = (ptrdiff_t)i;
    if (ui >= 0) {
        double tokens = context_tokens(assistant_usage(msgs->arr.items[ui]));
        for (size_t i = (size_t)ui + 1; i < msgs->arr.len; i++) tokens += (double)code_sessions_estimate(msgs->arr.items[i]);
        size_t at = 0;
        Str usage_id = {0};
        for (size_t i = 0; i < p->len; i++) {
            size_t next = at + p->ptr[i].msgs->arr.len;
            if ((size_t)ui < next) {
                usage_id = json_strs(json_get(p->ptr[i].src, "id"));
                break;
            }
            at = next;
        }
        ptrdiff_t usage_index = -1, invalid = -1;
        for (size_t i = 0; usage_id.ptr && i < branch->arr.len; i++)
            if (str_eq(json_strs(json_get(branch->arr.items[i], "id")), usage_id)) {
                usage_index = (ptrdiff_t)i;
                break;
            }
        for (size_t i = branch->arr.len; i-- > 0;) {
            Str t = type_of(branch->arr.items[i]);
            if (str_eqz(t, "context_edit") || str_eqz(t, "compaction")) {
                invalid = (ptrdiff_t)i;
                break;
            }
        }
        if (usage_index > invalid) return tokens;
    }
    const JValue *sys = code_current_system(a, msgs);
    double tokens = sys ? (double)code_sessions_estimate(sys) : 0;
    for (size_t i = 0; i < msgs->arr.len; i++)
        if (!str_eqz(role_of(msgs->arr.items[i]), "system")) tokens += (double)code_sessions_estimate(msgs->arr.items[i]);
    return tokens;
}

// ---- /fork and /clone (getUserMessagesForForking, AgentSessionRuntime.fork, createBranchedSession) --

JValue *code_sessions_fork_messages(Arena *a, const JValue *entries) {
    JValue *out = json_new(a, J_ARRAY);
    for (size_t i = 0; i < entries->arr.len; i++) {
        const JValue *e = entries->arr.items[i];
        if (!str_eqz(type_of(e), "message")) continue;
        const JValue *m = json_get(e, "message");
        if (!str_eqz(role_of(m), "user")) continue;
        Buf t = {0};
        content_text(&t, json_get(m, "content"), "");
        if (t.len) {
            JValue *o = json_new(a, J_OBJECT);
            json_put(a, o, S("entryId"), json_get(e, "id"));
            json_put(a, o, S("text"), json_string(a, str_n(arena_strndup(a, t.ptr, t.len), t.len)));
            json_push(a, out, o);
        }
        buf_free(&t);
    }
    return out;
}

static char *dir_of(Arena *a, const char *path) {
    const char *s = strrchr(path, '/');
    return s ? arena_strndup(a, path, (size_t)(s - path)) : arena_strdup(a, ".");
}

// join(sessionDir, `${timestamp.replace(/[:.]/g, "-")}_${id}.jsonl`) beside the current file.
static char *new_file(Arena *a, const char *beside, const char *ts, const char *id) {
    char *f = arena_strdup(a, ts);
    for (char *q = f; *q; q++)
        if (*q == ':' || *q == '.') *q = '-';
    return path_join(a, dir_of(a, beside), arena_printf(a, "%s_%s.jsonl", f, id));
}

// createBranchedSession(leafId): the path to leaf (labels dropped, the rest re-chained, the labels
// written again after it) in a new session whose parentSession is this one.
static Err branched(Arena *a, const CodeSession *cur, const char *cwd, const char *leaf, CodeSession **out) {
    const JValue *entries = code_session_entries(cur);
    JValue *path = code_sessions_branch(a, entries, leaf);
    if (!path->arr.len) return "EntryNotFound";
    JValue *kept = json_new(a, J_ARRAY);
    StrMap replacement = {0};
    VEC(Str) pending = {0};
    const char *parent = NULL;
    for (size_t i = 0; i < path->arr.len; i++) {
        const JValue *e = path->arr.items[i];
        if (str_eqz(type_of(e), "label")) {
            vec_push(&pending, json_strs(json_get(e, "id")));
            continue;
        }
        for (size_t k = 0; k < pending.len; k++) strmap_put(&replacement, pending.ptr[k], (void *)json_str(json_get(e, "id")));
        pending.len = 0;
        JValue *c = shallow(a, e);
        json_put(a, c, S("parentId"), parent ? json_string(a, str(parent)) : json_new(a, J_NULL));
        if (str_eqz(type_of(e), "compaction")) {
            Str fk = json_strs(json_get(e, "firstKeptEntryId"));
            if (!str_eq(fk, json_strs(json_get(e, "id")))) {
                void **r = strmap_get(&replacement, fk);
                if (r) json_put(a, c, S("firstKeptEntryId"), json_string(a, str((const char *)*r)));
            }
        }
        json_push(a, kept, c);
        parent = json_str(json_get(e, "id"));
    }
    vec_free(&pending);
    strmap_free(&replacement);
    double now = wall_ms();
    char *id = code_session_uuid(a, now);
    char *ts = code_session_iso(a, now);
    const char *prev = code_session_path(cur);
    JValue *h = json_new(a, J_OBJECT);
    json_put(a, h, S("type"), json_string(a, S("session")));
    json_put(a, h, S("version"), json_int(a, 3));
    json_put(a, h, S("id"), json_string(a, str(id)));
    json_put(a, h, S("timestamp"), json_string(a, str(ts)));
    json_put(a, h, S("cwd"), json_string(a, str(cwd)));
    if (prev) json_put(a, h, S("parentSession"), json_string(a, str(prev)));
    // the labels of the kept entries (labelsById: the latest label per target, in first-set order)
    JValue *labels = json_new(a, J_ARRAY);
    VEC(const JValue *) order = {0};
    for (size_t i = 0; i < entries->arr.len; i++) {
        const JValue *e = entries->arr.items[i];
        if (!str_eqz(type_of(e), "label")) continue;
        Str target = json_strs(json_get(e, "targetId"));
        size_t k = 0;
        while (k < order.len && !str_eq(json_strs(json_get(order.ptr[k], "targetId")), target)) k++;
        bool set = json_strs(json_get(e, "label")).len > 0;
        if (k < order.len) {
            if (set) order.ptr[k] = e;
            else {
                memmove(order.ptr + k, order.ptr + k + 1, (order.len - k - 1) * sizeof *order.ptr);
                order.len--;
            }
        } else if (set) vec_push(&order, e);
    }
    JValue *all = json_new(a, J_ARRAY);
    json_push(a, all, h);
    for (size_t i = 0; i < kept->arr.len; i++) json_push(a, all, kept->arr.items[i]);
    const char *lparent = parent;
    for (size_t k = 0; k < order.len; k++) {
        Str target = json_strs(json_get(order.ptr[k], "targetId"));
        bool in_path = false;
        for (size_t i = 0; i < kept->arr.len && !in_path; i++) in_path = str_eq(json_strs(json_get(kept->arr.items[i], "id")), target);
        if (!in_path) continue;
        JValue *l = json_new(a, J_OBJECT);
        json_put(a, l, S("type"), json_string(a, S("label")));
        char *lid = code_session_new_entry_id(a, all);
        json_put(a, l, S("id"), json_string(a, str(lid)));
        json_put(a, l, S("parentId"), lparent ? json_string(a, str(lparent)) : json_new(a, J_NULL));
        json_put(a, l, S("timestamp"), json_get(order.ptr[k], "timestamp"));
        json_put(a, l, S("targetId"), json_get(order.ptr[k], "targetId"));
        json_put(a, l, S("label"), json_get(order.ptr[k], "label"));
        json_push(a, labels, l);
        json_push(a, all, l);
        lparent = lid;
    }
    vec_free(&order);
    const char *file = prev ? new_file(a, prev, ts, id) : NULL;
    return code_session_from_entries(a, file, all, out);
}

// newSession({parentSession}): an empty session in the same folder.
static Err fresh(Arena *a, const CodeSession *cur, const char *cwd, CodeSession **out) {
    double now = wall_ms();
    char *id = code_session_uuid(a, now);
    char *ts = code_session_iso(a, now);
    const char *prev = code_session_path(cur);
    JValue *h = json_new(a, J_OBJECT);
    json_put(a, h, S("type"), json_string(a, S("session")));
    json_put(a, h, S("version"), json_int(a, 3));
    json_put(a, h, S("id"), json_string(a, str(id)));
    json_put(a, h, S("timestamp"), json_string(a, str(ts)));
    json_put(a, h, S("cwd"), json_string(a, str(cwd)));
    if (prev) json_put(a, h, S("parentSession"), json_string(a, str(prev)));
    JValue *all = json_new(a, J_ARRAY);
    json_push(a, all, h);
    const char *file = prev ? new_file(a, prev, ts, id) : NULL;
    return code_session_from_entries(a, file, all, out);
}

const char *code_sessions_fork(Arena *a, CodeSession *cur, const char *cwd, const char *entry_id, bool at, CodeSession **out, char **selected) {
    *selected = NULL;
    const JValue *entries = code_session_entries(cur);
    const JValue *e = entry_id ? by_id(entries, str(entry_id)) : NULL;
    if (!e) return "Invalid entry ID for forking";
    const char *target;
    if (at) target = json_str(json_get(e, "id"));
    else {
        if (!str_eqz(type_of(e), "message") || !str_eqz(role_of(json_get(e, "message")), "user")) return "Invalid entry ID for forking";
        const JValue *p = json_get(e, "parentId");
        target = json_is(p, J_STRING) ? p->s.ptr : NULL;
        // extractUserMessageText: the text parts joined
        Buf t = {0};
        content_text(&t, json_get(json_get(e, "message"), "content"), "");
        *selected = arena_strndup(a, t.ptr ? t.ptr : "", t.len);
        buf_free(&t);
    }
    const char *file = code_session_path(cur);
    Err err;
    if (file) {
        if (!target) err = fresh(a, cur, cwd, out);
        else {
            if (access(file, F_OK) != 0) return "This session has not been saved yet. Send a message before cloning or forking it.";
            // SessionManager.open(currentSessionFile): the file as written
            CodeSession *disk;
            if ((err = code_session_open_path(a, file, &disk))) return err;
            err = branched(a, disk, cwd, target, out);
        }
    } else err = target ? branched(a, cur, cwd, target, out) : fresh(a, cur, cwd, out);
    if (err) return !strcmp(err, "EntryNotFound") ? arena_printf(a, "Entry %s not found", target) : err;
    return NULL;
}

const char *code_sessions_fork_from(Arena *a, const char *path, const char *cwd, const char *dir, char **out) {
    CodeSession *src;
    char *real = realpath(path, NULL);
    const char *source = real ? arena_strdup(a, real) : path;
    free(real);
    if (code_session_open_path(a, source, &src)) return arena_printf(a, "Cannot fork: source session file is empty or invalid: %s", source);
    const JValue *entries = code_session_entries(src);
    if (!str_eqz(type_of(entries->arr.items[0]), "session")) return arena_printf(a, "Cannot fork: source session has no header: %s", source);
    if (mkdir_p(dir)) return arena_printf(a, "Cannot create %s", dir);
    double now = wall_ms();
    char *id = code_session_uuid(a, now), *ts = code_session_iso(a, now);
    char *f = arena_strdup(a, ts);
    for (char *q = f; *q; q++)
        if (*q == ':' || *q == '.') *q = '-';
    char *file = path_join(a, dir, arena_printf(a, "%s_%s.jsonl", f, id));
    JValue *h = json_new(a, J_OBJECT);
    json_put(a, h, S("type"), json_string(a, S("session")));
    json_put(a, h, S("version"), json_int(a, 3));
    json_put(a, h, S("id"), json_string(a, str(id)));
    json_put(a, h, S("timestamp"), json_string(a, str(ts)));
    json_put(a, h, S("cwd"), json_string(a, str(cwd)));
    json_put(a, h, S("parentSession"), json_string(a, str(source)));
    Buf b = {0};
    json_write(&b, h);
    buf_addc(&b, '\n');
    for (size_t i = 0; i < entries->arr.len; i++) {
        if (str_eqz(type_of(entries->arr.items[i]), "session")) continue;
        json_write(&b, entries->arr.items[i]);
        buf_addc(&b, '\n');
    }
    FILE *fp = fopen(file, "wx");
    bool ok = fp && fwrite(b.ptr, 1, b.len, fp) == b.len;
    if (fp) fclose(fp);
    buf_free(&b);
    if (!ok) return arena_printf(a, "Cannot write %s", file);
    *out = file;
    return NULL;
}

// ---- /session (getSessionStats) and the footer (getContextUsage) ----------------------------------

static void add_usage(CodeSessionStats *st, const JValue *u) {
    st->input += jnum(json_get(u, "input"));
    st->output += jnum(json_get(u, "output"));
    st->cache_read += jnum(json_get(u, "cacheRead"));
    st->cache_write += jnum(json_get(u, "cacheWrite"));
    st->cost += jnum(json_get(json_get(u, "cost"), "total"));
}

void code_sessions_stats(const JValue *entries, CodeSessionStats *st) {
    *st = (CodeSessionStats){.hit_rate = -1};
    for (size_t i = 0; i < entries->arr.len; i++) {
        const JValue *e = entries->arr.items[i];
        Str t = type_of(e);
        if (str_eqz(t, "usage")) add_usage(st, json_get(e, "usage"));
        else if ((str_eqz(t, "branch_summary") || str_eqz(t, "compaction")) && json_get(e, "usage")) add_usage(st, json_get(e, "usage"));
        if (!str_eqz(t, "message")) continue;
        st->total++;
        const JValue *m = json_get(e, "message");
        Str role = role_of(m);
        if (str_eqz(role, "user")) st->user++;
        else if (str_eqz(role, "toolResult")) {
            st->tool_results++;
            if (json_get(m, "usage")) add_usage(st, json_get(m, "usage"));
        } else if (str_eqz(role, "assistant")) {
            st->assistant++;
            const JValue *c = json_get(m, "content");
            for (size_t k = 0; json_is(c, J_ARRAY) && k < c->arr.len; k++)
                if (str_eqz(type_of(c->arr.items[k]), "toolCall")) st->tool_calls++;
            const JValue *u = json_get(m, "usage");
            add_usage(st, u);
            double prompt = jnum(json_get(u, "input")) + jnum(json_get(u, "cacheRead")) + jnum(json_get(u, "cacheWrite"));
            st->hit_rate = prompt > 0 ? jnum(json_get(u, "cacheRead")) / prompt * 100 : -1;
        }
    }
}

bool code_sessions_context_unknown(Arena *a, const JValue *entries, const char *leaf) {
    JValue *branch = code_sessions_branch(a, entries, leaf);
    ptrdiff_t ci = -1;
    for (size_t i = branch->arr.len; i-- > 0 && ci < 0;)
        if (str_eqz(type_of(branch->arr.items[i]), "compaction")) ci = (ptrdiff_t)i;
    if (ci < 0) return false;
    Projection p = {0};
    project(a, branch, &p, NULL);
    bool after = false;
    for (size_t i = (size_t)ci + 1; i < branch->arr.len && !after; i++) {
        Str id = json_strs(json_get(branch->arr.items[i], "id"));
        for (size_t k = 0; k < p.len && !after; k++) {
            if (!str_eq(json_strs(json_get(p.ptr[k].src, "id")), id)) continue;
            for (size_t j = 0; j < p.ptr[k].msgs->arr.len && !after; j++) after = assistant_usage(p.ptr[k].msgs->arr.items[j]) != NULL;
        }
    }
    vec_free(&p);
    return !after;
}

// ---- /compact (compaction.ts, utils.ts) ----------------------------------------------------------------

static const char SUMMARIZATION_SYSTEM_PROMPT[] =
    "You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then "
    "produce a structured summary following the exact format specified.\n\nDo NOT continue the conversation. Do NOT respond to "
    "any questions in the conversation. ONLY output the structured summary.";

static const char SUMMARIZATION_PROMPT[] =
    "The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will "
    "use to continue the work.\n\nUse this EXACT format:\n\n## Goal\n[What is the user trying to accomplish? Can be multiple "
    "items if the session covers different tasks.]\n\n## Constraints & Preferences\n- [Any constraints, preferences, or "
    "requirements mentioned by user]\n- [Or \"(none)\" if none were mentioned]\n\n## Progress\n### Done\n- [x] "
    "[Completed tasks/changes]\n\n### In Progress\n- [ ] [Current work]\n\n### Blocked\n- [Issues preventing progress, if "
    "any]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale]\n\n## Next Steps\n1. [Ordered list of what should happen "
    "next]\n\n## Critical Context\n- [Any data, examples, or references needed to continue]\n- [Or \"(none)\" if not "
    "applicable]\n\nKeep each section concise. Preserve exact file paths, function names, and error messages.";

static const char UPDATE_SUMMARIZATION_PROMPT[] =
    "The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> "
    "tags.\n\nUpdate the existing structured summary with new information. RULES:\n- PRESERVE all existing information from "
    "the previous summary\n- ADD new progress, decisions, and context from the new messages\n- UPDATE the Progress section: "
    "move items from \"In Progress\" to \"Done\" when completed\n- UPDATE \"Next Steps\" based on what was accomplished\n- "
    "PRESERVE exact file paths, function names, and error messages\n- If something is no longer relevant, you may remove "
    "it\n\nUse this EXACT format:\n\n## Goal\n[Preserve existing goals, add new ones if the task expanded]\n\n## Constraints "
    "& Preferences\n- [Preserve existing, add new ones discovered]\n\n## Progress\n### Done\n- [x] [Include previously done "
    "items AND newly completed items]\n\n### In Progress\n- [ ] [Current work - update based on progress]\n\n### Blocked\n- "
    "[Current blockers - remove if resolved]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale] (preserve all "
    "previous, add new)\n\n## Next Steps\n1. [Update based on current state]\n\n## Critical Context\n- [Preserve important "
    "context, add new if needed]\n\nKeep each section concise. Preserve exact file paths, function names, and error messages.";

static const char TURN_PREFIX_SUMMARIZATION_PROMPT[] =
    "The messages above are earlier context from an ongoing conversation. Later messages are stored separately and do not need "
    "to be reconstructed.\n\nCreate a concise checkpoint of the user's request and the progress shown above. This checkpoint "
    "will be placed before the later messages so the conversation can continue with the necessary context.\n\n## Original "
    "Request\n[What did the user ask for?]\n\n## Progress So Far\n- [Key decisions and work completed in these messages]\n\n## "
    "Context Needed to Continue\n- [Information from these messages needed to understand the later work]\n\nOnly summarize "
    "information explicitly present above. Do not infer or recreate later messages.";

static const char COMPACTION_PREFIX[] = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";
static const char COMPACTION_SUFFIX[] = "\n</summary>";
static const char BRANCH_PREFIX[] = "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n";
static const char BRANCH_SUFFIX[] = "</summary>";

// isCutPointMessage, isTurnStartMessage
static bool cut_point_message(const JValue *m) {
    Str r = role_of(m);
    return str_eqz(r, "user") || str_eqz(r, "assistant") || str_eqz(r, "bashExecution") || str_eqz(r, "custom") ||
           str_eqz(r, "branchSummary") || str_eqz(r, "compactionSummary");
}

static bool turn_start_message(const JValue *m) {
    Str r = role_of(m);
    return str_eqz(r, "user") || str_eqz(r, "bashExecution") || str_eqz(r, "custom") || str_eqz(r, "branchSummary") ||
           str_eqz(r, "compactionSummary");
}

static bool any_msg(const JValue *msgs, bool (*f)(const JValue *)) {
    for (size_t i = 0; i < msgs->arr.len; i++)
        if (f(msgs->arr.items[i])) return true;
    return false;
}

// isProjectedTurnStart
static bool projected_turn_start(const PEntry *e) { return !str_eqz(type_of(e->src), "compaction") && any_msg(e->msgs, turn_start_message); }

static int64_t msgs_tokens(const JValue *msgs) {
    int64_t t = 0;
    for (size_t i = 0; i < msgs->arr.len; i++) t += code_sessions_estimate(msgs->arr.items[i]);
    return t;
}

typedef struct {
    size_t first_kept;
    ptrdiff_t turn_start;
    bool split;
} CutPoint;

// findProjectedCutPoint
static CutPoint find_cut_point(Arena *a, const Projection *p, size_t start, size_t end, int64_t keep) {
    VEC(size_t) cuts = {0};
    for (size_t i = start; i < end; i++)
        if (!str_eqz(type_of(p->ptr[i].src), "compaction") && any_msg(p->ptr[i].msgs, cut_point_message)) vec_push(&cuts, i);
    if (!cuts.len) return (CutPoint){start, -1, false};
    int64_t acc = 0;
    bool exceeded = false;
    size_t cut = cuts.ptr[0];
    for (size_t i = end; i-- > start;) {
        int64_t t = msgs_tokens(p->ptr[i].msgs);
        if (!t) continue;
        acc += t;
        if (acc >= keep) {
            exceeded = true;
            cut = cuts.ptr[cuts.len - 1];
            for (size_t k = 0; k < cuts.len; k++)
                if (cuts.ptr[k] >= i) {
                    cut = cuts.ptr[k];
                    break;
                }
            break;
        }
    }
    vec_free(&cuts);
    // a closed suffix of omitted recovery attempts moves the cut past them
    if (exceeded && cut + 1 < end) {
        bool has_omitted_asst = false, external = false, all_omitted = true;
        for (size_t i = cut + 1; i < end; i++) {
            const PEntry *e = &p->ptr[i];
            JValue *own = json_new(a, J_ARRAY);
            code_sessions_entry_messages(a, e->src, own);
            bool visible = !str_eqz(type_of(e->src), "context_edit") && own->arr.len > 0;
            bool omitted = visible && e->msgs->arr.len == 0;
            if (omitted && str_eqz(type_of(e->src), "message") && str_eqz(role_of(json_get(e->src, "message")), "assistant")) has_omitted_asst = true;
            if (str_eqz(type_of(e->src), "compaction") || (visible && !omitted)) all_omitted = false;
        }
        for (size_t i = cut + 1; i < end; i++) {
            const PEntry *e = &p->ptr[i];
            if (!str_eqz(type_of(e->src), "context_edit")) continue;
            const JValue *rep = json_get(e->src, "replacement");
            if (!rep || rep->kind == J_NULL) continue;
            // the edit's target among the omitted suffix entries?
            bool target_omitted = false;
            Str target = json_strs(json_get(e->src, "targetId"));
            for (size_t k = cut + 1; k < end; k++) {
                const PEntry *x = &p->ptr[k];
                if (!str_eq(json_strs(json_get(x->src, "id")), target)) continue;
                JValue *own = json_new(a, J_ARRAY);
                code_sessions_entry_messages(a, x->src, own);
                target_omitted = !str_eqz(type_of(x->src), "context_edit") && own->arr.len > 0 && x->msgs->arr.len == 0;
            }
            if (!target_omitted) external = true;
        }
        if (!external && has_omitted_asst && all_omitted) cut++;
    }
    while (cut > start) {
        const PEntry *prev = &p->ptr[cut - 1];
        if (str_eqz(type_of(prev->src), "compaction") || prev->msgs->arr.len > 0) break;
        cut--;
    }
    bool starts = projected_turn_start(&p->ptr[cut]);
    ptrdiff_t ts = -1;
    if (!starts)
        for (size_t i = cut + 1; i-- > start;)
            if (projected_turn_start(&p->ptr[i])) {
                ts = (ptrdiff_t)i;
                break;
            }
    return (CutPoint){cut, ts, !starts && ts != -1};
}

// getMessagesFromProjectedEntryForCompaction
static void compaction_messages(Arena *a, const PEntry *e, JValue *out) {
    if (str_eqz(type_of(e->src), "compaction")) return;
    for (size_t i = 0; i < e->msgs->arr.len; i++)
        if (!str_eqz(role_of(e->msgs->arr.items[i]), "system")) json_push(a, out, e->msgs->arr.items[i]);
}

typedef struct {
    VEC(char *) read, written, edited;
} FileOps;

static void set_add(void *vec, const char *s) {
    VEC(char *) *v = vec;
    for (size_t i = 0; i < v->len; i++)
        if (!strcmp(v->ptr[i], s)) return;
    vec_push(v, (char *)s);
}

static bool set_has(void *vec, const char *s) {
    VEC(char *) *v = vec;
    for (size_t i = 0; i < v->len; i++)
        if (!strcmp(v->ptr[i], s)) return true;
    return false;
}

// extractFileOpsFromMessage (assistant tool calls; codemode's nested calls are not in Saetta Code)
static void file_ops_of(const JValue *m, FileOps *ops) {
    if (!str_eqz(role_of(m), "assistant")) return;
    const JValue *c = json_get(m, "content");
    for (size_t i = 0; json_is(c, J_ARRAY) && i < c->arr.len; i++) {
        const JValue *b = c->arr.items[i];
        if (!str_eqz(type_of(b), "toolCall")) continue;
        const char *path = json_str(json_get(json_get(b, "arguments"), "path"));
        if (!path || !*path) continue;
        Str name = json_strs(json_get(b, "name"));
        if (str_eqz(name, "read")) set_add(&ops->read, path);
        else if (str_eqz(name, "write")) set_add(&ops->written, path);
        else if (str_eqz(name, "edit")) set_add(&ops->edited, path);
    }
}

static int str_cmp(const void *x, const void *y) { return strcmp(*(char *const *)x, *(char *const *)y); }

// computeFileLists and formatFileOperations
static void file_lists(Arena *a, FileOps *ops, JValue **read_files, JValue **modified_files, Buf *text) {
    VEC(char *) modified = {0}, ro = {0};
    for (size_t i = 0; i < ops->edited.len; i++) set_add(&modified, ops->edited.ptr[i]);
    for (size_t i = 0; i < ops->written.len; i++) set_add(&modified, ops->written.ptr[i]);
    for (size_t i = 0; i < ops->read.len; i++)
        if (!set_has(&modified, ops->read.ptr[i])) vec_push(&ro, ops->read.ptr[i]);
    if (ro.len) qsort(ro.ptr, ro.len, sizeof *ro.ptr, str_cmp);
    if (modified.len) qsort(modified.ptr, modified.len, sizeof *modified.ptr, str_cmp);
    *read_files = json_new(a, J_ARRAY);
    *modified_files = json_new(a, J_ARRAY);
    for (size_t i = 0; i < ro.len; i++) json_push(a, *read_files, json_string(a, str(ro.ptr[i])));
    for (size_t i = 0; i < modified.len; i++) json_push(a, *modified_files, json_string(a, str(modified.ptr[i])));
    bool any = false;
    if (ro.len) {
        buf_adds(text, "\n\n<read-files>\n");
        for (size_t i = 0; i < ro.len; i++) buf_printf(text, "%s%s", i ? "\n" : "", ro.ptr[i]);
        buf_adds(text, "\n</read-files>");
        any = true;
    }
    if (modified.len) {
        buf_adds(text, any ? "\n\n<modified-files>\n" : "\n\n<modified-files>\n");
        for (size_t i = 0; i < modified.len; i++) buf_printf(text, "%s%s", i ? "\n" : "", modified.ptr[i]);
        buf_adds(text, "\n</modified-files>");
    }
    vec_free(&modified);
    vec_free(&ro);
}

// truncateForSummary(text, 2000) with sliceHead
static void truncated(Buf *b, Str text, size_t max) {
    size_t n = js_len(text);
    if (n <= max) {
        buf_addstr(b, text);
        return;
    }
    size_t units = 0, i = 0;
    while (i < text.len) {
        unsigned char c = (unsigned char)text.ptr[i];
        size_t l = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
        size_t u = l == 4 ? 2 : 1;
        if (units + u > max) break;
        units += u;
        i += l;
    }
    buf_add(b, text.ptr, i);
    buf_printf(b, "\n\n[... %zu more characters truncated]", n - max);
}

// convertToLlm and serializeConversation
static void serialize(Buf *out, const JValue *msgs) {
    bool first = true;
#define PART(...)                              \
    do {                                       \
        if (!first) buf_adds(out, "\n\n");     \
        buf_printf(out, __VA_ARGS__);          \
        first = false;                         \
    } while (0)
    for (size_t i = 0; i < msgs->arr.len; i++) {
        const JValue *m = msgs->arr.items[i];
        Str role = role_of(m);
        Buf t = {0};
        if (str_eqz(role, "user") || str_eqz(role, "custom") || str_eqz(role, "branchSummary") || str_eqz(role, "compactionSummary") ||
            str_eqz(role, "bashExecution")) {
            if (str_eqz(role, "branchSummary")) {
                buf_adds(&t, BRANCH_PREFIX);
                buf_addstr(&t, json_strs(json_get(m, "summary")));
                buf_adds(&t, BRANCH_SUFFIX);
            } else if (str_eqz(role, "compactionSummary")) {
                buf_adds(&t, COMPACTION_PREFIX);
                buf_addstr(&t, json_strs(json_get(m, "summary")));
                buf_adds(&t, COMPACTION_SUFFIX);
            } else if (str_eqz(role, "bashExecution")) {
                if (jtrue(json_get(m, "excludeFromContext"))) {
                    buf_free(&t);
                    continue;
                }
                // bashExecutionToText
                Str cmd = json_strs(json_get(m, "command")), o = json_strs(json_get(m, "output"));
                buf_printf(&t, "Ran `%.*s`\n", (int)cmd.len, cmd.ptr);
                if (o.len) buf_printf(&t, "```\n%.*s\n```", (int)o.len, o.ptr);
                else buf_adds(&t, "(no output)");
                const JValue *ec = json_get(m, "exitCode");
                if (jtrue(json_get(m, "cancelled"))) buf_adds(&t, "\n\n(command cancelled)");
                else if (jhas_num(ec) && jnum(ec) != 0) buf_printf(&t, "\n\nCommand exited with code %g", jnum(ec));
                Str full = json_strs(json_get(m, "fullOutputPath"));
                if (jtrue(json_get(m, "truncated")) && full.len) buf_printf(&t, "\n\n[Output truncated. Full output: %.*s]", (int)full.len, full.ptr);
            } else content_text(&t, json_get(m, "content"), "");
            if (t.len) PART("[User]: %.*s", (int)t.len, t.ptr);
        } else if (str_eqz(role, "assistant")) {
            const JValue *c = json_get(m, "content");
            Buf calls = {0};
            bool has_text = false;
            for (size_t k = 0; json_is(c, J_ARRAY) && k < c->arr.len; k++) {
                const JValue *b = c->arr.items[k];
                if (str_eqz(type_of(b), "text")) has_text = true;
                if (!str_eqz(type_of(b), "toolCall")) continue;
                if (calls.len) buf_adds(&calls, "; ");
                buf_addstr(&calls, json_strs(json_get(b, "name")));
                buf_addc(&calls, '(');
                const JValue *args = json_get(b, "arguments");
                for (size_t j = 0; json_is(args, J_OBJECT) && j < args->obj.len; j++) {
                    if (j) buf_adds(&calls, ", ");
                    buf_addstr(&calls, args->obj.items[j].key);
                    buf_addc(&calls, '=');
                    json_write(&calls, args->obj.items[j].val);
                }
                buf_addc(&calls, ')');
            }
            if (has_text) {
                content_text(&t, c, "\n");
                PART("[Assistant]: %.*s", (int)t.len, t.ptr);
            }
            if (calls.len) PART("[Assistant tool calls]: %.*s", (int)calls.len, calls.ptr);
            buf_free(&calls);
        } else if (str_eqz(role, "toolResult")) {
            content_text(&t, json_get(m, "content"), "");
            if (t.len) {
                Buf tr = {0};
                truncated(&tr, buf_str(&t), 2000);
                PART("[Tool result]: %.*s", (int)tr.len, tr.ptr);
                buf_free(&tr);
            }
        }
        buf_free(&t);
    }
#undef PART
}

// completeSummarization: one request with the summarization system prompt (no tools). NULL with
// *text and *usage, "aborted", or pi's failure text for label.
static const char *summarize(Arena *a, const CodeModel *m, const CodeEnv *env, Str prompt, const char *label, char **text, const JValue **usage) {
    JValue *sys = json_new(a, J_OBJECT);
    json_put(a, sys, S("role"), json_string(a, S("system")));
    json_put(a, sys, S("content"), json_string(a, str(SUMMARIZATION_SYSTEM_PROMPT)));
    json_put(a, sys, S("timestamp"), json_int(a, 0));
    JValue *user = json_new(a, J_OBJECT);
    json_put(a, user, S("role"), json_string(a, S("user")));
    JValue *c = json_new(a, J_ARRAY);
    json_push(a, c, text_block(a, prompt));
    json_put(a, user, S("content"), c);
    json_put(a, user, S("timestamp"), jnumber(a, wall_ms()));
    JValue *msgs = json_new(a, J_ARRAY);
    json_push(a, msgs, user);
    Buf body = {0};
    Err e = code_oai_body(a, m, sys, msgs, NULL, &body);
    JValue *resp = NULL;
    if (!e) e = code_oai_complete(a, m, env, buf_str(&body), NULL, &resp);
    buf_free(&body);
    if (e) return arena_printf(a, "%s failed: %s", label, e);
    if (env->abort && *env->abort) return "aborted";
    Str stop = json_strs(json_get(resp, "stopReason"));
    if (str_eqz(stop, "aborted")) return "aborted";
    if (str_eqz(stop, "error")) {
        Str em = json_strs(json_get(resp, "errorMessage"));
        return arena_printf(a, "%s failed: %.*s", label, (int)(em.len ? em.len : 13), em.len ? em.ptr : "Unknown error");
    }
    if (str_eqz(stop, "length")) return arena_printf(a, "%s failed: generation hit the token cap and the summary is incomplete", label);
    const JValue *rc = json_get(resp, "content");
    for (size_t i = 0; json_is(rc, J_ARRAY) && i < rc->arr.len; i++)
        if (str_eqz(type_of(rc->arr.items[i]), "toolCall")) return arena_printf(a, "%s attempted to call a tool", label);
    Buf t = {0};
    content_text(&t, rc, "\n");
    *text = arena_strndup(a, t.ptr ? t.ptr : "", t.len);
    buf_free(&t);
    *usage = json_get(resp, "usage");
    return NULL;
}

// combineUsage
static JValue *combine_usage(Arena *a, const JValue *x, const JValue *y) {
    JValue *u = json_new(a, J_OBJECT);
    static const char *const k1[] = {"input", "output", "cacheRead", "cacheWrite"};
    for (size_t i = 0; i < 4; i++) json_put(a, u, str(k1[i]), jnumber(a, jnum(json_get(x, k1[i])) + jnum(json_get(y, k1[i]))));
    if (json_get(x, "cacheWrite1h") || json_get(y, "cacheWrite1h"))
        json_put(a, u, S("cacheWrite1h"), jnumber(a, jnum(json_get(x, "cacheWrite1h")) + jnum(json_get(y, "cacheWrite1h"))));
    if (json_get(x, "reasoning") || json_get(y, "reasoning"))
        json_put(a, u, S("reasoning"), jnumber(a, jnum(json_get(x, "reasoning")) + jnum(json_get(y, "reasoning"))));
    json_put(a, u, S("totalTokens"), jnumber(a, jnum(json_get(x, "totalTokens")) + jnum(json_get(y, "totalTokens"))));
    JValue *c = json_new(a, J_OBJECT);
    const JValue *cx = json_get(x, "cost"), *cy = json_get(y, "cost");
    static const char *const k2[] = {"input", "output", "cacheRead", "cacheWrite", "total"};
    for (size_t i = 0; i < 5; i++) json_put(a, c, str(k2[i]), jnumber(a, jnum(json_get(cx, k2[i])) + jnum(json_get(cy, k2[i]))));
    json_put(a, c, S("totalFromProvider"), json_bool(a, jtrue(json_get(cx, "totalFromProvider")) && jtrue(json_get(cy, "totalFromProvider"))));
    json_put(a, u, S("cost"), c);
    return u;
}

// generateSummaryWithUsage
static const char *history_summary(Arena *a, const CodeModel *m, const CodeEnv *env, const JValue *msgs, const char *instructions,
                                   const char *previous, char **text, const JValue **usage) {
    Buf p = {0}, conv = {0};
    serialize(&conv, msgs);
    buf_adds(&p, "<conversation>\n");
    buf_addstr(&p, buf_str(&conv));
    buf_adds(&p, "\n</conversation>\n\n");
    if (previous) buf_printf(&p, "<previous-summary>\n%s\n</previous-summary>\n\n", previous);
    buf_adds(&p, previous ? UPDATE_SUMMARIZATION_PROMPT : SUMMARIZATION_PROMPT);
    if (instructions && *instructions) buf_printf(&p, "\n\nAdditional focus: %s", instructions);
    const char *err = summarize(a, m, env, buf_str(&p), "Summarization", text, usage);
    buf_free(&p);
    buf_free(&conv);
    return err;
}

// generateTurnPrefixSummary
static const char *turn_prefix_summary(Arena *a, const CodeModel *m, const CodeEnv *env, const JValue *msgs, const char *instructions,
                                       char **text, const JValue **usage) {
    Buf p = {0}, conv = {0};
    serialize(&conv, msgs);
    buf_adds(&p, "# Conversation\n");
    buf_addstr(&p, buf_str(&conv));
    buf_adds(&p, "\n\n# Instructions\n");
    buf_adds(&p, TURN_PREFIX_SUMMARIZATION_PROMPT);
    if (instructions && *instructions) buf_printf(&p, "\n\nAdditional focus: %s", instructions);
    const char *err = summarize(a, m, env, buf_str(&p), "Turn prefix summarization", text, usage);
    buf_free(&p);
    buf_free(&conv);
    return err;
}

const char *code_sessions_compact(Arena *a, CodeSession *s, const CodeModel *m, const CodeEnv *env, const char *instructions,
                                  int64_t keep_recent, const JValue **entry) {
    return code_sessions_compact_cb(a, s, m, env, instructions, keep_recent, entry, NULL, NULL);
}

double code_sessions_projected_tokens(Arena *a, const JValue *entries, const char *leaf) {
    JValue *path = code_sessions_branch(a, entries, leaf);
    Projection p = {0};
    JValue *all = json_new(a, J_ARRAY);
    project(a, path, &p, all);
    double t = projected_tokens(a, &p, all, path);
    vec_free(&p);
    return t;
}

JValue *code_sessions_projection(Arena *a, const JValue *entries, const char *leaf) {
    JValue *path = code_sessions_branch(a, entries, leaf);
    Projection p = {0};
    JValue *all = json_new(a, J_ARRAY);
    project(a, path, &p, all);
    vec_free(&p);
    return all;
}

const char *code_sessions_compact_cb(Arena *a, CodeSession *s, const CodeModel *m, const CodeEnv *env, const char *instructions,
                                     int64_t keep_recent, const JValue **entry, void (*started)(void *ctx), void *ctx) {
    *entry = NULL;
    const JValue *entries = code_session_entries(s);
    JValue *path = code_sessions_branch(a, entries, code_session_leaf(s));
    // prepareCompaction
    if (path->arr.len && str_eqz(type_of(path->arr.items[path->arr.len - 1]), "compaction")) return "Already compacted";
    Projection p = {0};
    JValue *all = json_new(a, J_ARRAY);
    project(a, path, &p, all);
    ptrdiff_t prev = -1;
    for (size_t i = 0; i < p.len && prev < 0; i++)
        if (str_eqz(type_of(p.ptr[i].src), "compaction") && p.ptr[i].msgs->arr.len) prev = (ptrdiff_t)i;
    const char *previous = NULL;
    size_t start = 0;
    if (prev >= 0) {
        previous = json_str(json_get(p.ptr[prev].src, "summary"));
        start = (size_t)prev + 1;
    }
    double tokens_before = projected_tokens(a, &p, all, path);
    const char *err = NULL;
    CutPoint cp = p.len ? find_cut_point(a, &p, start, p.len, keep_recent) : (CutPoint){0, -1, false};
    const JValue *first_kept = cp.first_kept < p.len ? p.ptr[cp.first_kept].src : NULL;
    JValue *to_sum = json_new(a, J_ARRAY), *prefix = json_new(a, J_ARRAY);
    if (first_kept && json_str(json_get(first_kept, "id"))) {
        size_t hist_end = cp.split ? (size_t)cp.turn_start : cp.first_kept;
        for (size_t i = start; i < hist_end; i++) compaction_messages(a, &p.ptr[i], to_sum);
        if (cp.split)
            for (size_t i = (size_t)cp.turn_start; i < cp.first_kept; i++) compaction_messages(a, &p.ptr[i], prefix);
    }
    if (!first_kept || !json_str(json_get(first_kept, "id")) || (!to_sum->arr.len && !prefix->arr.len)) {
        vec_free(&p);
        return "Nothing to compact (session too small)";
    }
    // extractFileOperations: the previous compaction's lists, then the tool calls
    FileOps ops = {0};
    if (prev >= 0) {
        const JValue *pc = p.ptr[prev].src, *d = json_get(pc, "details");
        if (!jtrue(json_get(pc, "fromHook")) && json_is(d, J_OBJECT)) {
            const JValue *rf = json_get(d, "readFiles"), *mf = json_get(d, "modifiedFiles");
            for (size_t i = 0; json_is(rf, J_ARRAY) && i < rf->arr.len; i++)
                if (json_str(rf->arr.items[i])) set_add(&ops.read, json_str(rf->arr.items[i]));
            for (size_t i = 0; json_is(mf, J_ARRAY) && i < mf->arr.len; i++)
                if (json_str(mf->arr.items[i])) set_add(&ops.edited, json_str(mf->arr.items[i]));
        }
    }
    for (size_t i = 0; i < to_sum->arr.len; i++) file_ops_of(to_sum->arr.items[i], &ops);
    if (cp.split)
        for (size_t i = 0; i < prefix->arr.len; i++) file_ops_of(prefix->arr.items[i], &ops);
    vec_free(&p);
    if (started) started(ctx); // the preparation is done: compaction_start (_runAutoCompaction)
    // compact()
    Buf summary = {0};
    const JValue *usage = NULL;
    if (cp.split && prefix->arr.len) {
        char *hist = (char *)(previous ? previous : "No prior history."), *pre = NULL;
        const JValue *hu = NULL, *pu = NULL;
        if (to_sum->arr.len) err = history_summary(a, m, env, to_sum, instructions, previous, &hist, &hu);
        if (!err) err = turn_prefix_summary(a, m, env, prefix, instructions, &pre, &pu);
        if (!err) {
            buf_printf(&summary, "%s\n\n---\n\n**Turn Context (split turn):**\n\n%s", hist, pre);
            usage = hu ? combine_usage(a, hu, pu) : pu;
        }
    } else {
        char *t = NULL;
        err = history_summary(a, m, env, to_sum, instructions, previous, &t, &usage);
        if (!err) buf_adds(&summary, t);
    }
    if (!err && env->abort && *env->abort) err = "aborted";
    if (err) {
        buf_free(&summary);
        vec_free(&ops.read);
        vec_free(&ops.written);
        vec_free(&ops.edited);
        return err;
    }
    JValue *rf, *mf;
    file_lists(a, &ops, &rf, &mf, &summary);
    vec_free(&ops.read);
    vec_free(&ops.written);
    vec_free(&ops.edited);
    // appendCompaction
    double now = wall_ms();
    JValue *f = json_new(a, J_OBJECT);
    json_put(a, f, S("summary"), json_string(a, str_n(arena_strndup(a, summary.ptr ? summary.ptr : "", summary.len), summary.len)));
    json_put(a, f, S("firstKeptEntryId"), json_get(first_kept, "id"));
    json_put(a, f, S("tokensBefore"), jnumber(a, tokens_before));
    JValue *details = json_new(a, J_OBJECT);
    json_put(a, details, S("readFiles"), rf);
    json_put(a, details, S("modifiedFiles"), mf);
    json_put(a, f, S("details"), details);
    if (usage) json_put(a, f, S("usage"), (JValue *)usage);
    json_put(a, f, S("fromHook"), json_bool(a, false));
    // the system message of the session's current projection, at the entry's time
    JValue *cur_path = code_sessions_branch(a, code_session_entries(s), code_session_leaf(s));
    Projection cp2 = {0};
    JValue *cur_msgs = json_new(a, J_ARRAY);
    project(a, cur_path, &cp2, cur_msgs);
    vec_free(&cp2);
    JValue *sys = code_current_system(a, cur_msgs);
    if (sys) {
        sys = shallow(a, sys);
        json_put(a, sys, S("timestamp"), jnumber(a, now));
        json_put(a, f, S("systemMessage"), sys);
    }
    buf_free(&summary);
    Err e = code_session_append_at(s, "compaction", f, now);
    if (e) return arena_printf(a, "%s", e);
    *entry = code_session_entry_last(s);
    return NULL;
}
