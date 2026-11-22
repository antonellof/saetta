// Saetta Code's configuration files (see code_cfg.h).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at
// v1.5.2, b4afddd: packages/coding-agent/src/core/settings-manager.ts, model-config.ts,
// resolve-config-value.ts).
#include "code_cfg.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "code_tui.h"

void cfg_json_pretty(Buf *b, const JValue *v, int depth) {
    if (json_is(v, J_OBJECT) && v->obj.len) {
        buf_adds(b, "{\n");
        for (size_t i = 0; i < v->obj.len; i++) {
            tui_spaces(b, (depth + 1) * 2);
            json_write_string(b, v->obj.items[i].key);
            buf_adds(b, ": ");
            cfg_json_pretty(b, v->obj.items[i].val, depth + 1);
            buf_adds(b, i + 1 < v->obj.len ? ",\n" : "\n");
        }
        tui_spaces(b, depth * 2);
        buf_addc(b, '}');
    } else if (json_is(v, J_ARRAY) && v->arr.len) {
        buf_adds(b, "[\n");
        for (size_t i = 0; i < v->arr.len; i++) {
            tui_spaces(b, (depth + 1) * 2);
            cfg_json_pretty(b, v->arr.items[i], depth + 1);
            buf_adds(b, i + 1 < v->arr.len ? ",\n" : "\n");
        }
        tui_spaces(b, depth * 2);
        buf_addc(b, ']');
    } else json_write(b, v);
}

static const char *base_name(const char *path) {
    const char *s = strrchr(path, '/');
    return s ? s + 1 : path;
}

// settings-manager.ts FileSettingsStorage.withLock: proper-lockfile's lock, the file read under
// it, the new text written whole (here through a temporary file and a rename).
bool cfg_json_update(const char *path, CfgEditFn edit, void *ctx, Buf *why) {
    Arena a = {0};
    Buf lock = {0}, tmp = {0}, text = {0};
    buf_printf(&lock, "%s.lock", path);
    buf_printf(&tmp, "%s.tmp-%d", path, (int)getpid());
    const char *name = base_name(path);
    bool locked = false;
    for (int i = 0; i < 40 && !locked; i++) {
        if (mkdir(lock.ptr, 0755) == 0) locked = true;
        else if (errno == EEXIST) {
            struct stat st;
            if (stat(lock.ptr, &st) == 0 && time(NULL) - st.st_mtime > 10) rmdir(lock.ptr); // stale
            else {
                struct timespec ts = {0, 25000000};
                nanosleep(&ts, NULL);
            }
        } else break;
    }
    bool ok = false;
    if (!locked) {
        if (why) buf_printf(why, "%s is locked by another program; it is not changed", name);
        goto done;
    }
    char *data = NULL;
    size_t n = 0;
    JValue *root = NULL;
    Err re = read_file(path, (size_t)16 << 20, &data, &n);
    if (re && access(path, F_OK) == 0) {
        if (why) buf_printf(why, "%s could not be read; it is not changed", name);
        rmdir(lock.ptr);
        goto done;
    }
    if (!re) {
        Str t = str_n(data ? data : "", n);
        if (str_starts(t, S("\xef\xbb\xbf"))) t = str_sub(t, 3, t.len); // stripBom
        if (str_trim(t).len && (json_parse(&a, t, false, &root) || !json_is(root, J_OBJECT))) {
            free(data);
            if (why) buf_printf(why, "%s does not parse; it is not changed (fix the file, then try again)", path);
            rmdir(lock.ptr);
            goto done;
        }
    }
    free(data);
    if (!json_is(root, J_OBJECT)) root = json_new(&a, J_OBJECT);
    if (!edit(&a, root, ctx)) {
        rmdir(lock.ptr);
        ok = true;
        goto done;
    }
    cfg_json_pretty(&text, root, 0);
    int fd = open(tmp.ptr, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    ok = fd >= 0;
    if (fd >= 0) {
        // keep the file's mode (a models.json holding keys may be private)
        struct stat st;
        fchmod(fd, stat(path, &st) == 0 ? (st.st_mode & 07777) : 0644);
    }
    for (size_t off = 0; ok && off < text.len;) {
        ssize_t w = write(fd, text.ptr + off, text.len - off);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) ok = false;
        else off += (size_t)w;
    }
    if (ok && fsync(fd)) ok = false;
    if (fd >= 0 && close(fd)) ok = false;
    if (ok && rename(tmp.ptr, path)) ok = false;
    if (!ok) {
        unlink(tmp.ptr);
        if (why) buf_printf(why, "Could not write %s: %s", name, strerror(errno));
    }
    rmdir(lock.ptr);
done:
    buf_free(&lock);
    buf_free(&tmp);
    buf_free(&text);
    arena_reset(&a);
    return ok;
}

// ---- settings.json ----------------------------------------------------------------------------

typedef struct {
    const char *provider, *model;
} Defaults;

static bool put_defaults(Arena *a, JValue *root, void *ctx) {
    const Defaults *d = ctx;
    // settings-manager.ts setDefaultModelAndProvider: both fields set, the rest kept
    json_put(a, root, S("defaultProvider"), json_string(a, str(d->provider)));
    json_put(a, root, S("defaultModel"), json_string(a, str(d->model)));
    return true;
}

bool cfg_set_default_model(const char *agent_dir, const char *provider, const char *model, Buf *why) {
    Arena a = {0};
    Defaults d = {provider, model};
    bool ok = cfg_json_update(path_join(&a, agent_dir, "settings.json"), put_defaults, &d, why);
    arena_reset(&a);
    return ok;
}

// ---- models.json ------------------------------------------------------------------------------

static void obj_remove(JValue *obj, size_t i) {
    memmove(&obj->obj.items[i], &obj->obj.items[i + 1], (obj->obj.len - i - 1) * sizeof(JMember));
    obj->obj.len--;
}

static long obj_find(const JValue *obj, const char *key) {
    for (size_t i = 0; json_is(obj, J_OBJECT) && i < obj->obj.len; i++)
        if (str_eqz(obj->obj.items[i].key, key)) return (long)i;
    return -1;
}

// A deep copy into the arena (the entry outlives the caller's tree).
static JValue *jcopy(Arena *a, const JValue *v) {
    JValue *c = json_new(a, v->kind);
    switch (v->kind) {
    case J_BOOL: c->b = v->b; break;
    case J_INTEGER: c->i = v->i; break;
    case J_FLOAT: c->f = v->f; break;
    case J_NUMBER_STRING:
    case J_STRING: c->s = str(str_z(a, v->s)); break;
    case J_ARRAY:
        for (size_t i = 0; i < v->arr.len; i++) json_push(a, c, jcopy(a, v->arr.items[i]));
        break;
    case J_OBJECT:
        for (size_t i = 0; i < v->obj.len; i++) json_put(a, c, str(str_z(a, v->obj.items[i].key)), jcopy(a, v->obj.items[i].val));
        break;
    default: break;
    }
    return c;
}

typedef struct {
    const char *name, *old_name;
    const JValue *entry;
} ProviderPut;

static bool put_provider(Arena *a, JValue *root, void *ctx) {
    const ProviderPut *p = ctx;
    JValue *providers = json_get(root, "providers");
    if (!json_is(providers, J_OBJECT)) {
        if (!p->entry) return false;
        providers = json_new(a, J_OBJECT);
        json_put(a, root, S("providers"), providers);
    }
    long at = p->old_name && strcmp(p->old_name, p->name) ? obj_find(providers, p->old_name) : -1;
    if (!p->entry) {
        long i = obj_find(providers, p->name);
        if (i < 0) return false;
        obj_remove(providers, (size_t)i);
        return true;
    }
    JValue *e = jcopy(a, p->entry);
    if (at >= 0) {
        // a rename: the entry takes the old one's place (and a provider already under the new
        // name gives way)
        long dup = obj_find(providers, p->name);
        providers->obj.items[at].key = str(str_z(a, str(p->name)));
        providers->obj.items[at].val = e;
        if (dup >= 0) obj_remove(providers, (size_t)dup);
    } else json_put(a, providers, str(p->name), e);
    return true;
}

bool cfg_put_provider(const char *agent_dir, const char *name, const char *old_name, const JValue *entry, Buf *why) {
    Arena a = {0};
    mkdir_p(agent_dir);
    ProviderPut p = {name, old_name, entry};
    bool ok = cfg_json_update(path_join(&a, agent_dir, "models.json"), put_provider, &p, why);
    arena_reset(&a);
    return ok;
}

JValue *cfg_read_models(Arena *a, const char *agent_dir) {
    char *text;
    size_t len;
    if (read_file(path_join(a, agent_dir, "models.json"), (size_t)16 << 20, &text, &len)) return NULL;
    Str t = str_n(text, len);
    if (str_starts(t, S("\xef\xbb\xbf"))) t = str_sub(t, 3, t.len);
    JValue *v = NULL;
    if (json_parse(a, t, false, &v) || !json_is(v, J_OBJECT)) v = NULL;
    free(text);
    return v;
}

JValue *cfg_openai_provider(Arena *a, const JValue *old, const char *base_url, const char *api_key, const char *ids) {
    JValue *e = json_new(a, J_OBJECT);
    // the order saetta's own entry has (launch.c pi_setup): name, baseUrl, api, apiKey, the rest, models
    if (json_is(old, J_OBJECT))
        for (size_t i = 0; i < old->obj.len; i++) json_put(a, e, old->obj.items[i].key, jcopy(a, old->obj.items[i].val));
    JValue *fresh = json_new(a, J_OBJECT);
    if (json_get(e, "name")) json_put(a, fresh, S("name"), json_get(e, "name"));
    json_put(a, fresh, S("baseUrl"), json_string(a, str_trim(str(base_url))));
    json_put(a, fresh, S("api"), json_string(a, S("openai-completions")));
    Str key = str_trim(str(api_key ? api_key : ""));
    json_put(a, fresh, S("apiKey"), json_string(a, key.len ? key : S("none")));
    for (size_t i = 0; i < e->obj.len; i++) {
        Str k = e->obj.items[i].key;
        if (str_eqz(k, "name") || str_eqz(k, "baseUrl") || str_eqz(k, "apiKey") || str_eqz(k, "api") || str_eqz(k, "models")) continue;
        json_put(a, fresh, k, e->obj.items[i].val);
    }
    const JValue *old_models = json_get(old, "models");
    JValue *models = json_new(a, J_ARRAY);
    for (const char *p = ids ? ids : ""; *p;) {
        while (*p == ',' || *p == ' ' || *p == '\t') p++;
        const char *q = p;
        while (*q && *q != ',' && *q != ' ' && *q != '\t') q++;
        if (q == p) break;
        Str id = str_n(p, (size_t)(q - p));
        p = q;
        bool seen = false;
        for (size_t k = 0; k < models->arr.len && !seen; k++) seen = str_eq(json_strs(json_get(models->arr.items[k], "id")), id);
        if (seen) continue;
        const JValue *keep = NULL;
        for (size_t k = 0; json_is(old_models, J_ARRAY) && k < old_models->arr.len && !keep; k++)
            if (str_eq(json_strs(json_get(old_models->arr.items[k], "id")), id)) keep = old_models->arr.items[k];
        if (keep) json_push(a, models, jcopy(a, keep));
        else {
            JValue *m = json_new(a, J_OBJECT);
            json_put(a, m, S("id"), json_string(a, id));
            json_put(a, m, S("name"), json_string(a, id));
            json_push(a, models, m);
        }
    }
    json_put(a, fresh, S("models"), models);
    return fresh;
}

// ---- resolveConfigValue (resolve-config-value.ts) ------------------------------------------------

static struct {
    char *config, *value;
} g_cmd_cache[16];
static size_t g_ncmd;

static bool env_name_char(char c, bool first) { return c == '_' || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (!first && c >= '0' && c <= '9'); }

// executeCommand: execSync(command) (/bin/sh -c), stdout trimmed, empty is undefined, 10 s
static const char *run_command(const char *command) {
    for (size_t i = 0; i < g_ncmd; i++)
        if (!strcmp(g_cmd_cache[i].config, command)) return g_cmd_cache[i].value;
    char *argv[] = {"/bin/sh", "-c", (char *)command + 1, NULL};
    Buf out = {0};
    char *value = NULL;
    if (tui_run(argv, NULL, S(""), &out, 10000)) {
        Str v = str_trim(buf_str(&out));
        if (v.len) value = strndup(v.ptr, v.len);
    }
    buf_free(&out);
    if (g_ncmd < COUNT(g_cmd_cache)) {
        g_cmd_cache[g_ncmd].config = strdup(command);
        g_cmd_cache[g_ncmd++].value = value;
    }
    return value;
}

const char *cfg_resolve_value(Arena *a, const char *config) {
    if (!config) return NULL;
    if (config[0] == '!') {
        const char *v = run_command(config);
        return v ? arena_strdup(a, v) : NULL;
    }
    // parseConfigValueTemplate, resolveTemplate
    Buf b = {0};
    bool ok = true;
    for (const char *p = config; *p && ok;) {
        if (*p != '$') {
            buf_addc(&b, *p++);
            continue;
        }
        char next = p[1];
        if (next == '$' || next == '!') {
            buf_addc(&b, next);
            p += 2;
        } else if (next == '{') {
            const char *end = strchr(p + 2, '}');
            if (!end) {
                buf_addc(&b, '$');
                p++;
                continue;
            }
            bool valid = end > p + 2;
            for (const char *q = p + 2; q < end && valid; q++) valid = env_name_char(*q, q == p + 2);
            if (valid) {
                char *nm = strndup(p + 2, (size_t)(end - p - 2));
                const char *e = getenv(nm);
                free(nm);
                if (e && *e) buf_adds(&b, e);
                else ok = false;
            } else buf_add(&b, p, (size_t)(end + 1 - p));
            p = end + 1;
        } else if (env_name_char(next, true)) {
            const char *q = p + 1;
            while (env_name_char(*q, q == p + 1)) q++;
            char *nm = strndup(p + 1, (size_t)(q - p - 1));
            const char *e = getenv(nm);
            free(nm);
            if (e && *e) buf_adds(&b, e);
            else ok = false;
            p = q;
        } else {
            buf_addc(&b, '$');
            p++;
        }
    }
    const char *r = ok ? arena_strndup(a, b.ptr ? b.ptr : "", b.len) : NULL;
    buf_free(&b);
    return r;
}

// ---- the footer's name for a pack -------------------------------------------------------------------

// A GGUF quantization type as a whole "-" or "." separated token of the file name: Q4_K_M, Q8_0,
// IQ4_XS, Q4_K_XL.
static bool gguf_quant(Str tok) {
    size_t i = 0;
    if (i < tok.len && tok.ptr[i] == 'I') i++;
    if (i >= tok.len || tok.ptr[i] != 'Q') return false;
    i++;
    size_t d = i;
    while (i < tok.len && tok.ptr[i] >= '0' && tok.ptr[i] <= '9') i++;
    if (i == d) return false;
    while (i < tok.len) { // _K, _M, _0, _XL, ...
        if (tok.ptr[i] != '_' || i + 1 >= tok.len) return false;
        i++;
        size_t s = i;
        while (i < tok.len && ((tok.ptr[i] >= 'A' && tok.ptr[i] <= 'Z') || (tok.ptr[i] >= '0' && tok.ptr[i] <= '9'))) i++;
        if (i == s || i - s > 2) return false;
    }
    return true;
}

static int int_cmp(const void *x, const void *y) { return *(const int *)x - *(const int *)y; }

void cfg_pack_label(Arena *a, const char *title, const char *gguf_file, const char *pack_dir, const char **full, const char **brief) {
    *full = *brief = NULL;
    Str t = str(title ? title : "");
    if (!t.len && pack_dir) {
        Str d = str(pack_dir);
        while (d.len > 1 && d.ptr[d.len - 1] == '/') d.len--;
        t = str(base_name(str_z(a, d)));
    }
    if (str_ends(t, S(" 4-bit")) || str_ends(t, S(" 8-bit"))) t.len -= 6;
    const char *quant = NULL, *mixed = NULL;
    for (Str rest = str(gguf_file ? gguf_file : ""); rest.len && !quant;) {
        size_t k = 0;
        while (k < rest.len && rest.ptr[k] != '-' && rest.ptr[k] != '.') k++;
        if (gguf_quant(str_n(rest.ptr, k))) quant = arena_strndup(a, rest.ptr, k);
        rest = k < rest.len ? str_sub(rest, k + 1, rest.len) : S("");
    }
    if (!quant && pack_dir) {
        char *text;
        size_t len;
        JValue *cfg = NULL;
        if (!read_file(path_join(a, pack_dir, "config.json"), (size_t)16 << 20, &text, &len)) {
            if (json_parse(a, str_n(text, len), true, &cfg)) cfg = NULL;
            free(text);
        }
        const JValue *q = json_get(cfg, "quantization");
        const JValue *bits = json_get(q, "bits");
        if (json_is(bits, J_INTEGER)) {
            int set[16], n = 0;
            set[n++] = (int)bits->i;
            for (size_t i = 0; i < q->obj.len; i++) {
                const JValue *b = json_get(q->obj.items[i].val, "bits");
                if (!json_is(b, J_INTEGER)) continue;
                bool have = false;
                for (int k = 0; k < n; k++) have |= set[k] == (int)b->i;
                if (!have && n < 16) set[n++] = (int)b->i;
            }
            quant = arena_printf(a, "%d-bit", (int)bits->i);
            if (n > 1) {
                qsort(set, (size_t)n, sizeof set[0], int_cmp);
                Buf m = {0};
                buf_adds(&m, "(mixed ");
                for (int k = 0; k < n; k++) buf_printf(&m, "%s%d", k ? "/" : "", set[k]);
                buf_addc(&m, ')');
                mixed = arena_strndup(a, m.ptr, m.len);
                buf_free(&m);
            }
        }
    }
    if (!quant) {
        if (t.len && title && t.len != strlen(title)) *full = *brief = str_z(a, str(title)); // the " 8-bit" kept
        return;
    }
    *brief = arena_printf(a, "%.*s%s%s", (int)t.len, t.ptr, t.len ? " " : "", quant);
    *full = mixed ? arena_printf(a, "%s %s", *brief, mixed) : *brief;
}
