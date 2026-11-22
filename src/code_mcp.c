// Saetta Code's MCP support: configuration, the JSON-RPC client over stdio and Streamable HTTP,
// the server connections, the tools offered to the model and the data of the /mcp manager.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at v1.5.2,
// b4afddd): packages/mcp/src/{client,config,naming,input-schema,output-limit}.ts,
// protocol/{jsonrpc,content,types}.ts, transports/{stdio,lines,transport,streamable-http}.ts;
// packages/coding-agent/src/extensions/mcp/{config,index,runtime,tools,resources,log}.ts,
// core/mcp-servers.ts, core/resolve-config-value.ts, core/tools/truncate.ts (formatSize,
// sliceHead, sliceTail), utils/output-files.ts.
//
// Threads: every server connects on a thread of its own; a stdio server's output is read by a
// reader thread per process; a request blocks its caller until the answer, a timeout or an abort.
// One mutex (g_mu) guards all the state; a stdio server's input is a queue under it, written
// without blocking (the sender, then the reader thread when the pipe drains), so no thread waits
// on a server that stops reading.
#include "code_mcp.h"
#include "code_cfg.h"
#include "code_oai.h"
#include "code_tui.h"
#include "hash.h"
#include "metal/shim.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

// ---------------------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------------------

static char *dupz(const char *s) { return s ? strdup(s) : NULL; }

static char *vmfmt(const char *fmt, va_list ap) {
    Buf b = {0};
    buf_vprintf(&b, fmt, ap);
    return b.ptr ? b.ptr : strdup("");
}

static char *mfmt(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static char *mfmt(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char *s = vmfmt(fmt, ap);
    va_end(ap);
    return s;
}

static void set_str(char **slot, char *owned) {
    free(*slot);
    *slot = owned;
}

static char *buf_take(Buf *b) { return b->ptr ? b->ptr : strdup(""); }

// Server-provided text made safe for the terminal before the UI shows it (stderr tails, HTTP
// error bodies, server and tool names, descriptions, notices): escape sequences dropped (as the
// tool results' text is), then every other C0 control (newlines kept when multiline, tabs kept),
// DEL and the C1 controls (U+0080..U+009F). Takes s (malloc'd or NULL), returns a malloc'd copy.
static char *term_safe(char *s, bool multiline) {
    if (!s) return NULL;
    Buf t = {0}, o = {0};
    tui_strip(&t, str(s));
    for (size_t i = 0; i < t.len; i++) {
        unsigned char c = (unsigned char)t.ptr[i];
        if (c == 0xc2 && i + 1 < t.len && (unsigned char)t.ptr[i + 1] >= 0x80 && (unsigned char)t.ptr[i + 1] <= 0x9f) {
            i++;
            continue;
        }
        if ((c < 0x20 && c != '\t' && !(multiline && c == '\n')) || c == 0x7f) continue;
        buf_addc(&o, (char)c);
    }
    buf_free(&t);
    free(s);
    return buf_take(&o);
}

// JSON.stringify, malloc'd
static char *js_text(const JValue *v) {
    Buf b = {0};
    if (v) code_oai_js_write(&b, v);
    else buf_adds(&b, "null");
    return buf_take(&b);
}

static JValue *js_parse(Arena *a, Str s) {
    char *err = NULL;
    return code_oai_js_parse(a, s, &err);
}

static JValue *jcopy(Arena *a, const JValue *v) {
    if (!v) return NULL;
    char *t = js_text(v);
    JValue *c = js_parse(a, str(t));
    free(t);
    return c;
}

// JSON.stringify(v, null, indent)
static void js_pretty(Buf *b, const JValue *v, const char *ind, int level) {
    if (v->kind == J_ARRAY || v->kind == J_OBJECT) {
        bool arr = v->kind == J_ARRAY;
        size_t n = arr ? v->arr.len : v->obj.len;
        if (!n) {
            buf_adds(b, arr ? "[]" : "{}");
            return;
        }
        buf_adds(b, arr ? "[\n" : "{\n");
        for (size_t i = 0; i < n; i++) {
            for (int k = 0; k <= level; k++) buf_adds(b, ind);
            if (arr) js_pretty(b, v->arr.items[i], ind, level + 1);
            else {
                json_write_string(b, v->obj.items[i].key);
                buf_adds(b, ": ");
                js_pretty(b, v->obj.items[i].val, ind, level + 1);
            }
            buf_adds(b, i + 1 < n ? ",\n" : "\n");
        }
        for (int k = 0; k < level; k++) buf_adds(b, ind);
        buf_addc(b, arr ? ']' : '}');
        return;
    }
    code_oai_js_write(b, v);
}

static bool is_record(const JValue *v) { return json_is(v, J_OBJECT); }

static bool jnum(const JValue *v, double *x) {
    if (!v) return false;
    if (v->kind == J_INTEGER) *x = (double)v->i;
    else if (v->kind == J_FLOAT) *x = v->f;
    else if (v->kind == J_NUMBER_STRING) {
        char z[400];
        size_t n = MIN(v->s.len, sizeof z - 1);
        memcpy(z, v->s.ptr, n);
        z[n] = 0;
        *x = strtod(z, NULL);
    } else return false;
    return true;
}

static JValue *jstr(Arena *a, const char *s) { return json_string(a, str(s ? s : "")); }
static void jput(Arena *a, JValue *o, const char *k, JValue *v) { json_put(a, o, str(k), v); }

static void obj_del(JValue *o, const char *k) {
    if (!is_record(o)) return;
    for (size_t i = 0; i < o->obj.len; i++)
        if (str_eqz(o->obj.items[i].key, k)) {
            memmove(&o->obj.items[i], &o->obj.items[i + 1], (o->obj.len - i - 1) * sizeof(JMember));
            o->obj.len--;
            return;
        }
}

// a shallow copy of an object (spread), in the arena
static JValue *obj_spread(Arena *a, const JValue *o) {
    JValue *c = json_new(a, J_OBJECT);
    for (size_t i = 0; is_record(o) && i < o->obj.len; i++) json_put(a, c, o->obj.items[i].key, o->obj.items[i].val);
    return c;
}

// node's error codes for the errno values a spawn or a write can fail with
static const char *errno_code(int e) {
    switch (e) {
    case ENOENT: return "ENOENT";
    case EACCES: return "EACCES";
    case EPERM: return "EPERM";
    case ENOTDIR: return "ENOTDIR";
    case E2BIG: return "E2BIG";
    case ENOEXEC: return "ENOEXEC";
    case EPIPE: return "EPIPE";
    case EAGAIN: return "EAGAIN";
    case ENOMEM: return "ENOMEM";
    case EMFILE: return "EMFILE";
    case EISDIR: return "EISDIR";
    case ELOOP: return "ELOOP";
    case ENAMETOOLONG: return "ENAMETOOLONG";
    default: return "UNKNOWN";
    }
}

// UTF-16 code units of a code point's UTF-8 sequence at s[i] (its byte length in *n)
static int u16_units(Str s, size_t i, size_t *n) {
    unsigned char c = (unsigned char)s.ptr[i];
    size_t len = c < 0x80 ? 1 : c >= 0xf0 ? 4 : c >= 0xe0 ? 3 : c >= 0xc0 ? 2 : 1;
    if (i + len > s.len) len = 1;
    *n = len;
    return len == 4 ? 2 : 1;
}

// sliceTail: at most the last max UTF-16 units, no surrogate pair split
static Str slice_tail(Str s, size_t max) {
    size_t total = 0;
    for (size_t i = 0, n; i < s.len; i += n) total += (size_t)u16_units(s, i, &n);
    if (total <= max) return s;
    size_t skip = total - max, units = 0, i = 0;
    while (i < s.len && units < skip) {
        size_t n;
        units += (size_t)u16_units(s, i, &n);
        i += n;
    }
    return str_sub(s, i, s.len);
}

// sliceHead: at most the first max UTF-16 units
static Str slice_head(Str s, size_t max) {
    size_t units = 0, i = 0;
    while (i < s.len) {
        size_t n;
        int u = u16_units(s, i, &n);
        if (units + (size_t)u > max) break;
        units += (size_t)u;
        i += n;
    }
    return str_sub(s, 0, i);
}

static size_t u16_len(Str s) {
    size_t total = 0;
    for (size_t i = 0, n; i < s.len; i += n) total += (size_t)u16_units(s, i, &n);
    return total;
}

static Str trim_js(Str s) { return str_trim(s); } // String.prototype.trim (ASCII whitespace is what servers send)

static Str first_line(Str s) {
    ptrdiff_t k = str_find_char(s, '\n', 0);
    return k >= 0 ? str_sub(s, 0, (size_t)k) : s;
}

// formatSize (truncate.ts)
static char *format_size(double bytes) {
    if (bytes < 1024) return mfmt("%gB", bytes);
    if (bytes < 1024 * 1024) return mfmt("%.1fKB", bytes / 1024);
    return mfmt("%.1fMB", bytes / (1024 * 1024));
}

static char *path_resolve2(const char *base, const char *p) {
    Buf b = {0};
    if (p[0] == '/') buf_adds(&b, p);
    else {
        buf_adds(&b, base);
        buf_addc(&b, '/');
        buf_adds(&b, p);
    }
    // normalize: segments, . and ..
    char *in = buf_take(&b);
    VEC(char *) seg = {0};
    for (char *save = NULL, *t = strtok_r(in, "/", &save); t; t = strtok_r(NULL, "/", &save)) {
        if (!strcmp(t, ".")) continue;
        if (!strcmp(t, "..")) {
            if (seg.len) seg.len--;
            continue;
        }
        vec_push(&seg, t);
    }
    Buf o = {0};
    for (size_t i = 0; i < seg.len; i++) {
        buf_addc(&o, '/');
        buf_adds(&o, seg.ptr[i]);
    }
    if (!o.len) buf_addc(&o, '/');
    vec_free(&seg);
    free(in);
    return buf_take(&o);
}

static const char *g_home(void) {
    const char *h = getenv("HOME");
    return h && *h ? h : "/";
}

// expandHome (runtime.ts): ~ and ~/... name the home folder
static char *expand_home(const char *v) {
    if (!strcmp(v, "~")) return strdup(g_home());
    if (!strncmp(v, "~/", 2)) return mfmt("%s/%s", g_home(), v + 2);
    return strdup(v);
}

// ---------------------------------------------------------------------------------------
// resolveConfigValueOrThrow (core/resolve-config-value.ts)
// ---------------------------------------------------------------------------------------

typedef struct {
    bool env;
    Buf text; // a literal, or the variable's name
} TplPart;
typedef VEC(TplPart) TplVec;
typedef VEC(char *) CStrVec;

static bool env_name_start(char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_'; }
static bool env_name_char(char c) { return env_name_start(c) || (c >= '0' && c <= '9'); }

static void tpl_literal(TplVec *parts, Str v) {
    if (!v.len) return;
    if (parts->len && !parts->ptr[parts->len - 1].env) {
        buf_addstr(&parts->ptr[parts->len - 1].text, v);
        return;
    }
    TplPart p = {0};
    buf_addstr(&p.text, v);
    vec_push(parts, p);
}

static void tpl_parse(Str c, TplVec *parts) {
    size_t i = 0;
    while (i < c.len) {
        ptrdiff_t d = str_find_char(c, '$', i);
        if (d < 0) {
            tpl_literal(parts, str_sub(c, i, c.len));
            break;
        }
        tpl_literal(parts, str_sub(c, i, (size_t)d));
        char next = (size_t)d + 1 < c.len ? c.ptr[d + 1] : 0;
        if (next == '$' || next == '!') {
            tpl_literal(parts, str_n(&c.ptr[d + 1], 1));
            i = (size_t)d + 2;
            continue;
        }
        if (next == '{') {
            ptrdiff_t e = str_find_char(c, '}', (size_t)d + 2);
            if (e < 0) {
                tpl_literal(parts, S("$"));
                i = (size_t)d + 1;
                continue;
            }
            Str name = str_sub(c, (size_t)d + 2, (size_t)e);
            bool ok = name.len && env_name_start(name.ptr[0]);
            for (size_t k = 0; ok && k < name.len; k++) ok = env_name_char(name.ptr[k]);
            if (ok) {
                TplPart p = {.env = true};
                buf_addstr(&p.text, name);
                vec_push(parts, p);
            } else tpl_literal(parts, str_sub(c, (size_t)d, (size_t)e + 1));
            i = (size_t)e + 1;
            continue;
        }
        size_t k = (size_t)d + 1;
        if (k < c.len && env_name_start(c.ptr[k])) {
            size_t e = k;
            while (e < c.len && env_name_char(c.ptr[e])) e++;
            TplPart p = {.env = true};
            buf_addstr(&p.text, str_sub(c, k, e));
            vec_push(parts, p);
            i = e;
            continue;
        }
        tpl_literal(parts, S("$"));
        i = (size_t)d + 1;
    }
}

// execSync(command): /bin/sh -c, stdout trimmed, 10 s; NULL when it fails or prints nothing
static char *run_shell(const char *command) {
    int out[2];
    if (pipe(out)) return NULL;
    fcntl(out[0], F_SETFD, FD_CLOEXEC);
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&fa, out[1], 1);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    posix_spawnattr_t at;
    posix_spawnattr_init(&at);
    sigset_t none, all;
    sigemptyset(&none);
    sigfillset(&all);
    posix_spawnattr_setsigmask(&at, &none);
    posix_spawnattr_setsigdefault(&at, &all);
    posix_spawnattr_setflags(&at, POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_CLOEXEC_DEFAULT | POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&at, 0);
    const char *argv[] = {"/bin/sh", "-c", command, NULL};
    pid_t pid;
    int rc = posix_spawn(&pid, "/bin/sh", &fa, &at, (char *const *)argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    posix_spawnattr_destroy(&at);
    close(out[1]);
    if (rc) {
        close(out[0]);
        return NULL;
    }
    Buf b = {0};
    uint64_t until = now_ns() + 10000000000ull;
    bool timed_out = false;
    for (;;) {
        uint64_t now = now_ns();
        if (now >= until) {
            timed_out = true;
            break;
        }
        struct pollfd pf = {out[0], POLLIN, 0};
        int r = poll(&pf, 1, (int)((until - now) / 1000000) + 1);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) continue;
        char tmp[4096];
        ssize_t n = read(out[0], tmp, sizeof tmp);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        buf_add(&b, tmp, (size_t)n);
    }
    close(out[0]);
    if (timed_out) kill(-pid, SIGKILL);
    int st = 0;
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
    Str t = str_trim(buf_str(&b));
    char *v = !timed_out && WIFEXITED(st) && WEXITSTATUS(st) == 0 && t.len ? strndup(t.ptr, t.len) : NULL;
    buf_free(&b);
    return v;
}

// resolveConfigValueOrThrow: the value, or NULL with *err set
static char *resolve_config_value(const char *config, const char *description, char **err) {
    *err = NULL;
    if (config[0] == '!') {
        char *v = run_shell(config + 1);
        if (!v) *err = mfmt("Failed to resolve %s from shell command: %s", description, config + 1);
        return v;
    }
    TplVec parts = {0};
    tpl_parse(str(config), &parts);
    Buf out = {0};
    CStrVec missing = {0};
    for (size_t i = 0; i < parts.len; i++) {
        if (!parts.ptr[i].env) {
            buf_addstr(&out, buf_str(&parts.ptr[i].text));
            continue;
        }
        const char *val = getenv(parts.ptr[i].text.ptr);
        if (val && *val) buf_adds(&out, val);
        else {
            bool seen = false;
            for (size_t k = 0; k < missing.len; k++) seen |= !strcmp(missing.ptr[k], parts.ptr[i].text.ptr);
            if (!seen) vec_push(&missing, parts.ptr[i].text.ptr);
        }
    }
    char *v = NULL;
    if (!missing.len) v = buf_take(&out);
    else {
        buf_free(&out);
        Buf m = {0};
        for (size_t k = 0; k < missing.len; k++) buf_printf(&m, "%s%s", k ? ", " : "", missing.ptr[k]);
        *err = missing.len == 1 ? mfmt("Failed to resolve %s from environment variable: %s", description, m.ptr)
                                : mfmt("Failed to resolve %s from environment variables: %s", description, m.ptr);
        buf_free(&m);
    }
    vec_free(&missing);
    for (size_t i = 0; i < parts.len; i++) buf_free(&parts.ptr[i].text);
    vec_free(&parts);
    return v;
}

// ---------------------------------------------------------------------------------------
// URLs (the parts of WHATWG URL the configuration checks use)
// ---------------------------------------------------------------------------------------

typedef struct {
    char scheme[16];
    char host[256]; // as URL.hostname: [::1] keeps its brackets
    char port[8];
    char *path, *query, *frag; // malloc'd
} Url;

static void url_free(Url *u) {
    free(u->path);
    free(u->query);
    free(u->frag);
}

static bool url_parse(const char *s, Url *u) {
    memset(u, 0, sizeof *u);
    const char *c = strstr(s, "://");
    if (!c || c == s || (size_t)(c - s) >= sizeof u->scheme) return false;
    for (const char *p = s; p < c; p++) {
        char ch = *p;
        bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (p > s && ((ch >= '0' && ch <= '9') || ch == '+' || ch == '-' || ch == '.'));
        if (!ok) return false;
        u->scheme[p - s] = (char)(ch | 0x20);
    }
    const char *h = c + 3;
    size_t hl = strcspn(h, "/?#");
    char auth[512];
    if (hl >= sizeof auth) return false;
    memcpy(auth, h, hl);
    auth[hl] = 0;
    char *at = strrchr(auth, '@');
    char *hp = at ? at + 1 : auth;
    char *colon = hp[0] == '[' ? strchr(strchr(hp, ']') ? strchr(hp, ']') : hp, ':') : strrchr(hp, ':');
    if (colon) {
        snprintf(u->port, sizeof u->port, "%s", colon + 1);
        *colon = 0;
        for (char *p = u->port; *p; p++)
            if (*p < '0' || *p > '9') return false;
    }
    if (!*hp) return false;
    snprintf(u->host, sizeof u->host, "%s", hp);
    for (char *p = u->host; *p; p++) *p = (char)((*p >= 'A' && *p <= 'Z') ? *p + 32 : *p);
    const char *rest = h + hl;
    size_t pl = strcspn(rest, "?#");
    u->path = pl ? strndup(rest, pl) : strdup("/");
    rest += pl;
    if (*rest == '?') {
        size_t ql = strcspn(rest, "#");
        u->query = strndup(rest, ql);
        rest += ql;
    } else u->query = strdup("");
    u->frag = strdup(*rest == '#' ? rest : "");
    // URL drops the default port
    if ((!strcmp(u->scheme, "http") && !strcmp(u->port, "80")) || (!strcmp(u->scheme, "https") && !strcmp(u->port, "443"))) u->port[0] = 0;
    if (!strcmp(u->query, "?")) u->query[0] = 0;
    if (!strcmp(u->frag, "#")) u->frag[0] = 0;
    return true;
}

static bool loopback_host(const char *h) { return !strcmp(h, "localhost") || !strcmp(h, "127.0.0.1") || !strcmp(h, "[::1]"); }

// pathToFileURL(path).href
static char *file_url(const char *path) {
    Buf b = {0};
    buf_adds(&b, "file://");
    for (const unsigned char *p = (const unsigned char *)path; *p; p++) {
        unsigned char c = *p;
        if (c <= 0x20 || c >= 0x7f || strchr("\"#%<>?`{}", c)) buf_printf(&b, "%%%02X", c);
        else buf_addc(&b, (char)c);
    }
    return buf_take(&b);
}

// ---------------------------------------------------------------------------------------
// configuration: the mcpServers shape (pi-mcp config.ts) and pi's fields (core/mcp-servers.ts)
// ---------------------------------------------------------------------------------------

static const char *const EXPOSURES[] = {"codemode", "deferred", "direct", "hidden"};

static bool is_exposure(const JValue *v) {
    for (size_t i = 0; i < COUNT(EXPOSURES); i++)
        if (json_is(v, J_STRING) && str_eqz(v->s, EXPOSURES[i])) return true;
    return false;
}

static bool is_string_record(const JValue *v) {
    if (!is_record(v)) return false;
    for (size_t i = 0; i < v->obj.len; i++)
        if (!json_is(v->obj.items[i].val, J_STRING)) return false;
    return true;
}

// isMcpServerOverride: no command, url or type
static bool is_override(const JValue *e) { return !json_get(e, "command") && !json_get(e, "url") && !json_get(e, "type"); }

static bool server_name_ok(const char *n) {
    if (!*n) return false;
    for (const char *p = n; *p; p++)
        if (!((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z') || (*p >= '0' && *p <= '9') || *p == '_' || *p == '-')) return false;
    return true;
}

// isClientMetadataDocumentCallback
static bool cimd_callback_ok(const JValue *cb) {
    if (!json_is(cb, J_STRING)) return true;
    Url u;
    if (!url_parse(cb->s.ptr, &u)) return false;
    bool ok = strcmp(u.host, "[::1]") && !strcmp(u.path, "/callback");
    url_free(&u);
    return ok;
}

static char *validate_oauth(const JValue *v) {
    if (!v) return NULL;
    if (!is_record(v)) return strdup("oauth must be an object");
    const JValue *x;
    if ((x = json_get(v, "clientId")) && !json_is(x, J_STRING)) return strdup("oauth.clientId must be a string");
    if ((x = json_get(v, "clientSecret")) && !json_is(x, J_STRING)) return strdup("oauth.clientSecret must be a string");
    const JValue *port = json_get(v, "callbackPort");
    double pn = 0;
    if (port && (!jnum(port, &pn) || pn != trunc(pn) || pn < 1 || pn > 65535)) return strdup("oauth.callbackPort must be a port number");
    const JValue *cb = json_get(v, "callbackUrl");
    if (cb) {
        Url u;
        bool ok = json_is(cb, J_STRING) && url_parse(cb->s.ptr, &u);
        if (ok) {
            ok = !strcmp(u.scheme, "http") && loopback_host(u.host) && !u.query[0] && !u.frag[0];
            if (ok && u.port[0] && port && atof(u.port) != pn) {
                url_free(&u);
                return strdup("oauth.callbackUrl and oauth.callbackPort name different ports");
            }
            url_free(&u);
        }
        if (!ok) return strdup("oauth.callbackUrl must be an http URI on localhost, 127.0.0.1, or [::1] without query or fragment");
    }
    if ((x = json_get(v, "scope")) && !json_is(x, J_STRING)) return strdup("oauth.scope must be a string");
    if ((x = json_get(v, "clientName")) && (!json_is(x, J_STRING) || !str_trim(x->s).len)) return strdup("oauth.clientName must be a non-empty string");
    const JValue *reg = json_get(v, "clientRegistration");
    if (reg && !(json_is(reg, J_STRING) && (str_eqz(reg->s, "auto") || str_eqz(reg->s, "dcr")))) {
        if (!(json_is(reg, J_STRING) && str_eqz(reg->s, "cimd"))) return strdup("oauth.clientRegistration must be \"auto\", \"dcr\", or \"cimd\"");
        if (json_get(v, "clientId") || json_get(v, "clientName"))
            return strdup("oauth.clientRegistration \"cimd\" cannot be combined with oauth.clientId or oauth.clientName");
        if (!cimd_callback_ok(cb))
            return strdup("oauth.clientRegistration \"cimd\" requires oauth.callbackUrl on localhost or 127.0.0.1 with path /callback");
    }
    const JValue *md = json_get(v, "authServerMetadataUrl");
    if (md) {
        Url u;
        bool ok = json_is(md, J_STRING) && url_parse(md->s.ptr, &u);
        if (ok) {
            ok = !strcmp(u.scheme, "https") || (!strcmp(u.scheme, "http") && loopback_host(u.host));
            url_free(&u);
        }
        if (!ok) return strdup("oauth.authServerMetadataUrl must be an https URL, or http on localhost, 127.0.0.1, or [::1]");
    }
    return NULL;
}

// validateMcpServerConfig (core/mcp-servers.ts over pi-mcp's): the config (exposure aliases
// resolved, in the arena), or NULL with *err set (malloc'd)
static JValue *validate_server(Arena *a, const char *name, const JValue *raw0, char **err) {
    *err = NULL;
    if (!server_name_ok(name)) {
        *err = mfmt("invalid server name \"%s\" (use letters, digits, \"_\" and \"-\")", name);
        return NULL;
    }
    if (!is_record(raw0)) {
        *err = mfmt("server \"%s\" must be an object", name);
        return NULL;
    }
    // resolveExposureAliases
    JValue *raw = obj_spread(a, raw0);
    const JValue *ex = json_get(raw, "exposure");
    if (json_is(ex, J_STRING) && str_eqz(ex->s, "codemode-deferred")) jput(a, raw, "exposure", jstr(a, "codemode"));
    const JValue *tex = json_get(raw, "toolExposure");
    if (is_record(tex)) {
        JValue *t = json_new(a, J_OBJECT);
        for (size_t i = 0; i < tex->obj.len; i++) {
            JValue *v = tex->obj.items[i].val;
            json_put(a, t, tex->obj.items[i].key, json_is(v, J_STRING) && str_eqz(v->s, "codemode-deferred") ? jstr(a, "codemode") : v);
        }
        jput(a, raw, "toolExposure", t);
    }
    // validateBaseFields
    const JValue *x;
    const char *base = NULL;
    double tn;
    if ((x = json_get(raw, "enabled")) && !json_is(x, J_BOOL)) base = "enabled must be a boolean";
    else if ((x = json_get(raw, "description")) && !json_is(x, J_STRING)) base = "description must be a string";
    else if ((x = json_get(raw, "timeout")) && (!jnum(x, &tn) || !(tn > 0))) base = "timeout must be a positive number of seconds";
    if (base) {
        *err = mfmt("server \"%s\": %s", name, base);
        return NULL;
    }
    const JValue *type = json_get(raw, "type"), *url = json_get(raw, "url"), *command = json_get(raw, "command");
    bool http = false;
    if (json_is(type, J_STRING) && str_eqz(type->s, "sse")) {
        *err = mfmt("server \"%s\": legacy SSE transport is not supported; use the streamable HTTP URL", name);
        return NULL;
    }
    if (json_is(url, J_STRING) && (!type || (json_is(type, J_STRING) && (str_eqz(type->s, "http") || str_eqz(type->s, "streamable-http"))))) {
        Url u;
        char *e = NULL;
        bool ok = url_parse(url->s.ptr, &u);
        if (ok) {
            ok = !strcmp(u.scheme, "http") || !strcmp(u.scheme, "https");
            url_free(&u);
        }
        if (!ok) e = strdup("url must be an http or https URL");
        else if ((x = json_get(raw, "headers")) && !is_string_record(x)) e = strdup("headers must map names to strings");
        else e = validate_oauth(json_get(raw, "oauth"));
        if (e) {
            *err = mfmt("server \"%s\": %s", name, e);
            free(e);
            return NULL;
        }
        http = true;
    } else if (json_is(command, J_STRING) && (!type || (json_is(type, J_STRING) && str_eqz(type->s, "stdio")))) {
        const char *e = NULL;
        const JValue *args = json_get(raw, "args");
        if (args) {
            bool ok = json_is(args, J_ARRAY);
            for (size_t i = 0; ok && i < args->arr.len; i++) ok = json_is(args->arr.items[i], J_STRING);
            if (!ok) e = "args must be an array of strings";
        }
        if (!e && (x = json_get(raw, "env")) && !is_string_record(x)) e = "env must map names to strings";
        if (!e && (x = json_get(raw, "cwd")) && !json_is(x, J_STRING)) e = "cwd must be a string";
        if (e) {
            *err = mfmt("server \"%s\": %s", name, e);
            return NULL;
        }
    } else {
        *err = mfmt("server \"%s\" needs either \"command\" (stdio) or \"url\" (streamable HTTP)", name);
        return NULL;
    }
    // checkExposures
    const char *exps = "\"codemode\", \"deferred\", \"direct\", \"hidden\"";
    ex = json_get(raw, "exposure");
    if (ex && !is_exposure(ex)) {
        *err = mfmt("server \"%s\": exposure must be one of %s", name, exps);
        return NULL;
    }
    tex = json_get(raw, "toolExposure");
    if (tex) {
        if (!is_record(tex)) {
            *err = mfmt("server \"%s\": toolExposure must map tool names to exposures", name);
            return NULL;
        }
        for (size_t i = 0; i < tex->obj.len; i++)
            if (!is_exposure(tex->obj.items[i].val)) {
                *err = mfmt("server \"%s\": toolExposure \"%.*s\" must be one of %s", name, (int)tex->obj.items[i].key.len,
                            tex->obj.items[i].key.ptr, exps);
                return NULL;
            }
    }
    // checkAuth
    const JValue *auth = json_get(raw, "auth");
    if (http && auth) {
        const JValue *p = json_get(auth, "provider");
        if (!is_record(auth) || !json_is(p, J_STRING) || !p->s.len) {
            *err = mfmt("server \"%s\": auth.provider must be a provider name", name);
            return NULL;
        }
        Url u;
        bool ok = url_parse(url->s.ptr, &u);
        if (ok) {
            ok = !strcmp(u.scheme, "https") || loopback_host(u.host);
            url_free(&u);
        }
        if (!ok) {
            *err = mfmt("server \"%s\": auth requires an https URL, or http on localhost, 127.0.0.1, or [::1]", name);
            return NULL;
        }
    }
    return raw;
}

// mcpNamespace (naming.ts)
static char *mcp_namespace(const char *server) {
    char *s = mfmt("mcp__%s", server);
    for (char *p = s; *p; p++)
        if (*p == '-') *p = '_';
    return s;
}

static bool is_http(const JValue *config) { return json_is(json_get(config, "url"), J_STRING); }

// toolPatternRegExp: * matches any characters, the rest literally
static bool pattern_match(const char *pat, const char *s) {
    if (!*pat) return !*s;
    if (*pat == '*') {
        for (const char *t = s;; t++) {
            if (pattern_match(pat + 1, t)) return true;
            if (!*t) return false;
        }
    }
    return *s == *pat && pattern_match(pat + 1, s + 1);
}

// getMcpToolExposure: the tool's toolExposure entry (exact, else the first matching pattern),
// else the server's exposure
static const char *tool_exposure(const JValue *config, const char *tool) {
    const JValue *o = json_get(config, "toolExposure");
    if (is_record(o)) {
        const JValue *exact = json_get(o, tool);
        if (json_is(exact, J_STRING)) return exact->s.ptr;
        for (size_t i = 0; i < o->obj.len; i++) {
            Str k = o->obj.items[i].key;
            if (str_find_char(k, '*', 0) >= 0 && pattern_match(k.ptr, tool) && json_is(o->obj.items[i].val, J_STRING))
                return o->obj.items[i].val->s.ptr;
        }
    }
    const char *e = json_str(json_get(config, "exposure"));
    return e ? e : "codemode";
}

static const char *exposure_of(const JValue *config) {
    const char *e = json_str(json_get(config, "exposure"));
    return e ? e : "codemode";
}

// Saetta Code has neither the codemode nor the tool_search tool, which pi reaches "codemode" and
// "deferred" tools with: their tools are declared to the model as "direct" ones are.
static bool declared_exposure(const char *e) { return strcmp(e, "hidden") != 0; }

// ---------------------------------------------------------------------------------------
// the state
// ---------------------------------------------------------------------------------------

typedef enum { ST_CONNECTING, ST_CONNECTED, ST_DISCONNECTED, ST_NEEDS_AUTH, ST_FAILED, ST_CLOSED } ConnState;
static const char *const STATE_NAMES[] = {"connecting", "connected", "disconnected", "needs-auth", "failed", "closed"};

typedef struct Pending {
    struct Pending *next;
    int64_t id;
    bool done, cancellable, wants_progress;
    char *result; // the result as JSON text
    char *err;
    int code;
    uint64_t timeout_ms, deadline_ns;
    CStrVec progress; // progress texts not handed to the caller yet, in order
} Pending;

typedef struct Client Client;
typedef struct Conn Conn;

typedef struct {
    char *name, *json; // the tool's name and its object (JSON text)
} ToolRec;

// McpClient with its transport
struct Client {
    int refs;
    char *server, *cwd;
    bool http;
    // stdio
    pid_t pid;
    int in_fd, out_fd, err_fd;
    // what goes to the server's stdin (g_mu): whole lines queued, written without blocking by the
    // sender and by the reader thread when stdin is writable again; the reader woken by wake_w
    Buf outq;
    size_t outq_off;
    int wake_r, wake_w;
    bool exited;
    Buf errtail;
    // http
    char *url;
    char **hdr_names, **hdr_values;
    size_t nhdr;
    char *session_id, *protocol_version;
    volatile int http_cancel; // set by close(): the requests in flight are cancelled
    // the client
    bool closed, closing, connected;
    Pending *pending;
    int64_t next_id;
    Conn *conn;
    bool cap_tools, cap_resources;
    char *instructions;
};

// McpServerConnection (runtime.ts)
struct Conn {
    char *name;
    JValue *config; // the entry's config when it was created (g_arena)
    ConnState state;
    char *error;
    VEC(ToolRec) tools;
    bool has_resources;
    char *instructions;
    Client *client;
    Client *connecting; // the client of a connect in progress (shutdown closes it)
    bool opening, shut;
    char *open_err; // why the last open failed, for the callers that waited on it
    char *stderr_tail;
};

// McpServer (index.ts)
typedef struct {
    char *name, *source, *scope, *override;
    JValue *config;
    Conn *conn;
    char *message;
    uint64_t attempt;
    int inflight; // connection attempts and reconnects not settled (server.ready)
    int closing;  // a detached connection still closing
} Server;

// a registered tool (pi.registerTool of an MCP tool definition)
typedef enum { REG_TOOL, REG_RESOURCES_LIST, REG_RESOURCES_TEMPLATES, REG_RESOURCES_READ } RegKind;
typedef struct {
    RegKind kind;
    char *name, *server, *tool, *label, *description;
    char *params; // the parameters schema, JSON text
    char *exposure;
    bool current; // offered by the server now (serverTools)
    bool hidden;
} Reg;

typedef struct {
    int kind;
    char *msg;
} Notice;

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
static struct {
    bool started, enabled, shutdown;
    Arena arena; // configs (grows; guarded by g_mu)
    char *cwd, *agent_dir, *project_config, *log_path;
    VEC(Server *) servers;
    VEC(char *) config_errors;
    VEC(Reg *) regs;
    VEC(char *) owners; // toolOwners: name \x01 owner, in order
    VEC(Notice) notices;
    // the allowlist (--tools)
    char **allow;
    size_t nallow;
    bool allow_given, allow_mcp;
    int startup_left;
    bool waited_startup;
    int bg; // background threads running
    uint64_t attempt_seq;
    const char *resources_exposure; // syncResourceTools' last exposure, NULL before
    void (*wake)(void);
    _Atomic uint64_t gen;
} G;

// process groups of running stdio servers, for the exit hook (signal-safe reads)
#define MAX_LIVE 256
static volatile pid_t g_live[MAX_LIVE];

static void live_add(pid_t p) {
    for (int i = 0; i < MAX_LIVE; i++)
        if (!g_live[i]) {
            g_live[i] = p;
            return;
        }
}

static void live_del(pid_t p) {
    for (int i = 0; i < MAX_LIVE; i++)
        if (g_live[i] == p) g_live[i] = 0;
}

void code_mcp_kill_signal_safe(void) {
    for (int i = 0; i < MAX_LIVE; i++)
        if (g_live[i] > 0) kill(-g_live[i], SIGTERM);
}

static void view_changed_l(void);

static void changed_l(void) {
    view_changed_l(); // emitChange: the open menu rebuilt (subscribe)
    atomic_fetch_add(&G.gen, 1);
    pthread_cond_broadcast(&g_cv);
    if (G.wake) G.wake();
}

uint64_t code_mcp_generation(void) { return atomic_load(&G.gen); }

void code_mcp_set_waker(void (*wake)(void)) {
    pthread_mutex_lock(&g_mu);
    G.wake = wake;
    pthread_mutex_unlock(&g_mu);
}

static void notify_l(int kind, char *owned) {
    vec_push(&G.notices, ((Notice){kind, owned}));
    changed_l();
}

void code_mcp_notify(int kind, const char *msg) {
    pthread_mutex_lock(&g_mu);
    notify_l(kind, strdup(msg));
    pthread_mutex_unlock(&g_mu);
}

bool code_mcp_ui_tick(void (*notify)(int kind, const char *msg)) {
    static uint64_t seen;
    pthread_mutex_lock(&g_mu);
    Notice *n = G.notices.ptr;
    size_t len = G.notices.len;
    G.notices.ptr = NULL;
    G.notices.len = G.notices.cap = 0;
    pthread_mutex_unlock(&g_mu);
    for (size_t i = 0; i < len; i++) {
        char *msg = term_safe(n[i].msg, true);
        notify(n[i].kind, msg);
        free(msg);
    }
    free(n);
    uint64_t g = code_mcp_generation();
    bool redraw = len > 0 || g != seen;
    seen = g;
    return redraw;
}

static void timed_wait_l(uint64_t until_ns) {
    uint64_t now = now_ns();
    if (now >= until_ns) return;
    uint64_t d = until_ns - now;
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t ns = (uint64_t)ts.tv_nsec + d % 1000000000ull;
    ts.tv_sec += (time_t)(d / 1000000000ull + ns / 1000000000ull);
    ts.tv_nsec = (long)(ns % 1000000000ull);
    pthread_cond_timedwait(&g_cv, &g_mu, &ts);
}

static Server *find_server_l(const char *name) {
    for (size_t i = 0; i < G.servers.len; i++)
        if (!strcmp(G.servers.ptr[i]->name, name)) return G.servers.ptr[i];
    return NULL;
}

static bool server_listed_l(const Server *s) {
    for (size_t i = 0; i < G.servers.len; i++)
        if (G.servers.ptr[i] == s) return true;
    return false;
}

static bool enabled_of(const JValue *config) {
    const JValue *e = json_get(config, "enabled");
    return !(json_is(e, J_BOOL) && !e->b);
}

static bool is_enabled(const Server *s) { return enabled_of(s->config); }

static uint64_t timeout_ms_of(const JValue *config) {
    double t = 60;
    jnum(json_get(config, "timeout"), &t);
    return (uint64_t)(t * 1000);
}

// configuredExposures has one of these
static bool has_exposure(const JValue *config, bool (*pred)(const char *)) {
    if (pred(exposure_of(config))) return true;
    const JValue *t = json_get(config, "toolExposure");
    for (size_t i = 0; is_record(t) && i < t->obj.len; i++)
        if (json_is(t->obj.items[i].val, J_STRING) && pred(t->obj.items[i].val->s.ptr)) return true;
    return false;
}

static void bg_begin_l(void) { G.bg++; }

static void bg_end(void) {
    pthread_mutex_lock(&g_mu);
    G.bg--;
    pthread_cond_broadcast(&g_cv);
    pthread_mutex_unlock(&g_mu);
}

static void spawn_detached(void *(*fn)(void *), void *arg) {
    pthread_t t;
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    pthread_create(&t, &at, fn, arg);
    pthread_attr_destroy(&at);
}

// ---------------------------------------------------------------------------------------
// the server log (log.ts): notifications/message lines appended to agent_dir/mcp.log
// ---------------------------------------------------------------------------------------

static void iso_now(char out[32]) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    gmtime_r(&ts.tv_sec, &tm);
    snprintf(out, 32, "%04d-%02d-%02dT%02d:%02d:%02d.%03ldZ", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min,
             tm.tm_sec, ts.tv_nsec / 1000000);
}

static void log_write(const char *server, const JValue *params) {
    if (!G.log_path) return;
    Arena t = {0};
    const JValue *msg = params;
    if (!is_record(msg)) {
        JValue *w = json_new(&t, J_OBJECT);
        if (params) jput(&t, w, "data", (JValue *)params);
        msg = w;
    }
    const char *level = json_str(json_get(msg, "level"));
    const char *logger = json_str(json_get(msg, "logger"));
    const JValue *data = json_get(msg, "data");
    Buf text = {0};
    if (json_is(data, J_STRING)) buf_addstr(&text, data->s);
    else if (data) code_oai_js_write(&text, data);
    else buf_adds(&text, "undefined");
    Buf line = {0};
    char ts[32];
    iso_now(ts);
    buf_printf(&line, "%s [%s] %s", ts, server, level ? level : "info");
    if (logger && *logger) buf_printf(&line, " %s:", logger);
    buf_addc(&line, ' ');
    Str tx = buf_str(&text);
    for (size_t i = 0; i < tx.len; i++) {
        if (tx.ptr[i] == '\r' && i + 1 < tx.len && tx.ptr[i + 1] == '\n') continue;
        if (tx.ptr[i] == '\n') buf_adds(&line, "\n    ");
        else buf_addc(&line, tx.ptr[i]);
    }
    buf_addc(&line, '\n');
    struct stat st;
    if (stat(G.log_path, &st) == 0 && st.st_size > 5 * 1024 * 1024) {
        char *rot = mfmt("%s.1", G.log_path);
        rename(G.log_path, rot);
        free(rot);
    }
    int fd = open(G.log_path, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0644);
    if (fd >= 0) {
        (void)!write(fd, line.ptr, line.len);
        close(fd);
    }
    buf_free(&text);
    buf_free(&line);
    arena_reset(&t);
}

// ---------------------------------------------------------------------------------------
// the client (client.ts) and the stdio transport (transports/stdio.ts, lines.ts)
// ---------------------------------------------------------------------------------------

enum { MAX_MESSAGE_BYTES = 16 * 1024 * 1024, MAX_STDERR_BYTES = 64 * 1024, STDERR_TAIL_CHARS = 2000 };
static const char *const SUPPORTED_VERSIONS[] = {"2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"};

static void client_ref_l(Client *c) { c->refs++; }

static void client_free(Client *c) {
    if (c->in_fd >= 0) close(c->in_fd);
    if (c->wake_r >= 0) close(c->wake_r);
    if (c->wake_w >= 0) close(c->wake_w);
    buf_free(&c->outq);
    buf_free(&c->errtail);
    free(c->server);
    free(c->cwd);
    free(c->url);
    for (size_t i = 0; i < c->nhdr; i++) {
        free(c->hdr_names[i]);
        free(c->hdr_values[i]);
    }
    free(c->hdr_names);
    free(c->hdr_values);
    free(c->session_id);
    free(c->protocol_version);
    free(c->instructions);
    free(c);
}

static void client_unref_l(Client *c) {
    if (--c->refs == 0) client_free(c);
}

static void client_unref(Client *c) {
    pthread_mutex_lock(&g_mu);
    client_unref_l(c);
    pthread_mutex_unlock(&g_mu);
}

static void fail_pending_l(Client *c, const char *msg) {
    for (Pending *p = c->pending; p; p = p->next) {
        p->done = true;
        if (!p->err && !p->result) p->err = strdup(msg);
    }
    c->pending = NULL;
    pthread_cond_broadcast(&g_cv);
}

static void unlink_pending_l(Client *c, Pending *p) {
    for (Pending **pp = &c->pending; *pp; pp = &(*pp)->next)
        if (*pp == p) {
            *pp = p->next;
            return;
        }
}

static char *http_send(Client *c, const char *json, bool is_request, int64_t id);
static void http_send_detached(Client *c, const char *json, double idle_s);

enum { MAX_OUTQ_BYTES = 64 * 1024 * 1024 };

// The queued bytes written as far as stdin takes them now (g_mu held): never blocks (stdin is
// O_NONBLOCK). A write error drops the queue (the reader sees the server go).
static void outq_flush_l(Client *c) {
    while (c->outq_off < c->outq.len && c->in_fd >= 0) {
        ssize_t w = write(c->in_fd, c->outq.ptr + c->outq_off, c->outq.len - c->outq_off);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) {
            if (w < 0 && errno != EAGAIN) c->outq_off = c->outq.len;
            break;
        }
        c->outq_off += (size_t)w;
    }
    if (c->outq_off == c->outq.len) c->outq.len = c->outq_off = 0;
}

// transport.send: one line on the server's stdin (or one POST). Returns NULL or the error. A
// stdio line is queued and written without blocking: what the pipe does not take now the reader
// thread writes when it can, so neither the agent nor the reader (answering a ping) ever waits on
// a server that does not read; the request's own timeout and abort still apply.
static char *client_send(Client *c, const char *json, bool is_request, int64_t id) {
    if (c->http) return http_send(c, json, is_request, id);
    char *err = NULL;
    size_t n = strlen(json);
    pthread_mutex_lock(&g_mu);
    if (c->in_fd < 0 || c->closing) err = strdup("MCP connection closed");
    else if (c->outq.len - c->outq_off + n + 1 > MAX_OUTQ_BYTES) err = strdup("MCP server is not reading its input");
    else {
        buf_add(&c->outq, json, n);
        buf_addc(&c->outq, '\n');
        outq_flush_l(c);
        if (c->outq.len && c->wake_w >= 0) (void)!write(c->wake_w, "", 1);
    }
    pthread_mutex_unlock(&g_mu);
    return err;
}

static void send_cancelled(Client *c, int64_t id, const char *reason) {
    Arena t = {0};
    JValue *m = json_new(&t, J_OBJECT);
    jput(&t, m, "jsonrpc", jstr(&t, "2.0"));
    jput(&t, m, "method", jstr(&t, "notifications/cancelled"));
    JValue *p = json_new(&t, J_OBJECT);
    jput(&t, p, "requestId", json_int(&t, id));
    if (reason && *reason) jput(&t, p, "reason", jstr(&t, reason));
    jput(&t, m, "params", p);
    char *s = js_text(m);
    // never waits: stdio queues it, HTTP posts it in the background with a 2 s idle timeout
    if (c->http) http_send_detached(c, s, 2);
    else free(client_send(c, s, false, -1));
    free(s);
    arena_reset(&t);
}

static void conn_client_closed(Client *c);
static void conn_tools_changed(Client *c);

// handleMessage: a response settles its request; a request from the server is answered (ping,
// roots/list); notifications: progress, server log messages, the tool list changed.
static void client_message(Client *c, Str line) {
    Arena t = {0};
    JValue *m = js_parse(&t, code_utf8_clean(&t, line));
    if (!is_record(m) || !str_eqz(json_strs(json_get(m, "jsonrpc")), "2.0")) goto done;
    const JValue *id = json_get(m, "id"), *method = json_get(m, "method");
    double idn = 0;
    bool id_num = id && jnum(id, &idn) && isfinite(idn);
    bool id_ok = id_num || json_is(id, J_STRING);
    const JValue *result = json_get(m, "result"), *error = json_get(m, "error");
    bool response = id_ok && ((result && !error) ||
                              (!result && is_record(error) && jnum(json_get(error, "code"), &(double){0}) && json_is(json_get(error, "message"), J_STRING)));
    if (response) {
        pthread_mutex_lock(&g_mu);
        Pending *p = NULL;
        if (id_num)
            for (Pending *q = c->pending; q; q = q->next)
                if ((double)q->id == idn) p = q;
        if (p) {
            unlink_pending_l(c, p);
            p->done = true;
            if (error) {
                double code = 0;
                jnum(json_get(error, "code"), &code);
                p->code = (int)code;
                p->err = strdup(json_str(json_get(error, "message")));
            } else p->result = js_text(result);
            pthread_cond_broadcast(&g_cv);
        }
        pthread_mutex_unlock(&g_mu);
        goto done;
    }
    if (id_ok && json_is(method, J_STRING)) {
        // a request from the server
        JValue *r = json_new(&t, J_OBJECT);
        jput(&t, r, "jsonrpc", jstr(&t, "2.0"));
        jput(&t, r, "id", (JValue *)id);
        if (str_eqz(method->s, "ping")) jput(&t, r, "result", json_new(&t, J_OBJECT));
        else if (str_eqz(method->s, "roots/list")) {
            JValue *res = json_new(&t, J_OBJECT), *roots = json_new(&t, J_ARRAY), *root = json_new(&t, J_OBJECT);
            char *u = file_url(c->cwd);
            jput(&t, root, "uri", jstr(&t, u));
            free(u);
            const char *base = strrchr(c->cwd, '/');
            jput(&t, root, "name", jstr(&t, base && base[1] ? base + 1 : c->cwd));
            json_push(&t, roots, root);
            jput(&t, res, "roots", roots);
            jput(&t, r, "result", res);
        } else {
            JValue *e = json_new(&t, J_OBJECT);
            jput(&t, e, "code", json_int(&t, -32601));
            jput(&t, e, "message", json_string(&t, str(arena_printf(&t, "Method not found: %.*s", (int)method->s.len, method->s.ptr))));
            jput(&t, r, "error", e);
        }
        char *s = js_text(r);
        if (c->http) http_send_detached(c, s, 0); // not inside the response stream that asked
        else free(client_send(c, s, false, -1));
        free(s);
        goto done;
    }
    if (!id && json_is(method, J_STRING)) {
        const JValue *params = json_get(m, "params");
        if (str_eqz(method->s, "notifications/progress")) {
            const JValue *tok = json_get(params, "progressToken");
            double tn, pv;
            if (!is_record(params) || !tok || !jnum(tok, &tn) || !jnum(json_get(params, "progress"), &pv)) goto done;
            pthread_mutex_lock(&g_mu);
            for (Pending *q = c->pending; q; q = q->next)
                if (q->wants_progress && (double)q->id == tn) {
                    q->deadline_ns = q->timeout_ms ? now_ns() + q->timeout_ms * 1000000ull : 0; // armTimeout
                    const JValue *msg = json_get(params, "message");
                    char *text;
                    if (json_is(msg, J_STRING)) text = strndup(msg->s.ptr, msg->s.len);
                    else {
                        char *pt = js_text(json_get(params, "progress"));
                        const JValue *total = json_get(params, "total");
                        char *tt = total ? js_text(total) : NULL;
                        text = tt ? mfmt("Progress %s/%s", pt, tt) : mfmt("Progress %s", pt);
                        free(pt);
                        free(tt);
                    }
                    vec_push(&q->progress, text);
                    pthread_cond_broadcast(&g_cv);
                }
            pthread_mutex_unlock(&g_mu);
        } else if (str_eqz(method->s, "notifications/message")) log_write(c->server, params);
        else if (str_eqz(method->s, "notifications/tools/list_changed")) conn_tools_changed(c);
    }
done:
    arena_reset(&t);
}

// JsonRpcLineReader.readStdout
static void client_stdout(Client *c, Buf *line, const char *p, size_t n) {
    for (size_t i = 0; i < n;) {
        const char *nl = memchr(p + i, '\n', n - i);
        size_t end = nl ? (size_t)(nl - p) : n;
        if (line->len + (end - i) > MAX_MESSAGE_BYTES) line->len = MAX_MESSAGE_BYTES + 1; // dropped at its newline
        else buf_add(line, p + i, end - i);
        if (!nl) return;
        i = end + 1;
        if (line->len <= MAX_MESSAGE_BYTES) {
            Str l = buf_str(line);
            if (l.len && l.ptr[l.len - 1] == '\r') l.len--;
            if (str_trim(l).len) client_message(c, l);
        }
        line->len = 0;
    }
}

static void *reader_main(void *arg) {
    Client *c = arg;
    Buf line = {0};
    bool out_open = true, err_open = c->err_fd >= 0;
    char *buf = xmalloc(65536);
    while (out_open || err_open) {
        struct pollfd fds[4];
        int n = 0, io = -1, ie = -1, iw = -1, iq = -1;
        pthread_mutex_lock(&g_mu);
        bool queued = c->outq.len > 0 && c->in_fd >= 0;
        int in_fd = c->in_fd;
        pthread_mutex_unlock(&g_mu);
        if (c->wake_r >= 0) {
            iw = n;
            fds[n++] = (struct pollfd){c->wake_r, POLLIN, 0};
        }
        if (queued) {
            iq = n;
            fds[n++] = (struct pollfd){in_fd, POLLOUT, 0};
        }
        if (out_open) {
            io = n;
            fds[n++] = (struct pollfd){c->out_fd, POLLIN, 0};
        }
        if (err_open) {
            ie = n;
            fds[n++] = (struct pollfd){c->err_fd, POLLIN, 0};
        }
        int r = poll(fds, (nfds_t)n, queued ? 100 : -1);
        if (r < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (iw >= 0 && fds[iw].revents) {
            char drain[64];
            while (read(c->wake_r, drain, sizeof drain) > 0) {}
        }
        if (iq >= 0 || iw >= 0) {
            pthread_mutex_lock(&g_mu);
            outq_flush_l(c);
            pthread_mutex_unlock(&g_mu);
        }
        if (io >= 0 && fds[io].revents) {
            ssize_t got = read(c->out_fd, buf, 65536);
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) out_open = false;
            else client_stdout(c, &line, buf, (size_t)got);
        }
        if (ie >= 0 && fds[ie].revents) {
            ssize_t got = read(c->err_fd, buf, 65536);
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) err_open = false;
            else {
                pthread_mutex_lock(&g_mu);
                buf_add(&c->errtail, buf, (size_t)got);
                if (c->errtail.len > MAX_STDERR_BYTES) {
                    size_t cut = c->errtail.len - MAX_STDERR_BYTES;
                    memmove(c->errtail.ptr, c->errtail.ptr + cut, MAX_STDERR_BYTES);
                    c->errtail.len = MAX_STDERR_BYTES;
                    c->errtail.ptr[c->errtail.len] = 0;
                }
                pthread_mutex_unlock(&g_mu);
            }
        }
    }
    free(buf);
    buf_free(&line);
    close(c->out_fd);
    if (c->err_fd >= 0) close(c->err_fd);
    int st;
    while (waitpid(c->pid, &st, 0) < 0 && errno == EINTR) {}
    pthread_mutex_lock(&g_mu);
    live_del(c->pid);
    c->exited = true;
    c->closed = true;
    c->connected = false;
    fail_pending_l(c, "MCP connection closed");
    pthread_mutex_unlock(&g_mu);
    conn_client_closed(c);
    pthread_mutex_lock(&g_mu);
    pthread_cond_broadcast(&g_cv);
    client_unref_l(c);
    pthread_mutex_unlock(&g_mu);
    return NULL;
}

// The server's stderr, trimmed, at most its last 2000 characters (sliceTail), or NULL.
static char *client_stderr_tail_l(Client *c) {
    if (c->http) return NULL;
    Arena t = {0};
    Str s = slice_tail(str_trim(code_utf8_clean(&t, buf_str(&c->errtail))), STDERR_TAIL_CHARS);
    char *r = s.len ? strndup(s.ptr, s.len) : NULL;
    arena_reset(&t);
    return r;
}

// StdioTransport.start: the server in its own process group, stdin, stdout and stderr piped.
static char *client_spawn(Client *c, char **argv, char **envp, const char *cwd) {
    int in[2] = {-1, -1}, out[2] = {-1, -1}, er[2] = {-1, -1};
    if (pipe(in) || pipe(out) || pipe(er)) {
        int e = errno;
        for (int i = 0; i < 2; i++) {
            if (in[i] >= 0) close(in[i]);
            if (out[i] >= 0) close(out[i]);
            if (er[i] >= 0) close(er[i]);
        }
        return mfmt("spawn %s %s", argv[0], errno_code(e));
    }
    fcntl(in[1], F_SETFD, FD_CLOEXEC);
    fcntl(out[0], F_SETFD, FD_CLOEXEC);
    fcntl(er[0], F_SETFD, FD_CLOEXEC);
    fcntl(in[1], F_SETNOSIGPIPE, 1);
    fcntl(in[1], F_SETFL, fcntl(in[1], F_GETFL) | O_NONBLOCK); // client_send never blocks
    int wk[2];
    if (pipe(wk) == 0) {
        for (int i = 0; i < 2; i++) {
            fcntl(wk[i], F_SETFD, FD_CLOEXEC);
            fcntl(wk[i], F_SETFL, O_NONBLOCK);
        }
        c->wake_r = wk[0];
        c->wake_w = wk[1];
    }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, in[0], 0);
    posix_spawn_file_actions_adddup2(&fa, out[1], 1);
    posix_spawn_file_actions_adddup2(&fa, er[1], 2);
    posix_spawn_file_actions_addchdir(&fa, cwd);
    posix_spawnattr_t at;
    posix_spawnattr_init(&at);
    sigset_t none, all;
    sigemptyset(&none);
    sigfillset(&all);
    posix_spawnattr_setsigmask(&at, &none);
    posix_spawnattr_setsigdefault(&at, &all); // the UI ignores SIGINT: the server must not
    posix_spawnattr_setflags(&at, POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_CLOEXEC_DEFAULT | POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&at, 0);
    pid_t pid = -1;
    // the PATH lookup of the environment the server gets, as node's spawn does
    int rc = posix_spawnp(&pid, argv[0], &fa, &at, argv, envp);
    posix_spawn_file_actions_destroy(&fa);
    posix_spawnattr_destroy(&at);
    close(in[0]);
    close(out[1]);
    close(er[1]);
    if (rc) {
        close(in[1]);
        close(out[0]);
        close(er[0]);
        return mfmt("spawn %s %s", argv[0], errno_code(rc));
    }
    c->pid = pid;
    c->in_fd = in[1];
    c->out_fd = out[0];
    c->err_fd = er[0];
    pthread_mutex_lock(&g_mu);
    live_add(pid);
    client_ref_l(c); // the reader's
    pthread_mutex_unlock(&g_mu);
    spawn_detached(reader_main, c);
    return NULL;
}

// McpClient.close + StdioTransport.close: stdin closed, 0.5 s for the server to exit, then
// SIGTERM to its group, SIGKILL 2 s later; the group gets SIGTERM once it exited (children that
// ignored stdin closing).
static void client_close(Client *c) {
    pthread_mutex_lock(&g_mu);
    if (c->closing) {
        pthread_mutex_unlock(&g_mu);
        return;
    }
    c->closing = true;
    c->closed = true;
    c->connected = false;
    fail_pending_l(c, "MCP connection closed");
    pthread_mutex_unlock(&g_mu);
    if (c->http) {
        // the requests in flight cancelled, then a DELETE ends the session (StreamableHttpTransport.close)
        c->http_cancel = 1;
        pthread_mutex_lock(&g_mu);
        bool session = c->session_id != NULL;
        pthread_mutex_unlock(&g_mu);
        if (session) free(http_send(c, NULL, false, -1));
        return;
    }
    if (c->pid <= 0) return;
    // stdin closed at once (no write ever holds it: client_send does not block), what was still
    // queued dropped
    pthread_mutex_lock(&g_mu);
    if (c->in_fd >= 0) close(c->in_fd);
    c->in_fd = -1;
    c->outq.len = c->outq_off = 0;
    if (c->wake_w >= 0) (void)!write(c->wake_w, "", 1);
    uint64_t t0 = now_ns();
    int stage = 0;
    while (!c->exited) {
        uint64_t el = now_ns() - t0;
        if (stage == 0 && el >= 500000000ull) {
            kill(-c->pid, SIGTERM);
            stage = 1;
        } else if (stage == 1 && el >= 2500000000ull) {
            kill(-c->pid, SIGKILL);
            stage = 2;
        } else if (stage == 2 && el >= 3500000000ull) break; // a process that outlives SIGKILL: given up
        timed_wait_l(now_ns() + 50000000ull);
    }
    bool exited = c->exited;
    pthread_mutex_unlock(&g_mu);
    if (exited) kill(-c->pid, SIGTERM);
}

// requestInternal: sends the request and waits for its answer, the timeout (re-armed by
// progress) or the abort. Returns the result's JSON text, or NULL with *err (and *code for an
// error response, else 0).
static char *client_request(Client *c, const char *method, const JValue *params, uint64_t timeout_ms, volatile int *abort,
                            void (*on_progress)(void *, const char *), void *pctx, char **err, int *code, bool allow_connecting) {
    *err = NULL;
    if (code) *code = 0;
    Pending p = {0};
    pthread_mutex_lock(&g_mu);
    if (c->closed || (!c->connected && !allow_connecting)) {
        pthread_mutex_unlock(&g_mu);
        *err = strdup(c->closed ? "MCP client is closed" : "MCP client is connecting");
        return NULL;
    }
    if (abort && *abort) {
        pthread_mutex_unlock(&g_mu);
        *err = strdup("MCP request aborted");
        return NULL;
    }
    p.id = c->next_id++;
    p.timeout_ms = timeout_ms;
    p.deadline_ns = timeout_ms ? now_ns() + timeout_ms * 1000000ull : 0;
    p.cancellable = strcmp(method, "initialize") != 0;
    p.wants_progress = on_progress != NULL;
    p.next = c->pending;
    c->pending = &p;
    pthread_mutex_unlock(&g_mu);
    Arena t = {0};
    JValue *m = json_new(&t, J_OBJECT);
    jput(&t, m, "jsonrpc", jstr(&t, "2.0"));
    jput(&t, m, "id", json_int(&t, p.id));
    jput(&t, m, "method", jstr(&t, method));
    JValue *rp = (JValue *)params;
    if (on_progress) {
        rp = obj_spread(&t, params);
        JValue *meta = obj_spread(&t, is_record(json_get(params, "_meta")) ? json_get(params, "_meta") : NULL);
        jput(&t, meta, "progressToken", json_int(&t, p.id));
        jput(&t, rp, "_meta", meta);
    }
    if (rp) jput(&t, m, "params", rp);
    char *line = js_text(m);
    arena_reset(&t);
    char *serr = client_send(c, line, true, p.id);
    free(line);
    pthread_mutex_lock(&g_mu);
    if (serr && !p.done) {
        unlink_pending_l(c, &p);
        p.done = true;
        p.err = serr;
        serr = NULL;
    }
    free(serr);
    for (;;) {
        // progress first: the server sent it before its answer
        if (p.progress.len) {
            char *tx = p.progress.ptr[0];
            memmove(p.progress.ptr, p.progress.ptr + 1, (p.progress.len - 1) * sizeof(char *));
            p.progress.len--;
            pthread_mutex_unlock(&g_mu);
            on_progress(pctx, tx);
            free(tx);
            pthread_mutex_lock(&g_mu);
            continue;
        }
        if (p.done) break;
        uint64_t now = now_ns();
        if (abort && *abort) {
            unlink_pending_l(c, &p);
            pthread_mutex_unlock(&g_mu);
            if (p.cancellable) send_cancelled(c, p.id, "AbortError: This operation was aborted");
            vec_free(&p.progress);
            *err = strdup("MCP request aborted");
            return NULL;
        }
        if (p.deadline_ns && now >= p.deadline_ns) {
            unlink_pending_l(c, &p);
            pthread_mutex_unlock(&g_mu);
            if (p.cancellable) send_cancelled(c, p.id, "Request timed out");
            vec_free(&p.progress);
            *err = mfmt("MCP request timed out after %llums", (unsigned long long)p.timeout_ms);
            return NULL;
        }
        uint64_t until = now + 50000000ull;
        if (p.deadline_ns && p.deadline_ns < until) until = p.deadline_ns;
        timed_wait_l(until);
    }
    pthread_mutex_unlock(&g_mu);
    vec_free(&p.progress);
    if (p.err) {
        *err = p.err;
        if (code) *code = p.code;
        free(p.result);
        return NULL;
    }
    return p.result;
}

static char *client_notify(Client *c, const char *method) {
    char *line = mfmt("{\"jsonrpc\":\"2.0\",\"method\":\"%s\"}", method);
    char *e = client_send(c, line, false, -1);
    free(line);
    return e;
}

// validateListPage + listAll: every item of a paginated list (JSON text of the array), or NULL
static char *client_list_all(Client *c, const char *method, const char *key, uint64_t timeout_ms, volatile int *abort, char **err,
                             int *code) {
    Arena t = {0};
    JValue *items = json_new(&t, J_ARRAY);
    VEC(char *) cursors = {0};
    char *cursor = NULL;
    char *out = NULL;
    for (int page = 0; page < 1000; page++) {
        JValue *params = NULL;
        if (cursor) {
            params = json_new(&t, J_OBJECT);
            jput(&t, params, "cursor", jstr(&t, cursor));
        }
        char *res = client_request(c, method, params, timeout_ms, abort, NULL, NULL, err, code, false);
        if (!res) goto end;
        JValue *v = js_parse(&t, str(res));
        free(res);
        const JValue *arr = json_get(v, key);
        if (!is_record(v) || !json_is(arr, J_ARRAY)) {
            *err = mfmt("Invalid MCP %s result", method);
            goto end;
        }
        for (size_t i = 0; i < arr->arr.len; i++) {
            const JValue *it = arr->arr.items[i];
            bool ok = is_record(it);
            if (ok && !strcmp(key, "tools")) ok = json_is(json_get(it, "name"), J_STRING) && is_record(json_get(it, "inputSchema"));
            else if (ok && !strcmp(key, "resources"))
                ok = json_is(json_get(it, "uri"), J_STRING) && (!json_get(it, "name") || json_is(json_get(it, "name"), J_STRING));
            else if (ok)
                ok = json_is(json_get(it, "uriTemplate"), J_STRING) && (!json_get(it, "name") || json_is(json_get(it, "name"), J_STRING));
            if (!ok) {
                *err = mfmt("Invalid entry in MCP %s result", method);
                goto end;
            }
            json_push(&t, items, arr->arr.items[i]);
        }
        const JValue *nc = json_get(v, "nextCursor");
        if (!nc || json_is(nc, J_NULL) || (json_is(nc, J_STRING) && !nc->s.len)) {
            out = js_text(items);
            goto end;
        }
        if (!json_is(nc, J_STRING)) {
            *err = mfmt("Invalid MCP %s cursor", method);
            goto end;
        }
        for (size_t k = 0; k < cursors.len; k++)
            if (!strcmp(cursors.ptr[k], nc->s.ptr)) {
                *err = mfmt("MCP %s returned duplicate cursor: %s", method, nc->s.ptr);
                goto end;
            }
        cursor = strdup(nc->s.ptr);
        vec_push(&cursors, cursor);
    }
    *err = mfmt("MCP %s exceeded 1000 pages", method);
end:
    for (size_t k = 0; k < cursors.len; k++) free(cursors.ptr[k]);
    vec_free(&cursors);
    arena_reset(&t);
    return out;
}

// ---------------------------------------------------------------------------------------
// the Streamable HTTP transport (transports/streamable-http.ts): each message POSTed, the answer
// a JSON body or an SSE stream; Mcp-Session-Id and MCP-Protocol-Version on every request after
// initialize; a DELETE ends the session
// ---------------------------------------------------------------------------------------

// error kinds (in a request's *code, outside JSON-RPC's codes): McpAuthRequiredError,
// McpSessionExpiredError, and the network failures and statuses worth another attempt
enum { ERR_AUTH = -1, ERR_SESSION_EXPIRED = -2, ERR_TRANSIENT = -3 };

typedef struct {
    Client *c;
    char *json; // the message, NULL for the session's DELETE
    bool is_request;
    int64_t id;
    double idle_s; // the exchange's idle timeout, 0 for the default (300 s, the DELETE 1 s)
    char ctype[128]; // the response's media type, lowercased
    char *session;
    Buf body; // a JSON or an error body
    Buf sse, data;
    char event[64];
    bool has_event;
    bool too_big, captured;
} HttpX;

static void http_header(void *ctx, const char *name, const char *value) {
    HttpX *h = ctx;
    if (!strcasecmp(name, "content-type")) {
        size_t n = strcspn(value, ";");
        Str t = str_trim(str_n(value, n));
        size_t k = MIN(t.len, sizeof h->ctype - 1);
        for (size_t i = 0; i < k; i++) h->ctype[i] = (char)((t.ptr[i] >= 'A' && t.ptr[i] <= 'Z') ? t.ptr[i] + 32 : t.ptr[i]);
        h->ctype[k] = 0;
    } else if (!strcasecmp(name, "mcp-session-id") && *value) set_str(&h->session, strdup(value));
}

// consumeSse's dispatch: an event's data (lines joined) handed over as one message, unless the
// event is not a "message" or has no data
static void sse_dispatch(HttpX *h) {
    if (h->data.len) {
        Str d = buf_str(&h->data);
        d.len--; // the last join
        if (str_trim(d).len && (!h->has_event || !strcmp(h->event, "message"))) client_message(h->c, d);
    }
    h->data.len = 0;
    h->has_event = false;
}

static void sse_line(HttpX *h, Str line) {
    if (!line.len) {
        sse_dispatch(h);
        return;
    }
    if (line.ptr[0] == ':') return;
    ptrdiff_t colon = str_find_char(line, ':', 0);
    Str field = colon >= 0 ? str_sub(line, 0, (size_t)colon) : line;
    Str value = colon >= 0 ? str_sub(line, (size_t)colon + 1, line.len) : S("");
    if (value.len && value.ptr[0] == ' ') value = str_sub(value, 1, value.len);
    if (str_eqz(field, "data")) {
        if (h->data.len + value.len > MAX_MESSAGE_BYTES) {
            h->too_big = true;
            return;
        }
        buf_addstr(&h->data, value);
        buf_addc(&h->data, '\n');
    } else if (str_eqz(field, "event")) {
        snprintf(h->event, sizeof h->event, "%.*s", (int)value.len, value.ptr);
        h->has_event = true;
    }
}

// the stream's lines: CRLF, LF or a lone CR end one
static void sse_feed(HttpX *h, const char *p, size_t n, bool end) {
    buf_add(&h->sse, p, n);
    Str s = buf_str(&h->sse);
    size_t start = 0;
    for (size_t i = 0; i < s.len; i++) {
        char ch = s.ptr[i];
        if (ch != '\r' && ch != '\n') continue;
        if (ch == '\r' && i + 1 >= s.len && !end) break; // its LF may follow in the next read
        sse_line(h, str_sub(s, start, i));
        if (ch == '\r' && i + 1 < s.len && s.ptr[i + 1] == '\n') i++;
        start = i + 1;
    }
    if (end) {
        if (start < s.len) sse_line(h, str_sub(s, start, s.len));
        sse_dispatch(h);
        start = s.len;
    }
    if (start) {
        memmove(h->sse.ptr, h->sse.ptr + start, h->sse.len - start);
        h->sse.len -= start;
        if (h->sse.ptr) h->sse.ptr[h->sse.len] = 0;
    }
}

static int http_data(void *ctx, int status, const void *p, size_t n) {
    HttpX *h = ctx;
    bool ok = status >= 200 && status < 300;
    if (ok && h->session && !h->captured) {
        // captureSession, before the stream's messages (the client goes on as its answer arrives)
        h->captured = true;
        pthread_mutex_lock(&g_mu);
        set_str(&h->c->session_id, strdup(h->session));
        pthread_mutex_unlock(&g_mu);
    }
    if (ok && h->is_request && !strcmp(h->ctype, "text/event-stream")) sse_feed(h, p, n, false);
    else if (h->body.len < (ok ? (size_t)MAX_MESSAGE_BYTES : 8 * 1024)) buf_add(&h->body, p, n);
    else if (ok) h->too_big = true;
    return h->c->http_cancel ? 1 : 0;
}

// describeHttpFailure
static char *http_failure(int status, Str body) {
    Str t = str_trim(body);
    if (u16_len(t) > 500) return mfmt("MCP HTTP request failed with status %d: %.*s...", status, (int)slice_head(t, 497).len, t.ptr);
    return t.len ? mfmt("MCP HTTP request failed with status %d: %.*s", status, (int)t.len, t.ptr) : mfmt("MCP HTTP request failed with status %d", status);
}

// StreamableHttpTransport.send (and close's DELETE): one exchange. NULL or the error, its kind in
// *kind.
static char *http_exchange(HttpX *h, int *kind) {
    Client *c = h->c;
    *kind = 0;
    VEC(sx_header) hs = {0};
    for (size_t i = 0; i < c->nhdr; i++) vec_push(&hs, ((sx_header){c->hdr_names[i], c->hdr_values[i]}));
    if (h->json) {
        vec_push(&hs, ((sx_header){"accept", "application/json, text/event-stream"}));
        vec_push(&hs, ((sx_header){"content-type", "application/json"}));
    }
    pthread_mutex_lock(&g_mu);
    char *sid = dupz(c->session_id), *pv = dupz(c->protocol_version);
    pthread_mutex_unlock(&g_mu);
    if (sid) vec_push(&hs, ((sx_header){"Mcp-Session-Id", sid}));
    if (pv) vec_push(&hs, ((sx_header){"MCP-Protocol-Version", pv}));
    int status = 0;
    char err[512] = {0};
    volatile int never = 0;
    int rc = sx_http_ex(h->json ? "POST" : "DELETE", c->url, hs.ptr, hs.len, h->json, h->json ? strlen(h->json) : 0,
                        h->idle_s > 0 ? h->idle_s : h->json ? 300 : 1,
                        http_header, http_data, h, &status, err, sizeof err, h->json ? &c->http_cancel : &never);
    vec_free(&hs);
    char *e = NULL;
    if (rc) {
        // fetch's TypeError (a network failure), or the transport closed meanwhile
        if (c->http_cancel) e = strdup("MCP connection closed");
        else {
            e = strdup("fetch failed");
            *kind = ERR_TRANSIENT;
        }
        goto out;
    }
    if (!h->json) goto out;
    if (status < 200 || status >= 300) {
        // checkResponse
        if (status == 401) {
            e = strdup("MCP server requires authentication");
            *kind = ERR_AUTH;
        } else if (status == 404 && sid) {
            e = strdup("MCP session expired");
            *kind = ERR_SESSION_EXPIRED;
        } else {
            e = http_failure(status, buf_str(&h->body));
            if (status == 408 || status == 429 || (status >= 500 && status != 501)) *kind = ERR_TRANSIENT;
        }
        goto out;
    }
    if (h->session) {
        // captureSession
        pthread_mutex_lock(&g_mu);
        set_str(&c->session_id, strdup(h->session));
        pthread_mutex_unlock(&g_mu);
    }
    if (!h->is_request) goto out; // notifications and responses: 202, no reply
    if (status == 202 || status == 204) {
        Arena t = {0};
        JValue *m = js_parse(&t, str(h->json));
        e = mfmt("MCP server accepted request %s without a response", json_str(json_get(m, "method")));
        arena_reset(&t);
        goto out;
    }
    if (!strcmp(h->ctype, "application/json")) {
        if (h->too_big) {
            e = strdup("MCP JSON response exceeds 16777216 bytes");
            goto out;
        }
        Arena t = {0};
        char *perr = NULL;
        JValue *body = code_oai_js_parse(&t, code_utf8_clean(&t, buf_str(&h->body)), &perr);
        if (!body) e = strdup(perr);
        else if (json_is(body, J_ARRAY))
            for (size_t i = 0; i < body->arr.len; i++) {
                char *one = js_text(body->arr.items[i]);
                client_message(c, str(one));
                free(one);
            }
        else client_message(c, buf_str(&h->body));
        arena_reset(&t);
    } else if (!strcmp(h->ctype, "text/event-stream")) {
        sse_feed(h, "", 0, true);
        // consumeResponseStream: a stream that ends before the answer fails its request (no
        // resumption: Saetta Code does not reopen streams)
        e = strdup(h->too_big ? "MCP response stream failed: MCP SSE event exceeds 16777216 bytes"
                              : "MCP response stream failed: stream ended without a response");
    } else e = mfmt("Unsupported MCP response content type: %s", h->ctype[0] ? h->ctype : "missing");
out:
    free(sid);
    free(pv);
    return e;
}

static void http_free(HttpX *h) {
    free(h->json);
    free(h->session);
    buf_free(&h->body);
    buf_free(&h->sse);
    buf_free(&h->data);
    free(h);
}

// a failed request's error, unless its answer came (cancelPending with the send's error)
static void fail_request(Client *c, int64_t id, const char *msg, int kind) {
    pthread_mutex_lock(&g_mu);
    for (Pending *p = c->pending; p; p = p->next)
        if (p->id == id) {
            unlink_pending_l(c, p);
            p->done = true;
            p->err = strdup(msg);
            p->code = kind;
            pthread_cond_broadcast(&g_cv);
            break;
        }
    pthread_mutex_unlock(&g_mu);
}

static void *http_request_main(void *arg) {
    HttpX *h = arg;
    int kind = 0;
    char *e = http_exchange(h, &kind);
    // the answer went through client_message; an error (or a stream without the answer) fails it
    if (e) fail_request(h->c, h->id, e, kind);
    free(e);
    client_unref(h->c);
    http_free(h);
    return NULL;
}

static void *http_detached_main(void *arg) {
    HttpX *h = arg;
    int kind = 0;
    free(http_exchange(h, &kind));
    client_unref(h->c);
    http_free(h);
    return NULL;
}

// A notification or a response POSTed on a thread of its own, its error dropped: the cancellation
// of an aborted or timed-out request (at most idle_s of silence) and the answers to the server's
// requests, so neither the agent nor a response stream waits on the POST.
static void http_send_detached(Client *c, const char *json, double idle_s) {
    HttpX *h = xcalloc(1, sizeof *h);
    h->c = c;
    h->json = dupz(json);
    h->id = -1;
    h->idle_s = idle_s;
    pthread_mutex_lock(&g_mu);
    client_ref_l(c);
    pthread_mutex_unlock(&g_mu);
    spawn_detached(http_detached_main, h);
}

// A request's POST runs on its own thread (its answer may stream for long; the caller waits with
// its timeout); notifications, responses and the DELETE are sent at once.
static char *http_send(Client *c, const char *json, bool is_request, int64_t id) {
    HttpX *h = xcalloc(1, sizeof *h);
    h->c = c;
    h->json = dupz(json);
    h->is_request = is_request;
    h->id = id;
    if (is_request) {
        pthread_mutex_lock(&g_mu);
        client_ref_l(c);
        pthread_mutex_unlock(&g_mu);
        spawn_detached(http_request_main, h);
        return NULL;
    }
    int kind = 0;
    char *e = http_exchange(h, &kind);
    http_free(h);
    return e;
}

// ---------------------------------------------------------------------------------------
// the connection (runtime.ts McpServerConnection)
// ---------------------------------------------------------------------------------------

static void register_tools_l(Conn *conn);
static void sync_resource_tools_l(void);

// The server's state changed: the manager shows it at once.
static void conn_changed_l(Conn *c) {
    (void)c;
    changed_l();
}

// handleClientClose: the transport dropped; the next call reconnects
static void conn_client_closed(Client *cl) {
    pthread_mutex_lock(&g_mu);
    Conn *c = cl->conn;
    if (c && c->client == cl && !c->shut) {
        c->client = NULL;
        c->state = ST_DISCONNECTED;
        char *tail = client_stderr_tail_l(cl);
        set_str(&c->error, tail ? mfmt("Connection closed\n%s", tail) : strdup("Connection closed"));
        free(tail);
        conn_changed_l(c);
        client_unref_l(cl); // the connection's
    }
    pthread_mutex_unlock(&g_mu);
}

typedef struct {
    Conn *conn;
    Client *client;
} Refresh;

static void *refresh_main(void *arg) {
    Refresh *r = arg;
    char *err = NULL;
    int code = 0;
    char *tools = client_list_all(r->client, "tools/list", "tools", timeout_ms_of(r->conn->config), NULL, &err, &code);
    pthread_mutex_lock(&g_mu);
    Conn *c = r->conn;
    if (tools && c->client == r->client && !c->shut) {
        Arena t = {0};
        JValue *arr = js_parse(&t, str(tools));
        for (size_t i = 0; i < c->tools.len; i++) {
            free(c->tools.ptr[i].name);
            free(c->tools.ptr[i].json);
        }
        c->tools.len = 0;
        for (size_t i = 0; json_is(arr, J_ARRAY) && i < arr->arr.len; i++)
            vec_push(&c->tools, ((ToolRec){strdup(json_str(json_get(arr->arr.items[i], "name"))), js_text(arr->arr.items[i])}));
        arena_reset(&t);
        register_tools_l(c);
    } else if (!tools) set_str(&c->error, mfmt("Failed to refresh tools: %s", err));
    conn_changed_l(c);
    client_unref_l(r->client);
    pthread_mutex_unlock(&g_mu);
    free(tools);
    free(err);
    free(r);
    bg_end();
    return NULL;
}

// notifications/tools/list_changed: the tools listed again
static void conn_tools_changed(Client *cl) {
    pthread_mutex_lock(&g_mu);
    Conn *c = cl->conn;
    if (c && !G.shutdown) {
        Refresh *r = xcalloc(1, sizeof *r);
        r->conn = c;
        r->client = cl;
        client_ref_l(cl);
        bg_begin_l();
        spawn_detached(refresh_main, r);
    }
    pthread_mutex_unlock(&g_mu);
}

// createDefaultTransport + connectOnce: a connected client, or NULL with *err (malloc'd)
static Client *connect_once(Conn *conn, char **err, int *kind) {
    *err = NULL;
    *kind = 0;
    Arena t = {0};
    pthread_mutex_lock(&g_mu);
    JValue *config = jcopy(&t, conn->config);
    char *cwd = strdup(G.cwd);
    pthread_mutex_unlock(&g_mu);
    uint64_t timeout = timeout_ms_of(config);
    Client *c = xcalloc(1, sizeof *c);
    c->refs = 1; // ours
    c->server = strdup(conn->name);
    c->cwd = cwd;
    c->in_fd = c->out_fd = c->err_fd = -1;
    c->wake_r = c->wake_w = -1;
    c->next_id = 1;
    c->conn = conn;
    pthread_mutex_lock(&g_mu);
    if (conn->shut) c->closed = true;
    conn->connecting = c;
    client_ref_l(c); // the connection's, while it connects
    pthread_mutex_unlock(&g_mu);
    char *e = NULL;
    if (is_http(config)) {
        c->http = true;
        c->url = strdup(json_str(json_get(config, "url")));
        const JValue *hd = json_get(config, "headers");
        size_t n = is_record(hd) ? hd->obj.len : 0;
        c->hdr_names = xcalloc(n + 1, sizeof(char *));
        c->hdr_values = xcalloc(n + 1, sizeof(char *));
        for (size_t i = 0; i < n && !e; i++) {
            char *desc = mfmt("MCP server \"%s\" header \"%.*s\"", conn->name, (int)hd->obj.items[i].key.len, hd->obj.items[i].key.ptr);
            char *v = resolve_config_value(json_str(hd->obj.items[i].val), desc, &e);
            free(desc);
            if (v) {
                c->hdr_names[c->nhdr] = strndup(hd->obj.items[i].key.ptr, hd->obj.items[i].key.len);
                c->hdr_values[c->nhdr++] = v;
            }
        }
    } else {
        // the environment: the process's, with the config's values (resolved) over it
        VEC(char *) envv = {0};
        const JValue *env = json_get(config, "env");
        VEC(char *) extra = {0};
        for (size_t i = 0; is_record(env) && i < env->obj.len && !e; i++) {
            Str k = env->obj.items[i].key;
            char *desc = mfmt("MCP server \"%s\" env \"%.*s\"", conn->name, (int)k.len, k.ptr);
            char *v = resolve_config_value(json_str(env->obj.items[i].val), desc, &e);
            free(desc);
            if (v) {
                vec_push(&extra, mfmt("%.*s=%s", (int)k.len, k.ptr, v));
                free(v);
            }
        }
        if (!e) {
            for (char **p = environ; *p; p++) {
                size_t kl = strcspn(*p, "=");
                bool over = false;
                for (size_t i = 0; i < extra.len && !over; i++) over = !strncmp(extra.ptr[i], *p, kl) && extra.ptr[i][kl] == '=';
                if (!over) vec_push(&envv, *p);
            }
            for (size_t i = 0; i < extra.len; i++) vec_push(&envv, extra.ptr[i]);
            vec_push(&envv, NULL);
            VEC(char *) argv = {0};
            vec_push(&argv, expand_home(json_str(json_get(config, "command"))));
            const JValue *args = json_get(config, "args");
            for (size_t i = 0; json_is(args, J_ARRAY) && i < args->arr.len; i++) vec_push(&argv, expand_home(json_str(args->arr.items[i])));
            vec_push(&argv, NULL);
            const char *cw = json_str(json_get(config, "cwd"));
            char *ecw = expand_home(cw ? cw : ".");
            char *dir = path_resolve2(c->cwd, ecw);
            free(ecw);
            e = client_spawn(c, argv.ptr, envv.ptr, dir);
            free(dir);
            for (size_t i = 0; i + 1 < argv.len; i++) free(argv.ptr[i]);
            vec_free(&argv);
        }
        for (size_t i = 0; i < extra.len; i++) free(extra.ptr[i]);
        vec_free(&extra);
        vec_free(&envv);
    }
    if (!e) {
        // initialize (client.ts connect)
        JValue *params = json_new(&t, J_OBJECT);
        jput(&t, params, "protocolVersion", jstr(&t, SUPPORTED_VERSIONS[0]));
        JValue *caps = json_new(&t, J_OBJECT);
        jput(&t, caps, "roots", json_new(&t, J_OBJECT));
        jput(&t, params, "capabilities", caps);
        JValue *info = json_new(&t, J_OBJECT);
        jput(&t, info, "name", jstr(&t, "saetta-code"));
        jput(&t, info, "version", jstr(&t, code_version));
        jput(&t, params, "clientInfo", info);
        int code = 0;
        char *res = client_request(c, "initialize", params, timeout, NULL, NULL, NULL, &e, &code, true);
        *kind = code;
        if (res) {
            JValue *v = js_parse(&t, str(res));
            free(res);
            const JValue *si = json_get(v, "serverInfo"), *ins = json_get(v, "instructions");
            if (!is_record(v) || !json_is(json_get(v, "protocolVersion"), J_STRING) || !is_record(json_get(v, "capabilities")) ||
                !is_record(si) || !json_is(json_get(si, "name"), J_STRING) || !json_is(json_get(si, "version"), J_STRING) ||
                (ins && !json_is(ins, J_STRING)))
                e = strdup("Invalid MCP initialize result");
            else {
                const char *pv = json_str(json_get(v, "protocolVersion"));
                bool ok = false;
                for (size_t i = 0; i < COUNT(SUPPORTED_VERSIONS); i++) ok |= !strcmp(pv, SUPPORTED_VERSIONS[i]);
                if (!ok) e = mfmt("MCP server selected unsupported protocol version %s", pv);
                else {
                    const JValue *caps2 = json_get(v, "capabilities");
                    c->cap_tools = json_get(caps2, "tools") != NULL;
                    c->cap_resources = json_get(caps2, "resources") != NULL;
                    Str is = trim_js(json_strs(ins));
                    c->instructions = is.len ? strndup(is.ptr, is.len) : NULL;
                    c->protocol_version = strdup(pv);
                    e = client_notify(c, "notifications/initialized");
                    if (!e) {
                        pthread_mutex_lock(&g_mu);
                        c->connected = !c->closed;
                        pthread_mutex_unlock(&g_mu);
                    }
                }
            }
        }
    }
    char *tools = NULL;
    if (!e && c->cap_tools) {
        int code = 0;
        tools = client_list_all(c, "tools/list", "tools", timeout, NULL, &e, &code);
        *kind = code;
    }
    pthread_mutex_lock(&g_mu);
    if (conn->connecting == c) {
        conn->connecting = NULL;
        client_unref_l(c);
    }
    if (!e && conn->shut) e = strdup("shut down while connecting");
    if (!e && !c->connected) e = strdup("connection closed during setup");
    if (!e) {
        conn->client = c; // takes our reference
        for (size_t i = 0; i < conn->tools.len; i++) {
            free(conn->tools.ptr[i].name);
            free(conn->tools.ptr[i].json);
        }
        conn->tools.len = 0;
        JValue *arr = tools ? js_parse(&t, str(tools)) : NULL;
        for (size_t i = 0; json_is(arr, J_ARRAY) && i < arr->arr.len; i++)
            vec_push(&conn->tools, ((ToolRec){strdup(json_str(json_get(arr->arr.items[i], "name"))), js_text(arr->arr.items[i])}));
        conn->has_resources = c->cap_resources;
        set_str(&conn->instructions, dupz(c->instructions));
        conn->state = ST_CONNECTED;
        set_str(&conn->error, NULL);
        register_tools_l(conn);
        conn_changed_l(conn);
        pthread_mutex_unlock(&g_mu);
        free(tools);
        arena_reset(&t);
        return c;
    }
    pthread_mutex_unlock(&g_mu);
    free(tools);
    client_close(c);
    pthread_mutex_lock(&g_mu);
    set_str(&conn->stderr_tail, client_stderr_tail_l(c));
    c->conn = NULL;
    client_unref_l(c);
    pthread_mutex_unlock(&g_mu);
    arena_reset(&t);
    *err = e;
    return NULL;
}

// getClient: the connected client (a reference the caller drops), opening a connection when
// there is none (one at a time: later callers wait for it). NULL with *err.
static bool needs_sign_in(const JValue *config);
static char *sign_in_message(const Conn *c);

static Client *conn_get_client(Conn *c, char **err) {
    *err = NULL;
    pthread_mutex_lock(&g_mu);
    for (;;) {
        if (c->shut) {
            pthread_mutex_unlock(&g_mu);
            *err = mfmt("MCP server \"%s\" is shut down", c->name);
            return NULL;
        }
        if (c->client && c->client->connected) {
            Client *cl = c->client;
            client_ref_l(cl);
            pthread_mutex_unlock(&g_mu);
            return cl;
        }
        if (!c->opening) break;
        uint64_t gen = atomic_load(&G.gen);
        while (c->opening && atomic_load(&G.gen) == gen) timed_wait_l(now_ns() + 100000000ull);
        if (!c->opening) {
            if (c->client && c->client->connected) continue;
            pthread_mutex_unlock(&g_mu);
            *err = dupz(c->open_err ? c->open_err : "MCP connection closed");
            return NULL;
        }
    }
    // open(): connecting, then the attempts (an HTTP server gets two more after a transient error,
    // 250 ms and 1 s later; a stdio server none)
    c->opening = true;
    c->state = ST_CONNECTING;
    conn_changed_l(c);
    bool http = is_http(c->config);
    pthread_mutex_unlock(&g_mu);
    static const int delays[] = {250, 1000};
    char *e = NULL;
    int kind = 0;
    Client *cl = NULL;
    for (int attempt = 0;; attempt++) {
        pthread_mutex_lock(&g_mu);
        set_str(&c->stderr_tail, NULL);
        pthread_mutex_unlock(&g_mu);
        cl = connect_once(c, &e, &kind);
        if (cl || c->shut || !http || attempt >= 2 || kind != ERR_TRANSIENT) break;
        uint64_t until = now_ns() + (uint64_t)delays[attempt] * 1000000ull;
        pthread_mutex_lock(&g_mu);
        while (!c->shut && now_ns() < until) timed_wait_l(until);
        pthread_mutex_unlock(&g_mu);
        if (c->shut) break;
        free(e);
        e = NULL;
    }
    pthread_mutex_lock(&g_mu);
    if (cl) client_ref_l(cl);
    else if (kind == ERR_AUTH && needs_sign_in(c->config) && !c->shut) {
        // markNeedsAuth
        c->state = ST_NEEDS_AUTH;
        set_str(&c->error, NULL);
        set_str(&c->open_err, sign_in_message(c));
        *err = strdup(c->open_err);
        conn_changed_l(c);
    } else {
        // connectFailed
        c->state = c->shut ? ST_CLOSED : ST_FAILED;
        set_str(&c->error, c->stderr_tail ? mfmt("%s\n%s", e, c->stderr_tail) : strdup(e));
        set_str(&c->open_err, mfmt("MCP server \"%s\" failed to connect: %s", c->name, c->error));
        *err = strdup(c->open_err);
        conn_changed_l(c);
    }
    c->opening = false;
    pthread_cond_broadcast(&g_cv);
    changed_l();
    pthread_mutex_unlock(&g_mu);
    free(e);
    return cl;
}

// dropClient
static void conn_drop_client(Conn *c) {
    pthread_mutex_lock(&g_mu);
    Client *cl = c->client;
    c->client = NULL;
    pthread_mutex_unlock(&g_mu);
    if (!cl) return;
    client_close(cl);
    client_unref(cl);
}

static void conn_wait_opening_l(Conn *c) {
    while (c->opening) timed_wait_l(now_ns() + 100000000ull);
}

// reconnect: the client dropped, then a new one; NULL or the error
static char *conn_reconnect(Conn *c) {
    pthread_mutex_lock(&g_mu);
    conn_wait_opening_l(c);
    pthread_mutex_unlock(&g_mu);
    conn_drop_client(c);
    char *err = NULL;
    Client *cl = conn_get_client(c, &err);
    if (cl) client_unref(cl);
    return err;
}

// close(): shut down, no transport left open
static void conn_close(Conn *c) {
    pthread_mutex_lock(&g_mu);
    c->shut = true;
    c->state = ST_CLOSED;
    conn_changed_l(c);
    Client *cl = c->client, *pend = c->connecting;
    c->client = NULL;
    if (pend) client_ref_l(pend);
    pthread_mutex_unlock(&g_mu);
    if (cl) {
        client_close(cl);
        client_unref(cl);
    }
    if (pend) {
        client_close(pend); // fails its initialize or tools/list
        client_unref(pend);
    }
    pthread_mutex_lock(&g_mu);
    conn_wait_opening_l(c);
    pthread_mutex_unlock(&g_mu);
}

// usesOAuth: an HTTP server without an Authorization header or auth; such servers, and the ones
// with auth.provider, need a sign-in when they answer 401
static bool needs_sign_in(const JValue *config) {
    if (!is_http(config)) return false;
    if (json_get(config, "auth")) return true;
    const JValue *h = json_get(config, "headers");
    for (size_t i = 0; is_record(h) && i < h->obj.len; i++)
        if (h->obj.items[i].key.len == 13 && !strncasecmp(h->obj.items[i].key.ptr, "authorization", 13)) return false;
    return true;
}

// signInRequiredMessage
static char *sign_in_message(const Conn *c) {
    const char *provider = json_str(json_get(json_get(c->config, "auth"), "provider"));
    return provider ? mfmt("MCP server \"%s\" requires sign-in. Run /login %s to sign in.", c->name, provider)
                    : mfmt("MCP server \"%s\" requires sign-in. Run /mcp to sign in.", c->name);
}

// withClient: a request through the connection, reconnecting when needed; a session the server
// forgot is retried once on a new one (the old client detached, not closed); a 401 from a server
// that needs a sign-in drops the client and marks it. NULL with *err (and *code).
static char *conn_request(Conn *conn, const char *method, const JValue *params, uint64_t timeout, volatile int *abort,
                          void (*on_progress)(void *, const char *), void *pctx, char **err, int *code) {
    for (int attempt = 1;; attempt++) {
        *code = 0;
        Client *cl = conn_get_client(conn, err);
        if (!cl) return NULL;
        char *res = client_request(cl, method, params, timeout, abort, on_progress, pctx, err, code, false);
        if (!res && *code == ERR_SESSION_EXPIRED && attempt == 1) {
            pthread_mutex_lock(&g_mu);
            if (conn->client == cl) {
                conn->client = NULL;
                client_unref_l(cl);
            }
            client_unref_l(cl);
            pthread_mutex_unlock(&g_mu);
            free(*err);
            *err = NULL;
            continue;
        }
        client_unref(cl);
        if (!res && *code == ERR_AUTH && needs_sign_in(conn->config)) {
            conn_drop_client(conn);
            pthread_mutex_lock(&g_mu);
            conn->state = ST_NEEDS_AUTH;
            set_str(&conn->error, NULL);
            conn_changed_l(conn);
            free(*err);
            *err = sign_in_message(conn);
            pthread_mutex_unlock(&g_mu);
        }
        return res;
    }
}

static Conn *conn_new_l(Server *s) {
    Conn *c = xcalloc(1, sizeof *c);
    c->name = strdup(s->name);
    c->config = s->config;
    c->state = ST_CONNECTING;
    return c;
}

// ---------------------------------------------------------------------------------------
// naming (naming.ts) and the registered tools (index.ts registerTools, hideTools)
// ---------------------------------------------------------------------------------------

// createMcpToolName: mcp__<server>__<tool> with everything but [A-Za-z0-9_] made "_", shortened
// with a hash suffix past 64 characters or when taken
static char *tool_name(const char *server, const char *tool, bool (*taken)(const char *, void *), void *ctx) {
    Buf b = {0};
    char *raw = mfmt("mcp__%s__%s", server, tool);
    Str r = str(raw);
    for (size_t i = 0, n; i < r.len; i += n) {
        int u = u16_units(r, i, &n);
        char ch = r.ptr[i];
        bool ok = n == 1 && ((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_');
        if (ok) buf_addc(&b, ch);
        else
            for (int k = 0; k < u; k++) buf_addc(&b, '_');
    }
    free(raw);
    char *name = buf_take(&b);
    if (strlen(name) <= 64 && !(taken && taken(name, ctx))) return name;
    Buf h = {0};
    buf_adds(&h, server);
    buf_addc(&h, 0);
    buf_adds(&h, tool);
    uint8_t d[SHA256_LEN];
    sha256(h.ptr, h.len, d);
    buf_free(&h);
    char *out = mfmt("%.*s_%02x%02x%02x%02x", 64 - 8 - 1, name, d[0], d[1], d[2], d[3]);
    free(name);
    return out;
}

static Reg *find_reg_l(const char *name) {
    for (size_t i = 0; i < G.regs.len; i++)
        if (!strcmp(G.regs.ptr[i]->name, name)) return G.regs.ptr[i];
    return NULL;
}

static const char *owner_of_l(const char *name) {
    size_t nl = strlen(name);
    for (size_t i = 0; i < G.owners.len; i++)
        if (!strncmp(G.owners.ptr[i], name, nl) && G.owners.ptr[i][nl] == 1) return G.owners.ptr[i] + nl + 1;
    return NULL;
}

static void set_owner_l(const char *name, const char *owner) {
    size_t nl = strlen(name);
    for (size_t i = 0; i < G.owners.len; i++)
        if (!strncmp(G.owners.ptr[i], name, nl) && G.owners.ptr[i][nl] == 1) {
            set_str(&G.owners.ptr[i], mfmt("%s\x01%s", name, owner));
            return;
        }
    vec_push(&G.owners, mfmt("%s\x01%s", name, owner));
}

typedef struct {
    const char *owner;
    CStrVec *current;
    char **plain;
    size_t nplain;
} TakenCtx;

static bool name_taken(const char *cand, void *ctx) {
    TakenCtx *t = ctx;
    const char *ex = owner_of_l(cand);
    if (ex && strcmp(ex, t->owner)) return true;
    for (size_t i = 0; i < t->current->len; i++)
        if (!strcmp(t->current->ptr[i], cand)) return true;
    int count = 0;
    for (size_t i = 0; i < t->nplain; i++) count += !strcmp(t->plain[i], cand);
    return count > 1;
}

static Reg *upsert_reg_l(const char *name) {
    Reg *r = find_reg_l(name);
    if (r) return r;
    r = xcalloc(1, sizeof *r);
    r->name = strdup(name);
    vec_push(&G.regs, r);
    return r;
}

// createMcpToolDefinition's description: the trimmed description, else the title, else a stand-in
static char *tool_description(const char *server, const JValue *tool) {
    Str d = trim_js(json_strs(json_get(tool, "description")));
    if (d.len) return strndup(d.ptr, d.len);
    const JValue *title = json_get(tool, "title");
    if (!title) title = json_get(json_get(tool, "annotations"), "title");
    if (json_is(title, J_STRING) && title->s.len) return strndup(title->s.ptr, title->s.len);
    if (title && !json_is(title, J_STRING) && !json_is(title, J_NULL)) {
        // String(title) for a value a server should not send
        return js_text(title);
    }
    return mfmt("MCP tool %s from server %s", json_str(json_get(tool, "name")), server);
}

// normalizeMcpInputSchema: {...schema, type: schema.type ?? "object"}, properties: {} when missing
static JValue *normalize_schema(Arena *a, const JValue *schema) {
    JValue *s = obj_spread(a, schema);
    const JValue *ty = json_get(schema, "type");
    jput(a, s, "type", ty && !json_is(ty, J_NULL) ? (JValue *)ty : jstr(a, "object"));
    if (!json_get(schema, "properties")) jput(a, s, "properties", json_new(a, J_OBJECT));
    return s;
}

static void register_tools_l(Conn *conn) {
    const char *server = conn->name;
    Server *srv = find_server_l(server);
    const JValue *config = srv ? srv->config : conn->config;
    Arena t = {0};
    // every tool whose name sanitizes to the same name gets the hash suffix
    VEC(char *) uniq = {0};
    for (size_t i = 0; i < conn->tools.len; i++) {
        bool seen = false;
        for (size_t k = 0; k < uniq.len && !seen; k++) seen = !strcmp(uniq.ptr[k], conn->tools.ptr[i].name);
        if (!seen) vec_push(&uniq, conn->tools.ptr[i].name);
    }
    char **plain = xcalloc(uniq.len + 1, sizeof(char *));
    for (size_t i = 0; i < uniq.len; i++) plain[i] = tool_name(server, uniq.ptr[i], NULL, NULL);
    CStrVec current = {0};
    for (size_t i = 0; i < G.regs.len; i++)
        if (G.regs.ptr[i]->kind == REG_TOOL && G.regs.ptr[i]->server && !strcmp(G.regs.ptr[i]->server, server)) G.regs.ptr[i]->current = false;
    for (size_t i = 0; i < conn->tools.len; i++) {
        const char *tn = conn->tools.ptr[i].name;
        char *owner = mfmt("%s%c%s", server, 2, tn);
        TakenCtx ctx = {owner, &current, plain, uniq.len};
        char *name = tool_name(server, tn, name_taken, &ctx);
        set_owner_l(name, owner);
        vec_push(&current, name);
        JValue *tool = js_parse(&t, str(conn->tools.ptr[i].json));
        Reg *r = upsert_reg_l(name);
        r->kind = REG_TOOL;
        set_str(&r->server, strdup(server));
        set_str(&r->tool, strdup(tn));
        set_str(&r->label, mfmt("%s/%s", server, tn));
        set_str(&r->description, tool_description(server, tool));
        set_str(&r->params, js_text(normalize_schema(&t, json_get(tool, "inputSchema"))));
        set_str(&r->exposure, strdup(tool_exposure(config, tn)));
        r->current = true;
        r->hidden = !strcmp(r->exposure, "hidden");
        free(owner);
    }
    // tools the server no longer offers stay registered, hidden
    for (size_t i = 0; i < G.regs.len; i++) {
        Reg *r = G.regs.ptr[i];
        if (r->kind == REG_TOOL && r->server && !strcmp(r->server, server) && !r->current) r->hidden = true;
    }
    for (size_t i = 0; i < current.len; i++) free(current.ptr[i]);
    vec_free(&current);
    for (size_t i = 0; i < uniq.len; i++) free(plain[i]);
    free(plain);
    vec_free(&uniq);
    arena_reset(&t);
    sync_resource_tools_l();
}

// hideTools: a disabled server's tools made unreachable
static void hide_tools_l(const char *server) {
    for (size_t i = 0; i < G.regs.len; i++) {
        Reg *r = G.regs.ptr[i];
        if (r->kind == REG_TOOL && r->server && !strcmp(r->server, server) && r->current) {
            r->hidden = true;
            r->current = false;
        }
    }
    sync_resource_tools_l();
}

// ---------------------------------------------------------------------------------------
// the resource tools (resources.ts)
// ---------------------------------------------------------------------------------------

static const char LIST_PARAMS[] =
    "{\"type\":\"object\",\"properties\":{\"server\":{\"type\":\"string\",\"description\":\"MCP server name. Omit to list every server with "
    "resources.\"},\"cursor\":{\"type\":\"string\",\"description\":\"Opaque cursor from a previous call with the same server; omit for the "
    "first page.\"}},\"additionalProperties\":false}";
static const char READ_PARAMS[] =
    "{\"type\":\"object\",\"properties\":{\"server\":{\"type\":\"string\",\"description\":\"MCP server name exactly as configured. Must "
    "match the 'server' field returned by list_mcp_resources.\"},\"uri\":{\"type\":\"string\",\"description\":\"Resource URI to read. Must "
    "be one of the URIs returned by list_mcp_resources.\"}},\"required\":[\"server\",\"uri\"],\"additionalProperties\":false}";

// serversWithResources: enabled, connected with resources, exposure not hidden
static bool resource_server_l(const Server *s) {
    return s->conn && s->conn->has_resources && is_enabled(s) && strcmp(exposure_of(s->config), "hidden") != 0;
}

// syncResourceTools: the three tools registered with the widest exposure of the servers they
// reach, hidden when none has resources
static void sync_resource_tools_l(void) {
    bool direct = false, codemode = false, deferred = false;
    for (size_t i = 0; i < G.servers.len; i++)
        if (resource_server_l(G.servers.ptr[i])) {
            const char *e = exposure_of(G.servers.ptr[i]->config);
            direct |= !strcmp(e, "direct");
            codemode |= !strcmp(e, "codemode");
            deferred |= !strcmp(e, "deferred");
        }
    const char *next = direct ? "direct" : codemode ? "codemode" : deferred ? "deferred" : "hidden";
    if ((G.resources_exposure && !strcmp(next, G.resources_exposure)) || (!G.resources_exposure && !strcmp(next, "hidden"))) return;
    G.resources_exposure = next;
    static const struct {
        RegKind kind;
        const char *name, *description, *params;
    } defs[] = {
        {REG_RESOURCES_LIST, "list_mcp_resources",
         "Lists resources provided by MCP servers. Resources allow servers to share data that provides context to language models, such as "
         "files, database schemas, or application-specific information. Prefer resources over web search when possible.",
         LIST_PARAMS},
        {REG_RESOURCES_TEMPLATES, "list_mcp_resource_templates",
         "Lists resource templates provided by MCP servers. Parameterized resource templates allow servers to share data that takes "
         "parameters and provides context to language models, such as files, database schemas, or application-specific information. "
         "Prefer resource templates over web search when possible.",
         LIST_PARAMS},
        {REG_RESOURCES_READ, "read_mcp_resource", "Read a specific resource from an MCP server given the server name and resource URI.",
         READ_PARAMS},
    };
    for (size_t i = 0; i < COUNT(defs); i++) {
        Reg *r = upsert_reg_l(defs[i].name);
        r->kind = defs[i].kind;
        set_str(&r->label, strdup(defs[i].name));
        set_str(&r->description, strdup(defs[i].description));
        set_str(&r->params, strdup(defs[i].params));
        set_str(&r->exposure, strdup(next));
        r->hidden = !strcmp(next, "hidden");
        r->current = true;
    }
}

// ---------------------------------------------------------------------------------------
// configuration files (extensions/mcp/config.ts)
// ---------------------------------------------------------------------------------------

static Str strip_bom(Str s) { return s.len >= 3 && !memcmp(s.ptr, "\xef\xbb\xbf", 3) ? str_sub(s, 3, s.len) : s; }

typedef struct {
    char *name, *source, *scope, *override;
    JValue *config;
} Entry;
typedef VEC(Entry) EntryVec;

static void read_config_file(const char *path, bool project, EntryVec *servers) {
    char *text = NULL;
    size_t len = 0;
    if (!file_exists(path) || read_file(path, 64 << 20, &text, &len)) {
        free(text);
        return;
    }
    Arena *a = &G.arena;
    char *perr = NULL;
    JValue *parsed = code_oai_js_parse(a, strip_bom(str_n(text, len)), &perr);
    free(text);
    if (!parsed) {
        vec_push(&G.config_errors, mfmt("%s: %s", path, perr));
        return;
    }
    const JValue *ms = json_get(parsed, "mcpServers");
    if (!is_record(parsed) || (ms && !is_record(ms))) {
        vec_push(&G.config_errors, mfmt("%s: expected an object with an \"mcpServers\" object", path));
        return;
    }
    const JValue *ac = json_get(parsed, "autoEnableCodemode");
    if (ac && !json_is(ac, J_BOOL)) vec_push(&G.config_errors, mfmt("%s: autoEnableCodemode must be a boolean", path));
    for (size_t i = 0; ms && i < ms->obj.len; i++) {
        char *name = str_z(a, ms->obj.items[i].key);
        const JValue *value = ms->obj.items[i].val;
        Entry *base = NULL;
        for (size_t k = 0; k < servers->len; k++)
            if (!strcmp(servers->ptr[k].name, name)) base = &servers->ptr[k];
        if (project && is_record(value) && is_override(value)) {
            static const char *const keys[] = {"enabled", "exposure", "toolExposure"};
            bool extra = false;
            for (size_t k = 0; k < value->obj.len; k++) {
                bool known = false;
                for (size_t j = 0; j < COUNT(keys); j++) known |= str_eqz(value->obj.items[k].key, keys[j]);
                extra |= !known;
            }
            if (!base) vec_push(&G.config_errors, mfmt("%s: server \"%s\" needs \"command\" or \"url\", or a global server to override", path, name));
            else if (extra) vec_push(&G.config_errors, mfmt("%s: server \"%s\": an override can only set enabled, exposure, toolExposure", path, name));
            else {
                JValue *merged = obj_spread(a, base->config);
                for (size_t k = 0; k < value->obj.len; k++) json_put(a, merged, value->obj.items[k].key, value->obj.items[k].val);
                char *err = NULL;
                JValue *config = validate_server(a, name, merged, &err);
                if (!config) {
                    vec_push(&G.config_errors, mfmt("%s: %s", path, err));
                    free(err);
                } else {
                    base->config = config;
                    base->override = arena_strdup(a, path);
                }
            }
            continue;
        }
        char *err = NULL;
        JValue *config = validate_server(a, name, value, &err);
        if (!config) {
            vec_push(&G.config_errors, mfmt("%s: %s", path, err));
            free(err);
            continue;
        }
        // names that differ only in - and _ would share a namespace
        char *ns = mcp_namespace(name);
        const char *clash = NULL;
        for (size_t k = 0; k < servers->len && !clash; k++) {
            if (!strcmp(servers->ptr[k].name, name)) continue;
            char *o = mcp_namespace(servers->ptr[k].name);
            if (!strcmp(o, ns)) clash = servers->ptr[k].name;
            free(o);
        }
        free(ns);
        if (clash) {
            vec_push(&G.config_errors, mfmt("%s: server \"%s\" conflicts with \"%s\"", path, name, clash));
            continue;
        }
        if (project && is_http(config) && json_get(config, "auth")) {
            vec_push(&G.config_errors, mfmt("%s: server \"%s\": auth is only allowed in the global mcp.json", path, name));
            continue;
        }
        Entry e = {name, arena_strdup(a, path), project ? "project" : "global", NULL, config};
        if (base) *base = e; // Map.set keeps the first position
        else vec_push(servers, e);
    }
}

// editMcpServers + updateMcpServerConfig: one server's enabled / exposure changed in the file
// that defines or overrides it, the rest kept, written back with its indentation. NULL or the
// error message.
static char *update_server_config(const char *path, const char *name, int enabled, const char *exposure, bool override) {
    Arena t = {0};
    char *text = NULL;
    size_t len = 0;
    bool have = file_exists(path) && !read_file(path, 64 << 20, &text, &len);
    char *err = NULL;
    JValue *parsed;
    Str body = have ? strip_bom(str_n(text, len)) : S("");
    if (have) {
        char *perr = NULL;
        parsed = code_oai_js_parse(&t, body, &perr);
        if (!parsed) {
            err = strdup(perr);
            goto done;
        }
    } else parsed = json_new(&t, J_OBJECT);
    JValue *servers = json_get(parsed, "mcpServers");
    if (!is_record(parsed) || (servers && !is_record(servers))) {
        err = mfmt("%s: expected an object with an \"mcpServers\" object", path);
        goto done;
    }
    JValue *server = servers ? json_get(servers, name) : NULL;
    if (!server && override) {
        server = json_new(&t, J_OBJECT);
        JValue *ns = obj_spread(&t, servers);
        jput(&t, ns, name, server);
        jput(&t, parsed, "mcpServers", ns);
    }
    if (!is_record(server)) {
        err = mfmt("%s does not define MCP server \"%s\"", path, name);
        goto done;
    }
    bool keep_defaults = is_override(server);
    if (enabled >= 0) {
        if (enabled && !keep_defaults) obj_del(server, "enabled");
        else jput(&t, server, "enabled", json_bool(&t, enabled));
    }
    if (exposure) {
        if (!strcmp(exposure, "codemode") && !keep_defaults) obj_del(server, "exposure");
        else jput(&t, server, "exposure", jstr(&t, exposure));
    }
    // the indentation of the first indented line, else two spaces
    char ind[64] = "  ";
    for (size_t i = 0; have && i < body.len; i++) {
        if (i && body.ptr[i - 1] != '\n') continue;
        size_t k = i;
        while (k < body.len && (body.ptr[k] == ' ' || body.ptr[k] == '\t')) k++;
        if (k > i && k < body.len && !strchr(" \t\r\n\v\f", body.ptr[k]) && k - i < sizeof ind) {
            memcpy(ind, body.ptr + i, k - i);
            ind[k - i] = 0;
            break;
        }
    }
    Buf out = {0};
    js_pretty(&out, parsed, ind, 0);
    buf_addc(&out, '\n');
    char *dir = strdup(path);
    char *sl = strrchr(dir, '/');
    if (sl && sl != dir) {
        *sl = 0;
        mkdir_p(dir);
    }
    free(dir);
    // writeFileSync through the shared writer (code_cfg.c): a symlinked mcp.json stays a link, its
    // mode kept, never half written
    if (!cfg_write_file(path, out.ptr, out.len, NULL)) err = mfmt("%s", strerror(errno));
    buf_free(&out);
done:
    free(text);
    arena_reset(&t);
    return err;
}

// ---------------------------------------------------------------------------------------
// the manager (index.ts)
// ---------------------------------------------------------------------------------------

// describeState
static char *describe_state_l(const Server *s, bool with_error) {
    if (!is_enabled(s)) return strdup("disabled");
    const Conn *c = s->conn;
    if (!c) return strdup("starting");
    switch (c->state) {
    case ST_NEEDS_AUTH: return strdup("needs sign-in");
    case ST_FAILED: {
        if (!with_error) return strdup("failed");
        Str f = first_line(str(c->error ? c->error : "unknown error"));
        return mfmt("failed: %.*s", (int)f.len, f.ptr);
    }
    case ST_CONNECTED:
        return mfmt("connected \xc2\xb7 %zu tool%s%s", c->tools.len, c->tools.len == 1 ? "" : "s", c->has_resources ? " \xc2\xb7 resources" : "");
    case ST_CONNECTING: return strdup("connecting\xe2\x80\xa6");
    default: return strdup(STATE_NAMES[c->state]);
    }
}

// attentionRank: servers that need the user first
static int attention_rank(const Server *s) {
    if (!is_enabled(s)) return 5;
    if (!s->conn) return 3;
    switch (s->conn->state) {
    case ST_NEEDS_AUTH: return 0;
    case ST_FAILED: return 1;
    case ST_DISCONNECTED: return 2;
    case ST_CONNECTED: return 4;
    default: return 3;
    }
}

// localeCompare for server names (letters, digits, _ and -): ICU's root collation, punctuation
// first, then digits, then letters without case; lowercase first when that ties
static int coll_key(char c) {
    if (c == '_') return 1;
    if (c == '-') return 2;
    if (c >= '0' && c <= '9') return 10 + (c - '0');
    if (c >= 'a' && c <= 'z') return 100 + (c - 'a');
    if (c >= 'A' && c <= 'Z') return 100 + (c - 'A');
    return 200 + (unsigned char)c;
}

static int locale_cmp(const char *a, const char *b) {
    for (size_t i = 0;; i++) {
        if (!a[i] || !b[i]) {
            if (a[i] || b[i]) return a[i] ? 1 : -1;
            break;
        }
        int x = coll_key(a[i]), y = coll_key(b[i]);
        if (x != y) return x < y ? -1 : 1;
    }
    for (size_t i = 0; a[i] && b[i]; i++) {
        bool ua = a[i] >= 'A' && a[i] <= 'Z', ub = b[i] >= 'A' && b[i] <= 'Z';
        if (ua != ub) return ua ? 1 : -1;
    }
    return 0;
}

// reportProblems: one notice for what needs the user (config errors, failures, sign-ins)
static void report_problems_l(Server **only, size_t nonly) {
    Buf b = {0};
    size_t n = 0;
    if (!only)
        for (size_t i = 0; i < G.config_errors.len; i++, n++) buf_printf(&b, "\n  config: %s", G.config_errors.ptr[i]);
    size_t count = only ? nonly : G.servers.len;
    for (size_t i = 0; i < count; i++) {
        Server *s = only ? only[i] : G.servers.ptr[i];
        if (s->conn && (s->conn->state == ST_NEEDS_AUTH || s->conn->state == ST_FAILED)) {
            char *d = describe_state_l(s, true);
            buf_printf(&b, "\n  %s: %s", s->name, d);
            free(d);
            n++;
        }
    }
    if (n) notify_l(1, mfmt("MCP servers need attention:%s\nRun /mcp to fix.", b.ptr));
    buf_free(&b);
}

typedef struct {
    Server *server;
    uint64_t attempt;
    bool startup;
    Conn *after; // a detached connection to wait for first (setEnabled after a disable)
} StartArg;

static bool is_current_l(Server *s, uint64_t attempt) { return !G.shutdown && s->attempt == attempt && is_enabled(s) && server_listed_l(s); }

// startConnection's work: the connection created, then its client opened
static void *start_main(void *arg) {
    StartArg *a = arg;
    Server *s = a->server;
    pthread_mutex_lock(&g_mu);
    while (s->closing > 0 && !G.shutdown) timed_wait_l(now_ns() + 100000000ull);
    Conn *c = NULL;
    if (is_current_l(s, a->attempt)) {
        c = conn_new_l(s);
        s->conn = c;
        changed_l();
    }
    pthread_mutex_unlock(&g_mu);
    if (c) {
        char *err = NULL;
        Client *cl = conn_get_client(c, &err);
        if (cl) client_unref(cl);
        free(err);
    }
    pthread_mutex_lock(&g_mu);
    s->inflight--;
    if (a->startup && --G.startup_left == 0 && !G.shutdown) report_problems_l(NULL, 0);
    changed_l();
    pthread_mutex_unlock(&g_mu);
    free(a);
    bg_end();
    return NULL;
}

static void start_connection_l(Server *s, bool startup) {
    StartArg *a = xcalloc(1, sizeof *a);
    a->server = s;
    a->attempt = s->attempt = ++G.attempt_seq;
    a->startup = startup;
    s->inflight++;
    bg_begin_l();
    spawn_detached(start_main, a);
}

void code_mcp_start(const CodeEnv *env, bool enabled, const char *const *allow, size_t nallow, bool allow_given) {
    pthread_mutex_lock(&g_mu);
    if (G.started) {
        pthread_mutex_unlock(&g_mu);
        return;
    }
    G.started = true;
    G.enabled = enabled;
    G.allow_given = allow_given;
    G.allow = xcalloc(nallow + 1, sizeof(char *));
    for (size_t i = 0; i < nallow; i++) {
        G.allow[i] = strdup(allow[i]);
        G.allow_mcp |= !strncmp(allow[i], "mcp__", 5);
    }
    G.nallow = nallow;
    G.cwd = strdup(env->cwd);
    G.agent_dir = strdup(env->agent_dir);
    G.log_path = mfmt("%s/mcp.log", env->agent_dir);
    if (!enabled) {
        pthread_mutex_unlock(&g_mu);
        return;
    }
    Arena t = {0};
    bool trusted = code_project_trusted(&t, env);
    arena_reset(&t);
    EntryVec entries = {0};
    char *global = mfmt("%s/mcp.json", env->agent_dir);
    read_config_file(global, false, &entries);
    free(global);
    if (trusted) {
        G.project_config = mfmt("%s/.pi/mcp.json", env->cwd);
        read_config_file(G.project_config, true, &entries);
    }
    for (size_t i = 0; i < entries.len; i++) {
        Server *s = xcalloc(1, sizeof *s);
        s->name = strdup(entries.ptr[i].name);
        s->source = strdup(entries.ptr[i].source);
        s->scope = strdup(entries.ptr[i].scope);
        s->override = dupz(entries.ptr[i].override);
        s->config = entries.ptr[i].config;
        vec_push(&G.servers, s);
    }
    vec_free(&entries);
    size_t n = 0;
    for (size_t i = 0; i < G.servers.len; i++) n += is_enabled(G.servers.ptr[i]);
    if (!n) report_problems_l(NULL, 0);
    G.startup_left = (int)n;
    for (size_t i = 0; i < G.servers.len; i++)
        if (is_enabled(G.servers.ptr[i])) start_connection_l(G.servers.ptr[i], true);
    pthread_mutex_unlock(&g_mu);
}

void code_mcp_shutdown(void) {
    pthread_mutex_lock(&g_mu);
    if (!G.started || G.shutdown) {
        pthread_mutex_unlock(&g_mu);
        return;
    }
    G.shutdown = true;
    VEC(Conn *) conns = {0};
    for (size_t i = 0; i < G.servers.len; i++)
        if (G.servers.ptr[i]->conn) vec_push(&conns, G.servers.ptr[i]->conn);
    pthread_mutex_unlock(&g_mu);
    for (size_t i = 0; i < conns.len; i++) conn_close(conns.ptr[i]);
    vec_free(&conns);
    // the background work (connects, reconnects, refreshes), at most 5 s more
    pthread_mutex_lock(&g_mu);
    uint64_t until = now_ns() + 5000000000ull;
    while (G.bg > 0 && now_ns() < until) {
        // a connection opened meanwhile
        for (size_t i = 0; i < G.servers.len; i++) {
            Conn *c = G.servers.ptr[i]->conn;
            if (c && !c->shut) {
                pthread_mutex_unlock(&g_mu);
                conn_close(c);
                pthread_mutex_lock(&g_mu);
            }
        }
        timed_wait_l(now_ns() + 50000000ull);
    }
    pthread_mutex_unlock(&g_mu);
}

// before_agent_start: the first prompt waits for the servers with declared tools, at most 10 s
// (waitForDirectServers)
void code_mcp_before_prompt(void) {
    pthread_mutex_lock(&g_mu);
    if (!G.started || G.waited_startup) {
        pthread_mutex_unlock(&g_mu);
        return;
    }
    G.waited_startup = true;
    uint64_t until = now_ns() + 10000000000ull;
    bool finished = true;
    for (;;) {
        bool waiting = false;
        for (size_t i = 0; i < G.servers.len; i++) {
            Server *s = G.servers.ptr[i];
            if (is_enabled(s) && has_exposure(s->config, declared_exposure) && s->inflight > 0) waiting = true;
        }
        if (!waiting) break;
        if (now_ns() >= until) {
            finished = false;
            break;
        }
        timed_wait_l(MIN(until, now_ns() + 100000000ull));
    }
    if (!finished) notify_l(0, strdup("MCP servers are still connecting; their tools become available once connected."));
    pthread_mutex_unlock(&g_mu);
}

static bool allowed_l(const char *name) {
    if (!G.allow_given) return true;
    if (!G.allow_mcp) return false;
    for (size_t i = 0; i < G.nallow; i++)
        if (strchr(G.allow[i], '*') ? pattern_match(G.allow[i], name) : !strcmp(G.allow[i], name)) return true;
    return false;
}

size_t code_mcp_declared(Arena *a, const char ***names) {
    pthread_mutex_lock(&g_mu);
    const char **v = arena_alloc(a, (G.regs.len + 1) * sizeof *v);
    size_t n = 0;
    for (size_t i = 0; G.enabled && i < G.regs.len; i++) {
        Reg *r = G.regs.ptr[i];
        if (!r->hidden && declared_exposure(r->exposure) && allowed_l(r->name)) v[n++] = arena_strdup(a, r->name);
    }
    pthread_mutex_unlock(&g_mu);
    *names = v;
    return n;
}

JValue *code_mcp_tool_parameters(Arena *a, Str name) {
    pthread_mutex_lock(&g_mu);
    char *z = strndup(name.ptr, name.len);
    Reg *r = find_reg_l(z);
    free(z);
    JValue *p = r ? js_parse(a, str(r->params)) : NULL;
    pthread_mutex_unlock(&g_mu);
    return p;
}

JValue *code_mcp_tool_schema(Arena *a, Str name) {
    pthread_mutex_lock(&g_mu);
    char *z = strndup(name.ptr, name.len);
    Reg *r = find_reg_l(z);
    free(z);
    JValue *out = NULL;
    if (r) {
        out = json_new(a, J_OBJECT);
        jput(a, out, "type", jstr(a, "function"));
        JValue *f = json_new(a, J_OBJECT);
        jput(a, f, "name", jstr(a, r->name));
        jput(a, f, "description", jstr(a, r->description));
        jput(a, f, "parameters", js_parse(a, str(r->params)));
        jput(a, out, "function", f);
    }
    pthread_mutex_unlock(&g_mu);
    return out;
}

char *code_mcp_tool_label(const char *name) {
    pthread_mutex_lock(&g_mu);
    Reg *r = find_reg_l(name);
    char *l = r ? strdup(r->label) : NULL;
    pthread_mutex_unlock(&g_mu);
    if (!l && !strncmp(name, "mcp__", 5)) {
        // a resumed session's call to a tool whose server did not connect: /^mcp__(.+?)__(.+)$/
        const char *rest = name + 5;
        const char *sep = strstr(rest + 1, "__");
        if (sep && sep > rest && sep[2]) l = mfmt("%.*s/%s", (int)(sep - rest), rest, sep + 2);
    }
    return term_safe(l, false);
}

// ---------------------------------------------------------------------------------------
// results (tools.ts convertMcpResult, protocol/content.ts toLlmContent, output-limit.ts)
// ---------------------------------------------------------------------------------------

static JValue *text_block(Arena *a, Str text) {
    JValue *b = json_new(a, J_OBJECT);
    jput(a, b, "type", jstr(a, "text"));
    jput(a, b, "text", json_string(a, text));
    return b;
}

static int b64v(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+' || c == '-') return 62;
    if (c == '/' || c == '_') return 63;
    return -1;
}

// Buffer.from(s, "base64"): invalid characters skipped, stops at "="
static void b64_decode(Str s, Buf *out) {
    uint32_t acc = 0;
    int bits = 0;
    for (size_t i = 0; i < s.len; i++) {
        if (s.ptr[i] == '=') break;
        int v = b64v((unsigned char)s.ptr[i]);
        if (v < 0) continue;
        acc = acc << 6 | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            buf_addc(out, (char)(acc >> bits & 0xff));
        }
    }
}

static const char *tmp_dir(void) {
    const char *t = getenv("TMPDIR");
    if (!t || !*t) t = getenv("TMP");
    if (!t || !*t) t = getenv("TEMP");
    if (!t || !*t) t = "/tmp";
    return t;
}

// writeOutputFile("pi-mcp", extension, data): <tmpdir>/pi-mcp-<16 hex><extension>, mode 0600
static char *save_output(const void *data, size_t n, const char *ext, char **err) {
    *err = NULL;
    uint8_t r[8];
    arc4random_buf(r, sizeof r);
    const char *t = tmp_dir();
    size_t tl = strlen(t);
    char *path = mfmt("%.*s/pi-mcp-%02x%02x%02x%02x%02x%02x%02x%02x%s", (int)(tl > 1 && t[tl - 1] == '/' ? tl - 1 : tl), t, r[0], r[1], r[2],
                      r[3], r[4], r[5], r[6], r[7], ext);
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) {
        *err = mfmt("%s, open '%s'", errno_code(errno), path);
        free(path);
        return NULL;
    }
    const char *p = data;
    while (n) {
        ssize_t w = write(fd, p, n);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) break;
        p += w;
        n -= (size_t)w;
    }
    close(fd);
    return path;
}

// isTextMimeType
static bool text_mime(Str m) {
    if (!m.len) return false;
    ptrdiff_t semi = str_find_char(m, ';', 0);
    Str t = str_trim(semi >= 0 ? str_sub(m, 0, (size_t)semi) : m);
    char z[256];
    size_t n = MIN(t.len, sizeof z - 1);
    for (size_t i = 0; i < n; i++) z[i] = (char)((t.ptr[i] >= 'A' && t.ptr[i] <= 'Z') ? t.ptr[i] + 32 : t.ptr[i]);
    z[n] = 0;
    Str l = str(z);
    return str_starts(l, S("text/")) || str_eqz(l, "application/json") || str_ends(l, S("+json")) || str_ends(l, S("+xml"));
}

// extensionOf: the URI's (path's) extension, else .bin
static char *extension_of(const char *uri) {
    Url u;
    const char *path = uri;
    bool parsed = url_parse(uri, &u);
    if (parsed) path = u.path;
    const char *dot = strrchr(path, '.');
    char *ext = NULL;
    if (dot && !strchr(dot, '/')) {
        size_t n = strlen(dot + 1);
        bool ok = n >= 1 && n <= 8;
        for (size_t i = 1; ok && dot[i]; i++) ok = (dot[i] >= 'A' && dot[i] <= 'Z') || (dot[i] >= 'a' && dot[i] <= 'z') || (dot[i] >= '0' && dot[i] <= '9');
        if (ok) ext = strdup(dot);
    }
    if (parsed) url_free(&u);
    return ext ? ext : strdup(".bin");
}

// blockToLlmContent
static void block_llm(Arena *a, const JValue *block, JValue *out) {
    Str type = json_strs(json_get(block, "type"));
    if (str_eqz(type, "text")) json_push(a, out, text_block(a, json_strs(json_get(block, "text"))));
    else if (str_eqz(type, "image")) {
        JValue *b = json_new(a, J_OBJECT);
        jput(a, b, "type", jstr(a, "image"));
        jput(a, b, "data", (JValue *)(json_get(block, "data") ? json_get(block, "data") : jstr(a, "")));
        jput(a, b, "mimeType", (JValue *)(json_get(block, "mimeType") ? json_get(block, "mimeType") : jstr(a, "")));
        json_push(a, out, b);
    } else if (str_eqz(type, "audio")) {
        Str m = json_strs(json_get(block, "mimeType"));
        json_push(a, out, text_block(a, str(arena_printf(a, "[audio %.*s omitted]", (int)m.len, m.ptr))));
    } else if (str_eqz(type, "resource_link")) {
        Str n = json_strs(json_get(block, "name")), u = json_strs(json_get(block, "uri"));
        json_push(a, out, text_block(a, str(arena_printf(a, "%.*s: %.*s", (int)n.len, n.ptr, (int)u.len, u.ptr))));
    } else if (str_eqz(type, "resource")) {
        const JValue *r = json_get(block, "resource");
        if (json_get(r, "text")) json_push(a, out, text_block(a, json_strs(json_get(r, "text"))));
        else if (str_starts(json_strs(json_get(r, "mimeType")), S("image/"))) {
            JValue *b = json_new(a, J_OBJECT);
            jput(a, b, "type", jstr(a, "image"));
            jput(a, b, "data", (JValue *)json_get(r, "blob"));
            jput(a, b, "mimeType", (JValue *)json_get(r, "mimeType"));
            json_push(a, out, b);
        } else {
            Str u = json_strs(json_get(r, "uri")), m = json_strs(json_get(r, "mimeType"));
            json_push(a, out, text_block(a, str(arena_printf(a, "[binary resource %.*s (%.*s) omitted]", (int)u.len, u.ptr, m.len ? (int)m.len : 12,
                                                              m.len ? m.ptr : "unknown type"))));
        }
    } else json_push(a, out, text_block(a, str(arena_printf(a, "[unsupported MCP content %.*s]", (int)type.len, type.ptr))));
}

// blockToContent (tools.ts): resource links name read_mcp_resource; binary resources saved
static void block_content(Arena *a, const char *server, const JValue *block, bool readable, JValue *out) {
    Str type = json_strs(json_get(block, "type"));
    if (str_eqz(type, "resource_link")) {
        Buf d = {0};
        Str mime = json_strs(json_get(block, "mimeType"));
        const JValue *size = json_get(block, "size");
        double sz;
        Buf det = {0};
        if (mime.len) buf_addstr(&det, mime);
        if (size && jnum(size, &sz)) {
            char *fs = format_size(sz);
            if (det.len) buf_adds(&det, ", ");
            buf_adds(&det, fs);
            free(fs);
        }
        const JValue *title = json_get(block, "title");
        Str label = json_is(title, J_STRING) ? title->s : json_strs(json_get(block, "name"));
        Str desc = json_strs(json_get(block, "description"));
        Str uri = json_strs(json_get(block, "uri"));
        buf_printf(&d, "[Resource %.*s \"%.*s\"", (int)uri.len, uri.ptr, (int)label.len, label.ptr);
        if (det.len) buf_printf(&d, " (%s)", det.ptr);
        if (desc.len) buf_printf(&d, ": %.*s", (int)desc.len, desc.ptr);
        if (readable) buf_printf(&d, ". Read it with read_mcp_resource (server \"%s\")", server);
        buf_addc(&d, ']');
        json_push(a, out, text_block(a, str(arena_strndup(a, d.ptr, d.len))));
        buf_free(&d);
        buf_free(&det);
        return;
    }
    const JValue *res = json_get(block, "resource");
    if (str_eqz(type, "resource") && json_get(res, "blob") && !str_starts(json_strs(json_get(res, "mimeType")), S("image/"))) {
        Str uri = json_strs(json_get(res, "uri")), mime = json_strs(json_get(res, "mimeType"));
        Buf data = {0};
        b64_decode(json_strs(json_get(res, "blob")), &data);
        if (text_mime(mime)) {
            json_push(a, out, text_block(a, code_utf8_clean(a, str(arena_strndup(a, data.ptr ? data.ptr : "", data.len)))));
            buf_free(&data);
            return;
        }
        char *fs = format_size((double)data.len);
        char *kind = mfmt("%.*s, %s", mime.len ? (int)mime.len : 12, mime.len ? mime.ptr : "unknown type", fs);
        free(fs);
        char *uz = strndup(uri.ptr, uri.len);
        char *ext = extension_of(uz);
        char *err = NULL;
        char *path = save_output(data.ptr ? data.ptr : "", data.len, ext, &err);
        char *t = path ? mfmt("[Binary resource %s (%s) saved to %s]", uz, kind, path) : mfmt("[Binary resource %s (%s) could not be saved: %s]", uz, kind, err);
        json_push(a, out, text_block(a, str(arena_strdup(a, t))));
        free(t);
        free(path);
        free(err);
        free(ext);
        free(uz);
        free(kind);
        buf_free(&data);
        return;
    }
    block_llm(a, block, out);
}

static Str text_of(Arena *a, const JValue *content) {
    Buf b = {0};
    bool first = true;
    for (size_t i = 0; i < content->arr.len; i++) {
        const JValue *bl = content->arr.items[i];
        if (!str_eqz(json_strs(json_get(bl, "type")), "text")) continue;
        if (!first) buf_addc(&b, '\n');
        first = false;
        buf_addstr(&b, json_strs(json_get(bl, "text")));
    }
    Str s = {arena_strndup(a, b.ptr ? b.ptr : "", b.len), b.len};
    buf_free(&b);
    return s;
}

// limitMcpContent: text past 20 KB keeps its start and end, the middle cut out, and the whole
// text saved to a file named after the cut
static JValue *limit_content(Arena *a, JValue *content, char **full_path) {
    *full_path = NULL;
    const size_t max = 20 * 1024;
    Str combined = text_of(a, content);
    if (combined.len <= max) return content;
    size_t head_end = max / 2;
    while (head_end > 0 && ((unsigned char)combined.ptr[head_end] & 0xc0) == 0x80) head_end--;
    size_t tail_start = combined.len - (max - max / 2);
    while (tail_start < combined.len && ((unsigned char)combined.ptr[tail_start] & 0xc0) == 0x80) tail_start++;
    size_t removed = 0;
    for (size_t i = head_end; i < tail_start; i++) removed += ((unsigned char)combined.ptr[i] & 0xc0) != 0x80;
    size_t lines = 1;
    for (size_t i = 0; i < combined.len; i++) lines += combined.ptr[i] == '\n';
    if (combined.len && combined.ptr[combined.len - 1] == '\n') lines--;
    char *err = NULL;
    char *path = save_output(combined.ptr, combined.len, ".txt", &err);
    Buf t = {0};
    buf_printf(&t, "Warning: truncated output (original token count: %zu)\nTotal output lines: %zu\n\n", (combined.len + 3) / 4, lines);
    buf_add(&t, combined.ptr, head_end);
    buf_printf(&t, "\xe2\x80\xa6%zu chars truncated\xe2\x80\xa6", removed);
    buf_add(&t, combined.ptr + tail_start, combined.len - tail_start);
    if (path) buf_printf(&t, "\n\n[Full output: %s (read it with offset/limit)]", path);
    else buf_printf(&t, "\n\n[Could not save the full output: %s]", err);
    JValue *out = json_new(a, J_ARRAY);
    json_push(a, out, text_block(a, str_n(arena_strndup(a, t.ptr, t.len), t.len)));
    for (size_t i = 0; i < content->arr.len; i++)
        if (str_eqz(json_strs(json_get(content->arr.items[i], "type")), "image")) json_push(a, out, content->arr.items[i]);
    buf_free(&t);
    if (path) *full_path = arena_strdup(a, path);
    free(path);
    free(err);
    return out;
}

static JValue *details_of(Arena *a, const char *server, const char *tool, const char *full) {
    JValue *d = json_new(a, J_OBJECT);
    jput(a, d, "server", jstr(a, server));
    jput(a, d, "tool", jstr(a, tool));
    if (full) jput(a, d, "fullOutputPath", jstr(a, full));
    return d;
}

typedef struct {
    const CodeEnv *env;
    const char *server, *tool;
} ProgressCtx;

// onProgress: the partial result tool_execution_update carries
static void on_progress(void *ctx, const char *text) {
    ProgressCtx *p = ctx;
    if (!p->env->on_partial) return;
    Arena t = {0};
    JValue *r = json_new(&t, J_OBJECT);
    JValue *content = json_new(&t, J_ARRAY);
    json_push(&t, content, text_block(&t, str(text)));
    jput(&t, r, "content", content);
    jput(&t, r, "details", details_of(&t, p->server, p->tool, NULL));
    p->env->on_partial(p->env->partial_ctx, r);
    arena_reset(&t);
}

// The tool_call hook for a direct call: its server's connection attempts settled first
static void wait_ready_l(const char *server, volatile int *abort) {
    for (;;) {
        Server *s = find_server_l(server);
        if (!s || !is_enabled(s) || s->inflight <= 0) return;
        if (abort && *abort) return;
        timed_wait_l(now_ns() + 50000000ull);
    }
}

static void resource_call(Arena *a, const CodeEnv *env, Reg *r, const JValue *args, JValue **content, JValue **details, JValue **structured,
                          const char **thrown);

void code_mcp_tool_call(Arena *a, const CodeEnv *env, Str name, const JValue *args, JValue **content, JValue **details,
                        JValue **structured, bool *is_error, const char **thrown) {
    *content = *details = *structured = NULL;
    *is_error = false;
    *thrown = NULL;
    char *nz = strndup(name.ptr, name.len);
    pthread_mutex_lock(&g_mu);
    Reg *r = find_reg_l(nz);
    free(nz);
    if (!r) {
        pthread_mutex_unlock(&g_mu);
        *thrown = arena_printf(a, "Tool %.*s not found", (int)name.len, name.ptr);
        return;
    }
    if (r->kind != REG_TOOL) {
        for (size_t i = 0; i < G.servers.len; i++) wait_ready_l(G.servers.ptr[i]->name, env->abort);
        pthread_mutex_unlock(&g_mu);
        resource_call(a, env, r, args, content, details, structured, thrown);
        return;
    }
    char *server = strdup(r->server), *tool = strdup(r->tool);
    wait_ready_l(server, env->abort);
    // the definition's getClient: the server's connection as it is now
    Server *s = find_server_l(server);
    Conn *conn = NULL;
    if (!s || !is_enabled(s)) *thrown = arena_printf(a, "MCP server \"%s\" is disabled.", server);
    else if (!(conn = s->conn)) *thrown = arena_printf(a, "MCP server \"%s\" is still starting.", server);
    else {
        bool offered = false;
        for (size_t i = 0; i < conn->tools.len && !offered; i++) offered = !strcmp(conn->tools.ptr[i].name, tool);
        if (!strcmp(tool_exposure(s->config, tool), "hidden") || (conn->state == ST_CONNECTED && !offered))
            *thrown = arena_printf(a, "MCP tool \"%s/%s\" is no longer available.", server, tool);
    }
    bool readable = s && resource_server_l(s);
    uint64_t timeout = s ? timeout_ms_of(s->config) : 60000;
    pthread_mutex_unlock(&g_mu);
    if (*thrown) goto out;
    char *err = NULL;
    Arena t = {0};
    JValue *params = json_new(&t, J_OBJECT);
    jput(&t, params, "name", jstr(&t, tool));
    if (args) jput(&t, params, "arguments", (JValue *)args);
    ProgressCtx pc = {env, server, tool};
    int code = 0;
    char *res = conn_request(conn, "tools/call", params, timeout, env->abort, on_progress, &pc, &err, &code);
    arena_reset(&t);
    if (!res) {
        *thrown = arena_strdup(a, err);
        free(err);
        goto out;
    }
    JValue *result = js_parse(a, str(res));
    free(res);
    // validateCallToolResult: content may be missing (servers that send only structuredContent)
    const JValue *rc = json_get(result, "content");
    if (!is_record(result) || (rc && !json_is(rc, J_ARRAY))) {
        *thrown = "Invalid MCP tools/call result";
        goto out;
    }
    if (!rc) {
        JValue *r2 = obj_spread(a, result);
        jput(a, r2, "content", json_new(a, J_ARRAY));
        result = r2;
        rc = json_get(result, "content");
    }
    JValue *conv = json_new(a, J_ARRAY);
    if (rc->arr.len)
        for (size_t i = 0; i < rc->arr.len; i++) block_content(a, server, rc->arr.items[i], readable, conv);
    else if (json_get(result, "structuredContent")) {
        Buf b = {0};
        js_pretty(&b, json_get(result, "structuredContent"), "  ", 0);
        json_push(a, conv, text_block(a, str_n(arena_strndup(a, b.ptr, b.len), b.len)));
        buf_free(&b);
    }
    const JValue *ie = json_get(result, "isError");
    bool err_result = ie && !(json_is(ie, J_BOOL) && !ie->b) && !json_is(ie, J_NULL);
    if (err_result && !text_of(a, conv).len) json_push(a, conv, text_block(a, str(arena_printf(a, "MCP tool %s/%s returned an error", server, tool))));
    char *full = NULL;
    *content = limit_content(a, conv, &full);
    *details = details_of(a, server, tool, full);
    JValue *sr = obj_spread(a, result);
    obj_del(sr, "_meta");
    *structured = sr;
    *is_error = err_result;
out:
    free(server);
    free(tool);
}

// ---- the resource tools' calls (resources.ts) ----

typedef struct {
    char *name;
    uint64_t timeout;
    Conn *conn;
} ResServer;

// isMcpAppResource
static bool app_resource(const JValue *item) {
    Str uri = json_strs(json_get(item, "uri"));
    if (!uri.len) uri = json_strs(json_get(item, "uriTemplate"));
    if (str_starts(uri, S("ui://"))) return true;
    Str m = json_strs(json_get(item, "mimeType"));
    // /;\s*profile\s*=\s*"?mcp-app"?/i
    for (size_t i = 0; i < m.len; i++) {
        if (m.ptr[i] != ';') continue;
        size_t k = i + 1;
        while (k < m.len && (m.ptr[k] == ' ' || m.ptr[k] == '\t')) k++;
        if (k + 7 > m.len || strncasecmp(m.ptr + k, "profile", 7)) continue;
        k += 7;
        while (k < m.len && (m.ptr[k] == ' ' || m.ptr[k] == '\t')) k++;
        if (k >= m.len || m.ptr[k] != '=') continue;
        k++;
        while (k < m.len && (m.ptr[k] == ' ' || m.ptr[k] == '\t')) k++;
        if (k < m.len && m.ptr[k] == '"') k++;
        if (k + 7 <= m.len && !strncasecmp(m.ptr + k, "mcp-app", 7)) return true;
    }
    return false;
}

// listed(): the item without _meta and icons, tagged with its server
static JValue *listed_item(Arena *a, const char *server, const JValue *item) {
    JValue *o = json_new(a, J_OBJECT);
    jput(a, o, "server", jstr(a, server));
    for (size_t i = 0; i < item->obj.len; i++)
        if (!str_eqz(item->obj.items[i].key, "_meta") && !str_eqz(item->obj.items[i].key, "icons")) json_put(a, o, item->obj.items[i].key, item->obj.items[i].val);
    // toResource: name defaults to the uri
    if (!json_get(o, "name")) jput(a, o, "name", (JValue *)(json_get(item, "uri") ? json_get(item, "uri") : json_get(item, "uriTemplate")));
    return o;
}

static const char *string_arg(Arena *a, const JValue *args, const char *key, const char **thrown) {
    const JValue *v = json_get(args, key);
    if (!v || json_is(v, J_NULL)) return NULL;
    if (!json_is(v, J_STRING)) {
        *thrown = arena_printf(a, "%s must be a string", key);
        return NULL;
    }
    Str t = trim_js(v->s);
    return t.len ? str_z(a, t) : NULL;
}

// a request through the server's connection (withClient); NULL with *err
static char *res_request(Conn *conn, const char *method, const JValue *params, uint64_t timeout, volatile int *abort, char **err, int *code) {
    return conn_request(conn, method, params, timeout, abort, NULL, NULL, err, code);
}

static void resource_call(Arena *a, const CodeEnv *env, Reg *r, const JValue *args, JValue **content, JValue **details, JValue **structured,
                          const char **thrown) {
    // the servers with resources, now
    pthread_mutex_lock(&g_mu);
    VEC(ResServer) servers = {0};
    for (size_t i = 0; i < G.servers.len; i++)
        if (resource_server_l(G.servers.ptr[i]))
            vec_push(&servers, ((ResServer){strdup(G.servers.ptr[i]->name), timeout_ms_of(G.servers.ptr[i]->config), G.servers.ptr[i]->conn}));
    pthread_mutex_unlock(&g_mu);
    const char *tool = r->kind == REG_RESOURCES_LIST ? "list_mcp_resources" : r->kind == REG_RESOURCES_TEMPLATES ? "list_mcp_resource_templates" : "read_mcp_resource";
    const char *server_name = string_arg(a, args, "server", thrown);
    ResServer *server = NULL;
    if (*thrown) goto out;
    if (server_name || r->kind == REG_RESOURCES_READ) {
        const char *uri = NULL;
        if (r->kind == REG_RESOURCES_READ) {
            uri = string_arg(a, args, "uri", thrown);
            if (*thrown) goto out;
            if (!server_name) {
                *thrown = "server must be provided";
                goto out;
            }
            if (!uri) {
                *thrown = "uri must be provided";
                goto out;
            }
        }
        for (size_t i = 0; i < servers.len; i++)
            if (!strcmp(servers.ptr[i].name, server_name)) server = &servers.ptr[i];
        if (!server) {
            Buf av = {0};
            for (size_t i = 0; i < servers.len; i++) buf_printf(&av, "%s%s", i ? ", " : "", servers.ptr[i].name);
            *thrown = av.len ? arena_printf(a, "MCP server \"%s\" has no resources. Servers with resources: %s", server_name, av.ptr)
                             : arena_printf(a, "MCP server \"%s\" has no resources", server_name);
            buf_free(&av);
            goto out;
        }
        char *err = NULL;
        int code = 0;
        if (r->kind == REG_RESOURCES_READ) {
            JValue *params = json_new(a, J_OBJECT);
            jput(a, params, "uri", jstr(a, uri));
            char *res = res_request(server->conn, "resources/read", params, server->timeout, env->abort, &err, &code);
            if (!res) {
                *thrown = arena_strdup(a, err);
                free(err);
                goto out;
            }
            JValue *v = js_parse(a, str(res));
            free(res);
            const JValue *cs = json_get(v, "contents");
            if (!is_record(v) || !json_is(cs, J_ARRAY)) {
                *thrown = "Invalid MCP resources/read result";
                goto out;
            }
            for (size_t i = 0; i < cs->arr.len; i++) {
                const JValue *c = cs->arr.items[i];
                if (!is_record(c) || !json_is(json_get(c, "uri"), J_STRING) || (!json_is(json_get(c, "text"), J_STRING) && !json_is(json_get(c, "blob"), J_STRING))) {
                    *thrown = "Invalid contents in MCP resources/read result";
                    goto out;
                }
            }
            JValue *conv = json_new(a, J_ARRAY);
            for (size_t i = 0; i < cs->arr.len; i++) {
                if (cs->arr.len > 1) json_push(a, conv, text_block(a, str(arena_printf(a, "%s:", json_str(json_get(cs->arr.items[i], "uri"))))));
                JValue *block = json_new(a, J_OBJECT);
                jput(a, block, "type", jstr(a, "resource"));
                jput(a, block, "resource", cs->arr.items[i]);
                block_content(a, server->name, block, false, conv);
            }
            if (!conv->arr.len) json_push(a, conv, text_block(a, str(arena_printf(a, "Resource %s is empty.", uri))));
            char *full = NULL;
            *content = limit_content(a, conv, &full);
            *details = details_of(a, server->name, tool, full);
            JValue *sc = json_new(a, J_OBJECT);
            jput(a, sc, "server", jstr(a, server->name));
            jput(a, sc, "uri", jstr(a, uri));
            JValue *contents = json_new(a, J_ARRAY);
            for (size_t i = 0; i < cs->arr.len; i++) {
                JValue *c = obj_spread(a, cs->arr.items[i]);
                obj_del(c, "_meta");
                json_push(a, contents, c);
            }
            jput(a, sc, "contents", contents);
            *structured = sc;
            goto out;
        }
    }
    // the listings: one page of one server, or every page of every server
    {
        bool templates = r->kind == REG_RESOURCES_TEMPLATES;
        const char *key = templates ? "resourceTemplates" : "resources";
        const char *method = templates ? "resources/templates/list" : "resources/list";
        const char *cursor = string_arg(a, args, "cursor", thrown);
        if (*thrown) goto out;
        JValue *payload = json_new(a, J_OBJECT);
        if (server) {
            JValue *params = NULL;
            if (cursor) {
                params = json_new(a, J_OBJECT);
                jput(a, params, "cursor", jstr(a, cursor));
            }
            char *err = NULL;
            int code = 0;
            char *res = res_request(server->conn, method, params, server->timeout, env->abort, &err, &code);
            JValue *items = json_new(a, J_ARRAY);
            const char *next = NULL;
            if (!res && templates && code == -32601) { // withoutTemplates: no templates
                free(err);
                err = NULL;
            } else if (!res) {
                *thrown = arena_strdup(a, err);
                free(err);
                goto out;
            } else {
                JValue *v = js_parse(a, str(res));
                free(res);
                const JValue *arr = json_get(v, key);
                if (!is_record(v) || !json_is(arr, J_ARRAY)) {
                    *thrown = arena_printf(a, "Invalid MCP %s result", method);
                    goto out;
                }
                for (size_t i = 0; i < arr->arr.len; i++)
                    if (is_record(arr->arr.items[i]) && !app_resource(arr->arr.items[i])) json_push(a, items, listed_item(a, server->name, arr->arr.items[i]));
                const JValue *nc = json_get(v, "nextCursor");
                if (json_is(nc, J_STRING) && nc->s.len) next = nc->s.ptr;
            }
            jput(a, payload, "server", jstr(a, server->name));
            jput(a, payload, key, items);
            if (next) jput(a, payload, "nextCursor", jstr(a, next));
        } else {
            if (cursor) {
                *thrown = "cursor can only be used when a server is specified";
                goto out;
            }
            // sorted by name
            for (size_t i = 1; i < servers.len; i++)
                for (size_t k = i; k > 0 && locale_cmp(servers.ptr[k - 1].name, servers.ptr[k].name) > 0; k--) {
                    ResServer tmp = servers.ptr[k];
                    servers.ptr[k] = servers.ptr[k - 1];
                    servers.ptr[k - 1] = tmp;
                }
            JValue *items = json_new(a, J_ARRAY), *errors = json_new(a, J_ARRAY);
            for (size_t i = 0; i < servers.len; i++) {
                char *err = NULL;
                int code = 0;
                char *all = NULL;
                Client *cl = conn_get_client(servers.ptr[i].conn, &err);
                if (cl) {
                    all = client_list_all(cl, method, key, servers.ptr[i].timeout, env->abort, &err, &code);
                    client_unref(cl);
                }
                if (!all && templates && code == -32601) {
                    free(err);
                    continue;
                }
                if (!all) {
                    JValue *e = json_new(a, J_OBJECT);
                    jput(a, e, "server", jstr(a, servers.ptr[i].name));
                    jput(a, e, "error", jstr(a, err));
                    json_push(a, errors, e);
                    free(err);
                    continue;
                }
                JValue *arr = js_parse(a, str(all));
                free(all);
                for (size_t k = 0; json_is(arr, J_ARRAY) && k < arr->arr.len; k++)
                    if (!app_resource(arr->arr.items[k])) json_push(a, items, listed_item(a, servers.ptr[i].name, arr->arr.items[k]));
            }
            jput(a, payload, key, items);
            if (errors->arr.len) jput(a, payload, "errors", errors);
        }
        // jsonResult
        JValue *conv = json_new(a, J_ARRAY);
        Buf b = {0};
        code_oai_js_write(&b, payload);
        json_push(a, conv, text_block(a, str_n(arena_strndup(a, b.ptr, b.len), b.len)));
        buf_free(&b);
        char *full = NULL;
        *content = limit_content(a, conv, &full);
        *details = details_of(a, server_name ? server_name : "", tool, full);
        *structured = payload;
    }
out:
    for (size_t i = 0; i < servers.len; i++) free(servers.ptr[i].name);
    vec_free(&servers);
}

// ---------------------------------------------------------------------------------------
// the manager's actions (index.ts saveConfig, setEnabled, setExposure, reconnect, runAction)
// ---------------------------------------------------------------------------------------

// saveConfig: the change saved to the server's mcp.json (inProject: as an override in the
// project's), then applied to the entry. NULL or the message.
static char *save_config_l(Server *s, int enabled, const char *exposure, bool in_project) {
    const char *override = in_project ? G.project_config : s->override;
    const char *path = override ? override : s->source;
    char *e = update_server_config(path, s->name, enabled, exposure, override != NULL);
    if (e) {
        char *m = mfmt("Could not update %s: %s", path, e);
        free(e);
        return m;
    }
    if (override && (!s->override || strcmp(s->override, override))) set_str(&s->override, strdup(override));
    JValue *c = obj_spread(&G.arena, s->config);
    if (enabled >= 0) jput(&G.arena, c, "enabled", json_bool(&G.arena, enabled));
    if (exposure) jput(&G.arena, c, "exposure", jstr(&G.arena, exposure));
    s->config = c;
    return NULL;
}

typedef struct {
    char *name, *action;
} ActionArg;

static void *action_main(void *arg) {
    ActionArg *a = arg;
    char *message = NULL;
    pthread_mutex_lock(&g_mu);
    Server *s = find_server_l(a->name);
    if (!s) {
        pthread_mutex_unlock(&g_mu);
        goto out;
    }
    uint64_t attempt;
    if (!strcmp(a->action, "signin") || !strcmp(a->action, "signout")) {
        // OAuth sign-in (oauth.ts) is not ported yet
        attempt = s->attempt;
        message = strdup("Sign-in failed: Saetta Code does not support MCP OAuth sign-in yet.");
    } else if (!strcmp(a->action, "reconnect")) {
        // reconnect: the connection state and error report failures
        attempt = s->attempt;
        Conn *c = s->conn;
        if (c) {
            s->inflight++;
            pthread_mutex_unlock(&g_mu);
            free(conn_reconnect(c));
            pthread_mutex_lock(&g_mu);
            s->inflight--;
        }
    } else {
        // setEnabled
        bool enable = !strncmp(a->action, "enable", 6);
        message = save_config_l(s, enable, NULL, strstr(a->action, "-project") != NULL);
        if (!message && !enable) {
            Conn *c = s->conn;
            s->attempt = 0;
            s->conn = NULL;
            hide_tools_l(s->name);
            changed_l();
            attempt = s->attempt;
            if (c) {
                s->closing++;
                pthread_mutex_unlock(&g_mu);
                conn_close(c);
                pthread_mutex_lock(&g_mu);
                while (s->inflight > 0 && !G.shutdown) timed_wait_l(now_ns() + 50000000ull);
                s->closing--;
            }
        } else if (!message) {
            start_connection_l(s, false);
            attempt = s->attempt;
            while (s->inflight > 0 && s->attempt == attempt && !G.shutdown) timed_wait_l(now_ns() + 50000000ull);
        } else attempt = s->attempt;
    }
    // runInBackground's finish: the message, unless the server changed meanwhile
    if (!G.shutdown && server_listed_l(s) && s->attempt == attempt) {
        set_str(&s->message, message);
        message = NULL;
    }
    changed_l();
    pthread_mutex_unlock(&g_mu);
out:
    free(message);
    free(a->name);
    free(a->action);
    free(a);
    bg_end();
    return NULL;
}

void code_mcp_run_action(const char *name, const char *action) {
    pthread_mutex_lock(&g_mu);
    Server *s = find_server_l(name);
    if (s) {
        // runAction: the message of the last action cleared, the background work started
        set_str(&s->message, NULL);
        ActionArg *a = xcalloc(1, sizeof *a);
        a->name = strdup(name);
        a->action = strdup(action);
        bg_begin_l();
        spawn_detached(action_main, a);
        changed_l();
    }
    pthread_mutex_unlock(&g_mu);
}

// chooseExposure + setExposure
void code_mcp_choose_exposure(const char *name, const char *exposure) {
    pthread_mutex_lock(&g_mu);
    Server *s = find_server_l(name);
    if (s) {
        char *message = NULL;
        if (exposure && strcmp(exposure, exposure_of(s->config))) {
            message = save_config_l(s, -1, exposure, false);
            if (!message) {
                if (s->conn && s->conn->state == ST_CONNECTED) register_tools_l(s->conn);
                sync_resource_tools_l();
            }
        }
        set_str(&s->message, message);
        changed_l();
    }
    pthread_mutex_unlock(&g_mu);
}

// ---- menus ----

static void menu_item(McpMenu *m, const char *value, const char *label, char *description) {
    m->items = xrealloc(m->items, (m->n + 1) * sizeof *m->items);
    m->items[m->n++] = (McpMenuItem){strdup(value), strdup(label), description};
}

void code_mcp_menu_free(McpMenu *m) {
    free(m->title);
    free(m->details);
    free(m->error);
    free(m->empty);
    free(m->selected);
    free(m->confirm_label);
    free(m->cancel_label);
    for (size_t i = 0; i < m->n; i++) {
        free(m->items[i].value);
        free(m->items[i].label);
        free(m->items[i].description);
    }
    free(m->items);
    memset(m, 0, sizeof *m);
}

static int server_order(const void *x, const void *y) {
    const Server *a = *(Server *const *)x, *b = *(Server *const *)y;
    int d = attention_rank(a) - attention_rank(b);
    return d ? d : locale_cmp(a->name, b->name);
}

static void menu_servers_l(McpMenu *m) {
    memset(m, 0, sizeof *m);
    m->title = strdup("MCP servers");
    Buf e = {0};
    for (size_t i = 0; i < G.config_errors.len; i++) buf_printf(&e, "%sconfig: %s", e.len ? "\n" : "", G.config_errors.ptr[i]);
    m->error = e.len ? buf_take(&e) : NULL;
    Server **sorted = xcalloc(G.servers.len + 1, sizeof *sorted);
    memcpy(sorted, G.servers.ptr, G.servers.len * sizeof *sorted);
    // a stable sort (Array.prototype.sort)
    for (size_t i = 1; i < G.servers.len; i++)
        for (size_t k = i; k > 0 && server_order(&sorted[k - 1], &sorted[k]) > 0; k--) {
            Server *t = sorted[k];
            sorted[k] = sorted[k - 1];
            sorted[k - 1] = t;
        }
    for (size_t i = 0; i < G.servers.len; i++) {
        Server *s = sorted[i];
        char *st = describe_state_l(s, true);
        menu_item(m, s->name, s->name, mfmt("%s \xc2\xb7 %s \xc2\xb7 %s", st, exposure_of(s->config), s->override ? "global, project override" : s->scope));
        free(st);
    }
    free(sorted);
    m->empty = mfmt("No MCP servers configured. Add them to %s/mcp.json or .pi/mcp.json.", G.agent_dir);
    m->confirm_label = strdup("manage");
    m->cancel_label = strdup("close");
}

// describeTransport
static char *describe_transport(const JValue *config) {
    if (is_http(config)) return strdup(json_str(json_get(config, "url")));
    Buf b = {0};
    buf_adds(&b, json_str(json_get(config, "command")));
    const JValue *args = json_get(config, "args");
    for (size_t i = 0; json_is(args, J_ARRAY) && i < args->arr.len; i++) {
        buf_addc(&b, ' ');
        buf_addstr(&b, json_strs(args->arr.items[i]));
    }
    return buf_take(&b);
}

static void menu_server_l(const char *name, McpMenu *m) {
    memset(m, 0, sizeof *m);
    Server *s = find_server_l(name);
    if (!s) {
        m->title = strdup(name);
        m->empty = strdup("This server is no longer configured.");
        m->confirm_label = strdup("");
        m->cancel_label = strdup("back");
        return;
    }
    const char *saved = s->override ? "saved to the project mcp.json" : !strcmp(s->scope, "global") ? "saved to the global mcp.json" : "saved to the project mcp.json";
    bool in_project = !strcmp(s->scope, "global") && !s->override && G.project_config;
    const char *ips = "saved to the project mcp.json";
    Conn *c = s->conn;
    if (!is_enabled(s)) {
        menu_item(m, "enable", "Enable", strdup(saved));
        if (in_project) menu_item(m, "enable-project", "Enable in this project", strdup(ips));
    } else {
        ConnState st = c ? c->state : (ConnState)-1;
        if (c && st == ST_NEEDS_AUTH) menu_item(m, "signin", "Sign in", strdup("opens the browser"));
        if (c && st == ST_CONNECTED) menu_item(m, "tools", "Tools", mfmt("%zu offered", c->tools.len));
        if (c && (st == ST_FAILED || st == ST_DISCONNECTED || st == ST_CONNECTED || st == ST_NEEDS_AUTH)) menu_item(m, "reconnect", "Reconnect", NULL);
        menu_item(m, "exposure", "Exposure", strdup(exposure_of(s->config)));
        menu_item(m, "disable", "Disable", strdup(saved));
        if (in_project) menu_item(m, "disable-project", "Disable in this project", strdup(ips));
    }
    char *tr = describe_transport(s->config);
    char *st = describe_state_l(s, false);
    m->details = s->override ? mfmt("%s\n%s: %s\nproject override: %s\nState: %s", tr, s->scope, s->source, s->override, st)
                             : mfmt("%s\n%s: %s\nState: %s", tr, s->scope, s->source, st);
    free(tr);
    free(st);
    Buf e = {0};
    if (s->message) buf_adds(&e, s->message);
    if (c && c->state != ST_CONNECTED && c->error) buf_printf(&e, "%s%s", e.len ? "\n" : "", c->error);
    m->error = e.len ? buf_take(&e) : NULL;
    m->title = mfmt("MCP server %s", name);
    m->selected = m->n ? strdup(m->items[0].value) : NULL;
    m->confirm_label = strdup("select");
    m->cancel_label = strdup("back");
}

static const char *exposure_description(const char *e) {
    if (!strcmp(e, "codemode")) return "called from codemode scripts, which find them with searchTools()";
    if (!strcmp(e, "deferred")) return "not declared until tool_search loads them, then called directly; no codemode needed";
    if (!strcmp(e, "direct")) return "declared to the model like built-in tools";
    return "unreachable";
}

static void menu_tools_l(const char *name, McpMenu *m) {
    memset(m, 0, sizeof *m);
    Server *s = find_server_l(name);
    const char *exposure = s ? exposure_of(s->config) : "codemode";
    const JValue *te = s ? json_get(s->config, "toolExposure") : NULL;
    bool overridden = is_record(te) && te->obj.len > 0;
    m->title = mfmt("Tools of %s", name);
    m->details = mfmt("Exposure %s: %s%s", exposure, exposure_description(exposure), overridden ? "\nSome tools override it with toolExposure." : "");
    Arena t = {0};
    for (size_t i = 0; s && s->conn && i < s->conn->tools.len; i++) {
        const char *tn = s->conn->tools.ptr[i].name;
        JValue *tool = js_parse(&t, str(s->conn->tools.ptr[i].json));
        const char *tx = tool_exposure(s->config, tn);
        Str d = first_line(json_strs(json_get(tool, "description")));
        menu_item(m, tn, tn, !strcmp(tx, exposure) ? strndup(d.ptr, d.len) : mfmt("[%s] %.*s", tx, (int)d.len, d.ptr));
    }
    arena_reset(&t);
    m->empty = strdup("The server offers no tools.");
    m->confirm_label = strdup("back");
    m->cancel_label = strdup("back");
}

static void menu_exposure_l(const char *name, McpMenu *m) {
    memset(m, 0, sizeof *m);
    Server *s = find_server_l(name);
    const char *current = s ? exposure_of(s->config) : "codemode";
    m->title = mfmt("Exposure of %s", name);
    m->details = mfmt("Saved to %s.", s ? (s->override ? s->override : s->source) : "mcp.json");
    static const char *const opts[] = {"codemode", "deferred", "direct"};
    for (size_t i = 0; i < COUNT(opts); i++) {
        char *label = mfmt("%s%s", !strcmp(opts[i], current) ? "\xe2\x9c\x93 " : "  ", opts[i]);
        menu_item(m, opts[i], label, strdup(exposure_description(opts[i])));
        free(label);
    }
    m->selected = strdup(current);
    m->confirm_label = strdup("save");
    m->cancel_label = strdup("back");
}

// ---- the subcommands (index.ts handler: /mcp login|logout|reconnect [server]) ----

typedef struct {
    char *name;
} ReconnectArg;

static void *reconnect_cmd_main(void *arg) {
    ReconnectArg *a = arg;
    pthread_mutex_lock(&g_mu);
    Server *s = find_server_l(a->name);
    Conn *c = s ? s->conn : NULL;
    if (s && c) {
        s->inflight++;
        pthread_mutex_unlock(&g_mu);
        char *e = conn_reconnect(c);
        pthread_mutex_lock(&g_mu);
        s->inflight--;
        if (e) notify_l(2, e);
        else {
            char *d = describe_state_l(s, true);
            notify_l(0, mfmt("Reconnected to MCP server \"%s\" (%s).", s->name, d));
            free(d);
        }
        changed_l();
    } else if (s) notify_l(2, mfmt("MCP server \"%s\" is disabled.", a->name));
    pthread_mutex_unlock(&g_mu);
    free(a->name);
    free(a);
    bg_end();
    return NULL;
}

char **code_mcp_subcommand(const char *args, size_t *npick) {
    *npick = 0;
    char **pick = NULL;
    // split on whitespace
    char *copy = strdup(args ? args : "");
    char *words[4] = {0};
    int n = 0;
    for (char *save = NULL, *w = strtok_r(copy, " \t\n", &save); w; w = strtok_r(NULL, " \t\n", &save))
        if (n < 4) words[n++] = w;
        else n++;
    const char *usage = "Usage: /mcp, /mcp login [server], /mcp logout [server], /mcp reconnect [server]";
    if (n > 2) {
        code_mcp_notify(1, usage);
        free(copy);
        return NULL;
    }
    const char *action = words[0], *name = words[1];
    pthread_mutex_lock(&g_mu);
    if (!strcmp(action, "login") || !strcmp(action, "logout")) {
        // only HTTP servers without an Authorization header use OAuth, which Saetta Code does not
        // support yet: no server is eligible
        const char *none = "No enabled MCP server uses OAuth. Only HTTP servers without an Authorization header do.";
        if (name) {
            if (!find_server_l(name)) notify_l(2, mfmt("No MCP server named \"%s\".", name));
            else notify_l(2, strdup(none));
        } else notify_l(0, strdup(none));
    } else if (!strcmp(action, "reconnect")) {
        Server *s = NULL;
        if (name) {
            s = find_server_l(name);
            if (!s) notify_l(2, mfmt("No MCP server named \"%s\".", name));
            else if (!s->conn) {
                notify_l(2, strdup("No enabled MCP server to reconnect."));
                s = NULL;
            }
        } else {
            // pickServer: the only candidate, or the only failed or disconnected one
            Server *only = NULL, *pref = NULL;
            int nc = 0, np = 0;
            for (size_t i = 0; i < G.servers.len; i++) {
                Server *x = G.servers.ptr[i];
                if (!x->conn) continue;
                nc++;
                only = x;
                if (x->conn->state == ST_FAILED || x->conn->state == ST_DISCONNECTED) {
                    np++;
                    pref = x;
                }
            }
            if (!nc) notify_l(0, strdup("No enabled MCP server to reconnect."));
            else if (nc == 1) s = only;
            else if (np == 1) s = pref;
            else {
                // ctx.ui.select("MCP server", the candidates)
                pick = xcalloc((size_t)nc + 1, sizeof(char *));
                for (size_t i = 0; i < G.servers.len; i++)
                    if (G.servers.ptr[i]->conn) pick[(*npick)++] = strdup(G.servers.ptr[i]->name);
            }
        }
        if (s) {
            ReconnectArg *a = xcalloc(1, sizeof *a);
            a->name = strdup(s->name);
            bg_begin_l();
            spawn_detached(reconnect_cmd_main, a);
        }
    } else notify_l(1, strdup(usage));
    pthread_mutex_unlock(&g_mu);
    free(copy);
    return pick;
}

// ---- the open view (McpManagerView.menu: the menu rebuilt on every change, its selection kept) ----

static struct {
    McpView kind;
    char *server;
    char *selected; // `selected`: the item the user moved to, or the one the last render chose
} MV;

static void view_menu_l(McpMenu *m) {
    switch (MV.kind) {
    case MCP_VIEW_SERVERS: menu_servers_l(m); break;
    case MCP_VIEW_SERVER: menu_server_l(MV.server, m); break;
    case MCP_VIEW_TOOLS: menu_tools_l(MV.server, m); break;
    case MCP_VIEW_EXPOSURE: menu_exposure_l(MV.server, m); break;
    default: memset(m, 0, sizeof *m);
    }
}

// render(): the wanted item (the kept one, else the menu's own) selected, else the first; the
// selection kept for the next render
static size_t view_resolve_l(McpMenu *m) {
    view_menu_l(m);
    const char *want = MV.selected ? MV.selected : m->selected;
    size_t sel = 0;
    for (size_t i = 0; want && i < m->n; i++)
        if (!strcmp(m->items[i].value, want)) {
            sel = i;
            break;
        }
    if (m->n) set_str(&MV.selected, strdup(m->items[sel].value));
    return sel;
}

static void view_changed_l(void) {
    if (MV.kind == MCP_VIEW_NONE) return;
    McpMenu m;
    view_resolve_l(&m);
    code_mcp_menu_free(&m);
}

void code_mcp_view_set(McpView kind, const char *server) {
    pthread_mutex_lock(&g_mu);
    char *sv = server ? strdup(server) : NULL;
    free(MV.server);
    MV.server = sv;
    MV.kind = kind;
    set_str(&MV.selected, NULL);
    view_changed_l();
    pthread_mutex_unlock(&g_mu);
}

McpView code_mcp_view(char **server) {
    pthread_mutex_lock(&g_mu);
    McpView k = MV.kind;
    if (server) *server = dupz(MV.server);
    pthread_mutex_unlock(&g_mu);
    return k;
}

size_t code_mcp_view_menu(McpMenu *m) {
    pthread_mutex_lock(&g_mu);
    size_t sel = view_resolve_l(m);
    pthread_mutex_unlock(&g_mu);
    // what the servers sent (names, descriptions, errors, stderr) never reaches the terminal raw
    char **multi[] = {&m->title, &m->details, &m->error, &m->empty};
    for (size_t i = 0; i < COUNT(multi); i++) *multi[i] = term_safe(*multi[i], true);
    for (size_t i = 0; i < m->n; i++) {
        m->items[i].label = term_safe(m->items[i].label, false);
        m->items[i].description = term_safe(m->items[i].description, true);
    }
    return sel;
}

void code_mcp_view_select(const char *value) {
    pthread_mutex_lock(&g_mu);
    set_str(&MV.selected, dupz(value));
    pthread_mutex_unlock(&g_mu);
}
