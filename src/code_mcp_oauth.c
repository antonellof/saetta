// Saetta Code's OAuth sign-in for Streamable HTTP MCP servers (see code_mcp_oauth.h).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at v1.5.2,
// b4afddd): packages/mcp/src/oauth/*.ts (adapted there from modelcontextprotocol/typescript-sdk
// v1.29.0 src/client/auth.ts, MIT, Copyright (c) 2024 Anthropic, PBC), packages/coding-agent/src/
// extensions/mcp/oauth.ts (the credential store, pi's client identity, signInMcpServer) and
// packages/ai/src/utils/oauth-page.ts (the browser page after the redirect).
//
// Saetta Code is a rebranded build in pi's terms (its client identifies as "saetta-code"), so a
// clientRegistration of "auto" registers a client dynamically (oauth.ts piSettings); "cimd" uses
// pi's Client ID Metadata Documents on pi.dev, as a rebranded pi does.
#include "code_mcp_oauth.h"
#include "code_cfg.h"
#include "code_oai.h"
#include "hash.h"
#include "json.h"
#include "metal/shim.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <spawn.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

#define APP_NAME "saetta-code"
#define PROTOCOL_VERSION "2025-11-25" // LATEST_PROTOCOL_VERSION
#define CIMD_BASE "https://pi.dev/oauth" // PI_CLIENT_METADATA_DOCUMENTS.baseUrl
#define DEFAULT_REDIRECT_URL "http://127.0.0.1/callback"
enum { REFRESH_SKEW_MS = 30000, OAUTH_REQUEST_TIMEOUT_S = 15, CALLBACK_TIMEOUT_MS = 5 * 60000 };

// ---------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------

static char *mfmt(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static char *mfmt(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    Buf b = {0};
    buf_vprintf(&b, fmt, ap);
    va_end(ap);
    return b.ptr ? b.ptr : strdup("");
}

static double now_ms(void) { return (double)(wall_ns() / 1000000ull); }

// errors: the error classes the flow tells apart (errors.ts, sign-in.ts, fetch's TypeError)
typedef enum { OE_NONE, OE_ERROR, OE_NETWORK, OE_OAUTH, OE_INSECURE, OE_ISSUER, OE_REGISTRATION, OE_CANCELLED, OE_REQUIRED, OE_LOCK_BUSY } OKind;
typedef struct {
    OKind kind;
    char *code; // OAuthError.code
    char *msg;
} OErr;

static bool fail(OErr *e, OKind k, char *msg) {
    free(e->msg);
    free(e->code);
    e->kind = k;
    e->code = NULL;
    e->msg = msg ? msg : strdup("");
    return false;
}

static void oerr_clear(OErr *e) {
    free(e->msg);
    free(e->code);
    memset(e, 0, sizeof *e);
}

// ---- JSON (JavaScript values: code_oai_js_parse's numbers are J_NUMBER_STRING) --------------------

static JValue *jo(Arena *a) { return json_new(a, J_OBJECT); }
static JValue *js(Arena *a, const char *s) { return json_string(a, str(s)); }
static void jset(Arena *a, JValue *o, const char *k, JValue *v) { json_put(a, o, str(k), v); }
static const char *jstrz(const JValue *o, const char *k) { return json_str(json_get(o, k)); }
static bool is_obj(const JValue *v) { return json_is(v, J_OBJECT); }

static void jdel(JValue *o, const char *k) {
    if (!is_obj(o)) return;
    for (size_t i = 0; i < o->obj.len; i++)
        if (str_eqz(o->obj.items[i].key, k)) {
            memmove(&o->obj.items[i], &o->obj.items[i + 1], (o->obj.len - i - 1) * sizeof o->obj.items[0]);
            o->obj.len--;
            return;
        }
}

// { ...o }
static JValue *jspread(Arena *a, const JValue *o) {
    JValue *c = jo(a);
    for (size_t i = 0; is_obj(o) && i < o->obj.len; i++) json_put(a, c, o->obj.items[i].key, o->obj.items[i].val);
    return c;
}

static JValue *jnum(Arena *a, double x) {
    JValue *v = json_new(a, J_NUMBER_STRING);
    char t[64];
    if (x == trunc(x) && fabs(x) < 9e15) snprintf(t, sizeof t, "%.0f", x);
    else snprintf(t, sizeof t, "%.17g", x);
    v->s = str(arena_strdup(a, t));
    return v;
}

static bool jnumber(const JValue *v, double *x) {
    if (json_is(v, J_INTEGER)) *x = (double)v->i;
    else if (json_is(v, J_FLOAT)) *x = v->f;
    else if (json_is(v, J_NUMBER_STRING)) *x = strtod(v->s.ptr, NULL);
    else return false;
    return true;
}

static JValue *jparse(Arena *a, Str text, OErr *e) {
    char *err = NULL;
    JValue *v = code_oai_js_parse(a, text, &err);
    if (!v) fail(e, OE_ERROR, strdup(err ? err : "Unexpected end of JSON input"));
    return v;
}

static char *jtext(const JValue *v) {
    Buf b = {0};
    if (v) code_oai_js_write(&b, v);
    else buf_adds(&b, "null");
    return b.ptr ? b.ptr : strdup("");
}

static bool str_in(const JValue *arr, const char *s) {
    for (size_t i = 0; json_is(arr, J_ARRAY) && i < arr->arr.len; i++)
        if (json_is(arr->arr.items[i], J_STRING) && !strcmp(arr->arr.items[i]->s.ptr, s)) return true;
    return false;
}

// ---- URLs (the parts of WHATWG URL the flow uses) ---------------------------------------------------

typedef struct {
    char *scheme, *host, *port, *path, *query, *hash; // host as URL.hostname ([::1] bracketed); query without "?"
    bool has_query, has_hash;
} OUrl;

static bool ourl_parse(Arena *a, const char *s, OUrl *u) {
    memset(u, 0, sizeof *u);
    if (!s) return false;
    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') s++;
    const char *c = strstr(s, "://");
    if (!c || c == s) return false;
    for (const char *p = s; p < c; p++) {
        char ch = *p;
        if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (p > s && ((ch >= '0' && ch <= '9') || ch == '+' || ch == '-' || ch == '.'))))
            return false;
    }
    u->scheme = arena_strndup(a, s, (size_t)(c - s));
    for (char *p = u->scheme; *p; p++) *p = (char)(*p | 0x20);
    const char *h = c + 3;
    size_t hl = strcspn(h, "/?#");
    char *auth = arena_strndup(a, h, hl);
    char *at = strrchr(auth, '@');
    char *hp = at ? at + 1 : auth;
    char *colon = NULL;
    if (hp[0] == '[') {
        char *rb = strchr(hp, ']');
        if (!rb) return false;
        if (rb[1] == ':') colon = rb + 1;
        else if (rb[1]) return false;
    } else colon = strrchr(hp, ':');
    u->port = "";
    if (colon) {
        *colon = 0;
        for (char *p = colon + 1; *p; p++)
            if (*p < '0' || *p > '9') return false;
        if (strlen(colon + 1) > 5 || atoi(colon + 1) > 65535) return false;
        u->port = colon[1] ? arena_printf(a, "%d", atoi(colon + 1)) : "";
    }
    if (!*hp) return false;
    for (char *p = hp; *p; p++) {
        if (*p >= 'A' && *p <= 'Z') *p = (char)(*p + 32);
        if (*p == ' ' || *p == '<' || *p == '>' || *p == '\\' || *p == '^' || *p == '|' || (unsigned char)*p < 0x20) return false;
    }
    u->host = hp;
    const char *rest = h + hl;
    size_t pl = strcspn(rest, "?#");
    u->path = pl ? arena_strndup(a, rest, pl) : "/";
    rest += pl;
    u->query = "";
    if (*rest == '?') {
        size_t ql = strcspn(rest + 1, "#");
        u->query = arena_strndup(a, rest + 1, ql);
        u->has_query = ql > 0;
        rest += 1 + ql;
    }
    u->hash = "";
    if (*rest == '#') {
        u->hash = arena_strdup(a, rest + 1);
        u->has_hash = rest[1] != 0;
    }
    if ((!strcmp(u->scheme, "http") && !strcmp(u->port, "80")) || (!strcmp(u->scheme, "https") && !strcmp(u->port, "443"))) u->port = "";
    return true;
}

static char *ourl_origin(Arena *a, const OUrl *u) { return arena_printf(a, "%s://%s%s%s", u->scheme, u->host, *u->port ? ":" : "", u->port); }

static char *ourl_href(Arena *a, const OUrl *u) {
    return arena_printf(a, "%s%s%s%s%s%s", ourl_origin(a, u), u->path, u->has_query ? "?" : "", u->has_query ? u->query : "", u->has_hash ? "#" : "",
                        u->has_hash ? u->hash : "");
}

// String(new URL(s)), NULL when s does not parse
static char *href_of(Arena *a, const char *s) {
    OUrl u;
    return ourl_parse(a, s, &u) ? ourl_href(a, &u) : NULL;
}

static bool is_http_url(const OUrl *u) { return !strcmp(u->scheme, "http") || !strcmp(u->scheme, "https"); }

// flow.ts loopback(): URL.hostname of a loopback address
static bool loopback_hostname(const char *h) { return !strcmp(h, "localhost") || !strcmp(h, "127.0.0.1") || !strcmp(h, "[::1]") || !strcmp(h, "::1"); }

// pathSuffix: the path without its trailing slash ("" for the root)
static char *path_suffix(Arena *a, const char *p) {
    size_t n = strlen(p);
    return arena_strndup(a, p, n && p[n - 1] == '/' ? n - 1 : n);
}

// application/x-www-form-urlencoded (URLSearchParams): the pairs, in order
typedef struct {
    VEC(char *) k;
    VEC(char *) v;
} Params;

static void params_free(Params *p) {
    vec_free(&p->k);
    vec_free(&p->v);
}

static void params_set(Params *p, const char *k, const char *v) {
    for (size_t i = 0; i < p->k.len; i++)
        if (!strcmp(p->k.ptr[i], k)) {
            p->v.ptr[i] = (char *)v;
            // URLSearchParams.set drops later pairs of the same name
            for (size_t j = p->k.len; j-- > i + 1;)
                if (!strcmp(p->k.ptr[j], k)) {
                    memmove(&p->k.ptr[j], &p->k.ptr[j + 1], (p->k.len - j - 1) * sizeof(char *));
                    memmove(&p->v.ptr[j], &p->v.ptr[j + 1], (p->v.len - j - 1) * sizeof(char *));
                    p->k.len--;
                    p->v.len--;
                }
            return;
        }
    vec_push(&p->k, (char *)k);
    vec_push(&p->v, (char *)v);
}

static const char *params_get(const Params *p, const char *k) {
    for (size_t i = 0; i < p->k.len; i++)
        if (!strcmp(p->k.ptr[i], k)) return p->v.ptr[i];
    return NULL;
}

static void form_encode(Buf *b, const char *s) {
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned char c = *p;
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '*' || c == '-' || c == '.' || c == '_') buf_addc(b, (char)c);
        else if (c == ' ') buf_addc(b, '+');
        else buf_printf(b, "%%%02X", c);
    }
}

static char *params_text(Arena *a, const Params *p) {
    Buf b = {0};
    for (size_t i = 0; i < p->k.len; i++) {
        if (i) buf_addc(&b, '&');
        form_encode(&b, p->k.ptr[i]);
        buf_addc(&b, '=');
        form_encode(&b, p->v.ptr[i]);
    }
    char *s = arena_strdup(a, b.ptr ? b.ptr : "");
    buf_free(&b);
    return s;
}

static int hexv(char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; }

static char *form_decode(Arena *a, const char *s, size_t n) {
    char *o = arena_alloc(a, n + 1);
    size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '+') o[k++] = ' ';
        else if (s[i] == '%' && i + 2 < n && hexv(s[i + 1]) >= 0 && hexv(s[i + 2]) >= 0) {
            o[k++] = (char)(hexv(s[i + 1]) * 16 + hexv(s[i + 2]));
            i += 2;
        } else o[k++] = s[i];
    }
    o[k] = 0;
    return o;
}

static void params_parse(Arena *a, const char *q, Params *p) {
    memset(p, 0, sizeof *p);
    while (q && *q) {
        size_t n = strcspn(q, "&");
        if (n) {
            const char *eq = memchr(q, '=', n);
            size_t kn = eq ? (size_t)(eq - q) : n;
            vec_push(&p->k, form_decode(a, q, kn));
            vec_push(&p->v, eq ? form_decode(a, eq + 1, n - kn - 1) : arena_strdup(a, ""));
        }
        q += n;
        if (*q == '&') q++;
    }
}

// ---- base64 ----

static void b64(Buf *b, const uint8_t *p, size_t n, bool url) {
    const char *al = url ? "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
                         : "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = (uint32_t)p[i] << 16 | (i + 1 < n ? (uint32_t)p[i + 1] << 8 : 0) | (i + 2 < n ? p[i + 2] : 0);
        buf_addc(b, al[v >> 18 & 63]);
        buf_addc(b, al[v >> 12 & 63]);
        if (i + 1 < n) buf_addc(b, al[v >> 6 & 63]);
        else if (!url) buf_addc(b, '=');
        if (i + 2 < n) buf_addc(b, al[v & 63]);
        else if (!url) buf_addc(b, '=');
    }
}

static char *b64z(Arena *a, const void *p, size_t n, bool url) {
    Buf b = {0};
    b64(&b, p, n, url);
    char *s = arena_strdup(a, b.ptr ? b.ptr : "");
    buf_free(&b);
    return s;
}

// ---------------------------------------------------------------------------------------
// requests to the authorization server (timedFetch: 15 s each)
// ---------------------------------------------------------------------------------------

typedef struct {
    int status;
    Buf body;
} OResp;

static int resp_sink(void *ctx, int status, const void *p, size_t n) {
    OResp *r = ctx;
    (void)status;
    if (r->body.len < (4u << 20)) buf_add(&r->body, p, n);
    return 0;
}

static bool ofetch(volatile int *cancel, const char *method, const char *url, const sx_header *hs, size_t nh, Str body, OResp *r, OErr *e) {
    memset(r, 0, sizeof *r);
    char err[256] = {0};
    volatile int never = 0;
    int rc = sx_http_ex(method, url, hs, nh, body.len ? body.ptr : NULL, body.len, OAUTH_REQUEST_TIMEOUT_S, NULL, resp_sink, r, &r->status, err,
                        sizeof err, cancel ? cancel : &never);
    if (rc) {
        buf_free(&r->body);
        if (cancel && *cancel) return fail(e, OE_CANCELLED, strdup("Sign-in cancelled"));
        return fail(e, OE_NETWORK, strdup("fetch failed")); // fetch's TypeError
    }
    return true;
}

static bool resp_ok(const OResp *r) { return r->status >= 200 && r->status < 300; }

// ---------------------------------------------------------------------------------------
// the credential store (oauth.ts McpOAuthCredentialStore over FileAuthStorageBackend)
// ---------------------------------------------------------------------------------------

struct McpAuth {
    char *agent_dir, *path;
    char *key, *legacy; // storeKeys: "<mcp__name>|<url>" and the URL alone
    char *server_url;   // as configured
    char *norm_url;     // String(new URL(serverUrl)): McpOAuthProvider's serverUrl
    McpOAuthSettingsFn settings;
    void *ctx;
    pthread_mutex_t refresh_mu; // `refreshing`: one refresh at a time in this process
    pthread_mutex_t mu;         // the challenge
    char *ch_rm, *ch_scope, *ch_error;
};

McpAuth *mcp_auth_new(const char *agent_dir, const char *name, const char *url, McpOAuthSettingsFn settings, void *ctx) {
    McpAuth *m = xcalloc(1, sizeof *m);
    m->agent_dir = strdup(agent_dir);
    m->path = mfmt("%s/mcp-auth.json", agent_dir);
    Arena t = {0};
    char *h = href_of(&t, url);
    m->norm_url = strdup(h ? h : url);
    m->server_url = strdup(url);
    Buf ns = {0};
    buf_printf(&ns, "mcp__%s", name); // mcpNamespace
    for (size_t i = 0; i < ns.len; i++)
        if (ns.ptr[i] == '-') ns.ptr[i] = '_';
    m->key = mfmt("%s|%s", ns.ptr, m->norm_url);
    m->legacy = strdup(m->norm_url);
    buf_free(&ns);
    arena_reset(&t);
    m->settings = settings;
    m->ctx = ctx;
    pthread_mutex_init(&m->refresh_mu, NULL);
    pthread_mutex_init(&m->mu, NULL);
    return m;
}

typedef bool (*StatesFn)(Arena *a, JValue *states, void *ctx, bool *write);

// withLock: the file created when missing ("{}", 0600, its folder 0700), locked (proper-lockfile's
// "<real path>.lock" folder, 10 tries 20 ms apart, stale after 10 s), read, changed, written.
static bool with_states(McpAuth *m, Arena *a, StatesFn fn, void *ctx, OErr *e) {
    if (!file_exists(m->agent_dir) && !mkdir_p(m->agent_dir)) chmod(m->agent_dir, 0700);
    if (!file_exists(m->path)) {
        int fd = open(m->path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd >= 0) {
            (void)!write(fd, "{}", 2);
            close(fd);
        }
    }
    char real[PATH_MAX];
    const char *lp = realpath(m->path, real) ? real : m->path;
    if (!cfg_lock(lp, 10, 20)) return fail(e, OE_ERROR, strdup("Lock file is already being held"));
    char *text = NULL;
    size_t n = 0;
    bool ok = true;
    if (read_file(m->path, 64u << 20, &text, &n)) text = NULL;
    JValue *states = NULL;
    if (text && str_trim(str_n(text, n)).len) {
        states = jparse(a, str_n(text, n), e);
        if (!states) ok = false;
        else if (!is_obj(states)) states = jo(a);
    } else states = jo(a);
    free(text);
    bool write_back = false;
    if (ok) ok = fn(a, states, ctx, &write_back);
    if (ok && write_back) {
        Buf b = {0};
        cfg_json_pretty(&b, states, 0);
        buf_addc(&b, '\n');
        int fd = open(m->path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        bool w = fd >= 0;
        for (size_t off = 0; w && off < b.len;) {
            ssize_t k = write(fd, b.ptr + off, b.len - off);
            if (k < 0 && errno == EINTR) continue;
            if (k <= 0) w = false;
            else off += (size_t)k;
        }
        if (fd >= 0) close(fd);
        buf_free(&b);
        if (!w) ok = fail(e, OE_ERROR, mfmt("EACCES: permission denied, open '%s'", m->path));
    }
    cfg_unlock(lp);
    return ok;
}

typedef struct {
    McpAuth *m;
    JValue *state;  // load's result, save's input
    bool removed;
} StoreOp;

// load: the first server to load legacy state takes it over
static bool load_fn(Arena *a, JValue *states, void *ctx, bool *write) {
    StoreOp *op = ctx;
    (void)a;
    JValue *cur = json_get(states, op->m->key), *old = json_get(states, op->m->legacy);
    if (cur || !old) {
        op->state = cur;
        return true;
    }
    json_put(a, states, str(op->m->key), old);
    jdel(states, op->m->legacy);
    op->state = old;
    *write = true;
    return true;
}

static bool save_fn(Arena *a, JValue *states, void *ctx, bool *write) {
    StoreOp *op = ctx;
    json_put(a, states, str(op->m->key), op->state);
    *write = true;
    return true;
}

static bool remove_fn(Arena *a, JValue *states, void *ctx, bool *write) {
    StoreOp *op = ctx;
    (void)a;
    const char *k = json_get(states, op->m->key) ? op->m->key : json_get(states, op->m->legacy) ? op->m->legacy : NULL;
    if (!k) return true;
    jdel(states, k);
    op->removed = true;
    *write = true;
    return true;
}

static bool peek_fn(Arena *a, JValue *states, void *ctx, bool *write) {
    StoreOp *op = ctx;
    (void)a;
    (void)write;
    JValue *s = json_get(states, op->m->key);
    op->state = s ? s : json_get(states, op->m->legacy);
    return true;
}

static bool store_load(McpAuth *m, Arena *a, JValue **out, OErr *e) {
    StoreOp op = {m, NULL, false};
    if (!with_states(m, a, load_fn, &op, e)) return false;
    *out = op.state;
    return true;
}

static bool store_save(McpAuth *m, Arena *a, JValue *state, OErr *e) {
    StoreOp op = {m, state, false};
    return with_states(m, a, save_fn, &op, e);
}

char *mcp_auth_stored_tokens(McpAuth *m) {
    Arena t = {0};
    OErr e = {0};
    StoreOp op = {m, NULL, false};
    char *s = with_states(m, &t, peek_fn, &op, &e) ? jtext(json_get(op.state, "tokens")) : strdup("null");
    oerr_clear(&e);
    arena_reset(&t);
    return s;
}

bool mcp_auth_remove(McpAuth *m) {
    Arena t = {0};
    OErr e = {0};
    StoreOp op = {m, NULL, false};
    bool ok = with_states(m, &t, remove_fn, &op, &e) && op.removed;
    oerr_clear(&e);
    arena_reset(&t);
    return ok;
}

// withRefreshLock: proper-lockfile's lock on <agent dir>/mcp-auth-refresh-<hash> (a ".lock"
// folder), stale after 20 s, waited for 25 s in 100 ms steps; McpRefreshLockBusyError after.
static bool refresh_lock(McpAuth *m, volatile int *cancel, char **lock_path, OErr *e) {
    (void)mkdir_p(m->agent_dir);
    uint8_t d[SHA256_LEN];
    sha256(m->key, strlen(m->key), d);
    char hex[2 * SHA256_LEN + 1];
    hash_hex(d, SHA256_LEN, hex);
    hex[16] = 0;
    char *p = mfmt("%s/mcp-auth-refresh-%s.lock", m->agent_dir, hex);
    for (int i = 0; i <= 250; i++) {
        if (mkdir(p, 0755) == 0) {
            *lock_path = p;
            return true;
        }
        if (errno != EEXIST) break;
        struct stat st;
        if (stat(p, &st) == 0 && time(NULL) - st.st_mtime > 20) {
            rmdir(p);
            continue;
        }
        if (cancel && *cancel) {
            free(p);
            return fail(e, OE_CANCELLED, strdup("Sign-in cancelled"));
        }
        struct timespec ts = {0, 100000000L};
        nanosleep(&ts, NULL);
    }
    free(p);
    return fail(e, OE_LOCK_BUSY, strdup("Another process is refreshing the MCP server's tokens"));
}

static void refresh_unlock(char *lock_path) {
    if (lock_path) rmdir(lock_path);
    free(lock_path);
}

// ---------------------------------------------------------------------------------------
// metadata and token types (types.ts)
// ---------------------------------------------------------------------------------------

static bool absent(const JValue *v) { return !v || json_is(v, J_NULL) || (json_is(v, J_STRING) && !v->s.len); }

static const char *required_string(const JValue *v, const char *name, OErr *e) {
    if (!json_is(v, J_STRING) || !v->s.len) {
        fail(e, OE_ERROR, mfmt("Invalid %s", name));
        return NULL;
    }
    return v->s.ptr;
}

// safeUrl: http and https only (the authorization endpoint is opened with the platform handler)
static const char *safe_url(Arena *a, const JValue *v, const char *name, OErr *e) {
    const char *t = required_string(v, name, e);
    if (!t) return NULL;
    OUrl u;
    if (!ourl_parse(a, t, &u) || !is_http_url(&u)) {
        fail(e, OE_ERROR, mfmt("Invalid %s", name));
        return NULL;
    }
    return t;
}

// optionalStrings: a copy of the array, NULL when absent; *bad when not an array of strings
static JValue *optional_strings(Arena *a, const JValue *v, const char *name, OErr *e, bool *bad) {
    *bad = false;
    if (!v || json_is(v, J_NULL)) return NULL;
    if (!json_is(v, J_ARRAY)) goto invalid;
    for (size_t i = 0; i < v->arr.len; i++)
        if (!json_is(v->arr.items[i], J_STRING)) goto invalid;
    JValue *c = json_new(a, J_ARRAY);
    for (size_t i = 0; i < v->arr.len; i++) json_push(a, c, v->arr.items[i]);
    return c;
invalid:
    *bad = true;
    fail(e, OE_ERROR, mfmt("Invalid %s", name));
    return NULL;
}

static bool put_strings(Arena *a, JValue *o, const JValue *in, const char *key, const char *name, OErr *e) {
    bool bad;
    JValue *v = optional_strings(a, json_get(in, key), name, e, &bad);
    if (bad) return false;
    if (v) jset(a, o, key, v);
    else jdel(o, key);
    return true;
}

static JValue *parse_resource_metadata(Arena *a, const JValue *in, OErr *e) {
    if (!is_obj(in)) return fail(e, OE_ERROR, strdup("Invalid OAuth protected resource metadata")), NULL;
    JValue *o = jspread(a, in);
    const char *r = safe_url(a, json_get(in, "resource"), "OAuth protected resource metadata resource", e);
    if (!r) return NULL;
    jset(a, o, "resource", js(a, r));
    bool bad;
    JValue *as = optional_strings(a, json_get(in, "authorization_servers"), "authorization_servers", e, &bad);
    if (bad) return NULL;
    if (as) {
        for (size_t i = 0; i < as->arr.len; i++)
            if (!safe_url(a, as->arr.items[i], "authorization server URL", e)) return NULL;
        jset(a, o, "authorization_servers", as);
    } else jdel(o, "authorization_servers");
    if (!put_strings(a, o, in, "scopes_supported", "scopes_supported", e)) return NULL;
    return o;
}

static JValue *parse_server_metadata(Arena *a, const JValue *in, OErr *e) {
    if (!is_obj(in)) return fail(e, OE_ERROR, strdup("Invalid authorization server metadata")), NULL;
    bool bad;
    JValue *rt = optional_strings(a, json_get(in, "response_types_supported"), "response_types_supported", e, &bad);
    if (bad) return NULL;
    if (!rt) return fail(e, OE_ERROR, strdup("Invalid response_types_supported")), NULL;
    JValue *o = jspread(a, in);
    static const char *const urls[][2] = {{"issuer", "authorization server issuer"},
                                          {"authorization_endpoint", "authorization endpoint"},
                                          {"token_endpoint", "token endpoint"}};
    for (size_t i = 0; i < COUNT(urls); i++) {
        const char *u = safe_url(a, json_get(in, urls[i][0]), urls[i][1], e);
        if (!u) return NULL;
        jset(a, o, urls[i][0], js(a, u));
    }
    const JValue *reg = json_get(in, "registration_endpoint");
    if (absent(reg)) jdel(o, "registration_endpoint");
    else if (!safe_url(a, reg, "registration endpoint", e)) return NULL;
    if (!put_strings(a, o, in, "scopes_supported", "scopes_supported", e)) return NULL;
    jset(a, o, "response_types_supported", rt);
    if (!put_strings(a, o, in, "grant_types_supported", "grant_types_supported", e)) return NULL;
    if (!put_strings(a, o, in, "token_endpoint_auth_methods_supported", "token_endpoint_auth_methods_supported", e)) return NULL;
    if (!put_strings(a, o, in, "code_challenge_methods_supported", "code_challenge_methods_supported", e)) return NULL;
    if (!json_is(json_get(in, "client_id_metadata_document_supported"), J_BOOL)) jdel(o, "client_id_metadata_document_supported");
    if (!json_is(json_get(in, "authorization_response_iss_parameter_supported"), J_BOOL)) jdel(o, "authorization_response_iss_parameter_supported");
    return o;
}

static bool optional_string_into(Arena *a, JValue *o, const JValue *in, const char *key, OErr *e) {
    const JValue *v = json_get(in, key);
    if (absent(v)) return true;
    const char *s = required_string(v, key, e);
    if (!s) return false;
    jset(a, o, key, js(a, s));
    return true;
}

static JValue *parse_tokens(Arena *a, const JValue *in, OErr *e) {
    if (!is_obj(in)) return fail(e, OE_ERROR, strdup("Invalid OAuth token response")), NULL;
    JValue *o = jo(a);
    const char *at = required_string(json_get(in, "access_token"), "access_token", e);
    if (!at) return NULL;
    const char *tt = required_string(json_get(in, "token_type"), "token_type", e);
    if (!tt) return NULL;
    jset(a, o, "access_token", js(a, at));
    jset(a, o, "token_type", js(a, tt));
    const JValue *ex = json_get(in, "expires_in");
    if (!absent(ex)) {
        // Number(expires_in)
        double x = NAN;
        if (!jnumber(ex, &x)) {
            if (json_is(ex, J_STRING)) {
                char *end;
                Str t = str_trim(ex->s);
                char *z = arena_strndup(a, t.ptr, t.len);
                x = strtod(z, &end);
                if (*end) x = NAN;
            } else if (json_is(ex, J_BOOL)) x = ex->b;
        }
        if (!isfinite(x)) return fail(e, OE_ERROR, strdup("Invalid expires_in")), NULL;
        jset(a, o, "expires_in", jnum(a, x));
    }
    if (!optional_string_into(a, o, in, "scope", e) || !optional_string_into(a, o, in, "refresh_token", e) ||
        !optional_string_into(a, o, in, "id_token", e))
        return NULL;
    return o;
}

static JValue *parse_client_information(Arena *a, const JValue *in, OErr *e) {
    if (!is_obj(in)) return fail(e, OE_ERROR, strdup("Invalid OAuth client registration response")), NULL;
    JValue *o = jspread(a, in);
    const char *id = required_string(json_get(in, "client_id"), "client_id", e);
    if (!id) return NULL;
    jset(a, o, "client_id", js(a, id));
    jdel(o, "client_secret");
    if (!optional_string_into(a, o, in, "client_secret", e)) return NULL;
    double x;
    if (!(jnumber(json_get(in, "client_id_issued_at"), &x))) jdel(o, "client_id_issued_at");
    if (!(jnumber(json_get(in, "client_secret_expires_at"), &x))) jdel(o, "client_secret_expires_at");
    bool bad;
    JValue *ru = optional_strings(a, json_get(in, "redirect_uris"), "redirect_uris", e, &bad);
    if (bad) return NULL;
    jset(a, o, "redirect_uris", ru ? ru : json_new(a, J_ARRAY));
    return o;
}

// ---------------------------------------------------------------------------------------
// discovery (discovery.ts)
// ---------------------------------------------------------------------------------------

typedef struct {
    char *rm_url, *scope, *error, *error_description;
} Challenge;

// field(): name="value" or name=token, after the start, a comma or a space; "" counts as absent
static char *challenge_field(Arena *a, const char *h, const char *name) {
    size_t nl = strlen(name);
    for (const char *p = h; *p; p++) {
        if (p != h && !(p[-1] == ',' || p[-1] == ' ' || p[-1] == '\t' || p[-1] == '\n' || p[-1] == '\r')) continue;
        if (strncasecmp(p, name, nl) || p[nl] != '=') continue;
        const char *v = p + nl + 1;
        const char *end = *v == '"' ? strchr(v + 1, '"') : NULL;
        if (end) return end > v + 1 ? arena_strndup(a, v + 1, (size_t)(end - v - 1)) : NULL; // `scope=""` is absent
        size_t n = strcspn(v, " \t\n\r\f\v,");
        if (n) return arena_strndup(a, v, n);
    }
    return NULL;
}

// parseWwwAuthenticate
static Challenge parse_www_authenticate(Arena *a, const char *header) {
    Challenge c = {0};
    if (!header) return c;
    const char *s = header;
    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') s++;
    size_t n = strcspn(s, " \t\n\r");
    if (!((n == 6 && !strncasecmp(s, "bearer", 6)) || (n == 4 && !strncasecmp(s, "dpop", 4)))) return c;
    char *rm = challenge_field(a, header, "resource_metadata");
    if (rm) c.rm_url = href_of(a, rm);
    c.scope = challenge_field(a, header, "scope");
    c.error = challenge_field(a, header, "error");
    c.error_description = challenge_field(a, header, "error_description");
    return c;
}

static bool fetch_metadata(volatile int *cancel, const char *url, OResp *r, OErr *e) {
    sx_header hs[] = {{"Accept", "application/json"}, {"MCP-Protocol-Version", PROTOCOL_VERSION}};
    return ofetch(cancel, "GET", url, hs, COUNT(hs), S(""), r, e);
}

// 4xx and 502 mean "not here"
static bool discovery_miss(int status) { return (status >= 400 && status < 500) || status == 502; }

static JValue *discover_resource_metadata(Arena *a, volatile int *cancel, const char *server_url, const char *rm_url, OErr *e) {
    OUrl s;
    if (!ourl_parse(a, server_url, &s)) return fail(e, OE_ERROR, strdup("Invalid URL")), NULL;
    const char *url = rm_url ? rm_url : arena_printf(a, "%s/.well-known/oauth-protected-resource%s", ourl_origin(a, &s), path_suffix(a, s.path));
    OResp r;
    if (!fetch_metadata(cancel, url, &r, e)) return NULL;
    if (!rm_url && strcmp(s.path, "/") && discovery_miss(r.status)) {
        buf_free(&r.body);
        if (!fetch_metadata(cancel, arena_printf(a, "%s/.well-known/oauth-protected-resource", ourl_origin(a, &s)), &r, e)) return NULL;
    }
    if (!resp_ok(&r)) {
        buf_free(&r.body);
        return fail(e, OE_ERROR, mfmt("HTTP %d loading OAuth protected resource metadata", r.status)), NULL;
    }
    JValue *v = jparse(a, buf_str(&r.body), e);
    buf_free(&r.body);
    return v ? parse_resource_metadata(a, v, e) : NULL;
}

// discoverAuthorizationServerMetadata: NULL with e->kind OE_NONE when no candidate has metadata
static JValue *discover_server_metadata(Arena *a, volatile int *cancel, const char *as_url, OErr *e) {
    OUrl u;
    if (!ourl_parse(a, as_url, &u)) return fail(e, OE_ERROR, strdup("Invalid URL")), NULL;
    char *origin = ourl_origin(a, &u), *path = path_suffix(a, u.path);
    const char *cands[3] = {arena_printf(a, "%s/.well-known/oauth-authorization-server%s", origin, path),
                            arena_printf(a, "%s/.well-known/openid-configuration%s", origin, path),
                            *path ? arena_printf(a, "%s%s/.well-known/openid-configuration", origin, path) : NULL};
    for (size_t i = 0; i < 3 && cands[i]; i++) {
        OResp r;
        if (!fetch_metadata(cancel, cands[i], &r, e)) return NULL;
        if (!resp_ok(&r)) {
            buf_free(&r.body);
            if (discovery_miss(r.status)) continue;
            return fail(e, OE_ERROR, mfmt("HTTP %d loading authorization server metadata from %s", r.status, cands[i])), NULL;
        }
        JValue *v = jparse(a, buf_str(&r.body), e);
        buf_free(&r.body);
        if (!v) return NULL;
        JValue *md = parse_server_metadata(a, v, e);
        if (!md) return NULL;
        // the issuer must be the authorization server's URL (a trailing slash aside)
        Str is = json_strs(json_get(md, "issuer")), ex = str(as_url);
        if (is.len && is.ptr[is.len - 1] == '/') is.len--;
        if (ex.len && ex.ptr[ex.len - 1] == '/') ex.len--;
        if (!str_eq(is, ex)) {
            Buf b = {0};
            json_write_string(&b, str(as_url));
            buf_adds(&b, ", received ");
            json_write_string(&b, json_strs(json_get(md, "issuer")));
            char *m = mfmt("OAuth issuer mismatch: expected %s", b.ptr);
            buf_free(&b);
            return fail(e, OE_ISSUER, m), NULL;
        }
        return md;
    }
    return NULL;
}

typedef struct {
    const char *as_url; // authorizationServerUrl
    JValue *as_md;      // authorizationServerMetadata (may be NULL)
    JValue *rm;         // resourceMetadata (may be NULL)
} Discovered;

static bool discover_server_info(Arena *a, volatile int *cancel, const char *server_url, const char *rm_url, const char *as_md_url, Discovered *d, OErr *e) {
    memset(d, 0, sizeof *d);
    OErr pe = {0};
    d->rm = discover_resource_metadata(a, cancel, server_url, rm_url, &pe);
    if (!d->rm && (pe.kind == OE_NETWORK || pe.kind == OE_CANCELLED)) {
        *e = pe;
        return false;
    }
    oerr_clear(&pe);
    if (as_md_url) {
        OResp r;
        if (!fetch_metadata(cancel, as_md_url, &r, e)) return false;
        if (!resp_ok(&r)) {
            buf_free(&r.body);
            return fail(e, OE_ERROR, mfmt("HTTP %d loading authorization server metadata from %s", r.status, as_md_url));
        }
        JValue *v = jparse(a, buf_str(&r.body), e);
        buf_free(&r.body);
        if (!v || !(d->as_md = parse_server_metadata(a, v, e))) return false;
        d->as_url = jstrz(d->as_md, "issuer");
        return true;
    }
    const JValue *as = json_get(d->rm, "authorization_servers");
    if (json_is(as, J_ARRAY) && as->arr.len) d->as_url = as->arr.items[0]->s.ptr;
    else {
        OUrl u;
        if (!ourl_parse(a, server_url, &u)) return fail(e, OE_ERROR, strdup("Invalid URL"));
        d->as_url = arena_printf(a, "%s/", ourl_origin(a, &u));
    }
    d->as_md = discover_server_metadata(a, cancel, d->as_url, e);
    return d->as_md || e->kind == OE_NONE;
}

// selectResource
static bool select_resource(Arena *a, const char *server_url, const JValue *rm, const char **out, OErr *e) {
    *out = NULL;
    if (!rm) return true;
    OUrl req, conf;
    if (!ourl_parse(a, server_url, &req) || !ourl_parse(a, jstrz(rm, "resource"), &conf)) return fail(e, OE_ERROR, strdup("Invalid URL"));
    req.has_hash = false;
    char *mism = mfmt("Protected resource %s does not match MCP server %s", jstrz(rm, "resource"), ourl_href(a, &req));
    if (strcmp(ourl_origin(a, &req), ourl_origin(a, &conf))) return fail(e, OE_ERROR, mism);
    char *rp = arena_printf(a, "%s%s", req.path, str_ends(str(req.path), S("/")) ? "" : "/");
    char *cp = arena_printf(a, "%s%s", conf.path, str_ends(str(conf.path), S("/")) ? "" : "/");
    if (strncmp(rp, cp, strlen(cp))) return fail(e, OE_ERROR, mism);
    free(mism);
    *out = jstrz(rm, "resource");
    return true;
}

// ---------------------------------------------------------------------------------------
// the client (client-settings.ts) and its provider (provider.ts McpOAuthProvider)
// ---------------------------------------------------------------------------------------

// isClientMetadataDocumentCallback: pi's documents list /callback on localhost and 127.0.0.1
static bool lists_redirect_url(Arena *a, const char *redirect) {
    OUrl u;
    if (!ourl_parse(a, redirect, &u)) return false;
    return strcmp(u.host, "[::1]") && !strcmp(u.path, "/callback");
}

// offersClientMetadataDocument
static bool offers_document(Arena *a, const McpOAuthSettings *st, const char *redirect) {
    const char *reg = st->client_registration ? st->client_registration : "auto";
    if (strcmp(reg, "auto")) return !strcmp(reg, "cimd");
    return !st->client_id && !st->client_name && lists_redirect_url(a, redirect);
}

// callbackId: 12 characters for the server URL (its hash dropped)
static char *callback_id(Arena *a, const char *server_url) {
    OUrl u;
    if (!ourl_parse(a, server_url, &u)) return arena_strdup(a, "");
    u.has_hash = false;
    char *h = ourl_href(a, &u);
    uint8_t d[SHA256_LEN];
    sha256(h, strlen(h), d);
    return b64z(a, d, 9, true);
}

static char *server_specific_path(Arena *a, const char *path, const char *server_url) {
    size_t n = strlen(path);
    return arena_printf(a, "%.*s/%s", (int)(n && path[n - 1] == '/' ? n - 1 : n), path, callback_id(a, server_url));
}

typedef struct {
    Arena *a;
    McpAuth *m;
    McpOAuthSettings st; // piSettings applied
    volatile int *cancel;
} OCtx;

typedef struct {
    OCtx *o;
    const char *redirect_url;
    JValue *client_metadata;
    bool offers_doc;
    const char *conf_id, *conf_secret; // configuredClient
    char *auth_url;                    // redirectToAuthorization's URL
} Prov;

static Prov prov_new(OCtx *o, const char *redirect) {
    Arena *a = o->a;
    Prov p = {o, arena_strdup(a, redirect), NULL, false, NULL, NULL, NULL};
    JValue *cm = jo(a);
    jset(a, cm, "client_name", js(a, o->st.client_name ? o->st.client_name : APP_NAME));
    JValue *ru = json_new(a, J_ARRAY);
    json_push(a, ru, js(a, redirect));
    jset(a, cm, "redirect_uris", ru);
    JValue *gt = json_new(a, J_ARRAY);
    json_push(a, gt, js(a, "authorization_code"));
    json_push(a, gt, js(a, "refresh_token"));
    jset(a, cm, "grant_types", gt);
    JValue *rt = json_new(a, J_ARRAY);
    json_push(a, rt, js(a, "code"));
    jset(a, cm, "response_types", rt);
    jset(a, cm, "token_endpoint_auth_method", js(a, o->st.client_secret ? "client_secret_post" : "none"));
    p.client_metadata = cm;
    p.offers_doc = offers_document(a, &o->st, redirect);
    if (o->st.client_id) {
        p.conf_id = o->st.client_id;
        p.conf_secret = o->st.client_secret;
    }
    return p;
}

// own(): state stored for another server URL is ignored
static bool prov_load(Prov *p, JValue **out, OErr *e) {
    JValue *s;
    if (!store_load(p->o->m, p->o->a, &s, e)) return false;
    if (!is_obj(s) || strcmp(jstrz(s, "serverUrl") ? jstrz(s, "serverUrl") : "", p->o->m->norm_url)) {
        s = jo(p->o->a);
        jset(p->o->a, s, "serverUrl", js(p->o->a, p->o->m->norm_url));
    } else s = jspread(p->o->a, s);
    *out = s;
    return true;
}

static bool prov_save(Prov *p, JValue *s, OErr *e) { return store_save(p->o->m, p->o->a, s, e); }

static JValue *configured_client(Prov *p) {
    if (!p->conf_id) return NULL;
    JValue *c = jo(p->o->a);
    jset(p->o->a, c, "client_id", js(p->o->a, p->conf_id));
    if (p->conf_secret && *p->conf_secret) jset(p->o->a, c, "client_secret", js(p->o->a, p->conf_secret));
    return c;
}

static bool prov_client_information(Prov *p, JValue **out, OErr *e) {
    JValue *c = configured_client(p);
    if (c) {
        *out = c;
        return true;
    }
    JValue *s;
    if (!prov_load(p, &s, e)) return false;
    *out = json_get(s, "clientInformation");
    return true;
}

static bool prov_set(Prov *p, const char *key, JValue *v, OErr *e) {
    JValue *s;
    if (!prov_load(p, &s, e)) return false;
    if (v) jset(p->o->a, s, key, v);
    else jdel(s, key);
    return prov_save(p, s, e);
}

static bool prov_save_tokens(Prov *p, JValue *tokens, OErr *e) {
    double ex;
    bool has = jnumber(json_get(tokens, "expires_in"), &ex);
    double at = has ? now_ms() + ex * 1000 : 0;
    JValue *s;
    if (!prov_load(p, &s, e)) return false;
    jset(p->o->a, s, "tokens", tokens);
    if (has) jset(p->o->a, s, "tokensExpireAt", jnum(p->o->a, at));
    else jdel(s, "tokensExpireAt");
    return prov_save(p, s, e);
}

// state(): the stored one, else a new one stored
static bool prov_state(Prov *p, const char **out, OErr *e) {
    JValue *s;
    if (!prov_load(p, &s, e)) return false;
    const char *st = jstrz(s, "oauthState");
    if (st && *st) {
        *out = st;
        return true;
    }
    uint8_t r[32];
    arc4random_buf(r, sizeof r);
    char hex[65];
    hash_hex(r, sizeof r, hex);
    *out = arena_strdup(p->o->a, hex);
    jset(p->o->a, s, "oauthState", js(p->o->a, hex));
    return prov_save(p, s, e);
}

static bool prov_invalidate(Prov *p, const char *kind, OErr *e) {
    JValue *s;
    if (!prov_load(p, &s, e)) return false;
    bool all = !strcmp(kind, "all");
    if (all || !strcmp(kind, "client")) jdel(s, "clientInformation");
    if (all || !strcmp(kind, "tokens")) {
        jdel(s, "tokens");
        jdel(s, "tokensExpireAt");
    }
    if (all || !strcmp(kind, "verifier")) jdel(s, "codeVerifier");
    if (all || !strcmp(kind, "discovery")) jdel(s, "discovery");
    if (all) jdel(s, "oauthState");
    return prov_save(p, s, e);
}

// clientMetadataDocument (createOAuthClientProvider's): NULL when the server does not support them
typedef struct {
    const char *url, *redirect;
} ClientDoc;

static bool client_document(Prov *p, const JValue *md, ClientDoc *doc, OErr *e) {
    Arena *a = p->o->a;
    doc->url = doc->redirect = NULL;
    if (!lists_redirect_url(a, p->redirect_url))
        return fail(e, OE_ERROR,
                    mfmt("oauth.clientRegistration \"cimd\" needs Client ID Metadata Documents that list the redirect URI %s", p->redirect_url));
    bool supported = json_is(json_get(md, "client_id_metadata_document_supported"), J_BOOL) &&
                     json_get(md, "client_id_metadata_document_supported")->b && str_in(json_get(md, "token_endpoint_auth_methods_supported"), "none");
    if (supported) {
        const JValue *iss = json_get(md, "authorization_response_iss_parameter_supported");
        if (json_is(iss, J_BOOL) && iss->b) {
            doc->url = CIMD_BASE "/client.json";
            doc->redirect = p->redirect_url;
        } else {
            OUrl r;
            if (!ourl_parse(a, p->redirect_url, &r)) return fail(e, OE_ERROR, strdup("Invalid URL"));
            r.path = server_specific_path(a, r.path, p->o->m->server_url);
            doc->url = arena_printf(a, CIMD_BASE "/%s/client.json", callback_id(a, p->o->m->server_url));
            doc->redirect = ourl_href(a, &r);
        }
    }
    if (!doc->url && p->o->st.client_registration && !strcmp(p->o->st.client_registration, "cimd"))
        return fail(e, OE_ERROR,
                    strdup("The authorization server does not support Client ID Metadata Documents for public clients; remove oauth.clientRegistration \"cimd\""));
    return true;
}

// ---------------------------------------------------------------------------------------
// the authorization code flow (flow.ts)
// ---------------------------------------------------------------------------------------

// secureEndpoint
static const char *secure_endpoint(Arena *a, const char *url, OErr *e) {
    OUrl u;
    if (!ourl_parse(a, url, &u)) return fail(e, OE_ERROR, strdup("Invalid URL")), NULL;
    char *h = ourl_href(a, &u);
    if (strcmp(u.scheme, "https") && !loopback_hostname(u.host)) return fail(e, OE_INSECURE, mfmt("Refusing to send OAuth credentials to non-HTTPS endpoint %s", h)), NULL;
    return h;
}

// selectClientAuthMethod
static const char *client_auth_method(const JValue *info, const JValue *supported) {
    const char *hint = jstrz(info, "token_endpoint_auth_method");
    bool secret = jstrz(info, "client_secret") && *jstrz(info, "client_secret");
    size_t ns = json_is(supported, J_ARRAY) ? supported->arr.len : 0;
    if (hint && (!strcmp(hint, "client_secret_basic") || !strcmp(hint, "client_secret_post") || !strcmp(hint, "none")) && (!ns || str_in(supported, hint)))
        return hint;
    if (!ns) return secret ? "client_secret_basic" : "none";
    if (secret && str_in(supported, "client_secret_basic")) return "client_secret_basic";
    if (secret && str_in(supported, "client_secret_post")) return "client_secret_post";
    if (str_in(supported, "none")) return "none";
    return secret ? "client_secret_post" : "none";
}

// tokenRequest: the token endpoint's answer, its OAuth error first (any status)
static JValue *token_request(OCtx *o, const char *as_url, const JValue *md, const JValue *client, const char *resource, Params *params, OErr *e) {
    Arena *a = o->a;
    const char *ep = jstrz(md, "token_endpoint");
    if (!ep) {
        OUrl u;
        if (!ourl_parse(a, as_url, &u)) return fail(e, OE_ERROR, strdup("Invalid URL")), NULL;
        ep = arena_printf(a, "%s/token", ourl_origin(a, &u));
    }
    const char *url = secure_endpoint(a, ep, e);
    if (!url) return NULL;
    if (resource) params_set(params, "resource", resource);
    const char *method = client_auth_method(client, json_get(md, "token_endpoint_auth_methods_supported"));
    const char *id = jstrz(client, "client_id"), *secret = jstrz(client, "client_secret");
    VEC(sx_header) hs = {0};
    vec_push(&hs, ((sx_header){"Accept", "application/json"}));
    vec_push(&hs, ((sx_header){"content-type", "application/x-www-form-urlencoded"}));
    if (!strcmp(method, "client_secret_basic")) {
        if (!secret || !*secret) {
            vec_free(&hs);
            return fail(e, OE_ERROR, strdup("client_secret_basic requires a client secret")), NULL;
        }
        char *pair = arena_printf(a, "%s:%s", id, secret);
        vec_push(&hs, ((sx_header){"Authorization", arena_printf(a, "Basic %s", b64z(a, pair, strlen(pair), false))}));
    } else {
        params_set(params, "client_id", id);
        if (!strcmp(method, "client_secret_post") && secret && *secret) params_set(params, "client_secret", secret);
    }
    OResp r;
    bool ok = ofetch(o->cancel, "POST", url, hs.ptr, hs.len, str(params_text(a, params)), &r, e);
    vec_free(&hs);
    if (!ok) return NULL;
    char *err = NULL;
    JValue *v = code_oai_js_parse(a, buf_str(&r.body), &err);
    const char *oe = jstrz(v, "error");
    if (is_obj(v) && oe) {
        const char *d = jstrz(v, "error_description");
        fail(e, OE_OAUTH, strdup(d && *d ? d : *oe ? oe : ""));
        e->code = strdup(oe);
        buf_free(&r.body);
        return NULL;
    }
    if (!resp_ok(&r)) {
        fail(e, OE_OAUTH, mfmt("HTTP %d: %.*s", r.status, (int)r.body.len, r.body.ptr ? r.body.ptr : ""));
        e->code = strdup("server_error");
        buf_free(&r.body);
        return NULL;
    }
    buf_free(&r.body);
    return parse_tokens(a, v, e);
}

// applicationType (MCP SEP-837)
static const char *application_type(Arena *a, const JValue *uris) {
    for (size_t i = 0; json_is(uris, J_ARRAY) && i < uris->arr.len; i++) {
        OUrl u;
        if (!json_is(uris->arr.items[i], J_STRING) || !ourl_parse(a, uris->arr.items[i]->s.ptr, &u)) continue;
        if (!is_http_url(&u) || loopback_hostname(u.host)) return "native";
    }
    return "web";
}

static JValue *register_client(OCtx *o, const char *as_url, const JValue *md, const JValue *cm, const char *scope, OErr *e) {
    Arena *a = o->a;
    const char *ep = jstrz(md, "registration_endpoint");
    if (md && !ep) return fail(e, OE_ERROR, strdup("Authorization server does not support dynamic client registration")), NULL;
    if (!ep) {
        OUrl u;
        if (!ourl_parse(a, as_url, &u)) return fail(e, OE_ERROR, strdup("Invalid URL")), NULL;
        ep = arena_printf(a, "%s/register", ourl_origin(a, &u));
    }
    JValue *body = jspread(a, cm);
    if (!json_get(cm, "application_type")) jset(a, body, "application_type", js(a, application_type(a, json_get(cm, "redirect_uris"))));
    if (scope && *scope) jset(a, body, "scope", js(a, scope));
    char *text = jtext(body);
    sx_header hs[] = {{"Accept", "application/json"}, {"content-type", "application/json"}};
    OResp r;
    bool ok = ofetch(o->cancel, "POST", ep, hs, COUNT(hs), str(text), &r, e);
    free(text);
    if (!ok) return NULL;
    if (!resp_ok(&r)) {
        fail(e, OE_REGISTRATION, mfmt("OAuth dynamic client registration failed with status %d: %.*s", r.status, (int)r.body.len, r.body.ptr ? r.body.ptr : ""));
        buf_free(&r.body);
        return NULL;
    }
    JValue *v = jparse(a, buf_str(&r.body), e);
    buf_free(&r.body);
    return v ? parse_client_information(a, v, e) : NULL;
}

// startAuthorization: the authorization URL and the PKCE verifier
static bool start_authorization(OCtx *o, const char *as_url, const JValue *md, const JValue *client, const char *redirect, const char *scope, const char *state,
                                const char *resource, char **url_out, char **verifier_out, OErr *e) {
    Arena *a = o->a;
    if (md && !str_in(json_get(md, "response_types_supported"), "code")) return fail(e, OE_ERROR, strdup("Authorization server does not support authorization codes"));
    const JValue *ccm = json_get(md, "code_challenge_methods_supported");
    if (ccm && !str_in(ccm, "S256")) return fail(e, OE_ERROR, strdup("Authorization server does not support PKCE S256"));
    const char *ep = jstrz(md, "authorization_endpoint");
    OUrl u;
    if (!ep) {
        if (!ourl_parse(a, as_url, &u)) return fail(e, OE_ERROR, strdup("Invalid URL"));
        ep = arena_printf(a, "%s/authorize", ourl_origin(a, &u));
    }
    if (!ourl_parse(a, ep, &u) || !is_http_url(&u)) return fail(e, OE_ERROR, mfmt("Invalid authorization endpoint %s", ep));
    // pkce(): 32 random bytes as base64url, and its SHA-256
    uint8_t r[32], d[SHA256_LEN];
    arc4random_buf(r, sizeof r);
    char *verifier = b64z(a, r, sizeof r, true);
    sha256(verifier, strlen(verifier), d);
    char *challenge = b64z(a, d, sizeof d, true);
    Params p;
    params_parse(a, u.query, &p);
    params_set(&p, "response_type", "code");
    params_set(&p, "client_id", jstrz(client, "client_id"));
    params_set(&p, "code_challenge", challenge);
    params_set(&p, "code_challenge_method", "S256");
    params_set(&p, "redirect_uri", redirect);
    if (state && *state) params_set(&p, "state", state);
    if (scope && *scope) params_set(&p, "scope", scope);
    bool offline = false;
    for (const char *s = scope; s && *s;) {
        while (*s == ' ' || *s == '\t' || *s == '\n') s++;
        size_t n = strcspn(s, " \t\n");
        offline |= n == 14 && !strncmp(s, "offline_access", 14);
        s += n;
    }
    if (offline) params_set(&p, "prompt", "consent");
    if (resource) params_set(&p, "resource", resource);
    u.query = params_text(a, &p);
    u.has_query = *u.query != 0;
    params_free(&p);
    *url_out = ourl_href(a, &u);
    *verifier_out = verifier;
    return true;
}

typedef struct {
    const char *rm_url, *as_md_url, *scope;
    const char *code, *iss, *auth_redirect;
    bool skip_refresh, refresh_only;
} FlowOpts;
typedef enum { FLOW_AUTHORIZED, FLOW_REDIRECT } FlowResult;

static JValue *with_scope(Arena *a, JValue *tokens, const char *scope) {
    if (json_get(tokens, "scope") || !scope || !*scope) return tokens;
    JValue *t = jspread(a, tokens);
    jset(a, t, "scope", js(a, scope));
    return t;
}

static char *join_strings(Arena *a, const JValue *arr) {
    Buf b = {0};
    for (size_t i = 0; json_is(arr, J_ARRAY) && i < arr->arr.len; i++) {
        if (i) buf_addc(&b, ' ');
        buf_addstr(&b, json_strs(arr->arr.items[i]));
    }
    char *s = arena_strdup(a, b.ptr ? b.ptr : "");
    buf_free(&b);
    return s;
}

// runFlow
static bool run_flow(Prov *p, const FlowOpts *f, const char **stored_id, FlowResult *res, OErr *e) {
    OCtx *o = p->o;
    Arena *a = o->a;
    const char *server_url = o->m->server_url;
    const char *md_url = NULL;
    if (f->as_md_url && !(md_url = secure_endpoint(a, f->as_md_url, e))) return false;
    // with a configured metadata URL, discovery is not cached
    JValue *state;
    if (!prov_load(p, &state, e)) return false;
    const JValue *cached = md_url ? NULL : json_get(state, "discovery");
    Discovered d = {0};
    if (jstrz(cached, "authorizationServerUrl")) {
        d.as_url = jstrz(cached, "authorizationServerUrl");
        d.as_md = json_get(cached, "authorizationServerMetadata");
        d.rm = json_get(cached, "resourceMetadata");
        if (!d.as_md) {
            d.as_md = discover_server_metadata(a, o->cancel, d.as_url, e);
            if (!d.as_md && e->kind != OE_NONE) return false;
        }
    } else if (!discover_server_info(a, o->cancel, server_url, f->rm_url, md_url, &d, e)) return false;
    if (!md_url) {
        JValue *ds = jo(a);
        jset(a, ds, "authorizationServerUrl", js(a, d.as_url));
        if (d.as_md) jset(a, ds, "authorizationServerMetadata", d.as_md);
        if (d.rm) jset(a, ds, "resourceMetadata", d.rm);
        if (f->rm_url) jset(a, ds, "resourceMetadataUrl", js(a, f->rm_url));
        if (!prov_set(p, "discovery", ds, e)) return false;
    }
    const JValue *md = d.as_md;
    const char *resource;
    if (!select_resource(a, server_url, d.rm, &resource, e)) return false;
    const char *scope = f->scope && *f->scope ? f->scope : NULL;
    if (!scope) {
        char *ss = join_strings(a, json_get(d.rm, "scopes_supported"));
        scope = *ss ? ss : jstrz(p->client_metadata, "scope");
    }
    JValue *stored;
    if (!prov_client_information(p, &stored, e)) return false;
    *stored_id = jstrz(stored, "client_id");
    ClientDoc doc = {0};
    if (!stored && p->offers_doc && !client_document(p, md, &doc, e)) return false;
    if (doc.url) {
        OUrl du;
        if (!ourl_parse(a, doc.url, &du) || strcmp(du.scheme, "https") || !strcmp(du.path, "/")) return fail(e, OE_ERROR, strdup("Invalid OAuth client metadata URL"));
    }
    const char *redirect = doc.redirect ? doc.redirect : p->redirect_url;
    if (f->code && f->auth_redirect && strcmp(redirect, f->auth_redirect))
        return fail(e, OE_ERROR, mfmt("The code exchange would use the redirect URI %s, but the authorization request used %s", redirect, f->auth_redirect));
    JValue *client = stored;
    if (!client && doc.url) {
        client = jo(a);
        jset(a, client, "client_id", js(a, doc.url));
    }
    if (!client) {
        if (f->refresh_only) {
            *res = FLOW_REDIRECT;
            return true;
        }
        if (f->code)
            return fail(e, OE_ERROR, strdup("OAuth client information is missing during code exchange: the authorization request's client was removed, or the client settings changed"));
        if (!(client = register_client(o, d.as_url, md, p->client_metadata, scope, e))) return false;
        if (!p->conf_id && !prov_set(p, "clientInformation", client, e)) return false;
    }
    if (f->code) {
        // RFC 9207: never send a code from another authorization server to this one
        const JValue *issp = json_get(md, "authorization_response_iss_parameter_supported");
        if (f->iss || (json_is(issp, J_BOOL) && issp->b)) {
            const char *expected = jstrz(md, "issuer");
            if (!f->iss || !expected || strcmp(f->iss, expected)) {
                Buf b = {0};
                if (expected) json_write_string(&b, str(expected));
                else buf_adds(&b, "none");
                buf_adds(&b, ", received ");
                if (f->iss) json_write_string(&b, str(f->iss));
                else buf_adds(&b, "none");
                char *m = mfmt("OAuth issuer mismatch: expected %s", b.ptr);
                buf_free(&b);
                return fail(e, OE_ISSUER, m);
            }
        }
        JValue *s2;
        if (!prov_load(p, &s2, e)) return false;
        const char *verifier = jstrz(s2, "codeVerifier");
        if (!verifier || !*verifier) return fail(e, OE_ERROR, strdup("No OAuth PKCE code verifier is stored"));
        Params pa = {0};
        params_set(&pa, "grant_type", "authorization_code");
        params_set(&pa, "code", f->code);
        params_set(&pa, "code_verifier", verifier);
        params_set(&pa, "redirect_uri", redirect);
        JValue *t = token_request(o, d.as_url, md, client, resource, &pa, e);
        params_free(&pa);
        if (!t || !prov_save_tokens(p, with_scope(a, t, scope), e)) return false;
        *res = FLOW_AUTHORIZED;
        return true;
    }
    const JValue *existing = NULL;
    if (!f->skip_refresh) {
        JValue *s3;
        if (!prov_load(p, &s3, e)) return false;
        existing = json_get(s3, "tokens");
    }
    const char *rt = jstrz(existing, "refresh_token");
    if (rt) {
        Params pa = {0};
        params_set(&pa, "grant_type", "refresh_token");
        params_set(&pa, "refresh_token", rt);
        OErr re = {0};
        JValue *t = token_request(o, d.as_url, md, client, resource, &pa, &re);
        params_free(&pa);
        if (t) {
            // refreshAuthorization: { refresh_token, ...tokens }; a refresh without scope keeps the grant's
            JValue *nt = jo(a);
            jset(a, nt, "refresh_token", js(a, rt));
            for (size_t i = 0; i < t->obj.len; i++) json_put(a, nt, t->obj.items[i].key, t->obj.items[i].val);
            oerr_clear(&re);
            if (!prov_save_tokens(p, with_scope(a, nt, jstrz(existing, "scope")), e)) return false;
            *res = FLOW_AUTHORIZED;
            return true;
        }
        if ((o->cancel && *o->cancel) || re.kind == OE_INSECURE || re.kind == OE_CANCELLED || (re.kind == OE_OAUTH && strcmp(re.code, "server_error"))) {
            oerr_clear(e);
            *e = re;
            return false;
        }
        oerr_clear(&re);
    }
    if (f->refresh_only) {
        *res = FLOW_REDIRECT;
        return true;
    }
    const char *st;
    if (!prov_state(p, &st, e)) return false;
    char *url, *verifier;
    if (!start_authorization(o, d.as_url, md, client, redirect, scope, st, resource, &url, &verifier, e)) return false;
    if (!prov_set(p, "codeVerifier", js(a, verifier), e)) return false;
    p->auth_url = url;
    *res = FLOW_REDIRECT;
    return true;
}

// authorizeMcp: a rejected client registers again, a rejected grant signs in again
static bool authorize(Prov *p, const FlowOpts *f, FlowResult *res, OErr *e) {
    const char *stored_id = NULL;
    if (run_flow(p, f, &stored_id, res, e)) return true;
    if (e->kind == OE_OAUTH && (!strcmp(e->code, "invalid_client") || !strcmp(e->code, "unauthorized_client"))) {
        OErr ie = {0};
        bool ok = true;
        if (f->refresh_only) {
            JValue *cur = NULL;
            ok = prov_client_information(p, &cur, &ie);
            const char *cid = jstrz(cur, "client_id");
            if (ok && ((!cid && !stored_id) || (cid && stored_id && !strcmp(cid, stored_id))))
                ok = prov_invalidate(p, "client", &ie) && prov_invalidate(p, "tokens", &ie) && prov_invalidate(p, "discovery", &ie);
        } else ok = prov_invalidate(p, "all", &ie);
        if (!ok) {
            oerr_clear(e);
            *e = ie;
            return false;
        }
        oerr_clear(e);
        return run_flow(p, f, &stored_id, res, e);
    }
    if (e->kind == OE_OAUTH && !strcmp(e->code, "invalid_grant")) {
        OErr ie = {0};
        if (!prov_invalidate(p, "tokens", &ie)) {
            oerr_clear(e);
            *e = ie;
            return false;
        }
        oerr_clear(e);
        return run_flow(p, f, &stored_id, res, e);
    }
    return false;
}

// ---------------------------------------------------------------------------------------
// settings (oauth.ts piSettings, client-settings.ts loopbackSettings)
// ---------------------------------------------------------------------------------------

static bool load_settings(McpAuth *m, Arena *a, McpOAuthSettings *st, OErr *e) {
    memset(st, 0, sizeof *st);
    char *err = NULL;
    if (m->settings && !m->settings(m->ctx, a, st, &err)) return fail(e, OE_ERROR, err ? err : strdup("invalid OAuth settings"));
    // a rebranded build would show up as pi with pi's documents, so "auto" registers
    if (!st->client_registration || !strcmp(st->client_registration, "auto")) st->client_registration = "dcr";
    return true;
}

typedef struct {
    const char *url;   // the redirect URI, without the port unless one is configured
    const char *host;  // the address to listen on
    const char *redirect_host;
    int port; // -1: none
    const char *path;
    const char *fixed; // the exact redirect URI when the port is fixed
} Loopback;

static Loopback loopback_settings(Arena *a, const McpOAuthSettings *st) {
    Loopback l = {0};
    l.url = st->callback_url ? st->callback_url : DEFAULT_REDIRECT_URL;
    OUrl u;
    if (!ourl_parse(a, l.url, &u)) ourl_parse(a, DEFAULT_REDIRECT_URL, &u);
    char *addr = u.host;
    if (addr[0] == '[') addr = arena_strndup(a, addr + 1, strlen(addr) - 2);
    l.port = *u.port ? atoi(u.port) : st->callback_port ? st->callback_port : -1;
    if (*u.port) l.fixed = st->callback_url;
    else if (l.port >= 0) {
        u.port = arena_printf(a, "%d", l.port);
        if ((!strcmp(u.scheme, "http") && l.port == 80)) u.port = "";
        l.fixed = ourl_href(a, &u);
    }
    l.host = !strcmp(addr, "localhost") ? "127.0.0.1" : addr;
    l.redirect_host = addr;
    l.path = u.path;
    return l;
}

static const char *first_registered_redirect(const JValue *client) {
    const JValue *r = json_get(client, "redirect_uris");
    return json_is(r, J_ARRAY) && r->arr.len ? json_str(r->arr.items[0]) : NULL;
}

// refreshStoredTokens: a refresh with the stored client, or REDIRECT when that is not possible
static bool refresh_stored_tokens(OCtx *o, const JValue *state, const char *rm_url, const char *redirect_url, FlowResult *res, OErr *e) {
    Arena *a = o->a;
    Loopback l = loopback_settings(a, &o->st);
    const char *cands[3] = {redirect_url, l.fixed, first_registered_redirect(json_get(state, "clientInformation"))};
    const char *redirect = DEFAULT_REDIRECT_URL;
    for (size_t i = 0; i < 3; i++)
        if (cands[i] && lists_redirect_url(a, cands[i])) {
            redirect = cands[i];
            break;
        }
    Prov p = prov_new(o, redirect);
    FlowOpts f = {.rm_url = rm_url, .as_md_url = o->st.auth_server_metadata_url, .refresh_only = true};
    return authorize(&p, &f, res, e);
}

// ---------------------------------------------------------------------------------------
// the connection's auth provider (auth-provider.ts createMcpAuthProvider)
// ---------------------------------------------------------------------------------------

// refresh(staleToken): under the refresh lock, unless another refresh replaced the token already
static int auth_refresh(McpAuth *m, const char *stale, const char *rm_url, char **err) {
    pthread_mutex_lock(&m->refresh_mu);
    Arena t = {0};
    OErr e = {0};
    int rc = MCP_AUTH_RETRY;
    char *lock = NULL;
    if (!refresh_lock(m, NULL, &lock, &e)) rc = MCP_AUTH_FAILED;
    else {
        JValue *state;
        if (!store_load(m, &t, &state, &e)) rc = MCP_AUTH_FAILED;
        else {
            const char *cur = jstrz(json_get(state, "tokens"), "access_token");
            if ((cur == NULL) != (stale == NULL) || (cur && strcmp(cur, stale))) rc = MCP_AUTH_RETRY;
            else if (!jstrz(json_get(state, "tokens"), "refresh_token")) rc = MCP_AUTH_REQUIRED;
            else {
                OCtx o = {&t, m, {0}, NULL};
                FlowResult res;
                if (!load_settings(m, &t, &o.st, &e) || !refresh_stored_tokens(&o, state, rm_url, NULL, &res, &e)) rc = MCP_AUTH_FAILED;
                else rc = res == FLOW_REDIRECT ? MCP_AUTH_REQUIRED : MCP_AUTH_RETRY;
            }
        }
        refresh_unlock(lock);
    }
    if (rc == MCP_AUTH_FAILED && err) *err = strdup(e.msg ? e.msg : "OAuth refresh failed");
    oerr_clear(&e);
    arena_reset(&t);
    pthread_mutex_unlock(&m->refresh_mu);
    return rc;
}

char *mcp_auth_token(McpAuth *m) {
    pthread_mutex_lock(&m->refresh_mu); // a refresh in progress finishes first
    pthread_mutex_unlock(&m->refresh_mu);
    Arena t = {0};
    OErr e = {0};
    JValue *state = NULL;
    char *tok = NULL;
    if (store_load(m, &t, &state, &e)) {
        const JValue *tk = json_get(state, "tokens");
        // own(): state of another URL is not this server's
        if (!jstrz(state, "serverUrl") || strcmp(jstrz(state, "serverUrl"), m->norm_url)) tk = NULL;
        const char *at = jstrz(tk, "access_token");
        double exp;
        bool expired = jnumber(json_get(state, "tokensExpireAt"), &exp) && exp - REFRESH_SKEW_MS <= now_ms();
        if (expired && jstrz(tk, "refresh_token")) {
            char *er = NULL;
            (void)auth_refresh(m, at, NULL, &er); // failures fall through: a 401 decides
            free(er);
            arena_reset(&t);
            oerr_clear(&e);
            if (store_load(m, &t, &state, &e) && jstrz(state, "serverUrl") && !strcmp(jstrz(state, "serverUrl"), m->norm_url))
                at = jstrz(json_get(state, "tokens"), "access_token");
            else at = NULL;
        }
        tok = at ? strdup(at) : NULL;
    }
    oerr_clear(&e);
    arena_reset(&t);
    return tok;
}

int mcp_auth_unauthorized(McpAuth *m, const char *www, const char *stale, char **err) {
    Arena t = {0};
    Challenge c = parse_www_authenticate(&t, www);
    pthread_mutex_lock(&m->mu);
    free(m->ch_rm);
    free(m->ch_scope);
    free(m->ch_error);
    m->ch_rm = c.rm_url ? strdup(c.rm_url) : NULL;
    m->ch_scope = c.scope ? strdup(c.scope) : NULL;
    m->ch_error = c.error ? strdup(c.error) : NULL;
    pthread_mutex_unlock(&m->mu);
    int rc;
    // a refresh keeps the granted scope, so more scope needs a new sign-in
    if (c.error && !strcmp(c.error, "insufficient_scope")) rc = MCP_AUTH_REQUIRED;
    else rc = auth_refresh(m, stale, c.rm_url, err);
    arena_reset(&t);
    return rc;
}

void mcp_auth_settled(McpAuth *m) {
    pthread_mutex_lock(&m->refresh_mu);
    pthread_mutex_unlock(&m->refresh_mu);
}

void mcp_auth_clear_challenge(McpAuth *m) {
    pthread_mutex_lock(&m->mu);
    free(m->ch_rm);
    free(m->ch_scope);
    free(m->ch_error);
    m->ch_rm = m->ch_scope = m->ch_error = NULL;
    pthread_mutex_unlock(&m->mu);
}

// ---------------------------------------------------------------------------------------
// the loopback callback (callback.ts OAuthCallbackServer, loopback.ts) and its page
// ---------------------------------------------------------------------------------------

static void html_escape(Buf *b, const char *s) {
    for (; s && *s; s++) switch (*s) {
        case '&': buf_adds(b, "&amp;"); break;
        case '<': buf_adds(b, "&lt;"); break;
        case '>': buf_adds(b, "&gt;"); break;
        case '"': buf_adds(b, "&quot;"); break;
        case '\'': buf_adds(b, "&#39;"); break;
        default: buf_addc(b, *s);
        }
}

// oauth-page.ts renderPage (pi's logo left out: this is Saetta Code's page)
static void render_page(Buf *b, bool ok, const char *message, const char *details) {
    const char *title = ok ? "Authentication successful" : "Authentication failed";
    buf_adds(b, "<!doctype html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"utf-8\" />\n"
                "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n  <title>");
    html_escape(b, title);
    buf_adds(b, "</title>\n  <style>\n"
                "    :root {\n      --text: #fafafa;\n      --text-dim: #a1a1aa;\n      --page-bg: #09090b;\n"
                "      --font-sans: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif;\n"
                "      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace;\n    }\n"
                "    * { box-sizing: border-box; }\n    html { color-scheme: dark; }\n"
                "    body {\n      margin: 0;\n      min-height: 100vh;\n      display: flex;\n      align-items: center;\n      justify-content: center;\n"
                "      padding: 24px;\n      background: var(--page-bg);\n      color: var(--text);\n      font-family: var(--font-sans);\n      text-align: center;\n    }\n"
                "    main {\n      width: 100%;\n      max-width: 560px;\n      display: flex;\n      flex-direction: column;\n      align-items: center;\n"
                "      justify-content: center;\n    }\n"
                "    h1 {\n      margin: 0 0 10px;\n      font-size: 28px;\n      line-height: 1.15;\n      font-weight: 650;\n      color: var(--text);\n    }\n"
                "    p {\n      margin: 0;\n      line-height: 1.7;\n      color: var(--text-dim);\n      font-size: 15px;\n    }\n"
                "    .details {\n      margin-top: 16px;\n      font-family: var(--font-mono);\n      font-size: 13px;\n      color: var(--text-dim);\n"
                "      white-space: pre-wrap;\n      word-break: break-word;\n    }\n  </style>\n</head>\n<body>\n  <main>\n    <h1>");
    html_escape(b, title);
    buf_adds(b, "</h1>\n    <p>");
    html_escape(b, message);
    buf_adds(b, "</p>\n    ");
    if (details) {
        buf_adds(b, "<div class=\"details\">");
        html_escape(b, details);
        buf_adds(b, "</div>");
    }
    buf_adds(b, "\n  </main>\n</body>\n</html>");
}

static void reply(int fd, int status, bool ok, const char *message, const char *details) {
    Buf page = {0}, b = {0};
    if (ok) render_page(&page, true, "Signed in to the MCP server. You may now close this page.", NULL);
    else render_page(&page, false, message, details);
    const char *reason = status == 200 ? "OK" : status == 400 ? "Bad Request" : "Not Found";
    buf_printf(&b, "HTTP/1.1 %d %s\r\ncontent-type: text/html; charset=utf-8\r\ncache-control: no-store\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
               status, reason, page.len);
    buf_add(&b, page.ptr, page.len);
    for (size_t off = 0; off < b.len;) {
        ssize_t k = send(fd, b.ptr + off, b.len - off, 0);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) break;
        off += (size_t)k;
    }
    buf_free(&page);
    buf_free(&b);
}

typedef struct {
    int fd;
    char *redirect_url; // http://<redirect host>:<port><path>
    const char *paths[2];
    size_t npaths;
} Callback;

static const char *errno_name(int e) {
    switch (e) {
    case EADDRINUSE: return "EADDRINUSE";
    case EADDRNOTAVAIL: return "EADDRNOTAVAIL";
    case EACCES: return "EACCES";
    default: return "EINVAL";
    }
}

// OAuthCallbackServer.listen
static bool callback_listen(Arena *a, const char *host, const char *redirect_host, int port, const char *path, const char *extra, Callback *cb, OErr *e) {
    memset(cb, 0, sizeof *cb);
    cb->fd = -1;
    struct addrinfo hints = {0}, *ai = NULL;
    hints.ai_flags = AI_NUMERICHOST | AI_PASSIVE;
    hints.ai_socktype = SOCK_STREAM;
    char ps[16];
    snprintf(ps, sizeof ps, "%d", port > 0 ? port : 0);
    if (getaddrinfo(host, ps, &hints, &ai) || !ai) return fail(e, OE_ERROR, mfmt("getaddrinfo ENOTFOUND %s", host));
    int fd = socket(ai->ai_family, SOCK_STREAM, 0);
    int one = 1;
    if (fd >= 0) setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (fd < 0 || bind(fd, ai->ai_addr, ai->ai_addrlen) || listen(fd, 16)) {
        int er = errno;
        if (fd >= 0) close(fd);
        freeaddrinfo(ai);
        const char *desc = er == EADDRINUSE ? "address already in use" : er == EADDRNOTAVAIL ? "address not available" : strerror(er);
        return fail(e, OE_ERROR, mfmt("listen %s: %s %s:%d", errno_name(er), desc, host, port > 0 ? port : 0));
    }
    freeaddrinfo(ai);
    fcntl(fd, F_SETFD, FD_CLOEXEC);
    struct sockaddr_storage ss;
    socklen_t sl = sizeof ss;
    getsockname(fd, (struct sockaddr *)&ss, &sl);
    int bound = ntohs(ss.ss_family == AF_INET6 ? ((struct sockaddr_in6 *)&ss)->sin6_port : ((struct sockaddr_in *)&ss)->sin_port);
    cb->fd = fd;
    cb->redirect_url = arena_printf(a, "http://%s%s%s:%d%s", strchr(redirect_host, ':') ? "[" : "", redirect_host, strchr(redirect_host, ':') ? "]" : "", bound, path);
    cb->paths[cb->npaths++] = path;
    if (extra) cb->paths[cb->npaths++] = extra;
    return true;
}

static void callback_close(Callback *cb) {
    if (cb->fd >= 0) close(cb->fd);
    cb->fd = -1;
}

typedef struct {
    const char *code, *state, *iss;
} AuthResponse;

// one request on the callback server: 1 when it carried the response (or failed the sign-in), 0 to
// go on waiting
static int callback_handle(Arena *a, Callback *cb, int fd, const char *want_state, const char *want_path, AuthResponse *out, OErr *e) {
    char req[8192];
    size_t n = 0;
    uint64_t until = now_ns() + 2000000000ull;
    while (n < sizeof req - 1) {
        struct pollfd p = {fd, POLLIN, 0};
        int left = (int)((until - MIN(until, now_ns())) / 1000000ull);
        if (left <= 0 || poll(&p, 1, left) <= 0) break;
        ssize_t k = recv(fd, req + n, sizeof req - 1 - n, 0);
        if (k <= 0) break;
        n += (size_t)k;
        req[n] = 0;
        if (strstr(req, "\r\n\r\n") || strstr(req, "\n\n")) break;
    }
    req[n] = 0;
    char *sp = strchr(req, ' ');
    if (!sp) return 0;
    char *target = sp + 1;
    size_t tl = strcspn(target, " \r\n");
    target = arena_strndup(a, target, tl);
    // new URL(rawUrl, redirectUrl)
    OUrl u;
    if (!ourl_parse(a, target[0] == '/' ? arena_printf(a, "http://x%s", target) : target, &u)) {
        reply(fd, 404, false, "Not found", NULL);
        return 0;
    }
    bool known = false;
    for (size_t i = 0; i < cb->npaths; i++) known |= !strcmp(cb->paths[i], u.path);
    if (!known) {
        reply(fd, 404, false, "Not found", NULL);
        return 0;
    }
    Params q;
    params_parse(a, u.query, &q);
    const char *state = params_get(&q, "state");
    if (!state || !*state || strcmp(state, want_state)) {
        params_free(&q);
        reply(fd, 400, false, "Invalid or expired OAuth state", NULL);
        return 0;
    }
    int rc = 1;
    if (want_path && strcmp(u.path, want_path)) {
        fail(e, OE_ERROR, strdup("The authorization response arrived on another redirect URI"));
        reply(fd, 400, false, "Unexpected redirect URI", NULL);
    } else if (params_get(&q, "error") && *params_get(&q, "error")) {
        const char *d = params_get(&q, "error_description");
        const char *desc = d ? d : params_get(&q, "error");
        fail(e, OE_ERROR, strdup(desc));
        reply(fd, 200, false, "Authorization failed. You may close this window.", desc);
    } else if (!params_get(&q, "code") || !*params_get(&q, "code")) {
        fail(e, OE_ERROR, strdup("OAuth callback did not include an authorization code"));
        reply(fd, 400, false, "Missing authorization code", NULL);
    } else {
        out->code = params_get(&q, "code");
        out->state = state;
        const char *iss = params_get(&q, "iss");
        out->iss = iss && *iss ? iss : NULL;
        reply(fd, 200, true, NULL, NULL);
        rc = 2;
    }
    params_free(&q);
    return rc;
}

// responseFromRedirectUrl: the authorization response in a pasted redirect URL
static bool response_from_redirect(Arena *a, const char *input, const char *state, const char *redirect, AuthResponse *out, OErr *e) {
    Str t = str_trim(str(input));
    OUrl u, r;
    if (!ourl_parse(a, arena_strndup(a, t.ptr, t.len), &u)) return fail(e, OE_ERROR, strdup("Expected the full redirect URL from the browser address bar"));
    if (!ourl_parse(a, redirect, &r) || strcmp(ourl_origin(a, &u), ourl_origin(a, &r)) || strcmp(u.path, r.path))
        return fail(e, OE_ERROR, strdup("The redirect URL does not match this sign-in's redirect URI"));
    Params q;
    params_parse(a, u.query, &q);
    const char *er = params_get(&q, "error");
    bool ok = false;
    if (er && *er) {
        const char *d = params_get(&q, "error_description");
        fail(e, OE_ERROR, strdup(d ? d : er));
    } else if (!params_get(&q, "state") || strcmp(params_get(&q, "state"), state)) fail(e, OE_ERROR, strdup("The redirect URL belongs to a different sign-in"));
    else if (!params_get(&q, "code") || !*params_get(&q, "code")) fail(e, OE_ERROR, strdup("The redirect URL does not contain an authorization code"));
    else {
        out->code = params_get(&q, "code");
        out->state = params_get(&q, "state");
        const char *iss = params_get(&q, "iss");
        out->iss = iss && *iss ? iss : NULL;
        ok = true;
    }
    params_free(&q);
    return ok;
}

// waitForLoopbackSignIn: the callback, a pasted redirect URL, or the cancel, whichever comes first
static bool wait_for_response(Arena *a, Callback *cb, McpSignIn *si, const char *state, const char *redirect, AuthResponse *out, OErr *e) {
    OUrl ru;
    const char *want_path = ourl_parse(a, redirect, &ru) ? ru.path : NULL;
    uint64_t until = now_ns() + (uint64_t)CALLBACK_TIMEOUT_MS * 1000000ull;
    for (;;) {
        if (si->cancel) return fail(e, OE_CANCELLED, strdup("Sign-in cancelled"));
        pthread_mutex_lock(&si->mu);
        bool gave_up = si->gave_up;
        char *pasted = si->pasted;
        si->pasted = NULL;
        pthread_mutex_unlock(&si->mu);
        if (gave_up) return fail(e, OE_CANCELLED, strdup("Sign-in cancelled"));
        if (pasted) {
            bool ok = response_from_redirect(a, pasted, state, redirect, out, e);
            free(pasted);
            return ok;
        }
        if (now_ns() >= until) return fail(e, OE_ERROR, strdup("OAuth callback timed out"));
        struct pollfd p = {cb->fd, POLLIN, 0};
        if (poll(&p, 1, 100) <= 0) continue;
        int fd = accept(cb->fd, NULL, NULL);
        if (fd < 0) continue;
        int rc = callback_handle(a, cb, fd, state, want_path, out, e);
        close(fd);
        if (rc == 2) return true;
        if (rc == 1) return false;
    }
}

// ---------------------------------------------------------------------------------------
// the sign-in (sign-in.ts startMcpSignIn / completeMcpSignIn, oauth.ts signInMcpServer)
// ---------------------------------------------------------------------------------------

void mcp_sign_in_init(McpSignIn *si) {
    memset(si, 0, sizeof *si);
    pthread_mutex_init(&si->mu, NULL);
}

void mcp_sign_in_destroy(McpSignIn *si) {
    free(si->pasted);
    pthread_mutex_destroy(&si->mu);
}

void mcp_sign_in_paste(McpSignIn *si, const char *url) {
    pthread_mutex_lock(&si->mu);
    if (url && str_trim(str(url)).len) {
        free(si->pasted);
        si->pasted = strdup(url);
    } else si->gave_up = true;
    pthread_mutex_unlock(&si->mu);
}

static void merge_scopes(Buf *b, const char *s) {
    for (const char *p = s; p && *p;) {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == '\f' || *p == '\v') p++;
        size_t n = strcspn(p, " \t\n\r\f\v");
        if (!n) break;
        bool seen = false;
        Str have = buf_str(b);
        for (size_t i = 0; i < have.len && !seen;) {
            size_t m = strcspn(have.ptr + i, " ");
            seen = m == n && !strncmp(have.ptr + i, p, n);
            i += m + 1;
        }
        if (!seen) {
            if (b->len) buf_addc(b, ' ');
            buf_add(b, p, n);
        }
        p += n;
    }
}

// stepUpScope: the challenged scopes plus the granted ones; NULL without challenged scopes
static char *step_up_scope(Arena *a, const char *granted, const char *challenged) {
    if (!challenged || !*challenged) return NULL;
    Buf b = {0};
    merge_scopes(&b, granted);
    merge_scopes(&b, challenged);
    char *s = arena_strdup(a, b.ptr ? b.ptr : "");
    buf_free(&b);
    return s;
}

// refreshBeforeSignIn
static bool refresh_before_sign_in(OCtx *o, const char *redirect, const char *rm_url, FlowResult *res, OErr *e) {
    McpAuth *m = o->m;
    *res = FLOW_REDIRECT;
    pthread_mutex_lock(&m->refresh_mu);
    char *lock = NULL;
    bool ok = refresh_lock(m, o->cancel, &lock, e);
    if (ok) {
        if (!*o->cancel) {
            JValue *state;
            ok = store_load(m, o->a, &state, e);
            if (ok && jstrz(json_get(state, "tokens"), "refresh_token") &&
                !(!strcmp(o->st.client_registration, "cimd") && json_get(state, "clientInformation") && !o->st.client_id)) {
                // not cancellable: a refresh the server answered must save its tokens
                volatile int *c = o->cancel;
                o->cancel = NULL;
                ok = refresh_stored_tokens(o, state, rm_url, redirect, res, e);
                o->cancel = c;
            }
        }
        refresh_unlock(lock);
    } else if (e->kind == OE_LOCK_BUSY) {
        oerr_clear(e);
        ok = true; // another process is refreshing and took too long: the user signs in instead
    }
    pthread_mutex_unlock(&m->refresh_mu);
    if (ok && *o->cancel) return fail(e, OE_CANCELLED, strdup("Sign-in cancelled"));
    return ok;
}

typedef struct {
    const char *url, *redirect, *state, *scope;
} Redirect;

// startBrowserStep
static bool start_browser_step(OCtx *o, const char *redirect, const Challenge *ch, bool step_up, Redirect *out, OErr *e) {
    Arena *a = o->a;
    JValue *stored;
    if (!store_load(o->m, a, &stored, e)) return false;
    if (stored) {
        // stateForSignIn: without the last sign-in's state, and without a client that cannot use
        // this redirect URI (with its tokens)
        JValue *next = jspread(a, stored);
        jdel(next, "oauthState");
        const JValue *ci = json_get(stored, "clientInformation");
        bool keep;
        if (o->st.client_id) keep = true;
        else if (ci) {
            keep = strcmp(o->st.client_registration, "cimd");
            if (keep) {
                const JValue *ru = json_get(ci, "redirect_uris");
                keep = str_in(ru, redirect);
            }
        } else keep = offers_document(a, &o->st, redirect);
        if (!keep) {
            jdel(next, "clientInformation");
            jdel(next, "tokens");
            jdel(next, "tokensExpireAt");
        }
        if (!store_save(o->m, a, next, e)) return false;
    }
    Prov p = prov_new(o, redirect);
    Buf sb = {0};
    merge_scopes(&sb, o->st.scope);
    merge_scopes(&sb, step_up ? step_up_scope(a, jstrz(json_get(stored, "tokens"), "scope"), ch->scope) : ch->scope);
    const char *scope = sb.len ? arena_strdup(a, sb.ptr) : NULL;
    buf_free(&sb);
    FlowOpts f = {.rm_url = ch->rm_url, .as_md_url = o->st.auth_server_metadata_url, .scope = scope, .skip_refresh = true};
    FlowResult res;
    if (!authorize(&p, &f, &res, e)) return false;
    if (!p.auth_url) return fail(e, OE_ERROR, strdup("OAuth flow did not produce an authorization URL"));
    OUrl u;
    ourl_parse(a, p.auth_url, &u);
    Params q;
    params_parse(a, u.query, &q);
    out->url = p.auth_url;
    const char *ru = params_get(&q, "redirect_uri");
    out->redirect = ru ? ru : redirect;
    const char *rs = params_get(&q, "scope");
    out->scope = rs && *rs ? rs : NULL;
    params_free(&q);
    return prov_state(&p, &out->state, e);
}

// completeMcpSignIn
static bool complete_sign_in(OCtx *o, const char *redirect, const Challenge *ch, const Redirect *auth, const AuthResponse *resp, OErr *e) {
    if (!resp->state || strcmp(resp->state, auth->state)) return fail(e, OE_ERROR, strdup("The authorization response belongs to a different sign-in"));
    JValue *state;
    if (!store_load(o->m, o->a, &state, e)) return false;
    if (!jstrz(state, "oauthState") || strcmp(jstrz(state, "oauthState"), auth->state))
        return fail(e, OE_ERROR, strdup("Another sign-in to this MCP server started since; sign in again"));
    Prov p = prov_new(o, redirect);
    FlowOpts f = {.rm_url = ch->rm_url,
                  .as_md_url = o->st.auth_server_metadata_url,
                  .scope = auth->scope,
                  .auth_redirect = auth->redirect,
                  .code = resp->code,
                  .iss = resp->iss};
    FlowResult res;
    return authorize(&p, &f, &res, e);
}

int mcp_auth_sign_in(McpAuth *m, McpSignIn *si, char **err) {
    Arena t = {0};
    OErr e = {0};
    OCtx o = {&t, m, {0}, &si->cancel};
    Challenge ch = {0};
    pthread_mutex_lock(&m->mu);
    ch.rm_url = m->ch_rm ? arena_strdup(&t, m->ch_rm) : NULL;
    ch.scope = m->ch_scope ? arena_strdup(&t, m->ch_scope) : NULL;
    ch.error = m->ch_error ? arena_strdup(&t, m->ch_error) : NULL;
    pthread_mutex_unlock(&m->mu);
    Callback cb = {.fd = -1};
    bool ok = load_settings(m, &t, &o.st, &e);
    const char *redirect = NULL;
    if (ok) {
        // listenForLoopbackSignIn: callbackUrl / callbackPort, else the stored client's port, else a free one
        Loopback l = loopback_settings(&t, &o.st);
        JValue *stored = NULL;
        ok = store_load(m, &t, &stored, &e);
        if (ok) {
            const char *reg = first_registered_redirect(json_get(stored, "clientInformation"));
            int port = l.port;
            OUrl ru;
            if (port < 0 && reg && ourl_parse(&t, reg, &ru) && *ru.port && atoi(ru.port) > 0) port = atoi(ru.port);
            bool doc = offers_document(&t, &o.st, l.url);
            const char *extra = doc ? server_specific_path(&t, l.path, m->server_url) : NULL;
            ok = callback_listen(&t, l.host, l.redirect_host, port, l.path, extra, &cb, &e);
            if (!ok && l.port < 0 && port >= 0) {
                // the stored client's port is only preferred
                oerr_clear(&e);
                ok = callback_listen(&t, l.host, l.redirect_host, 0, l.path, extra, &cb, &e);
            }
            redirect = l.fixed ? l.fixed : cb.redirect_url;
        }
    }
    if (ok && si->cancel) ok = fail(&e, OE_CANCELLED, strdup("Sign-in cancelled"));
    bool step_up = ch.error && !strcmp(ch.error, "insufficient_scope");
    FlowResult res = FLOW_REDIRECT;
    if (ok && !step_up) ok = refresh_before_sign_in(&o, redirect, ch.rm_url, &res, &e);
    if (ok && res != FLOW_AUTHORIZED) {
        Redirect auth = {0};
        ok = start_browser_step(&o, redirect, &ch, step_up, &auth, &e);
        if (ok) {
            if (si->show_url) si->show_url(si->ctx, auth.url);
            AuthResponse resp = {0};
            ok = wait_for_response(&t, &cb, si, auth.state, auth.redirect, &resp, &e);
            if (si->responded) si->responded(si->ctx);
            if (ok) ok = complete_sign_in(&o, redirect, &ch, &auth, &resp, &e);
        }
    }
    callback_close(&cb);
    // an abort shows as the cancellation it is
    if (!ok && si->cancel) {
        oerr_clear(&e);
        e.kind = OE_CANCELLED;
    }
    int rc = ok ? MCP_SIGNIN_OK : e.kind == OE_CANCELLED ? MCP_SIGNIN_CANCELLED : MCP_SIGNIN_FAILED;
    if (rc == MCP_SIGNIN_FAILED) *err = strdup(e.msg ? e.msg : "unknown error");
    oerr_clear(&e);
    arena_reset(&t);
    return rc;
}

// ---------------------------------------------------------------------------------------
// the browser (utils/open-browser.ts openBrowser) and the tests' hook
// ---------------------------------------------------------------------------------------

static void *reap_main(void *arg) {
    pid_t pid = (pid_t)(intptr_t)arg;
    int st;
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {
    }
    return NULL;
}

static int discard_sink(void *ctx, int status, const void *p, size_t n) {
    (void)ctx;
    (void)status;
    (void)p;
    (void)n;
    return 0;
}

static void *fetch_main(void *arg) {
    char *url = arg;
    int status = 0;
    char err[128];
    (void)sx_http_ex("GET", url, NULL, 0, NULL, 0, OAUTH_REQUEST_TIMEOUT_S, NULL, discard_sink, NULL, &status, err, sizeof err, NULL);
    free(url);
    return NULL;
}

static void detach(void *(*fn)(void *), void *arg) {
    pthread_t th;
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    if (pthread_create(&th, &at, fn, arg)) {
        if (fn == fetch_main) free(arg);
    }
    pthread_attr_destroy(&at);
}

void mcp_oauth_open_url(const char *url) {
    const char *hook = getenv("SAETTA_MCP_OAUTH_FETCH");
    if (hook && *hook) {
        detach(fetch_main, strdup(url));
        return;
    }
    // `open <url>`, its output dropped, in "/" (best effort: the screen shows the URL too)
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addchdir(&fa, "/");
    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_CLOEXEC_DEFAULT);
    posix_spawnattr_setpgroup(&attr, 0);
    char *argv[] = {"open", (char *)url, NULL};
    pid_t pid;
    if (posix_spawnp(&pid, "open", &fa, &attr, argv, environ) == 0) detach(reap_main, (void *)(intptr_t)pid);
    posix_spawnattr_destroy(&attr);
    posix_spawn_file_actions_destroy(&fa);
}
