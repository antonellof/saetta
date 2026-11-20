// `saetta status`, `saetta stop`, `saetta update` (see control.h).
#include "control.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <ftw.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "hash.h"
#include "instance.h"
#include "json.h"
#include "metal/shim.h"
#include "stats.h"

extern char **environ;

static double num(const JValue *v, const char *k) {
    const JValue *x = json_get(v, k);
    if (!x) return 0;
    if (x->kind == J_INTEGER) return (double)x->i;
    if (x->kind == J_FLOAT) return x->f;
    return 0;
}

static const char *jstr(const JValue *v, const char *k) {
    const char *s = json_str(json_get(v, k));
    return s ? s : "";
}

static const char *fx(Arena *a, double x, int prec) {
    char b[400];
    fmt_fixed(b, sizeof b, x, prec);
    return arena_strdup(a, b);
}

static Err status_in(Arena *a, FILE *out, int *code) {
    Server s;
    if (!instance_running(&s)) {
        fprintf(out, "saetta: no server is running\n");
        *code = 1;
        return OK;
    }
    JValue *m = NULL;
    const char *home = getenv("HOME");
    char *path = path_join(a, path_join(a, path_join(a, home ? home : "", ".saetta"), "run"), "metrics.json");
    char *text;
    size_t len;
    if (!read_file(path, 1 << 20, &text, &len)) {
        Str t = str_n(arena_dup(a, text, len + 1), len);
        free(text);
        if (json_parse(a, t, true, &m)) m = NULL;
        if (m && (int64_t)num(m, "pid") != s.pid) m = NULL;
    }
    fprintf(out, "saetta: server running, pid %d, http://127.0.0.1:%u (%s)\n", s.pid, s.port,
            s.autostop ? "started by saetta start, stops after its last session" : "started by hand, stop it with saetta stop");
    if (m) {
        double up = num(m, "now") - num(m, "started");
        fprintf(out, "  model    %s (%s), context %s tokens, up %s min\n", jstr(m, "model"), jstr(m, "pack"),
                fx(a, num(m, "ctx"), 0), fx(a, up / 60, 0));
        double convs = num(m, "convs");
        fprintf(out, "  state    %s, %s requests, %s conversation(s) kept\n", jstr(m, "state"), fx(a, num(m, "requests"), 0),
                fx(a, convs > 1 ? convs : 1, 0));
        double gws = num(m, "gpu_working_set");
        if (gws > 0)
            fprintf(out, "  GPU      %s of %s GB\n", fx(a, num(m, "gpu_allocated") / (double)(1 << 30), 1),
                    fx(a, gws / (double)(1 << 30), 1));
    }
    Client *cl;
    size_t n = instance_clients(a, &cl);
    fprintf(out, "  sessions %zu", n);
    for (size_t i = 0; i < n; i++) fprintf(out, "%s%s (pid %d)", i == 0 ? ": " : ", ", cl[i].agent, cl[i].pid);
    fprintf(out, "\n");
    if (instance_kept()) fprintf(out, "  kept     yes (saetta start --keep): stays up with no session\n");
    *code = 0;
    return OK;
}

Err control_status(FILE *out, int *code) {
    Arena a = {0};
    Err e = status_in(&a, out, code);
    arena_reset(&a);
    return e;
}

static void sleep_ms(long ms) {
    struct timespec ts = {ms / 1000, (ms % 1000) * 1000000};
    nanosleep(&ts, NULL);
}

Err control_stop(FILE *out, int *code) {
    Server s;
    if (!instance_running(&s)) {
        fprintf(out, "saetta: no server is running\n");
        *code = 0;
        return OK;
    }
    Arena a = {0};
    Client *cl;
    size_t n = instance_clients(&a, &cl);
    arena_reset(&a);
    if (n > 0) fprintf(out, "saetta: note: %zu agent session(s) still use it; their next request will fail\n", n);
    instance_set_keep(false);
    if (kill(s.pid, SIGTERM) != 0) {
        fprintf(out, "saetta: could not signal pid %d\n", s.pid);
        *code = 1;
        return OK;
    }
    for (uint32_t waited = 0; instance_running(NULL) && waited < 60; waited++) sleep_ms(500);
    if (instance_running(NULL)) {
        fprintf(out, "saetta: pid %d did not stop within 30 s\n", s.pid);
        *code = 1;
        return OK;
    }
    fprintf(out, "saetta: stopped the server (pid %d, port %u)\n", s.pid, s.port);
    *code = 0;
    return OK;
}

// --- semantic versions (std.SemanticVersion) --------------------------------

static bool all_digits(Str s) {
    for (size_t i = 0; i < s.len; i++)
        if (!isdigit((unsigned char)s.ptr[i])) return false;
    return true;
}

// std.fmt.parseUnsigned: digits with inner underscores, no sign.
static Err parse_unsigned(Str t, uint64_t *v) {
    if (t.len > 0 && (t.ptr[0] == '+' || t.ptr[0] == '-')) return "InvalidCharacter";
    return parse_u64(t, v);
}

static Err parse_num(Str t, size_t *out) {
    if (t.len > 1 && t.ptr[0] == '0') return "InvalidVersion";
    uint64_t v;
    Err e = parse_unsigned(t, &v);
    if (e && strcmp(e, "Overflow") == 0) return e;
    if (e) return "InvalidVersion";
    *out = (size_t)v;
    return OK;
}

// Splits s at sep: the next part from *pos, false after the last.
static bool split_next(Str s, char sep, size_t *pos, bool *done, Str *part) {
    if (*done) return false;
    ptrdiff_t k = str_find_char(s, sep, *pos);
    if (k < 0) {
        *part = str_sub(s, *pos, s.len);
        *done = true;
    } else {
        *part = str_sub(s, *pos, (size_t)k);
        *pos = (size_t)k + 1;
    }
    return true;
}

static bool ident_ok(Str id) {
    if (id.len == 0) return false;
    for (size_t i = 0; i < id.len; i++)
        if (!isalnum((unsigned char)id.ptr[i]) && id.ptr[i] != '-') return false;
    return true;
}

Err semver_parse(Str text, SemanticVersion *out) {
    size_t extra_index = text.len;
    for (size_t i = 0; i < text.len; i++)
        if (text.ptr[i] == '-' || text.ptr[i] == '+') {
            extra_index = i;
            break;
        }
    Str required = str_sub(text, 0, extra_index);
    size_t pos = 0;
    bool done = false;
    Str part;
    SemanticVersion v = {0};
    split_next(required, '.', &pos, &done, &part);
    TRY(parse_num(part, &v.major));
    if (!split_next(required, '.', &pos, &done, &part)) return "InvalidVersion";
    TRY(parse_num(part, &v.minor));
    if (!split_next(required, '.', &pos, &done, &part)) return "InvalidVersion";
    TRY(parse_num(part, &v.patch));
    if (split_next(required, '.', &pos, &done, &part)) return "InvalidVersion";
    if (extra_index < text.len) {
        Str extra = str_sub(text, extra_index, text.len);
        if (extra.ptr[0] == '-') {
            ptrdiff_t b = str_find_char(extra, '+', 0);
            v.pre = str_sub(extra, 1, b < 0 ? extra.len : (size_t)b);
            if (b >= 0) v.build = str_sub(extra, (size_t)b + 1, extra.len);
        } else {
            v.build = str_sub(extra, 1, extra.len);
        }
        if (v.pre.ptr) {
            pos = 0;
            done = false;
            while (split_next(v.pre, '.', &pos, &done, &part)) {
                if (!ident_ok(part)) return "InvalidVersion";
                if (all_digits(part)) {
                    size_t x;
                    TRY(parse_num(part, &x));
                }
            }
        }
        if (v.build.ptr) {
            pos = 0;
            done = false;
            while (split_next(v.build, '.', &pos, &done, &part))
                if (!ident_ok(part)) return "InvalidVersion";
        }
    }
    *out = v;
    return OK;
}

static bool unsigned_num(Str s, size_t *out) {
    uint64_t v;
    if (parse_unsigned(s, &v)) return false;
    *out = (size_t)v;
    return true;
}

int semver_order(const SemanticVersion *l, const SemanticVersion *r) {
    if (l->major != r->major) return l->major < r->major ? -1 : 1;
    if (l->minor != r->minor) return l->minor < r->minor ? -1 : 1;
    if (l->patch != r->patch) return l->patch < r->patch ? -1 : 1;
    if (l->pre.ptr && !r->pre.ptr) return -1;
    if (!l->pre.ptr && !r->pre.ptr) return 0;
    if (!l->pre.ptr && r->pre.ptr) return 1;
    size_t lp = 0, rp = 0;
    bool ld = false, rd = false;
    for (;;) {
        Str lid, rid;
        bool hl = split_next(l->pre, '.', &lp, &ld, &lid);
        bool hr = split_next(r->pre, '.', &rp, &rd, &rid);
        if (!hl && hr) return -1;
        if (!hl && !hr) return 0;
        if (hl && !hr) return 1;
        size_t ln, rn;
        bool lnum = unsigned_num(lid, &ln), rnum = unsigned_num(rid, &rn);
        if (lnum && !rnum) return -1;
        if (!lnum && rnum) return 1;
        if (lnum && rnum) {
            if (ln < rn) return -1;
            if (ln > rn) return 1;
        } else {
            // std.mem.order: bytewise, then by length
            size_t n = MIN(lid.len, rid.len);
            int c = n ? memcmp(lid.ptr, rid.ptr, n) : 0;
            if (c != 0) return c < 0 ? -1 : 1;
            if (lid.len != rid.len) return lid.len < rid.len ? -1 : 1;
        }
    }
}

// --- update ------------------------------------------------------------------

#define REPO "antonellof/saetta"
#define ASSET "saetta-macos-arm64.tar.gz"

typedef struct {
    int status;
    Buf body;
} Got;

static int sink_buf(void *ctx, int status, const void *p, size_t n) {
    buf_add(ctx, p, n);
    return 0;
}

static _Thread_local char http_err[256];

static Err get(const char *url, Got *g) {
    sx_header h[2] = {{"accept-encoding", "identity"}, {"user-agent", "saetta"}};
    char err[256] = "";
    *g = (Got){0};
    if (sx_http("GET", url, h, 2, NULL, 0, 300.0, sink_buf, &g->body, &g->status, err, sizeof err) != 0) {
        buf_free(&g->body);
        snprintf(http_err, sizeof http_err, "%s", err[0] ? err : "HttpFailed");
        return http_err;
    }
    return OK;
}

// std.process.executablePathAlloc: _NSGetExecutablePath, symlinks resolved.
static Err exe_path(Arena *a, char **out) {
    char sym[PATH_MAX + 1];
    uint32_t n = sizeof sym;
    if (_NSGetExecutablePath(sym, &n) != 0) return "NameTooLong";
    char real[PATH_MAX];
    if (!realpath(sym, real)) return err_errno(errno);
    *out = arena_strdup(a, real);
    return OK;
}

static int rm_one(const char *p, const struct stat *st, int flag, struct FTW *ftw) { return remove(p); }

// std.Io.Dir.deleteTree, errors ignored.
static void delete_tree(const char *path) { nftw(path, rm_one, 16, FTW_DEPTH | FTW_PHYS); }

// Runs argv with stdin, stdout and stderr on /dev/null; true when it exited with 0.
static Err run_quiet(const char *const *argv, bool *success) {
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    pid_t pid;
    int rc = posix_spawn(&pid, argv[0], &fa, NULL, (char *const *)argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) return err_errno(rc);
    int st;
    while (waitpid(pid, &st, 0) < 0)
        if (errno != EINTR) return err_errno(errno);
    *success = WIFEXITED(st) && WEXITSTATUS(st) == 0;
    return OK;
}

static Err update_in(Arena *a, FILE *out, const char *current, int *code, Got *latest, Got *tgz, Got *sum) {
    // a Homebrew install is Homebrew's to replace (its Cellar, its links)
    char *self;
    if (!exe_path(a, &self) && strstr(self, "/Cellar/saetta/") != NULL) {
        fprintf(out, "saetta update: installed with Homebrew (%s); run: brew upgrade saetta\n", self);
        *code = 1;
        return OK;
    }
    // the latest release tag, from the GitHub API
    TRY(get("https://api.github.com/repos/" REPO "/releases/latest", latest));
    if (latest->status != 200) {
        fprintf(out, "saetta update: could not read the latest release (HTTP %d)%s\n", latest->status,
                latest->status == 404 ? "; the repository may be private" : "");
        *code = 1;
        return OK;
    }
    JValue *v;
    TRY(json_parse(a, str_n(arena_dup(a, buf_str(&latest->body).ptr, latest->body.len + 1), latest->body.len), true, &v));
    const char *tag = jstr(v, "tag_name");
    const char *ver = tag[0] == 'v' ? tag + 1 : tag;
    if (!ver[0]) return "BadRelease";
    SemanticVersion cur, nv;
    if (semver_parse(str(current), &cur)) return "BadVersion";
    if (semver_parse(str(ver), &nv)) return "BadRelease";
    if (semver_order(&nv, &cur) != 1) {
        fprintf(out, "saetta %s is the latest release\n", current);
        *code = 0;
        return OK;
    }
    fprintf(out, "saetta update: %s -> %s, downloading...\n", current, ver);
    fflush(out);
    const char *base = "https://github.com/" REPO "/releases/download/";
    TRY(get(arena_printf(a, "%sv%s/%s", base, ver, ASSET), tgz));
    TRY(get(arena_printf(a, "%sv%s/%s.sha256", base, ver, ASSET), sum));
    if (tgz->status != 200 || sum->status != 200) {
        fprintf(out, "saetta update: download failed (HTTP %d / %d)\n", tgz->status, sum->status);
        *code = 1;
        return OK;
    }
    uint8_t d[SHA256_LEN];
    sha256(tgz->body.ptr ? tgz->body.ptr : "", tgz->body.len, d);
    char hex[2 * SHA256_LEN + 1];
    hash_hex(d, sizeof d, hex);
    if (sum->body.len < 64 || strncasecmp(sum->body.ptr, hex, 64) != 0) {
        fprintf(out, "saetta update: checksum mismatch, nothing changed\n");
        *code = 1;
        return OK;
    }
    // unpack with the system tar into a temporary folder next to this binary, then rename
    char *exe;
    TRY(exe_path(a, &exe));
    char *slash = strrchr(exe, '/');
    const char *dir = slash ? (slash == exe ? "/" : arena_strndup(a, exe, (size_t)(slash - exe))) : ".";
    char *tmp = arena_printf(a, "%s/.saetta-update-%d", dir, (int)getpid());
    TRY(mkdir_p(tmp));
    Err e = OK;
    char *tpath = arena_printf(a, "%s/%s", tmp, ASSET);
    e = write_file(tpath, tgz->body.ptr ? tgz->body.ptr : "", tgz->body.len);
    if (e) goto out;
    const char *argv[] = {"/usr/bin/tar", "-xzf", tpath, "-C", tmp, NULL};
    bool ok = false;
    e = run_quiet(argv, &ok);
    if (e) goto out;
    if (!ok) {
        e = "UnpackFailed";
        goto out;
    }
    char *newbin = arena_printf(a, "%s/saetta-%s/bin/saetta", tmp, ver);
    if (rename(newbin, exe) != 0) {
        e = err_errno(errno);
        goto out;
    }
    fprintf(out, "saetta update: installed %s at %s\n", ver, exe);
    *code = 0;
out:
    delete_tree(tmp);
    return e;
}

Err control_update(FILE *out, const char *current, int *code) {
    Arena a = {0};
    Got latest = {0}, tgz = {0}, sum = {0};
    Err e = update_in(&a, out, current, code, &latest, &tgz, &sum);
    buf_free(&latest.body);
    buf_free(&tgz.body);
    buf_free(&sum.body);
    arena_reset(&a);
    return e;
}
