// `saetta monitor` (see monitor.h).
#include "monitor.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/sysctl.h>
#include <sys/wait.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include "instance.h"
#include "json.h"
#include "metal/shim.h"
#include "stats.h"

extern char **environ;

// colors (truecolor): saetta purple and orange, from the logo
#define PURPLE "\x1b[38;2;167;112;255m"
#define ORANGE "\x1b[38;2;242;140;40m"
#define GREEN "\x1b[38;2;80;220;150m"
#define RED "\x1b[38;2;240;90;90m"
#define DIM "\x1b[38;2;120;120;140m"
#define WHITE "\x1b[38;2;230;230;240m"
#define BOLD "\x1b[1m"
#define RESET "\x1b[0m"
#define BADGE "\x1b[48;2;138;63;252m\x1b[38;2;255;255;255m\x1b[1m"

typedef MonitorHist Hist;
typedef MonitorSys Sys;

static void hist_push(Hist *h, double x) {
    if (h->n == COUNT(h->v)) {
        memmove(h->v, h->v + 1, (COUNT(h->v) - 1) * sizeof(double));
        h->n -= 1;
    }
    h->v[h->n++] = x;
}

typedef struct {
    size_t w, h;
} Size;

static Err write_all(const char *s, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t n = write(1, s + off, len - off);
        if (n <= 0) return "WriteFailed";
        off += (size_t)n;
    }
    return OK;
}

static Size size(void) {
    struct winsize ws = {0};
    if (ioctl(1, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) return (Size){ws.ws_col, ws.ws_row};
    return (Size){120, 40};
}

// std.process.run: stdout captured up to limit (more is an error); stdin and stderr closed
// to /dev/null.
static bool run_capture(const char *const *argv, size_t limit, Buf *out) {
    int p[2];
    if (pipe(p) != 0) return false;
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&fa, p[1], 1);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addclose(&fa, p[0]);
    posix_spawn_file_actions_addclose(&fa, p[1]);
    pid_t pid;
    int rc = posix_spawn(&pid, argv[0], &fa, NULL, (char *const *)argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    close(p[1]);
    if (rc != 0) {
        close(p[0]);
        return false;
    }
    bool ok = true;
    char b[16384];
    for (;;) {
        ssize_t n = read(p[0], b, sizeof b);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        if (out->len + (size_t)n > limit) {
            ok = false;
            kill(pid, SIGKILL);
            break;
        }
        buf_add(out, b, (size_t)n);
    }
    close(p[0]);
    int st;
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
    return ok;
}

static size_t digits_len(Str v) {
    size_t e = 0;
    while (e < v.len && isdigit((unsigned char)v.ptr[e])) e++;
    return e;
}

MonitorSys monitor_sample(void) {
    Sys s = {.thermal = sx_thermal_state()};
    // GPU: "Device Utilization %"=N in the IOAccelerator's PerformanceStatistics
    Buf r = {0};
    const char *ioreg[] = {"/usr/sbin/ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator", NULL};
    if (run_capture(ioreg, 1 << 20, &r)) {
        Str t = buf_str(&r);
        Str key = S("\"Device Utilization %\"=");
        ptrdiff_t i = str_find(t, key, 0);
        if (i >= 0) {
            Str v = str_sub(t, (size_t)i + key.len, t.len);
            size_t e = digits_len(v);
            if (e > 0) { // parseFloat of digits only: empty is an error
                char tmp[64];
                snprintf(tmp, sizeof tmp, "%.*s", (int)MIN(e, sizeof tmp - 1), v.ptr);
                s.gpu_util = strtod(tmp, NULL);
                s.has_gpu = true;
            }
        }
    }
    buf_free(&r);
    uint64_t total = 0;
    size_t len = sizeof total;
    if (sysctlbyname("hw.memsize", &total, &len, NULL, 0) == 0) s.ram_total = total;
    // used memory as Activity Monitor counts it: app (active) + wired + compressed
    Buf vm = {0};
    const char *vm_stat[] = {"/usr/bin/vm_stat", NULL};
    if (run_capture(vm_stat, 1 << 16, &vm)) {
        Str t = buf_str(&vm);
        uint64_t page = 16384;
        ptrdiff_t i = str_find(t, S("page size of "), 0);
        if (i >= 0) {
            Str v = str_sub(t, (size_t)i + 13, t.len);
            uint64_t x;
            if (!parse_u64(str_n(v.ptr, digits_len(v)), &x)) page = x;
        }
        static const char *keys[] = {"Pages active:", "Pages wired down:", "Pages occupied by compressor:"};
        uint64_t used = 0;
        for (size_t k = 0; k < COUNT(keys); k++) {
            Str key = str(keys[k]);
            ptrdiff_t j = str_find(t, key, 0);
            if (j < 0) continue;
            Str v = str_sub(t, (size_t)j + key.len, t.len);
            while (v.len > 0 && v.ptr[0] == ' ') v = str_sub(v, 1, v.len);
            uint64_t x;
            if (parse_u64(str_n(v.ptr, digits_len(v)), &x)) x = 0;
            used += x * page;
        }
        s.ram_used = used;
    }
    buf_free(&vm);
    struct xsw_usage sw;
    len = sizeof sw;
    if (sysctlbyname("vm.swapusage", &sw, &len, NULL, 0) == 0) s.swap_used = sw.xsu_used;
    return s;
}

static const JValue *get(const JValue *v, const char *k) { return json_get(v, k); }

static double num(const JValue *v, const char *k) {
    const JValue *x = get(v, k);
    if (!x) return 0;
    if (x->kind == J_INTEGER) return (double)x->i;
    if (x->kind == J_FLOAT) return x->f;
    return 0;
}

static const char *jstr(const JValue *v, const char *k) {
    const char *s = json_str(get(v, k));
    return s ? s : "";
}

// Memory in GB as macOS reports it (binary: 128 GB of RAM is 128).
static double gb(double b) { return b / (1024.0 * 1024.0 * 1024.0); }

// Zig "{d:.N}" into the arena.
static const char *fx(Arena *a, double x, int prec) {
    char b[400];
    fmt_fixed(b, sizeof b, x, prec);
    return arena_strdup(a, b);
}

// "19.5K" style token counts.
static const char *kfmt(Arena *a, double x) {
    if (x >= 1e6) return arena_printf(a, "%sM", fx(a, x / 1e6, 1));
    if (x >= 1e4) return arena_printf(a, "%sK", fx(a, x / 1e3, 1));
    return fx(a, x, 0);
}

// Printable width of s (ANSI sequences skipped, UTF-8 counted by code point).
static size_t vis(Str s) {
    size_t n = 0, i = 0;
    while (i < s.len) {
        if (s.ptr[i] == 0x1b) {
            while (i < s.len && !isalpha((unsigned char)s.ptr[i])) i++;
            i += 1;
            continue;
        }
        if (((unsigned char)s.ptr[i] & 0xC0) != 0x80) n += 1;
        i += 1;
    }
    return n;
}

// Plain text cut to at most w code points.
static Str fit(Str s, size_t w) {
    size_t n = 0;
    for (size_t i = 0; i < s.len; i++) {
        if (((unsigned char)s.ptr[i] & 0xC0) != 0x80) {
            if (n == w) return str_n(s.ptr, i);
            n += 1;
        }
    }
    return s;
}

static void pad(Buf *o, Str s, size_t w) {
    buf_addstr(o, s);
    size_t v = vis(s);
    for (size_t k = v; k < w; k++) buf_addc(o, ' ');
}

static const char *bar(Arena *a, double frac, size_t w, const char *color) {
    Buf o = {0};
    double f = fmax(0, fmin(frac, 1));
    size_t full = (size_t)round(f * (double)w);
    buf_adds(&o, color);
    for (size_t k = 0; k < full; k++) buf_adds(&o, "█");
    buf_adds(&o, DIM);
    for (size_t k = full; k < w; k++) buf_adds(&o, "░");
    buf_adds(&o, RESET);
    char *r = arena_strndup(a, o.ptr, o.len);
    buf_free(&o);
    return r;
}

static const char *spark(Arena *a, const Hist *h, size_t w, double max_hint, const char *color) {
    static const char *ticks[] = {" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
    Buf o = {0};
    double mx = max_hint;
    for (size_t i = 0; i < h->n; i++) mx = fmax(mx, h->v[i]);
    buf_adds(&o, color);
    size_t start = h->n > w ? h->n - w : 0;
    size_t shown = h->n - start;
    for (size_t k = 0; k < w - shown; k++) buf_addc(&o, ' ');
    for (size_t i = start; i < h->n; i++) {
        size_t k = mx <= 0 ? 0 : (size_t)round(fmax(0, fmin(h->v[i] / mx, 1)) * 8);
        buf_adds(&o, ticks[k]);
    }
    buf_adds(&o, RESET);
    char *r = arena_strndup(a, o.ptr, o.len);
    buf_free(&o);
    return r;
}

typedef struct {
    const char *title, *value, *sub, *graph;
} Card;

// A row of cards side by side, each `cw` columns wide (5 lines).
static void cards(Arena *a, Buf *o, const Card *row, size_t n, size_t cw) {
    size_t inner = cw - 4;
    for (int line = 0; line < 5; line++) {
        buf_addc(o, ' ');
        for (size_t c = 0; c < n; c++) {
            const Card *cd = &row[c];
            switch (line) {
            case 0:
                buf_printf(o, "%s╭─ %s%s%s ", DIM, WHITE, cd->title, DIM);
                for (size_t k = 0; k < cw - 5 - vis(str(cd->title)); k++) buf_adds(o, "─");
                buf_printf(o, "╮%s", RESET);
                break;
            case 4:
                buf_printf(o, "%s╰", DIM);
                for (size_t k = 0; k < cw - 2; k++) buf_adds(o, "─");
                buf_printf(o, "╯%s", RESET);
                break;
            default: {
                buf_printf(o, "%s│%s ", DIM, RESET);
                const char *t;
                if (line == 1) {
                    Str v = fit(str(cd->value), inner);
                    t = arena_printf(a, "%s%.*s%s", BOLD, (int)v.len, v.ptr, RESET);
                } else if (line == 2) {
                    Str v = fit(str(cd->sub), inner);
                    t = arena_printf(a, "%s%.*s%s", DIM, (int)v.len, v.ptr, RESET);
                } else t = cd->graph;
                pad(o, str(t), inner);
                buf_printf(o, " %s│%s", DIM, RESET);
            }
            }
        }
        buf_addc(o, '\n');
    }
}

// HH:MM:SS local time of a unix timestamp.
static const char *clock_str(Arena *a, int64_t t) {
    time_t tt = (time_t)t;
    struct tm tm;
    if (!localtime_r(&tt, &tm)) return "--:--:--";
    return arena_printf(a, "%02u:%02u:%02u", (unsigned)tm.tm_hour, (unsigned)tm.tm_min, (unsigned)tm.tm_sec);
}

static void requests_panel(Arena *a, Buf *o, const JValue *m, size_t room) {
    // recent requests, newest first
    buf_printf(o, "\n %s%sRecent requests%s\n", BOLD, PURPLE, RESET);
    buf_printf(o, " %s%-10s%-11s%9s%9s%8s%9s%9s%10s  %s%s\n", DIM, "TIME", "API", "PROMPT", "REUSED", "OUTPUT", "TOK/S",
               "PREFILL", "DURATION", "FINISH", RESET);
    const JValue *rec = get(m, "recent");
    if (!json_is(rec, J_ARRAY)) return;
    size_t k = rec->arr.len;
    for (size_t shown = 0; k > 0 && shown < room; shown++) {
        k -= 1;
        const JValue *r = rec->arr.items[k];
        int64_t end = (int64_t)num(r, "end");
        double dec = num(r, "decode_ms");
        double outn = num(r, "output");
        const char *fin = jstr(r, "finish");
        const char *fc = strcmp(fin, "tool_calls") == 0 ? ORANGE : strcmp(fin, "length") == 0 ? RED : GREEN;
        buf_printf(o, " %-10s%-11s%9s%9s%8s%9s%9s%10s  %s%s%s\n", clock_str(a, end), jstr(r, "api"),
                   kfmt(a, num(r, "prompt")), kfmt(a, num(r, "reused")), fx(a, outn, 0),
                   fx(a, dec > 0 ? outn * 1000 / dec : 0, 1),
                   arena_printf(a, "%s s", fx(a, num(r, "prefill_ms") / 1000, 1)),
                   arena_printf(a, "%s s", fx(a, (num(r, "prefill_ms") + dec) / 1000, 1)), fc, fin, RESET);
    }
    if (rec->arr.len == 0) buf_printf(o, " %sno requests yet%s\n", DIM, RESET);
}

// The conversations the server keeps: tokens kept for the next turn, capacity, last use.
static void conversations_panel(Arena *a, Buf *o, const JValue *m, size_t room) {
    buf_printf(o, "\n %s%sConversations%s\n", BOLD, PURPLE, RESET);
    buf_printf(o, " %s%-6s%10s%11s%14s%s\n", DIM, "#", "TOKENS", "CAPACITY", "LAST USED", RESET);
    const JValue *cv = get(m, "conversations");
    if (!json_is(cv, J_ARRAY) || cv->arr.len == 0) {
        buf_printf(o, " %snone yet%s\n", DIM, RESET);
        return;
    }
    double now = num(m, "now");
    for (size_t i = 0; i < cv->arr.len; i++) {
        if (i >= room) break;
        const JValue *conv = cv->arr.items[i];
        const JValue *av = get(conv, "active");
        bool act = av && av->kind == J_BOOL && av->b;
        double ago = now - num(conv, "last");
        const char *last = num(conv, "last") <= 0 ? "-"
                           : ago < 60             ? arena_printf(a, "%s s ago", fx(a, ago, 0))
                                                  : arena_printf(a, "%s min ago", fx(a, ago / 60, 0));
        buf_printf(o, " %s%-6s%10s%11s%14s  %s%s\n", act ? GREEN : WHITE, arena_printf(a, "%zu", i + 1),
                   kfmt(a, num(conv, "tokens")), kfmt(a, num(conv, "cap")), last, act ? "active" : "", RESET);
    }
}

// The tail of the server log (~/Library/Logs/saetta/serve-<port>.log).
static void log_panel(Arena *a, Buf *o, const JValue *m, size_t room, size_t w, bool inproc) {
    if (inproc) {
        buf_printf(o, "\n %s%sServer log%s\n %sthe model runs in this process: no server log%s\n", BOLD, PURPLE, RESET, DIM, RESET);
        return;
    }
    uint16_t port = (uint16_t)num(m, "port");
    const char *home = getenv("HOME");
    char *path = arena_printf(a, "%s/Library/Logs/saetta/serve-%u.log", home ? home : "", port > 0 ? port : 8080);
    buf_printf(o, "\n %s%sServer log%s  %s%s%s\n", BOLD, PURPLE, RESET, DIM, path, RESET);
    char *text;
    size_t len;
    if (read_file(path, 4 << 20, &text, &len)) {
        buf_printf(o, " %sno log file (a server started by hand logs to its terminal)%s\n", DIM, RESET);
        return;
    }
    VecStr lines = {0};
    size_t s = 0;
    for (size_t i = 0; i <= len; i++) {
        if (i == len || text[i] == '\n') {
            if (i > s) vec_push(&lines, str_n(text + s, i - s));
            s = i + 1;
        }
    }
    size_t first = lines.len > room ? lines.len - room : 0;
    for (size_t i = first; i < lines.len; i++) {
        Str ln = fit(lines.ptr[i], w > 2 ? w - 2 : 0);
        buf_printf(o, " %.*s\n", (int)ln.len, ln.ptr);
    }
    vec_free(&lines);
    free(text);
}

JValue *monitor_read_snapshot(Arena *a, const char *path) {
    // server snapshot; stale or missing means no server
    JValue *m = NULL;
    char *text;
    size_t len;
    if (!read_file(path, 1 << 20, &text, &len)) {
        Str t = str_n(arena_dup(a, text, len + 1), len);
        free(text);
        if (t.len > 0 && json_parse(a, t, true, &m)) m = NULL;
    }
    int32_t pid = (int32_t)num(m, "pid");
    bool up = m && pid > 0 && kill(pid, 0) == 0 && instance_running(NULL);
    return up ? m : NULL; // the last server's numbers are stale
}

void monitor_push(MonitorHists *h, const JValue *m, const MonitorSys *sys) {
    hist_push(&h->tok, m ? num(m, "tok_s") : 0);
    hist_push(&h->gpu, sys->has_gpu ? sys->gpu_util : 0);
}

void monitor_frame(Arena *a, Buf *o, const MonitorFrame *f) {
    if (f->unavailable) {
        buf_printf(o, " %s saetta %s \n\n %s%s%s\n %s%s%s\n\n %s%s%s\n", BADGE, RESET, WHITE, f->unavailable, RESET, DIM,
                   f->unavailable_hint ? f->unavailable_hint : "", RESET, DIM, f->keys, RESET);
        return;
    }
    size_t w = MAX(f->w, 60);
    const JValue *m = f->m;
    bool up = m != NULL;
    Sys sys = f->sys;
    const Hist *tok_hist = &f->hist->tok, *gpu_hist = &f->hist->gpu;

    const char *state = up ? jstr(m, "state") : "offline";
    double tok_s = up ? num(m, "tok_s") : 0;

    // header
    buf_printf(o, " %s saetta %s ", BADGE, RESET);
    if (up) {
        double uptime = num(m, "now") - num(m, "started");
        const char *where = f->inproc ? "in this process" : arena_printf(a, "http://127.0.0.1:%s", fx(a, num(m, "port"), 0));
        buf_printf(o, " %s%s%s  %s%s%s  %s%s%s  %sup %sm%s", WHITE, jstr(m, "model"), RESET, PURPLE, jstr(m, "pack"), RESET,
                   DIM, where, RESET, DIM, fx(a, uptime / 60, 0), RESET);
    } else
        buf_printf(o, " %s%s%s  %s%s%s", WHITE, f->offline, RESET, DIM, f->offline_hint, RESET);
    buf_adds(o, "\n");
    buf_addc(o, '\n');

    // state strip and progress
    static const char *names[] = {"idle", "reading", "generating"};
    static const char *labels[] = {"Idle", "Reading prompt", "Generating"};
    buf_printf(o, " %s%sModel state%s  ", BOLD, PURPLE, RESET);
    for (size_t k = 0; k < 3; k++) {
        if (strcmp(names[k], state) == 0) buf_printf(o, "%s%s[%s]%s  ", BOLD, ORANGE, labels[k], RESET);
        else buf_printf(o, "%s%s%s  ", DIM, labels[k], RESET);
    }
    double frac = 0;
    const char *right = "";
    bool generating = strcmp(state, "generating") == 0;
    if (strcmp(state, "reading") == 0) {
        double tot = num(m, "prompt_total");
        double done = num(m, "prompt_done");
        frac = tot > 0 ? done / tot : 0;
        right = arena_printf(a, "%s / %s tokens read (%s reused)", kfmt(a, done), kfmt(a, tot), kfmt(a, num(m, "reused")));
    } else if (generating) {
        double g = num(m, "gen");
        double mx = num(m, "max_tokens");
        frac = mx > 0 ? g / mx : 0;
        right = arena_printf(a, "%s tokens out, %s tok/s (%s API)", fx(a, g, 0), fx(a, tok_s, 1), jstr(m, "api"));
    }
    buf_printf(o, "%s%s%s\n ", DIM, right, RESET);
    buf_adds(o, bar(a, frac, w - 2, generating ? GREEN : PURPLE));
    buf_adds(o, "\n\n");

    // cards
    size_t per = w >= 100 ? 4 : 2;
    size_t cw = (w - 2) / per;
    size_t gw = cw - 4;
    double reqs = num(m, "requests");
    double passes = num(m, "output_passes");
    double outs = num(m, "output_tokens");
    double ptoks = num(m, "prompt_tokens");
    double rtoks = num(m, "reused_tokens");
    double ctx = num(m, "ctx");
    double conv = 0;
    const JValue *ck = get(m, "ck");
    if (json_is(ck, J_ARRAY))
        for (size_t i = 0; i < ck->arr.len; i++)
            if (ck->arr.items[i]->kind == J_INTEGER) conv = fmax(conv, (double)ck->arr.items[i]->i);
    double galloc = num(m, "gpu_allocated");
    double gws = num(m, "gpu_working_set");
    static const char *thermal_names[] = {"nominal", "fair", "serious", "critical"};
    size_t th = (size_t)CLAMP(sys.thermal, 0, 3);
    const Client *sessions = f->sessions;
    size_t n_sessions = f->n_sessions;
    Buf who = {0};
    for (size_t i = 0; i < n_sessions; i++) {
        if (i > 0) buf_adds(&who, ", ");
        buf_adds(&who, sessions[i].agent);
        if (who.len > gw - 4) break;
    }
    const char *who_s = arena_strndup(a, who.ptr ? who.ptr : "", who.len);
    buf_free(&who);

    double ram_used = (double)sys.ram_used, ram_total = (double)sys.ram_total;
    const Card row1[4] = {
        {"Speed", arena_printf(a, "%s tok/s", fx(a, tok_s, 1)), generating ? "now" : "idle", spark(a, tok_hist, gw, 80, GREEN)},
        {"GPU load", sys.has_gpu ? arena_printf(a, "%s %%", fx(a, sys.gpu_util, 0)) : "n/a",
         arena_printf(a, "thermal %s", thermal_names[th]), spark(a, gpu_hist, gw, 100, PURPLE)},
        {"GPU memory", up ? arena_printf(a, "%s / %s GB", fx(a, gb(galloc), 1), fx(a, gb(gws), 0)) : "-",
         "saetta of the GPU limit", bar(a, gws > 0 ? galloc / gws : 0, gw, PURPLE)},
        {"System RAM", arena_printf(a, "%s / %s GB", fx(a, gb(ram_used), 1), fx(a, gb(ram_total), 0)),
         arena_printf(a, "swap %s GB", fx(a, gb((double)sys.swap_used), 1)),
         bar(a, sys.ram_total > 0 ? ram_used / ram_total : 0, gw, sys.swap_used > 1ull << 30 ? ORANGE : GREEN)},
    };
    const Card row2[4] = {
        {"Prompt cache", ptoks > 0 ? arena_printf(a, "%s %% reused", fx(a, 100 * rtoks / ptoks, 1)) : "-",
         arena_printf(a, "%s of %s prompt tokens", kfmt(a, rtoks), kfmt(a, ptoks)), bar(a, ptoks > 0 ? rtoks / ptoks : 0, gw, GREEN)},
        {"Speculation", passes > 0 ? arena_printf(a, "%s tokens/pass", fx(a, outs / passes, 2)) : "-",
         arena_printf(a, "MTP drafts, %s requests", fx(a, reqs, 0)), bar(a, passes > 0 ? (outs / passes - 1) / 3 : 0, gw, ORANGE)},
        {"Context", up ? arena_printf(a, "%s / %s", kfmt(a, conv), kfmt(a, ctx)) : "-",
         num(m, "convs") > 1 ? arena_printf(a, "%s conversations kept", fx(a, num(m, "convs"), 0)) : "kept for the next turn",
         bar(a, ctx > 0 ? conv / ctx : 0, gw, PURPLE)},
        {"Sessions", arena_printf(a, "%zu connected", n_sessions), n_sessions > 0 ? who_s : "saetta start <agent>", ""},
    };
    if (per == 4) {
        cards(a, o, row1, 4, cw);
        cards(a, o, row2, 4, cw);
    } else {
        cards(a, o, row1, 2, cw);
        cards(a, o, row1 + 2, 2, cw);
        cards(a, o, row2, 2, cw);
        cards(a, o, row2 + 2, 2, cw);
    }

    size_t used_lines = per == 4 ? 18 : 28;
    size_t room = f->h > used_lines + 2 ? f->h - used_lines - 2 : 3;
    switch (f->view) {
    case MONITOR_REQUESTS: requests_panel(a, o, m, room); break;
    case MONITOR_CONVERSATIONS: conversations_panel(a, o, m, room); break;
    case MONITOR_LOG: log_panel(a, o, m, room, w, f->inproc); break;
    }
    if (!up && f->waiting) buf_printf(o, " %s%s%s\n", DIM, f->waiting, RESET);
    buf_printf(o, "\n %s%s%s\n", DIM, f->keys, RESET);
}

// The monitor's own frame: the snapshot at path, the sessions registered on this Mac.
static void frame(Arena *a, Buf *o, const char *path, MonitorHists *hist, Size sz, MonitorView view) {
    Sys sys = monitor_sample();
    const JValue *m = monitor_read_snapshot(a, path);
    monitor_push(hist, m, &sys);
    Client *sessions;
    size_t n_sessions = instance_clients(a, &sessions);
    MonitorFrame f = {
        .m = m, .sys = sys, .hist = hist, .w = sz.w, .h = sz.h, .view = view, .sessions = sessions,
        .n_sessions = n_sessions, .offline = "no saetta server is running",
        .offline_hint = "start one with: saetta start claude", .waiting = "waiting for a server...",
        .keys = "r requests  ·  c conversations  ·  l log  ·  space freeze  ·  q quit (the server keeps running)",
    };
    monitor_frame(a, o, &f);
}

static Err live_loop(const char *path, MonitorHists *hist, int *code) {
    bool frozen = false;
    MonitorView view = MONITOR_REQUESTS;
    for (;;) {
        if (!frozen) {
            Arena a = {0};
            Buf body = {0};
            frame(&a, &body, path, hist, size(), view);
            // redrawn in place: clear the rest of each line, then everything below
            Buf out = {0};
            buf_adds(&out, "\x1b[H");
            for (size_t i = 0; i < body.len; i++) {
                if (body.ptr[i] == '\n') buf_adds(&out, "\x1b[K");
                buf_addc(&out, body.ptr[i]);
            }
            buf_adds(&out, "\x1b[J");
            Err e = write_all(out.ptr, out.len);
            buf_free(&out);
            buf_free(&body);
            arena_reset(&a);
            TRY(e);
        }
        // wait up to 500 ms for a key
        struct pollfd fds = {.fd = 0, .events = POLLIN, .revents = 0};
        if (poll(&fds, 1, 500) > 0) {
            char b[8];
            ssize_t n = read(0, b, sizeof b);
            if (n <= 0) {
                *code = 0;
                return OK;
            }
            for (ssize_t i = 0; i < n; i++) {
                switch (b[i]) {
                case 'q': case 'Q': case 3: case 27: *code = 0; return OK;
                case ' ': frozen = !frozen; break;
                case 'r': view = MONITOR_REQUESTS; break;
                case 'c': view = MONITOR_CONVERSATIONS; break;
                case 'l': view = MONITOR_LOG; break;
                default: break;
                }
            }
        }
    }
}

Err monitor_run(const char *const *args, size_t n, int *code) {
    bool once = false;
    MonitorView view_once = MONITOR_REQUESTS;
    for (size_t i = 0; i < n; i++) {
        if (strcmp(args[i], "--once") == 0) once = true;
        else if (strcmp(args[i], "--conversations") == 0) view_once = MONITOR_CONVERSATIONS;
        else if (strcmp(args[i], "--log") == 0) view_once = MONITOR_LOG;
        else {
            dbg("usage: saetta monitor [--once [--conversations | --log]]\n");
            *code = 64;
            return OK;
        }
    }
    const char *home = getenv("HOME");
    if (!home) return "NoHome";
    Arena pa = {0};
    const char *path = path_join(&pa, path_join(&pa, path_join(&pa, home, ".saetta"), "run"), "metrics.json");

    MonitorHists hist = {0};
    Err e = OK;
    if (once) {
        Arena a = {0};
        Buf out = {0};
        frame(&a, &out, path, &hist, size(), view_once);
        e = write_all(out.ptr ? out.ptr : "", out.len);
        buf_free(&out);
        arena_reset(&a);
        arena_reset(&pa);
        if (!e) *code = 0;
        return e;
    }

    // full screen: alternate buffer, hidden cursor, keys unbuffered
    struct termios saved;
    bool tty = tcgetattr(0, &saved) == 0;
    if (tty) {
        struct termios raw = saved;
        raw.c_lflag &= ~(tcflag_t)(ICANON | ECHO);
        tcsetattr(0, TCSANOW, &raw);
    }
    e = write_all("\x1b[?1049h\x1b[?25l", 14);
    if (!e) {
        e = live_loop(path, &hist, code);
        (void)write_all("\x1b[?25h\x1b[?1049l", 14);
    }
    if (tty) tcsetattr(0, TCSANOW, &saved);
    arena_reset(&pa);
    return e;
}
