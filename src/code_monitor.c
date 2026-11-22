// Saetta Code's /monitor (see code_monitor.h). Saetta Code's own command, not a port: the frame
// is `saetta monitor`'s (monitor_frame), only the data source and the keys differ.
#include "code_monitor.h"

#include <pthread.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>

#include "instance.h"
#include "monitor.h"
#include "server.h"
#include "stats.h"

#define SAMPLE_NS 500000000ull // `saetta monitor` redraws every 500 ms

// The model, as code_main.c last set it.
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static struct Engine *g_engine;
static char *g_provider, *g_base_url;

// The view (UI thread only).
typedef enum { SRC_NONE, SRC_ENGINE, SRC_SERVER, SRC_SAETTA_DOWN } Source;
static struct {
    bool open, frozen;
    MonitorView view;
    uint64_t next_ns;
    Arena a;      // the last sample's snapshot and sessions
    Source src;
    const JValue *m;
    MonitorSys sys;
    MonitorHists hist;
    Client *sessions;
    size_t n_sessions;
    char provider[128];
} M;

void code_monitor_set_model(struct Engine *engine, const char *provider, const char *base_url) {
    pthread_mutex_lock(&g_lock);
    g_engine = engine;
    free(g_provider);
    free(g_base_url);
    g_provider = provider ? strdup(provider) : NULL;
    g_base_url = base_url ? strdup(base_url) : NULL;
    pthread_mutex_unlock(&g_lock);
}

// The port of a base URL on this Mac ("http://127.0.0.1:8080/v1"), 0 for another host.
static unsigned local_port(const char *url) {
    if (!url) return 0;
    const char *h = strstr(url, "://");
    bool https = !strncasecmp(url, "https", 5);
    h = h ? h + 3 : url;
    static const char *const local[] = {"127.0.0.1", "localhost", "[::1]", "0.0.0.0"};
    for (size_t i = 0; i < COUNT(local); i++) {
        size_t n = strlen(local[i]);
        if (strncasecmp(h, local[i], n)) continue;
        const char *p = h + n;
        if (*p == ':') return (unsigned)strtoul(p + 1, NULL, 10);
        if (*p == 0 || *p == '/') return https ? 443 : 80;
    }
    return 0;
}

// The Mac's numbers; SAETTA_CODE_MONITOR_SYSTEM=off leaves them out (zero), for the TUI tests,
// whose screens must not depend on this machine's load.
static MonitorSys sample_sys(void) {
    const char *v = getenv("SAETTA_CODE_MONITOR_SYSTEM");
    if (v && !strcmp(v, "off")) return (MonitorSys){0};
    return monitor_sample();
}

static void sample(void) {
    arena_reset(&M.a);
    M.m = NULL;
    M.sessions = NULL;
    M.n_sessions = 0;
    pthread_mutex_lock(&g_lock);
    struct Engine *e = g_engine;
    unsigned port = local_port(g_base_url);
    snprintf(M.provider, sizeof M.provider, "%s", g_provider ? g_provider : "");
    pthread_mutex_unlock(&g_lock);
    Server srv;
    if (e) M.src = SRC_ENGINE;
    else if (port && instance_running(&srv) && srv.port == port) M.src = SRC_SERVER;
    else M.src = !strcmp(M.provider, "saetta") ? SRC_SAETTA_DOWN : SRC_NONE;
    if (M.src == SRC_NONE) return;
    M.sys = sample_sys();
    if (M.src == SRC_ENGINE) {
        // the engine's counters, read as the server's snapshot thread reads them
        Buf j = {0};
        stats_json(server_engine_stats(e), &j);
        Str t = str_n(arena_dup(&M.a, j.ptr, j.len + 1), j.len);
        buf_free(&j);
        JValue *m;
        if (!json_parse(&M.a, t, true, &m)) M.m = m;
        M.sessions = ARENA_NEW(&M.a, Client);
        M.sessions[0] = (Client){.pid = (int32_t)getpid(), .agent = "code"};
        M.n_sessions = 1;
    } else if (M.src == SRC_SERVER) {
        const char *home = getenv("HOME");
        if (home) M.m = monitor_read_snapshot(&M.a, arena_printf(&M.a, "%s/.saetta/run/metrics.json", home));
        M.n_sessions = instance_clients(&M.a, &M.sessions);
    }
    monitor_push(&M.hist, M.m, &M.sys);
}

void code_monitor_open(void) {
    M.open = true;
    M.frozen = false;
    M.view = MONITOR_REQUESTS;
    M.hist = (MonitorHists){0};
    sample();
    M.next_ns = now_ns() + SAMPLE_NS;
}

bool code_monitor_is_open(void) { return M.open; }

bool code_monitor_key(const Key *k) {
    if (!M.open) return false;
    if (k->kind == K_ESC || (k->kind == K_CTRL && k->ch == 'c') || (k->kind == K_CHAR && (k->ch == 'q' || k->ch == 'Q'))) {
        M.open = false;
        arena_reset(&M.a);
        M.m = NULL;
        M.sessions = NULL;
    } else if (k->kind == K_CHAR && k->ch == ' ') M.frozen = !M.frozen;
    else if (k->kind == K_CHAR && k->ch == 'r') M.view = MONITOR_REQUESTS;
    else if (k->kind == K_CHAR && k->ch == 'c') M.view = MONITOR_CONVERSATIONS;
    else if (k->kind == K_CHAR && k->ch == 'l') M.view = MONITOR_LOG;
    return true;
}

int code_monitor_wait_ms(void) {
    if (!M.open) return -1;
    uint64_t now = now_ns();
    return M.next_ns > now ? (int)((M.next_ns - now) / 1000000) + 1 : 0;
}

bool code_monitor_tick(void) {
    if (!M.open) return false;
    uint64_t now = now_ns();
    if (now < M.next_ns) return false;
    M.next_ns = now + SAMPLE_NS;
    if (M.frozen) return false;
    sample();
    return true;
}

static const char KEYS[] = "r requests  ·  c conversations  ·  l log  ·  space freeze  ·  esc close";

void code_monitor_render(int width, int height, Lines *out) {
    Arena a = {0};
    Buf o = {0};
    MonitorFrame f = {
        .m = M.m, .sys = M.sys, .hist = &M.hist, .w = (size_t)MAX(width, 0), .h = (size_t)MAX(height, 0), .view = M.view,
        .sessions = M.sessions, .n_sessions = M.n_sessions, .inproc = M.src == SRC_ENGINE,
        .offline = "no saetta server is running", .offline_hint = "this provider's server keeps no statistics saetta can read",
        .waiting = M.src == SRC_SAETTA_DOWN ? "the monitor shows a model run in this process (/model) or a saetta server on this Mac" : NULL,
        .keys = KEYS,
    };
    if (M.src == SRC_NONE) {
        // another provider: its server keeps no statistics saetta can read
        f.unavailable = arena_printf(&a, "The monitor is not available for this provider (%s).", M.provider);
        f.unavailable_hint = "It shows a model run in this process (/model) or a saetta server on this Mac.";
        f.keys = "esc close";
    }
    monitor_frame(&a, &o, &f);
    Str s = buf_str(&o);
    size_t i = 0;
    while (i < s.len && out->n < (size_t)height) {
        size_t j = i;
        while (j < s.len && s.ptr[j] != '\n') j++;
        Buf line = {0};
        tui_truncate(&line, str_n(s.ptr + i, j - i), width, "", false);
        lines_addb(out, &line);
        buf_free(&line);
        i = j + 1;
    }
    buf_free(&o);
    arena_reset(&a);
}
