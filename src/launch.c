// `saetta start` (see launch.h).
#include "launch.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "download.h"
#include "instance.h"
#include "metal/shim.h"

extern char **environ;

#define MODEL_ID "qwen3.8-flash-next"

const char *launch_usage(void) {
    return "usage: saetta start <claude|pi|opencode> [--pack coder|full|<dir>] [--port N] [--ctx N] [--keep] [-- <agent args>]\n";
}

typedef enum { AGENT_CLAUDE, AGENT_PI, AGENT_OPENCODE } Agent;
static const char *agent_names[] = {"claude", "pi", "opencode"};

// A copy of the environment ("K=V" strings in the arena), as std.process.Environ.Map.
typedef struct {
    Arena *a;
    VEC(char *) kv;
} Env;

static void env_clone(Env *e, Arena *a) {
    *e = (Env){.a = a};
    for (char **p = environ; *p; p++) vec_push(&e->kv, *p);
}

static ptrdiff_t env_find(const Env *e, const char *k) {
    size_t n = strlen(k);
    for (size_t i = 0; i < e->kv.len; i++)
        if (strncmp(e->kv.ptr[i], k, n) == 0 && e->kv.ptr[i][n] == '=') return (ptrdiff_t)i;
    return -1;
}

static void env_put(Env *e, const char *k, const char *v) {
    char *s = arena_printf(e->a, "%s=%s", k, v);
    ptrdiff_t i = env_find(e, k);
    if (i >= 0) e->kv.ptr[i] = s;
    else vec_push(&e->kv, s);
}

static void env_swap_remove(Env *e, const char *k) {
    ptrdiff_t i = env_find(e, k);
    if (i < 0) return;
    e->kv.ptr[i] = e->kv.ptr[e->kv.len - 1];
    e->kv.len--;
}

// NULL terminated, for posix_spawn.
static char **env_block(Env *e) {
    char **b = ARENA_ARRAY(e->a, char *, e->kv.len + 1);
    memcpy(b, e->kv.ptr, e->kv.len * sizeof(char *));
    b[e->kv.len] = NULL;
    return b;
}

static void env_free(Env *e) { vec_free(&e->kv); }

static void sleep_ms(long ms) {
    struct timespec ts = {ms / 1000, (ms % 1000) * 1000000};
    nanosleep(&ts, NULL);
}

static const char *install_hint(Agent agent) {
    switch (agent) {
    case AGENT_CLAUDE: return "Install Claude Code: curl -fsSL https://claude.ai/install.sh | bash";
    case AGENT_PI: return "Install pi: curl -fsSL https://pi.dev/install.sh | sh";
    default: return "Install opencode: curl -fsSL https://opencode.ai/install | bash";
    }
}

static bool has_config(Arena *a, const char *dir) {
    struct stat st;
    return stat(path_join(a, dir, "config.json"), &st) == 0;
}

// The pack folder: --pack (name or path), else the first downloaded of coder, full.
// *out NULL (after a message) when there is none.
static Err pack_for(Arena *a, const char *arg, const char **out) {
    *out = NULL;
    if (arg) {
        const DownloadPack *pk = download_find(arg);
        char *dir = (char *)arg;
        if (pk) TRY(download_pack_dir(a, pk, &dir));
        if (has_config(a, dir)) {
            *out = dir;
            return OK;
        }
        dbg("saetta: no model at %s; download it with: saetta download %s\n", dir, pk ? arg : "coder");
        return OK;
    }
    for (size_t i = 0; i < DOWNLOAD_NPACKS; i++) {
        char *dir;
        TRY(download_pack_dir(a, &download_packs[i], &dir));
        if (has_config(a, dir)) {
            *out = dir;
            return OK;
        }
    }
    dbg("saetta: no model downloaded yet. Run: saetta download coder\n");
    return OK;
}

typedef struct {
    char b[256];
    size_t n;
    bool overflow;
} Sink;

static int sink_fixed(void *ctx, int status, const void *p, size_t n) {
    Sink *s = ctx;
    if (s->n + n > sizeof s->b) {
        s->overflow = true;
        return 1;
    }
    memcpy(s->b + s->n, p, n);
    s->n += n;
    return 0;
}

static bool healthy(const char *base) {
    char url[256];
    if ((size_t)snprintf(url, sizeof url, "%s/health", base) >= sizeof url) return false;
    Sink s = {0};
    int status = 0;
    char err[256];
    // no timeout of its own in the Zig engine: a busy server answers after its request
    if (sx_http("GET", url, NULL, 0, NULL, 0, 86400.0, sink_fixed, &s, &status, err, sizeof err) != 0) return false;
    return status == 200;
}

static Err exe_path(Arena *a, char **out) {
    char sym[PATH_MAX + 1];
    uint32_t n = sizeof sym;
    if (_NSGetExecutablePath(sym, &n) != 0) return "NameTooLong";
    char real[PATH_MAX];
    if (!realpath(sym, real)) return err_errno(errno);
    *out = arena_strdup(a, real);
    return OK;
}

static Err start_server(Arena *a, const char *dir, uint16_t port, uint32_t ctx, pid_t *pid) {
    char *exe;
    TRY(exe_path(a, &exe));
    const char *home = getenv("HOME");
    if (!home) return "NoHome";
    char *log_dir = path_join(a, path_join(a, path_join(a, home, "Library"), "Logs"), "saetta");
    TRY(mkdir_p(log_dir));
    char *log_path = arena_printf(a, "%s/serve-%u.log", log_dir, port);
    int log = open(log_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
    if (log < 0) return err_errno(errno);
    Env senv;
    env_clone(&senv, a);
    env_put(&senv, "SAETTA_CTX", arena_printf(a, "%u", ctx));
    env_put(&senv, "SAETTA_AUTOSTOP", "1"); // stopped by the last `saetta start` session
    char *port_s = arena_printf(a, "%u", port);
    const char *argv[] = {exe, "serve", dir, "--port", port_s, NULL};
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&fa, log, 1);
    posix_spawn_file_actions_adddup2(&fa, log, 2);
    posix_spawnattr_t at;
    posix_spawnattr_init(&at);
    posix_spawnattr_setflags(&at, POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&at, 0); // its own process group: the terminal's Ctrl-C goes to the agent only
    int rc = posix_spawn(pid, exe, &fa, &at, (char *const *)argv, env_block(&senv));
    posix_spawnattr_destroy(&at);
    posix_spawn_file_actions_destroy(&fa);
    env_free(&senv);
    close(log);
    if (rc != 0) return err_errno(rc);
    return OK;
}

static int wait_child(pid_t pid) {
    int st;
    while (waitpid(pid, &st, 0) < 0)
        if (errno != EINTR) return -1;
    return st;
}

static void stop_server(pid_t pid) {
    kill(pid, SIGTERM);
    wait_child(pid);
}

// After the agent is running: Ctrl-C is the agent's, and a closed terminal (SIGHUP) ends the
// agent while this process survives to stop the server (the server's watchdog covers a
// launcher killed any other way).
static void ignore_interrupt(void) {
    struct sigaction act = {0};
    act.sa_handler = SIG_IGN;
    sigemptyset(&act.sa_mask);
    act.sa_flags = 0;
    sigaction(SIGINT, &act, NULL);
    sigaction(SIGHUP, &act, NULL);
}

typedef VEC(const char *) Argv;

static void claude_setup(Arena *a, Env *env, Argv *argv, const char *base, uint32_t ctx) {
    const char *ctx_s = arena_printf(a, "%u", ctx);
    const char *vars[][2] = {
        {"ANTHROPIC_BASE_URL", base},
        {"ANTHROPIC_AUTH_TOKEN", "saetta"},
        {"ANTHROPIC_MODEL", MODEL_ID},
        {"ANTHROPIC_DEFAULT_OPUS_MODEL", MODEL_ID},
        {"ANTHROPIC_DEFAULT_SONNET_MODEL", MODEL_ID},
        {"ANTHROPIC_DEFAULT_HAIKU_MODEL", MODEL_ID},
        {"ANTHROPIC_DEFAULT_FABLE_MODEL", MODEL_ID},
        {"CLAUDE_CODE_SUBAGENT_MODEL", MODEL_ID},
        {"CLAUDE_CODE_MAX_CONTEXT_TOKENS", ctx_s},
        {"CLAUDE_CODE_MAX_OUTPUT_TOKENS", "32000"},
        {"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1"},
        {"CLAUDE_CODE_DISABLE_TERMINAL_TITLE", "1"},
        {"CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "1"},
        {"CLAUDE_CODE_ATTRIBUTION_HEADER", "0"},
        {"DISABLE_AUTOUPDATER", "1"},
        {"DISABLE_TELEMETRY", "1"},
    };
    // an Anthropic key in the environment would be sent too and could prompt for approval
    env_swap_remove(env, "ANTHROPIC_API_KEY");
    Buf settings = {0};
    buf_adds(&settings, "{\"env\":{");
    for (size_t k = 0; k < COUNT(vars); k++) {
        env_put(env, vars[k][0], vars[k][1]);
        if (k > 0) buf_addc(&settings, ',');
        buf_printf(&settings, "\"%s\":\"%s\"", vars[k][0], vars[k][1]);
    }
    buf_adds(&settings, "}}");
    const char *s = arena_strndup(a, settings.ptr, settings.len);
    buf_free(&settings);
    const char *args[] = {"claude", "--model", MODEL_ID, "--settings", s};
    for (size_t k = 0; k < COUNT(args); k++) vec_push(argv, args[k]);
}

static Err write_in(Arena *a, const char *dir, const char *name, const char *data) {
    return write_file(path_join(a, dir, name), data, strlen(data));
}

static Err pi_setup(Arena *a, Env *env, Argv *argv, const char *base, uint32_t ctx) {
    const char *home = getenv("HOME");
    if (!home) return "NoHome";
    char *dir = path_join(a, path_join(a, home, ".saetta"), "pi");
    TRY(mkdir_p(dir));
    const char *models = arena_printf(a,
        "{\n"
        "  \"providers\": {\n"
        "    \"saetta\": {\n"
        "      \"baseUrl\": \"%s/v1\",\n"
        "      \"api\": \"openai-completions\",\n"
        "      \"apiKey\": \"saetta\",\n"
        "      \"compat\": {\n"
        "        \"supportsDeveloperRole\": false,\n"
        "        \"supportsStore\": false,\n"
        "        \"supportsReasoningEffort\": false,\n"
        "        \"maxTokensField\": \"max_tokens\",\n"
        "        \"thinkingFormat\": \"qwen-chat-template\"\n"
        "      },\n"
        "      \"models\": [{\n"
        "        \"id\": \"%s\",\n"
        "        \"name\": \"Qwen3.8-Flash-Next (saetta)\",\n"
        "        \"reasoning\": true,\n"
        "        \"input\": [\"text\"],\n"
        "        \"contextWindow\": %u,\n"
        "        \"maxTokens\": 32768,\n"
        "        \"cost\": { \"input\": 0, \"output\": 0, \"cacheRead\": 0, \"cacheWrite\": 0 }\n"
        "      }]\n"
        "    }\n"
        "  }\n"
        "}\n",
        base, MODEL_ID, ctx);
    TRY(write_in(a, dir, "models.json", models));
    TRY(write_in(a, dir, "settings.json", "{\n  \"defaultProvider\": \"saetta\",\n  \"defaultModel\": \"" MODEL_ID "\"\n}\n"));
    env_put(env, "PI_CODING_AGENT_DIR", dir);
    env_put(env, "PI_SKIP_VERSION_CHECK", "1");
    const char *args[] = {"pi", "--provider", "saetta", "--model", MODEL_ID};
    for (size_t k = 0; k < COUNT(args); k++) vec_push(argv, args[k]);
    return OK;
}

static Err opencode_setup(Arena *a, Env *env, Argv *argv, const char *base, uint32_t ctx) {
    const char *home = getenv("HOME");
    if (!home) return "NoHome";
    char *dir = path_join(a, path_join(a, home, ".saetta"), "opencode");
    TRY(mkdir_p(dir));
    const char *cfg = arena_printf(a,
        "{\n"
        "  \"$schema\": \"https://opencode.ai/config.json\",\n"
        "  \"model\": \"saetta/%s\",\n"
        "  \"small_model\": \"saetta/%s\",\n"
        "  \"provider\": {\n"
        "    \"saetta\": {\n"
        "      \"npm\": \"@ai-sdk/openai-compatible\",\n"
        "      \"name\": \"saetta (local)\",\n"
        "      \"options\": { \"baseURL\": \"%s/v1\", \"timeout\": false },\n"
        "      \"models\": {\n"
        "        \"%s\": {\n"
        "          \"name\": \"Qwen3.8-Flash-Next (saetta)\",\n"
        "          \"limit\": { \"context\": %u, \"output\": 8192 },\n"
        "          \"options\": { \"temperature\": 0 }\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "}\n",
        MODEL_ID, MODEL_ID, base, MODEL_ID, ctx);
    TRY(write_in(a, dir, "opencode.json", cfg));
    env_put(env, "OPENCODE_CONFIG", path_join(a, dir, "opencode.json"));
    vec_push(argv, "opencode");
    return OK;
}

static Err parse_u16(const char *s, uint16_t *out) {
    uint64_t v;
    TRY(parse_u64(str(s), &v));
    if (v > UINT16_MAX) return "Overflow";
    *out = (uint16_t)v;
    return OK;
}

static Err parse_u32(const char *s, uint32_t *out) {
    uint64_t v;
    TRY(parse_u64(str(s), &v));
    if (v > UINT32_MAX) return "Overflow";
    *out = (uint32_t)v;
    return OK;
}

// What the session needs after the server part, for the deferred cleanup.
typedef struct {
    bool keep;
    pid_t server; // 0: none started here
} Session;

// The part of run after registering (the Zig `defer` covers it: cleanup runs on every exit).
static Err session(Arena *a, Session *ss, Agent agent, const char *pack_arg, uint16_t port, uint32_t ctx,
                   const char *const *extra, size_t n_extra, const char *base, int *code) {
    // join the running server; if it is stopping (its last session just exited), start anew
    bool joined = false;
    Server s;
    if (instance_running(&s)) {
        dbg("saetta: a server is already running (pid %d); this session shares it at %s\n", s.pid, base);
        // a busy server answers /health after its current request; one still loading refuses
        joined = true;
        if (!healthy(base)) {
            while (!healthy(base)) {
                if (!instance_running(NULL)) {
                    dbg("saetta: that server has just stopped; starting a new one\n");
                    joined = false;
                    break;
                }
                sleep_ms(500);
            }
        }
    }
    if (!joined) {
        const char *dir;
        TRY(pack_for(a, pack_arg, &dir));
        if (!dir) {
            *code = 1;
            return OK;
        }
        TRY(start_server(a, dir, port, ctx, &ss->server));
        // std.fs.path.basename: the last component, trailing slashes ignored
        size_t end = strlen(dir);
        while (end > 1 && dir[end - 1] == '/') end--;
        size_t start = end;
        while (start > 0 && dir[start - 1] != '/') start--;
        dbg("saetta: loading %.*s (context %u tokens)...\n", (int)(end - start), dir + start, ctx);
        for (uint32_t waited = 0; !healthy(base); waited++) {
            if (waited >= 600) {
                dbg("saetta: the server did not come up in 5 minutes; see ~/Library/Logs/saetta/serve-%u.log\n", port);
                stop_server(ss->server);
                ss->server = 0;
                *code = 1;
                return OK;
            }
            sleep_ms(500);
        }
        dbg("saetta: ready at %s\n", base);
    }

    // the agent
    Env cenv;
    env_clone(&cenv, a);
    Argv argv = {0};
    Err e = OK;
    switch (agent) {
    case AGENT_CLAUDE: claude_setup(a, &cenv, &argv, base, ctx); break;
    case AGENT_PI: e = pi_setup(a, &cenv, &argv, base, ctx); break;
    case AGENT_OPENCODE: e = opencode_setup(a, &cenv, &argv, base, ctx); break;
    }
    if (e) goto out;
    for (size_t k = 0; k < n_extra; k++) vec_push(&argv, extra[k]);
    vec_push(&argv, NULL);
    pid_t child;
    // PATH search as std.process.spawn does, with this process's PATH (the child's is the same)
    int rc = posix_spawnp(&child, argv.ptr[0], NULL, NULL, (char *const *)argv.ptr, env_block(&cenv));
    if (rc != 0) {
        if (rc == ENOENT) {
            dbg("saetta: '%s' is not installed or not on PATH. %s\n", argv.ptr[0], install_hint(agent));
            *code = 127;
        } else e = err_errno(rc);
        goto out;
    }
    // the terminal's Ctrl-C belongs to the agent now; this process only waits for it
    ignore_interrupt();
    int st = wait_child(child);
    if (st < 0) {
        e = err_errno(errno);
        goto out;
    }
    *code = WIFEXITED(st) ? WEXITSTATUS(st) : 1;
out:
    vec_free(&argv);
    env_free(&cenv);
    return e;
}

static Err run_in(Arena *a, const char *const *args, size_t n, int *code) {
    if (n == 0) {
        dbg("%s", launch_usage());
        *code = 64;
        return OK;
    }
    Agent agent;
    size_t ai = 0;
    while (ai < COUNT(agent_names) && strcmp(args[0], agent_names[ai]) != 0) ai++;
    if (ai == COUNT(agent_names)) {
        dbg("saetta start: unknown agent '%s'\n%s", args[0], launch_usage());
        *code = 64;
        return OK;
    }
    agent = (Agent)ai;
    const char *pack_arg = NULL;
    uint16_t port = 8080;
    const char *v = getenv("SAETTA_PORT");
    if (v) {
        uint16_t p;
        if (!parse_u16(v, &p)) port = p;
    }
    uint32_t ctx = 204800;
    v = getenv("SAETTA_CTX");
    if (v) {
        uint32_t c;
        if (!parse_u32(v, &c)) ctx = c;
    }
    bool keep = false;
    const char *const *extra = NULL;
    size_t n_extra = 0;
    for (size_t i = 1; i < n; i++) {
        const char *x = args[i];
        if (strcmp(x, "--") == 0) {
            extra = args + i + 1;
            n_extra = n - i - 1;
            break;
        } else if (strcmp(x, "--keep") == 0) {
            keep = true;
        } else if (i + 1 < n && strcmp(x, "--pack") == 0) {
            i += 1;
            pack_arg = args[i];
        } else if (i + 1 < n && strcmp(x, "--port") == 0) {
            i += 1;
            TRY(parse_u16(args[i], &port));
        } else if (i + 1 < n && strcmp(x, "--ctx") == 0) {
            i += 1;
            TRY(parse_u32(args[i], &ctx));
        } else {
            extra = args + i; // anything else goes to the agent
            n_extra = n - i;
            break;
        }
    }
    // the server: join the one running on this machine, else start one
    TRY(instance_register(agent_names[agent]));
    instance_set_keep(keep);
    Server s;
    if (instance_running(&s)) {
        if (s.port != port) dbg("saetta: note: the running server is on port %u, not %u\n", s.port, port);
        port = s.port;
        if (pack_arg) dbg("saetta: note: --pack ignored, the running server keeps its model\n");
    }
    const char *base = arena_printf(a, "http://127.0.0.1:%u", port);
    Session ss = {.keep = keep};
    Err e = session(a, &ss, agent, pack_arg, port, ctx, extra, n_extra, base, code);
    // deferred: leave, and stop an auto-started server nobody else uses
    size_t others = instance_unregister();
    bool up = instance_running(&s);
    if (keep || !up || !s.autostop) {
        if (keep && up) dbg("saetta: server left running at http://127.0.0.1:%u (pid %d)\n", s.port, s.pid);
    } else if (others > 0) {
        dbg("saetta: %zu other session(s) still use the server; it keeps running\n", others);
    } else {
        kill(s.pid, SIGTERM);
        if (ss.server) wait_child(ss.server);
    }
    return e;
}

Err launch_run(const char *const *args, size_t n, int *code) {
    Arena a = {0};
    Err e = run_in(&a, args, n, code);
    arena_reset(&a);
    return e;
}
