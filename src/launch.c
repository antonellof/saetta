// `saetta start` (see launch.h).
#include "launch.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <netinet/in.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "ctxlimits.h"
#include "fit.h"
#include "download.h"
#include "instance.h"
#include "metal/shim.h"

extern char **environ;

#define MODEL_ID "qwen3.8-flash-next"

// The model id the agent asks for and its display name: the server's base model, or with
// --lora the adapter's name (the server picks the adapter by the request's model).
static const char *model_id = MODEL_ID;
static const char *model_label = "Qwen3.8-Flash-Next (saetta)";
// --lora name=path, passed to serve as --lora-modules (thinking off by default, as the pi
// adapter was trained)
static const char *lora_arg = NULL;

const char *launch_usage(void) {
    return "usage: saetta start <claude|pi|opencode|code> [--pack coder|full|27b-turbo|27b-coder390|27b-fable5|27b-8bit|27b-4bit|<dir>] [--lora name=path] [--port N] "
           "[--ctx N] [--keep] [-- <agent args>]\n";
}

typedef enum { AGENT_CLAUDE, AGENT_PI, AGENT_OPENCODE, AGENT_CODE } Agent;
static const char *agent_names[] = {"claude", "pi", "opencode", "code"};

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

const char *launch_install_command(const char *agent) {
    if (strcmp(agent, "claude") == 0) return "curl -fsSL https://claude.ai/install.sh | bash";
    if (strcmp(agent, "pi") == 0) return "curl -fsSL https://pi.dev/install.sh | sh";
    if (strcmp(agent, "opencode") == 0) return "curl -fsSL https://opencode.ai/install | bash";
    return NULL;
}

static const char *install_hint(Agent agent) {
    static char b[160];
    static const char *const label[] = {"Claude Code", "pi", "opencode", "Saetta Code"};
    const char *cmd = launch_install_command(agent_names[agent]);
    snprintf(b, sizeof b, "Install %s: %s", label[agent], cmd ? cmd : "it is part of saetta");
    return b;
}

static bool has_config(Arena *a, const char *dir) {
    struct stat st;
    return stat(path_join(a, dir, "config.json"), &st) == 0;
}

// The pack folder: --pack (name or path), else the first downloaded pack that fits this Mac's
// memory, in download table order (coder, full, then the 27B packs).
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
        dbg("saetta: no model at %s; download it with: saetta download %s\n", dir, pk ? arg : (fit_default_pack() ? fit_default_pack() : "27b-4bit"));
        return OK;
    }
    // the first downloaded pack that fits this Mac, in table order: coder, full, then the 27B packs
    const uint64_t ram = fit_ram_bytes();
    bool any = false;
    for (size_t i = 0; i < DOWNLOAD_NPACKS; i++) {
        const DownloadPack *p = &download_packs[i];
        if (p->min_ram_gb == 0) continue; // adapters
        char *dir;
        TRY(download_pack_dir(a, p, &dir));
        if (!has_config(a, dir)) continue;
        any = true;
        if (ram && ((uint64_t)p->min_ram_gb << 30) > ram) continue;
        *out = dir;
        return OK;
    }
    const char *rec = fit_default_pack();
    if (any) dbg("saetta: no downloaded model fits this Mac's memory (%llu GB).\n", (unsigned long long)(ram >> 30));
    if (rec)
        dbg("saetta: %s Run: saetta download %s\n", any ? "" : "no model downloaded yet.", rec);
    else
        fit_suggest(ram);
    return OK;
}

// Whether any model pack that fits this Mac is downloaded.
static bool any_pack(Arena *a) {
    const uint64_t ram = fit_ram_bytes();
    for (size_t i = 0; i < DOWNLOAD_NPACKS; i++) {
        const DownloadPack *p = &download_packs[i];
        char *dir;
        if (!p->min_ram_gb || (ram && ((uint64_t)p->min_ram_gb << 30) > ram)) continue;
        if (!download_pack_dir(a, p, &dir) && has_config(a, dir)) return true;
    }
    return false;
}

typedef struct {
    char b[256];
    size_t n;
    bool overflow;
} Sink;

static int sink_fixed(void *ctx, int status, const void *p, size_t n) {
    Sink *s = ctx;
    // keeps what fits; the rest stops the transfer
    size_t room = sizeof s->b - s->n, k = n < room ? n : room;
    memcpy(s->b + s->n, p, k);
    s->n += k;
    if (k < n) {
        s->overflow = true;
        return 1;
    }
    return 0;
}

static bool healthy_within(const char *base, double timeout_s) {
    char url[256];
    if ((size_t)snprintf(url, sizeof url, "%s/health", base) >= sizeof url) return false;
    Sink s = {0};
    int status = 0;
    char err[256];
    if (sx_http("GET", url, NULL, 0, NULL, 0, timeout_s, sink_fixed, &s, &status, err, sizeof err) != 0) return false;
    return status == 200;
}

// The context a running server serves: "ctx" in its metrics file (~/.saetta/run/metrics.json,
// written after the memory cap, readable while the server is busy), else max_model_len from
// GET /v1/models; 0 when unknown.
static uint32_t server_context(Arena *a, const char *base) {
    const char *home = getenv("HOME");
    char *path = path_join(a, path_join(a, path_join(a, home ? home : "", ".saetta"), "run"), "metrics.json");
    char *text;
    size_t len;
    if (!read_file(path, 1 << 20, &text, &len)) {
        // only the running server's file (a stopped one leaves its last)
        Server srv;
        uint64_t v = 0, pid = 0;
        const char *k = strstr(text, "\"ctx\":"), *kp = strstr(text, "\"pid\":");
        if (k)
            for (const char *c = k + 6; *c >= '0' && *c <= '9'; c++) v = v * 10 + (uint64_t)(*c - '0');
        if (kp)
            for (const char *c = kp + 6; *c >= '0' && *c <= '9'; c++) pid = pid * 10 + (uint64_t)(*c - '0');
        free(text);
        if (v > 0 && v <= UINT32_MAX && instance_running(&srv) && pid == (uint64_t)srv.pid) return (uint32_t)v;
    }
    char url[256];
    if ((size_t)snprintf(url, sizeof url, "%s/v1/models", base) >= sizeof url) return 0;
    Sink s = {0};
    int status = 0;
    char err[256];
    // the sink keeps the first 256 bytes, which hold the base model's entry
    (void)sx_http("GET", url, NULL, 0, NULL, 0, 10.0, sink_fixed, &s, &status, err, sizeof err);
    if (status != 200 && !s.overflow) return 0;
    const char *k = "\"max_model_len\":";
    size_t kl = strlen(k);
    for (size_t i = 0; i + kl < s.n; i++)
        if (memcmp(s.b + i, k, kl) == 0) {
            uint64_t v = 0;
            for (size_t j = i + kl; j < s.n && s.b[j] >= '0' && s.b[j] <= '9'; j++) v = v * 10 + (uint64_t)(s.b[j] - '0');
            return v > UINT32_MAX ? 0 : (uint32_t)v;
        }
    return 0;
}

// A running server: no timeout, a busy server answers after its current request.
static bool healthy(const char *base) { return healthy_within(base, 86400.0); }

// Whether 127.0.0.1:port can be bound (the server binds it the same way, with SO_REUSEADDR).
static bool port_free(uint16_t port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return true; // let the server report it
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in sa = {0};
    sa.sin_family = AF_INET;
    sa.sin_port = htons(port);
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    bool ok = bind(fd, (struct sockaddr *)&sa, sizeof sa) == 0 || errno != EADDRINUSE;
    close(fd);
    return ok;
}

// The last non-empty line of a small log file, into out ("" if none).
static void last_line(const char *path, char *out, size_t cap) {
    out[0] = 0;
    char *data;
    size_t len;
    if (read_file(path, 1 << 20, &data, &len)) return;
    while (len > 0 && (data[len - 1] == '\n' || data[len - 1] == '\r')) len--;
    size_t start = len;
    while (start > 0 && data[start - 1] != '\n') start--;
    snprintf(out, cap, "%.*s", (int)(len - start), data + start);
    free(data);
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

static Err start_server(Arena *a, const char *dir, uint16_t port, uint32_t ctx, pid_t *pid, char **log_out) {
    char *exe;
    TRY(exe_path(a, &exe));
    const char *home = getenv("HOME");
    if (!home) return "NoHome";
    char *log_dir = path_join(a, path_join(a, path_join(a, home, "Library"), "Logs"), "saetta");
    TRY(mkdir_p(log_dir));
    char *log_path = arena_printf(a, "%s/serve-%u.log", log_dir, port);
    *log_out = log_path;
    int log = open(log_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
    if (log < 0) return err_errno(errno);
    Env senv;
    env_clone(&senv, a);
    env_put(&senv, "SAETTA_CTX", arena_printf(a, "%u", ctx));
    env_put(&senv, "SAETTA_AUTOSTOP", "1"); // stopped by the last `saetta start` session
    char *port_s = arena_printf(a, "%u", port);
    const char *argv[10] = {exe, "serve", dir, "--port", port_s};
    if (lora_arg) {
        argv[5] = "--lora-modules";
        argv[6] = lora_arg;
        argv[7] = "--default-chat-template-kwargs";
        argv[8] = "{\"enable_thinking\":false}";
    }
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
        {"ANTHROPIC_MODEL", model_id},
        {"ANTHROPIC_DEFAULT_OPUS_MODEL", model_id},
        {"ANTHROPIC_DEFAULT_SONNET_MODEL", model_id},
        {"ANTHROPIC_DEFAULT_HAIKU_MODEL", model_id},
        {"ANTHROPIC_DEFAULT_FABLE_MODEL", model_id},
        {"CLAUDE_CODE_SUBAGENT_MODEL", model_id},
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
    const char *args[] = {"claude", "--model", model_id, "--settings", s};
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
        "        \"name\": \"%s\",\n"
        "        \"reasoning\": %s,\n"
        "        \"input\": [\"text\"],\n"
        "        \"contextWindow\": %u,\n"
        "        \"maxTokens\": 32768,\n"
        "        \"cost\": { \"input\": 0, \"output\": 0, \"cacheRead\": 0, \"cacheWrite\": 0 }\n"
        "      }]\n"
        "    }\n"
        "  }\n"
        "}\n",
        base, model_id, model_label, lora_arg ? "false" : "true", ctx);
    TRY(write_in(a, dir, "models.json", models));
    TRY(write_in(a, dir, "settings.json",
                 arena_printf(a, "{\n  \"defaultProvider\": \"saetta\",\n  \"defaultModel\": \"%s\"\n}\n", model_id)));
    env_put(env, "PI_CODING_AGENT_DIR", dir);
    env_put(env, "PI_SKIP_VERSION_CHECK", "1");
    const char *args[] = {"pi", "--provider", "saetta", "--model", model_id};
    for (size_t k = 0; k < COUNT(args); k++) vec_push(argv, args[k]);
    return OK;
}

static Err code_in_process(Arena *a, const char *dir, uint32_t ctx, const char *const *extra, size_t n_extra, int *code);

// Saetta Code (`saetta code`, this binary): pi's configuration, so the two share settings,
// models and sessions under ~/.saetta/pi.
static Err code_setup(Arena *a, Env *env, Argv *argv, const char *base, uint32_t ctx) {
    TRY(pi_setup(a, env, argv, base, ctx));
    char self[PATH_MAX];
    uint32_t len = sizeof self;
    if (_NSGetExecutablePath(self, &len) != 0) return "NoExecutablePath";
    argv->ptr[0] = arena_strdup(a, self);
    // pi_setup pushed {"pi", "--provider", "saetta", "--model", id}: insert "code" after the binary
    vec_push(argv, NULL);
    memmove(&argv->ptr[2], &argv->ptr[1], (argv->len - 2) * sizeof argv->ptr[0]);
    argv->ptr[1] = "code";
    return OK;
}

// saetta code --pack <dir> [extra], the engine in its own process, waited for.
static Err code_in_process(Arena *a, const char *dir, uint32_t ctx, const char *const *extra, size_t n_extra, int *code) {
    const char *home = getenv("HOME");
    if (!home) return "NoHome";
    char *cfg = path_join(a, path_join(a, home, ".saetta"), "pi");
    TRY(mkdir_p(cfg));
    char self[PATH_MAX];
    uint32_t len = sizeof self;
    if (_NSGetExecutablePath(self, &len) != 0) return "NoExecutablePath";
    Env env;
    env_clone(&env, a);
    env_put(&env, "PI_CODING_AGENT_DIR", cfg);
    Argv argv = {0};
    vec_push(&argv, arena_strdup(a, self));
    vec_push(&argv, "code");
    if (dir) {
        vec_push(&argv, "--pack");
        vec_push(&argv, dir);
        vec_push(&argv, "--ctx");
        vec_push(&argv, arena_printf(a, "%u", ctx));
    }
    for (size_t k = 0; k < n_extra; k++) vec_push(&argv, extra[k]);
    vec_push(&argv, NULL);
    pid_t child;
    int rc = posix_spawn(&child, argv.ptr[0], NULL, NULL, (char *const *)argv.ptr, env_block(&env));
    if (rc != 0) {
        vec_free(&argv);
        return err_errno(rc);
    }
    ignore_interrupt(); // Ctrl-C belongs to Saetta Code now
    int st = wait_child(child);
    vec_free(&argv);
    if (st < 0) return err_errno(errno);
    *code = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
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
        "          \"name\": \"%s\",\n"
        "          \"limit\": { \"context\": %u, \"output\": 8192 },\n"
        "          \"options\": { \"temperature\": 0 }\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "}\n",
        model_id, model_id, base, model_id, model_label, ctx);
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
static Err session(Arena *a, Session *ss, Agent agent, const char *pack_arg, uint16_t port, bool port_set,
                   uint32_t ctx, const char *const *extra, size_t n_extra, const char *base, int *code) {
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
        int ipid = 0;
        if (instance_inproc_running(&ipid)) {
            dbg("saetta: Saetta Code (pid %d) is running the model in its own process; one model runs at a time. Close it "
                "first, or run it against a server (start the server first, then saetta start code joins it).\n", ipid);
            *code = 1;
            return OK;
        }
        const char *dir;
        // Saetta Code in a terminal with nothing downloaded: it guides the first run itself (the
        // model that fits this Mac, downloaded on confirmation), or uses a server of its models.json
        if (agent == AGENT_CODE && !pack_arg && !lora_arg && isatty(0) && isatty(1) && !any_pack(a))
            return code_in_process(a, NULL, ctx, extra, n_extra, code);
        TRY(pack_for(a, pack_arg, &dir));
        if (!dir) {
            *code = 1;
            return OK;
        }
        // Saetta Code without a server: the model runs inside it (saetta code --pack), with the
        // configuration and sessions `saetta start pi` uses
        // (an adapter needs the server: it serves the adapter under its own name)
        if (agent == AGENT_CODE && !lora_arg) return code_in_process(a, dir, ctx, extra, n_extra, code);
        // another program on the port: the default moves to the next free one, a chosen one is an error
        if (!port_free(port)) {
            if (port_set) {
                dbg("saetta: port %u is in use by another program; choose another with --port N or SAETTA_PORT\n", port);
                *code = 1;
                return OK;
            }
            uint16_t p = port;
            while (p < port + 100 && p < UINT16_MAX && !port_free(p)) p++;
            if (!port_free(p)) {
                dbg("saetta: ports %u to %u are in use; choose one with --port N\n", port, p);
                *code = 1;
                return OK;
            }
            dbg("saetta: warning: port %u is in use by another program; using port %u\n", port, p);
            port = p;
            base = arena_printf(a, "http://127.0.0.1:%u", port);
        }
        char *log_path;
        TRY(start_server(a, dir, port, ctx, &ss->server, &log_path));
        // std.fs.path.basename: the last component, trailing slashes ignored
        size_t end = strlen(dir);
        while (end > 1 && dir[end - 1] == '/') end--;
        size_t start = end;
        while (start > 0 && dir[start - 1] != '/') start--;
        bool tty = isatty(2);
        uint64_t t0 = now_ns();
        dbg("saetta: loading %.*s (context %u tokens)...%s", (int)(end - start), dir + start, ctx, tty ? "" : "\n");
        // a loading server refuses connections; the short timeout covers a listener that never answers
        while (!healthy_within(base, 2.0)) {
            unsigned secs = (unsigned)((now_ns() - t0) / 1000000000ull);
            int st;
            if (waitpid(ss->server, &st, WNOHANG) == ss->server) {
                ss->server = 0;
                char line[512];
                last_line(log_path, line, sizeof line);
                dbg("%ssaetta: the server stopped while loading%s%s\nsaetta: log: %s\n", tty ? "\n" : "",
                    line[0] ? ": " : "", line, log_path);
                *code = 1;
                return OK;
            }
            if (secs >= 300) {
                dbg("%ssaetta: the server did not come up in 5 minutes; see %s\n", tty ? "\n" : "", log_path);
                stop_server(ss->server);
                ss->server = 0;
                *code = 1;
                return OK;
            }
            if (tty) dbg("\rsaetta: loading %.*s (context %u tokens)... %u s", (int)(end - start), dir + start, ctx, secs);
            sleep_ms(500);
        }
        if (tty) dbg("\n");
        dbg("saetta: ready at %s (%.1f s)\n", base, (double)(now_ns() - t0) / 1e9);
    }

    // the context the server actually serves (it can be smaller than asked: this Mac's memory)
    uint32_t served = server_context(a, base);
    if (!served) dbg("saetta: warning: could not read the server's context; the agent is told %u tokens\n", ctx);
    if (served && served < ctx) {
        dbg("saetta: the server's context is %u tokens (this Mac's memory); the agent is told so\n", served);
        ctx = served;
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
    case AGENT_CODE: e = code_setup(a, &cenv, &argv, base, ctx); break;
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
    bool port_set = false; // chosen by the user: never moved to another port
    const char *v = getenv("SAETTA_PORT");
    if (v) {
        uint16_t p;
        if (!parse_u16(v, &p)) port = p, port_set = true;
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
        } else if (i + 1 < n && strcmp(x, "--lora") == 0) {
            i += 1;
            const char *eq = strchr(args[i], '=');
            if (!eq || eq == args[i] || !eq[1]) {
                dbg("saetta: --lora wants name=path, got %s\n", args[i]);
                *code = 64;
                return OK;
            }
            lora_arg = args[i];
        } else if (i + 1 < n && strcmp(x, "--port") == 0) {
            i += 1;
            TRY(parse_u16(args[i], &port));
            port_set = true;
        } else if (i + 1 < n && strcmp(x, "--ctx") == 0) {
            i += 1;
            TRY(parse_u32(args[i], &ctx));
        } else {
            extra = args + i; // anything else goes to the agent
            n_extra = n - i;
            break;
        }
    }
    // the server's rule (session_ctx_parse in src/model.h, limits in src/ctxlimits.h)
    ctx -= ctx % SESSION_CTX_ALIGN;
    if (ctx < SESSION_CTX_MIN || ctx > SESSION_CTX_QWEN_MAX) {
        dbg("saetta: --ctx / SAETTA_CTX wants %d to %zu tokens\n", SESSION_CTX_MIN, SESSION_CTX_QWEN_MAX);
        *code = 64;
        return OK;
    }
    // the model the agent asks for: the adapter's name with --lora, else the pack's base id
    bool is27 = false;
    if (pack_arg) {
        const char *dir;
        TRY(pack_for(a, pack_arg, &dir));
        if (dir) {
            char *cfg;
            size_t cl;
            if (!read_file(path_join(a, dir, "config.json"), 4u << 20, &cfg, &cl)) {
                is27 = strstr(cfg, "\"qwen3_5_text\"") != NULL;
                free(cfg);
            }
        }
    }
    if (lora_arg) {
        if (!is27) {
            dbg("saetta: --lora needs a Qwen3.8-27B pack, e.g. --pack 27b-8bit\n");
            *code = 64;
            return OK;
        }
        const char *eq = strchr(lora_arg, '=');
        model_id = arena_strndup(a, lora_arg, (size_t)(eq - lora_arg));
        model_label = arena_printf(a, "%s (Qwen3.8-27B LoRA, saetta)", model_id);
    } else if (is27) {
        model_id = "qwen3.8-27b";
        model_label = "Qwen3.8-27B (saetta)";
    }
    // the server: join the one running on this machine, else start one
    TRY(instance_register(agent_names[agent]));
    instance_set_keep(keep);
    Server s;
    if (instance_running(&s)) {
        if (s.port != port) dbg("saetta: note: the running server is on port %u, not %u\n", s.port, port);
        port = s.port;
        if (pack_arg) dbg("saetta: note: --pack ignored, the running server keeps its model\n");
        if (lora_arg) {
            // the agent would ask for an adapter the running server may not have
            dbg("saetta: a server is already running (pid %d); --lora needs a new one: stop it first (saetta stop)\n", s.pid);
            (void)instance_unregister();
            *code = 1;
            return OK;
        }
    }
    const char *base = arena_printf(a, "http://127.0.0.1:%u", port);
    Session ss = {.keep = keep};
    Err e = session(a, &ss, agent, pack_arg, port, port_set, ctx, extra, n_extra, base, code);
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
