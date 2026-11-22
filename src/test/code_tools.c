// Tests for src/code_tools.c against pi 1.5.2's own behaviour: each scenario's scripted tool
// calls run through code_tool_run in a fresh workspace, and the toolResult messages, normalized
// as the captures are normalized, must equal the session lines pi wrote, byte for byte.
// SAETTA_CODE_GOLDEN is the folder of pi captures, one per scenario (skipped when unset); the scenarios are read from
// SAETTA_CODE_SCENARIOS, else the scenarios folder beside it.
#include "../code.h"
#include "../metal/shim.h"
#include "../test.h"

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

static const char *SCENARIOS[] = {
    "read-answer", "read-truncated", "read-missing", "parallel-calls", "edit-ok", "edit-multi", "edit-nomatch",
    "edit-ambiguous", "edit-missing-file", "write-new", "bash-ok", "bash-fail", "bash-timeout", "json-tools",
    "tools-args", "tools-read", "tools-edit", "tools-bash", "tools-search",
};

static Str replace_all(Arena *a, Str s, Str from, Str to) {
    Buf b = {0};
    size_t pos = 0;
    for (;;) {
        ptrdiff_t i = str_find(s, from, pos);
        if (i < 0) break;
        buf_add(&b, s.ptr + pos, (size_t)i - pos);
        buf_addstr(&b, to);
        pos = (size_t)i + from.len;
    }
    buf_add(&b, s.ptr + pos, s.len - pos);
    Str r = {arena_strndup(a, b.ptr ? b.ptr : "", b.len), b.len};
    buf_free(&b);
    return r;
}

static bool is_hex(char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }

// "key":<digits> -> "key":0, and pi-bash-<16 hex> -> pi-bash-$RANDOM (a random temp file name, as the captures hold it)
static Str normalize(Arena *a, Str s, const char *ws, const char *ws_real, const char *home, const char *home_real) {
    // the longer (real) path first: it contains the other
    s = replace_all(a, s, str(ws_real), S("$WS"));
    s = replace_all(a, s, str(ws), S("$WS"));
    s = replace_all(a, s, str(home_real), S("$HOME"));
    s = replace_all(a, s, str(home), S("$HOME"));
    Buf b = {0};
    for (size_t i = 0; i < s.len;) {
        if (str_starts(str_n(s.ptr + i, s.len - i), S("\"durationMs\":"))) {
            buf_adds(&b, "\"durationMs\":0");
            i += 13;
            while (i < s.len && s.ptr[i] >= '0' && s.ptr[i] <= '9') i++;
            continue;
        }
        if (str_starts(str_n(s.ptr + i, s.len - i), S("pi-bash-")) && i + 24 <= s.len) {
            bool hex = true;
            for (size_t k = 8; k < 24; k++) hex = hex && is_hex(s.ptr[i + k]);
            if (hex) {
                buf_adds(&b, "pi-bash-$RANDOM");
                i += 24;
                continue;
            }
        }
        buf_addc(&b, s.ptr[i++]);
    }
    Str r = {arena_strndup(a, b.ptr ? b.ptr : "", b.len), b.len};
    buf_free(&b);
    return r;
}

static Str write_json(Arena *a, const JValue *v) {
    Buf b = {0};
    json_write(&b, v);
    Str r = {arena_strndup(a, b.ptr, b.len), b.len};
    buf_free(&b);
    return r;
}

// The JSON lines of a file whose objects pass keep, re-serialized (numbers kept verbatim).
static void golden_lines(Arena *a, const char *path, const char *field, VecStr *out, bool tool_results) {
    char *text;
    size_t len;
    if (read_file(path, 1 << 28, &text, &len)) return;
    Str s = str_n(text, len);
    for (size_t pos = 0; pos < s.len;) {
        ptrdiff_t nl = str_find_char(s, '\n', pos);
        size_t end = nl < 0 ? s.len : (size_t)nl;
        JValue *v;
        if (end > pos && !json_parse(a, str_n(s.ptr + pos, end - pos), false, &v)) {
            if (tool_results) {
                JValue *m = json_get(v, "message");
                if (m && str_eqz(json_strs(json_get(m, "role")), "toolResult")) vec_push(out, write_json(a, m));
            } else if (str_eqz(json_strs(json_get(v, "type")), field)) vec_push(out, write_json(a, json_get(v, "result")));
        }
        pos = end + 1;
    }
    free(text);
}

static int run_scenario(const char *golden, const char *scen_dir, const char *name, size_t *count) {
    Arena a = {0};
    int bad = 0;
    char *text;
    size_t len;
    const char *sp = arena_printf(&a, "%s/%s.json", scen_dir, name);
    if (read_file(sp, 1 << 26, &text, &len)) {
        dbg("%s: cannot read %s\n", name, sp);
        return 1;
    }
    char tmpl[] = "/tmp/sct-XXXXXX";
    char *root = mkdtemp(tmpl);
    const char *ws = arena_printf(&a, "%s/ws", root), *home = arena_printf(&a, "%s/home", root);
    mkdir_p(ws);
    mkdir_p(arena_printf(&a, "%s/.pi/agent", home));
    char real[4096];
    const char *ws_real = arena_strdup(&a, realpath(ws, real) ? real : ws);
    const char *home_real = arena_strdup(&a, realpath(home, real) ? real : home);
    Str src = replace_all(&a, str_n(text, len), S("$WS"), str(ws));
    src = replace_all(&a, src, S("$HOME"), str(home));
    free(text);
    JValue *sc;
    if (json_parse(&a, src, true, &sc)) {
        dbg("%s: bad scenario\n", name);
        return 1;
    }
    JValue *files = json_get(sc, "files");
    for (size_t i = 0; files && i < files->obj.len; i++) {
        Str p = files->obj.items[i].key;
        const char *dst = str_starts(p, S("~/")) ? arena_printf(&a, "%s/%.*s", home, (int)p.len - 2, p.ptr + 2)
                                                 : arena_printf(&a, "%s/%.*s", ws, (int)p.len, p.ptr);
        char *dir = arena_strdup(&a, dst);
        *strrchr(dir, '/') = 0;
        mkdir_p(dir);
        Str body = json_strs(files->obj.items[i].val);
        write_file(dst, body.ptr, body.len);
    }
    VecStr want = {0}, want_raw = {0};
    golden_lines(&a, arena_printf(&a, "%s/%s/sessions/0.jsonl", golden, name), NULL, &want, true);
    golden_lines(&a, arena_printf(&a, "%s/%s/stdout-0.txt", golden, name), "tool_execution_end", &want_raw, false);
    CodeEnv env = {.cwd = ws_real, .home = home, .agent_dir = arena_printf(&a, "%s/.pi/agent", home), .now_ms = 0};
    size_t n = 0;
    JValue *runs = json_get(sc, "runs");
    for (size_t r = 0; runs && r < runs->arr.len; r++) {
        JValue *script = json_get(runs->arr.items[r], "script");
        for (size_t t = 0; script && t < script->arr.len; t++) {
            JValue *calls = json_get(script->arr.items[t], "tool_calls");
            for (size_t k = 0; calls && k < calls->arr.len; k++, n++) {
                JValue *c = calls->arr.items[k];
                JValue *call = json_new(&a, J_OBJECT);
                json_put(&a, call, S("type"), json_string(&a, S("toolCall")));
                json_put(&a, call, S("id"), json_string(&a, str(arena_printf(&a, "call_%zu_%zu", t + 1, k))));
                json_put(&a, call, S("name"), json_get(c, "name"));
                JValue *args = json_get(c, "arguments");
                if (json_is(args, J_STRING) && json_parse(&a, args->s, true, &args)) {
                    dbg("%s: call %zu has arguments that are not JSON; not supported here\n", name, n);
                    bad = 1;
                    continue;
                }
                json_put(&a, call, S("arguments"), args);
                JValue *res, *raw;
                if (code_tool_run(&a, &env, call, &res, &raw)) { bad = 1; continue; }
                Str got = normalize(&a, write_json(&a, res), ws, ws_real, home, home_real);
                Str exp = n < want.len ? normalize(&a, want.ptr[n], ws, ws_real, home, home_real) : S("<missing>");
                if (!str_eq(got, exp)) {
                    dbg("%s: call %zu differs\n  got  %.*s\n  want %.*s\n", name, n, (int)MIN(got.len, 3000), got.ptr, (int)MIN(exp.len, 3000), exp.ptr);
                    bad = 1;
                }
                if (want_raw.len) {
                    Str rg = normalize(&a, write_json(&a, raw), ws, ws_real, home, home_real);
                    Str re = n < want_raw.len ? normalize(&a, want_raw.ptr[n], ws, ws_real, home, home_real) : S("<missing>");
                    if (!str_eq(rg, re)) {
                        dbg("%s: raw result %zu differs\n  got  %.*s\n  want %.*s\n", name, n, (int)rg.len, rg.ptr, (int)re.len, re.ptr);
                        bad = 1;
                    }
                }
            }
        }
    }
    if (n != want.len) {
        dbg("%s: %zu calls, golden has %zu results\n", name, n, want.len);
        bad = 1;
    }
    *count = n;
    vec_free(&want);
    vec_free(&want_raw);
    char *cmd = arena_printf(&a, "rm -rf '%s'", root);
    if (system(cmd)) {}
    arena_reset(&a);
    return bad;
}

static void setup_path(void) {
    // pi finds rg and fd in its agent folder's bin; the scenarios were captured with it on PATH
    const char *h = getenv("HOME");
    const char *p = getenv("PATH");
    static char buf[16384];
    snprintf(buf, sizeof buf, "%s/.pi/agent/bin:%s", h ? h : "", p ? p : "");
    setenv("PATH", buf, 1);
    unsetenv("TMPDIR");
}

static const char *scen_dir(Arena *a, const char *golden) {
    const char *s = getenv("SAETTA_CODE_SCENARIOS");
    return s && *s ? s : arena_printf(a, "%s/../scenarios", golden);
}

TEST(code_tools_golden_pi) {
    const char *golden = getenv("SAETTA_CODE_GOLDEN");
    if (!golden || !*golden) SKIP("no $SAETTA_CODE_GOLDEN");
    setup_path();
    Arena a = {0};
    const char *sd = scen_dir(&a, golden);
    int bad = 0;
    for (size_t i = 0; i < COUNT(SCENARIOS); i++) {
        if (!file_exists(arena_printf(&a, "%s/%s/sessions/0.jsonl", golden, SCENARIOS[i]))) {
            dbg("code_tools: no golden for %s, skipped\n", SCENARIOS[i]);
            continue;
        }
        size_t n = 0;
        int b = run_scenario(golden, sd, SCENARIOS[i], &n);
        if (!b) dbg("code_tools: %s matches (%zu tool results)\n", SCENARIOS[i], n);
        bad |= b;
    }
    arena_reset(&a);
    EXPECT(!bad);
}

TEST(code_tools_schemas) {
    const char *golden = getenv("SAETTA_CODE_GOLDEN");
    if (!golden || !*golden) SKIP("no $SAETTA_CODE_GOLDEN");
    Arena a = {0};
    const char *cases[][8] = {{"read-answer", "read", "bash", "edit", "write"},
                              {"tools-search", "read", "bash", "edit", "write", "grep", "find", "ls"}};
    for (size_t c = 0; c < COUNT(cases); c++) {
        char *text;
        size_t len;
        if (read_file(arena_printf(&a, "%s/%s/requests.jsonl", golden, cases[c][0]), 1 << 26, &text, &len)) continue;
        Str s = str_n(text, len);
        ptrdiff_t nl = str_find_char(s, '\n', 0);
        JValue *req;
        EXPECT_OK(json_parse(&a, str_n(text, nl < 0 ? len : (size_t)nl), false, &req));
        free(text);
        size_t n = 0;
        while (n < 7 && cases[c][n + 1]) n++;
        JValue *mine = code_tool_schemas(&a, &cases[c][1], n);
        EXPECT_STR(write_json(&a, mine), write_json(&a, json_get(req, "tools")));
    }
    arena_reset(&a);
}

// ---- abort and large directories (no golden needed)

typedef struct {
    volatile int *flag;
    int after_ms;
    bool kill_detached; // call code_tools_kill_detached instead of setting the flag
} Aborter;

static void *aborter(void *p) {
    Aborter *ab = p;
    usleep((useconds_t)ab->after_ms * 1000);
    if (ab->kill_detached) code_tools_kill_detached();
    else *ab->flag = 1;
    return NULL;
}

static JValue *bash_call(Arena *a, const char *cmd) {
    JValue *call = json_new(a, J_OBJECT);
    json_put(a, call, S("type"), json_string(a, S("toolCall")));
    json_put(a, call, S("id"), json_string(a, S("call_1")));
    json_put(a, call, S("name"), json_string(a, S("bash")));
    JValue *args = json_new(a, J_OBJECT);
    json_put(a, args, S("command"), json_string(a, str(cmd)));
    json_put(a, call, S("arguments"), args);
    return call;
}

static Str result_text(Arena *a, const JValue *res) {
    const JValue *c = json_get(res, "content");
    return c && c->arr.len ? json_strs(json_get(c->arr.items[0], "text")) : S("");
}

// Runs cmd (which writes its shell pid, the group id, to root/pid) and aborts it after 200 ms.
static JValue *abort_case(Arena *a, const char *root, const char *cmd, bool kill_detached, double *secs, bool *group_gone) {
    volatile int flag = 0;
    CodeEnv env = {.cwd = root, .home = root, .agent_dir = arena_printf(a, "%s/agent", root), .abort = &flag};
    Aborter ab = {.flag = &flag, .after_ms = 200, .kill_detached = kill_detached};
    pthread_t th;
    pthread_create(&th, NULL, aborter, &ab);
    uint64_t t0 = now_ns();
    JValue *res = NULL, *raw;
    if (code_tool_run(a, &env, bash_call(a, cmd), &res, &raw)) res = NULL;
    *secs = (double)(now_ns() - t0) / 1e9;
    pthread_join(th, NULL);
    char *text;
    size_t len;
    *group_gone = false;
    if (!read_file(arena_printf(a, "%s/pid", root), 64, &text, &len)) {
        pid_t g = (pid_t)atoi(text);
        free(text);
        // SIGKILL is delivered at once, but reaping a non-child takes the kernel a moment
        for (int i = 0; i < 50 && !*group_gone; i++) {
            *group_gone = g > 0 && kill(-g, 0) < 0 && errno == ESRCH;
            if (!*group_gone) usleep(10000);
        }
    }
    return res;
}

TEST(code_tools_bash_abort) {
    Arena a = {0};
    char tmpl[] = "/tmp/sct-abort-XXXXXX";
    char *root = mkdtemp(tmpl);
    char real[4096];
    root = arena_strdup(&a, realpath(root, real) ? real : root);
    const char *pidf = arena_printf(&a, "%s/pid", root);
    double secs;
    bool gone;
    // a silent command with a background job in its group: pi's text, everything killed
    JValue *res = abort_case(&a, root, arena_printf(&a, "sleep 30 & echo $$ > '%s'; sleep 30", pidf), false, &secs, &gone);
    EXPECT(res);
    EXPECT_STR(result_text(&a, res), S("Command aborted"));
    EXPECT(json_get(res, "isError")->b);
    dbg("code_tools: silent bash aborted after %.3f s\n", secs);
    EXPECT(secs < 1.0);
    EXPECT(gone);
    // partial output is kept above the status, its last newline too (as in the timeout golden)
    res = abort_case(&a, root, arena_printf(&a, "echo partial; echo $$ > '%s'; sleep 30", pidf), false, &secs, &gone);
    EXPECT(res);
    EXPECT_STR(result_text(&a, res), S("partial\n\n\nCommand aborted"));
    EXPECT(secs < 1.0);
    EXPECT(gone);
    // code_tools_kill_detached kills a running command's group (the shell then dies of SIGKILL)
    res = abort_case(&a, root, arena_printf(&a, "echo $$ > '%s'; sleep 30", pidf), true, &secs, &gone);
    EXPECT(res);
    EXPECT_STR(result_text(&a, res), S("(no output)\n\nCommand exited with code 137"));
    EXPECT(secs < 1.0);
    EXPECT(gone);
    // a call made while the flag is set does not run
    volatile int flag = 1;
    CodeEnv env = {.cwd = root, .home = root, .agent_dir = root, .abort = &flag};
    const char *marker = arena_printf(&a, "%s/ran", root);
    JValue *raw;
    EXPECT_OK(code_tool_run(&a, &env, bash_call(&a, arena_printf(&a, "touch '%s'", marker)), &res, &raw));
    EXPECT_STR(result_text(&a, res), S("Operation aborted"));
    EXPECT(json_get(res, "isError")->b);
    EXPECT(!json_get(res, "durationMs"));
    EXPECT(!file_exists(marker));
    if (system(arena_printf(&a, "rm -rf '%s'", root))) {}
    arena_reset(&a);
}

TEST(code_tools_ls_large) {
    Arena a = {0};
    char tmpl[] = "/tmp/sct-ls-XXXXXX";
    char *root = mkdtemp(tmpl);
    char real[4096];
    root = arena_strdup(&a, realpath(root, real) ? real : root);
    const char *dir = arena_printf(&a, "%s/big", root);
    mkdir_p(dir);
    // mixed case, digits and punctuation, so the locale order differs from the byte order
    static const char *stems[] = {"Alpha", "beta", "_gamma", "Delta-", "eps.ilon", "10x", "9x", "zeta", "Ägir", "ä"};
    for (int i = 0; i < 2000; i++) {
        const char *p = arena_printf(&a, "%s/%s%d", dir, stems[i % COUNT(stems)], i);
        write_file(p, "", 0);
    }
    CodeEnv env = {.cwd = root, .home = root, .agent_dir = root};
    JValue *call = json_new(&a, J_OBJECT);
    json_put(&a, call, S("type"), json_string(&a, S("toolCall")));
    json_put(&a, call, S("id"), json_string(&a, S("call_1")));
    json_put(&a, call, S("name"), json_string(&a, S("ls")));
    JValue *args = json_new(&a, J_OBJECT);
    json_put(&a, args, S("path"), json_string(&a, S("big")));
    json_put(&a, args, S("limit"), json_int(&a, 5000));
    json_put(&a, call, S("arguments"), args);
    uint64_t t0 = now_ns();
    JValue *res, *raw;
    EXPECT_OK(code_tool_run(&a, &env, call, &res, &raw));
    double secs = (double)(now_ns() - t0) / 1e9;
    dbg("code_tools: ls of 2000 entries in %.3f s\n", secs);
    EXPECT(secs < 1.0);
    Str t = result_text(&a, res);
    size_t lines = 0;
    Str prev = {0};
    bool ordered = true;
    for (size_t pos = 0; pos < t.len;) {
        ptrdiff_t nl = str_find_char(t, '\n', pos);
        Str l = str_n(t.ptr + pos, (nl < 0 ? t.len : (size_t)nl) - pos);
        if (prev.ptr && sx_locale_compare_lower(prev.ptr, prev.len, l.ptr, l.len) > 0) ordered = false;
        prev = l;
        lines++;
        pos = nl < 0 ? t.len : (size_t)nl + 1;
    }
    EXPECT_EQ(lines, 2000);
    EXPECT(ordered);
    if (system(arena_printf(&a, "rm -rf '%s'", root))) {}
    arena_reset(&a);
}
