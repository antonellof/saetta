// Tests for src/code_oai.c against pi 1.5.2's real behaviour (tests/code/golden-pi in
// saetta_private, $SAETTA_CODE_GOLDEN): every request body byte for byte, and every answer
// (the SSE bytes the mock sends, parsed without a socket and over HTTP) as pi's assistant
// message and json-mode events. Skipped without $SAETTA_CODE_GOLDEN.
#include "../code_oai.h"
#include "../metal/shim.h"
#include "../test.h"

#include <dirent.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

// run_scenario.py models_json(): the "saetta" provider of every capture.
static const char *COMPAT = "{\"supportsDeveloperRole\":false,\"supportsStore\":false,\"supportsReasoningEffort\":false,"
                            "\"maxTokensField\":\"max_tokens\",\"thinkingFormat\":\"qwen-chat-template\"}";

static CodeModel mock_model(Arena *a, const char *base_url, const char *thinking) {
    JValue *compat = NULL;
    json_parse(a, str(COMPAT), true, &compat);
    return (CodeModel){.provider = "saetta", .id = "mock-model", .name = "Mock", .base_url = base_url,
                       .api_key = "saetta", .reasoning = true, .context_window = 131072, .max_tokens = 32768,
                       .max_tokens_field = "max_tokens", .thinking_format = "qwen-chat-template",
                       .compat = compat, .thinking = thinking};
}

// ---- golden files -------------------------------------------------------------------------

typedef VEC(Str) Lines;

static bool read_lines(Arena *a, const char *path, Lines *out) {
    char *text;
    size_t len;
    if (read_file(path, 1u << 30, &text, &len)) return false;
    char *t = arena_strndup(a, text, len);
    free(text);
    size_t start = 0;
    for (size_t i = 0; i < len; i++)
        if (t[i] == '\n') {
            if (i > start) vec_push(out, str_n(t + start, i - start));
            start = i + 1;
        }
    if (len > start) vec_push(out, str_n(t + start, len - start));
    return true;
}

static int cmp_str(const void *x, const void *y) { return strcmp(*(char *const *)x, *(char *const *)y); }

// The scenario folders of the golden data, sorted.
static size_t scenarios(Arena *a, const char *golden, char ***out) {
    DIR *d = opendir(golden);
    VEC(char *) v = {0};
    if (d) {
        struct dirent *e;
        while ((e = readdir(d)))
            if (e->d_name[0] != '.' && e->d_type == DT_DIR) vec_push(&v, arena_strdup(a, e->d_name));
        closedir(d);
    }
    if (v.len) qsort(v.ptr, v.len, sizeof(char *), cmp_str);
    char **r = ARENA_ARRAY(a, char *, v.len + 1);
    if (v.len) memcpy(r, v.ptr, v.len * sizeof(char *));
    size_t n = v.len;
    vec_free(&v);
    *out = r;
    return n;
}

// The "message" member of a session line, as the raw bytes pi wrote.
static Str raw_message(Str line) {
    ptrdiff_t i = str_find(line, S(",\"message\":"), 0);
    if (i < 0) return S("");
    return str_n(line.ptr + i + 11, line.len - (size_t)i - 12);
}

// "durationMs":N -> "durationMs":0, as run_scenario.py normalizes.
static Str norm_duration(Arena *a, Str s) {
    Buf b = {0};
    size_t i = 0;
    for (;;) {
        ptrdiff_t k = str_find(s, S("\"durationMs\":"), i);
        if (k < 0) break;
        buf_add(&b, s.ptr + i, (size_t)k + 13 - i);
        buf_addc(&b, '0');
        i = (size_t)k + 13;
        while (i < s.len && s.ptr[i] >= '0' && s.ptr[i] <= '9') i++;
    }
    buf_add(&b, s.ptr + i, s.len - i);
    Str r = {arena_strndup(a, b.ptr ? b.ptr : "", b.len), b.len};
    buf_free(&b);
    return r;
}

static void report_diff(const char *what, Str got, Str want) {
    size_t i = 0;
    while (i < got.len && i < want.len && got.ptr[i] == want.ptr[i]) i++;
    size_t from = i > 120 ? i - 120 : 0;
    dbg("%s: differs at byte %zu\n  got:  ...%.*s\n  want: ...%.*s\n", what, i, (int)((got.len > from + 300 ? from + 300 : got.len) - from),
        got.ptr + from, (int)((want.len > from + 300 ? from + 300 : want.len) - from), want.ptr + from);
}

// ---- the system prompt seam -----------------------------------------------------------------

static Str g_system_text;
static Str fixed_system_text(Arena *a, const JValue *system) {
    (void)a;
    (void)system;
    return g_system_text;
}

// One golden session: the requests' inputs.
typedef struct {
    JValue *system;
    JValue *before;      // the conversation before this answer
    const char *thinking;
    Str answer;          // the golden assistant message (raw)
} Turn;

static bool single_edit(const JValue *v) {
    return json_is(v, J_OBJECT) && json_is(json_get(v, "oldText"), J_STRING) && json_is(json_get(v, "newText"), J_STRING);
}

// pi's edit tool (core/tools/edit.ts prepareEditArguments) rewrites a call's "edits" in place
// after the session line was written: the requests of the same process carry the rewritten
// arguments, a resumed session the written ones. This is the agent's tool layer, not the
// client: the test applies it to rebuild what pi sent.
static JValue *edit_prepared(Arena *a, JValue *msg) {
    JValue *content = json_get(msg, "content");
    JValue *copy = NULL;
    for (size_t i = 0; json_is(content, J_ARRAY) && i < content->arr.len; i++) {
        JValue *b = content->arr.items[i];
        if (!str_eqz(json_strs(json_get(b, "type")), "toolCall") || !str_eqz(json_strs(json_get(b, "name")), "edit")) continue;
        JValue *args = json_get(b, "arguments"), *edits = json_get(args, "edits"), *nv = NULL;
        if (json_is(edits, J_STRING)) {
            JValue *parsed = code_oai_js_parse(a, edits->s, NULL);
            if (json_is(parsed, J_ARRAY)) nv = parsed;
            else if (single_edit(parsed)) {
                nv = json_new(a, J_ARRAY);
                json_push(a, nv, parsed);
            }
        } else if (single_edit(edits)) {
            nv = json_new(a, J_ARRAY);
            json_push(a, nv, edits);
        }
        if (!nv) continue;
        if (!copy) {
            copy = json_new(a, J_OBJECT);
            for (size_t k = 0; k < msg->obj.len; k++) json_put(a, copy, msg->obj.items[k].key, msg->obj.items[k].val);
            JValue *nc = json_new(a, J_ARRAY);
            for (size_t k = 0; k < content->arr.len; k++) json_push(a, nc, content->arr.items[k]);
            json_put(a, copy, S("content"), nc);
        }
        JValue *nb = json_new(a, J_OBJECT), *na = json_new(a, J_OBJECT);
        for (size_t k = 0; k < b->obj.len; k++) json_put(a, nb, b->obj.items[k].key, b->obj.items[k].val);
        for (size_t k = 0; k < args->obj.len; k++) json_put(a, na, args->obj.items[k].key, args->obj.items[k].val);
        json_put(a, na, S("edits"), nv);
        json_put(a, nb, S("arguments"), na);
        json_get(copy, "content")->arr.items[i] = nb;
    }
    return copy ? copy : msg;
}

// runs: the number of answers of each run of the scenario (NULL: one run).
static size_t session_turns(Arena *a, const char *path, const size_t *runs, size_t nruns, Turn **out) {
    Lines lines = {0};
    if (!read_lines(a, path, &lines)) return 0;
    VEC(Turn) turns = {0};
    JValue *system = NULL, *conv = json_new(a, J_ARRAY);
    VEC(size_t) edited = {0}; // conv indexes holding a rewritten message, with the original below
    VEC(JValue *) originals = {0};
    size_t run = 0, left = runs && nruns ? runs[0] : SIZE_MAX;
    const char *thinking = "off";
    for (size_t i = 0; i < lines.len; i++) {
        JValue *e;
        if (json_parse(a, lines.ptr[i], false, &e)) continue;
        const char *type = json_str(json_get(e, "type"));
        if (type && !strcmp(type, "thinking_level_change")) thinking = json_str(json_get(e, "thinkingLevel"));
        if (!type || strcmp(type, "message")) continue;
        JValue *m = json_get(e, "message");
        const char *role = json_str(json_get(m, "role"));
        if (role && !strcmp(role, "system")) {
            system = m;
            continue;
        }
        if (role && !strcmp(role, "assistant")) {
            if (left == 0 && run + 1 < nruns) {
                // a new process: the session is read back from the file
                run++;
                left = runs[run];
                for (size_t k = 0; k < edited.len; k++) conv->arr.items[edited.ptr[k]] = originals.ptr[k];
                edited.len = originals.len = 0;
            }
            left--;
            JValue *before = json_new(a, J_ARRAY);
            for (size_t k = 0; k < conv->arr.len; k++) json_push(a, before, conv->arr.items[k]);
            vec_push(&turns, ((Turn){system, before, thinking, raw_message(lines.ptr[i])}));
            JValue *prepared = edit_prepared(a, m);
            if (prepared != m) {
                vec_push(&edited, conv->arr.len);
                vec_push(&originals, m);
                m = prepared;
            }
        }
        json_push(a, conv, m);
    }
    vec_free(&edited);
    vec_free(&originals);
    vec_free(&lines);
    *out = turns.ptr;
    return turns.len;
}

// The answers per run of a scenario (the length of each run's script), 0 when unknown.
static size_t scenario_runs(Arena *a, const char *golden, const char *name, size_t *out, size_t cap) {
    char *text;
    size_t len;
    if (read_file(arena_printf(a, "%s/../scenarios/%s.json", golden, name), 1u << 26, &text, &len)) return 0;
    JValue *sc;
    Err e = json_parse(a, str_n(text, len), false, &sc);
    free(text);
    if (e) return 0;
    JValue *runs = json_get(sc, "runs");
    size_t n = 0;
    for (size_t r = 0; json_is(runs, J_ARRAY) && r < runs->arr.len && n < cap; r++)
        out[n++] = json_get(runs->arr.items[r], "script")->arr.len;
    return n;
}

static const char *golden_dir(void) {
    const char *g = getenv("SAETTA_CODE_GOLDEN");
    return g && *g ? g : NULL;
}

// ---- (1) request bodies -----------------------------------------------------------------------

TEST(code_oai_body_golden) {
    const char *golden = golden_dir();
    if (!golden) SKIP("no $SAETTA_CODE_GOLDEN");
    Arena a = {0};
    char **names;
    size_t n = scenarios(&a, golden, &names), checked = 0, skipped = 0;
    Str (*saved)(Arena *, const JValue *) = code_oai_system_text;
    code_oai_system_text = fixed_system_text;
    for (size_t s = 0; s < n; s++) {
        Lines reqs = {0};
        Turn *turns = NULL;
        size_t runs[64], nruns = scenario_runs(&a, golden, names[s], runs, 64);
        size_t nt = session_turns(&a, arena_printf(&a, "%s/%s/sessions/0.jsonl", golden, names[s]), runs, nruns, &turns);
        if (!read_lines(&a, arena_printf(&a, "%s/%s/requests.jsonl", golden, names[s]), &reqs) || nt != reqs.len) {
            skipped++;
            vec_free(&reqs);
            free(turns);
            continue;
        }
        for (size_t k = 0; k < nt; k++) {
            JValue *req;
            EXPECT_OK(json_parse(&a, reqs.ptr[k], false, &req));
            JValue *msgs = json_get(req, "messages"), *first = msgs->arr.len ? msgs->arr.items[0] : NULL;
            g_system_text = first && str_eqz(json_strs(json_get(first, "role")), "system") ? json_strs(json_get(first, "content")) : S("");
            CodeModel m = mock_model(&a, "http://127.0.0.1:1/v1", turns[k].thinking);
            for (int pass = 0; pass < 2; pass++) {
                // pass 0: the golden tools array; pass 1: NULL, the tools from the system message's toolsAdded
                Buf body = {0};
                code_oai_system_text = fixed_system_text;
                Err e = code_oai_body(&a, &m, turns[k].system, turns[k].before, pass ? NULL : json_get(req, "tools"), &body);
                code_oai_system_text = saved;
                EXPECT_OK(e);
                if (!str_eq(buf_str(&body), reqs.ptr[k])) {
                    report_diff(arena_printf(&a, "%s request %zu (%s)", names[s], k, pass ? "tools from system" : "golden tools"),
                                buf_str(&body), reqs.ptr[k]);
                    buf_free(&body);
                    *failed_ = 1;
                    return;
                }
                buf_free(&body);
            }
            checked++;
        }
        vec_free(&reqs);
        free(turns);
    }
    code_oai_system_text = saved;
    dbg("code_oai_body_golden: %zu requests byte-identical, %zu scenarios without a session to rebuild them from\n", checked, skipped);
    EXPECT(checked > 0);
    arena_reset(&a);
}

// ---- (2) answers ------------------------------------------------------------------------------

typedef struct {
    pid_t pid;
    int port;
} Mock;

static int free_port(void) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in sa = {.sin_family = AF_INET, .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    bind(fd, (struct sockaddr *)&sa, sizeof sa);
    socklen_t l = sizeof sa;
    getsockname(fd, (struct sockaddr *)&sa, &l);
    close(fd);
    return ntohs(sa.sin_port);
}

static bool mock_start(Arena *a, const char *mock_py, const char *script, const char *reqlog, Mock *m) {
    m->port = free_port();
    int pipefd[2];
    if (pipe(pipefd)) return false;
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, pipefd[1], 1);
    posix_spawn_file_actions_addclose(&fa, pipefd[0]);
    char *port = arena_printf(a, "%d", m->port);
    char *argv[] = {"python3", "-I", (char *)mock_py, (char *)script, (char *)reqlog, "--port", port, NULL};
    int rc = posix_spawnp(&m->pid, "python3", &fa, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    close(pipefd[1]);
    if (rc) {
        close(pipefd[0]);
        return false;
    }
    char c;
    while (read(pipefd[0], &c, 1) == 1 && c != '\n') {} // the ready line
    close(pipefd[0]);
    return true;
}

static void mock_stop(Mock *m) {
    kill(m->pid, SIGTERM);
    waitpid(m->pid, NULL, 0);
}

typedef struct {
    Arena *a;
    VEC(Str) lines; // json-mode lines: message_start (from "start") and message_update
} Rec;

// The json-mode line pi prints for an event (modes/json-event.ts toJsonEvent).
static void rec_event(void *ctx, const JValue *ev, const JValue *partial) {
    Rec *r = ctx;
    const char *type = json_str(json_get(ev, "type"));
    Buf b = {0};
    if (!strcmp(type, "start")) {
        buf_adds(&b, "{\"type\":\"message_start\",\"message\":");
        code_oai_js_write(&b, partial);
        buf_addc(&b, '}');
    } else if (!strcmp(type, "done") || !strcmp(type, "error")) {
        return;
    } else {
        buf_adds(&b, "{\"type\":\"message_update\",\"usage\":");
        code_oai_js_write(&b, json_get(partial, "usage"));
        buf_adds(&b, ",\"assistantMessageEvent\":");
        if (!strcmp(type, "toolcall_start")) {
            const JValue *idx = json_get(ev, "contentIndex");
            const JValue *tc = json_get(partial, "content")->arr.items[atoi(idx->s.ptr)];
            buf_adds(&b, "{\"type\":\"toolcall_start\",\"contentIndex\":");
            code_oai_js_write(&b, idx);
            buf_adds(&b, ",\"id\":");
            code_oai_js_write(&b, json_get(tc, "id"));
            buf_adds(&b, ",\"toolName\":");
            code_oai_js_write(&b, json_get(tc, "name"));
            buf_addc(&b, '}');
        } else
            code_oai_js_write(&b, ev);
        buf_addc(&b, '}');
    }
    vec_push(&r->lines, ((Str){arena_strndup(r->a, b.ptr, b.len), b.len}));
    buf_free(&b);
}

// The json-mode lines of the assistant answers in a golden stdout: per answer, its
// message_start (when the stream started) and message_update lines.
typedef struct {
    VEC(Str) lines;
} Answer;

static size_t golden_answers(Arena *a, const char *path, Answer **out) {
    Lines lines = {0};
    VEC(Answer) ans = {0};
    if (!read_lines(a, path, &lines)) return 0;
    bool in = false;
    for (size_t i = 0; i < lines.len; i++) {
        Str l = lines.ptr[i];
        if (str_starts(l, S("{\"type\":\"message_start\",\"message\":{\"role\":\"assistant\""))) {
            vec_push(&ans, (Answer){0});
            in = true;
            // pi's agent loop emits message_start with the final message when the stream never
            // started (an HTTP error): no "start" event then.
            if (str_find(l, S("\"stopReason\":\"pending\""), 0) >= 0) vec_push(&ans.ptr[ans.len - 1].lines, l);
            continue;
        }
        if (in && str_starts(l, S("{\"type\":\"message_update\""))) vec_push(&ans.ptr[ans.len - 1].lines, l);
        if (str_starts(l, S("{\"type\":\"message_end\""))) in = false;
    }
    vec_free(&lines);
    *out = ans.ptr;
    return ans.len;
}

typedef struct {
    Buf body;
    int status;
} Raw;

static int raw_sink(void *ctx, int status, const void *p, size_t n) {
    Raw *r = ctx;
    r->status = status;
    buf_add(&r->body, p, n);
    return 0;
}

// Strips the "usage" of message_update lines: in pi it depends on timing (which events its
// consumer copied before the usage chunk was parsed), so the live HTTP check leaves it out.
static Str no_usage(Arena *a, Str l) {
    ptrdiff_t u = str_find(l, S(",\"usage\":"), 0), e = str_find(l, S(",\"assistantMessageEvent\":"), 0);
    if (u < 0 || e < 0) return l;
    return str(arena_printf(a, "%.*s%.*s", (int)u, l.ptr, (int)(l.len - (size_t)e), l.ptr + e));
}

static bool same_lines(Arena *a, const Answer *g, const Rec *r, bool usage) {
    bool same = g->lines.len == r->lines.len;
    for (size_t i = 0; same && i < g->lines.len; i++)
        same = usage ? str_eq(g->lines.ptr[i], r->lines.ptr[i]) : str_eq(no_usage(a, g->lines.ptr[i]), no_usage(a, r->lines.ptr[i]));
    if (!same)
        for (size_t i = 0; i < g->lines.len || i < r->lines.len; i++)
            dbg("  got:  %s\n  want: %s\n", i < r->lines.len ? r->lines.ptr[i].ptr : "-", i < g->lines.len ? g->lines.ptr[i].ptr : "-");
    return same;
}

static Str final_text(Arena *a, JValue *msg, const char *thinking) {
    json_put(a, msg, S("thinkingLevel"), json_string(a, str(thinking))); // the agent loop's field
    Buf b = {0};
    code_oai_js_write(&b, msg);
    Str s = norm_duration(a, buf_str(&b));
    buf_free(&b);
    return s;
}

// Answer k of a scenario: (a) the mock's bytes in one read through the parser, events with
// their usage; (b) the same over HTTP with code_oai_complete, events without the usage.
static bool check_answer(Arena *a, const char *name, size_t k, const Turn *turn, Str req, int raw_port, int live_port,
                         const Answer *gold, size_t *event_lines) {
    CodeEnv env = {.now_ms = 0};
    char *base = arena_printf(a, "http://127.0.0.1:%d/v1", raw_port);
    CodeModel m = mock_model(a, base, turn->thinking);
    Raw raw = {0};
    char err[256];
    sx_header h[] = {{"content-type", "application/json"}};
    if (sx_http("POST", arena_printf(a, "%s/chat/completions", base), h, 1, req.ptr, req.len, 30, raw_sink, &raw, &raw.status,
                err, sizeof err)) {
        dbg("%s answer %zu: mock request failed: %s\n", name, k, err);
        return false;
    }
    Rec rec = {.a = a};
    CodeSink sink = {rec_event, &rec};
    CodeOaiStream *st = code_oai_stream_new(a, &m, &env, &sink);
    JValue *msg;
    if (raw.status == 200) {
        code_oai_stream_start(st);
        code_oai_stream_feed(st, raw.body.ptr, raw.body.len);
        msg = code_oai_stream_end(st, NULL);
    } else
        msg = code_oai_stream_end(st, code_oai_http_error(a, raw.status, buf_str(&raw.body)));
    buf_free(&raw.body);
    Str got = final_text(a, msg, turn->thinking);
    bool ok = true;
    if (!str_eq(got, turn->answer)) {
        report_diff(arena_printf(a, "%s answer %zu (parser)", name, k), got, turn->answer);
        ok = false;
    } else if (gold && !same_lines(a, gold, &rec, true)) {
        dbg("%s answer %zu: json-mode events differ\n", name, k);
        ok = false;
    }
    if (gold && ok) *event_lines += gold->lines.len;
    Rec rec2 = {.a = a};
    CodeSink sink2 = {rec_event, &rec2};
    CodeModel lm = mock_model(a, arena_printf(a, "http://127.0.0.1:%d/v1", live_port), turn->thinking);
    JValue *live = NULL;
    if (ok && code_oai_complete(a, &lm, &env, req, &sink2, &live)) ok = false;
    if (ok) {
        Str lgot = final_text(a, live, turn->thinking);
        if (!str_eq(lgot, turn->answer)) {
            report_diff(arena_printf(a, "%s answer %zu (HTTP)", name, k), lgot, turn->answer);
            ok = false;
        } else if (gold && !same_lines(a, gold, &rec2, false)) {
            dbg("%s answer %zu: json-mode events over HTTP differ\n", name, k);
            ok = false;
        }
    }
    vec_free(&rec.lines);
    vec_free(&rec2.lines);
    return ok;
}

TEST(code_oai_stream_golden) {
    const char *golden = golden_dir();
    if (!golden) SKIP("no $SAETTA_CODE_GOLDEN");
    Arena a = {0};
    const char *mock_py = getenv("SAETTA_CODE_MOCK");
    if (!mock_py) mock_py = arena_printf(&a, "%s/../../../tools/code/mock_openai.py", golden);
    const char *tmp = getenv("TMPDIR");
    char **names;
    size_t n = scenarios(&a, golden, &names), answers = 0, event_lines = 0;
    for (size_t s = 0; s < n; s++) {
        char *scpath = arena_printf(&a, "%s/../scenarios/%s.json", golden, names[s]);
        char *text;
        size_t len;
        if (read_file(scpath, 1u << 26, &text, &len)) continue;
        JValue *sc;
        Err pe = json_parse(&a, str_n(text, len), false, &sc);
        free(text);
        if (pe) continue;
        Lines reqs = {0};
        Turn *turns = NULL;
        size_t runs_n[64], nruns = scenario_runs(&a, golden, names[s], runs_n, 64);
        size_t nt = session_turns(&a, arena_printf(&a, "%s/%s/sessions/0.jsonl", golden, names[s]), runs_n, nruns, &turns);
        read_lines(&a, arena_printf(&a, "%s/%s/requests.jsonl", golden, names[s]), &reqs);
        JValue *runs = json_get(sc, "runs");
        size_t total = 0;
        for (size_t r = 0; r < runs->arr.len; r++) total += json_get(runs->arr.items[r], "script")->arr.len;
        if (nt != reqs.len || total != nt) {
            vec_free(&reqs);
            free(turns);
            continue;
        }
        size_t k = 0;
        for (size_t r = 0; r < runs->arr.len; r++) {
            JValue *script = json_get(runs->arr.items[r], "script");
            Buf sb = {0};
            json_write(&sb, script);
            char *spath = arena_printf(&a, "%s/saetta-code-oai-%d.json", tmp ? tmp : "/tmp", getpid());
            char *rpath = arena_printf(&a, "%s/saetta-code-oai-%d.jsonl", tmp ? tmp : "/tmp", getpid());
            EXPECT_OK(write_file(spath, sb.ptr, sb.len));
            buf_free(&sb);
            Answer *gold = NULL;
            size_t ng = golden_answers(&a, arena_printf(&a, "%s/%s/stdout-%zu.txt", golden, names[s], r), &gold);
            Mock raw_mock, live_mock;
            EXPECT(mock_start(&a, mock_py, spath, rpath, &raw_mock));
            EXPECT(mock_start(&a, mock_py, spath, rpath, &live_mock));
            bool ok = true;
            for (size_t t = 0; ok && t < script->arr.len; t++, k++) {
                ok = check_answer(&a, names[s], k, &turns[k], reqs.ptr[k], raw_mock.port, live_mock.port, t < ng ? &gold[t] : NULL, &event_lines);
                answers += ok;
            }
            if (!ok) *failed_ = 1;
            mock_stop(&raw_mock);
            mock_stop(&live_mock);
            unlink(spath);
            unlink(rpath);
            for (size_t i = 0; i < ng; i++) vec_free(&gold[i].lines);
            free(gold);
            if (!ok) return;
        }
        vec_free(&reqs);
        free(turns);
    }
    dbg("code_oai_stream_golden: %zu answers identical (parser and HTTP), %zu json-mode event lines\n", answers, event_lines);
    EXPECT(answers > 0);
    arena_reset(&a);
}

// ---- JSON.parse and number formatting ---------------------------------------------------------

static Str js_round(Arena *a, const char *text) {
    char *err = NULL;
    JValue *v = code_oai_js_parse(a, str(text), &err);
    if (!v) return str(err);
    Buf b = {0};
    code_oai_js_write(&b, v);
    Str s = {arena_strndup(a, b.ptr, b.len), b.len};
    buf_free(&b);
    return s;
}

TEST(code_oai_js_parse) {
    Arena a = {0};
    // expected values are node v26's JSON.stringify(JSON.parse(text)) and its error messages
    EXPECT_STR(js_round(&a, "{\"b\":1,\"2\":2,\"1\":3,\"a\":4,\"01\":5,\"4294967295\":6,\"4294967294\":7}"),
               S("{\"1\":3,\"2\":2,\"4294967294\":7,\"b\":1,\"a\":4,\"01\":5,\"4294967295\":6}"));
    EXPECT_STR(js_round(&a, "[1e21,1e-7,123456789012345680000,0.1,-0,1.5e300,9007199254740993,5e-324,1e-6,123e-20]"),
               S("[1e+21,1e-7,123456789012345680000,0.1,0,1.5e+300,9007199254740992,5e-324,0.000001,1.23e-18]"));
    EXPECT_STR(js_round(&a, "{\"a\":1,\"a\":2,\"b\":3}"), S("{\"a\":2,\"b\":3}"));
    EXPECT_STR(js_round(&a, "{\"a\":1"), S("Expected ',' or '}' after property value in JSON at position 6 (line 1 column 7)"));
    EXPECT_STR(js_round(&a, "{\"a\":\"x"), S("Unterminated string in JSON at position 7 (line 1 column 8)"));
    EXPECT_STR(js_round(&a, "{\"a\" 1}"), S("Expected ':' after property name in JSON at position 5 (line 1 column 6)"));
    EXPECT_STR(js_round(&a, "{\"a\":1,}"), S("Expected double-quoted property name in JSON at position 7 (line 1 column 8)"));
    EXPECT_STR(js_round(&a, "{a:1}"), S("Expected property name or '}' in JSON at position 1 (line 1 column 2)"));
    EXPECT_STR(js_round(&a, "{\"a\":tru}"), S("Unexpected token '}', \"{\"a\":tru}\" is not valid JSON"));
    EXPECT_STR(js_round(&a, "[1,2"), S("Expected ',' or ']' after array element in JSON at position 4 (line 1 column 5)"));
    EXPECT_STR(js_round(&a, "{\"a\":1}\n{\"b\":2}"), S("Unexpected non-whitespace character after JSON at position 8 (line 2 column 1)"));
    EXPECT_STR(js_round(&a, "{\"a\":-}"), S("No number after minus sign in JSON at position 6 (line 1 column 7)"));
    EXPECT_STR(js_round(&a, "{\"a\":01}"), S("Unexpected number in JSON at position 6 (line 1 column 7)"));
    EXPECT_STR(js_round(&a, "{\"a\":\"\\x\"}"), S("Bad escaped character in JSON at position 7 (line 1 column 8)"));
    EXPECT_STR(js_round(&a, "{\"a\":1.}"), S("Unterminated fractional number in JSON at position 7 (line 1 column 8)"));
    EXPECT_STR(js_round(&a, "nul"), S("Unexpected end of JSON input"));
    EXPECT_STR(js_round(&a, "{\"a\nb\":1}"), S("Bad control character in string literal in JSON at position 3 (line 1 column 4)"));
    EXPECT_STR(js_round(&a, "{\"a\":\"\\u12x4\"}"), S("Bad Unicode escape in JSON at position 10 (line 1 column 11)"));
    EXPECT_STR(js_round(&a, "{\"a\":1e+}"), S("Exponent part is missing a number in JSON at position 8 (line 1 column 9)"));
    EXPECT_STR(js_round(&a, "[1,]"), S("Unexpected token ']', \"[1,]\" is not valid JSON"));
    EXPECT_STR(js_round(&a, "\r\n\r{x"), S("Expected property name or '}' in JSON at position 4 (line 3 column 2)"));
    EXPECT_STR(js_round(&a, "{\"aaaaaaaaaaaaaaaaaaaa\":tru}"), S("Unexpected token '}', ...\"aaaaa\":tru}\" is not valid JSON"));
    EXPECT_STR(js_round(&a, "{\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\": +1}"),
               S("Unexpected token '+', ...\"aaaaaaa\": +1}\" is not valid JSON"));
    EXPECT_STR(js_round(&a, "undefined"), S("\"undefined\" is not valid JSON"));
    EXPECT_STR(js_round(&a, ""), S("Unexpected end of JSON input"));
    EXPECT_STR(js_round(&a, "{\"\xc3\xa9\":\"\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\", x}"),
               S("Expected double-quoted property name in JSON at position 29 (line 1 column 30)"));
    arena_reset(&a);
}

// HTTP error messages as pi records them (captured from pi 1.5.2 against the mock).
TEST(code_oai_http_errors) {
    Arena a = {0};
    EXPECT_STR(str(code_oai_http_error(&a, 500, S("{\"error\":{\"message\":\"boom\",\"type\":\"server_error\"}}"))),
               S("500: {\"message\":\"boom\",\"type\":\"server_error\"}"));
    EXPECT_STR(str(code_oai_http_error(&a, 400, S("bad request text"))), S("400 bad request text"));
    EXPECT_STR(str(code_oai_http_error(&a, 429, S("{\"error\":\"slow down\"}"))), S("429 \"slow down\""));
    EXPECT_STR(str(code_oai_http_error(&a, 503, S(""))), S("503 status code (no body)"));
    arena_reset(&a);
}
