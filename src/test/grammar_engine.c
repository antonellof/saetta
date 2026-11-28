// The tool-call grammar in the engine (SAETTA_GRAMMAR=1, src/server.c). Needs SAETTA_GRAMMAR_TEST_PACK
// (a Qwen3.8 pack: it loads it and runs prompts on the GPU):
// - greedy, with speculation on: requests whose calls are well formed (a read, a bash command, an
//   edit whose edits JSON copies lines of a file, a reply with reasoning) answer byte for byte the
//   same with the grammar as without, and the grammar replaced no token;
// - a request whose read tool takes its paths as an array of strings (the model writes a bare path
//   there unasked): with the grammar the value is a JSON array of strings (tokens replaced), and
//   every call, greedy and sampled, names a tool of the request with arguments of the schema's types.
#include "../json.h"
#include "../server.h"
#include "../stats.h"
#include "../test.h"

#include <stdatomic.h>
#include <stdlib.h>

static int sink_keep(void *ctx, int status, const void *p, size_t n) {
    (void)status;
    if (p && n) buf_add((Buf *)ctx, p, n);
    return 0;
}

static Engine *shared_engine;
static Engine *engine_for(const char *pack) {
    if (shared_engine) return shared_engine;
    setenv("SAETTA_CONVERSATIONS", "4", 1);
    setenv("SAETTA_CTX", "32768", 1);
    setenv("SAETTA_BENCH_GREEDY", "1", 1); // greedy with depths chosen without the clock (the sampled
                                          // requests turn it off for themselves)
    static Arena a;
    if (server_engine_open(&a, pack, NULL, 0, NULL, &shared_engine)) return NULL;
    return shared_engine;
}

static const char tools_all[] =
    "[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"description\":\"Read file contents\",\"parameters\":{\"type\":\"object\","
    "\"required\":[\"path\"],\"properties\":{\"path\":{\"type\":\"string\"},\"offset\":{\"type\":\"number\"},\"limit\":{\"type\":"
    "\"number\"}}}}},{\"type\":\"function\",\"function\":{\"name\":\"bash\",\"description\":\"Execute bash commands\",\"parameters\":"
    "{\"type\":\"object\",\"required\":[\"command\"],\"properties\":{\"command\":{\"type\":\"string\"},\"timeout\":{\"type\":\"number\"}}}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"edit\",\"description\":\"Edit a file with exact text replacement\",\"parameters\":"
    "{\"type\":\"object\",\"required\":[\"path\",\"edits\"],\"properties\":{\"path\":{\"type\":\"string\"},\"edits\":{\"type\":\"array\","
    "\"items\":{\"type\":\"object\",\"required\":[\"oldText\",\"newText\"],\"properties\":{\"oldText\":{\"type\":\"string\"},"
    "\"newText\":{\"type\":\"string\"}}}}}}}}]";

static const char tools_two[] =
    "[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"description\":\"Read file contents\",\"parameters\":{\"type\":\"object\","
    "\"required\":[\"path\"],\"properties\":{\"path\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}}}}},{\"type\":\"function\",\"function\":{\"name\":\"bash\","
    "\"description\":\"Execute bash commands\",\"parameters\":{\"type\":\"object\",\"required\":[\"command\"],\"properties\":{\"command\":"
    "{\"type\":\"string\"}}}}}]";

static char *body_of(const char *user, const char *tools, bool thinking, const char *sampling) {
    Buf j = {0}, b = {0};
    json_write_string(&j, str(user));
    buf_printf(&b, "{\"messages\":[{\"role\":\"user\",\"content\":%s}],\"tools\":%s,\"max_tokens\":500,%s,"
                   "\"chat_template_kwargs\":{\"enable_thinking\":%s}}",
               j.ptr, tools, sampling, thinking ? "true" : "false");
    char *s = strdup(b.ptr);
    buf_free(&j);
    buf_free(&b);
    return s;
}

typedef struct {
    char *content, *reasoning, *calls; // calls: "name args\n" per call
    size_t gen;
} Reply;

static char *dup_str(const JValue *v) { return v && v->kind == J_STRING ? strndup(v->s.ptr, v->s.len) : strdup(""); }

static Reply chat(Engine *e, const char *body, const char *grammar) {
    if (grammar) setenv("SAETTA_GRAMMAR", grammar, 1);
    else unsetenv("SAETTA_GRAMMAR");
    Buf out = {0};
    Err err = server_engine_chat_session(e, str(body), "grammar-test", sink_keep, &out);
    unsetenv("SAETTA_GRAMMAR");
    Reply r = {0};
    r.gen = atomic_load(&server_engine_stats(e)->gen);
    Arena a = {0};
    JValue *v;
    Buf calls = {0};
    if (!err && !json_parse(&a, str_n(out.ptr, out.len), false, &v)) {
        const JValue *ch = json_get(v, "choices");
        const JValue *m = ch && ch->kind == J_ARRAY && ch->arr.len ? json_get(ch->arr.items[0], "message") : NULL;
        r.content = dup_str(m ? json_get(m, "content") : NULL);
        r.reasoning = dup_str(m ? json_get(m, "reasoning_content") : NULL);
        const JValue *tc = m ? json_get(m, "tool_calls") : NULL;
        for (size_t i = 0; tc && tc->kind == J_ARRAY && i < tc->arr.len; i++) {
            const JValue *f = json_get(tc->arr.items[i], "function");
            buf_printf(&calls, "%s %s\n", json_str(json_get(f, "name")), json_str(json_get(f, "arguments")));
        }
    }
    r.calls = strdup(calls.ptr ? calls.ptr : "");
    if (!r.content) r.content = strdup("<error>");
    if (!r.reasoning) r.reasoning = strdup("");
    buf_free(&calls);
    buf_free(&out);
    arena_reset(&a);
    return r;
}

static void reply_free(Reply *r) {
    free(r->content);
    free(r->reasoning);
    free(r->calls);
}

static const char stats_file[] = "def mean(xs):\n    return sum(xs) / len(xs)\n\n\ndef median(xs):\n    s = sorted(xs)\n    n = len(s)\n"
                                 "    mid = n // 2\n    if n % 2 == 1:\n        return s[mid]\n    return s[mid]\n";

TEST(grammar_engine_greedy_unchanged) {
    const char *pack = getenv("SAETTA_GRAMMAR_TEST_PACK");
    if (!pack) SKIP("SAETTA_GRAMMAR_TEST_PACK not set (loads a model)");
    Engine *e = engine_for(pack);
    EXPECT(e != NULL);
    if (!e) return;
    Buf edit = {0};
    buf_printf(&edit, "This is mathkit/stats.py:\n\n%s\nmedian() is wrong for lists of even length. Fix it with the edit tool now.",
               stats_file);
    struct {
        const char *user;
        bool thinking;
    } cases[] = {
        {"Read the file mathkit/stats.py.", false},
        {"List the files in the current folder with bash, then stop.", false},
        {edit.ptr, false},
        {edit.ptr, true},
        {"What is 17 * 23? Answer in one line, no tools.", true},
    };
    for (size_t i = 0; i < COUNT(cases); i++) {
        char *body = body_of(cases[i].user, tools_all, cases[i].thinking, "\"temperature\":0");
        Reply off = chat(e, body, NULL);
        Reply on = chat(e, body, "1");
        size_t replaced = server_engine_grammar_replaced(e);
        dbg("grammar case %zu: %zu tokens, calls: %s", i, on.gen, on.calls[0] ? on.calls : "none\n");
        EXPECT_STR(str(on.content), str(off.content));
        EXPECT_STR(str(on.reasoning), str(off.reasoning));
        EXPECT_STR(str(on.calls), str(off.calls));
        EXPECT_EQ(on.gen, off.gen);
        EXPECT_EQ(replaced, 0);
        if (i < 4) EXPECT(on.calls[0] != 0); // the cases that call a tool did
        reply_free(&off);
        reply_free(&on);
        free(body);
    }
    buf_free(&edit);
}

// every call names a tool of tools_two, with its required argument as a string
static bool valid_calls(const char *calls) {
    Arena a = {0};
    bool ok = true;
    for (const char *p = calls; *p && ok;) {
        const char *nl = strchr(p, '\n');
        Str line = str_n(p, nl ? (size_t)(nl - p) : strlen(p));
        ptrdiff_t sp = str_find_char(line, ' ', 0);
        JValue *args;
        if (sp <= 0 || json_parse(&a, str_sub(line, (size_t)sp + 1, line.len), false, &args)) {
            ok = false;
            break;
        }
        Str name = str_sub(line, 0, (size_t)sp);
        if (str_eq(name, S("read")))
            ok = json_is(json_get(args, "path"), J_ARRAY) && json_get(args, "path")->arr.len > 0 &&
                 json_is(json_get(args, "path")->arr.items[0], J_STRING);
        else if (str_eq(name, S("bash"))) ok = json_is(json_get(args, "command"), J_STRING);
        else ok = false;
        p = nl ? nl + 1 : p + line.len;
    }
    arena_reset(&a);
    return ok;
}

TEST(grammar_engine_array_argument_masked) {
    const char *pack = getenv("SAETTA_GRAMMAR_TEST_PACK");
    if (!pack) SKIP("SAETTA_GRAMMAR_TEST_PACK not set (loads a model)");
    Engine *e = engine_for(pack);
    EXPECT(e != NULL);
    if (!e) return;
    const char *user = "Read the file mathkit/stats.py.";
    char *body = body_of(user, tools_two, false, "\"temperature\":0");
    Reply off = chat(e, body, NULL);
    Reply on = chat(e, body, "1");
    size_t replaced = server_engine_grammar_replaced(e);
    dbg("grammar array argument: without: %s with (%zu replaced): %s", off.calls[0] ? off.calls : "no call\n", replaced,
        on.calls[0] ? on.calls : "no call\n");
    EXPECT(on.calls[0] != 0);
    EXPECT(valid_calls(on.calls));
    if (!valid_calls(off.calls)) EXPECT(replaced > 0); // the model's own path was not an array
    reply_free(&off);
    reply_free(&on);
    free(body);
    // the count is the request's own: a request the grammar does not change after one it changed reports 0
    body = body_of("Read the file mathkit/stats.py.", tools_all, false, "\"temperature\":0");
    Reply plain = chat(e, body, "1");
    EXPECT_EQ(server_engine_grammar_replaced(e), 0);
    reply_free(&plain);
    free(body);
    for (int seed = 1; seed <= 3; seed++) {
        char sampling[96];
        snprintf(sampling, sizeof sampling, "\"temperature\":1,\"seed\":%d", seed);
        body = body_of(user, tools_two, false, sampling);
        setenv("SAETTA_BENCH_GREEDY", "0", 1); // sampled as asked
        Reply r = chat(e, body, "1");
        setenv("SAETTA_BENCH_GREEDY", "1", 1);
        EXPECT(r.calls[0] != 0);
        EXPECT(valid_calls(r.calls));
        reply_free(&r);
        free(body);
    }
}

// ---- edit match ("saetta_edit_match", Saetta Code's SAETTA_EDIT_MATCH=1) -------------------------

// body_of's request with "saetta_edit_match": mathkit/stats.py in /w holds text
static char *body_em(const char *user, const char *text) {
    char *plain = body_of(user, tools_all, false, "\"temperature\":0");
    Buf t = {0}, b = {0};
    json_write_string(&t, str(text));
    buf_printf(&b, "%.*s,\"saetta_edit_match\":{\"cwd\":\"/w\",\"home\":\"/h\",\"files\":[{\"path\":\"/w/mathkit/stats.py\",\"text\":%s}]}}",
               (int)strlen(plain) - 1, plain, t.ptr);
    char *s = strdup(b.ptr);
    free(plain);
    buf_free(&t);
    buf_free(&b);
    return s;
}

// every edit call's oldTexts occur exactly once in text
static bool old_texts_in(const char *calls, const char *text, size_t *edits) {
    Arena a = {0};
    bool ok = true;
    *edits = 0;
    for (const char *p = calls; *p && ok;) {
        const char *nl = strchr(p, '\n');
        Str line = str_n(p, nl ? (size_t)(nl - p) : strlen(p));
        p = nl ? nl + 1 : p + line.len;
        if (!str_starts(line, S("edit "))) continue;
        JValue *args;
        if (json_parse(&a, str_sub(line, 5, line.len), false, &args)) {
            ok = false;
            break;
        }
        const JValue *es = json_get(args, "edits");
        for (size_t i = 0; json_is(es, J_ARRAY) && i < es->arr.len && ok; i++) {
            const JValue *o = json_get(es->arr.items[i], "oldText");
            size_t n = 0;
            if (json_is(o, J_STRING) && o->s.len)
                for (ptrdiff_t at = str_find(str(text), o->s, 0); at >= 0; at = str_find(str(text), o->s, (size_t)at + o->s.len)) n++;
            ok = n == 1;
            ++*edits;
        }
    }
    arena_reset(&a);
    return ok;
}

TEST(grammar_engine_edit_match) {
    const char *pack = getenv("SAETTA_GRAMMAR_TEST_PACK");
    if (!pack) SKIP("SAETTA_GRAMMAR_TEST_PACK not set (loads a model)");
    Engine *e = engine_for(pack);
    EXPECT(e != NULL);
    if (!e) return;
    Buf user = {0};
    buf_printf(&user, "This is mathkit/stats.py:\n\n%s\nmedian() is wrong for lists of even length. Fix it with the edit tool now.",
               stats_file);
    // the file as the prompt shows it: the reply is the grammar's alone, nothing replaced
    char *plain = body_of(user.ptr, tools_all, false, "\"temperature\":0");
    char *same = body_em(user.ptr, stats_file);
    Reply g = chat(e, plain, "1");
    Reply m = chat(e, same, NULL); // the field turns the grammar on by itself
    size_t replaced = server_engine_grammar_replaced(e);
    size_t edits;
    dbg("edit match, same file: %s", m.calls);
    EXPECT_STR(str(m.calls), str(g.calls));
    EXPECT_STR(str(m.content), str(g.content));
    EXPECT_EQ(replaced, 0);
    EXPECT(old_texts_in(m.calls, stats_file, &edits));
    EXPECT(edits > 0);
    // the file on disk differs from what the model was shown: its oldText follows the file
    static const char disk[] = "def mean(xs):\n    return sum(xs) / len(xs)\n\n\ndef median(xs):\n    s = sorted(xs)\n    n = len(s)\n"
                               "    mid = n // 2\n    if n % 2:\n        return s[mid]\n    return s[mid]\n";
    char *other = body_em(user.ptr, disk);
    Reply d = chat(e, other, NULL);
    replaced = server_engine_grammar_replaced(e);
    dbg("edit match, other file (%zu replaced): %s", replaced, d.calls);
    EXPECT(old_texts_in(d.calls, disk, &edits));
    EXPECT(edits > 0);
    size_t ge;
    if (!old_texts_in(g.calls, disk, &ge)) EXPECT(replaced > 0);
    reply_free(&g);
    reply_free(&m);
    reply_free(&d);
    free(plain);
    free(same);
    free(other);
    buf_free(&user);
}


// ---- the reasoning budget (thinking_token_budget) -------------------------------------------------
// Greedy, with speculation on: a budget the reasoning stays under leaves the reply byte for byte as
// without one; a budget of 0 gives a reply with no reasoning (closed in the prompt) that still
// answers; a small budget cuts the reasoning (the engine writes the close) and the answer follows,
// with and without the tool-call grammar, greedy and sampled.

static const char tools_rb[] =
    "[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"description\":\"Read file contents\",\"parameters\":{\"type\":\"object\","
    "\"required\":[\"path\"],\"properties\":{\"path\":{\"type\":\"string\"}}}}},{\"type\":\"function\",\"function\":{\"name\":\"bash\","
    "\"description\":\"Execute bash commands\",\"parameters\":{\"type\":\"object\",\"required\":[\"command\"],\"properties\":{\"command\":"
    "{\"type\":\"string\"}}}}}]";

// budget < 0: none
static char *budget_body(const char *user, long budget, const char *sampling) {
    Buf j = {0}, b = {0};
    json_write_string(&j, str(user));
    buf_printf(&b, "{\"messages\":[{\"role\":\"user\",\"content\":%s}],\"tools\":%s,\"max_tokens\":1500,%s,"
                   "\"chat_template_kwargs\":{\"enable_thinking\":true}",
               j.ptr, tools_rb, sampling);
    if (budget >= 0) buf_printf(&b, ",\"thinking_token_budget\":%ld", budget);
    buf_adds(&b, "}");
    char *s = strdup(b.ptr);
    buf_free(&j);
    buf_free(&b);
    return s;
}

static const char *const budget_prompts[] = {
    "A train leaves at 9:40 and arrives at 13:15. How long is the trip? Think it through, then answer in one line, no tools.",
    "Find out which Python files in the current folder import os, using bash.",
};

static bool answered(const Reply *r) { return strcmp(r->content, "<error>") && (r->content[0] || r->calls[0]); }

TEST(grammar_engine_budget_under_unchanged_zero_and_small) {
    const char *pack = getenv("SAETTA_GRAMMAR_TEST_PACK");
    if (!pack) SKIP("SAETTA_GRAMMAR_TEST_PACK not set (loads a model)");
    Engine *e = engine_for(pack);
    EXPECT(e != NULL);
    if (!e) return;
    for (size_t i = 0; i < COUNT(budget_prompts); i++) {
        char *b_none = budget_body(budget_prompts[i], -1, "\"temperature\":0");
        char *b_big = budget_body(budget_prompts[i], 100000, "\"temperature\":0");
        char *b_zero = budget_body(budget_prompts[i], 0, "\"temperature\":0");
        char *b_small = budget_body(budget_prompts[i], 12, "\"temperature\":0");
        Reply none = chat(e, b_none, NULL), big = chat(e, b_big, NULL), zero = chat(e, b_zero, NULL), small = chat(e, b_small, NULL);
        dbg("budget case %zu: reasoning %zu bytes (none), %zu (0), %zu (12); answer (12): %.80s; calls (12): %s", i,
            strlen(none.reasoning), strlen(zero.reasoning), strlen(small.reasoning), small.content, small.calls);
        EXPECT(answered(&none) && answered(&big) && answered(&zero) && answered(&small));
        EXPECT(strlen(none.reasoning) > 100); // the case reasons, past the small budget
        EXPECT_STR(str(big.content), str(none.content));
        EXPECT_STR(str(big.reasoning), str(none.reasoning));
        EXPECT_STR(str(big.calls), str(none.calls));
        EXPECT_EQ(big.gen, none.gen);
        EXPECT_EQ(strlen(zero.reasoning), 0);
        // 12 tokens of reasoning are far under 12 * 32 bytes; the close is not in the reply's text
        EXPECT(strlen(small.reasoning) > 0 && strlen(small.reasoning) < 12 * 32);
        EXPECT(!strstr(small.reasoning, "</think>") && !strstr(small.content, "</think>"));
        if (i == 1) EXPECT(zero.calls[0] && small.calls[0]);
        reply_free(&none), reply_free(&big), reply_free(&zero), reply_free(&small);
        free(b_none), free(b_big), free(b_zero), free(b_small);
    }
}

TEST(grammar_engine_budget_with_grammar_and_sampled) {
    const char *pack = getenv("SAETTA_GRAMMAR_TEST_PACK");
    if (!pack) SKIP("SAETTA_GRAMMAR_TEST_PACK not set (loads a model)");
    Engine *e = engine_for(pack);
    EXPECT(e != NULL);
    if (!e) return;
    char *body = budget_body(budget_prompts[1], 12, "\"temperature\":0");
    Reply g = chat(e, body, "1");
    Reply p = chat(e, body, NULL);
    EXPECT(answered(&g) && answered(&p));
    EXPECT(g.calls[0] != 0);
    EXPECT_STR(str(g.reasoning), str(p.reasoning)); // the close is valid in the grammar's reasoning
    reply_free(&g), reply_free(&p);
    free(body);
    setenv("SAETTA_BENCH_GREEDY", "0", 1); // sampled as asked
    for (int seed = 1; seed <= 3; seed++) {
        char sampling[96];
        snprintf(sampling, sizeof sampling, "\"temperature\":1,\"seed\":%d", seed);
        body = budget_body(budget_prompts[1], 12, sampling);
        Reply r = chat(e, body, NULL);
        EXPECT(answered(&r));
        EXPECT(strlen(r.reasoning) < 12 * 32 && !strstr(r.content, "</think>"));
        reply_free(&r);
        free(body);
    }
    setenv("SAETTA_BENCH_GREEDY", "1", 1);
    // a budget that is no integer from 0 to SESSION_CTX_MAX is refused
    Buf out = {0};
    char *bad = budget_body(budget_prompts[0], -1, "\"temperature\":0,\"thinking_token_budget\":-3");
    (void)server_engine_chat_session(e, str(bad), "grammar-test", sink_keep, &out); // a request error is the answer
    EXPECT(out.ptr && strstr(out.ptr, "BadThinkingBudget"));
    buf_free(&out);
    free(bad);
}
