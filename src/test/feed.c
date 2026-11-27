// Speculation fed by the agent (server_engine_feed, src/server.c). The engine tests need
// SAETTA_FEED_TEST_PACK (a Qwen3.8 pack: the Coder or a 27B; they load it and run prompts on the GPU)
// and run from the repository's root (they copy src/spec.c):
// - a request that copies a file's text into an edit's JSON answers the same, token for token
//   (greedy, speculation on), with that file fed to its session as without, in fewer passes;
// - the fed texts stay bounded: a text's head, a session's total, the number of sessions, and a
//   key fed again replaces its text;
// - clearing a session drops its texts: its next request drafts from none, and answers the same.
#include "../json.h"
#include "../server.h"
#include "../stats.h"
#include "../test.h"

#include <stdatomic.h>
#include <stdio.h>
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
    // greedy, and depths chosen without the clock: two runs repeat token for token
    setenv("SAETTA_BENCH_GREEDY", "1", 1);
    static Arena a;
    if (server_engine_open(&a, pack, NULL, 0, NULL, &shared_engine)) return NULL;
    return shared_engine;
}

static char *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    Buf b = {0};
    char tmp[4096];
    size_t k;
    while ((k = fread(tmp, 1, sizeof tmp, f)) > 0) buf_add(&b, tmp, k);
    fclose(f);
    *n = b.len;
    char *s = strndup(b.ptr ? b.ptr : "", b.len);
    buf_free(&b);
    return s;
}

// The file (shown as a read's result) and a request for an edit copying one of its functions, as an
// edit tool call's edits argument is written: JSON. Greedy, no thinking.
static char *edit_body(Str file, unsigned salt) {
    Buf b = {0}, j = {0};
    buf_printf(&b, "Request %u. Here is the file src/spec.c:\n\n", salt);
    buf_add(&b, file.ptr, file.len);
    buf_adds(&b, "\n\nWrite the edits argument of an edit tool call as a JSON array with one object: its oldText is the "
                 "complete function map_grow copied exactly from the file, and its newText is the same function with the "
                 "local variable ncap renamed to new_cap. Output only the JSON array.");
    json_write_string(&j, buf_str(&b));
    Buf body = {0};
    buf_adds(&body, "{\"messages\":[{\"role\":\"user\",\"content\":");
    buf_add(&body, j.ptr, j.len);
    buf_adds(&body, "}],\"max_tokens\":400,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}");
    char *s = strdup(body.ptr);
    buf_free(&b);
    buf_free(&j);
    buf_free(&body);
    return s;
}

// The reply's content (strdup'd), with its decode passes and generated tokens.
static char *chat(Engine *e, const char *body, const char *sess, size_t *passes, size_t *gen) {
    Buf out = {0};
    Err err = server_engine_chat_session(e, str(body), sess, sink_keep, &out);
    *passes = atomic_load(&server_engine_stats(e)->passes);
    *gen = atomic_load(&server_engine_stats(e)->gen);
    char *content = NULL;
    Arena a = {0};
    JValue *r;
    if (!err && !json_parse(&a, str_n(out.ptr, out.len), false, &r)) {
        const JValue *ch = json_get(r, "choices");
        const JValue *m = ch && ch->kind == J_ARRAY && ch->arr.len ? json_get(ch->arr.items[0], "message") : NULL;
        const JValue *c = m ? json_get(m, "content") : NULL;
        if (c && c->kind == J_STRING) content = strndup(c->s.ptr, c->s.len);
    }
    arena_reset(&a);
    buf_free(&out);
    return content ? content : strdup("<error>");
}

TEST(feed_engine_same_answers) {
    const char *pack = getenv("SAETTA_FEED_TEST_PACK");
    if (!pack) SKIP("SAETTA_FEED_TEST_PACK not set (loads a model)");
    size_t fn;
    char *file = slurp("src/spec.c", &fn);
    if (!file) SKIP("src/spec.c not found (run from the repository's root)");
    Engine *e = engine_for(pack);
    EXPECT(e != NULL);
    if (!e) return;
    // the same prompt three times: read (plain 1), then from its restore point without (plain 2) and
    // with the file fed (another session, routed to the same conversation by its restore point, so
    // plain 2 and the fed request both start from it with a speculation controller one request warm)
    char *body = edit_body(str_n(file, fn), 1);
    size_t p1, g1, pp, pg, fp, fg;
    char *plain1 = chat(e, body, "feed-plain", &p1, &g1);
    char *plain = chat(e, body, "feed-plain", &pp, &pg);
    EXPECT_EQ(server_engine_feed_drafts(e), 0);
    EXPECT_EQ(server_engine_feed(e, "feed-fed", S("/ws/src/spec.c"), str_n(file, fn)), OK);
    size_t texts;
    EXPECT(server_engine_feed_tokens(e, "feed-fed", &texts, NULL) > 1000);
    EXPECT_EQ(texts, 1);
    char *fed = chat(e, body, "feed-fed", &fp, &fg);
    size_t fd = server_engine_feed_drafts(e);
    dbg("feed: plain %zu tokens in %zu passes (first read: %zu in %zu), fed %zu in %zu (%zu drafted from the feed): \"%s\"\n", pg,
        pp, g1, p1, fg, fp, fd, fed);
    EXPECT_STR(str(plain), str(plain1));
    EXPECT_STR(str(fed), str(plain));
    EXPECT_EQ(fg, pg);
    EXPECT(strstr(plain, "new_cap") != NULL); // the request copied the function
    // the fed copy drafts the oldText's lines across their \n escapes: fewer passes
    EXPECT(fd > 0);
    EXPECT(fp * 10 < pp * 9);
    free(plain1);
    free(plain);
    free(fed);
    free(body);
    free(file);
}

TEST(feed_engine_bounded) {
    const char *pack = getenv("SAETTA_FEED_TEST_PACK");
    if (!pack) SKIP("SAETTA_FEED_TEST_PACK not set (loads a model)");
    Engine *e = engine_for(pack);
    EXPECT(e != NULL);
    if (!e) return;
    // a text of more than 65,536 tokens (digits are single tokens) keeps its first 65,536
    Buf big = {0};
    for (size_t i = 0; i < 40000; i++) buf_printf(&big, "x%zu\n", i * 7919 % 100003);
    EXPECT_EQ(server_engine_feed(e, "feed-bound", S("big"), buf_str(&big)), OK);
    size_t texts;
    EXPECT_EQ(server_engine_feed_tokens(e, "feed-bound", &texts, NULL), 65536);
    // fed again under the same key: replaced, not added
    EXPECT_EQ(server_engine_feed(e, "feed-bound", S("big"), buf_str(&big)), OK);
    EXPECT_EQ(server_engine_feed_tokens(e, "feed-bound", &texts, NULL), 65536);
    EXPECT_EQ(texts, 1);
    // two more keys: the session holds 131,072 tokens at most, the oldest text goes first
    EXPECT_EQ(server_engine_feed(e, "feed-bound", S("big2"), buf_str(&big)), OK);
    EXPECT_EQ(server_engine_feed(e, "feed-bound", S("big3"), buf_str(&big)), OK);
    EXPECT_EQ(server_engine_feed_tokens(e, "feed-bound", &texts, NULL), 131072);
    EXPECT_EQ(texts, 2);
    // 6 sessions: the 4 most recently fed are kept, and the memory stays bounded
    char sess[32];
    for (int k = 0; k < 6; k++) {
        snprintf(sess, sizeof sess, "feed-many-%d", k);
        EXPECT_EQ(server_engine_feed(e, sess, S("big"), buf_str(&big)), OK);
    }
    EXPECT_EQ(server_engine_feed_tokens(e, "feed-bound", NULL, NULL), 0);
    EXPECT_EQ(server_engine_feed_tokens(e, "feed-many-1", NULL, NULL), 0);
    size_t bytes;
    EXPECT_EQ(server_engine_feed_tokens(e, "feed-many-5", NULL, &bytes), 65536);
    EXPECT_EQ(server_engine_feed_tokens(e, "feed-many-2", NULL, NULL), 65536);
    dbg("feed: 4 sessions of 65,536 tokens hold %zu bytes\n", bytes);
    EXPECT(bytes < (size_t)4 * 6 * 1024 * 1024);
    for (int k = 0; k < 6; k++) {
        snprintf(sess, sizeof sess, "feed-many-%d", k);
        server_engine_feed_clear(e, sess);
    }
    (void)server_engine_feed_tokens(e, NULL, NULL, &bytes);
    EXPECT_EQ(bytes, 0); // nothing held by any session
    buf_free(&big);
}

TEST(feed_engine_cleared) {
    const char *pack = getenv("SAETTA_FEED_TEST_PACK");
    if (!pack) SKIP("SAETTA_FEED_TEST_PACK not set (loads a model)");
    size_t fn;
    char *file = slurp("src/spec.c", &fn);
    if (!file) SKIP("src/spec.c not found (run from the repository's root)");
    Engine *e = engine_for(pack);
    EXPECT(e != NULL);
    if (!e) return;
    char *body = edit_body(str_n(file, fn), 2);
    size_t pp, pg, cp, cg;
    EXPECT_EQ(server_engine_feed(e, "feed-clear", S("/ws/src/spec.c"), str_n(file, fn)), OK);
    char *fed = chat(e, body, "feed-clear", &pp, &pg);
    EXPECT(server_engine_feed_drafts(e) > 0);
    server_engine_feed_clear(e, "feed-clear");
    size_t texts;
    EXPECT_EQ(server_engine_feed_tokens(e, "feed-clear", &texts, NULL), 0);
    EXPECT_EQ(texts, 0);
    // the same request again drafts from its history alone, and answers the same
    char *after = chat(e, body, "feed-clear", &cp, &cg);
    EXPECT_EQ(server_engine_feed_drafts(e), 0);
    EXPECT_STR(str(after), str(fed));
    free(fed);
    free(after);
    free(body);
    free(file);
}
