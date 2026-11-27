// Shared prefixes through the in-process engine (src/server.c), with a model (one engine for both
// tests): subagents start where their parent's prompt leaves off (the parent's request asks for a
// checkpoint before its generation prompt, "saetta_fork_point"), sessions that share a system prompt
// and tools start where the second of them left the first (a checkpoint where a prompt leaves
// another conversation's tokens), each answering as the same prompt read from 0 (greedy, speculation
// as configured), the parent's conversation kept, and the checkpoints inside the fit's reserve.
// Needs SAETTA_FORK_TEST_PACK (a Qwen3.8 pack: the Coder or a 27B; loads it, runs prompts on the GPU).
#include "../json.h"
#include "../model.h"
#include "../server.h"
#include "../stats.h"
#include "../test.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>

// words of text, distinct per seed
static void words(Buf *b, uint32_t seed, size_t n) {
    uint64_t x = 0x9E3779B97F4A7C15ull ^ seed;
    static const char *const w[] = {"river", "stone", "lamp", "orbit", "cedar", "quartz", "harbor", "ember", "violet", "anchor",
                                    "meadow", "signal", "copper", "lantern", "glacier", "thimble"};
    buf_printf(b, "Topic %u.", seed);
    for (size_t i = 0; i < n; i++) {
        x ^= x << 13, x ^= x >> 7, x ^= x << 17;
        buf_printf(b, " %s%llu", w[x % 16], (unsigned long long)(x >> 50));
    }
}

static const char TOOLS[] = ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"subagent\",\"description\":\"Hand a task to a "
                            "child agent.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"prompt\":{\"type\":\"string\"}},"
                            "\"required\":[\"prompt\"]}}}]";

// [system (seed sys, nsys words), user (seed u, nu words)], then, with a, [assistant a, user (seed
// u2, nu2 words)], then, when task > 0, a user message with that task: a parent's prompt (task 0;
// fork asks for the fork point) or a child's (its parent's messages, then its task, as Saetta Code's
// subagents send them)
static char *body2(uint32_t sys, size_t nsys, uint32_t u, size_t nu, const char *a, uint32_t u2, size_t nu2, uint32_t task, bool fork,
                   size_t max_tokens) {
    Buf b = {0};
    buf_adds(&b, "{\"messages\":[{\"role\":\"system\",\"content\":\"");
    words(&b, sys, nsys);
    buf_adds(&b, "\"},{\"role\":\"user\",\"content\":\"");
    words(&b, u, nu);
    buf_adds(&b, " Name the topic number.\"}");
    if (a) {
        buf_adds(&b, ",{\"role\":\"assistant\",\"content\":");
        json_write_string(&b, str(a));
        buf_adds(&b, "},{\"role\":\"user\",\"content\":\"");
        words(&b, u2, nu2);
        buf_adds(&b, " And this topic number?\"}");
    }
    if (task) {
        buf_adds(&b, ",{\"role\":\"user\",\"content\":\"You are a subagent. ");
        words(&b, task, 40);
        buf_adds(&b, " Which word comes first? One word.\"}");
    }
    buf_adds(&b, "]");
    buf_adds(&b, TOOLS);
    buf_printf(&b, ",\"max_tokens\":%zu,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}%s}", max_tokens,
               fork ? ",\"saetta_fork_point\":true" : "");
    char *r = strdup(b.ptr);
    buf_free(&b);
    return r;
}
static char *body(uint32_t sys, size_t nsys, uint32_t u, size_t nu, uint32_t task, bool fork, size_t max_tokens) {
    return body2(sys, nsys, u, nu, NULL, 0, 0, task, fork, max_tokens);
}

static int sink_keep(void *ctx, int status, const void *p, size_t n) {
    (void)status;
    if (p && n) buf_add((Buf *)ctx, p, n);
    return 0;
}

// The reply's content (strdup'd) and the request's reused tokens, for agent session sess.
static char *chat_s(Engine *e, const char *b, const char *sess, size_t *reused) {
    Buf out = {0};
    Err err = server_engine_chat_session(e, str(b), sess, sink_keep, &out);
    *reused = atomic_load(&server_engine_stats(e)->reused);
    char *content = NULL;
    Arena a = {0};
    JValue *r;
    if (!err && !json_parse(&a, str_n(out.ptr, out.len), false, &r)) {
        const JValue *ch = json_get(r, "choices");
        const JValue *m = ch && ch->kind == J_ARRAY && ch->arr.len ? json_get(ch->arr.items[0], "message") : NULL;
        const JValue *c = m ? json_get(m, "content") : NULL;
        if (c && c->kind == J_STRING) content = strndup(c->s.ptr, c->s.len);
        const JValue *tc = m ? json_get(m, "tool_calls") : NULL;
        if (!content && tc && tc->kind == J_ARRAY) {
            // a tool call: its functions (names and arguments; the ids are random)
            Buf t = {0};
            for (size_t i = 0; i < tc->arr.len; i++) json_write(&t, json_get(tc->arr.items[i], "function"));
            content = strdup(t.len ? t.ptr : "");
            buf_free(&t);
        }
    }
    arena_reset(&a);
    buf_free(&out);
    return content ? content : strdup("<error>");
}

typedef struct {
    Engine *e;
    const char *body, *sess;
    char *content;
    size_t reused, prompt;
} Job;

// each child on a thread of its own, as Saetta Code runs them (one request at a time in the engine)
static void *run_job(void *arg) {
    Job *j = arg;
    j->content = chat_s(j->e, j->body, j->sess, &j->reused);
    j->prompt = atomic_load(&server_engine_stats(j->e)->prompt_total);
    return NULL;
}
static Job on_thread(Engine *e, const char *b, const char *sess) {
    Job j = {e, b, sess, NULL, 0, 0};
    pthread_t t;
    pthread_create(&t, NULL, run_job, &j);
    pthread_join(t, NULL);
    return j;
}

// short prompts from 0 in every conversation (other callers, nothing shared): what the earlier
// requests left is gone
static bool fill(Engine *e, uint32_t seed) {
    bool ok = true;
    for (uint32_t i = 0; i < 3; i++) {
        Buf b = {0};
        buf_adds(&b, "{\"messages\":[{\"role\":\"user\",\"content\":\"");
        words(&b, seed + i, 30);
        buf_adds(&b, "\"}],\"max_tokens\":2,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}");
        Job j = on_thread(e, b.ptr, NULL);
        ok &= j.reused == 0;
        free(j.content);
        buf_free(&b);
    }
    return ok;
}

static Engine *shared_engine;
static Engine *engine_for(const char *pack) {
    if (shared_engine) return shared_engine;
    setenv("SAETTA_CONVERSATIONS", "3", 1);
    setenv("SAETTA_CTX", "32768", 1);
    static Arena a;
    if (server_engine_open(&a, pack, NULL, 0, NULL, &shared_engine)) return NULL;
    return shared_engine;
}

// a guard: the checkpoint buffers within what the fit reserves, PT_MAX for the active conversation
// and PT_HELD for each of the 2 others (activate trims an inactive conversation to PT_HELD)
static bool slots_in_reserve(Engine *e) {
    size_t act = 0;
    size_t all = server_engine_point_slots(e, &act, NULL);
    return act <= PT_MAX && all <= PT_MAX + 2 * PT_HELD;
}

// A parent that asks for the fork point, then two subagents of it (its messages and a task each, on
// threads of their own, its session id): both start at the checkpoint before the parent's generation
// prompt (n - 1 - PT_TAIL), copying its rows, and answer as their prompts read from 0; the parent
// answers as without the fork point. Then the parent's next request (a long new message, which
// continues its own conversation) and a second batch of three subagents in the three conversations:
// the first starts at the parent's new fork point, not at the older one its conversation holds
// (plan takes the longest start), and the third does not take the parent's conversation (kept),
// whose next request still continues it.
TEST(fork_engine_children_start_at_the_fork_point) {
    const char *pack = getenv("SAETTA_FORK_TEST_PACK");
    if (!pack) SKIP("SAETTA_FORK_TEST_PACK not set (loads a model)");
    Engine *e = engine_for(pack);
    EXPECT(e != NULL);
    if (!e) return;
    // the history: a system prompt and a long user message (about 12K tokens)
    char *c1 = body(101, 600, 102, 2400, 103, false, 12), *c2 = body(101, 600, 102, 2400, 104, false, 12);
    char *p = body(101, 600, 102, 2400, 0, true, 16), *p0 = body(101, 600, 102, 2400, 0, false, 16);
    // what the prompts compute read from 0 in a conversation of their own (fill: nothing shared)
    Job r1 = on_thread(e, c1, "ref");
    EXPECT(fill(e, 900));
    Job r2 = on_thread(e, c2, "ref");
    EXPECT(fill(e, 910));
    Job rp = on_thread(e, p0, "ref");
    EXPECT(fill(e, 920));
    EXPECT_EQ(r1.reused, 0);
    EXPECT_EQ(r2.reused, 0);
    EXPECT_EQ(rp.reused, 0);
    // the parent (this thread, session P): from 0, with the fork point; the same answer
    size_t rpp;
    char *ap = chat_s(e, p, "session-P", &rpp);
    size_t np = atomic_load(&server_engine_stats(e)->prompt_total);
    EXPECT_EQ(rpp, 0);
    EXPECT_STR(str(ap), str(rp.content));
    EXPECT(slots_in_reserve(e));
    // the children: the fork point (the parent's generation prompt is 7 tokens without thinking:
    // <|im_start|>assistant\n<think>\n\n</think>\n\n, so n - 1 - PT_TAIL is before its <|im_start|>)
    Job j1 = on_thread(e, c1, "session-P");
    EXPECT(slots_in_reserve(e));
    Job j2 = on_thread(e, c2, "session-P");
    EXPECT(slots_in_reserve(e));
    EXPECT_EQ(j1.reused, np - 1 - PT_TAIL);
    EXPECT_EQ(j2.reused, np - 1 - PT_TAIL);
    EXPECT_STR(str(j1.content), str(r1.content));
    EXPECT_STR(str(j2.content), str(r2.content));
    // the parent's next request: its own conversation from its reply's end, a new fork point
    char *p2 = body2(101, 600, 102, 2400, ap, 105, 1200, 0, true, 16);
    size_t r2p;
    char *ap2 = chat_s(e, p2, "session-P", &r2p);
    size_t np2 = atomic_load(&server_engine_stats(e)->prompt_total);
    EXPECT(r2p >= np);
    // a second batch: the conversations are the parent's (kept) and the two children's, whose own
    // checkpoints include the old fork point np - 1 - PT_TAIL
    char *c3 = body2(101, 600, 102, 2400, ap, 105, 1200, 106, false, 12), *c4 = body2(101, 600, 102, 2400, ap, 105, 1200, 107, false, 4),
         *c5 = body2(101, 600, 102, 2400, ap, 105, 1200, 108, false, 4);
    Job j3 = on_thread(e, c3, "session-P"), j4 = on_thread(e, c4, "session-P"), j5 = on_thread(e, c5, "session-P");
    EXPECT_EQ(j3.reused, np2 - 1 - PT_TAIL);
    EXPECT_EQ(j4.reused, np2 - 1 - PT_TAIL);
    EXPECT_EQ(j5.reused, np2 - 1 - PT_TAIL);
    EXPECT(slots_in_reserve(e));
    // the parent again: still its conversation, from its last reply's end
    char *p3 = body2(101, 600, 102, 2400, ap, 105, 1200, 0, false, 4);
    {
        // p2's messages, its answer, a short message
        Arena a = {0};
        JValue *pj = NULL;
        EXPECT(!json_parse(&a, str(p2), true, &pj));
        if (pj) {
            JValue *ms = (JValue *)json_get(pj, "messages");
            JValue *as = json_new(&a, J_OBJECT), *us = json_new(&a, J_OBJECT);
            json_put(&a, as, S("role"), json_string(&a, S("assistant")));
            json_put(&a, as, S("content"), json_string(&a, str(ap2)));
            json_put(&a, us, S("role"), json_string(&a, S("user")));
            json_put(&a, us, S("content"), json_string(&a, S("Thanks. And the second word?")));
            json_push(&a, ms, as);
            json_push(&a, ms, us);
            Buf nb = {0};
            json_write(&nb, pj);
            size_t rn;
            free(chat_s(e, nb.ptr, "session-P", &rn));
            EXPECT(rn >= np2);
            buf_free(&nb);
        }
        arena_reset(&a);
    }
    dbg("fork_engine: parent %zu then %zu tokens; children at %zu, %zu, then %zu, %zu, %zu; answers %s | %s (from 0: %s | %s)\n", np,
        np2, j1.reused, j2.reused, j3.reused, j4.reused, j5.reused, j1.content, j2.content, r1.content, r2.content);
    free(c1), free(c2), free(c3), free(c4), free(c5), free(p), free(p0), free(p2), free(p3), free(ap), free(ap2);
    free(r1.content), free(r2.content), free(rp.content), free(j1.content), free(j2.content), free(j3.content), free(j4.content),
        free(j5.content);
}

// Three sessions with one system prompt and tools (about 2.4K tokens), different first messages,
// from three threads: the first reads from 0; the second too (nothing is saved where they part),
// with a checkpoint where it leaves the first one's tokens; the third starts there (main: from 0),
// and answers as its prompt read from 0.
TEST(fork_engine_sessions_share_the_system_prompt) {
    const char *pack = getenv("SAETTA_FORK_TEST_PACK");
    if (!pack) SKIP("SAETTA_FORK_TEST_PACK not set (loads a model)");
    Engine *e = engine_for(pack);
    EXPECT(e != NULL);
    if (!e) return;
    char *sa = body(201, 600, 202, 60, 0, false, 12), *sb = body(201, 600, 203, 60, 0, false, 12),
         *sc = body(201, 600, 204, 60, 0, false, 12);
    Job ref = on_thread(e, sc, "ref");
    EXPECT(fill(e, 950));
    Job ja = on_thread(e, sa, "session-A");
    Job jb = on_thread(e, sb, "session-B");
    Job jc = on_thread(e, sc, "session-C");
    EXPECT_EQ(ja.reused, 0);
    EXPECT_EQ(jb.reused, 0);
    // B and C part from A where their first messages' topic numbers differ ("Topic 20" and a
    // digit): C starts there, at B's checkpoint, and reads its first message only (60 words)
    EXPECT(jc.reused >= PT_GAP);
    EXPECT(jc.reused + 400 > jc.prompt);
    EXPECT_STR(str(jc.content), str(ref.content));
    EXPECT(slots_in_reserve(e));
    dbg("fork_engine: sessions of %zu, %zu, %zu tokens start at %zu, %zu, %zu; C answers %s (from 0: %s)\n", ja.prompt, jb.prompt,
        jc.prompt, ja.reused, jb.reused, jc.reused, jc.content, ref.content);
    free(sa), free(sb), free(sc), free(ref.content), free(ja.content), free(jb.content), free(jc.content);
}
