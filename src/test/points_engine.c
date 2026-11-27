// Checkpoints at message boundaries through the in-process engine (src/server.c), with a model (one
// engine for all four tests, which run in this order):
// a request whose prompt leaves the conversation's history at a message start restarts at that
// start's checkpoint, and what it generates equals the same request read on without a restart
// (greedy, speculation as configured). Needs SAETTA_POINTS_TEST_PACK (a Qwen3.8 pack: the Coder or
// a 27B; loads it, runs prompts on the GPU).
#include "../json.h"
#include "../model.h"
#include "../server.h"
#include "../stats.h"
#include "../test.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>

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

// [user u1], [user u1, assistant a1, user u2] or, with a2, [user u1, assistant a1, user u2,
// assistant a2, user u3], greedy, no thinking
static char *body5(uint32_t u1, const char *a1, uint32_t u2, const char *a2, uint32_t u3, size_t max_tokens) {
    Buf b = {0};
    buf_adds(&b, "{\"messages\":[{\"role\":\"user\",\"content\":\"");
    words(&b, u1, 2500);
    buf_adds(&b, "\"}");
    if (a1) {
        buf_adds(&b, ",{\"role\":\"assistant\",\"content\":\"");
        buf_adds(&b, a1);
        buf_adds(&b, "\"},{\"role\":\"user\",\"content\":\"");
        words(&b, u2, 1500);
        buf_adds(&b, " Summarize the topic number in one line.\"}");
    }
    if (a2) {
        buf_adds(&b, ",{\"role\":\"assistant\",\"content\":\"");
        buf_adds(&b, a2);
        buf_adds(&b, "\"},{\"role\":\"user\",\"content\":\"");
        words(&b, u3, 200);
        buf_adds(&b, " Which topic numbers came up? One line.\"}");
    }
    buf_printf(&b, "],\"max_tokens\":%zu,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}", max_tokens);
    char *s = strdup(b.ptr);
    buf_free(&b);
    return s;
}
static char *body(uint32_t u1, const char *a1, uint32_t u2, size_t max_tokens) { return body5(u1, a1, u2, NULL, 0, max_tokens); }

static int sink_keep(void *ctx, int status, const void *p, size_t n) {
    (void)status;
    if (p && n) buf_add((Buf *)ctx, p, n);
    return 0;
}

// The reply's content (strdup'd) and the request's reused tokens, for agent session sess (NULL:
// none, as another caller).
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
    }
    arena_reset(&a);
    buf_free(&out);
    return content ? content : strdup("<error>");
}
// this thread's requests in one agent session (Saetta Code's)
static char *chat(Engine *e, const char *b, size_t *reused) { return chat_s(e, b, "session-main", reused); }

typedef struct {
    int fd;
    const char *b;
    Buf out;
} Sender;

// the client's side: the request (larger than the socket's buffer), then the whole response (its
// head and content-length bytes), then it closes (a client that half-closes early is gone)
static void *send_request(void *arg) {
    Sender *s = arg;
    char head[200];
    int hl = snprintf(head, sizeof head, "POST /v1/chat/completions HTTP/1.1\r\nhost: test\r\ncontent-type: application/json\r\ncontent-length: %zu\r\n\r\n", strlen(s->b));
    Buf all = {0};
    buf_add(&all, head, (size_t)hl);
    buf_adds(&all, s->b);
    for (size_t at = 0; at < all.len;) {
        ssize_t w = write(s->fd, all.ptr + at, all.len - at);
        if (w <= 0) break;
        at += (size_t)w;
    }
    buf_free(&all);
    char tmp[4096];
    for (;;) {
        Str got = str_n(s->out.ptr, s->out.len);
        ptrdiff_t at = str_find(got, S("\r\n\r\n"), 0);
        if (at >= 0) {
            ptrdiff_t cl = str_find(got, S("content-length: "), 0);
            if (cl >= 0 && cl < at && s->out.len >= (size_t)at + 4 + (size_t)strtoull(s->out.ptr + cl + 16, NULL, 10)) break;
        }
        ssize_t r = read(s->fd, tmp, sizeof tmp);
        if (r <= 0) break;
        buf_add(&s->out, tmp, (size_t)r);
    }
    close(s->fd);
    return NULL;
}

// One HTTP request on a connection served on this thread (as saetta serve serves every client on
// one thread; only the client's write runs on another): the reply's content (strdup'd), the
// request's reused tokens.
static char *http_chat(Engine *e, const char *b, size_t *reused) {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv)) return strdup("<socketpair>");
    Sender snd = {sv[0], b, {0}};
    pthread_t t;
    pthread_create(&t, NULL, send_request, &snd);
    (void)server_engine_connection(e, sv[1], NULL, NULL);
    pthread_join(t, NULL);
    close(sv[1]);
    Buf out = snd.out;
    *reused = atomic_load(&server_engine_stats(e)->reused);
    char *content = NULL;
    Str all = str_n(out.ptr, out.len);
    ptrdiff_t at = str_find(all, S("\r\n\r\n"), 0);
    Arena a = {0};
    JValue *j;
    if (at >= 0 && !json_parse(&a, str_sub(all, (size_t)at + 4, all.len), false, &j)) {
        const JValue *ch = json_get(j, "choices");
        const JValue *m = ch && ch->kind == J_ARRAY && ch->arr.len ? json_get(ch->arr.items[0], "message") : NULL;
        const JValue *c = m ? json_get(m, "content") : NULL;
        if (c && c->kind == J_STRING) content = strndup(c->s.ptr, c->s.len);
    }
    arena_reset(&a);
    buf_free(&out);
    return content ? content : strdup("<error>");
}

// One engine for the tests below (a second model would not fit beside the first): three
// conversations, the state store in a temporary folder.
static Engine *shared_engine;
static char store_root[64];
static Engine *engine_for(const char *pack) {
    if (shared_engine) return shared_engine;
    setenv("SAETTA_CONVERSATIONS", "3", 1);
    setenv("SAETTA_CTX", "32768", 1);
    static Arena a;
    snprintf(store_root, sizeof store_root, "/tmp/saetta-points-store-XXXXXX");
    if (!mkdtemp(store_root)) return NULL;
    if (server_engine_open(&a, pack, NULL, 0, NULL, &shared_engine)) return NULL;
    if (server_engine_state_enable(shared_engine, store_root, "test")) shared_engine = NULL;
    return shared_engine;
}

TEST(points_engine_restart_equals_continuous) {
    const char *pack = getenv("SAETTA_POINTS_TEST_PACK");
    if (!pack) SKIP("SAETTA_POINTS_TEST_PACK not set (loads a model)");
    Engine *e = engine_for(pack);
    EXPECT(e != NULL);
    if (!e) return;
    size_t r1, r2a, r2b, r2c;
    char *b1 = body(1, NULL, 0, 16);
    char *c1 = chat(e, b1, &r1);
    EXPECT_EQ(r1, 0);
    // fixed earlier answers (so both prompts share them exactly), then two different last user
    // messages: B and A share [u1, a1, u2, a2] (about 8K tokens past the first request)
    const char *a1 = "The topic is number one.", *a2 = "Topic two was about lanterns.";
    char *bA = body5(1, a1, 2, a2, 4, 24), *bB = body5(1, a1, 2, a2, 3, 24);
    // B read on from the first request's restore point: its prefill keeps checkpoints at message
    // starts and multiples of PT_EVERY inside the new messages (none forced)
    char *cB1 = chat(e, bB, &r2a);
    // A leaves B's history inside its last user message: it restarts at the last checkpoint before
    // that, a cut inside B's prefill (past B's own start)
    char *cA = chat(e, bA, &r2b);
    EXPECT(r2b > r2a + PT_GAP);
    // B again: its own history left at the same message start: the same checkpoint (saved while B
    // was read on the first time, kept by A, which started there), and the same answer as when it
    // was read without a restart
    char *cB2 = chat(e, bB, &r2c);
    EXPECT_EQ(r2c, r2b);
    EXPECT_STR(str(cB2), str(cB1));
    dbg("points_engine: reused %zu then %zu, %zu; answers %s | %s | %s\n", r2a, r2b, r2c, cB1, cA, cB2);
    free(b1), free(c1), free(bA), free(bB), free(cB1), free(cA), free(cB2);
}

// The state store keeps the last two checkpoints of a conversation: after the conversation is gone
// from memory (here: two other prompts take both slots from 0), a request that leaves the saved
// state between them (the last request again: a regenerated answer) restarts at the second one,
// and answers as the resident conversation did.
TEST(points_engine_store_second_snapshot) {
    const char *pack = getenv("SAETTA_POINTS_TEST_PACK");
    if (!pack) SKIP("SAETTA_POINTS_TEST_PACK not set (loads a model)");
    Engine *e = engine_for(pack);
    EXPECT(e != NULL);
    if (!e) return;
    size_t r1, r2, r3, r4;
    char *b1 = body(5, NULL, 0, 16);
    free(chat(e, b1, &r1));
    const char *a1 = "The topic is number five.";
    char *bB = body(5, a1, 6, 24);
    char *cB1 = chat(e, bB, &r2);
    size_t nB = atomic_load(&server_engine_stats(e)->prompt_total);
    server_engine_state_save(e);
    server_engine_state_flush(e);
    // three other conversations from 0 in every slot (other callers: no in-place restart)
    char *bX = body(77, NULL, 0, 4), *bY = body(78, NULL, 0, 4), *bZ = body(79, NULL, 0, 4);
    free(chat_s(e, bX, NULL, &r3));
    EXPECT_EQ(r3, 0);
    free(chat_s(e, bY, NULL, &r3));
    EXPECT_EQ(r3, 0);
    free(chat_s(e, bZ, NULL, &r3));
    EXPECT_EQ(r3, 0);
    // B again: its saved state's restore point is past this prompt (it holds the answer); its
    // second snapshot, the prompt's own restore point one token before its end, is not
    char *cB2 = chat(e, bB, &r4);
    EXPECT_EQ(r4, nB - 1);
    EXPECT_STR(str(cB2), str(cB1));
    dbg("points_engine: B %zu tokens, restored at %zu; answers %s | %s\n", nB, r4, cB1, cB2);
    free(b1), free(bB), free(cB1), free(bX), free(bY), free(bZ), free(cB2);
}

typedef struct {
    Engine *e;
    const char *body;
    char *content;
    size_t reused;
} Job;

static void *run_job(void *arg) {
    Job *j = arg;
    j->content = chat_s(j->e, j->body, NULL, &j->reused);
    return NULL;
}

// A retry by the caller whose conversation holds the checkpoint restarts there in that
// conversation; the same prompt from another thread (a subagent sharing the prefix) gets the other
// conversation, which starts from a copy of the rows at that checkpoint, and answers the same.
TEST(points_engine_route_retry_and_shared) {
    const char *pack = getenv("SAETTA_POINTS_TEST_PACK");
    if (!pack) SKIP("SAETTA_POINTS_TEST_PACK not set (loads a model)");
    Engine *e = engine_for(pack);
    EXPECT(e != NULL);
    if (!e) return;
    size_t r1, r2, r3;
    char *b1 = body(9, NULL, 0, 16);
    free(chat(e, b1, &r1));
    const char *a1 = "The topic is number nine.", *a2 = "Topic ten was about cedars.";
    char *bA = body5(9, a1, 10, a2, 12, 24), *bB = body5(9, a1, 10, a2, 11, 24);
    // B read on without a restart, then A, a retry of that last user message, by the same caller
    char *cB = chat(e, bB, &r2);
    free(chat(e, bA, &r3));
    size_t nA = atomic_load(&server_engine_stats(e)->prompt_total);
    EXPECT(r3 > r2 + PT_GAP);
    // B on another thread: its own conversation, from the rows copied at the same checkpoint (A,
    // which started there in conversation 1, kept it), and B's answer
    Job j = {e, bB, NULL, 0};
    pthread_t t;
    pthread_create(&t, NULL, run_job, &j);
    pthread_join(t, NULL);
    EXPECT_EQ(j.reused, r3);
    EXPECT_STR(str(j.content), str(cB));
    // the same thread in another agent session (after /new, /resume or a /fork to a new file):
    // a prompt that A's conversation's checkpoints start does not restart there in place but in
    // another conversation, so the first session's conversation stays as it was
    char *bC = body5(9, a1, 10, a2, 13, 24);
    size_t r4, r5;
    free(chat_s(e, bC, "session-other", &r4));
    EXPECT(r4 > r2 + PT_GAP);
    free(chat(e, bA, &r5));
    EXPECT_EQ(r5, nA - 1);
    dbg("points_engine: own retry at %zu, shared start at %zu, other session at %zu, A again at %zu; answers %s | %s\n", r3,
        j.reused, r4, r5, cB, j.content);
    free(b1), free(bA), free(bB), free(bC), free(cB), free(j.content);
}

// Two HTTP sessions on one thread (one saetta serve, two pi sessions): Q's prompt shares P's first
// message, so P's conversation holds a checkpoint that starts it; Q gets the other conversation
// (from a copy of P's rows), never P's in place, and P's next request still finds its own
// conversation as it left it.
TEST(points_engine_http_sessions_keep_their_conversations) {
    const char *pack = getenv("SAETTA_POINTS_TEST_PACK");
    if (!pack) SKIP("SAETTA_POINTS_TEST_PACK not set (loads a model)");
    Engine *e = engine_for(pack);
    EXPECT(e != NULL);
    if (!e) return;
    size_t r;
    char *p1 = body(30, NULL, 0, 8);
    free(http_chat(e, p1, &r));
    const char *a1 = "The topic is number thirty.";
    char *pB = body(30, a1, 31, 16), *qB = body(30, a1, 32, 16);
    free(http_chat(e, pB, &r));
    size_t nP = atomic_load(&server_engine_stats(e)->prompt_total);
    // Q, on the same thread: it starts from P's checkpoint by copying (another conversation)
    free(http_chat(e, qB, &r));
    size_t rq = r;
    EXPECT(rq > 0);
    // P again: its conversation still ends where P left it (its prompt's restore point)
    free(http_chat(e, pB, &r));
    EXPECT_EQ(r, nP - 1);
    // and Q again likewise
    free(http_chat(e, qB, &r));
    EXPECT_EQ(r, atomic_load(&server_engine_stats(e)->prompt_total) - 1);
    dbg("points_engine: http P %zu tokens, Q started at %zu, P again at %zu\n", nP, rq, nP - 1);
    free(p1), free(pB), free(qB);
    char cmd[200];
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", store_root);
    (void)system(cmd);
}
