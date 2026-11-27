// Prefill ahead of the request (server_engine_ahead, src/server.c). The chunk test needs
// SAETTA_AHEAD_CHUNK_PACK and the engine tests SAETTA_AHEAD_TEST_PACK (a Qwen3.8 pack: the Coder or a
// 27B; each loads it and runs prompts on the GPU; set one at a time: the engine tests share one engine,
// and a second model would not fit beside it):
// - prefill is bitwise the same whatever its chunk boundaries (chunks of more than MODEL_SMALL
//   tokens): the logits, every KV row and the recurrent state, so a prompt read in part ahead and
//   in part by its request computes what one read whole computes;
// - a multi-turn tool session answers the same, token for token (greedy, speculation on), with its
//   tool results read ahead while they grow as without, and its requests start past what was read;
// - a request whose tool result differs from what was read ahead (early or late) starts where the
//   two still agree and answers as without;
// - an ahead read yields to a request that waits for the engine's turn, and to its stop flag.
#include "../gpu.h"
#include "../json.h"
#include "../model.h"
#include "../server.h"
#include "../state_store.h"
#include "../stats.h"
#include "../test.h"
#include "../turns.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

TEST(turns_waiting) {
    Turns t;
    turns_init(&t);
    uint64_t a = turns_take(&t);
    EXPECT(turns_wait(&t, a, 0));
    EXPECT(!turns_waiting(&t));
    uint64_t b = turns_take(&t);
    EXPECT(turns_waiting(&t));
    turns_done(&t);
    EXPECT(turns_wait(&t, b, 0));
    EXPECT(!turns_waiting(&t));
    turns_done(&t);
}

// ---- chunk boundaries ------------------------------------------------------------------------

static uint64_t fnv(uint64_t h, const uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

// Reads ids[0..n) from 0 in calls ending at splits (then to n - 1, then the last token alone, as a
// request does): the last logits (copied) and *h, a hash of every row below n of every plane and
// of the recurrent state at n.
static float *read_split(Model *m, Session *s, const uint32_t *ids, size_t n, const size_t *splits, size_t ns, uint64_t *h) {
    session_reset(s);
    size_t at = 0;
    float *lg = NULL;
    for (size_t k = 0; k <= ns; k++) {
        size_t end = k < ns ? splits[k] : n - 1;
        if (end > at) {
            if (model_prefill_with(m, s, ids + at, end - at, NULL, NULL, &lg)) return NULL;
            at = end;
        }
    }
    if (model_prefill(m, s, ids + n - 1, 1, NULL, &lg)) return NULL;
    size_t vocab = model_info(m)->vocab;
    float *c = malloc(vocab * sizeof(float));
    memcpy(c, lg, vocab * sizeof(float));
    *h = 1469598103934665603ull;
    if (session_save_checkpoint_at(s, 1, ids, n)) return c;
    StLayout L;
    if (!session_state_layout(s, 1, n, &L)) return c;
    for (size_t r = 0; r < L.nreg; r++)
        for (size_t p = 0; p < L.reg[r].planes; p++)
            *h = fnv(*h, L.reg[r].base + p * L.reg[r].plane_stride, (n / L.reg[r].per_row) * L.reg[r].row_bytes);
    for (size_t r = 0; r < L.nrec; r++) *h = fnv(*h, L.rec[r], L.rec_bytes[r]);
    return c;
}

TEST(ahead_chunk_boundaries_bitwise) {
    const char *dir = getenv("SAETTA_AHEAD_CHUNK_PACK");
    if (!dir) SKIP("SAETTA_AHEAD_CHUNK_PACK not set (loads a model)");
    Arena arena = {0};
    Pack pack;
    EXPECT_OK(pack_open(&arena, dir, &pack));
    Gpu g;
    EXPECT_OK(gpu_init(&arena, &pack, &g));
    Model *m;
    EXPECT_OK(model_load(&arena, &pack, &g, &m));
    Session *s;
    EXPECT_OK(session_create_cap(m, &g, true, 16384, 16384, &s));
    size_t n = 6000, vocab = model_info(m)->vocab;
    uint32_t *ids = malloc(n * sizeof *ids);
    uint64_t x = 88172645463325252ull;
    for (size_t i = 0; i < n; i++) {
        x ^= x << 13, x ^= x >> 7, x ^= x << 17;
        ids[i] = 200 + (uint32_t)(x % 100000);
    }
    uint64_t href = 0, h = 0;
    float *ref = read_split(m, s, ids, n, NULL, 0, &href);
    EXPECT(ref != NULL);
    // boundaries anywhere (none on a multiple of anything), the shortest chunk 10 tokens (a first
    // chunk of 9 from position 0 differs: the MTP head skips its first row there, and runs its other 8
    // on the small-chunk kernels; an ahead read never leaves a chunk that short)
    const size_t cases[][3] = {{1000, 0, 0}, {1023, 0, 0}, {4097, 0, 0}, {513, 3001, 0}, {10, 2047, 4999}};
    const size_t nc[] = {1, 1, 1, 2, 3};
    for (size_t c = 0; c < COUNT(nc); c++) {
        float *got = read_split(m, s, ids, n, cases[c], nc[c], &h);
        EXPECT(got != NULL);
        size_t diff = 0;
        for (size_t i = 0; i < vocab; i++) diff += got[i] != ref[i];
        free(got);
        dbg("ahead: splits at %zu %zu %zu: %zu logits differ, state %s\n", cases[c][0], cases[c][1], cases[c][2], diff,
            h == href ? "equal" : "differs");
        EXPECT_EQ(diff, 0);
        EXPECT(h == href);
    }
    // a chunk of MODEL_SMALL tokens or fewer runs on the decode-style kernels: not the same rounding
    // (what an ahead read never leaves: server.c AHEAD_MIN, live_continues)
    const size_t small[] = {3000, 3000 + MODEL_SMALL};
    float *got = read_split(m, s, ids, n, small, 2, &h);
    size_t diff = 0;
    for (size_t i = 0; i < vocab; i++) diff += got[i] != ref[i];
    free(got);
    EXPECT(diff > 0 || h != href);
    free(ref);
    free(ids);
}

// ---- through the engine ----------------------------------------------------------------------

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

// A tool's output: lines [0, lines) of a log, seed giving its words; line `odd` (if below lines)
// differs from the log's (a result that changes after it was read ahead).
static void log_lines(Buf *b, uint32_t seed, size_t lines, size_t odd) {
    uint64_t x = 0xD1B54A32D192ED03ull ^ seed;
    static const char *const w[] = {"PASS", "ok", "test_", "build", "src/", ".c:", "warning", "linking", "  ", "[", "]", "ms"};
    for (size_t l = 0; l < lines; l++) {
        if (l == odd) buf_adds(b, "error: something else happened here");
        else {
            buf_printf(b, "%zu", l + 1);
            for (int k = 0; k < 14; k++) {
                x ^= x << 13, x ^= x >> 7, x ^= x << 17;
                buf_printf(b, " %s%llu", w[x % 12], (unsigned long long)(x >> 54));
            }
        }
        buf_addc(b, '\n');
    }
}

typedef struct {
    const char *answer; // the assistant's text answer before a tool result (NULL: none yet)
    uint32_t seed;      // the tool result's log
    size_t lines, odd;  // its length; a line that differs (SIZE_MAX: none)
} Round;

// [user u, then per round: assistant answer, tool result], greedy, no thinking. With only, the
// last round's tool result is lines lines long (a partial result).
static char *body_of(uint32_t user, const Round *rounds, size_t nr, size_t max_tokens) {
    Buf b = {0};
    buf_adds(&b, "{\"messages\":[{\"role\":\"user\",\"content\":\"");
    words(&b, user, 700);
    buf_adds(&b, " Read the logs that follow and answer in one line.\"}");
    for (size_t i = 0; i < nr; i++) {
        buf_adds(&b, ",{\"role\":\"assistant\",\"content\":");
        Buf t = {0};
        Buf j = {0};
        buf_adds(&t, rounds[i].answer);
        json_write_string(&j, buf_str(&t));
        buf_add(&b, j.ptr, j.len);
        buf_free(&t);
        buf_free(&j);
        buf_printf(&b, "},{\"role\":\"tool\",\"tool_call_id\":\"call_%zu\",\"content\":", i);
        Buf lg = {0};
        log_lines(&lg, rounds[i].seed, rounds[i].lines, rounds[i].odd);
        Buf js = {0};
        json_write_string(&js, buf_str(&lg));
        buf_add(&b, js.ptr, js.len);
        buf_free(&lg);
        buf_free(&js);
        buf_adds(&b, "}");
    }
    buf_printf(&b, "],\"max_tokens\":%zu,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}", max_tokens);
    char *s = strdup(b.ptr);
    buf_free(&b);
    return s;
}

static int sink_keep(void *ctx, int status, const void *p, size_t n) {
    (void)status;
    if (p && n) buf_add((Buf *)ctx, p, n);
    return 0;
}

// The reply's content (strdup'd) and the request's reused tokens, for agent session sess.
static char *chat(Engine *e, const char *b, const char *sess, size_t *reused) {
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

static Engine *shared_engine;
static Engine *engine_for(const char *pack) {
    if (shared_engine) return shared_engine;
    setenv("SAETTA_CONVERSATIONS", "4", 1);
    setenv("SAETTA_CTX", "32768", 1);
    // greedy, and depths chosen without the clock: two runs of a session repeat token for token
    setenv("SAETTA_BENCH_GREEDY", "1", 1);
    static Arena a;
    if (server_engine_open(&a, pack, NULL, 0, NULL, &shared_engine)) return NULL;
    return shared_engine;
}

#define ROUNDS 3
// A tool session of ROUNDS rounds for session sess: each round's tool result read ahead in steps
// (ahead: after 1/4, 1/2 and 3/4 of its lines; odd_at: the round whose partial results carry a line
// that differs from the final one, at line odd), then the request with the whole result. answers[i]:
// the reply to request i (0: the user's first request), reused[i] its reused tokens, ahead_read[i]
// the tokens read ahead before it.
static bool session_run(Engine *e, const char *sess, bool ahead, size_t odd_at, size_t odd, char *answers[ROUNDS + 1],
                        size_t reused[ROUNDS + 1], size_t ahead_read[ROUNDS + 1]) {
    Round rounds[ROUNDS];
    const size_t lines[ROUNDS] = {120, 40, 260};
    char *b = body_of(42, NULL, 0, 12);
    answers[0] = chat(e, b, sess, &reused[0]);
    free(b);
    ahead_read[0] = 0;
    for (size_t i = 0; i < ROUNDS; i++) {
        rounds[i] = (Round){answers[i], 100 + (uint32_t)i, lines[i], SIZE_MAX};
        ahead_read[i + 1] = 0;
        if (ahead)
            for (size_t q = 1; q <= 3; q++) {
                rounds[i].lines = lines[i] * q / 4;
                rounds[i].odd = i == odd_at ? odd : SIZE_MAX;
                char *pb = body_of(42, rounds, i + 1, 12);
                char id[16];
                snprintf(id, sizeof id, "call_%zu", i);
                size_t got = 0;
                if (server_engine_ahead(e, str(pb), sess, id, NULL, &got)) return false;
                ahead_read[i + 1] += got;
                free(pb);
            }
        rounds[i].lines = lines[i];
        rounds[i].odd = SIZE_MAX;
        b = body_of(42, rounds, i + 1, 12);
        answers[i + 1] = chat(e, b, sess, &reused[i + 1]);
        free(b);
    }
    return true;
}

// Every conversation taken by a short prompt of another caller, read from 0: no restore point or
// checkpoint of an earlier session is left to start from (the engine has 4).
static void flush(Engine *e) {
    for (int k = 0; k < 4; k++) {
        char b[160];
        snprintf(b, sizeof b, "{\"messages\":[{\"role\":\"user\",\"content\":\"Flush %d %llu.\"}],\"max_tokens\":1}", k,
                 (unsigned long long)now_ns());
        Buf out = {0};
        (void)server_engine_chat_session(e, str(b), NULL, sink_keep, &out);
        buf_free(&out);
    }
}

static void free_answers(char *a[ROUNDS + 1]) {
    for (size_t i = 0; i <= ROUNDS; i++) free(a[i]);
}

TEST(ahead_engine_same_answers) {
    const char *pack = getenv("SAETTA_AHEAD_TEST_PACK");
    if (!pack) SKIP("SAETTA_AHEAD_TEST_PACK not set (loads a model)");
    Engine *e = engine_for(pack);
    EXPECT(e != NULL);
    if (!e) return;
    char *pa[ROUNDS + 1], *aa[ROUNDS + 1];
    size_t pr[ROUNDS + 1], ar[ROUNDS + 1], pz[ROUNDS + 1], az[ROUNDS + 1];
    flush(e);
    EXPECT(session_run(e, "sess-plain", false, SIZE_MAX, 0, pa, pr, pz));
    flush(e);
    EXPECT(session_run(e, "sess-ahead", true, SIZE_MAX, 0, aa, ar, az));
    for (size_t i = 0; i <= ROUNDS; i++) {
        dbg("ahead: request %zu: plain reused %zu, ahead read %zu, reused %zu: \"%s\" | \"%s\"\n", i, pr[i], az[i], ar[i], pa[i], aa[i]);
        EXPECT_STR(str(aa[i]), str(pa[i]));
    }
    // each tool round's request starts past the tokens read ahead for it (the 40-line result too)
    for (size_t i = 1; i <= ROUNDS; i++) {
        EXPECT(az[i] > 0);
        EXPECT_EQ(ar[i], pr[i] + az[i]);
    }
    free_answers(pa);
    free_answers(aa);
}

TEST(ahead_engine_mismatch_rolls_back) {
    const char *pack = getenv("SAETTA_AHEAD_TEST_PACK");
    if (!pack) SKIP("SAETTA_AHEAD_TEST_PACK not set (loads a model)");
    Engine *e = engine_for(pack);
    EXPECT(e != NULL);
    if (!e) return;
    char *pa[ROUNDS + 1], *ea[ROUNDS + 1], *la[ROUNDS + 1];
    size_t pr[ROUNDS + 1], er[ROUNDS + 1], lr[ROUNDS + 1], pz[ROUNDS + 1], ez[ROUNDS + 1], lz[ROUNDS + 1];
    flush(e);
    EXPECT(session_run(e, "sess-plain2", false, SIZE_MAX, 0, pa, pr, pz));
    flush(e);
    // the third round's partial results differ from its final one at its second line (early: before
    // any checkpoint of what was read) and at line 151 of 260 (late: in the last partial only, past a
    // checkpoint at a multiple of PT_EVERY inside the result)
    EXPECT(session_run(e, "sess-early", true, 2, 1, ea, er, ez));
    flush(e);
    EXPECT(session_run(e, "sess-late", true, 2, 150, la, lr, lz));
    for (size_t i = 0; i <= ROUNDS; i++) {
        dbg("ahead: mismatch request %zu: plain reused %zu; early read %zu reused %zu; late read %zu reused %zu\n", i, pr[i], ez[i],
            er[i], lz[i], lr[i]);
        EXPECT_STR(str(ea[i]), str(pa[i]));
        EXPECT_STR(str(la[i]), str(pa[i]));
    }
    // the request of the changed round starts at the last checkpoint before the line that changed:
    // the round's start (its last reply) when that line comes before any, else one inside the result
    EXPECT(ez[3] > 0);
    EXPECT_EQ(er[3], pr[3]);
    EXPECT(lr[3] > pr[3]);
    EXPECT(lr[3] < pr[3] + lz[3]);
    free_answers(pa);
    free_answers(ea);
    free_answers(la);
}

typedef struct {
    Engine *e;
    const char *body, *sess, *id;
    volatile int *stop;
    size_t read;
    uint64_t end;
} AheadJob;

static void *run_ahead(void *arg) {
    AheadJob *j = arg;
    (void)server_engine_ahead(j->e, str(j->body), j->sess, j->id, j->stop, &j->read);
    j->end = now_ns();
    return NULL;
}

TEST(ahead_engine_yields) {
    const char *pack = getenv("SAETTA_AHEAD_TEST_PACK");
    if (!pack) SKIP("SAETTA_AHEAD_TEST_PACK not set (loads a model)");
    Engine *e = engine_for(pack);
    EXPECT(e != NULL);
    if (!e) return;
    size_t r0, r1;
    char *b = body_of(7, NULL, 0, 8);
    char *a0 = chat(e, b, "sess-yield", &r0);
    free(b);
    // a long result (about 9K tokens: 9 chunks or more) read ahead on another thread; a request of
    // another session arrives meanwhile
    Round rd = {a0, 9, 900, SIZE_MAX};
    char *pb = body_of(7, &rd, 1, 8);
    AheadJob j = {e, pb, "sess-yield", "call_0", NULL, 0, 0};
    pthread_t t;
    pthread_create(&t, NULL, run_ahead, &j);
    struct timespec ts = {0, 700 * 1000000L};
    nanosleep(&ts, NULL);
    char *ob = body_of(8, NULL, 0, 4);
    uint64_t t0 = now_ns();
    char *oa = chat(e, ob, "sess-other", &r1);
    uint64_t t1 = now_ns();
    pthread_join(t, NULL);
    dbg("ahead: yielded after %zu tokens, %.0f ms after the request came; the request took %.0f ms\n", j.read,
        (double)(j.end - t0) / 1e6, (double)(t1 - t0) / 1e6);
    // it stopped at the chunk it was reading when the request came, and ended before the request
    EXPECT(j.read > 0);
    EXPECT(j.read < 6000);
    EXPECT(j.end < t1);
    // a stop flag set before: nothing read
    volatile int stop = 1;
    size_t got = 99;
    EXPECT_OK(server_engine_ahead(e, str(pb), "sess-yield", "call_0", &stop, &got));
    EXPECT_EQ(got, 0);
    free(a0), free(pb), free(ob), free(oa);
}
