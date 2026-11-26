// Inkling 2 Mini at long contexts, the engine against itself (no oracle reaches these lengths).
// Off in a plain `make test`: it runs with SAETTA_TEST_LONG=1, SAETTA_PACK=<Inkling pack> and
// SAETTA_INK_LONG_IDS (a whitespace-separated file of token ids, repeated to the length), at
// SAETTA_INK_LONG tokens (default and at most 65536; longer contexts are a manual run at release).
// The prompt is read three ways on f16-KV sessions:
//   A: prompt chunks of 256 tokens (the default prefill);
//   B: chunks of 200 (SAETTA_INK_PCH), so chunk boundaries and the ring wraps fall elsewhere;
//   C: chunks of 256 up to the last 32 tokens, then those 32 one at a time (the decode kernels);
// and the last token's distributions must agree: KL(A||B), KL(A||C) at most TOL_PREFILL_FAST_KL
// and the same argmax. Times per path go to stderr. Skipped without the variables.
#include <math.h>
#include <stdlib.h>

#include "../gpu.h"
#include "../model.h"
#include "../test.h"
#include "../tolerances.h"

static double kl_last(const float *ref, const float *eng, size_t n) {
    double mr = -INFINITY, me = -INFINITY, zr = 0, ze = 0, d = 0;
    for (size_t i = 0; i < n; i++) mr = fmax(mr, ref[i]), me = fmax(me, eng[i]);
    for (size_t i = 0; i < n; i++) zr += exp((double)ref[i] - mr), ze += exp((double)eng[i] - me);
    for (size_t i = 0; i < n; i++) {
        const double lp = (double)ref[i] - mr - log(zr), lq = (double)eng[i] - me - log(ze);
        d += exp(lp) * (lp - lq);
    }
    return d;
}

TEST(ink_long_self_consistency) {
    const char *dir = getenv("SAETTA_PACK"), *nv = getenv("SAETTA_INK_LONG"), *idf = getenv("SAETTA_INK_LONG_IDS");
    const char *on = getenv("SAETTA_TEST_LONG");
    if (!on || strcmp(on, "1") != 0 || !dir || !idf) SKIP("SAETTA_TEST_LONG=1, SAETTA_PACK and SAETTA_INK_LONG_IDS not set");
    const size_t N = MIN(nv ? (size_t)strtoull(nv, NULL, 10) : 65536, (size_t)65536);
    EXPECT(N > 64);
    Arena a = {0};
    char *text;
    size_t len;
    EXPECT_OK(read_file(idf, 1u << 30, &text, &len));
    VEC(uint32_t) src = {0};
    for (char *p = text, *e; p < text + len; p = e) {
        unsigned long v = strtoul(p, &e, 10);
        if (e == p) break;
        vec_push(&src, (uint32_t)v);
    }
    free(text);
    EXPECT(src.len > 0);
    uint32_t *toks = ARENA_ARRAY(&a, uint32_t, N);
    for (size_t i = 0; i < N; i++) toks[i] = src.ptr[i % src.len];
    Pack pack;
    EXPECT_OK(pack_open(&a, dir, &pack));
    Gpu g;
    EXPECT_OK(gpu_init(&a, &pack, &g));
    Model *m;
    EXPECT_OK(model_load(&a, &pack, &g, &m));
    const ModelInfo *mi = model_info(m);
    const size_t V = mi->vocab, cap = ALIGN_UP(N + 64, SESSION_CTX_ALIGN);
    float *la = ARENA_ARRAY(&a, float, V), *lb = ARENA_ARRAY(&a, float, V), *lc = ARENA_ARRAY(&a, float, V);
    float *lg;
    Session *s;
    EXPECT_OK(session_create_cap(m, &g, true, cap, cap, &s));
    double t0 = now_s();
    EXPECT_OK(model_prefill(m, s, toks, N, NULL, &lg));
    memcpy(la, lg, V * 4);
    double ta = now_s() - t0;
    double tb = 0;
    {
        session_reset(s);
        setenv("SAETTA_INK_PCH", "200", 1);
        t0 = now_s();
        Err e = model_prefill(m, s, toks, N, NULL, &lg);
        unsetenv("SAETTA_INK_PCH");
        EXPECT_OK(e);
        memcpy(lb, lg, V * 4);
        tb = now_s() - t0;
    }
    session_reset(s);
    t0 = now_s();
    EXPECT_OK(model_prefill(m, s, toks, N - 32, NULL, &lg));
    double tc0 = now_s() - t0;
    t0 = now_s();
    for (size_t i = N - 32; i < N; i++) EXPECT_OK(model_step(m, s, toks[i], &lg));
    memcpy(lc, lg, V * 4);
    double tc1 = now_s() - t0;
    const double kab = kl_last(la, lb, V), kac = kl_last(la, lc, V);
    const uint32_t xa = model_argmax(la, V), xb = model_argmax(lb, V), xc = model_argmax(lc, V);
    dbg("ink_long %zu tokens: prefill 256-chunks %.1f s (%.0f tok/s), 200-chunks %.1f s; %zu + 32 decode steps %.1f s + %.1f s "
        "(%.1f tok/s at the end)\n",
        N, ta, (double)N / ta, tb, N - 32, tc0, tc1, 32.0 / tc1);
    dbg("ink_long %zu tokens: KL(A||B) %.3g, KL(A||C) %.3g, argmax %u %u %u\n", N, kab, kac, xa, xb, xc);
    EXPECT(kab <= TOL_PREFILL_FAST_KL && kac <= TOL_PREFILL_FAST_KL);
    EXPECT(xa == xb && xa == xc);
    vec_free(&src);
}

// Speculative decoding with the DSpark drafter against plain decoding: with SAETTA_PACK=<Inkling
// pack> and the drafter found (src/inkling.c ds_load) and SAETTA_INK_LONG_IDS (the prompt: its
// first 2048 tokens), 200 greedy tokens both ways must be identical, token for token (the
// verification chunk computes each token with the same bits as decode, and only accepted rows are
// committed). Then session_roll_to into the last verification chunk and decoding on from there
// must give the plain run's tokens again (the rings and conv states restored exactly).
static uint32_t pick_greedy(void *ctx, const float *row, bool has_draft, uint32_t draft) {
    (void)has_draft, (void)draft;
    return model_argmax(row, *(const size_t *)ctx);
}

TEST(ink_dspark_greedy_equal) {
    const char *dir = getenv("SAETTA_PACK"), *idf = getenv("SAETTA_INK_LONG_IDS");
    if (!dir || !idf) SKIP("SAETTA_PACK and SAETTA_INK_LONG_IDS not set");
    Arena a = {0};
    char *text;
    size_t len;
    EXPECT_OK(read_file(idf, 1u << 30, &text, &len));
    VEC(uint32_t) src = {0};
    for (char *p = text, *e; p < text + len && src.len < 2048; p = e) {
        unsigned long v = strtoul(p, &e, 10);
        if (e == p) break;
        vec_push(&src, (uint32_t)v);
    }
    free(text);
    // SAETTA_TEST_REPEAT=1: the first 1024 tokens, then their first 512 again (a continuation the
    // drafter should mostly copy)
    if (getenv("SAETTA_TEST_REPEAT") && src.len >= 1024) {
        src.len = 1024;
        for (size_t i = 0; i < 512; i++) vec_push(&src, src.ptr[i]);
    }
    Pack pack;
    EXPECT_OK(pack_open(&a, dir, &pack));
    Gpu g;
    EXPECT_OK(gpu_init(&a, &pack, &g));
    Model *m;
    EXPECT_OK(model_load(&a, &pack, &g, &m));
    if (!model_has_mtp(m)) SKIP("no DSpark drafter downloaded (saetta download inkling-dspark)");
    size_t V = model_info(m)->vocab;
    enum { NG = 200 };
    uint32_t plain[NG + 1], spec[NG + 1];
    Session *s;
    EXPECT_OK(session_create_cap(m, &g, true, 4096, 4096, &s));
    float *lg;
    // plain
    EXPECT_OK(model_prefill(m, s, src.ptr, src.len, NULL, &lg));
    uint32_t t = model_argmax(lg, V);
    double t0 = now_s();
    for (size_t i = 0; i < NG; i++) {
        plain[i] = t;
        EXPECT_OK(model_step(m, s, t, &lg));
        t = model_argmax(lg, V);
    }
    plain[NG] = t;
    const double tp = now_s() - t0;
    // speculative: drafts from the drafter, verified in chunks
    session_reset(s);
    EXPECT_OK(model_prefill(m, s, src.ptr, src.len, NULL, &lg));
    t = model_argmax(lg, V);
    size_t n = 0, passes = 0, acc = 0;
    const char *kde = getenv("SAETTA_TEST_DRAFTS");
    const size_t kd = kde ? (size_t)atoi(kde) : KMAX - 1;
    double tdraft = 0, tver = 0;
    size_t hist[KMAX] = {0};
    t0 = now_s();
    while (n < NG) {
        uint32_t drafts[KMAX], nd = 0;
        size_t ndr = 0;
        const double d0 = now_s();
        EXPECT_OK(model_draft_mtp(m, s, t, kd, drafts, &ndr));
        tdraft += now_s() - d0;
        nd = (uint32_t)ndr;
        passes++;
        if (nd == 0) {
            spec[n++] = t;
            EXPECT_OK(model_step(m, s, t, &lg));
            t = model_argmax(lg, V);
            continue;
        }
        SpecResult r;
        const double v0 = now_s();
        EXPECT_OK(model_spec_verify(m, s, t, drafts, nd, pick_greedy, &V, &r));
        tver += now_s() - v0;
        spec[n++] = t;
        for (size_t i = 0; i < r.accepted && n < NG; i++) spec[n++] = drafts[i];
        acc += r.accepted;
        hist[r.accepted]++;
        t = r.next;
    }
    const double ts = now_s() - t0;
    size_t same = 0;
    while (same < NG && spec[same] == plain[same]) same++;
    dbg("ink_dspark: %d tokens, plain %.1f tok/s, speculative %.1f tok/s (%zu passes, %.2f drafts accepted per pass, draft %.1f ms, "
        "verify %.1f ms per pass); equal for %zu\n",
        NG, NG / tp, NG / ts, passes, (double)acc / (double)passes, 1e3 * tdraft / passes, 1e3 * tver / passes, same);
    dbg("ink_dspark: accepted per pass 0..7:");
    for (size_t i = 0; i < KMAX; i++) dbg(" %zu", hist[i]);
    dbg("\n");
    EXPECT_EQ(same, NG);
    // roll back into the last verification chunk and decode on: the plain tokens again
    const size_t pos = s->pos, base = src.len;
    if (s->chunk_t > 1) {
        const size_t p = s->chunk_pos0 + 1;
        EXPECT(session_roll_to(s, p, NULL));
        t = plain[p - base];
        for (size_t i = p - base; i < NG; i++) {
            EXPECT_EQ(t, plain[i]);
            EXPECT_OK(model_step(m, s, t, &lg));
            t = model_argmax(lg, V);
        }
        dbg("ink_dspark: rolled back from %zu to %zu, plain tokens again\n", pos, p);
    }
    vec_free(&src);
}

// ---- speculative decoding is exact: every verification against plain decoding, bit for bit ----
//
// Runs in `SAETTA_PACK=<Inkling pack> make test` (skipped for other packs) on a self-generated
// prompt (pseudo-random ids repeating with period 200, so prompt lookup finds copies). Two
// sessions read the same tokens: A decodes plainly, B speculates; at probe positions (1000: the
// 1024 ring wrap; 2290: queries of one chunk on different split counts; 5100: past 4K and the
// wrap at 5120) A decodes 48 greedy tokens, B reaches the same 48 with a schedule of single
// steps, the DSpark drafter's drafts (depth 1..7), prompt-lookup drafts, the right tokens (all
// accepted, then a rollback 2..7 deep, then a verification chunk again) and the right tokens
// with one wrong. Every logits row a verification returns for an accepted position, and every
// single step, must equal A's row at that position bit for bit (FNV-1a over the floats), on
// f16 and f32 caches.
#include "../spec.h"

typedef struct {
    size_t V, n;
    uint64_t got[KMAX];
} RowHash;

static uint64_t row_hash(const float *row, size_t n) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t *b = (const uint8_t *)row;
    for (size_t i = 0; i < n * 4; i++) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

static uint32_t pick_hash(void *ctx, const float *row, bool has_draft, uint32_t draft) {
    (void)has_draft, (void)draft;
    RowHash *h = ctx;
    h->got[h->n++] = row_hash(row, h->V);
    return model_argmax(row, h->V);
}

enum { SPX_G = 48 };
typedef struct {
    size_t rows, mism, depth_seen[KMAX], wrap_verify, wrap_rollback, drafter, lookup, verifies;
} SpxStats;

// One probe at A's position Q (A and B hold the same tokens; t0 the token after them): both end
// at Q + SPX_G + 8 with the same tokens, B's rows checked against A's.
static void spx_probe(Model *m, Session *A, Session *B, Lookup *lk, size_t V, uint32_t t0, SpxStats *st, uint64_t *step,
                      int *failed_) {
    uint32_t ptok[SPX_G + 9];
    uint64_t hA[SPX_G + 8];
    float *lg;
    const size_t Q = A->pos;
    ptok[0] = t0;
    for (size_t i = 0; i < SPX_G + 8; i++) {
        EXPECT_OK(model_step(m, A, ptok[i], &lg));
        hA[i] = row_hash(lg, V);
        ptok[i + 1] = model_argmax(lg, V);
    }
    size_t i = 0;
    uint32_t x = ptok[0];
    bool wrap_done = false;
    while (i < SPX_G) {
        const uint64_t s = (*step)++;
        const size_t pos = Q + i, wrap = (pos / 1024 + 1) * 1024;
        const bool at_wrap = !wrap_done && wrap <= pos + 7; // a chunk of 8 from here crosses the wrap
        int kind = at_wrap ? 3 : (int)(s % 5);
        uint32_t dr[KMAX];
        size_t nd = 0;
        if (kind == 1) {
            EXPECT_OK(model_draft_mtp(m, B, x, 1 + s % 7, dr, &nd));
            st->drafter += nd > 0;
        } else if (kind == 2) {
            nd = lookup_draft(lk, dr, 1 + s % 7);
            st->lookup += nd > 0;
        } else if (kind >= 3) {
            nd = kind == 3 ? KMAX - 1 : 2 + s % 6; // all 7 right (then a rollback), or 2..7 with one wrong
            for (size_t j = 0; j < nd; j++) dr[j] = ptok[i + 1 + j];
            if (kind == 4) dr[s % nd] = (dr[s % nd] + 1) % (uint32_t)V; // one wrong
        }
        if (nd == 0) {
            EXPECT_OK(model_step(m, B, x, &lg));
            st->rows++;
            st->mism += row_hash(lg, V) != hA[i];
            x = ptok[++i];
            continue;
        }
        RowHash h = {.V = V};
        SpecResult r;
        EXPECT_OK(model_spec_verify(m, B, x, dr, nd, pick_hash, &h, &r));
        st->verifies++;
        EXPECT_EQ(h.n, r.accepted + 1);
        for (size_t j = 0; j <= r.accepted; j++) {
            st->rows++;
            st->mism += h.got[j] != hA[i + j];
        }
        if (wrap <= pos + nd) st->wrap_verify++;
        EXPECT_EQ(r.next, ptok[i + r.accepted + 1]);
        i += r.accepted + 1;
        x = r.next;
        if (kind == 3) {
            // all right: roll back 2..7 rows (at a wrap: to just after x, the wrap's row undone),
            // and the next pass verifies from there
            EXPECT_EQ(r.accepted, nd);
            const size_t end = B->pos, p = at_wrap ? pos + 1 : end - (2 + (s / 5) % 6);
            EXPECT(session_roll_to(B, p, NULL));
            EXPECT_EQ(B->pos, p);
            st->depth_seen[end - p]++;
            if (at_wrap) {
                wrap_done = true;
                st->wrap_rollback++;
            }
            i = p - Q;
            x = ptok[i];
        }
    }
    for (; i < SPX_G + 8; i++) {
        EXPECT_OK(model_step(m, B, ptok[i], &lg));
        st->rows++;
        st->mism += row_hash(lg, V) != hA[i];
    }
    EXPECT_EQ(B->pos, A->pos);
    for (size_t j = 0; j < SPX_G + 8; j++) lookup_push(lk, ptok[j]);
}

TEST(ink_spec_exact) {
    const char *dir = getenv("SAETTA_PACK");
    if (!dir) SKIP("SAETTA_PACK not set");
    Arena a = {0};
    Pack pack;
    EXPECT_OK(pack_open(&a, dir, &pack));
    Gpu g;
    EXPECT_OK(gpu_init(&a, &pack, &g));
    Model *m;
    EXPECT_OK(model_load(&a, &pack, &g, &m));
    if (model_info(m)->chat != CHAT_INKLING) SKIP("not an Inkling pack");
    const size_t V = model_info(m)->vocab;
    enum { NP = 5100 };
    static uint32_t prompt[NP], block[200];
    uint64_t lcg = 12345;
    for (size_t i = 0; i < 200; i++) {
        lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
        block[i] = 1000 + (uint32_t)((lcg >> 33) % 90000);
    }
    for (size_t i = 0; i < NP; i++) prompt[i] = block[i % 200];
    SpxStats st = {0};
    uint64_t step = 0;
    const double t0 = now_s();
    for (int half = 1; half >= 0; half--) {
        static const size_t probes_h[] = {1000, 2290, 5100}, probes_f[] = {1000, 2290};
        const size_t *pr = half ? probes_h : probes_f, np = half ? 3 : 2;
        Session *A, *B;
        EXPECT_OK(session_create_cap(m, &g, half, 6144, 6144, &A));
        EXPECT_OK(session_create_cap(m, &g, half, 6144, 6144, &B));
        Lookup lk = {0};
        for (size_t k = 0; k < np; k++) {
            const size_t from = A->pos, n = pr[k] - from;
            float *la, *lb;
            EXPECT_OK(model_prefill(m, A, prompt + from, n, NULL, &la));
            const uint64_t ha = row_hash(la, V);
            const uint32_t t = model_argmax(la, V);
            EXPECT_OK(model_prefill(m, B, prompt + from, n, NULL, &lb));
            EXPECT_EQ(row_hash(lb, V), ha);
            for (size_t j = 0; j < n; j++) lookup_push(&lk, prompt[from + j]);
            spx_probe(m, A, B, &lk, V, t, &st, &step, failed_);
            if (*failed_) return;
        }
        lookup_free(&lk);
    }
    dbg("ink_spec_exact: %zu rows checked bit for bit (%zu verification chunks: %zu with drafter drafts, %zu with prompt-lookup "
        "drafts, %zu across a ring wrap), %zu different; rollbacks by depth 2..7:",
        st.rows, st.verifies, st.drafter, st.lookup, st.wrap_verify, st.mism);
    for (size_t d = 2; d < KMAX; d++) dbg(" %zu", st.depth_seen[d]);
    dbg(", %zu across a wrap; drafter %s; %.1f s\n", st.wrap_rollback, model_has_mtp(m) ? "loaded" : "not loaded", now_s() - t0);
    EXPECT_EQ(st.mism, 0);
    for (size_t d = 2; d < KMAX; d++) EXPECT(st.depth_seen[d] > 0);
    EXPECT(st.wrap_verify > 0 && st.wrap_rollback > 0 && st.lookup > 0);
    if (model_has_mtp(m)) EXPECT(st.drafter > 0);
}
