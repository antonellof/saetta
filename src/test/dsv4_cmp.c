// DeepSeek-V4-Flash compressor and indexer kernels (src/kernels/dsv4_cmp.metal) against the
// scalar reference (src/dsv4_ref_cmp.c), on random data.
//
// Compressor, for the three shapes (CSA ratio 4 d 512, HCA ratio 128 d 512, the indexer's
// ratio 4 d 128): the kernels repeat the reference's operations in its order (its exp
// included), so the pooled values, the entries (also from the GPU's own pooled values, to
// locate a failure) and the state after the run are bit-exact.
// The chunk kernels (emit + state, any start, R's prefill at 0) must give the same entries,
// pooled values and state as the decode kernel, bit for bit; the reference's prefill and its
// decode steps must agree bit for bit too.
//
// Indexer: the rotated query and head weights bit-exact; scores against the reference
// (TOL_DSV4_IDX_SCORE_*); the GPU's index list equal to the reference's list on the GPU's own
// scores (stable descending order, ties to the lower index, truncated to min(512, n)), and to
// the reference end to end when the scores agree. Ties: duplicated keys and synthetic score sets
// with ties across rank 512/513.
#include <math.h>
#include <stdlib.h>

#include "../dsv4_cmp.h"
#include "../dsv4_ref_cmp.h"
#include "../gpu.h"
#include "../qgen.h"
#include "../test.h"
#include "../tolerances.h"

typedef struct {
    const char *name;
    uint32_t ratio, d;
    bool rot;
} Kind;
static const Kind KINDS[3] = {{"csa", 4, 512, false}, {"hca", 128, 512, false}, {"idx", 4, 128, true}};

typedef struct {
    Arena arena;
    Gpu g;
    QgenPrng rng;
    sx_pipeline *step[3], *emit[3], *state[3], *iq, *isc, *itk;
    float freqs[DSV4C_PAIRS];
    bool ready;
} Ctx;
static Ctx C;

static Err ctx_get(void) {
    if (C.ready) return OK;
    Pack empty = {0};
    TRY(gpu_init(&C.arena, &empty, &C.g));
    qgen_prng_init(&C.rng, 0xd5c0c0);
    char name[64];
    for (int i = 0; i < 3; i++) {
        snprintf(name, sizeof name, "dsv4_cmp_step_%s", KINDS[i].name);
        TRY(gpu_pipeline(&C.g, dsv4_cmp_src, name, &C.step[i]));
        snprintf(name, sizeof name, "dsv4_cmp_emit_%s", KINDS[i].name);
        TRY(gpu_pipeline(&C.g, dsv4_cmp_src, name, &C.emit[i]));
        snprintf(name, sizeof name, "dsv4_cmp_state_%s", KINDS[i].name);
        TRY(gpu_pipeline(&C.g, dsv4_cmp_src, name, &C.state[i]));
    }
    TRY(gpu_pipeline(&C.g, dsv4_cmp_src, "dsv4_idx_q", &C.iq));
    TRY(gpu_pipeline(&C.g, dsv4_cmp_src, "dsv4_idx_score", &C.isc));
    TRY(gpu_pipeline(&C.g, dsv4_cmp_src, "dsv4_idx_topk", &C.itk));
    dsv4_cmp_rope_freqs(C.freqs);
    C.ready = true;
    return OK;
}

static float frand(void) { return qgen_prng_float(&C.rng); }
static sx_arg arg_at(sx_buffer *b, size_t off) { return (sx_arg){b, off, NULL, 0}; }
static uint16_t bf16_bits(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    return (uint16_t)(u >> 16);
}
static float bf16_val(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

// ------------------------------------------------------------------------- compressor -----

typedef struct {
    int ki;
    Dsv4Cmp k;
    size_t N, nent;
    sx_buffer *kv, *sc, *ape, *nw, *cs;
    float *normw; // bf16 values
} Case;

static Err case_make(int ki, size_t N, Case *c) {
    const Kind *K = &KINDS[ki];
    c->ki = ki;
    c->k = dsv4_cmp_shape(K->ratio, K->d, K->rot);
    c->N = N;
    c->nent = N / K->ratio;
    const size_t cd = c->k.cd, d = c->k.d;
    TRY(gpu_new_buffer(&C.g, N * cd * 4, &c->kv));
    TRY(gpu_new_buffer(&C.g, N * cd * 4, &c->sc));
    TRY(gpu_new_buffer(&C.g, K->ratio * cd * 4, &c->ape));
    TRY(gpu_new_buffer(&C.g, d * 2, &c->nw));
    TRY(gpu_new_buffer(&C.g, (c->nent + 1) * DSV4C_PAIRS * 8, &c->cs));
    float *kv = gpu_floats(c->kv), *sc = gpu_floats(c->sc), *ape = gpu_floats(c->ape);
    for (size_t i = 0; i < N * cd; i++) {
        kv[i] = frand() * 4 - 2;
        sc[i] = frand() * 6 - 3;
    }
    for (size_t i = 0; i < K->ratio * cd; i++) ape[i] = frand() * 2 - 1;
    c->normw = arena_alloc(&C.arena, d * 4);
    uint16_t *nw = sx_buffer_contents(c->nw);
    for (size_t i = 0; i < d; i++) {
        c->normw[i] = dsv4c_bf16(0.5f + frand());
        nw[i] = bf16_bits(c->normw[i]);
    }
    float *cs = gpu_floats(c->cs);
    for (size_t e = 0; e <= c->nent; e++) dsv4_cmp_rope_cs(C.freqs, (uint32_t)(e * K->ratio), cs + e * 2 * DSV4C_PAIRS);
    return OK;
}

typedef struct {
    float *pool, *ent, *kvs, *scs; // [nent][d], [nent][d], [rows][cd] x 2
} Out;

static void out_alloc(const Case *c, Out *o) {
    o->pool = arena_alloc(&C.arena, (c->nent + 1) * c->k.d * 4);
    o->ent = arena_alloc(&C.arena, (c->nent + 1) * c->k.d * 4);
    o->kvs = arena_alloc(&C.arena, c->k.rows * c->k.cd * 4);
    o->scs = arena_alloc(&C.arena, c->k.rows * c->k.cd * 4);
}

// Runs the N tokens on the GPU: decode steps (chunks NULL) or the chunk kernels with the chunk
// sizes in chunks (summing to N, 0-terminated).
static Err gpu_run(const Case *c, const size_t *chunks, Out *o) {
    const Dsv4Cmp *k = &c->k;
    const size_t cd = k->cd, d = k->d, ratio = k->ratio;
    sx_buffer *kvs, *scs, *cache, *pool;
    TRY(gpu_new_buffer(&C.g, k->rows * cd * 4, &kvs));
    TRY(gpu_new_buffer(&C.g, k->rows * cd * 4, &scs));
    TRY(gpu_new_buffer(&C.g, (c->nent + 1) * d * 2, &cache));
    TRY(gpu_new_buffer(&C.g, (c->nent + 1) * d * 4, &pool));
    dsv4_cmp_state_init(k, gpu_floats(kvs), gpu_floats(scs));
    sx_cmd *cmd = sx_cmd_begin(C.g.dev);
    if (!cmd) return "CmdFailed";
    Dsv4CmpP *ps = arena_alloc(&C.arena, (c->N + 1) * sizeof(Dsv4CmpP));
    if (!chunks) {
        for (size_t pos = 0; pos < c->N; pos++) {
            const size_t e = pos / ratio;
            ps[pos] = (Dsv4CmpP){(uint32_t)pos, 1, (uint32_t)e, (pos + 1) % ratio == 0, 1, {0}};
            sx_arg a[] = {arg_at(c->kv, pos * cd * 4), arg_at(c->sc, pos * cd * 4), gpu_buf_arg(c->ape), gpu_buf_arg(kvs),
                          gpu_buf_arg(scs), gpu_buf_arg(c->nw), arg_at(c->cs, e * DSV4C_PAIRS * 8), gpu_buf_arg(cache),
                          arg_at(pool, e * d * 4), gpu_bytes_arg(&ps[pos], sizeof(Dsv4CmpP))};
            sx_cmd_dispatch(cmd, C.step[c->ki], a, COUNT(a), 1, 1, d);
        }
    } else {
        size_t p0 = 0, n = 0;
        for (const size_t *ch = chunks; *ch; ch++, n++) {
            const size_t T = *ch, e0 = p0 / ratio, ne = (p0 + T) / ratio - e0;
            ps[n] = (Dsv4CmpP){(uint32_t)p0, (uint32_t)T, (uint32_t)e0, (uint32_t)ne, 1, {0}};
            sx_arg a[] = {arg_at(c->kv, p0 * cd * 4), arg_at(c->sc, p0 * cd * 4), gpu_buf_arg(c->ape), gpu_buf_arg(kvs),
                          gpu_buf_arg(scs), gpu_buf_arg(c->nw), arg_at(c->cs, e0 * DSV4C_PAIRS * 8), gpu_buf_arg(cache),
                          arg_at(pool, e0 * d * 4), gpu_bytes_arg(&ps[n], sizeof(Dsv4CmpP))};
            if (ne) sx_cmd_dispatch(cmd, C.emit[c->ki], a, COUNT(a), ne, 1, d);
            sx_cmd_dispatch(cmd, C.state[c->ki], a, COUNT(a), cd / 256, 1, 256);
            p0 += T;
        }
        if (p0 != c->N) die("gpu_run: chunks do not sum to N\n");
    }
    char err[256];
    if (sx_cmd_finish(cmd, err, sizeof err) != 0) {
        dbg("dsv4_cmp: %s\n", err);
        return "GpuFailed";
    }
    const uint16_t *cb = sx_buffer_contents(cache);
    for (size_t i = 0; i < c->nent * d; i++) o->ent[i] = bf16_val(cb[i]);
    memcpy(o->pool, gpu_floats(pool), c->nent * d * 4);
    memcpy(o->kvs, gpu_floats(kvs), k->rows * cd * 4);
    memcpy(o->scs, gpu_floats(scs), k->rows * cd * 4);
    sx_buffer_release(kvs);
    sx_buffer_release(scs);
    sx_buffer_release(cache);
    sx_buffer_release(pool);
    return OK;
}

static const float *cs_at(const Case *c, size_t e) { return gpu_floats(c->cs) + e * 2 * DSV4C_PAIRS; }

// The reference decode run: pooled values and states after every token (only the last kept).
static void ref_decode(const Case *c, Out *o) {
    const Dsv4Cmp *k = &c->k;
    dsv4_cmp_state_init(k, o->kvs, o->scs);
    const float *kv = gpu_floats(c->kv), *sc = gpu_floats(c->sc), *ape = gpu_floats(c->ape);
    for (size_t pos = 0; pos < c->N; pos++) {
        float pooled[512];
        if (dsv4_cmp_ref_step(k, o->kvs, o->scs, kv + pos * k->cd, sc + pos * k->cd, ape, (uint32_t)pos, pooled)) {
            const size_t e = pos / k->ratio;
            memcpy(o->pool + e * k->d, pooled, k->d * 4);
            dsv4_cmp_ref_post(k, pooled, c->normw, cs_at(c, e), o->ent + e * k->d);
        }
    }
}

static size_t ndiff(const float *a, const float *b, size_t n) {
    size_t r = 0;
    for (size_t i = 0; i < n; i++) r += memcmp(&a[i], &b[i], 4) != 0;
    return r;
}

TEST(dsv4_cmp_decode) {
    EXPECT_OK(ctx_get());
    static const size_t NS[3] = {23, 300, 23}; // windows 0-5 (and a partial), HCA windows 0-1
    for (int ki = 0; ki < 3; ki++) {
        Case c;
        EXPECT_OK(case_make(ki, NS[ki], &c));
        Out g, r;
        out_alloc(&c, &g);
        out_alloc(&c, &r);
        EXPECT_OK(gpu_run(&c, NULL, &g));
        ref_decode(&c, &r);
        const size_t d = c.k.d;
        size_t pd = 0, ed = 0;
        for (size_t e = 0; e < c.nent; e++) {
            float ent[512];
            dsv4_cmp_ref_post(&c.k, g.pool + e * d, c.normw, cs_at(&c, e), ent);
            pd += ndiff(g.pool + e * d, r.pool + e * d, d);
            ed += ndiff(ent, g.ent + e * d, d);
        }
        if (pd || ed) dbg("dsv4_cmp_decode %s: pooled values differing %zu, entries from them %zu, end to end %zu (of %zu)\n",
                          KINDS[ki].name, pd, ed, ndiff(g.ent, r.ent, c.nent * d), c.nent * d);
        EXPECT(pd == 0 && ed == 0);
        EXPECT(ndiff(g.ent, r.ent, c.nent * d) == 0);
        EXPECT(ndiff(g.kvs, r.kvs, c.k.rows * c.k.cd) == 0);
        EXPECT(ndiff(g.scs, r.scs, c.k.rows * c.k.cd) == 0);
    }
}

TEST(dsv4_cmp_prefill) {
    EXPECT_OK(ctx_get());
    static const size_t NS[3] = {23, 300, 23};
    static const size_t P4[][8] = {{23}, {3, 20}, {4, 4, 15}, {5, 2, 1, 7, 8}, {1, 22}, {8, 1, 1, 1, 12}, {7, 16}};
    static const size_t P128[][8] = {{300}, {127, 1, 172}, {100, 100, 100}, {128, 172}, {1, 299}, {255, 2, 43}};
    for (int ki = 0; ki < 3; ki++) {
        Case c;
        EXPECT_OK(case_make(ki, NS[ki], &c));
        const Dsv4Cmp *k = &c.k;
        const size_t d = k->d, sn = k->rows * k->cd;
        Out dec;
        out_alloc(&c, &dec);
        EXPECT_OK(gpu_run(&c, NULL, &dec));
        const size_t np = c.k.ratio == 4 ? COUNT(P4) : COUNT(P128);
        for (size_t p = 0; p < np; p++) {
            const size_t *ch = c.k.ratio == 4 ? P4[p] : P128[p];
            Out g;
            out_alloc(&c, &g);
            EXPECT_OK(gpu_run(&c, ch, &g));
            if (ndiff(g.pool, dec.pool, c.nent * d) || ndiff(g.ent, dec.ent, c.nent * d) || ndiff(g.kvs, dec.kvs, sn) ||
                ndiff(g.scs, dec.scs, sn)) {
                dbg("dsv4_cmp_prefill %s: chunks %zu, %zu, ... differ from decode (pool %zu, entries %zu, state %zu + %zu)\n",
                    KINDS[ki].name, ch[0], ch[1], ndiff(g.pool, dec.pool, c.nent * d), ndiff(g.ent, dec.ent, c.nent * d),
                    ndiff(g.kvs, dec.kvs, sn), ndiff(g.scs, dec.scs, sn));
                EXPECT(0);
            }
        }
        // the reference: R's prefill of T0 tokens then decode steps, against decode from 0
        Out rd;
        out_alloc(&c, &rd);
        ref_decode(&c, &rd);
        const float *kv = gpu_floats(c.kv), *sc = gpu_floats(c.sc), *ape = gpu_floats(c.ape);
        for (size_t T0 = 1; T0 <= c.N; T0 += (c.k.ratio == 4 ? 1 : 37)) {
            float *kvs = malloc(sn * 4), *scs = malloc(sn * 4), *pooled = malloc((c.nent + 1) * d * 4);
            EXPECT(kvs && scs && pooled);
            dsv4_cmp_state_init(k, kvs, scs);
            size_t n = dsv4_cmp_ref_prefill(k, kvs, scs, kv, sc, ape, T0, pooled);
            EXPECT_EQ(n, T0 / k->ratio);
            for (size_t pos = T0; pos < c.N; pos++)
                if (dsv4_cmp_ref_step(k, kvs, scs, kv + pos * k->cd, sc + pos * k->cd, ape, (uint32_t)pos, pooled + n * d)) n++;
            EXPECT_EQ(n, c.nent);
            EXPECT(ndiff(pooled, rd.pool, c.nent * d) == 0);
            free(kvs);
            free(scs);
            free(pooled);
        }
    }
}

// The softmax exp shared by the reference and the kernel, against exp in double.
TEST(dsv4_cmp_exp) {
    double worst = 0;
    for (int i = 0; i <= 2000000; i++) {
        const float x = -87.0f * (float)i / 2000000.0f;
        const double e = exp((double)x), g = dsv4c_exp(x);
        const double ulp = ldexp(1.0, ilogb(e) - 23);
        worst = fmax(worst, fabs(g - e) / ulp);
    }
    if (worst > 2.0) dbg("dsv4c_exp: %.3f ulp\n", worst); // measured 1.18 ulp, 2026-11-20
    EXPECT(worst <= 2.0);
    EXPECT(dsv4c_exp(0.0f) == 1.0f);
}

// ---------------------------------------------------------------------------- indexer -----

// A key as the indexer's compressor makes them: bf16 values, Hadamard, FP4.
static void rand_key(float *x) {
    for (size_t i = 0; i < DSV4C_IDX_DIM; i++) x[i] = dsv4c_bf16((frand() * 2 - 1) * (i % 7 == 0 ? 4.0f : 1.0f));
    dsv4c_hadamard(x, DSV4C_IDX_DIM);
    dsv4c_fp4_qdq(x, DSV4C_IDX_DIM);
}

static Err run_topk(sx_buffer *scores, size_t n_vis, size_t n_total, int32_t offset, int32_t *out) {
    sx_buffer *ib;
    TRY(gpu_new_buffer(&C.g, DSV4C_TOPK * 4, &ib));
    Dsv4IdxP P = {(uint32_t)n_vis, (uint32_t)n_total, offset, 0};
    sx_arg a[] = {gpu_buf_arg(scores), gpu_buf_arg(ib), gpu_bytes_arg(&P, sizeof P)};
    char err[256];
    if (sx_dispatch_sync(C.g.dev, C.itk, a, COUNT(a), 1, 1024, err, sizeof err) != 0) return "GpuFailed";
    memcpy(out, sx_buffer_contents(ib), DSV4C_TOPK * 4);
    sx_buffer_release(ib);
    return OK;
}

// The GPU list against the reference list on the same scores.
static bool topk_same(sx_buffer *scores, size_t n_vis, size_t n_total, int32_t offset) {
    int32_t got[DSV4C_TOPK], want[DSV4C_TOPK];
    if (run_topk(scores, n_vis, n_total, offset, got)) return false;
    const size_t k = dsv4_idx_ref_topk(gpu_floats(scores), n_vis, n_total, offset, want);
    for (size_t r = 0; r < k; r++)
        if (got[r] != want[r]) {
            dbg("topk n_vis %zu n_total %zu: rank %zu got %d want %d\n", n_vis, n_total, r, got[r], want[r]);
            return false;
        }
    return true;
}

TEST(dsv4_idx_scores) {
    EXPECT_OK(ctx_get());
    const size_t NMAX = 3000;
    sx_buffer *qb, *wpb, *csb, *qxb, *wb, *kb, *sb;
    EXPECT_OK(gpu_new_buffer(&C.g, DSV4C_IDX_HEADS * DSV4C_IDX_DIM * 4, &qb));
    EXPECT_OK(gpu_new_buffer(&C.g, DSV4C_IDX_HEADS * 4, &wpb));
    EXPECT_OK(gpu_new_buffer(&C.g, DSV4C_PAIRS * 8, &csb));
    EXPECT_OK(gpu_new_buffer(&C.g, DSV4C_IDX_HEADS * DSV4C_IDX_DIM * 4, &qxb));
    EXPECT_OK(gpu_new_buffer(&C.g, DSV4C_IDX_HEADS * 4, &wb));
    EXPECT_OK(gpu_new_buffer(&C.g, NMAX * DSV4C_IDX_DIM * 2, &kb));
    EXPECT_OK(gpu_new_buffer(&C.g, NMAX * 4, &sb));
    float *keys = malloc(NMAX * DSV4C_IDX_DIM * 4), *ref = malloc(NMAX * 4);
    EXPECT(keys && ref);
    // keys: 1000 distinct, then copies (ties of equal keys) of the first 37
    for (size_t s = 0; s < NMAX; s++) {
        if (s < 1000) rand_key(keys + s * DSV4C_IDX_DIM);
        else memcpy(keys + s * DSV4C_IDX_DIM, keys + (s % 37) * DSV4C_IDX_DIM, DSV4C_IDX_DIM * 4);
    }
    uint16_t *kbits = sx_buffer_contents(kb);
    for (size_t i = 0; i < NMAX * DSV4C_IDX_DIM; i++) kbits[i] = bf16_bits(keys[i]);
    static const uint32_t POS[] = {11, 4099, 199999};
    for (size_t pi = 0; pi < COUNT(POS); pi++) {
        float *q = gpu_floats(qb), *wp = gpu_floats(wpb);
        for (size_t i = 0; i < DSV4C_IDX_HEADS * DSV4C_IDX_DIM; i++) q[i] = dsv4c_bf16((frand() * 2 - 1) * 3);
        for (size_t h = 0; h < DSV4C_IDX_HEADS; h++) wp[h] = dsv4c_bf16((frand() * 2 - 0.6f) * 8);
        dsv4_cmp_rope_cs(C.freqs, POS[pi], gpu_floats(csb));
        sx_arg a[] = {gpu_buf_arg(qb), gpu_buf_arg(wpb), gpu_buf_arg(csb), gpu_buf_arg(qxb), gpu_buf_arg(wb)};
        char err[256];
        EXPECT(sx_dispatch_sync(C.g.dev, C.iq, a, COUNT(a), DSV4C_IDX_HEADS, DSV4C_IDX_DIM, err, sizeof err) == 0);
        float qx[DSV4C_IDX_HEADS * DSV4C_IDX_DIM], w[DSV4C_IDX_HEADS];
        dsv4_idx_ref_q(q, gpu_floats(csb), qx);
        dsv4_idx_ref_w(wp, w);
        EXPECT(ndiff(qx, gpu_floats(qxb), COUNT(qx)) == 0);
        EXPECT(ndiff(w, gpu_floats(wb), COUNT(w)) == 0);
        static const size_t NV[] = {1, 100, 511, 512, 513, 700, 3000};
        for (size_t ni = 0; ni < COUNT(NV); ni++) {
            const size_t n = NV[ni];
            Dsv4IdxP P = {(uint32_t)n, (uint32_t)n, 128, 0};
            sx_arg b[] = {gpu_buf_arg(kb), gpu_buf_arg(qxb), gpu_buf_arg(wb), gpu_buf_arg(sb), gpu_bytes_arg(&P, sizeof P)};
            EXPECT(sx_dispatch_sync(C.g.dev, C.isc, b, COUNT(b), (n + 31) / 32, 128, err, sizeof err) == 0);
            dsv4_idx_ref_scores(qx, w, keys, n, ref);
            const float *got = gpu_floats(sb);
            float m = 1e-30f, e = 0;
            size_t nd = 0;
            for (size_t s = 0; s < n; s++) {
                m = fmaxf(m, fabsf(ref[s]));
                e = fmaxf(e, fabsf(got[s] - ref[s]));
                nd += got[s] != ref[s];
            }
            if (ni == COUNT(NV) - 1) dbg("dsv4_idx_scores pos %u: score rel %.2e, %zu of %zu differ\n", POS[pi], e / m, nd, n);
            EXPECT(e / m <= TOL_DSV4_IDX_SCORE_REL);
            EXPECT((double)nd <= TOL_DSV4_IDX_SCORE_FRAC * (double)n);
            // decode (n_vis = n_total) with the attention's window offset, and prefill-style masking
            EXPECT(topk_same(sb, n, n, 128));
            if (n > 1) EXPECT(topk_same(sb, n - 1, n, 0));
            if (n > 300) EXPECT(topk_same(sb, 300, n, 7));
            if (nd == 0) { // end to end
                int32_t g1[DSV4C_TOPK], r1[DSV4C_TOPK];
                EXPECT_OK(run_topk(sb, n, n, 128, g1));
                const size_t k = dsv4_idx_ref_topk(ref, n, n, 128, r1);
                EXPECT(memcmp(g1, r1, k * 4) == 0);
            }
        }
    }
    free(keys);
    free(ref);
}

TEST(dsv4_idx_topk_ties) {
    EXPECT_OK(ctx_get());
    const size_t NMAX = 70000;
    sx_buffer *sb;
    EXPECT_OK(gpu_new_buffer(&C.g, NMAX * 4, &sb));
    float *s = gpu_floats(sb);
    // 400 entries at 3, 300 at 2 (rank 512 falls inside them), the rest at 1, scattered
    for (size_t i = 0; i < 1000; i++) {
        const uint32_t r = (uint32_t)(qgen_prng_next(&C.rng) % 1000);
        s[i] = r < 400 ? 3.0f : r < 700 ? 2.0f : 1.0f;
    }
    EXPECT(topk_same(sb, 1000, 1000, 0));
    // every score equal: the first 512 indices in order
    for (size_t i = 0; i < 2000; i++) s[i] = 0.5f;
    EXPECT(topk_same(sb, 2000, 2000, 0));
    int32_t got[DSV4C_TOPK];
    EXPECT_OK(run_topk(sb, 2000, 2000, 10, got));
    for (size_t r = 0; r < DSV4C_TOPK; r++) EXPECT_EQ(got[r], (int32_t)r + 10);
    // exactly 511 above a tie of many at rank 512/513; -0 and +0 tie; negatives
    for (size_t i = 0; i < 1500; i++) s[i] = (i % 3 == 0 && i / 3 < 511) ? 1.0f + (float)(i % 5) : (i % 2 ? -0.0f : 0.0f);
    EXPECT(topk_same(sb, 1500, 1500, 0));
    for (size_t i = 0; i < 1500; i++) s[i] = i % 4 == 0 ? -1.5f : i % 4 == 1 ? -0.25f : i % 4 == 2 ? 0.0f : -0.0f;
    EXPECT(topk_same(sb, 1500, 1500, 0));
    // few distinct bf16 values over a long cache, with the threshold in the low byte
    for (size_t i = 0; i < NMAX; i++) s[i] = dsv4c_bf16(1.0f + (float)(qgen_prng_next(&C.rng) % 9) * 0.0078125f);
    EXPECT(topk_same(sb, NMAX, NMAX, 128));
    EXPECT(topk_same(sb, 513, 513, 0));
    EXPECT(topk_same(sb, 512, 512, 0));
    EXPECT(topk_same(sb, 512, 600, 0));
    EXPECT(topk_same(sb, 37, 40, 3));
    // random bf16 scores, both signs
    for (size_t i = 0; i < NMAX; i++) s[i] = dsv4c_bf16((frand() * 2 - 1) * 50);
    EXPECT(topk_same(sb, NMAX, NMAX, 0));
    EXPECT(topk_same(sb, 50000, NMAX, 5));
    sx_buffer_release(sb);
}
