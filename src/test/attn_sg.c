// Prompt attention without tensor ops (src/kernels/attn_sg.metal) against the f64 reference in
// src/attn_sg.c, on random data (no pack needed): the 27B causal kernels, direct and split-K
// (merged by the engine's attn_merge), both widths; the Flash-Next QSA kernel across the
// dense / list boundary. Ragged chunk lengths, chunks starting mid-cache, NaN past the cache.
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../attn_sg.h"
#include "../qgen.h"
#include "../test.h"
#include "../tolerances.h"

#define NQ ATTN_SG_NQ
#define HD ATTN_SG_HD

// attn_merge (decode.metal) with the helpers it uses
static const char merge_src[] = {
#embed "../kernels/qmatvec.metal"
    , '\n',
#embed "../kernels/decode.metal"
    , 0};

typedef struct {
    uint32_t pos0, budget_blocks, ratio, head_dim, kv_stride, q_per_kv, gate_stride, nq, nsplit, maxsel;
    float scale;
} AttnSP;

static float urand(QgenCtx *x, float lo, float hi) { return lo + (hi - lo) * qgen_prng_float(&x->rng); }

// Largest |a - b| of a (token, head) row relative to the largest |b| of that row.
static double row_rel(const float *a, const float *b) {
    double mx = 0, md = 0;
    for (size_t d = 0; d < HD; d++) {
        mx = fmax(mx, fabs((double)b[d]));
        const double e = fabs((double)a[d] - (double)b[d]);
        if (!(e <= md)) md = e; // NaN propagates
    }
    return mx > 0 ? md / mx : md;
}

typedef struct {
    size_t pos0, T;
    float qa;      // queries uniform in [-qa, qa): sets how peaked the softmax is
    size_t nsplit; // causal: 0 the engine's choice (attn_sg_nsplit), else forced
} Case;

static Err run3(QgenCtx *x, sx_pipeline *p, const sx_arg *a, size_t na, size_t gx, size_t gy, size_t gz, size_t th) {
    char err[256];
    sx_cmd *c = sx_cmd_begin(x->g.dev);
    if (!c) return "DispatchFailed";
    sx_cmd_dispatch3(c, p, a, na, gx, gy, gz, th);
    if (sx_cmd_finish(c, err, sizeof err) != 0) {
        dbg("attn_sg: %s\n", err);
        return "DispatchFailed";
    }
    return OK;
}

// Random q, gate, K, V for a case (rows of K / V past the chunk hold NaN); tokens to check.
typedef struct {
    sx_buffer *qb, *qgb, *kvb, *ob, *partb, *selb, *cntb;
    float *q, *qg, *o;
    _Float16 *K, *V;
    size_t cap, toks[48], nt;
} Data;

static Err data_make(QgenCtx *x, const Case *c, size_t nkv, size_t maxsel, Data *d) {
    const size_t n = c->pos0 + c->T;
    d->cap = n + 200;
    TRY(gpu_new_buffer(&x->g, c->T * NQ * HD * 4, &d->qb));
    TRY(gpu_new_buffer(&x->g, c->T * NQ * 2 * HD * 4, &d->qgb));
    TRY(gpu_new_buffer(&x->g, 2 * d->cap * nkv * HD * 2, &d->kvb));
    TRY(gpu_new_buffer(&x->g, c->T * NQ * HD * 4, &d->ob));
    TRY(gpu_new_buffer(&x->g, (size_t)512 * NQ * 8 * (HD + 2) * 4, &d->partb));
    TRY(gpu_new_buffer(&x->g, c->T * maxsel * 4 + 4, &d->selb));
    TRY(gpu_new_buffer(&x->g, c->T * 4, &d->cntb));
    d->q = sx_buffer_contents(d->qb);
    d->qg = sx_buffer_contents(d->qgb);
    d->o = sx_buffer_contents(d->ob);
    d->K = sx_buffer_contents(d->kvb);
    d->V = d->K + d->cap * nkv * HD;
    for (size_t i = 0; i < c->T * NQ * HD; i++) d->q[i] = urand(x, -c->qa, c->qa);
    for (size_t i = 0; i < c->T * NQ * 2 * HD; i++) d->qg[i] = urand(x, -3, 3);
    for (size_t i = 0; i < d->cap * nkv * HD; i++) {
        const bool live = i < n * nkv * HD;
        d->K[i] = live ? (_Float16)urand(x, -1, 1) : (_Float16)NAN;
        d->V[i] = live ? (_Float16)urand(x, -1, 1) : (_Float16)NAN;
    }
    for (size_t i = 0; i < c->T * NQ * HD; i++) d->o[i] = NAN;
    // tokens checked: the first, the last two, a stride through the rest
    d->nt = 0;
    const size_t step = MAX((size_t)1, c->T / 40);
    for (size_t t = 0; t < c->T && d->nt < 45; t += step) d->toks[d->nt++] = t;
    d->toks[d->nt++] = c->T - 1;
    if (c->T > 1) d->toks[d->nt++] = c->T - 2;
    return OK;
}

static void data_free(Data *d) {
    sx_buffer *bufs[] = {d->qb, d->qgb, d->kvb, d->ob, d->partb, d->selb, d->cntb};
    for (size_t i = 0; i < COUNT(bufs); i++) sx_buffer_release(bufs[i]);
}

// Worst row error of the checked tokens against the reference; false when an output is not finite.
static bool check(const Data *d, const Case *c, size_t nkv, const uint32_t *sel, const uint32_t *cnt, size_t maxsel,
                  double *worst) {
    bool finite = true;
    for (size_t i = 0; i < c->T * NQ * HD; i++) finite &= isfinite(d->o[i]);
    float ref[NQ * HD];
    *worst = 0;
    for (size_t i = 0; i < d->nt; i++) {
        const size_t t = d->toks[i], n = c->pos0 + t + 1;
        const bool list = cnt && cnt[t] > 0;
        attn_sg_ref(d->q, d->qg, d->K, d->V, nkv, t, list ? sel + t * maxsel : NULL, list ? cnt[t] : n, ref);
        for (size_t h = 0; h < NQ; h++) *worst = fmax(*worst, row_rel(d->o + (t * NQ + h) * HD, ref + h * HD));
    }
    return finite;
}

TEST(attn_sg_causal) {
    static Arena ar;
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&ar, &x));
    static const struct {
        const char *dir, *part;
        size_t nsg;
    } vs[] = {{"attn_sg_27", "attn_sg_27_part", 8}, {"attn_sg_27_s4", "attn_sg_27_s4_part", 4}};
    sx_pipeline *merge;
    EXPECT_OK(qgen_ctx_pipe(&x, merge_src, "attn_merge", &merge));
    // chunk at 0, odd sizes, a cached prefix, long rows, peaked rows, steps one key short of
    // full, blocks that straddle tokens; small chunks over long caches take split-K (forced
    // counts too, with empty splits for the early rows)
    static const Case cases[] = {{0, 24, 1.0f, 0},    {0, 37, 4.0f, 0},    {5, 9, 2.0f, 0},      {100, 300, 4.0f, 0},
                                 {3000, 517, 4.0f, 0}, {127, 129, 12.0f, 0}, {9000, 200, 24.0f, 0}, {0, 1024, 8.0f, 0},
                                 {30, 40, 4.0f, 0},    {502, 100, 8.0f, 0},  {20000, 20, 4.0f, 0},  {4000, 9, 8.0f, 0},
                                 {0, 50, 4.0f, 3},     {700, 33, 4.0f, 8},   {63, 2, 4.0f, 2}};
    // gpu_pipeline refuses sources naming the tensor-op header when the accelerators are off
    EXPECT(strstr(attn_sg_src, "<metal_tensor>") == NULL);
    const size_t nkv = 4, qh = NQ / nkv;
    double worst = 0;
    for (size_t v = 0; v < COUNT(vs); v++) {
        sx_pipeline *dir, *part;
        EXPECT_OK(qgen_ctx_pipe(&x, attn_sg_src, vs[v].dir, &dir));
        EXPECT_OK(qgen_ctx_pipe(&x, attn_sg_src, vs[v].part, &part));
        const size_t th = vs[v].nsg * 32;
        if (sx_pipeline_max_threads(dir) < th || sx_pipeline_max_threads(part) < th) {
            dbg("attn_sg_causal: %s allows %u / %u threads, needs %zu: skipped\n", vs[v].dir, sx_pipeline_max_threads(dir),
                sx_pipeline_max_threads(part), th);
            EXPECT(v > 0); // the widest variant must run on the machine the tests run on
            continue;
        }
        for (size_t ci = 0; ci < COUNT(cases); ci++) {
            const Case c = cases[ci];
            Data d;
            EXPECT_OK(data_make(&x, &c, nkv, 1, &d));
            const size_t nb = attn_sg_blocks(c.T, qh, vs[v].nsg);
            const size_t ns = c.nsplit ? c.nsplit : attn_sg_nsplit(c.T, qh, nkv, vs[v].nsg, c.pos0 + c.T, 512, 8);
            const AttnSgP p = {(uint32_t)c.pos0, (uint32_t)c.T, (uint32_t)ns, 0, 1, 0};
            sx_arg a[] = {gpu_buf_arg(d.qb), {d.kvb, 0, NULL, 0}, {d.kvb, d.cap * nkv * HD * 2, NULL, 0}, {d.qgb, HD * 4, NULL, 0},
                          gpu_buf_arg(ns > 1 ? d.partb : d.ob), gpu_bytes_arg(&p, sizeof p), gpu_buf_arg(d.selb), gpu_buf_arg(d.cntb)};
            EXPECT_OK(run3(&x, ns > 1 ? part : dir, a, COUNT(a), nkv, nb, ns, th));
            if (ns > 1) {
                const AttnSP mp = {(uint32_t)c.pos0, 0, 1, HD, (uint32_t)(nkv * HD), (uint32_t)qh, 2 * HD, NQ, (uint32_t)ns, 0,
                                   1.0f / 16.0f};
                sx_arg b[] = {gpu_buf_arg(d.partb), {d.qgb, HD * 4, NULL, 0}, gpu_buf_arg(d.ob), gpu_bytes_arg(&mp, sizeof mp)};
                EXPECT_OK(run3(&x, merge, b, COUNT(b), NQ, c.T, 1, 256));
            }
            double cw;
            const bool finite = check(&d, &c, nkv, NULL, NULL, 0, &cw);
            dbg("attn_sg_causal: pos0 %5zu T %4zu qa %4.1f %-14s splits %zu: rel %.3g\n", c.pos0, c.T, (double)c.qa,
                vs[v].dir, ns, cw);
            data_free(&d);
            EXPECT(finite);
            EXPECT(cw <= TOL_ATTN27_REL);
            worst = fmax(worst, cw);
        }
    }
    dbg("attn_sg_causal: worst %.3g\n", worst);
}

TEST(attn_sg_qsa) {
    static Arena ar;
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&ar, &x));
    sx_pipeline *fn;
    EXPECT_OK(qgen_ctx_pipe(&x, attn_sg_src, "attn_sg_fn", &fn));
    EXPECT(sx_pipeline_max_threads(fn) >= ATTN_SG_FN_THREADS);
    enum { BUDGET = 512, RATIO = 4, MAXSEL = BUDGET * RATIO + RATIO - 1 + 13 };
    // dense rows, the boundary at 2,052 visible tokens inside a chunk, long caches with lists
    // (tails of 0 to 3 tokens), peaked rows
    static const Case cases[] = {{0, 24, 1.0f, 0},     {0, 300, 4.0f, 0},    {1900, 300, 4.0f, 0},
                                 {5000, 64, 8.0f, 0},  {20001, 33, 4.0f, 0}, {9000, 7, 24.0f, 0},
                                 {2047, 9, 12.0f, 0}};
    const size_t nkv = 2;
    double worst = 0;
    for (size_t ci = 0; ci < COUNT(cases); ci++) {
        const Case c = cases[ci];
        Data d;
        EXPECT_OK(data_make(&x, &c, nkv, MAXSEL, &d));
        uint32_t *sel = sx_buffer_contents(d.selb), *cnt = sx_buffer_contents(d.cntb);
        attn_sg_lists(&x.rng, c.pos0, c.T, BUDGET, RATIO, MAXSEL, sel, cnt);
        const AttnSgP p = {(uint32_t)c.pos0, (uint32_t)c.T, 1, BUDGET, RATIO, MAXSEL};
        sx_arg a[] = {gpu_buf_arg(d.qb), {d.kvb, 0, NULL, 0}, {d.kvb, d.cap * nkv * HD * 2, NULL, 0}, {d.qgb, HD * 4, NULL, 0},
                      gpu_buf_arg(d.ob), gpu_bytes_arg(&p, sizeof p), gpu_buf_arg(d.selb), gpu_buf_arg(d.cntb)};
        EXPECT_OK(run3(&x, fn, a, COUNT(a), nkv, c.T, 1, ATTN_SG_FN_THREADS));
        double cw;
        const bool finite = check(&d, &c, nkv, sel, cnt, MAXSEL, &cw);
        dbg("attn_sg_qsa: pos0 %5zu T %4zu qa %4.1f: rel %.3g\n", c.pos0, c.T, (double)c.qa, cw);
        data_free(&d);
        EXPECT(finite);
        EXPECT(cw <= TOL_ATTN27_REL);
        worst = fmax(worst, cw);
    }
    dbg("attn_sg_qsa: worst %.3g\n", worst);
}
