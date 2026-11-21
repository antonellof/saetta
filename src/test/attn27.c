// Qwen3.8-27B prompt attention on the accelerators (src/kernels/dense_attn.metal) against the
// scalar reference in src/attn27.c, on random data (no pack needed).
#include <math.h>
#include <stdlib.h>

#include "../attn27.h"
#include "../qgen.h"
#include "../test.h"
#include "../tolerances.h"

#define NQ ATTN27_NQ
#define NKV ATTN27_NKV
#define HD ATTN27_HD

static float urand(QgenCtx *x, float lo, float hi) { return lo + (hi - lo) * qgen_prng_float(&x->rng); }

typedef struct {
    size_t pos0, T;
    float qa; // queries uniform in [-qa, qa): sets how peaked the softmax is
} Case;

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

TEST(attn27_flash) {
    static Arena ar;
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&ar, &x));
    sx_pipeline *q6h, *fl[2];
    EXPECT_OK(qgen_ctx_pipe(&x, attn27_src, "attn_fl_q6h", &q6h));
    EXPECT_OK(qgen_ctx_pipe(&x, attn27_src, "attn_fl_6_q8", &fl[0]));
    EXPECT_OK(qgen_ctx_pipe(&x, attn27_src, "attn_fl_6_q16", &fl[1]));
    static const size_t qbs[2] = {8, 16};
    // chunk at 0 (the golden prompt's 24 tokens), odd sizes, a cached prefix, long rows, peaked rows
    static const Case cases[] = {{0, 24, 1.0f},     {0, 37, 4.0f},    {5, 9, 2.0f},       {100, 300, 4.0f},
                                 {3000, 517, 4.0f}, {127, 129, 12.0f}, {9000, 200, 24.0f}, {0, 1024, 8.0f}};
    double worst = 0, worst_exact = 0;
    for (size_t ci = 0; ci < COUNT(cases); ci++) {
        const Case c = cases[ci];
        const size_t n = c.pos0 + c.T, cap = n + 200;
        sx_buffer *qb, *qgb, *kvb, *qhb, *ob;
        EXPECT_OK(gpu_new_buffer(&x.g, c.T * NQ * HD * 4, &qb));
        EXPECT_OK(gpu_new_buffer(&x.g, c.T * NQ * 2 * HD * 4, &qgb));
        EXPECT_OK(gpu_new_buffer(&x.g, 2 * cap * NKV * HD * 2, &kvb));
        EXPECT_OK(gpu_new_buffer(&x.g, (size_t)NKV * attn27_tq(c.T, 16) * (NQ / NKV) * HD * 2, &qhb));
        EXPECT_OK(gpu_new_buffer(&x.g, c.T * NQ * HD * 4, &ob));
        float *q = sx_buffer_contents(qb), *qg = sx_buffer_contents(qgb);
        _Float16 *K = sx_buffer_contents(kvb), *V = K + cap * NKV * HD;
        for (size_t i = 0; i < c.T * NQ * HD; i++) q[i] = urand(&x, -c.qa, c.qa);
        for (size_t i = 0; i < c.T * NQ * 2 * HD; i++) qg[i] = urand(&x, -3, 3);
        // rows past the chunk hold NaN: the kernel must not read them into a visible result
        for (size_t i = 0; i < cap * NKV * HD; i++) {
            const bool live = i < n * NKV * HD;
            K[i] = live ? (_Float16)urand(&x, -1, 1) : (_Float16)NAN;
            V[i] = live ? (_Float16)urand(&x, -1, 1) : (_Float16)NAN;
        }
        // tokens checked: the first, the last, a stride through the rest
        size_t toks[40], nt = 0;
        const size_t step = MAX((size_t)1, c.T / 30);
        for (size_t t = 0; t < c.T && nt < 38; t += step) toks[nt++] = t;
        toks[nt++] = c.T - 1;
        if (c.T > 1) toks[nt++] = c.T - 2;
        float *ref = malloc(nt * NQ * HD * 4), *ex = malloc(nt * NQ * HD * 4);
        for (size_t i = 0; i < nt; i++) {
            attn27_ref(q, qg, K, V, c.pos0, toks[i], true, ref + i * NQ * HD);
            attn27_ref(q, qg, K, V, c.pos0, toks[i], false, ex + i * NQ * HD);
        }
        for (int v = 0; v < 2; v++) {
            const Attn27P p = {(uint32_t)c.pos0, (uint32_t)c.T, attn27_tq(c.T, qbs[v])};
            float *o = sx_buffer_contents(ob);
            for (size_t i = 0; i < c.T * NQ * HD; i++) o[i] = NAN;
            const size_t n4 = (size_t)NKV * p.tq * (NQ / NKV) * HD / 4;
            sx_arg a[] = {gpu_buf_arg(qb), gpu_buf_arg(qhb), gpu_bytes_arg(&p, sizeof p)};
            EXPECT_OK(qgen_ctx_run(&x, q6h, a, COUNT(a), (n4 + 255) / 256, 1, 256));
            sx_arg b[] = {gpu_buf_arg(qhb), {kvb, 0, NULL, 0}, {kvb, cap * NKV * HD * 2, NULL, 0},
                          {qgb, HD * 4, NULL, 0}, gpu_buf_arg(ob), gpu_bytes_arg(&p, sizeof p)};
            EXPECT_OK(qgen_ctx_run(&x, fl[v], b, COUNT(b), NKV, p.tq / qbs[v], qbs[v] * 32));
            bool finite = true;
            for (size_t i = 0; i < c.T * NQ * HD; i++) finite &= isfinite(o[i]);
            if (!finite) dbg("attn27_flash: case %zu q%zu: non-finite output\n", ci, qbs[v]);
            EXPECT(finite);
            double cw = 0, cx = 0;
            for (size_t i = 0; i < nt; i++)
                for (size_t h = 0; h < NQ; h++) {
                    const float *gv = o + (toks[i] * NQ + h) * HD;
                    cw = fmax(cw, row_rel(gv, ref + (i * NQ + h) * HD));
                    cx = fmax(cx, row_rel(gv, ex + (i * NQ + h) * HD));
                }
            dbg("attn27_flash: pos0 %5zu T %4zu qa %4.1f q%-2zu: rel to the mirrored reference %.3g, to the exact %.3g\n",
                c.pos0, c.T, (double)c.qa, qbs[v], cw, cx);
            worst = fmax(worst, cw);
            worst_exact = fmax(worst_exact, cx);
            EXPECT(cw <= TOL_ATTN27_REL);
            EXPECT(cx <= TOL_ATTN27_EXACT_REL);
        }
        free(ref);
        free(ex);
        sx_buffer *bufs[] = {qb, qgb, kvb, qhb, ob};
        for (size_t i = 0; i < COUNT(bufs); i++) sx_buffer_release(bufs[i]);
    }
    dbg("attn27_flash: worst %.3g (mirrored), %.3g (exact)\n", worst, worst_exact);
}
