// Inkling 2 Mini kernels (src/kernels/inkling_*.metal) against the scalar references in
// src/inkling_ref.c, on random data (no pack needed).
#include <math.h>
#include <stdlib.h>

#include "../inkling_ref.h"
#include "../qgen.h"
#include "../test.h"
#include "../tolerances.h"

static const char ink_src[] = {
#embed "../kernels/inkling_ops.metal"
    , '\n',
#embed "../kernels/inkling_attn.metal"
    , '\n',
#embed "../kernels/inkling_moe.metal"
    , 0};
static const char ink_mpp_src[] = {
#embed "../kernels/inkling_mpp.metal"
    , 0};

#define BYTES(p) gpu_bytes_arg((p), sizeof(*(p)))
static sx_arg BA(sx_buffer *b) { return gpu_buf_arg(b); }
static sx_arg BO(sx_buffer *b, size_t off) { return (sx_arg){b, off, NULL, 0}; }

static float ur(QgenCtx *x, float lo, float hi) { return lo + (hi - lo) * qgen_prng_float(&x->rng); }
static void fill(QgenCtx *x, float *p, size_t n, float lo, float hi) {
    for (size_t i = 0; i < n; i++) p[i] = ur(x, lo, hi);
}
// a GPU buffer holding n floats (copied from src when given)
static Err fbuf(QgenCtx *x, const float *src, size_t n, sx_buffer **b) {
    TRY(gpu_new_buffer(&x->g, MAX(n * 4, (size_t)16), b));
    if (src) memcpy(sx_buffer_contents(*b), src, n * 4);
    return OK;
}
static Err run_src(QgenCtx *x, const char *src, const char *name, const sx_arg *args, size_t nargs, size_t gx, size_t gy, size_t gz,
                   size_t th) {
    sx_pipeline *p;
    TRY(qgen_ctx_pipe(x, src, name, &p));
    sx_cmd *c = sx_cmd_begin(x->g.dev);
    sx_cmd_dispatch3(c, p, args, nargs, gx, gy, gz, th);
    char err[512];
    if (sx_cmd_finish(c, err, sizeof err)) {
        dbg("%s: %s\n", name, err);
        return "DispatchFailed";
    }
    return OK;
}
static Err run(QgenCtx *x, const char *name, const sx_arg *args, size_t nargs, size_t gx, size_t gy, size_t gz, size_t th) {
    return run_src(x, ink_src, name, args, nargs, gx, gy, gz, th);
}
#define RUN(x, name, gx, gy, gz, th, ...)                                     \
    do {                                                                     \
        sx_arg a_[] = {__VA_ARGS__};                                         \
        EXPECT_OK(run((x), (name), a_, COUNT(a_), (gx), (gy), (gz), (th)));  \
    } while (0)

static float rel(const float *a, const float *b, size_t n) {
    float m = 0, s = 1e-30f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (!(d <= m)) m = d;
        s = fmaxf(s, fabsf(b[i]));
    }
    return m / s;
}
#define EXPECT_CLOSE(what, a, b, n)                                                                      \
    do {                                                                                                 \
        float e_ = rel((a), (b), (n));                                                                   \
        if (!(e_ <= TOL_INK_KERNEL_REL)) {                                                               \
            dbg("%s: rel err %.3g (bound %.3g)\n", what, (double)e_, (double)TOL_INK_KERNEL_REL);       \
            *failed_ = 1;                                                                                \
            return;                                                                                      \
        }                                                                                                \
    } while (0)

enum { H = 2304, NH = 24, NKV = 8, HD = 128, DREL = 16, R = 64 };

TEST(ink_mix) {
    Arena a = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&a, &x));
    const size_t n = 4, nq = 3, T = 3, PH = T * H;
    float *st = ARENA_ARRAY(&a, float, n * PH), *pt = ARENA_ARRAY(&a, float, PH), *q = ARENA_ARRAY(&a, float, nq * H);
    for (size_t j = 0; j < n; j++) fill(&x, st + j * PH, PH, -(float)(j + 1) * 3, (float)(j + 1) * 3);
    fill(&x, pt, PH, -50, 50);
    fill(&x, q, nq * H, -0.3f, 0.3f);
    sx_buffer *bs, *bp, *bq, *bo;
    EXPECT_OK(fbuf(&x, st, n * PH, &bs));
    EXPECT_OK(fbuf(&x, pt, PH, &bp));
    EXPECT_OK(fbuf(&x, q, nq * H, &bq));
    EXPECT_OK(fbuf(&x, NULL, T * 4 * H, &bo));
    struct { uint32_t n, nq, h, sstride, pstride, ostride; float eps; } p = {n, nq, H, PH, H, 4 * H, 1e-6f};
    RUN(&x, "ink_mix", T, 1, 1, 1024, BA(bs), BA(bp), BA(bq), BA(bo), BYTES(&p));
    float *ref = ARENA_ARRAY(&a, float, nq * H);
    for (size_t t = 0; t < T; t++) {
        ink_ref_mix(st + t * H, PH, n, pt + t * H, q, nq, H, 1e-6f, ref);
        EXPECT_CLOSE("mix", gpu_floats(bo) + t * 4 * H, ref, nq * H);
    }
}

TEST(ink_rmsnorm_msg_gate) {
    Arena a = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&a, &x));
    const size_t T = 2;
    float *xs = ARENA_ARRAY(&a, float, T * H), *ms = ARENA_ARRAY(&a, float, T * 4 * H), *w = ARENA_ARRAY(&a, float, H);
    float *F = ARENA_ARRAY(&a, float, H * R), *G = ARENA_ARRAY(&a, float, H * R);
    uint16_t *D = ARENA_ARRAY(&a, uint16_t, R * 2 * H);
    fill(&x, xs, T * H, -20, 20);
    fill(&x, ms, T * 4 * H, -2, 2);
    fill(&x, w, H, 0.2f, 1.5f);
    fill(&x, F, H * R, -0.2f, 0.2f);
    fill(&x, G, H * R, -0.2f, 0.2f);
    for (size_t i = 0; i < R * 2 * H; i++) {
        float f = ur(&x, -0.05f, 0.05f);
        uint32_t u;
        memcpy(&u, &f, 4);
        D[i] = (uint16_t)(u >> 16);
    }
    sx_buffer *bx, *bm, *bw, *bF, *bG, *bD, *bd, *bo;
    EXPECT_OK(fbuf(&x, xs, T * H, &bx));
    EXPECT_OK(fbuf(&x, ms, T * 4 * H, &bm));
    EXPECT_OK(fbuf(&x, w, H, &bw));
    EXPECT_OK(fbuf(&x, F, H * R, &bF));
    EXPECT_OK(fbuf(&x, G, H * R, &bG));
    EXPECT_OK(gpu_new_buffer(&x.g, R * 2 * H * 2, &bD));
    memcpy(sx_buffer_contents(bD), D, R * 2 * H * 2);
    EXPECT_OK(fbuf(&x, NULL, T * R, &bd));
    EXPECT_OK(fbuf(&x, NULL, T * H, &bo));
    struct { uint32_t n, xs, os, hasw; float eps; } np = {H, H, H, 1, 1e-6f};
    RUN(&x, "ink_rmsnorm", T, 1, 1, 256, BA(bx), BA(bw), BA(bo), BYTES(&np));
    float *ref = ARENA_ARRAY(&a, float, H);
    for (size_t t = 0; t < T; t++) {
        ink_ref_rmsnorm(xs + t * H, w, H, 1e-6f, ref);
        EXPECT_CLOSE("rmsnorm", gpu_floats(bo) + t * H, ref, H);
    }
    // the gate on the message at column 2 of a [T][4][H] message block
    struct { uint32_t h, r, xs, ms; float eps; } mp = {H, R, H, 4 * H, 1e-6f};
    RUN(&x, "ink_msg_down", R / 8, T, 1, 256, BA(bx), BO(bm, 2 * H * 4), BA(bD), BA(bd), BYTES(&mp));
    RUN(&x, "ink_msg_out", H / 256, T, 1, 256, BA(bx), BO(bm, 2 * H * 4), BA(bd), BA(bF), BA(bG), BA(bo), BYTES(&mp));
    for (size_t t = 0; t < T; t++) {
        ink_ref_msg_gate(xs + t * H, ms + t * 4 * H + 2 * H, D, F, G, H, R, 1e-6f, ref);
        EXPECT_CLOSE("msg_gate", gpu_floats(bo) + t * H, ref, H);
    }
}

TEST(ink_qr_prep) {
    Arena a = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&a, &x));
    const size_t T = 3, pos0 = 9000;
    for (int logscale = 0; logscale < 2; logscale++) {
        float *q = ARENA_ARRAY(&a, float, T * NH * HD), *r = ARENA_ARRAY(&a, float, T * NH * DREL);
        float *qs = ARENA_ARRAY(&a, float, NH * HD), *rs = ARENA_ARRAY(&a, float, NH * DREL);
        fill(&x, q, T * NH * HD, -4, 4);
        fill(&x, r, T * NH * DREL, -3, 3);
        fill(&x, qs, NH * HD, 0, 2);
        fill(&x, rs, NH * DREL, 0, 2);
        for (size_t i = 0; i < HD; i++) q[i] *= 0.01f; // head 0 of token 0 under the clip threshold
        sx_buffer *bq, *br, *bqs, *brs;
        EXPECT_OK(fbuf(&x, q, T * NH * HD, &bq));
        EXPECT_OK(fbuf(&x, r, T * NH * DREL, &br));
        EXPECT_OK(fbuf(&x, qs, NH * HD, &bqs));
        EXPECT_OK(fbuf(&x, rs, NH * DREL, &brs));
        struct { uint32_t nh, hd, drel, pos0, logscale; float tq, tr, alpha, nfloor; } p = {NH, HD, DREL, pos0, logscale, 22.627417f, 8.0f, 0.1f, 8192.0f};
        RUN(&x, "ink_qr_prep", NH, T, 1, HD, BA(bq), BA(br), BA(bqs), BA(brs), BYTES(&p));
        for (size_t t = 0; t < T; t++)
            ink_ref_qr(q + t * NH * HD, r + t * NH * DREL, qs, rs, NH, HD, DREL, pos0 + t, logscale, 22.627417f, 8.0f, 0.1f, 8192.0f);
        EXPECT_CLOSE("qr_prep q", gpu_floats(bq), q, T * NH * HD);
        EXPECT_CLOSE("qr_prep r", gpu_floats(br), r, T * NH * DREL);
    }
}

TEST(ink_convs) {
    Arena a = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&a, &x));
    const size_t C = NKV * HD;
    for (size_t T = 1; T <= 5; T += 4)
        for (int dyn = 0; dyn < 2; dyn++)
            for (int mode = 0; mode < 2; mode++) {
                float *raw = ARENA_ARRAY(&a, float, T * C), *st = ARENA_ARRAY(&a, float, 3 * C), *w = ARENA_ARRAY(&a, float, C * 4);
                float *dw = ARENA_ARRAY(&a, float, T * NKV * 4), *ref = ARENA_ARRAY(&a, float, T * C);
                fill(&x, raw, T * C, -3, 3);
                fill(&x, st, 3 * C, -3, 3);
                fill(&x, w, C * 4, -1, 1);
                fill(&x, dw, T * NKV * 4, -1, 1);
                sx_buffer *br, *bs, *bw, *bd, *bo;
                EXPECT_OK(fbuf(&x, raw, T * C, &br));
                EXPECT_OK(fbuf(&x, st, 3 * C, &bs));
                EXPECT_OK(fbuf(&x, w, C * 4, &bw));
                EXPECT_OK(fbuf(&x, dw, T * NKV * 4, &bd));
                EXPECT_OK(fbuf(&x, NULL, T * C, &bo));
                const float den = dyn ? 3.0f : sqrtf(5.0f);
                struct { uint32_t c, hd, dyn, mode, update; float den, thr, eps; } p = {C, HD, dyn, mode, T == 1, den, 4.0f, 1e-6f};
                RUN(&x, "ink_kv_conv", NKV, T, 1, HD, BA(br), BA(bs), BA(bw), BA(bd), BA(bo), BYTES(&p));
                ink_ref_kv_conv(raw, st, w, dyn ? dw : NULL, T, C, HD, mode, den, 4.0f, 1e-6f, ref);
                EXPECT_CLOSE("kv_conv", gpu_floats(bo), ref, T * C);
                if (T > 1) {
                    struct { uint32_t c, t; } cs = {C, T};
                    RUN(&x, "ink_conv_state", C / 256, 1, 1, 256, BA(br), BA(bs), BYTES(&cs));
                }
                // the state now holds the last 3 inputs
                const float *ns = gpu_floats(bs);
                for (size_t k = 0; k < 3; k++)
                    for (size_t c = 0; c < C; c++) {
                        const long s = (long)T - 3 + (long)k;
                        EXPECT(ns[k * C + c] == (s >= 0 ? raw[s * C + c] : st[(3 + s) * C + c]));
                    }
            }
    // post-attention conv with the residual, both kinds of layer
    for (int boundary = 0; boundary < 2; boundary++) {
        const size_t T = 4;
        float *av = ARENA_ARRAY(&a, float, T * H), *st = ARENA_ARRAY(&a, float, 3 * H), *w = ARENA_ARRAY(&a, float, H * 4);
        float *hin = ARENA_ARRAY(&a, float, T * H), *ref = ARENA_ARRAY(&a, float, T * H);
        fill(&x, av, T * H, -3, 3);
        fill(&x, st, 3 * H, -3, 3);
        fill(&x, w, H * 4, -1, 1);
        fill(&x, hin, T * H, -10, 10);
        sx_buffer *ba, *bs, *bw, *bh, *br, *bb;
        EXPECT_OK(fbuf(&x, av, T * H, &ba));
        EXPECT_OK(fbuf(&x, st, 3 * H, &bs));
        EXPECT_OK(fbuf(&x, w, H * 4, &bw));
        EXPECT_OK(fbuf(&x, hin, T * H, &bh));
        EXPECT_OK(fbuf(&x, NULL, T * H, &br));
        EXPECT_OK(fbuf(&x, NULL, T * H, &bb));
        struct { uint32_t c, boundary, update; float den; } p = {H, boundary, 0, sqrtf(5.0f)};
        RUN(&x, "ink_conv_res", H / 256, T, 1, 256, BA(ba), BA(bs), BA(bw), BA(bh), BA(br), BA(bb), BYTES(&p));
        ink_ref_conv_res(av, st, w, hin, T, H, boundary, sqrtf(5.0f), ref);
        EXPECT_CLOSE("conv_res", gpu_floats(br), ref, T * H);
        if (boundary) EXPECT(memcmp(gpu_floats(bb), hin, T * H * 4) == 0);
    }
}

TEST(ink_router) {
    Arena a = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&a, &x));
    const size_t E = 128, K = 8, T = 3;
    float *xs = ARENA_ARRAY(&a, float, T * H), *W = ARENA_ARRAY(&a, float, E * H), *b = ARENA_ARRAY(&a, float, E);
    fill(&x, xs, T * H, -1, 1);
    fill(&x, W, E * H, -0.05f, 0.05f);
    fill(&x, b, E, -0.2f, 0.2f);
    sx_buffer *bx, *bW, *bb, *bi, *bw;
    EXPECT_OK(fbuf(&x, xs, T * H, &bx));
    EXPECT_OK(fbuf(&x, W, E * H, &bW));
    EXPECT_OK(fbuf(&x, b, E, &bb));
    EXPECT_OK(fbuf(&x, NULL, T * K, &bi));
    EXPECT_OK(fbuf(&x, NULL, T * K, &bw));
    struct { uint32_t e, h, k; float scale; uint32_t forced, check, pos0; float thr; } p = {E, H, K, 8.0f, 0, 0, 0, 1e-4f};
    uint32_t lgp[2] = {E, H};
    sx_buffer *bl;
    EXPECT_OK(fbuf(&x, NULL, T * E, &bl));
    sx_buffer *bst, *bmg;
    EXPECT_OK(gpu_new_buffer(&x.g, 32, &bst));
    EXPECT_OK(fbuf(&x, NULL, T, &bmg));
    RUN(&x, "ink_router_logits", E / 8, T, 1, 256, BA(bx), BA(bW), BA(bl), gpu_bytes_arg(lgp, 8));
    RUN(&x, "ink_router", T, 1, 1, E, BA(bl), BA(bb), BA(bi), BA(bw), BYTES(&p), BA(bi), BA(bmg), BA(bst));
    for (size_t t = 0; t < T; t++) {
        uint32_t idx[8];
        float wt[8];
        ink_ref_router(xs + t * H, W, b, E, H, K, 8.0f, idx, wt);
        const uint32_t *gi = (const uint32_t *)sx_buffer_contents(bi) + t * K;
        for (size_t k = 0; k < K; k++) EXPECT_EQ(gi[k], idx[k]);
        EXPECT_CLOSE("router weights", gpu_floats(bw) + t * K, wt, K);
    }
    // forced experts (verify's replay): the given ids, weights from their own scores
    uint32_t *fi = ARENA_ARRAY(&a, uint32_t, T * K);
    for (size_t i = 0; i < T * K; i++) fi[i] = (uint32_t)((i * 37 + 5) % E);
    sx_buffer *bf;
    EXPECT_OK(gpu_new_buffer(&x.g, T * K * 4, &bf));
    memcpy(sx_buffer_contents(bf), fi, T * K * 4);
    // forced and checked: token 0 at a clear margin, the others at a near tie; every forced list
    // differs from the selection
    float mg[3] = {1.0f, 1e-6f, 1e-6f};
    memcpy(sx_buffer_contents(bmg), mg, sizeof mg);
    uint32_t *stt = sx_buffer_contents(bst);
    stt[0] = stt[1] = stt[3] = 0;
    stt[2] = stt[4] = UINT32_MAX;
    p.forced = 1;
    p.check = 1;
    p.pos0 = 100;
    RUN(&x, "ink_router", T, 1, 1, E, BA(bl), BA(bb), BA(bi), BA(bw), BYTES(&p), BA(bf), BA(bmg), BA(bst));
    EXPECT_EQ(stt[0], 1);
    EXPECT_EQ(stt[1], 2);
    EXPECT_EQ(stt[2], 100);
    for (size_t t = 0; t < T; t++) {
        double sc[8], sum = 0;
        for (size_t k = 0; k < K; k++) {
            const size_t e = fi[t * K + k];
            double z = 0;
            for (size_t i = 0; i < H; i++) z += (double)W[e * H + i] * xs[t * H + i];
            sc[k] = 1.0 / (1.0 + exp(-z));
            sum += sc[k];
        }
        float wt[8];
        for (size_t k = 0; k < K; k++) wt[k] = (float)(sc[k] / (sum + 1e-20) * 8.0);
        const uint32_t *gi = (const uint32_t *)sx_buffer_contents(bi) + t * K;
        for (size_t k = 0; k < K; k++) EXPECT_EQ(gi[k], fi[t * K + k]);
        EXPECT_CLOSE("forced router weights", gpu_floats(bw) + t * K, wt, K);
    }
}

// A chunk of T queries at pos0 over a ring of `win` rows (positions below pos0) and the chunk's
// own rows: the window, the relative bias extent, the sink and the gate. half: the cache in f16
// (the reference then sees the cached keys and values rounded to f16). flash: the prompt-chunk
// kernel instead of the split-K one.
static Err attn_case(Arena *a, QgenCtx *x, size_t pos0, size_t T, size_t win, size_t ext, size_t nsplit, bool half, bool flash,
                     float *err) {
    const size_t ld = NKV * HD, P = pos0 + T;
    float *K = ARENA_ARRAY(a, float, P * ld), *V = ARENA_ARRAY(a, float, P * ld);
    float *q = ARENA_ARRAY(a, float, T * NH * HD), *r = ARENA_ARRAY(a, float, T * NH * DREL), *proj = ARENA_ARRAY(a, float, DREL * ext);
    float *sink = ARENA_ARRAY(a, float, T * NH), *gate = ARENA_ARRAY(a, float, T * NH * HD);
    fill(x, K, P * ld, -0.3f, 0.3f);
    fill(x, V, P * ld, -1, 1);
    fill(x, q, T * NH * HD, -2, 2);
    fill(x, r, T * NH * DREL, -1, 1);
    fill(x, proj, DREL * ext, -1, 1);
    fill(x, sink, T * NH, -2, 2);
    fill(x, gate, T * NH * HD, -3, 3);
    if (half)
        for (size_t i = 0; i < pos0 * ld; i++) K[i] = (float)(_Float16)K[i], V[i] = (float)(_Float16)V[i];
    const size_t rows = win ? win : P, es = half ? 2 : 4;
    uint8_t *kc = ARENA_ARRAY(a, uint8_t, rows * ld * es), *vc = ARENA_ARRAY(a, uint8_t, rows * ld * es);
    memset(kc, 0x7c, rows * ld * es); // rows the kernel must not read: huge or infinite values
    memset(vc, 0x7c, rows * ld * es);
    for (size_t j = 0; j < pos0; j++)
        if (!win || j + win >= pos0)
            for (size_t i = 0; i < ld; i++) {
                const size_t o = (win ? j % win : j) * ld + i;
                if (half) {
                    ((_Float16 *)kc)[o] = (_Float16)K[j * ld + i];
                    ((_Float16 *)vc)[o] = (_Float16)V[j * ld + i];
                } else {
                    ((float *)kc)[o] = K[j * ld + i];
                    ((float *)vc)[o] = V[j * ld + i];
                }
            }
    sx_buffer *bq, *br, *bp, *bkc, *bvc, *bkn, *bvn, *bpart, *bs, *bg, *bo;
    TRY(fbuf(x, q, T * NH * HD, &bq));
    TRY(fbuf(x, r, T * NH * DREL, &br));
    TRY(fbuf(x, proj, DREL * ext, &bp));
    TRY(fbuf(x, (const float *)kc, rows * ld * es / 4, &bkc));
    TRY(fbuf(x, (const float *)vc, rows * ld * es / 4, &bvc));
    TRY(fbuf(x, K + pos0 * ld, T * ld, &bkn));
    TRY(fbuf(x, V + pos0 * ld, T * ld, &bvn));
    TRY(fbuf(x, NULL, T * NH * nsplit * (HD + 2), &bpart));
    TRY(fbuf(x, sink, T * NH, &bs));
    TRY(fbuf(x, gate, T * NH * HD, &bg));
    TRY(fbuf(x, NULL, T * NH * HD, &bo));
    const size_t range = win ? MIN(P, win) : P;
    const uint32_t span = (uint32_t)((range + nsplit - 1) / nsplit);
    struct { uint32_t nh, nkv, hd, drel, ext, pos0, win, ring, nsplit, span, t; float scale; } p = {
        NH, NKV, HD, DREL, (uint32_t)ext, (uint32_t)pos0, (uint32_t)win, (uint32_t)win, (uint32_t)nsplit, span, (uint32_t)T,
        1.0f / sqrtf((float)HD)};
    if (flash) {
        sx_arg af[] = {BA(bq), BA(br), BA(bp), BA(bkc), BA(bvc), BA(bkn), BA(bvn), BA(bs), BA(bg), BA(bo), BYTES(&p)};
        TRY(run(x, half ? "ink_attn_flash_mma" : "ink_attn_flash", af, COUNT(af), NKV, (T + 15) / 16, 1, half ? 192 : 256));
    } else {
        sx_arg ar[] = {BA(bq), BA(br), BA(bp), BA(bkc), BA(bvc), BA(bkn), BA(bvn), BA(bpart), BYTES(&p)};
        TRY(run(x, half ? "ink_attn_part_h" : "ink_attn_part", ar, COUNT(ar), NKV, nsplit, T, 128));
        sx_arg am[] = {BA(bpart), BA(bs), BA(bg), BA(bo), BYTES(&p)};
        TRY(run(x, "ink_attn_merge", am, COUNT(am), NH, T, 1, HD));
    }
    float *ref = ARENA_ARRAY(a, float, NH * HD);
    *err = 0;
    for (size_t t = 0; t < T; t++) {
        // the split-K kernel reads the chunk's earlier rows rounded to the cache's type (as a
        // decode of that query would read them from the cache): so does the reference
        if (half && !flash && t > 0)
            for (size_t i = 0; i < ld; i++) {
                const size_t o = (pos0 + t - 1) * ld + i;
                K[o] = (float)(_Float16)K[o];
                V[o] = (float)(_Float16)V[o];
            }
        ink_ref_attn(q + t * NH * HD, r + t * NH * DREL, proj, K, V, sink + t * NH, gate + t * NH * HD, NH, NKV, HD, DREL, ext,
                     pos0 + t, win, ref);
        *err = fmaxf(*err, rel(gpu_floats(bo) + t * NH * HD, ref, NH * HD));
    }
    return OK;
}

TEST(ink_attention) {
    Arena a = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&a, &x));
    // decode and chunks; sliding with the ring wrapped (a chunk longer than the window too), full
    // with distances past the bias extent; both kernels, f32 and f16 caches
    static const struct { size_t pos0, T, win, ext, nsplit; bool flash; } cs[] = {
        {0, 1, 64, 64, 1, false},    {200, 1, 64, 64, 3, false},  {130, 7, 64, 64, 2, false}, {40, 150, 64, 64, 4, false},
        {300, 1, 0, 96, 8, false},   {90, 20, 0, 96, 4, false},   {0, 37, 64, 64, 1, true},   {130, 70, 64, 64, 1, true},
        {40, 150, 64, 64, 1, true},  {90, 50, 0, 96, 1, true},    {2000, 33, 0, 96, 1, true}, {5000, 1, 0, 96, 64, false}};
    for (int half = 0; half < 2; half++)
        for (size_t i = 0; i < COUNT(cs); i++) {
            float e;
            EXPECT_OK(attn_case(&a, &x, cs[i].pos0, cs[i].T, cs[i].win, cs[i].ext, cs[i].nsplit, half, cs[i].flash, &e));
            // the f16-cache prompt kernel multiplies half Q, K, V and P
            const float bound = half && cs[i].flash ? TOL_INK_ATTN_HALF_REL : TOL_INK_KERNEL_REL;
            dbg("attention case %zu%s: rel err %.3g\n", i, half ? " (f16 cache)" : "", (double)e);
            if (!(e <= bound)) {
                EXPECT(0);
            }
        }
}

static void rand_e4m3(QgenCtx *x, uint8_t *p, size_t n, int emin, int emax) {
    for (size_t i = 0; i < n; i++) {
        uint32_t e = (uint32_t)emin + (uint32_t)(qgen_prng_next(&x->rng) % (uint64_t)(emax - emin + 1));
        uint32_t m = (uint32_t)(qgen_prng_next(&x->rng) & 7), s = (uint32_t)(qgen_prng_next(&x->rng) & 1);
        if (e == 15 && m == 7) m = 6; // not the NaN code
        p[i] = (uint8_t)((s << 7) | (e << 3) | m);
    }
}

// One expert layer: gate_up (rows alternating gate / up) and down, K slots over T tokens, the
// tensors at byte offset sh in their buffers.
static Err expert_case(Arena *a, QgenCtx *x, bool mx, size_t sh, size_t T, float *err) {
    const size_t E = 4, K = 3, I = 1152, in_gu = H, in_dn = I;
    const size_t gw_row = mx ? in_gu : in_gu / 2, gs_row = mx ? in_gu / 32 : in_gu / 16;
    const size_t dw_row = mx ? in_dn : in_dn / 2, ds_row = mx ? in_dn / 32 : in_dn / 16;
    const size_t gwE = 2 * I * gw_row, gsE = 2 * I * gs_row, dwE = H * dw_row, dsE = H * ds_row;
    uint8_t *gw = ARENA_ARRAY(a, uint8_t, E * gwE), *gs = ARENA_ARRAY(a, uint8_t, E * gsE);
    uint8_t *dw = ARENA_ARRAY(a, uint8_t, E * dwE), *ds = ARENA_ARRAY(a, uint8_t, E * dsE);
    if (mx) {
        rand_e4m3(x, gw, E * gwE, 0, 15);
        rand_e4m3(x, dw, E * dwE, 0, 15);
        for (size_t i = 0; i < E * gsE; i++) gs[i] = (uint8_t)(112 + qgen_prng_next(&x->rng) % 6);
        for (size_t i = 0; i < E * dsE; i++) ds[i] = (uint8_t)(112 + qgen_prng_next(&x->rng) % 6);
    } else {
        qgen_prng_bytes(&x->rng, gw, E * gwE);
        qgen_prng_bytes(&x->rng, dw, E * dwE);
        rand_e4m3(x, gs, E * gsE, 1, 10);
        rand_e4m3(x, ds, E * dsE, 1, 10);
    }
    float gf[4], df[4];
    for (size_t e = 0; e < E; e++) gf[e] = ur(x, 1e-4f, 1e-3f), df[e] = ur(x, 1e-4f, 1e-3f);
    float *xs = ARENA_ARRAY(a, float, T * H), *wts = ARENA_ARRAY(a, float, T * K);
    uint32_t *idx = ARENA_ARRAY(a, uint32_t, T * K);
    fill(x, xs, T * H, -2, 2);
    fill(x, wts, T * K, 0.1f, 2);
    for (size_t i = 0; i < T * K; i++) idx[i] = (uint32_t)(qgen_prng_next(&x->rng) % E);
    sx_buffer *b[4];
    const uint8_t *src[4] = {gw, gs, dw, ds};
    const size_t len[4] = {E * gwE, E * gsE, E * dwE, E * dsE};
    for (int k = 0; k < 4; k++) {
        TRY(gpu_new_buffer(&x->g, len[k] + 64, &b[k]));
        memcpy((uint8_t *)sx_buffer_contents(b[k]) + sh, src[k], len[k]);
    }
    sx_buffer *bgf, *bdf, *bx, *bi, *bw, *bh, *by;
    TRY(fbuf(x, gf, E, &bgf));
    TRY(fbuf(x, df, E, &bdf));
    TRY(fbuf(x, xs, T * H, &bx));
    TRY(gpu_new_buffer(&x->g, T * K * 4, &bi));
    memcpy(sx_buffer_contents(bi), idx, T * K * 4);
    TRY(fbuf(x, wts, T * K, &bw));
    TRY(fbuf(x, NULL, T * K * I, &bh));
    TRY(fbuf(x, NULL, T * K * H, &by));
    struct { uint32_t h, i, k, sh_w, sh_s; uint64_t wsE, ssE; float gcap, ucap; } gp = {H, I, K, (uint32_t)sh, (uint32_t)sh, gwE, gsE, 4.0f, 25.0f},
                                                                                       dp = {H, I, K, (uint32_t)sh, (uint32_t)sh, dwE, dsE, 4.0f, 25.0f};
    {
        sx_arg ag[] = {BA(b[0]), BA(b[1]), BA(bgf), BA(bx), BA(bi), BA(bh), BYTES(&gp)};
        TRY(run(x, mx ? "ink_moe_gu_mx" : "ink_moe_gu_nv", ag, COUNT(ag), I / 8, K, T, 256));
        sx_arg ad[] = {BA(b[2]), BA(b[3]), BA(bdf), BA(bh), BA(bi), BA(bw), BA(by), BYTES(&dp)};
        TRY(run(x, mx ? "ink_moe_dn_mx" : "ink_moe_dn_nv", ad, COUNT(ad), H / 32, K, T, 256));
    }
    // the chunk kernels on the same data: pairs grouped by expert, tiles of 32
    sx_buffer *bpairs, *btiles, *bnt, *bhg, *byg;
    TRY(gpu_new_buffer(&x->g, T * K * 4, &bpairs));
    TRY(gpu_new_buffer(&x->g, (T * K / 32 + 128) * 16, &btiles));
    TRY(gpu_new_buffer(&x->g, 16, &bnt));
    TRY(fbuf(x, NULL, T * K * I, &bhg));
    TRY(fbuf(x, NULL, T * K * H, &byg));
    {
        struct { uint32_t e, k, t; } pp = {E, K, (uint32_t)T};
        sx_arg ap[] = {BA(bi), BA(bpairs), BA(btiles), BA(bnt), BYTES(&pp)};
        TRY(run(x, "ink_moe_plan", ap, COUNT(ap), 1, 1, 1, 128));
        const uint32_t maxt = (uint32_t)(T * K / 32 + E);
        sx_arg ag[] = {BA(b[0]), BA(b[1]), BA(bgf), BA(bx), BA(bpairs), BA(btiles), BA(bnt), BA(bhg), BYTES(&gp)};
        TRY(run(x, mx ? "ink_moe_gu_mx_g" : "ink_moe_gu_nv_g", ag, COUNT(ag), 2 * I / 64, maxt, 1, 128));
        sx_arg ad[] = {BA(b[2]), BA(b[3]), BA(bdf), BA(bhg), BA(bpairs), BA(btiles), BA(bnt), BA(bw), BA(byg), BYTES(&dp)};
        TRY(run(x, mx ? "ink_moe_dn_mx_g" : "ink_moe_dn_nv_g", ad, COUNT(ad), H / 64, maxt, 1, 128));
    }
    // the plan: every pair once, grouped by expert in token order
    const uint32_t *pr = (const uint32_t *)sx_buffer_contents(bpairs);
    for (size_t i = 1; i < T * K; i++)
        if (idx[pr[i - 1]] > idx[pr[i]] || (idx[pr[i - 1]] == idx[pr[i]] && pr[i - 1] >= pr[i])) return "BadPlan";
    float *row = ARENA_ARRAY(a, float, H), *hr = ARENA_ARRAY(a, float, I), *yr = ARENA_ARRAY(a, float, H);
    float *hsort = ARENA_ARRAY(a, float, T * K * I);
    for (size_t q = 0; q < T * K; q++) memcpy(hsort + pr[q] * I, gpu_floats(bhg) + q * I, I * 4);
    *err = 0;
    for (size_t t = 0; t < T; t++)
        for (size_t k = 0; k < K; k++) {
            const size_t e = idx[t * K + k];
            for (size_t j = 0; j < I; j++) {
                double g = 0, u = 0;
                if (mx) ink_ref_row_mx(gw + e * gwE, gs + e * gsE, H, 2 * j, row);
                else ink_ref_row_nv(gw + e * gwE, gs + e * gsE, gf[e], H, 2 * j, row);
                for (size_t i = 0; i < H; i++) g += (double)row[i] * xs[t * H + i];
                if (mx) ink_ref_row_mx(gw + e * gwE, gs + e * gsE, H, 2 * j + 1, row);
                else ink_ref_row_nv(gw + e * gwE, gs + e * gsE, gf[e], H, 2 * j + 1, row);
                for (size_t i = 0; i < H; i++) u += (double)row[i] * xs[t * H + i];
                hr[j] = ink_ref_swiglu((float)g, (float)u, 4.0f, 25.0f);
            }
            *err = fmaxf(*err, rel(gpu_floats(bh) + (t * K + k) * I, hr, I));
            *err = fmaxf(*err, rel(hsort + (t * K + k) * I, hr, I));
            // down from the GPU's own activations, so this checks the down kernel alone
            const float *hg = gpu_floats(bh) + (t * K + k) * I;
            for (size_t r = 0; r < H; r++) {
                if (mx) ink_ref_row_mx(dw + e * dwE, ds + e * dsE, I, r, row);
                else ink_ref_row_nv(dw + e * dwE, ds + e * dsE, df[e], I, r, row);
                double acc = 0;
                for (size_t i = 0; i < I; i++) acc += (double)row[i] * hg[i];
                yr[r] = (float)(acc * wts[t * K + k]);
            }
            *err = fmaxf(*err, rel(gpu_floats(by) + (t * K + k) * H, yr, H));
            // the chunk kernels' down from their own activations
            const float *hgg = hsort + (t * K + k) * I;
            for (size_t r = 0; r < H; r++) {
                if (mx) ink_ref_row_mx(dw + e * dwE, ds + e * dsE, I, r, row);
                else ink_ref_row_nv(dw + e * dwE, ds + e * dsE, df[e], I, r, row);
                double acc = 0;
                for (size_t i = 0; i < I; i++) acc += (double)row[i] * hgg[i];
                yr[r] = (float)(acc * wts[t * K + k]);
            }
            *err = fmaxf(*err, rel(gpu_floats(byg) + (t * K + k) * H, yr, H));
        }
    return OK;
}

TEST(ink_experts) {
    Arena a = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&a, &x));
    for (int mx = 0; mx < 2; mx++)
        for (size_t sh = 0; sh < 4; sh++) {
            float e;
            EXPECT_OK(expert_case(&a, &x, mx, sh * 5 % 16, sh == 0 ? 2 : sh == 1 ? 40 : 1, &e));
            if (!(e <= TOL_INK_KERNEL_REL)) {
                dbg("experts %s shift %zu: rel err %.3g\n", mx ? "mxfp8" : "nvfp4", sh * 5 % 16, (double)e);
                EXPECT(0);
            }
        }
}

TEST(ink_e4m3) {
    // the kernel's half-based decode against the reference over all 255 non-NaN codes
    Arena a = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&a, &x));
    (void)x;
    for (int b = 0; b < 256; b++) {
        if ((b & 0x7f) == 0x7f) continue;
        const float v = ink_ref_e4m3((uint8_t)b);
        const int e = (b >> 3) & 15, m = b & 7;
        const double want = (e == 0 ? m / 8.0 * pow(2, -6) : (1 + m / 8.0) * pow(2, e - 7)) * ((b & 0x80) ? -1 : 1);
        EXPECT(v == (float)want);
    }
}

// The tensor-op chunk experts (src/kernels/inkling_mpp.metal) on random data: the plan (every
// pair once, grouped by expert in token order, rows padded to 32, tiles of one expert), gate_up's
// half activations against the f64 reference on the half-rounded inputs (the weights are exact in
// half, so an activation must be the half rounding of a value within TOL_INK_KERNEL_REL of the
// reference, relative to the largest), and down from the kernel's own activations within
// TOL_INK_KERNEL_REL.
static float half_ulp(float v) {
    int e;
    (void)frexpf(v, &e); // v in [2^(e-1), 2^e)
    return ldexpf(1.0f, MAX(e - 11, -24));
}
static Err mpp_expert_case(Arena *a, QgenCtx *x, bool mx, size_t sh, size_t T, size_t E, size_t K, float *eh, float *ey) {
    const size_t I = 1152, in_gu = H, in_dn = I;
    const size_t gw_row = mx ? in_gu : in_gu / 2, gs_row = mx ? in_gu / 32 : in_gu / 16;
    const size_t dw_row = mx ? in_dn : in_dn / 2, ds_row = mx ? in_dn / 32 : in_dn / 16;
    const size_t gwE = 2 * I * gw_row, gsE = 2 * I * gs_row, dwE = H * dw_row, dsE = H * ds_row;
    uint8_t *gw = ARENA_ARRAY(a, uint8_t, E * gwE), *gs = ARENA_ARRAY(a, uint8_t, E * gsE);
    uint8_t *dw = ARENA_ARRAY(a, uint8_t, E * dwE), *ds = ARENA_ARRAY(a, uint8_t, E * dsE);
    if (mx) {
        rand_e4m3(x, gw, E * gwE, 0, 15);
        rand_e4m3(x, dw, E * dwE, 0, 15);
        // block scales on both sides of the kernel's half / f32 switch (105..134)
        for (size_t i = 0; i < E * gsE; i++) gs[i] = (uint8_t)(100 + qgen_prng_next(&x->rng) % 20);
        for (size_t i = 0; i < E * dsE; i++) ds[i] = (uint8_t)(100 + qgen_prng_next(&x->rng) % 20);
    } else {
        qgen_prng_bytes(&x->rng, gw, E * gwE);
        qgen_prng_bytes(&x->rng, dw, E * dwE);
        rand_e4m3(x, gs, E * gsE, 0, 12);
        rand_e4m3(x, ds, E * dsE, 0, 12);
    }
    float *gf = ARENA_ARRAY(a, float, E), *df = ARENA_ARRAY(a, float, E);
    for (size_t e = 0; e < E; e++) gf[e] = ur(x, 1e-4f, 1e-3f), df[e] = ur(x, 1e-4f, 1e-3f);
    float *xs = ARENA_ARRAY(a, float, T * H), *wts = ARENA_ARRAY(a, float, T * K);
    uint32_t *idx = ARENA_ARRAY(a, uint32_t, T * K);
    fill(x, xs, T * H, -2, 2);
    if (mx) // MXFP8 sums are larger: keep them in the softcaps' working range
        for (size_t i = 0; i < T * H; i++) xs[i] *= 0.1f;
    fill(x, wts, T * K, 0.1f, 2);
    // distinct experts per token, skewed so some experts fill several tiles
    for (size_t t = 0; t < T; t++)
        for (size_t k = 0; k < K; k++) {
            uint32_t e;
            bool dup;
            do {
                const uint64_t r = qgen_prng_next(&x->rng);
                e = (uint32_t)(r % 3 == 0 ? r / 3 % 2 : r / 3 % E);
                dup = false;
                for (size_t j = 0; j < k; j++) dup = dup || idx[t * K + j] == e;
            } while (dup);
            idx[t * K + k] = e;
        }
    sx_buffer *b[4];
    const uint8_t *src[4] = {gw, gs, dw, ds};
    const size_t len[4] = {E * gwE, E * gsE, E * dwE, E * dsE};
    for (int k = 0; k < 4; k++) {
        TRY(gpu_new_buffer(&x->g, len[k] + 64, &b[k]));
        memcpy((uint8_t *)sx_buffer_contents(b[k]) + sh, src[k], len[k]);
    }
    const size_t n = T * K, rows_cap = n + 31 * E, tiles_cap = rows_cap / 128 + E;
    sx_buffer *bgf, *bdf, *bx, *bi, *bw, *bsrt, *bt, *bm, *bxs, *bhs, *by;
    TRY(fbuf(x, gf, E, &bgf));
    TRY(fbuf(x, df, E, &bdf));
    TRY(fbuf(x, xs, T * H, &bx));
    TRY(gpu_new_buffer(&x->g, n * 4, &bi));
    memcpy(sx_buffer_contents(bi), idx, n * 4);
    TRY(fbuf(x, wts, n, &bw));
    TRY(gpu_new_buffer(&x->g, rows_cap * 4, &bsrt));
    TRY(gpu_new_buffer(&x->g, tiles_cap * 16, &bt));
    TRY(gpu_new_buffer(&x->g, 16, &bm));
    TRY(gpu_new_buffer(&x->g, rows_cap * H * 2, &bxs));
    TRY(gpu_new_buffer(&x->g, rows_cap * I * 2, &bhs));
    TRY(fbuf(x, NULL, n * H, &by));
    struct { uint32_t e, n; } pp = {(uint32_t)E, (uint32_t)n};
    uint32_t gp2[2] = {H, (uint32_t)K};
    struct { uint32_t h, i, k, sh_w, sh_s; uint64_t wsE, ssE; float gcap, ucap; } gp = {H, I, (uint32_t)K, (uint32_t)sh, (uint32_t)sh, gwE, gsE, 4.0f, 25.0f},
                                                                                       dp = {H, I, (uint32_t)K, (uint32_t)sh, (uint32_t)sh, dwE, dsE, 4.0f, 25.0f};
    {
        sx_arg ap[] = {BA(bi), BA(bsrt), BA(bt), BA(bm), BYTES(&pp)};
        TRY(run_src(x, ink_mpp_src, "ink_mpp_plan", ap, COUNT(ap), 1, 1, 1, 1024));
        sx_arg ag0[] = {BA(bx), BA(bsrt), BA(bm), BA(bxs), gpu_bytes_arg(gp2, 8)};
        TRY(run_src(x, ink_mpp_src, "ink_mpp_gather", ag0, COUNT(ag0), rows_cap, 1, 1, 256));
        sx_arg ag[] = {BA(b[0]), BA(b[1]), BA(bgf), BA(bxs), BA(bt), BA(bm), BA(bhs), BYTES(&gp)};
        TRY(run_src(x, ink_mpp_src, mx ? "ink_mpp_gu_mx" : "ink_mpp_gu_nv", ag, COUNT(ag), 2 * I / 64, tiles_cap, 1, 128));
        sx_arg ad[] = {BA(b[2]), BA(b[3]), BA(bdf), BA(bhs), BA(bt), BA(bm), BA(bsrt), BA(bw), BA(by), BYTES(&dp)};
        TRY(run_src(x, ink_mpp_src, mx ? "ink_mpp_dn_mx" : "ink_mpp_dn_nv", ad, COUNT(ad), H / 64, tiles_cap, 1, 128));
    }
    // the plan
    const int32_t *srt = sx_buffer_contents(bsrt);
    const uint32_t *meta = sx_buffer_contents(bm), (*tl)[4] = sx_buffer_contents(bt);
    const size_t rows = meta[1], ntl = meta[0];
    if (rows > rows_cap || ntl > tiles_cap || rows % 32) return "BadPlan";
    size_t *row_of = ARENA_ARRAY(a, size_t, n);
    for (size_t i = 0; i < n; i++) row_of[i] = SIZE_MAX;
    int64_t prev = -1;
    uint32_t pe = 0;
    for (size_t r = 0; r < rows; r++) {
        if (srt[r] < 0) continue;
        const size_t i = (size_t)srt[r];
        if (i >= n || row_of[i] != SIZE_MAX) return "BadPlan";
        row_of[i] = r;
        if (prev >= 0 && (idx[i] < pe || (idx[i] == pe && (int64_t)i <= prev))) return "BadPlan";
        prev = (int64_t)i, pe = idx[i];
    }
    for (size_t i = 0; i < n; i++)
        if (row_of[i] == SIZE_MAX) return "BadPlan";
    size_t covered = 0;
    for (size_t j = 0; j < ntl; j++) {
        if (tl[j][3] - tl[j][1] > 128 || (tl[j][3] - tl[j][1]) % 32 || tl[j][2] > tl[j][3]) return "BadPlan";
        for (size_t r = tl[j][1]; r < tl[j][3]; r++)
            if (srt[r] >= 0 && idx[srt[r]] != tl[j][0]) return "BadPlan";
        covered += tl[j][3] - tl[j][1];
    }
    if (covered != rows) return "BadPlan";
    // activations and down
    float *row = ARENA_ARRAY(a, float, H), *hr = ARENA_ARRAY(a, float, I), *hg = ARENA_ARRAY(a, float, I), *yr = ARENA_ARRAY(a, float, H);
    const _Float16 *hs = sx_buffer_contents(bhs);
    *eh = *ey = 0;
    for (size_t i = 0; i < n; i++) {
        const size_t t = i / K, e = idx[i], r = row_of[i];
        float hmax = 0;
        for (size_t j = 0; j < I; j++) {
            double g = 0, u = 0;
            if (mx) ink_ref_row_mx(gw + e * gwE, gs + e * gsE, H, 2 * j, row);
            else ink_ref_row_nv(gw + e * gwE, gs + e * gsE, gf[e], H, 2 * j, row);
            for (size_t c = 0; c < H; c++) g += (double)row[c] * (float)(_Float16)xs[t * H + c];
            if (mx) ink_ref_row_mx(gw + e * gwE, gs + e * gsE, H, 2 * j + 1, row);
            else ink_ref_row_nv(gw + e * gwE, gs + e * gsE, gf[e], H, 2 * j + 1, row);
            for (size_t c = 0; c < H; c++) u += (double)row[c] * (float)(_Float16)xs[t * H + c];
            hr[j] = ink_ref_swiglu((float)g, (float)u, 4.0f, 25.0f);
            hmax = fmaxf(hmax, fabsf(hr[j]));
            hg[j] = (float)hs[r * I + j];
        }
        for (size_t j = 0; j < I; j++) {
            const float d = TOL_INK_KERNEL_REL * hmax, bound = 0.5f * half_ulp(fabsf(hr[j]) + d) + d;
            const float over = fabsf(hg[j] - hr[j]) - bound;
            *eh = fmaxf(*eh, over > 0 ? over / fmaxf(hmax, 1e-30f) + TOL_INK_KERNEL_REL : 0);
        }
        for (size_t c = 0; c < H; c++) {
            if (mx) ink_ref_row_mx(dw + e * dwE, ds + e * dsE, I, c, row);
            else ink_ref_row_nv(dw + e * dwE, ds + e * dsE, df[e], I, c, row);
            double acc = 0;
            for (size_t j = 0; j < I; j++) acc += (double)row[j] * hg[j];
            yr[c] = (float)(acc * wts[i]);
        }
        *ey = fmaxf(*ey, rel(gpu_floats(by) + i * H, yr, H));
    }
    return OK;
}

TEST(ink_mpp_experts) {
    Arena a = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&a, &x));
    sx_pipeline *p;
    if (qgen_ctx_pipe(&x, ink_mpp_src, "ink_mpp_plan", &p)) SKIP("accelerator kernels do not compile");
    static const struct { size_t T, E, K; } cs[] = {{1, 3, 2}, {40, 5, 3}, {300, 9, 8}};
    for (int mx = 0; mx < 2; mx++)
        for (size_t c = 0; c < COUNT(cs); c++) {
            const size_t sh = (c * 5 + (size_t)mx * 3) % 16;
            float eh, ey;
            EXPECT_OK(mpp_expert_case(&a, &x, mx, sh, cs[c].T, cs[c].E, cs[c].K, &eh, &ey));
            dbg("ink_mpp experts %s T %zu shift %zu: gate_up beyond the half rounding %.3g, down rel err %.3g\n", mx ? "mxfp8" : "nvfp4",
                cs[c].T, sh, (double)eh, (double)ey);
            if (!(eh <= TOL_INK_KERNEL_REL && ey <= TOL_INK_KERNEL_REL)) EXPECT(0);
        }
}
