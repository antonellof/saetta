// DeepSeek-V4-Flash hyper-connection, norm and router kernels (src/kernels/dsv4_hc.metal)
// against the scalar references in src/dsv4_ref_hc.c, on random data (no pack needed).
#include <math.h>
#include <stdlib.h>

#include "../dsv4_ref_hc.h"
#include "../qgen.h"
#include "../test.h"
#include "../tolerances.h"

static const char dsv4_hc_src[] = {
#embed "../kernels/dsv4_hc.metal"
    , 0};

#define BUF(b) gpu_buf_arg(b)
static sx_arg gpu_buffer_at(sx_buffer *b, size_t off) { return (sx_arg){b, off, NULL, 0}; }
#define BYTES(p) gpu_bytes_arg((p), sizeof(*(p)))

static float urand(QgenCtx *x, float lo, float hi) { return lo + (hi - lo) * qgen_prng_float(&x->rng); }
static uint16_t bf16_bits(float v) {
    float r = dsv4_bf16_round(v);
    uint32_t u;
    memcpy(&u, &r, 4);
    return (uint16_t)(u >> 16);
}

static Err newbuf(QgenCtx *x, size_t bytes, sx_buffer **b) { return gpu_new_buffer(&x->g, MAX(bytes, (size_t)16), b); }

static Err run3(QgenCtx *x, const char *name, const sx_arg *args, size_t nargs, size_t gx, size_t gy, size_t gz,
                size_t threads) {
    sx_pipeline *p;
    TRY(qgen_ctx_pipe(x, dsv4_hc_src, name, &p));
    sx_cmd *cmd = sx_cmd_begin(x->g.dev);
    if (!cmd) return "DispatchFailed";
    sx_cmd_dispatch3(cmd, p, args, nargs, gx, gy, gz, threads);
    char err[512];
    if (sx_cmd_finish(cmd, err, sizeof err) != 0) {
        dbg("dispatch %s: %s\n", name, err);
        return "DispatchFailed";
    }
    return OK;
}

// Max |a - b| over n values, divided by max(max |b|, floor).
static float rel_err(const float *a, const float *b, size_t n, float floor_) {
    float m = 0, s = floor_;
    for (size_t i = 0; i < n; i++) {
        s = fmaxf(s, fabsf(b[i]));
        float d = fabsf(a[i] - b[i]);
        if (!(d <= m)) m = d; // NaN propagates
    }
    return m / s;
}

// BF16 outputs: every value equal to the reference or one BF16 step from it (an f32
// difference of a few ulp before rounding can only cross one rounding boundary), and at most
// max_frac of them different. *frac gets the fraction that differ.
static bool bf16_close(const float *a, const float *b, size_t n, float max_frac, float *frac) {
    size_t diff = 0;
    for (size_t i = 0; i < n; i++) {
        if (a[i] == b[i]) continue;
        float m = fmaxf(fabsf(a[i]), fabsf(b[i]));
        int ex;
        frexpf(m, &ex);
        float step = ldexpf(1.0f, ex - 8); // BF16 spacing below m (8 significant bits)
        if (!(fabsf(a[i] - b[i]) <= step)) {
            dbg("bf16 value %zu: %.9g, expected %.9g\n", i, (double)a[i], (double)b[i]);
            *frac = 1;
            return false;
        }
        diff++;
    }
    *frac = (float)diff / (float)n;
    return *frac <= max_frac;
}

#define EXPECT_REL(what, a, b, n, floor_, bound)                                                  \
    do { float e_ = rel_err(a, b, n, floor_); worst_##what = fmaxf(worst_##what, e_);           \
        if (!(e_ < (bound))) { dbg("%s: rel err %.3g (bound %.3g)\n", #what, (double)e_, (double)(bound)); \
            *failed_ = 1; return; } } while (0)
#define EXPECT_BF16(what, a, b, n)                                                                \
    do { float f_; if (!bf16_close(a, b, n, TOL_DSV4_BF16_FRAC, &f_)) {                           \
            dbg("%s: %.3g of BF16 values differ (bound %.3g)\n", #what, (double)f_, (double)TOL_DSV4_BF16_FRAC); \
            *failed_ = 1; return; } worst_##what = fmaxf(worst_##what, f_); } while (0)

static void bf16_fill(QgenCtx *x, float *p, size_t n, float lo, float hi) {
    for (size_t i = 0; i < n; i++) p[i] = dsv4_bf16_round(urand(x, lo, hi));
}

// Mixer configurations: 0 typical; 1 every comb row peaked on column 0 (Sinkhorn has to
// spread mass the softmax concentrated); 2 comb logits scaled by 1000 (near one-hot rows);
// 3 all mixes inputs flat (uniform comb, pre and post at sigmoid(0)); 4 pre logits far
// below the sigmoid's exp overflow and post far above.
enum { NCFG = 5, NT = 3 };
static void make_mixer(QgenCtx *x, int cfg, float *fn, size_t nmix, float *scale, float *base) {
    float a = cfg == 3 ? 0 : 0.05f;
    for (size_t i = 0; i < nmix * DSV4_HCD; i++) fn[i] = urand(x, -a, a);
    for (size_t i = 0; i < 3; i++) scale[i] = cfg == 3 ? 0 : urand(x, 0.5f, 2);
    for (size_t i = 0; i < nmix; i++) base[i] = cfg == 3 ? 0 : urand(x, -1, 1);
    if (cfg == 1)
        for (size_t j = 0; j < DSV4_HC && nmix == DSV4_MIX; j++) base[2 * DSV4_HC + 4 * j] = 30;
    if (cfg == 2) scale[2] = 1000;
    if (cfg == 4) {
        for (size_t j = 0; j < DSV4_HC; j++) base[j] = -300;
        if (nmix == DSV4_MIX)
            for (size_t j = 0; j < DSV4_HC; j++) base[DSV4_HC + j] = 300;
    }
}

// Token streams: token 0 in [-1, 1), token 1 in [-100, 100), token 2 in [-1e-3, 1e-3).
static void make_streams(QgenCtx *x, float *xs) {
    static const float mag[NT] = {1, 100, 1e-3f};
    for (size_t t = 0; t < NT; t++) bf16_fill(x, xs + t * DSV4_HCD, DSV4_HCD, -mag[t], mag[t]);
}

// The mixes the split kernel forms from the partials (slices in order, f32), for feeding
// the reference split the GPU's own matvec result.
static void mixes_from_part(const float *part, size_t nmix, size_t t, float *m) {
    const float *pt = part + t * DSV4_HC_SLICES * (nmix + 1);
    float ss = 0;
    for (size_t s = 0; s < DSV4_HC_SLICES; s++) ss += pt[s * (nmix + 1) + nmix];
    float r = 1.0f / sqrtf(ss * (1.0f / DSV4_HCD) + DSV4_NORM_EPS);
    for (size_t i = 0; i < nmix; i++) {
        float a = 0;
        for (size_t s = 0; s < DSV4_HC_SLICES; s++) a += pt[s * (nmix + 1) + i];
        m[i] = a * r;
    }
}

TEST(dsv4_hc_pre) {
    Arena arena = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&arena, &x));
    float worst_mixes = 0, worst_split = 0, worst_split_e2e = 0, worst_y = 0, worst_ynorm = 0;
    sx_buffer *fnb, *xb, *partb, *scb, *bsb, *yb, *ynb, *postb, *combb, *nwb;
    EXPECT_OK(newbuf(&x, DSV4_MIX * DSV4_HCD * 4, &fnb));
    EXPECT_OK(newbuf(&x, NT * DSV4_HCD * 4, &xb));
    EXPECT_OK(newbuf(&x, NT * DSV4_HC_SLICES * (DSV4_MIX + 1) * 4, &partb));
    EXPECT_OK(newbuf(&x, 3 * 4, &scb));
    EXPECT_OK(newbuf(&x, DSV4_MIX * 4, &bsb));
    EXPECT_OK(newbuf(&x, NT * DSV4_D * 4, &yb));
    EXPECT_OK(newbuf(&x, NT * DSV4_D * 4, &ynb));
    EXPECT_OK(newbuf(&x, NT * DSV4_HC * 4, &postb));
    EXPECT_OK(newbuf(&x, NT * DSV4_HC * DSV4_HC * 4, &combb));
    EXPECT_OK(newbuf(&x, DSV4_D * 2, &nwb));
    float *fn = gpu_floats(fnb), *xs = gpu_floats(xb), *part = gpu_floats(partb), *sc = gpu_floats(scb),
          *bs = gpu_floats(bsb);
    uint16_t *nw = sx_buffer_contents(nwb);
    for (size_t i = 0; i < DSV4_D; i++) nw[i] = bf16_bits(urand(&x, 0.5f, 1.5f));
    float *ref_y = ARENA_ARRAY(&arena, float, DSV4_D), *ref_yn = ARENA_ARRAY(&arena, float, DSV4_D);
    for (int cfg = 0; cfg < NCFG; cfg++) {
        make_mixer(&x, cfg, fn, DSV4_MIX, sc, bs);
        make_streams(&x, xs);
        const uint32_t mp[2] = {DSV4_MIX, NT};
        sx_arg ma[] = {BUF(fnb), BUF(xb), BUF(partb), BYTES(&mp)};
        EXPECT_OK(run3(&x, "dsv4_hc_mix", ma, COUNT(ma), DSV4_MIX, DSV4_HC_SLICES, NT, 128));
        sx_arg sa[] = {BUF(partb), BUF(scb), BUF(bsb), BUF(xb), BUF(yb), BUF(postb), BUF(combb)};
        EXPECT_OK(run3(&x, "dsv4_hc_split", sa, COUNT(sa), NT, 1, 1, 256));
        sx_arg na[] = {BUF(partb), BUF(scb), BUF(bsb), BUF(xb), BUF(ynb), BUF(postb), BUF(combb), BUF(nwb)};
        EXPECT_OK(run3(&x, "dsv4_hc_split_norm", na, COUNT(na), NT, 1, 1, 256));
        for (size_t t = 0; t < NT; t++) {
            const float *xt = xs + t * DSV4_HCD;
            float m_ref[DSV4_MIX], m_gpu[DSV4_MIX];
            dsv4_ref_hc_mixes(fn, DSV4_MIX, xt, m_ref);
            mixes_from_part(part, DSV4_MIX, t, m_gpu);
            EXPECT_REL(mixes, m_gpu, m_ref, DSV4_MIX, 1e-3f, TOL_DSV4_MIX_REL);
            // split from the GPU's own mixes: only the elementwise math differs
            float pre[DSV4_HC], post[DSV4_HC], comb[16], pc[DSV4_HC + 16], gc[DSV4_HC + 16];
            dsv4_ref_hc_split(m_gpu, sc, bs, pre, post, comb);
            memcpy(pc, post, sizeof post);
            memcpy(pc + DSV4_HC, comb, sizeof comb);
            memcpy(gc, gpu_floats(postb) + t * DSV4_HC, sizeof post);
            memcpy(gc + DSV4_HC, gpu_floats(combb) + t * 16, sizeof comb);
            EXPECT_REL(split, gc, pc, DSV4_HC + 16, 1, TOL_DSV4_SPLIT_ABS);
            dsv4_ref_hc_weighted(pre, xt, ref_y);
            EXPECT_BF16(y, gpu_floats(yb) + t * DSV4_D, ref_y, DSV4_D);
            dsv4_ref_rmsnorm(ref_y, nw, DSV4_D, ref_yn);
            EXPECT_BF16(ynorm, gpu_floats(ynb) + t * DSV4_D, ref_yn, DSV4_D);
            // end to end from the f64 mixes, where the mixer is well conditioned (cfg 2
            // amplifies the mixes' f32 rounding by its scale of 1000)
            if (cfg != 2) {
                float pre2[DSV4_HC], post2[DSV4_HC], comb2[16];
                dsv4_ref_hc_split(m_ref, sc, bs, pre2, post2, comb2);
                memcpy(pc, post2, sizeof post2);
                memcpy(pc + DSV4_HC, comb2, sizeof comb2);
                EXPECT_REL(split_e2e, gc, pc, DSV4_HC + 16, 1, TOL_DSV4_SPLIT_E2E_ABS);
            }
            // the Sinkhorn output is doubly stochastic up to eps: columns sum to 1
            for (size_t k = 0; k < DSV4_HC; k++) {
                const float *c = gpu_floats(combb) + t * 16;
                float s = c[k] + c[4 + k] + c[8 + k] + c[12 + k];
                EXPECT(fabsf(s - 1) < 1e-4f);
            }
        }
    }
    dbg("dsv4_hc_pre: mixes rel %.3g, split abs %.3g (end to end %.3g), BF16 differing y %.3g, normed %.3g\n",
        (double)worst_mixes, (double)worst_split, (double)worst_split_e2e, (double)worst_y, (double)worst_ynorm);
    arena_reset(&arena);
}

TEST(dsv4_hc_head) {
    Arena arena = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&arena, &x));
    float worst_mixes = 0, worst_y = 0, worst_ynorm = 0;
    sx_buffer *fnb, *xb, *partb, *scb, *bsb, *yb, *ynb, *nwb;
    EXPECT_OK(newbuf(&x, DSV4_HC * DSV4_HCD * 4, &fnb));
    EXPECT_OK(newbuf(&x, NT * DSV4_HCD * 4, &xb));
    EXPECT_OK(newbuf(&x, NT * DSV4_HC_SLICES * (DSV4_HC + 1) * 4, &partb));
    EXPECT_OK(newbuf(&x, 3 * 4, &scb));
    EXPECT_OK(newbuf(&x, DSV4_HC * 4, &bsb));
    EXPECT_OK(newbuf(&x, NT * DSV4_D * 4, &yb));
    EXPECT_OK(newbuf(&x, NT * DSV4_D * 4, &ynb));
    EXPECT_OK(newbuf(&x, DSV4_D * 2, &nwb));
    float *fn = gpu_floats(fnb), *xs = gpu_floats(xb), *part = gpu_floats(partb), *sc = gpu_floats(scb),
          *bs = gpu_floats(bsb);
    uint16_t *nw = sx_buffer_contents(nwb);
    for (size_t i = 0; i < DSV4_D; i++) nw[i] = bf16_bits(urand(&x, -2, 2));
    float *ref_y = ARENA_ARRAY(&arena, float, DSV4_D), *ref_yn = ARENA_ARRAY(&arena, float, DSV4_D);
    for (int cfg = 0; cfg < NCFG; cfg++) {
        make_mixer(&x, cfg, fn, DSV4_HC, sc, bs);
        make_streams(&x, xs);
        const uint32_t mp[2] = {DSV4_HC, NT};
        sx_arg ma[] = {BUF(fnb), BUF(xb), BUF(partb), BYTES(&mp)};
        EXPECT_OK(run3(&x, "dsv4_hc_mix", ma, COUNT(ma), DSV4_HC, DSV4_HC_SLICES, NT, 128));
        sx_arg ha[] = {BUF(partb), BUF(scb), BUF(bsb), BUF(xb), BUF(yb)};
        EXPECT_OK(run3(&x, "dsv4_hc_head", ha, COUNT(ha), NT, 1, 1, 256));
        sx_arg hn[] = {BUF(partb), BUF(scb), BUF(bsb), BUF(xb), BUF(ynb), BUF(nwb)};
        EXPECT_OK(run3(&x, "dsv4_hc_head_norm", hn, COUNT(hn), NT, 1, 1, 256));
        for (size_t t = 0; t < NT; t++) {
            const float *xt = xs + t * DSV4_HCD;
            float m_ref[DSV4_HC], m_gpu[DSV4_HC], pre[DSV4_HC];
            dsv4_ref_hc_mixes(fn, DSV4_HC, xt, m_ref);
            mixes_from_part(part, DSV4_HC, t, m_gpu);
            EXPECT_REL(mixes, m_gpu, m_ref, DSV4_HC, 1e-3f, TOL_DSV4_MIX_REL);
            dsv4_ref_hc_head_pre(m_gpu, sc[0], bs, pre);
            dsv4_ref_hc_weighted(pre, xt, ref_y);
            EXPECT_BF16(y, gpu_floats(yb) + t * DSV4_D, ref_y, DSV4_D);
            dsv4_ref_rmsnorm(ref_y, nw, DSV4_D, ref_yn);
            EXPECT_BF16(ynorm, gpu_floats(ynb) + t * DSV4_D, ref_yn, DSV4_D);
        }
    }
    dbg("dsv4_hc_head: mixes rel %.3g, BF16 differing y %.3g, normed %.3g\n", (double)worst_mixes,
        (double)worst_y, (double)worst_ynorm);
    arena_reset(&arena);
}

TEST(dsv4_hc_post) {
    Arena arena = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&arena, &x));
    float worst_out = 0;
    sx_buffer *fb, *rb, *pb, *cb, *ob;
    EXPECT_OK(newbuf(&x, NT * DSV4_D * 4, &fb));
    EXPECT_OK(newbuf(&x, NT * DSV4_HCD * 4, &rb));
    EXPECT_OK(newbuf(&x, NT * DSV4_HC * 4, &pb));
    EXPECT_OK(newbuf(&x, NT * 16 * 4, &cb));
    EXPECT_OK(newbuf(&x, NT * DSV4_HCD * 4, &ob));
    float *f = gpu_floats(fb), *res = gpu_floats(rb), *post = gpu_floats(pb), *comb = gpu_floats(cb);
    float *ref = ARENA_ARRAY(&arena, float, DSV4_HCD);
    for (int rep = 0; rep < 2; rep++) {
        make_streams(&x, res);
        bf16_fill(&x, f, NT * DSV4_D, -10, 10);
        // post and comb from the reference split of random mixes
        for (size_t t = 0; t < NT; t++) {
            float m[DSV4_MIX], sc[3] = {1, 1, 1}, bs[DSV4_MIX] = {0}, pre[DSV4_HC];
            for (size_t i = 0; i < DSV4_MIX; i++) m[i] = urand(&x, -3, 3);
            dsv4_ref_hc_split(m, sc, bs, pre, post + t * DSV4_HC, comb + t * 16);
        }
        const uint32_t T = NT;
        // rep 1 writes in place over the residual
        sx_buffer *outb = rep ? rb : ob;
        float *res_copy = ARENA_ARRAY(&arena, float, NT * DSV4_HCD);
        memcpy(res_copy, res, NT * DSV4_HCD * 4);
        sx_arg a[] = {BUF(fb), BUF(rb), BUF(pb), BUF(cb), BUF(outb), BYTES(&T)};
        EXPECT_OK(run3(&x, "dsv4_hc_post", a, COUNT(a), NT * DSV4_D / 4 / 256, 1, 1, 256));
        for (size_t t = 0; t < NT; t++) {
            dsv4_ref_hc_post(f + t * DSV4_D, res_copy + t * DSV4_HCD, post + t * DSV4_HC, comb + t * 16, ref);
            EXPECT_BF16(out, gpu_floats(outb) + t * DSV4_HCD, ref, DSV4_HCD);
        }
    }
    dbg("dsv4_hc_post: BF16 differing %.3g\n", (double)worst_out);
    arena_reset(&arena);
}

TEST(dsv4_hc_rmsnorm) {
    Arena arena = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&arena, &x));
    float worst_out = 0;
    static const uint32_t ns[] = {4096, 1024, 512, 128};
    for (size_t ni = 0; ni < COUNT(ns); ni++) {
        const uint32_t n = ns[ni];
        sx_buffer *xb, *wb, *ob;
        EXPECT_OK(newbuf(&x, NT * n * 4, &xb));
        EXPECT_OK(newbuf(&x, n * 2, &wb));
        EXPECT_OK(newbuf(&x, NT * n * 4, &ob));
        static const float mag[NT] = {1, 300, 1e-4f};
        for (size_t t = 0; t < NT; t++) bf16_fill(&x, gpu_floats(xb) + t * n, n, -mag[t], mag[t]);
        uint16_t *w = sx_buffer_contents(wb);
        for (size_t i = 0; i < n; i++) w[i] = bf16_bits(urand(&x, -3, 3));
        const uint32_t p[2] = {n, NT};
        sx_arg a[] = {BUF(xb), BUF(wb), BUF(ob), BYTES(&p)};
        EXPECT_OK(run3(&x, "dsv4_rmsnorm", a, COUNT(a), NT, 1, 1, 256));
        float *ref = ARENA_ARRAY(&arena, float, n);
        for (size_t t = 0; t < NT; t++) {
            dsv4_ref_rmsnorm(gpu_floats(xb) + t * n, w, n, ref);
            EXPECT_BF16(out, gpu_floats(ob) + t * n, ref, n);
        }
    }
    dbg("dsv4_hc_rmsnorm: BF16 differing %.3g\n", (double)worst_out);
    arena_reset(&arena);
}

// Router tokens: 0-1 random; 2 x = 0 (every logit exactly 0: ties everywhere, broken by the
// bias pattern and then by index); 3 x scaled so logits pass softplus's threshold of 20;
// 4 random with gate rows 7, 100 and 200 identical and their biases equal and dominant.
enum { RT = 5 };
TEST(dsv4_hc_router) {
    Arena arena = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&arena, &x));
    float worst_logits = 0, worst_weights = 0, worst_hash = 0;
    sx_buffer *wb, *xb, *lb, *bb, *ib, *gb;
    EXPECT_OK(newbuf(&x, DSV4_NE * DSV4_D * 2, &wb));
    EXPECT_OK(newbuf(&x, RT * DSV4_D * 4, &xb));
    EXPECT_OK(newbuf(&x, RT * DSV4_NE * 4, &lb));
    EXPECT_OK(newbuf(&x, RT * DSV4_NE * 4, &bb)); // bias per token case: [RT][256], one dispatch each
    EXPECT_OK(newbuf(&x, RT * DSV4_TOPK * 4, &ib));
    EXPECT_OK(newbuf(&x, RT * DSV4_TOPK * 4, &gb));
    uint16_t *w = sx_buffer_contents(wb);
    float *xs = gpu_floats(xb), *bias = gpu_floats(bb);
    for (size_t i = 0; i < DSV4_NE * DSV4_D; i++) w[i] = bf16_bits(urand(&x, -0.05f, 0.05f));
    memcpy(w + 100 * DSV4_D, w + 7 * DSV4_D, DSV4_D * 2);
    memcpy(w + 200 * DSV4_D, w + 7 * DSV4_D, DSV4_D * 2);
    bf16_fill(&x, xs, RT * DSV4_D, -1, 1);
    memset(xs + 2 * DSV4_D, 0, DSV4_D * 4);
    for (size_t i = 0; i < DSV4_D; i++) xs[3 * DSV4_D + i] = dsv4_bf16_round(xs[3 * DSV4_D + i] * 40);
    for (size_t t = 0; t < RT; t++)
        for (size_t e = 0; e < DSV4_NE; e++) bias[t * DSV4_NE + e] = urand(&x, -0.1f, 0.1f);
    for (size_t e = 0; e < DSV4_NE; e++) bias[2 * DSV4_NE + e] = (e % 3 == 1) ? 0.5f : 0;
    bias[4 * DSV4_NE + 7] = bias[4 * DSV4_NE + 100] = bias[4 * DSV4_NE + 200] = 10;

    sx_arg la[] = {BUF(wb), BUF(xb), BUF(lb)};
    EXPECT_OK(run3(&x, "dsv4_router_logits", la, COUNT(la), DSV4_NE / 8, RT, 1, 256));
    const float *lg = gpu_floats(lb);
    bool over20 = false;
    for (size_t t = 0; t < RT; t++) {
        float ref[DSV4_NE];
        dsv4_ref_router_logits(w, xs + t * DSV4_D, ref);
        EXPECT_REL(logits, lg + t * DSV4_NE, ref, DSV4_NE, 1e-3f, TOL_DSV4_ROUTER_REL);
        for (size_t e = 0; e < DSV4_NE; e++) over20 |= t == 3 && lg[t * DSV4_NE + e] > 20;
    }
    EXPECT(over20);
    for (size_t t = 0; t < RT; t++) {
        sx_arg sa[] = {gpu_buffer_at(lb, t * DSV4_NE * 4), gpu_buffer_at(bb, t * DSV4_NE * 4),
                       gpu_buffer_at(ib, t * DSV4_TOPK * 4), gpu_buffer_at(gb, t * DSV4_TOPK * 4)};
        EXPECT_OK(run3(&x, "dsv4_router_select", sa, COUNT(sa), 1, 1, 1, 256));
    }
    const uint32_t *ids = sx_buffer_contents(ib);
    const float *wts = gpu_floats(gb);
    for (size_t t = 0; t < RT; t++) {
        uint32_t rid[DSV4_TOPK];
        float rw[DSV4_TOPK];
        dsv4_ref_router_select(lg + t * DSV4_NE, bias + t * DSV4_NE, rid, rw);
        for (size_t k = 0; k < DSV4_TOPK; k++) EXPECT_EQ(ids[t * DSV4_TOPK + k], rid[k]);
        EXPECT_REL(weights, wts + t * DSV4_TOPK, rw, DSV4_TOPK, 0, TOL_DSV4_ROUTER_REL);
    }
    // the constructed ties
    static const uint32_t tie2[DSV4_TOPK] = {1, 4, 7, 10, 13, 16};
    for (size_t k = 0; k < DSV4_TOPK; k++) EXPECT_EQ(ids[2 * DSV4_TOPK + k], tie2[k]);
    EXPECT_EQ(ids[4 * DSV4_TOPK + 0], 7);
    EXPECT_EQ(ids[4 * DSV4_TOPK + 1], 100);
    EXPECT_EQ(ids[4 * DSV4_TOPK + 2], 200);
    EXPECT(wts[4 * DSV4_TOPK] == wts[4 * DSV4_TOPK + 2]);
    for (size_t k = 1; k < DSV4_TOPK; k++) EXPECT(wts[2 * DSV4_TOPK + k] == wts[2 * DSV4_TOPK]);

    // hash routing over a 1000-row table of 6 distinct ids per row
    enum { V = 1000 };
    sx_buffer *eb, *tb;
    EXPECT_OK(newbuf(&x, V * DSV4_TOPK * 4, &eb));
    EXPECT_OK(newbuf(&x, RT * 4, &tb));
    uint32_t *eid = sx_buffer_contents(eb), *tok = sx_buffer_contents(tb);
    for (size_t v = 0; v < V; v++)
        for (size_t k = 0; k < DSV4_TOPK; k++) {
            uint32_t e;
            bool dup;
            do {
                e = (uint32_t)(qgen_prng_next(&x.rng) % DSV4_NE);
                dup = false;
                for (size_t j = 0; j < k; j++) dup |= eid[v * DSV4_TOPK + j] == e;
            } while (dup);
            eid[v * DSV4_TOPK + k] = e;
        }
    for (size_t t = 0; t < RT; t++) tok[t] = (uint32_t)(qgen_prng_next(&x.rng) % V);
    tok[4] = 0;
    memcpy(eid, (uint32_t[DSV4_TOPK]){200, 7, 100, 0, 255, 128}, DSV4_TOPK * 4); // identical rows again
    sx_arg ha[] = {BUF(wb), BUF(xb), BUF(eb), BUF(tb), BUF(ib), BUF(gb)};
    EXPECT_OK(run3(&x, "dsv4_router_hash", ha, COUNT(ha), RT, 1, 1, 32 * DSV4_TOPK));
    for (size_t t = 0; t < RT; t++) {
        uint32_t rid[DSV4_TOPK];
        float rw[DSV4_TOPK];
        dsv4_ref_router_hash(lg + t * DSV4_NE, eid + tok[t] * DSV4_TOPK, rid, rw);
        for (size_t k = 0; k < DSV4_TOPK; k++) EXPECT_EQ(ids[t * DSV4_TOPK + k], rid[k]);
        EXPECT_REL(hash, wts + t * DSV4_TOPK, rw, DSV4_TOPK, 0, TOL_DSV4_ROUTER_REL);
    }
    EXPECT(wts[4 * DSV4_TOPK] == wts[4 * DSV4_TOPK + 1] && wts[4 * DSV4_TOPK] == wts[4 * DSV4_TOPK + 2]);
    dbg("dsv4_hc_router: logits rel %.3g, weights rel %.3g, hash weights rel %.3g\n", (double)worst_logits,
        (double)worst_weights, (double)worst_hash);
    arena_reset(&arena);
}
