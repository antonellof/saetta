// DeepSeek-V4-Flash attention kernels (src/kernels/dsv4_attn.metal) against the scalar
// references in src/dsv4_ref_attn.c, on random data (no pack needed).
#include <math.h>
#include <stdlib.h>

#include "../dsv4_ref_attn.h"
#include "../qgen.h"
#include "../test.h"
#include "../tolerances.h"

static const char dsv4_attn_src[] = {
#embed "../kernels/dsv4_attn.metal"
    , 0};

#define BUF(b) gpu_buf_arg(b)
#define BYTES(p) gpu_bytes_arg((p), sizeof(*(p)))

typedef struct {
    uint32_t pos0, T, rope_pos0, cap, ls;
} AttnP; // DsAttnP

static float urand(QgenCtx *x, float lo, float hi) { return lo + (hi - lo) * qgen_prng_float(&x->rng); }
static Err newbuf(QgenCtx *x, size_t bytes, sx_buffer **b) { return gpu_new_buffer(&x->g, MAX(bytes, (size_t)16), b); }
static uint16_t *bits(sx_buffer *b) { return sx_buffer_contents(b); }

static void put_bf16(uint16_t *dst, const float *src, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = dsv4a_bf16_bits(src[i]);
}
static void fill_bf16(QgenCtx *x, float *p, size_t n, float lo, float hi) {
    for (size_t i = 0; i < n; i++) p[i] = dsv4a_bf16(urand(x, lo, hi));
}

typedef struct {
    const char *name;
    size_t nargs, gx, gy, gz, threads;
    sx_arg args[12];
} Disp;

// The dispatches in order in one command buffer.
static Err run(QgenCtx *x, const Disp *d, size_t n) {
    sx_pipeline *p[4];
    for (size_t i = 0; i < n; i++) TRY(qgen_ctx_pipe(x, dsv4_attn_src, d[i].name, &p[i]));
    sx_cmd *cmd = sx_cmd_begin(x->g.dev);
    if (!cmd) return "DispatchFailed";
    for (size_t i = 0; i < n; i++) sx_cmd_dispatch3(cmd, p[i], d[i].args, d[i].nargs, d[i].gx, d[i].gy, d[i].gz, d[i].threads);
    char err[512];
    if (sx_cmd_finish(cmd, err, sizeof err) != 0) {
        dbg("dispatch %s: %s\n", d[0].name, err);
        return "DispatchFailed";
    }
    return OK;
}

// One BF16 step below |v| (8 significant bits).
static float bf16_step(float v) {
    int ex;
    frexpf(fabsf(v), &ex);
    return ldexpf(1.0f, ex - 8);
}

// GPU BF16 values a against reference values b: the largest |a - b| relative to the largest
// |b| of the same group of `group` values (a head), and the fraction not identical.
typedef struct {
    double rel, frac;
} Cmp;
static Cmp compare(const uint16_t *a, const float *b, size_t n, size_t group) {
    Cmp c = {0, 0};
    size_t diff = 0;
    for (size_t g0 = 0; g0 < n; g0 += group) {
        float mx = 0, md = 0;
        for (size_t i = g0; i < g0 + group; i++) {
            mx = fmaxf(mx, fabsf(b[i]));
            const float d = fabsf(dsv4a_bf16_val(a[i]) - b[i]);
            if (!(d <= md)) md = d; // NaN propagates
            diff += dsv4a_bf16_val(a[i]) != b[i];
        }
        const double r = mx > 0 ? (double)md / mx : (double)md;
        if (!(r <= c.rel)) c.rel = r;
    }
    c.frac = (double)diff / (double)n;
    return c;
}
// Every differing value within one BF16 step of the reference.
static bool within_step(const uint16_t *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        const float v = dsv4a_bf16_val(a[i]);
        if (v == b[i]) continue;
        if (!(fabsf(v - b[i]) <= bf16_step(fmaxf(fabsf(v), fabsf(b[i]))))) {
            dbg("value %zu: %.9g, expected %.9g\n", i, (double)v, (double)b[i]);
            return false;
        }
    }
    return true;
}

TEST(dsv4_attn_rope) {
    EXPECT(DSV4A_SCALE == (float)pow(512.0, -0.5));
    float f0[DSV4A_NPAIR], f1[DSV4A_NPAIR];
    dsv4a_rope_freqs(false, f0);
    dsv4a_rope_freqs(true, f1);
    for (int i = 0; i < DSV4A_NPAIR; i++) {
        const double e0 = pow(10000.0, -(double)(2 * i) / 64), e1 = pow(160000.0, -(double)(2 * i) / 64);
        EXPECT(fabs(f0[i] - e0) <= 2e-7 * e0);
        // YaRN ramp (i - 15) / 10: unchanged up to 15, divided by 16 from 25 on
        const double ramp = fmin(fmax((i - 15) / 10.0, 0.0), 1.0);
        const double e = e1 * (1 - ramp) + e1 / 16 * ramp;
        EXPECT(fabs(f1[i] - e) <= 4e-7 * e);
        float g1 = 1.0f / powf(160000.0f, (float)(2 * i) / 64.0f);
        if (i <= 15) EXPECT(f1[i] == g1);
        if (i >= 25) EXPECT(f1[i] == g1 / 16);
    }
    // the rotation and its inverse are a rotation by +-angle (to BF16 rounding)
    float cs[DSV4A_NPAIR * 2], x[DSV4A_RD], y[DSV4A_RD];
    dsv4a_rope_table(f1, 77777, 1, cs);
    for (int i = 0; i < DSV4A_RD; i++) x[i] = y[i] = dsv4a_bf16((float)(i % 7) - 3.0f);
    dsv4a_ref_rope(y, cs, false);
    dsv4a_ref_rope(y, cs, true);
    for (int i = 0; i < DSV4A_RD; i++) EXPECT(fabsf(y[i] - x[i]) <= 0.05f);
}

// Rope tables for both thetas over positions [p0, p0 + n), uploaded as a buffer.
static Err rope_buf(QgenCtx *x, bool compressed, uint32_t p0, size_t n, float *host, sx_buffer **b) {
    float f[DSV4A_NPAIR];
    dsv4a_rope_freqs(compressed, f);
    dsv4a_rope_table(f, p0, n, host);
    TRY(newbuf(x, n * DSV4A_NPAIR * 2 * 4, b));
    memcpy(gpu_floats(*b), host, n * DSV4A_NPAIR * 2 * 4);
    return OK;
}

TEST(dsv4_attn_q) {
    Arena arena = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&arena, &x));
    enum { T = 4 };
    const size_t N = (size_t)T * DSV4A_NH * DSV4A_HD;
    float *q = ARENA_ARRAY(&arena, float, N), *cs = ARENA_ARRAY(&arena, float, 16 * DSV4A_NPAIR * 2);
    sx_buffer *qb;
    EXPECT_OK(newbuf(&x, N * 2, &qb));
    static const uint32_t pos0s[] = {0, 125, 65530, 200000};
    double worst_rel = 0, worst_frac = 0;
    for (int theta = 0; theta < 2; theta++)
        for (size_t pi = 0; pi < COUNT(pos0s); pi++) {
            // per-head magnitudes from 1e-3 to 1e2
            for (size_t r = 0; r < (size_t)T * DSV4A_NH; r++) {
                const float a = powf(10.0f, urand(&x, -3, 2));
                fill_bf16(&x, q + r * DSV4A_HD, DSV4A_HD, -a, a);
            }
            put_bf16(bits(qb), q, N);
            const uint32_t rp0 = pos0s[pi] >= 3 ? pos0s[pi] - 3 : 0;
            sx_buffer *csb;
            EXPECT_OK(rope_buf(&x, theta, rp0, 16, cs, &csb));
            const AttnP p = {pos0s[pi], T, rp0, 0, 0};
            Disp d = {"dsv4_attn_q", 3, DSV4A_NH, T, 1, 128, {BUF(qb), BUF(csb), BYTES(&p)}};
            EXPECT_OK(run(&x, &d, 1));
            for (size_t t = 0; t < T; t++) dsv4a_ref_q(q + t * DSV4A_NH * DSV4A_HD, cs + (pos0s[pi] + t - rp0) * DSV4A_NPAIR * 2);
            Cmp c = compare(bits(qb), q, N, DSV4A_HD);
            EXPECT(within_step(bits(qb), q, N));
            EXPECT(c.frac <= TOL_DSV4_ATTN_Q_FRAC);
            worst_rel = fmax(worst_rel, c.rel);
            worst_frac = fmax(worst_frac, c.frac);
            sx_buffer_release(csb);
        }
    dbg("dsv4_attn_q: BF16 values differing %.3g (within one step), rel %.3g\n", worst_frac, worst_rel);
    arena_reset(&arena);
}

TEST(dsv4_attn_kv) {
    Arena arena = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&arena, &x));
    enum { T = DSV4A_MAX_T };
    const size_t N = (size_t)T * DSV4A_HD;
    float *xs = ARENA_ARRAY(&arena, float, N), *ref = ARENA_ARRAY(&arena, float, N);
    float *cs = ARENA_ARRAY(&arena, float, 16 * DSV4A_NPAIR * 2);
    uint16_t w[DSV4A_HD];
    sx_buffer *xb, *wb, *kb, *rb;
    EXPECT_OK(newbuf(&x, N * 2, &xb));
    EXPECT_OK(newbuf(&x, DSV4A_HD * 2, &wb));
    EXPECT_OK(newbuf(&x, N * 2, &kb));
    EXPECT_OK(newbuf(&x, DSV4A_WIN * DSV4A_HD * 2, &rb));
    static const uint32_t pos0s[] = {0, 124, 4000, 300001};
    double worst_frac = 0, worst_rel = 0;
    size_t fp8_diff = 0, total = 0;
    for (int theta = 0; theta < 2; theta++)
        for (size_t pi = 0; pi < COUNT(pos0s); pi++) {
            for (size_t i = 0; i < DSV4A_HD; i++) w[i] = dsv4a_bf16_bits(urand(&x, 0.2f, 3.0f));
            if (pi == 3) // weights of mixed magnitudes: FP8 blocks with small and large amax
                for (size_t i = 0; i < DSV4A_HD; i++) w[i] = dsv4a_bf16_bits(powf(10.0f, urand(&x, -5, 1)));
            memcpy(bits(wb), w, sizeof w);
            for (size_t t = 0; t < T; t++) {
                const float a = powf(10.0f, urand(&x, -2, 2));
                fill_bf16(&x, xs + t * DSV4A_HD, DSV4A_HD, -a, a);
            }
            // row 1: one block of zeros (amax floored at 1e-4)
            memset(xs + DSV4A_HD + 64, 0, 64 * 4);
            put_bf16(bits(xb), xs, N);
            const uint32_t rp0 = pos0s[pi];
            sx_buffer *csb;
            EXPECT_OK(rope_buf(&x, theta, rp0, 16, cs, &csb));
            const AttnP p = {pos0s[pi], T, rp0, 0, 0};
            Disp d[2] = {{"dsv4_attn_kv", 5, T, 1, 1, 128, {BUF(xb), BUF(wb), BUF(csb), BUF(kb), BYTES(&p)}},
                         {"dsv4_attn_kv_store", 3, T, 1, 1, 128, {BUF(kb), BUF(rb), BYTES(&p)}}};
            EXPECT_OK(run(&x, d, 2));
            for (size_t t = 0; t < T; t++)
                dsv4a_ref_kv(xs + t * DSV4A_HD, w, cs + t * DSV4A_NPAIR * 2, ref + t * DSV4A_HD);
            const uint16_t *kv = bits(kb);
            // rope dims within one BF16 step; nope dims are FP8 values times a power of two
            for (size_t t = 0; t < T; t++) {
                const uint16_t *kt = kv + t * DSV4A_HD;
                const float *rt = ref + t * DSV4A_HD;
                EXPECT(within_step(kt + DSV4A_NOPE, rt + DSV4A_NOPE, DSV4A_RD));
                for (size_t b = 0; b < DSV4A_NOPE; b += DSV4A_QBLK) {
                    float amax = 0;
                    for (size_t i = 0; i < DSV4A_QBLK; i++) amax = fmaxf(amax, fabsf(dsv4a_bf16_val(kt[b + i])));
                    const float s = dsv4a_act_scale(amax);
                    for (size_t i = 0; i < DSV4A_QBLK; i++) {
                        const float v = dsv4a_bf16_val(kt[b + i]);
                        EXPECT(dsv4a_e4m3_round(v / s) == v / s); // representable: GPU output is FP8 * scale
                        fp8_diff += v != rt[b + i];
                        total++;
                    }
                }
            }
            Cmp c = compare(kv, ref, N, DSV4A_QBLK);
            EXPECT(c.frac <= TOL_DSV4_ATTN_KV_FRAC);
            EXPECT(c.rel <= TOL_DSV4_ATTN_KV_REL);
            worst_frac = fmax(worst_frac, c.frac);
            worst_rel = fmax(worst_rel, c.rel);
            // the ring holds row t at slot (pos0 + t) % 128
            for (size_t t = 0; t < T; t++)
                EXPECT(memcmp(bits(rb) + ((pos0s[pi] + t) % DSV4A_WIN) * DSV4A_HD, kv + t * DSV4A_HD, DSV4A_HD * 2) == 0);
            sx_buffer_release(csb);
        }
    dbg("dsv4_attn_kv: values differing %.3g, rel to the FP8 block max %.3g, nope differing %zu of %zu\n", worst_frac,
        worst_rel, fp8_diff, total);
    arena_reset(&arena);
}

// Sparse attention cases. kind: 0 window only, 1 HCA (every visible entry, ascending),
// 2 CSA (min(512, visible) distinct entries in random order), 3 CSA with -1 entries mixed in.
// big scales q for logits of order 100, and puts the strongest key in a late block (or the
// first block when big == 2); sink: 0 random in [-2, 2], 1 large (+12), 2 very negative.
typedef struct {
    uint32_t pos0, T;
    int kind, big, sink;
} Case;

static uint32_t visible(int kind, uint32_t p) {
    if (kind == 0) return 0;
    if (kind == 1) return (p + 1) / 128;
    const uint32_t n = (p + 1) / 4;
    return n < 512 ? n : 512;
}


// Plain softmax attention in f64 (no BF16 probabilities, no blocks), BF16 output, then the
// de-rotation: what a kernel without kernel.py's probability rounding would give. The GPU
// must be much closer to the reference than this is, or the rounding is not implemented.
static void exact_attn(const float *q, const float *const *keys, size_t L, const float *sink, const float *cs, float *o,
                       double *s) {
    for (int h = 0; h < DSV4A_NH; h++) {
        const float *qh = q + (size_t)h * DSV4A_HD;
        double m = sink[h];
        for (size_t j = 0; j < L; j++) {
            s[j] = -INFINITY;
            if (!keys[j]) continue;
            double d = 0;
            for (int i = 0; i < DSV4A_HD; i++) d += (double)qh[i] * keys[j][i];
            s[j] = d * DSV4A_SCALE;
            m = fmax(m, s[j]);
        }
        double den = exp(sink[h] - m), acc[DSV4A_HD] = {0};
        for (size_t j = 0; j < L; j++) {
            if (!keys[j]) continue;
            const double e = exp(s[j] - m);
            den += e;
            for (int i = 0; i < DSV4A_HD; i++) acc[i] += e * keys[j][i];
        }
        for (int i = 0; i < DSV4A_HD; i++) o[(size_t)h * DSV4A_HD + i] = dsv4a_bf16((float)(acc[i] / den));
    }
    dsv4a_ref_derotate(o, cs);
}
static double rms_diff(const uint16_t *a, const float *b, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double)dsv4a_bf16_val(a[i]) - b[i];
        s += d * d;
    }
    return sqrt(s / (double)n);
}
static double rms_diff_f(const float *a, const float *b, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; i++) s += ((double)a[i] - b[i]) * ((double)a[i] - b[i]);
    return sqrt(s / (double)n);
}

TEST(dsv4_attn_sparse) {
    Arena arena = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&arena, &x));
    static const Case cases[] = {
        {0, 1, 0, 0, 0},     {5, 1, 0, 0, 0},     {63, 1, 0, 0, 1},    {64, 1, 0, 0, 0},  {126, 1, 0, 0, 0},
        {127, 1, 0, 0, 0},   {128, 1, 0, 0, 2},   {1000, 1, 0, 0, 0},  {0, 8, 0, 0, 0},   {124, 8, 0, 0, 0},
        {250, 5, 0, 0, 0},   {254, 4, 1, 0, 0},   {40000, 1, 1, 0, 0}, {16383, 2, 1, 1, 0}, {300, 1, 2, 0, 0},
        {5003, 1, 2, 0, 0},  {2050, 3, 2, 0, 1},  {9000, 8, 3, 0, 0},  {5003, 1, 2, 1, 0}, {5003, 1, 2, 2, 2},
        {200, 1, 0, 1, 0},   {131072, 2, 1, 1, 1}, {70001, 1, 2, 1, 0},
    };
    enum { MAXN = 1100, NCOMP = 2048 };
    const size_t QN = (size_t)DSV4A_MAX_T * DSV4A_NH * DSV4A_HD;
    float *q = ARENA_ARRAY(&arena, float, QN), *o = ARENA_ARRAY(&arena, float, DSV4A_NH * DSV4A_HD);
    float *ox = ARENA_ARRAY(&arena, float, DSV4A_NH * DSV4A_HD);
    double *sx = ARENA_ARRAY(&arena, double, DSV4A_WIN + MAXN);
    float *ring = ARENA_ARRAY(&arena, float, DSV4A_WIN * DSV4A_HD);
    float *chunk = ARENA_ARRAY(&arena, float, DSV4A_MAX_T * DSV4A_HD);
    float *comp = ARENA_ARRAY(&arena, float, (size_t)NCOMP * DSV4A_HD);
    float *vring = ARENA_ARRAY(&arena, float, DSV4A_WIN * DSV4A_HD);
    float *cs = ARENA_ARRAY(&arena, float, 16 * DSV4A_NPAIR * 2);
    const float **keys = ARENA_ARRAY(&arena, const float *, DSV4A_WIN + MAXN);
    int32_t *idx = ARENA_ARRAY(&arena, int32_t, DSV4A_MAX_T * MAXN);
    uint32_t cnt[DSV4A_MAX_T];
    float sink[DSV4A_NH];
    const uint32_t ls = ((DSV4A_WIN + MAXN + 63) / 64) * 64;
    sx_buffer *qb, *rb, *chb, *cb, *ib, *nb, *sb, *skb, *ob;
    EXPECT_OK(newbuf(&x, QN * 2, &qb));
    EXPECT_OK(newbuf(&x, DSV4A_WIN * DSV4A_HD * 2, &rb));
    EXPECT_OK(newbuf(&x, DSV4A_MAX_T * DSV4A_HD * 2, &chb));
    EXPECT_OK(newbuf(&x, (size_t)NCOMP * DSV4A_HD * 2, &cb));
    EXPECT_OK(newbuf(&x, DSV4A_MAX_T * MAXN * 4, &ib));
    EXPECT_OK(newbuf(&x, DSV4A_MAX_T * 4, &nb));
    EXPECT_OK(newbuf(&x, (size_t)DSV4A_MAX_T * DSV4A_NH * ls * 4, &sb));
    EXPECT_OK(newbuf(&x, DSV4A_NH * 4, &skb));
    EXPECT_OK(newbuf(&x, QN * 2, &ob));
    double worst_rel = 0, worst_frac = 0, worst_ratio = 0;
    for (size_t ci = 0; ci < COUNT(cases); ci++) {
        const Case *k = &cases[ci];
        const uint32_t T = k->T, pos0 = k->pos0;
        fill_bf16(&x, q, (size_t)T * DSV4A_NH * DSV4A_HD, -1.7f, 1.7f);
        fill_bf16(&x, ring, DSV4A_WIN * DSV4A_HD, -1.7f, 1.7f);
        fill_bf16(&x, chunk, T * DSV4A_HD, -1.7f, 1.7f);
        fill_bf16(&x, comp, (size_t)NCOMP * DSV4A_HD, -1.7f, 1.7f);
        for (int h = 0; h < DSV4A_NH; h++)
            sink[h] = k->sink == 0 ? urand(&x, -2, 2) : k->sink == 1 ? urand(&x, 10, 14) : urand(&x, -300, -100);
        // index lists: entries of the compressed cache (indices folded into NCOMP rows)
        uint32_t nmax = 0;
        for (uint32_t t = 0; t < T; t++) {
            const uint32_t p = pos0 + t, vis = k->kind == 0 ? 0 : k->kind == 1 ? (p + 1) / 128 : (p + 1) / 4;
            const uint32_t n = visible(k->kind, p);
            EXPECT(n <= MAXN);
            int32_t *row = idx + t * MAXN;
            if (k->kind == 1) {
                for (uint32_t i = 0; i < n; i++) row[i] = (int32_t)(i % NCOMP);
            } else if (n) {
                // n distinct of vis in random order (partial Fisher-Yates over a small pool)
                for (uint32_t i = 0; i < n; i++) row[i] = (int32_t)i;
                for (uint32_t i = n; i < vis && i < 4 * n; i++) {
                    const uint32_t j = (uint32_t)(qgen_prng_next(&x.rng) % (i + 1));
                    if (j < n) row[j] = (int32_t)i;
                }
                for (uint32_t i = n - 1; i > 0; i--) {
                    const uint32_t j = (uint32_t)(qgen_prng_next(&x.rng) % (i + 1));
                    const int32_t tmp = row[i];
                    row[i] = row[j];
                    row[j] = tmp;
                }
                for (uint32_t i = 0; i < n; i++) row[i] %= NCOMP;
                if (k->kind == 3)
                    for (uint32_t i = 0; i < n; i++)
                        if (qgen_prng_next(&x.rng) % 5 == 0) row[i] = -1;
            }
            cnt[t] = n;
            nmax = n > nmax ? n : nmax;
        }
        if (k->big) { // a strong key: comp entry (late block) or the oldest window entry (block 0)
            float *key = k->big == 1 && cnt[0] ? comp + (size_t)(idx[cnt[0] - 1] < 0 ? 0 : idx[cnt[0] - 1]) * DSV4A_HD
                                               : (pos0 >= 127 ? ring + ((pos0 - 127) % DSV4A_WIN) * DSV4A_HD : chunk);
            if (k->big == 1 && cnt[0] && idx[cnt[0] - 1] < 0) idx[cnt[0] - 1] = 0;
            for (size_t r = 0; r < (size_t)T * DSV4A_NH; r++)
                for (size_t i = 0; i < DSV4A_HD; i++)
                    q[r * DSV4A_HD + i] = dsv4a_bf16(q[r * DSV4A_HD + i] * 0.3f + key[i] * (float)(r % 8) * 1.2f);
        }
        put_bf16(bits(qb), q, (size_t)T * DSV4A_NH * DSV4A_HD);
        put_bf16(bits(rb), ring, DSV4A_WIN * DSV4A_HD);
        put_bf16(bits(chb), chunk, T * DSV4A_HD);
        put_bf16(bits(cb), comp, (size_t)NCOMP * DSV4A_HD);
        memcpy(sx_buffer_contents(ib), idx, DSV4A_MAX_T * MAXN * 4);
        memcpy(sx_buffer_contents(nb), cnt, sizeof cnt);
        memcpy(gpu_floats(skb), sink, sizeof sink);
        const uint32_t rp0 = pos0 >= 5 ? pos0 - 5 : 0;
        const int theta = k->kind != 0;
        sx_buffer *csb;
        EXPECT_OK(rope_buf(&x, theta, rp0, 16, cs, &csb));
        const AttnP p = {pos0, T, rp0, MAXN, ls};
        const size_t nbmax = (DSV4A_WIN + nmax + 63) / 64;
        Disp d[2] = {
            {"dsv4_attn_scores", 8, nbmax, DSV4A_NH / 16, T, 128,
             {BUF(qb), BUF(rb), BUF(chb), BUF(cb), BUF(ib), BUF(nb), BUF(sb), BYTES(&p)}},
            {"dsv4_attn_pv", 10, DSV4A_HD / 64, DSV4A_NH / 16, T, 128,
             {BUF(sb), BUF(rb), BUF(chb), BUF(cb), BUF(ib), BUF(nb), BUF(skb), BUF(csb), BUF(ob), BYTES(&p)}},
        };
        EXPECT_OK(run(&x, d, 2));
        for (uint32_t t = 0; t < T; t++) {
            const uint32_t pq = pos0 + t;
            // R's ring at this decode step: the old ring with the chunk's rows up to pq written
            memcpy(vring, ring, DSV4A_WIN * DSV4A_HD * 4);
            for (uint32_t u = pos0; u <= pq; u++)
                memcpy(vring + (u % DSV4A_WIN) * DSV4A_HD, chunk + (u - pos0) * DSV4A_HD, DSV4A_HD * 4);
            int32_t slots[DSV4A_WIN];
            dsv4a_ref_window_list(pq, slots);
            for (int j = 0; j < DSV4A_WIN; j++) keys[j] = slots[j] < 0 ? NULL : vring + (size_t)slots[j] * DSV4A_HD;
            for (uint32_t j = 0; j < cnt[t]; j++) {
                const int32_t e = idx[t * MAXN + j];
                keys[DSV4A_WIN + j] = e < 0 ? NULL : comp + (size_t)e * DSV4A_HD;
            }
            dsv4a_ref_sparse_attn(q + (size_t)t * DSV4A_NH * DSV4A_HD, keys, DSV4A_WIN + cnt[t], sink, o);
            dsv4a_ref_derotate(o, cs + (pq - rp0) * DSV4A_NPAIR * 2);
            const uint16_t *og = bits(ob) + (size_t)t * DSV4A_NH * DSV4A_HD;
            Cmp c = compare(og, o, DSV4A_NH * DSV4A_HD, DSV4A_HD);
            exact_attn(q + (size_t)t * DSV4A_NH * DSV4A_HD, keys, DSV4A_WIN + cnt[t], sink, cs + (pq - rp0) * DSV4A_NPAIR * 2,
                       ox, sx);
            const double e_gpu = rms_diff(og, o, DSV4A_NH * DSV4A_HD), e_exact = rms_diff_f(ox, o, DSV4A_NH * DSV4A_HD);
            const double ratio = e_exact > 0 ? e_gpu / e_exact : 0;
            if (!(c.rel <= TOL_DSV4_ATTN_O_REL && c.frac <= TOL_DSV4_ATTN_O_FRAC && ratio <= TOL_DSV4_ATTN_O_RMS_RATIO))
                dbg("case %zu row %u: rel %.3g, differing %.3g, rms ratio %.3g\n", ci, t, c.rel, c.frac, ratio);
            EXPECT(c.rel <= TOL_DSV4_ATTN_O_REL);
            EXPECT(c.frac <= TOL_DSV4_ATTN_O_FRAC);
            EXPECT(ratio <= TOL_DSV4_ATTN_O_RMS_RATIO);
            worst_rel = fmax(worst_rel, c.rel);
            worst_frac = fmax(worst_frac, c.frac);
            worst_ratio = fmax(worst_ratio, ratio);
        }
        sx_buffer_release(csb);
    }
    dbg("dsv4_attn_sparse: rel to the head's max %.3g, BF16 values differing %.3g, rms against the reference / rms "
        "of plain softmax %.3g\n",
        worst_rel, worst_frac, worst_ratio);
    arena_reset(&arena);
}

TEST(dsv4_attn_wo_a) {
    Arena arena = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&arena, &x));
    const size_t WN = (size_t)DSV4A_WOA_ROWS * DSV4A_GD, ON = (size_t)DSV4A_NH * DSV4A_HD;
    sx_buffer *wb, *ob, *yb;
    EXPECT_OK(newbuf(&x, WN * 2, &wb));
    EXPECT_OK(newbuf(&x, DSV4A_MAX_T * ON * 2, &ob));
    EXPECT_OK(newbuf(&x, DSV4A_MAX_T * DSV4A_WOA_ROWS * 2, &yb));
    uint16_t *w = bits(wb);
    for (size_t i = 0; i < WN; i++) w[i] = dsv4a_bf16_bits(urand(&x, -0.05f, 0.05f));
    float *o = ARENA_ARRAY(&arena, float, DSV4A_MAX_T * ON), *y = ARENA_ARRAY(&arena, float, DSV4A_WOA_ROWS);
    double worst_rel = 0, worst_frac = 0;
    static const uint32_t Ts[] = {1, 3, 8};
    for (size_t ti = 0; ti < COUNT(Ts); ti++) {
        const uint32_t T = Ts[ti];
        for (size_t t = 0; t < T; t++) fill_bf16(&x, o + t * ON, ON, -2, 2);
        put_bf16(bits(ob), o, T * ON);
        const AttnP p = {0, T, 0, 0, 0};
        Disp d = {"dsv4_attn_wo_a", 4, DSV4A_WOA_ROWS / 8, 1, 1, 256, {BUF(ob), BUF(wb), BUF(yb), BYTES(&p)}};
        EXPECT_OK(run(&x, &d, 1));
        for (size_t t = 0; t < T; t++) {
            if (ti == 2 && t > 1 && t < 7) continue; // T = 8: the first two rows and the last
            dsv4a_ref_wo_a(o + t * ON, w, y);
            const uint16_t *yg = bits(yb) + t * DSV4A_WOA_ROWS;
            EXPECT(within_step(yg, y, DSV4A_WOA_ROWS));
            Cmp c = compare(yg, y, DSV4A_WOA_ROWS, DSV4A_WOA_ROWS);
            EXPECT(c.frac <= TOL_DSV4_ATTN_WOA_FRAC);
            worst_rel = fmax(worst_rel, c.rel);
            worst_frac = fmax(worst_frac, c.frac);
        }
    }
    dbg("dsv4_attn_wo_a: BF16 values differing %.3g (within one step), rel %.3g\n", worst_frac, worst_rel);
    arena_reset(&arena);
}
