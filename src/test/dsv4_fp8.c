// DeepSeek-V4-Flash FP8 linears (src/kernels/dsv4_fp8.metal) against the scalar references
// (src/dsv4_ref_fp8.c), on random e4m3 weights and e8m0 scales at the model's shapes. No pack.
#include <math.h>
#include <stdlib.h>

#include "../dsv4_fp8.h"
#include "../dsv4_ref_fp8.h"
#include "../ops.h"
#include "../qgen.h"
#include "../test.h"
#include "../tolerances.h"

// f32 to the nearest bf16 (ties to even), back as f32: the model's activations are bf16.
static float bf16_round(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    memcpy(&f, &u, 4);
    return f;
}

static uint32_t rnd(QgenPrng *r, uint32_t n) { return (uint32_t)(qgen_prng_next(r) % n); }

TEST(dsv4_fp8_codecs) {
    EXPECT(dsv4_e4m3_to_f32(0x7e) == 448.0f);
    EXPECT(dsv4_e4m3_to_f32(0xfe) == -448.0f);
    EXPECT(dsv4_e4m3_to_f32(0x38) == 1.0f);
    EXPECT(dsv4_e4m3_to_f32(0x01) == 0x1p-9f);
    EXPECT(dsv4_e4m3_to_f32(0x07) == 7 * 0x1p-9f);
    EXPECT(dsv4_e4m3_to_f32(0x08) == 0x1p-6f);
    EXPECT(isnan(dsv4_e4m3_to_f32(0x7f)) && isnan(dsv4_e4m3_to_f32(0xff)));
    EXPECT(dsv4_e8m0_to_f32(127) == 1.0f && dsv4_e8m0_to_f32(120) == 0x1p-7f && dsv4_e8m0_to_f32(130) == 8.0f);
    // every finite code survives encode(decode)
    for (int b = 0; b < 256; b++)
        if ((b & 0x7f) != 0x7f) EXPECT_EQ(dsv4_f32_to_e4m3(dsv4_e4m3_to_f32((uint8_t)b)), b);
    // ties to even, normal and subnormal; saturation
    EXPECT_EQ(dsv4_f32_to_e4m3(1.0625f), 0x38);       // between 1 and 1.125: to 1 (even)
    EXPECT_EQ(dsv4_f32_to_e4m3(1.1875f), 0x3a);       // between 1.125 and 1.25: to 1.25
    EXPECT_EQ(dsv4_f32_to_e4m3(1.5f * 0x1p-9f), 0x02); // between 1 and 2 (x 2^-9): to 2
    EXPECT_EQ(dsv4_f32_to_e4m3(0.5f * 0x1p-9f), 0x00);
    EXPECT_EQ(dsv4_f32_to_e4m3(7.5f * 0x1p-9f), 0x08); // to 2^-6, the first normal
    EXPECT_EQ(dsv4_f32_to_e4m3(432.0f), 0x7e);         // between 416 and 448: to 448 (even)
    EXPECT_EQ(dsv4_f32_to_e4m3(1000.0f), 0x7e);
    EXPECT_EQ(dsv4_f32_to_e4m3(-0.0f), 0x80);
    // scales: 2^ceil(log2(amax / 448)), amax floored at 1e-4
    EXPECT(dsv4_act_scale(448.0f) == 1.0f);
    EXPECT(dsv4_act_scale(449.0f) == 2.0f);
    EXPECT(dsv4_act_scale(224.0f) == 0.5f);
    EXPECT(dsv4_act_scale(0.0f) == 0x1p-22f); // 1e-4 / 448 = 2.2e-7, in (2^-23, 2^-22]
}

// Random inputs with the ranges that matter to act_quant: per block magnitudes 2^-10 .. 2^10,
// outliers, all-zero blocks, blocks whose amax is exactly 448 * 2^j, values on e4m3 ties.
static void fill_x(QgenPrng *r, float *x, size_t T, size_t K) {
    for (size_t t = 0; t < T; t++)
        for (size_t b = 0; b < K / 128; b++) {
            float *xb = x + t * K + b * 128;
            const uint32_t kind = rnd(r, 16);
            const float mag = ldexpf(1.0f, (int)rnd(r, 21) - 10);
            for (size_t i = 0; i < 128; i++) {
                float v = (qgen_prng_float(r) * 2 - 1) * mag;
                if (rnd(r, 64) == 0) v *= 30.0f;
                xb[i] = kind == 0 ? 0.0f : bf16_round(v);
            }
            if (kind == 1) xb[rnd(r, 128)] = 448.0f * mag; // amax a power-of-two multiple of 448
            if (kind == 2) {                               // ties: midpoints of e4m3 codes times the scale
                xb[0] = 448.0f * mag;
                for (size_t i = 1; i < 128; i++) {
                    const uint8_t c = (uint8_t)rnd(r, 0x7e);
                    const float lo = dsv4_e4m3_to_f32(c), hi = dsv4_e4m3_to_f32((uint8_t)(c + 1));
                    xb[i] = bf16_round((lo + hi) * 0.5f * mag * (rnd(r, 2) ? 1.0f : -1.0f));
                }
            }
        }
}

// A random e4m3 weight [N][K] (no NaN codes) at byte wshift of its buffer, e8m0 scales
// 2^-13 .. 2^-6 at byte sshift.
typedef struct {
    sx_buffer *wb, *sb;
    uint8_t *w, *s;
    Dsv4Fp8Lin l;
} Weight;

static Err new_weight(QgenCtx *x, size_t N, size_t K, uint32_t wshift, uint32_t sshift, Weight *o) {
    TRY(gpu_new_buffer(&x->g, N * K + wshift + 16, &o->wb));
    const size_t ns = (N + 127) / 128 * (K / 128);
    TRY(gpu_new_buffer(&x->g, ns + sshift + 16, &o->sb));
    o->w = (uint8_t *)sx_buffer_contents(o->wb) + wshift;
    o->s = (uint8_t *)sx_buffer_contents(o->sb) + sshift;
    qgen_prng_bytes(&x->rng, o->w, N * K);
    for (size_t i = 0; i < N * K; i++)
        if ((o->w[i] & 0x7f) == 0x7f) o->w[i] ^= 0x01; // NaN to 0x7e (448)
    for (size_t i = 0; i < ns; i++) o->s[i] = (uint8_t)(127 - 6 - rnd(&x->rng, 8));
    o->l = (Dsv4Fp8Lin){gpu_buf_arg(o->wb), gpu_buf_arg(o->sb), wshift, sshift, (uint32_t)N, (uint32_t)K};
    return OK;
}

static void free_weight(Weight *w) {
    sx_buffer_release(w->wb);
    sx_buffer_release(w->sb);
}

static Err finish(sx_cmd *c) {
    char err[512];
    if (sx_cmd_finish(c, err, sizeof err) != 0) {
        dbg("dispatch: %s\n", err);
        return "DispatchFailed";
    }
    return OK;
}

// Error of y (GPU) against ref for one token row, two measures: max |diff| / max |ref| (the
// tolerance) and max |diff| / (2^-24 sum_k |terms|) (the f32 rounding budget, in units).
typedef struct {
    float rel;
    double units;
} ErrStat;

static void row_err(const float *y, const float *ref, const double *mag, size_t N, ErrStat *e) {
    e->rel = fmaxf(e->rel, ops_max_rel_err(y, ref, N));
    for (size_t n = 0; n < N; n++) {
        const double d = fabs((double)y[n] - (double)ref[n]);
        if (mag[n] > 0) e->units = fmax(e->units, d / (mag[n] * 0x1p-24));
        else if (d > 0) e->units = INFINITY;
    }
}

// sum_k |code(w) code(x) sa sb| per row n, for one token.
static void mag_q(const Weight *w, const uint8_t *codes, const float *sc, double *mag) {
    const size_t N = w->l.out, K = w->l.in, nb = K / 128;
    for (size_t n = 0; n < N; n++) {
        double m = 0;
        for (size_t k = 0; k < K; k++)
            m += fabs((double)dsv4_e4m3_to_f32(w->w[n * K + k]) * dsv4_e4m3_to_f32(codes[k]) * sc[k / 128] *
                      dsv4_e8m0_to_f32(w->s[(n / 128) * nb + k / 128]));
        mag[n] = m;
    }
}

static const size_t Ts[] = {1, 2, 8, 100, 512}; // 100: a partial matmul tile

// Tokens checked against the reference: all for T <= 8, else a sample across tiles.
static size_t sample_tokens(size_t T, size_t *ts) {
    if (T <= 8) {
        for (size_t i = 0; i < T; i++) ts[i] = i;
        return T;
    }
    static const size_t s[] = {0, 1, 63, 64, 99, 255, 448, 511};
    size_t n = 0;
    for (size_t i = 0; i < COUNT(s); i++)
        if (s[i] < T) ts[n++] = s[i];
    return n;
}

TEST(dsv4_fp8_act_quant_bit_exact) {
    Arena arena = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&arena, &x));
    Dsv4Fp8Pipes p;
    EXPECT_OK(dsv4_fp8_init(&x.g, &p));
    const size_t T = 64, K = 8192;
    sx_buffer *xb, *xcb, *xsb;
    EXPECT_OK(gpu_new_buffer(&x.g, T * K * 4, &xb));
    EXPECT_OK(gpu_new_buffer(&x.g, T * K * 2, &xcb));
    EXPECT_OK(gpu_new_buffer(&x.g, T * K / 128 * 4, &xsb));
    fill_x(&x.rng, gpu_floats(xb), T, K);
    sx_cmd *c = sx_cmd_begin(x.g.dev);
    dsv4_encode_act_quant(&p, c, gpu_buf_arg(xb), (uint32_t)K, gpu_buf_arg(xcb), gpu_buf_arg(xsb), (uint32_t)T, (uint32_t)K);
    EXPECT_OK(finish(c));
    uint8_t *codes = ARENA_ARRAY(&arena, uint8_t, T * K);
    float *sc = ARENA_ARRAY(&arena, float, T * K / 128);
    dsv4_ref_act_quant(gpu_floats(xb), T, K, K, codes, sc);
    const _Float16 *xc = sx_buffer_contents(xcb);
    const float *xs = gpu_floats(xsb);
    size_t ties = 0;
    for (size_t i = 0; i < T * K / 128; i++) EXPECT(xs[i] == sc[i]);
    for (size_t i = 0; i < T * K; i++) {
        const float g = (float)xc[i], r = dsv4_e4m3_to_f32(codes[i]);
        if (g != r) {
            dbg("act_quant: element %zu x %g scale %g: GPU %g, reference %g\n", i, (double)gpu_floats(xb)[i],
                (double)sc[i / 128], (double)g, (double)r);
            EXPECT(g == r);
        }
        const float q = gpu_floats(xb)[i] / sc[i / 128];
        const uint8_t cq = dsv4_f32_to_e4m3(q);
        const float a = dsv4_e4m3_to_f32(cq);
        if (a != q) { // count inputs exactly halfway between two codes
            const uint8_t m = cq & 0x7f;
            const float up = m < 0x7e ? dsv4_e4m3_to_f32((uint8_t)(cq + 1)) : a;
            const float dn = m > 0 ? dsv4_e4m3_to_f32((uint8_t)(cq - 1)) : a;
            if (fabsf(q - a) == fabsf(q - up) || fabsf(q - a) == fabsf(q - dn)) ties++;
        }
    }
    dbg("act_quant: %zu values, %zu on ties, codes and scales bit-exact\n", T * K, ties);
    EXPECT(ties > 1000);
    sx_buffer_release(xb);
    sx_buffer_release(xcb);
    sx_buffer_release(xsb);
    arena_reset(&arena);
}

// The reasoning behind f32 per-block sums: a product of two e4m3 codes has at most 8
// significant bits and lies in [2^-18, 200704], so each product is exact in f32 (and in the
// accelerators' f32 accumulation of half x half); only the 127 additions of a block round.
// Measures the f32 block sum against the exact (f64) one over random and worst-range blocks.
TEST(dsv4_fp8_block_sums_in_f32) {
    QgenPrng r;
    qgen_prng_init(&r, 7);
    double worst = 0, worst_rel_out = 0;
    for (int it = 0; it < 200000; it++) {
        uint8_t a[128], b[128];
        qgen_prng_bytes(&r, a, 128);
        qgen_prng_bytes(&r, b, 128);
        float s = 0;
        double d = 0, m = 0;
        for (int i = 0; i < 128; i++) {
            if ((a[i] & 0x7f) == 0x7f) a[i] ^= 1;
            if ((b[i] & 0x7f) == 0x7f) b[i] ^= 1;
            const float pa = dsv4_e4m3_to_f32(a[i]), pb = dsv4_e4m3_to_f32(b[i]);
            const float pr = pa * pb;
            if ((double)pr != (double)pa * (double)pb) {
                dbg("product of codes 0x%02x 0x%02x not exact in f32\n", a[i], b[i]);
                *failed_ = 1;
                return;
            }
            s += pr;
            d += (double)pa * (double)pb;
            m += fabs((double)pa * (double)pb);
        }
        worst = fmax(worst, fabs((double)s - d) / (m * 0x1p-24));
        if (fabs(d) > 0) worst_rel_out = fmax(worst_rel_out, fabs((double)s - d) / fabs(d));
    }
    dbg("block sums: worst f32 error %.2f units of 2^-24 sum|p| (bound 127)\n", worst);
    EXPECT(worst <= 127);
}

// One linear: act_quant on the GPU (checked equal to the reference's codes), then the path
// chosen by T (or forced: mm 0 matvec, 1 matmul, -1 by T), against dsv4_ref_fp8_gemm on the
// reference's codes.
static Err check_linear(Arena *a, QgenCtx *x, const Dsv4Fp8Pipes *p, const Weight *w, size_t T, int force,
                        ErrStat *es) {
    const size_t N = w->l.out, K = w->l.in, nb = K / 128;
    sx_buffer *xb, *xcb, *xsb, *yb;
    TRY(gpu_new_buffer(&x->g, T * K * 4, &xb));
    TRY(gpu_new_buffer(&x->g, T * K * 2, &xcb));
    TRY(gpu_new_buffer(&x->g, T * nb * 4, &xsb));
    TRY(gpu_new_buffer(&x->g, T * N * 4, &yb));
    fill_x(&x->rng, gpu_floats(xb), T, K);
    sx_cmd *c = sx_cmd_begin(x->g.dev);
    dsv4_encode_act_quant(p, c, gpu_buf_arg(xb), (uint32_t)K, gpu_buf_arg(xcb), gpu_buf_arg(xsb), (uint32_t)T, (uint32_t)K);
    if (force < 0) dsv4_encode_linear(p, c, &w->l, gpu_buf_arg(xcb), gpu_buf_arg(xsb), gpu_buf_arg(yb), (uint32_t)T, (uint32_t)N);
    else if (force == 0)
        dsv4_encode_mv(p, c, &w->l, false, gpu_buf_arg(xcb), gpu_buf_arg(xsb), (uint32_t)K, (uint32_t)N, gpu_buf_arg(yb),
                       (uint32_t)T, (uint32_t)N);
    else
        dsv4_encode_mm(p, c, &w->l, false, gpu_buf_arg(xcb), gpu_buf_arg(xsb), (uint32_t)K, (uint32_t)N, gpu_buf_arg(yb),
                       (uint32_t)T, (uint32_t)N);
    TRY(finish(c));
    uint8_t *codes = ARENA_ARRAY(a, uint8_t, T * K);
    float *sc = ARENA_ARRAY(a, float, T * nb);
    dsv4_ref_act_quant(gpu_floats(xb), T, K, K, codes, sc);
    const _Float16 *xc = sx_buffer_contents(xcb);
    for (size_t i = 0; i < T * K; i++)
        if ((float)xc[i] != dsv4_e4m3_to_f32(codes[i])) return "ActQuantMismatch";
    for (size_t i = 0; i < T * nb; i++)
        if (gpu_floats(xsb)[i] != sc[i]) return "ActQuantMismatch";
    float *ref = ARENA_ARRAY(a, float, N);
    double *mag = ARENA_ARRAY(a, double, N);
    size_t ts[8];
    const size_t nts = sample_tokens(T, ts);
    for (size_t i = 0; i < nts; i++) {
        const size_t t = ts[i];
        dsv4_ref_fp8_gemm(w->w, w->s, N, K, codes + t * K, sc + t * nb, 1, ref, N);
        mag_q(w, codes + t * K, sc + t * nb, mag);
        row_err(gpu_floats(yb) + t * N, ref, mag, N, es);
    }
    sx_buffer_release(xb);
    sx_buffer_release(xcb);
    sx_buffer_release(xsb);
    sx_buffer_release(yb);
    return OK;
}

TEST(dsv4_fp8_linear_matches_the_reference) {
    Arena arena = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&arena, &x));
    Dsv4Fp8Pipes p;
    EXPECT_OK(dsv4_fp8_init(&x.g, &p));
    static const struct {
        const char *name;
        size_t out, in;
    } shapes[] = {{"wq_a", 1024, 4096},          {"wq_b", 32768, 1024},       {"wkv", 512, 4096},
                  {"wo_b", 4096, 8192},          {"indexer.wq_b", 8192, 1024}, {"shared w1/w3", 2048, 4096},
                  {"shared w2", 4096, 2048}};
    Arena scratch = {0};
    ErrStat worst_mv = {0}, worst_mm = {0};
    for (size_t si = 0; si < COUNT(shapes); si++) {
        Weight w;
        // odd shapes bound at unaligned bytes, as pack tensors can be
        EXPECT_OK(new_weight(&x, shapes[si].out, shapes[si].in, si % 2 ? 3 : 0, si % 2 ? 5 : 0, &w));
        for (size_t ti = 0; ti < COUNT(Ts); ti++) {
            ErrStat e = {0};
            EXPECT_OK(check_linear(&scratch, &x, &p, &w, Ts[ti], -1, &e));
            dbg("%-13s [%5zu, %4zu] T %3zu %s: max rel err %.3g, %.2f units of 2^-24 sum|terms|\n", shapes[si].name,
                shapes[si].out, shapes[si].in, Ts[ti], Ts[ti] <= DSV4_FP8_SMALL_T ? "mv" : "mm", (double)e.rel, e.units);
            const size_t bound_units = 128 + shapes[si].in / 128 + 64;
            EXPECT(e.rel < TOL_DSV4_FP8_REL);
            EXPECT(e.units < (double)bound_units);
            ErrStat *wst = Ts[ti] <= DSV4_FP8_SMALL_T ? &worst_mv : &worst_mm;
            wst->rel = fmaxf(wst->rel, e.rel);
            wst->units = fmax(wst->units, e.units);
        }
        if (si == 0) { // the other path at the same T: matmul at T 1 and 8, matvec passes at T 12
            ErrStat e = {0};
            EXPECT_OK(check_linear(&scratch, &x, &p, &w, 1, 1, &e));
            EXPECT_OK(check_linear(&scratch, &x, &p, &w, 8, 1, &e));
            EXPECT_OK(check_linear(&scratch, &x, &p, &w, 12, 0, &e));
            dbg("wq_a forced paths: max rel err %.3g, %.2f units\n", (double)e.rel, e.units);
            EXPECT(e.rel < TOL_DSV4_FP8_REL);
        }
        free_weight(&w);
        arena_reset(&scratch);
    }
    arena_reset(&arena);
    dbg("dsv4 fp8 linear: worst max rel err %.3g (mv), %.3g (mm); %.2f, %.2f units\n", (double)worst_mv.rel,
        (double)worst_mm.rel, worst_mv.units, worst_mm.units);
}

// wo_a: [8 groups x 1024, 4096] on the raw f32 (bf16-valued) attention output [T][8 x 4096].
TEST(dsv4_fp8_wo_a_grouped_matches_the_reference) {
    Arena arena = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&arena, &x));
    Dsv4Fp8Pipes p;
    EXPECT_OK(dsv4_fp8_init(&x.g, &p));
    const size_t G = 8, N = 8192, K = 4096, ldx = G * K, nb = K / 128;
    Weight w;
    EXPECT_OK(new_weight(&x, N, K, 1, 2, &w));
    ErrStat worst_mv = {0}, worst_mm = {0};
    for (size_t ti = 0; ti < COUNT(Ts); ti++) {
        const size_t T = Ts[ti];
        sx_buffer *xb, *yb;
        EXPECT_OK(gpu_new_buffer(&x.g, T * ldx * 4, &xb));
        EXPECT_OK(gpu_new_buffer(&x.g, T * N * 4, &yb));
        float *xv = gpu_floats(xb);
        for (size_t i = 0; i < T * ldx; i++) xv[i] = bf16_round((qgen_prng_float(&x.rng) * 2 - 1) * 0.5f);
        sx_cmd *c = sx_cmd_begin(x.g.dev);
        dsv4_encode_grouped(&p, c, &w.l, (uint32_t)G, gpu_buf_arg(xb), gpu_buf_arg(yb), (uint32_t)T, (uint32_t)N);
        EXPECT_OK(finish(c));
        float *ref = ARENA_ARRAY(&arena, float, N);
        double *mag = ARENA_ARRAY(&arena, double, N);
        size_t ts[8];
        const size_t nts = sample_tokens(T, ts);
        ErrStat e = {0};
        for (size_t i = 0; i < nts; i++) {
            const size_t t = ts[i];
            dsv4_ref_grouped(w.w, w.s, N, K, G, xv + t * ldx, ldx, 1, ref, N);
            for (size_t n = 0; n < N; n++) {
                double m = 0;
                for (size_t k = 0; k < K; k++)
                    m += fabs((double)dsv4_e4m3_to_f32(w.w[n * K + k]) * dsv4_e8m0_to_f32(w.s[(n / 128) * nb + k / 128]) *
                              xv[t * ldx + (n / (N / G)) * K + k]);
                mag[n] = m;
            }
            row_err(gpu_floats(yb) + t * N, ref, mag, N, &e);
        }
        dbg("wo_a grouped T %3zu %s: max rel err %.3g, %.2f units of 2^-24 sum|terms|\n", T,
            T <= DSV4_FP8_SMALL_T ? "mv" : "mm", (double)e.rel, e.units);
        EXPECT(e.rel < TOL_DSV4_FP8_REL);
        EXPECT(e.units < (double)(K + 64));
        ErrStat *wst = T <= DSV4_FP8_SMALL_T ? &worst_mv : &worst_mm;
        wst->rel = fmaxf(wst->rel, e.rel);
        wst->units = fmax(wst->units, e.units);
        sx_buffer_release(xb);
        sx_buffer_release(yb);
    }
    dbg("wo_a grouped: worst max rel err %.3g (mv), %.3g (mm); %.2f, %.2f units\n", (double)worst_mv.rel,
        (double)worst_mm.rel, worst_mv.units, worst_mm.units);
    free_weight(&w);
    arena_reset(&arena);
}
