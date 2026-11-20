// DeepSeek-V4-Flash routed-expert kernels (src/kernels/dsv4_moe.metal) against the scalar
// reference (src/dsv4_ref.c), on random FP4 experts in a simulated expert cache.
//
// Each stage is checked on the GPU's own input to that stage, so a rounding decision taken
// differently upstream (a bf16 or e4m3 tie broken the other way after a different f32
// summation order) does not compound into the next stage's check:
//   act_quant(x)                   bit-exact
//   h (gate, up, SwiGLU, weight)   against the reference on the same xq
//   hq = act_quant(h)              bit-exact against the reference applied to the GPU's h
//   y (down, sum of 6)             against the reference on the GPU's hq
// and then end to end against dsv4_moe_ref, where an e4m3 code of hq may differ.
#include <math.h>
#include <stdlib.h>

#include "../dsv4_moe.h"
#include "../dsv4_ref.h"
#include "../gpu.h"
#include "../qgen.h"
#include "../test.h"
#include "../tolerances.h"

enum { NSLOT = 12 };
#define WSTRIDE (DSV4_EXPERT_W_BYTES + 16384u) // not the bare expert size, as a cache may pad
#define SSTRIDE (DSV4_EXPERT_S_BYTES + 16384u)

typedef struct {
    Arena arena;
    Gpu g;
    QgenPrng rng;
    sx_buffer *W, *S;
    sx_pipeline *aq, *gu, *dn;
    bool ready;
} Ctx;
static Ctx C;

static Err ctx_get(void) {
    if (C.ready) return OK;
    Pack empty = {0};
    TRY(gpu_init(&C.arena, &empty, &C.g));
    qgen_prng_init(&C.rng, 0xd5e4f4);
    TRY(gpu_new_buffer(&C.g, (size_t)NSLOT * WSTRIDE, &C.W));
    TRY(gpu_new_buffer(&C.g, (size_t)NSLOT * SSTRIDE, &C.S));
    qgen_prng_bytes(&C.rng, sx_buffer_contents(C.W), (size_t)NSLOT * WSTRIDE);
    uint8_t *s = sx_buffer_contents(C.S);
    // scales 2^-13 .. 2^-3: outputs from about 0.1 to well past the clamp at 10
    for (size_t i = 0; i < (size_t)NSLOT * SSTRIDE; i++) s[i] = (uint8_t)(114 + qgen_prng_next(&C.rng) % 11);
    TRY(gpu_pipeline(&C.g, dsv4_moe_src, "dsv4_act_quant", &C.aq));
    TRY(gpu_pipeline(&C.g, dsv4_moe_src, "dsv4_moe_gate_up", &C.gu));
    TRY(gpu_pipeline(&C.g, dsv4_moe_src, "dsv4_moe_down", &C.dn));
    C.ready = true;
    return OK;
}

static Dsv4Expert expert_at(uint32_t slot) {
    const uint8_t *w = (const uint8_t *)sx_buffer_contents(C.W) + (size_t)slot * WSTRIDE;
    const uint8_t *s = (const uint8_t *)sx_buffer_contents(C.S) + (size_t)slot * SSTRIDE;
    return (Dsv4Expert){w + DSV4_W1_OFF, s + DSV4_S1_OFF, w + DSV4_W2_OFF, s + DSV4_S2_OFF, w + DSV4_W3_OFF, s + DSV4_S3_OFF};
}

static float frand(void) { return qgen_prng_float(&C.rng); }

// Largest |got - ref| relative to the largest |ref|, and the count of values not bit-identical.
typedef struct {
    double rel;
    size_t diff, n;
} Cmp;
static void cmp_add(Cmp *c, const float *got, const float *ref, size_t n) {
    float m = 1e-30f, e = 0;
    for (size_t i = 0; i < n; i++) {
        m = fmaxf(m, fabsf(ref[i]));
        e = fmaxf(e, fabsf(got[i] - ref[i]));
        if (memcmp(&got[i], &ref[i], 4) != 0) c->diff++;
    }
    c->rel = fmax(c->rel, (double)(e / m));
    c->n += n;
}

typedef struct {
    Cmp h, y, e2e;
    size_t hq_codes; // e4m3 codes of hq that differ between the GPU path and the reference path
} Stats;

static void run_case(int *failed_, size_t T, const uint32_t *slots, Stats *st) {
    EXPECT_OK(ctx_get());
    const size_t np = T * DSV4_TOPK;
    sx_buffer *xb, *xqb, *sb, *rwb, *hqb, *hrb, *yb;
    EXPECT_OK(gpu_new_buffer(&C.g, T * DSV4_DIM * 4, &xb));
    EXPECT_OK(gpu_new_buffer(&C.g, T * DSV4_DIM * 4, &xqb));
    EXPECT_OK(gpu_new_buffer(&C.g, np * 4, &sb));
    EXPECT_OK(gpu_new_buffer(&C.g, np * 4, &rwb));
    EXPECT_OK(gpu_new_buffer(&C.g, np * DSV4_INTER * 4, &hqb));
    EXPECT_OK(gpu_new_buffer(&C.g, np * DSV4_INTER * 4, &hrb));
    EXPECT_OK(gpu_new_buffer(&C.g, T * DSV4_DIM * 4, &yb));
    float *x = gpu_floats(xb), *rw = gpu_floats(rwb);
    memcpy(sx_buffer_contents(sb), slots, np * 4);
    for (size_t i = 0; i < T * DSV4_DIM; i++) x[i] = dsv4_bf16_round(frand() * 4 - 2);
    for (size_t i = 0; i < 128; i++) x[3 * 128 + i] = 0;                                         // amax floor
    for (size_t i = 0; i < 128; i++) x[5 * 128 + i] = dsv4_bf16_round((frand() - 0.5f) * 1e-5f); // below the floor
    for (size_t i = 0; i < 128; i++) x[(T - 1) * DSV4_DIM + 7 * 128 + i] = dsv4_bf16_round((frand() - 0.5f) * 600);
    for (size_t t = 0; t < T; t++) {
        float sum = 0;
        for (size_t k = 0; k < DSV4_TOPK; k++) sum += rw[t * DSV4_TOPK + k] = 0.05f + frand();
        for (size_t k = 0; k < DSV4_TOPK; k++) rw[t * DSV4_TOPK + k] = rw[t * DSV4_TOPK + k] / sum * 1.5f;
    }

    Dsv4MoeP P = {(uint32_t)T, WSTRIDE, SSTRIDE, 1, {DSV4_W1_OFF, DSV4_W2_OFF, DSV4_W3_OFF, 0}, {DSV4_S1_OFF, DSV4_S2_OFF, DSV4_S3_OFF, 0}};
    const uint32_t K = DSV4_DIM;
    sx_cmd *cmd = sx_cmd_begin(C.g.dev);
    EXPECT(cmd);
    sx_arg a0[] = {gpu_buf_arg(xb), gpu_buf_arg(xqb), gpu_u32_arg(&K)};
    sx_cmd_dispatch(cmd, C.aq, a0, COUNT(a0), DSV4_DIM / 128, T, 32);
    sx_arg a1[] = {gpu_buf_arg(C.W), gpu_buf_arg(C.S), gpu_buf_arg(xqb), gpu_buf_arg(sb), gpu_buf_arg(rwb),
                   gpu_buf_arg(hqb), gpu_buf_arg(hrb), gpu_bytes_arg(&P, sizeof P)};
    sx_cmd_dispatch(cmd, C.gu, a1, COUNT(a1), DSV4_INTER / 128, np, 256);
    sx_arg a2[] = {gpu_buf_arg(C.W), gpu_buf_arg(C.S), gpu_buf_arg(hqb), gpu_buf_arg(sb), gpu_buf_arg(yb), gpu_bytes_arg(&P, sizeof P)};
    sx_cmd_dispatch(cmd, C.dn, a2, COUNT(a2), DSV4_DIM / 16, 1, 256);
    char err[256];
    EXPECT(sx_cmd_finish(cmd, err, sizeof err) == 0);

    const float *xq = gpu_floats(xqb), *hq = gpu_floats(hqb), *hr = gpu_floats(hrb), *y = gpu_floats(yb);
    float *ref = malloc(DSV4_DIM * 4), *h = malloc(DSV4_INTER * 4), *d = malloc(DSV4_DIM * 4);
    float *yref = calloc(T * DSV4_DIM, 4), *ye2e = calloc(T * DSV4_DIM, 4);
    EXPECT(ref && h && d && yref && ye2e);
    for (size_t t = 0; t < T; t++) {
        dsv4_act_quant(x + t * DSV4_DIM, ref, DSV4_DIM, DSV4_ACT_BLOCK);
        EXPECT(memcmp(ref, xq + t * DSV4_DIM, DSV4_DIM * 4) == 0);
    }
    for (size_t p = 0; p < np; p++) {
        const size_t t = p / DSV4_TOPK;
        const Dsv4Expert e = expert_at(slots[p]);
        dsv4_gate_up_ref(&e, xq + t * DSV4_DIM, rw[p], h);
        cmp_add(&st->h, hr + p * DSV4_INTER, h, DSV4_INTER);
        float hq_gpu_ref[DSV4_INTER];
        dsv4_act_quant(hr + p * DSV4_INTER, hq_gpu_ref, DSV4_INTER, DSV4_ACT_BLOCK);
        EXPECT(memcmp(hq_gpu_ref, hq + p * DSV4_INTER, DSV4_INTER * 4) == 0);
        dsv4_down_ref(&e, hq + p * DSV4_INTER, d);
        for (size_t i = 0; i < DSV4_DIM; i++) yref[t * DSV4_DIM + i] += d[i];
        // the reference path: its own h quantized; down again only where hq differs
        dsv4_act_quant(h, h, DSV4_INTER, DSV4_ACT_BLOCK);
        size_t nd = 0;
        for (size_t i = 0; i < DSV4_INTER; i++) nd += memcmp(&h[i], &hq[p * DSV4_INTER + i], 4) != 0;
        st->hq_codes += nd;
        if (nd) dsv4_down_ref(&e, h, d);
        for (size_t i = 0; i < DSV4_DIM; i++) ye2e[t * DSV4_DIM + i] += d[i];
    }
    cmp_add(&st->y, y, yref, T * DSV4_DIM);
    cmp_add(&st->e2e, y, ye2e, T * DSV4_DIM);
    free(ref), free(h), free(d), free(yref), free(ye2e);
    sx_buffer *bufs[] = {xb, xqb, sb, rwb, hqb, hrb, yb};
    for (size_t i = 0; i < COUNT(bufs); i++) sx_buffer_release(bufs[i]);
}

static void report(const char *name, const Stats *s) {
    dbg("dsv4 %s: h rel %.2e diff %zu/%zu, y rel %.2e diff %zu/%zu, end to end rel %.2e (hq codes differing %zu)\n", name,
        s->h.rel, s->h.diff, s->h.n, s->y.rel, s->y.diff, s->y.n, s->e2e.rel, s->hq_codes);
}

static bool within(const Stats *s) {
    return s->h.rel <= TOL_DSV4_STAGE_REL && s->y.rel <= TOL_DSV4_STAGE_REL && s->e2e.rel <= TOL_DSV4_E2E_REL &&
           (double)s->h.diff <= TOL_DSV4_DIFF_FRAC * (double)s->h.n && (double)s->y.diff <= TOL_DSV4_DIFF_FRAC * (double)s->y.n;
}

TEST(dsv4_formats) {
    // e4m3: every finite code decodes and rounds to itself; midpoints round to even codes
    for (int c = 0; c < 256; c++) {
        if ((c & 0x7f) == 0x7f) continue;
        const float v = dsv4_e4m3((uint8_t)c);
        EXPECT(dsv4_e4m3_round(v) == v);
    }
    EXPECT(dsv4_e4m3(0x7e) == 448.0f && dsv4_e4m3(0x08) == 0.015625f && dsv4_e4m3(0x01) == 0.001953125f);
    EXPECT(dsv4_e4m3_round(1.0625f) == 1.0f && dsv4_e4m3_round(1.1875f) == 1.25f); // ties to even
    EXPECT(dsv4_e4m3_round(0.0029296875f) == 0.00390625f);                         // subnormal tie 1.5 * 2^-9
    EXPECT(dsv4_e2m1(7) == 6.0f && dsv4_e2m1(9) == -0.5f && dsv4_e8m0(127) == 1.0f && dsv4_e8m0(0) == 0x1p-127f);
    EXPECT(dsv4_act_scale(448.0f) == 1.0f && dsv4_act_scale(449.0f) == 2.0f && dsv4_act_scale(0) == 0x1p-13f * 0x1p-9f);
    EXPECT(dsv4_bf16_round(1.00390625f) == 1.0f && dsv4_bf16_round(1.01171875f) == 1.015625f);
}

TEST(dsv4_moe_decode) {
    Stats s = {0};
    static const uint32_t slots[6] = {3, 7, 0, 11, 5, 1};
    run_case(failed_, 1, slots, &s);
    if (*failed_) return;
    report("T=1", &s);
    EXPECT(within(&s));
}

TEST(dsv4_moe_chunk) {
    // T = 3 (rows share some slots) and T = 8 over 12 slots (most slots used by several rows)
    Stats s = {0};
    static const uint32_t s3[18] = {3, 7, 0, 11, 5, 1, 7, 2, 3, 9, 10, 4, 6, 8, 11, 1, 0, 2};
    run_case(failed_, 3, s3, &s);
    if (*failed_) return;
    uint32_t s8[48];
    for (uint32_t t = 0; t < 8; t++)
        for (uint32_t k = 0; k < 6; k++) s8[t * 6 + k] = (t * 5 + k * 7) % NSLOT; // distinct within a row
    run_case(failed_, 8, s8, &s);
    if (*failed_) return;
    report("T=3,8", &s);
    EXPECT(within(&s));
}
