// DeepSeek-V4-Flash scalar references (see dsv4_ref.h).
#include "dsv4_ref.h"

#include <math.h>

static inline uint32_t f2u(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}
static inline float u2f(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}

float dsv4_e8m0(uint8_t v) { return u2f(v == 0 ? 0x00400000u : (uint32_t)v << 23); }

float dsv4_e2m1(uint8_t code) {
    static const float t[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = t[code & 7];
    return (code & 8) ? -v : v;
}

float dsv4_e4m3(uint8_t code) {
    const uint32_t e = (code >> 3) & 15, m = code & 7;
    float v;
    if (e == 15 && m == 7) v = NAN;
    else if (e == 0) v = ldexpf((float)m, -9);
    else v = ldexpf((float)(8 + m), (int)e - 10);
    return (code & 0x80) ? -v : v;
}

float dsv4_e4m3_round(float v) {
    const float a = fabsf(v);
    float r;
    if (a < 0.015625f) {
        r = nearbyintf(a * 512.0f) * 0.001953125f; // subnormals: steps of 2^-9 (default rounding: to even)
    } else {
        uint32_t u = f2u(a);
        u += 0x7ffffu + ((u >> 20) & 1u); // keep 3 mantissa bits, ties to even
        r = u2f(u & 0xfff00000u);
    }
    return copysignf(r, v);
}

float dsv4_bf16_round(float v) {
    uint32_t u = f2u(v);
    u += 0x7fffu + ((u >> 16) & 1u);
    return u2f(u & 0xffff0000u);
}

// The reference takes ceil(log2(.)) from the bits of the f32 product amax * f32(1 / 448).
float dsv4_act_scale(float amax) {
    const uint32_t b = f2u(fmaxf(amax, 1e-4f) * (float)(1.0 / 448.0));
    const int e = (int)((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0 ? 1 : 0);
    return u2f((uint32_t)(127 + e) << 23);
}

void dsv4_act_quant(const float *x, float *xq, size_t n, size_t block) {
    for (size_t b = 0; b < n; b += block) {
        float amax = 0;
        for (size_t i = 0; i < block; i++) amax = fmaxf(amax, fabsf(x[b + i]));
        const float s = dsv4_act_scale(amax), inv = 1.0f / s; // exact: s is a power of two
        for (size_t i = 0; i < block; i++) xq[b + i] = dsv4_e4m3_round(fminf(fmaxf(x[b + i] * inv, -448.0f), 448.0f)) * s;
    }
}

float dsv4_fp4_dot(const uint8_t *w, const uint8_t *s, const float *xq, size_t k) {
    float acc = 0;
    for (size_t b = 0; b < k / DSV4_FP4_BLOCK; b++) {
        float part = 0;
        for (size_t i = 0; i < DSV4_FP4_BLOCK; i += 2) {
            const uint8_t byte = w[(b * DSV4_FP4_BLOCK + i) / 2];
            part += dsv4_e2m1(byte & 15) * xq[b * DSV4_FP4_BLOCK + i];
            part += dsv4_e2m1(byte >> 4) * xq[b * DSV4_FP4_BLOCK + i + 1];
        }
        acc += part * dsv4_e8m0(s[b]);
    }
    return acc;
}

static float silu(float g) { return g / (1.0f + expf(-g)); }

void dsv4_gate_up_ref(const Dsv4Expert *e, const float *xq, float route_w, float *h) {
    for (size_t r = 0; r < DSV4_INTER; r++) {
        float g = dsv4_bf16_round(dsv4_fp4_dot(e->w1 + r * (DSV4_DIM / 2), e->s1 + r * (DSV4_DIM / 32), xq, DSV4_DIM));
        float u = dsv4_bf16_round(dsv4_fp4_dot(e->w3 + r * (DSV4_DIM / 2), e->s3 + r * (DSV4_DIM / 32), xq, DSV4_DIM));
        u = fminf(fmaxf(u, -DSV4_SWIGLU_LIMIT), DSV4_SWIGLU_LIMIT);
        g = fminf(g, DSV4_SWIGLU_LIMIT);
        h[r] = dsv4_bf16_round(silu(g) * u * route_w);
    }
}

void dsv4_down_ref(const Dsv4Expert *e, const float *hq, float *d) {
    for (size_t r = 0; r < DSV4_DIM; r++)
        d[r] = dsv4_bf16_round(dsv4_fp4_dot(e->w2 + r * (DSV4_INTER / 2), e->s2 + r * (DSV4_INTER / 32), hq, DSV4_INTER));
}

void dsv4_moe_ref(const Dsv4Expert *experts, const float *rw, const float *x, size_t T, float *y, float *scratch) {
    float *xq = scratch, *d = scratch + DSV4_DIM, *h = d + DSV4_DIM;
    for (size_t t = 0; t < T; t++) {
        dsv4_act_quant(x + t * DSV4_DIM, xq, DSV4_DIM, DSV4_ACT_BLOCK);
        float *yt = y + t * DSV4_DIM;
        for (size_t i = 0; i < DSV4_DIM; i++) yt[i] = 0;
        for (size_t k = 0; k < DSV4_TOPK; k++) {
            const Dsv4Expert *e = &experts[t * DSV4_TOPK + k];
            dsv4_gate_up_ref(e, xq, rw[t * DSV4_TOPK + k], h);
            dsv4_act_quant(h, h, DSV4_INTER, DSV4_ACT_BLOCK);
            dsv4_down_ref(e, h, d);
            for (size_t i = 0; i < DSV4_DIM; i++) yt[i] += d[i];
        }
    }
}
