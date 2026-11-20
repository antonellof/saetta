// Scalar references of the DeepSeek-V4-Flash FP8 linears (see dsv4_ref_fp8.h).
#include "dsv4_ref_fp8.h"

#include <math.h>
#include <string.h>

static inline uint32_t f2u(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}


float dsv4_e4m3_to_f32(uint8_t b) {
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    float v;
    if (e == 15 && m == 7) v = NAN;
    else if (e == 0) v = (float)m * 0x1p-9f;
    else v = ldexpf(1.0f + (float)m / 8.0f, (int)e - 7);
    return (b & 0x80u) ? -v : v;
}

float dsv4_e8m0_to_f32(uint8_t v) { return v == 0xff ? NAN : ldexpf(1.0f, (int)v - 127); }

uint8_t dsv4_f32_to_e4m3(float v) {
    if (isnan(v)) return 0x7f;
    const uint8_t sign = signbit(v) ? 0x80 : 0;
    float a = fabsf(v);
    if (a > DSV4_FP8_MAX) a = DSV4_FP8_MAX;
    if (a < 0x1p-6f) return sign | (uint8_t)rintf(a * 512.0f); // 8 is 0x08 = 2^-6
    uint32_t u = f2u(a);
    u += 0x7ffffu + ((u >> 20) & 1u); // to 3 mantissa bits, ties to even
    const uint32_t e = ((u >> 23) & 0xffu) - 127u + 7u, m = (u >> 20) & 7u;
    return sign | (uint8_t)((e << 3) | m);
}

// dsv4_act_scale is defined once, in src/dsv4_ref.c (the same formula).

void dsv4_ref_act_quant(const float *x, size_t T, size_t K, size_t ldx, uint8_t *codes, float *scales) {
    const size_t nb = K / DSV4_FP8_BLOCK;
    for (size_t t = 0; t < T; t++)
        for (size_t b = 0; b < nb; b++) {
            const float *xb = x + t * ldx + b * DSV4_FP8_BLOCK;
            float amax = 0;
            for (size_t i = 0; i < DSV4_FP8_BLOCK; i++) amax = fmaxf(amax, fabsf(xb[i]));
            const float s = dsv4_act_scale(amax);
            scales[t * nb + b] = s;
            for (size_t i = 0; i < DSV4_FP8_BLOCK; i++) {
                float q = xb[i] / s;
                q = q < -DSV4_FP8_MAX ? -DSV4_FP8_MAX : q > DSV4_FP8_MAX ? DSV4_FP8_MAX : q;
                codes[t * K + b * DSV4_FP8_BLOCK + i] = dsv4_f32_to_e4m3(q);
            }
        }
}

void dsv4_ref_fp8_gemm(const uint8_t *w, const uint8_t *ws, size_t N, size_t K, const uint8_t *codes,
                       const float *scales, size_t T, float *y, size_t ldy) {
    const size_t nb = K / DSV4_FP8_BLOCK;
    float lut[256];
    for (int i = 0; i < 256; i++) lut[i] = dsv4_e4m3_to_f32((uint8_t)i);
    for (size_t t = 0; t < T; t++)
        for (size_t n = 0; n < N; n++) {
            const uint8_t *wr = w + n * K, *xr = codes + t * K;
            float acc = 0;
            for (size_t b = 0; b < nb; b++) {
                float part = 0;
                for (size_t i = b * DSV4_FP8_BLOCK; i < (b + 1) * DSV4_FP8_BLOCK; i++) part += lut[xr[i]] * lut[wr[i]];
                const float sc = scales[t * nb + b] * dsv4_e8m0_to_f32(ws[(n / DSV4_FP8_BLOCK) * nb + b]);
                acc += part * sc;
            }
            y[t * ldy + n] = acc;
        }
}

void dsv4_ref_grouped(const uint8_t *w, const uint8_t *ws, size_t N, size_t K, size_t groups, const float *x,
                      size_t ldx, size_t T, float *y, size_t ldy) {
    const size_t nb = K / DSV4_FP8_BLOCK, rpg = N / groups;
    float lut[256];
    for (int i = 0; i < 256; i++) lut[i] = dsv4_e4m3_to_f32((uint8_t)i);
    for (size_t t = 0; t < T; t++)
        for (size_t n = 0; n < N; n++) {
            const uint8_t *wr = w + n * K;
            const float *xr = x + t * ldx + (n / rpg) * K;
            const uint8_t *sr = ws + (n / DSV4_FP8_BLOCK) * nb;
            float acc = 0;
            for (size_t b = 0; b < nb; b++) {
                const float sb = dsv4_e8m0_to_f32(sr[b]);
                for (size_t k = b * DSV4_FP8_BLOCK; k < (b + 1) * DSV4_FP8_BLOCK; k++) acc += (lut[wr[k]] * sb) * xr[k];
            }
            y[t * ldy + n] = acc;
        }
}
