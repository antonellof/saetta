// DeepSeek-V4-Flash attention scalar references (see dsv4_ref_attn.h). Written from
// DeepSeek-V4-Flash-0731 at 7872f01, inference/model.py and inference/kernel.py (MIT,
// Copyright (c) DeepSeek); the FP8 rounding helpers follow kernel.py's act_quant.
#include "dsv4_ref_attn.h"

#include <assert.h>
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

void dsv4a_rope_freqs(bool compressed, float freqs[DSV4A_NPAIR]) {
    const int dim = DSV4A_RD;
    const double base = compressed ? 160000.0 : 10000.0;
    // freqs = 1 / base ** (arange(0, dim, 2) / dim), f32 tensors
    for (int i = 0; i < DSV4A_NPAIR; i++) freqs[i] = 1.0f / powf((float)base, (float)(2 * i) / (float)dim);
    if (!compressed) return;
    // YaRN: original_seq_len 65536, factor 16, beta_fast 32, beta_slow 1
    const double orig = 65536.0, factor = 16.0;
    const double pi = 3.14159265358979323846;
#define CORR_DIM(rot) (dim * log(orig / ((rot) * 2 * pi)) / (2 * log(base)))
    int low = (int)floor(CORR_DIM(32.0)), high = (int)ceil(CORR_DIM(1.0));
#undef CORR_DIM
    if (low < 0) low = 0;
    if (high > dim - 1) high = dim - 1;
    float hi = (float)high;
    if (low == high) hi += 0.001f;
    for (int i = 0; i < DSV4A_NPAIR; i++) {
        float ramp = ((float)i - (float)low) / (hi - (float)low);
        ramp = fminf(fmaxf(ramp, 0.0f), 1.0f);
        const float smooth = 1.0f - ramp;
        const float f = freqs[i];
        freqs[i] = f / (float)factor * (1.0f - smooth) + f * smooth;
    }
}

void dsv4a_rope_table(const float freqs[DSV4A_NPAIR], uint32_t pos0, size_t n, float *cs) {
    for (size_t k = 0; k < n; k++)
        for (int i = 0; i < DSV4A_NPAIR; i++) {
            const float a = (float)(pos0 + k) * freqs[i]; // torch.outer in f32
            cs[(k * DSV4A_NPAIR + i) * 2] = cosf(a);      // torch.polar, complex64
            cs[(k * DSV4A_NPAIR + i) * 2 + 1] = sinf(a);
        }
}

void dsv4a_ref_rope(float *x, const float *cs, bool inverse) {
    for (int i = 0; i < DSV4A_NPAIR; i++) {
        const float a = x[2 * i], b = x[2 * i + 1];
        const float c = cs[2 * i], s = inverse ? -cs[2 * i + 1] : cs[2 * i + 1];
        // complex product (a + bi)(c + si), f32, then copy_ into BF16
        const float re = a * c - b * s, im = a * s + b * c;
        x[2 * i] = dsv4a_bf16(re);
        x[2 * i + 1] = dsv4a_bf16(im);
    }
}

void dsv4a_ref_q(float *q, const float *cs) {
    for (int h = 0; h < DSV4A_NH; h++) {
        float *qh = q + (size_t)h * DSV4A_HD;
        // q *= rsqrt(q.square().mean(-1) + eps), each op on a BF16 tensor (f32 math, BF16 result)
        double sum = 0;
        for (int i = 0; i < DSV4A_HD; i++) sum += dsv4a_bf16(qh[i] * qh[i]);
        const float ms = dsv4a_bf16((float)sum / (float)DSV4A_HD);
        const float r = dsv4a_bf16(1.0f / sqrtf(dsv4a_bf16(ms + DSV4A_EPS)));
        for (int i = 0; i < DSV4A_HD; i++) qh[i] = dsv4a_bf16(qh[i] * r);
        dsv4a_ref_rope(qh + DSV4A_NOPE, cs, false);
    }
}

float dsv4a_e4m3_round(float v) {
    const float a = fabsf(v);
    float r;
    if (a < 0.015625f) {
        r = nearbyintf(a * 512.0f) * 0.001953125f; // subnormals: steps of 2^-9, ties to even
    } else {
        uint32_t u = f2u(a);
        u += 0x7ffffu + ((u >> 20) & 1u); // 3 mantissa bits, ties to even
        r = u2f(u & 0xfff00000u);
    }
    return copysignf(r, v);
}

float dsv4a_act_scale(float amax) {
    const uint32_t b = f2u(fmaxf(amax, 1e-4f) * (float)(1.0 / 448.0));
    const int e = (int)((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0 ? 1 : 0);
    return u2f((uint32_t)(127 + e) << 23);
}

void dsv4a_ref_kv(const float *x, const uint16_t *w, const float *cs, float *kv) {
    // RMSNorm: f32 x, var = mean(x^2), x * rsqrt(var + eps), weight * that, BF16
    double ss = 0;
    for (int i = 0; i < DSV4A_HD; i++) ss += (double)x[i] * x[i];
    const float r = 1.0f / sqrtf((float)ss / (float)DSV4A_HD + DSV4A_EPS);
    for (int i = 0; i < DSV4A_HD; i++) kv[i] = dsv4a_bf16(dsv4a_bf16_val(w[i]) * (x[i] * r));
    dsv4a_ref_rope(kv + DSV4A_NOPE, cs, false);
    // act_quant(kv[:448], 64, ue8m0, inplace): bf16(f32(e4m3(clamp(x / s))) * s)
    for (int b = 0; b < DSV4A_NOPE; b += DSV4A_QBLK) {
        float amax = 0;
        for (int i = 0; i < DSV4A_QBLK; i++) amax = fmaxf(amax, fabsf(kv[b + i]));
        const float s = dsv4a_act_scale(amax);
        for (int i = 0; i < DSV4A_QBLK; i++)
            kv[b + i] = dsv4a_bf16(dsv4a_e4m3_round(fminf(fmaxf(kv[b + i] / s, -448.0f), 448.0f)) * s);
    }
}

void dsv4a_ref_window_list(uint32_t p, int32_t slots[DSV4A_WIN]) {
    if (p >= DSV4A_WIN - 1) {
        const uint32_t s = p % DSV4A_WIN; // cat(arange(s + 1, win), arange(0, s + 1))
        int n = 0;
        for (uint32_t i = s + 1; i < DSV4A_WIN; i++) slots[n++] = (int32_t)i;
        for (uint32_t i = 0; i <= s; i++) slots[n++] = (int32_t)i;
    } else { // pad(arange(p + 1), value -1)
        for (uint32_t i = 0; i < DSV4A_WIN; i++) slots[i] = i <= p ? (int32_t)i : -1;
    }
}

// kernel.py sparse_attn, one (batch, position): blocks of 64 list entries in order, online
// softmax with the running max; the numerator takes the exponentials rounded to BF16 against
// the block's running max, the denominator the f32 ones; the sink enters the denominator.
void dsv4a_ref_sparse_attn(const float *q, const float *const *keys, size_t L, const float *sink, float *o) {
    const size_t nb = (L + DSV4A_BLK - 1) / DSV4A_BLK;
    for (int h = 0; h < DSV4A_NH; h++) {
        const float *qh = q + (size_t)h * DSV4A_HD;
        float m = -INFINITY, se = 0;
        float acc[DSV4A_HD] = {0};
        for (size_t b = 0; b < nb; b++) {
            float s[DSV4A_BLK], p[DSV4A_BLK];
            float bmax = -INFINITY;
            for (int j = 0; j < DSV4A_BLK; j++) {
                const size_t e = b * DSV4A_BLK + j;
                const float *k = e < L ? keys[e] : NULL;
                if (!k) {
                    s[j] = -INFINITY;
                    continue;
                }
                double d = 0;
                for (int i = 0; i < DSV4A_HD; i++) d += (double)qh[i] * k[i];
                s[j] = (float)d * DSV4A_SCALE;
                bmax = fmaxf(bmax, s[j]);
            }
            const float mn = fmaxf(m, bmax);
            assert(mn > -INFINITY); // the first block holds a valid key
            const float sc = expf(m - mn);
            float sum = 0;
            for (int j = 0; j < DSV4A_BLK; j++) {
                p[j] = expf(s[j] - mn);
                sum += p[j];
            }
            se = se * sc + sum;
            for (int i = 0; i < DSV4A_HD; i++) {
                double d = 0;
                for (int j = 0; j < DSV4A_BLK; j++) {
                    const size_t e = b * DSV4A_BLK + j;
                    const float *k = e < L ? keys[e] : NULL;
                    if (k) d += (double)dsv4a_bf16(p[j]) * k[i];
                }
                acc[i] = acc[i] * sc + (float)d;
            }
            m = mn;
        }
        se += expf(sink[h] - m);
        for (int i = 0; i < DSV4A_HD; i++) o[(size_t)h * DSV4A_HD + i] = dsv4a_bf16(acc[i] / se);
    }
}

void dsv4a_ref_derotate(float *o, const float *cs) {
    for (int h = 0; h < DSV4A_NH; h++) dsv4a_ref_rope(o + (size_t)h * DSV4A_HD + DSV4A_NOPE, cs, true);
}

void dsv4a_ref_wo_a(const float *o, const uint16_t *w, float *y) {
    for (size_t r = 0; r < DSV4A_WOA_ROWS; r++) {
        const float *og = o + (r / DSV4A_OL) * DSV4A_GD;
        const uint16_t *wr = w + r * DSV4A_GD;
        double d = 0;
        for (size_t i = 0; i < DSV4A_GD; i++) d += (double)dsv4a_bf16_val(wr[i]) * og[i];
        y[r] = dsv4a_bf16((float)d);
    }
}
