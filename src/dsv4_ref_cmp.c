// DeepSeek-V4-Flash compressor and indexer scalar references (see dsv4_ref_cmp.h).
#include "dsv4_ref_cmp.h"

#include <math.h>
#include <stdlib.h>

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

Dsv4Cmp dsv4_cmp_shape(uint32_t ratio, uint32_t d, bool rotate) {
    const uint32_t coff = ratio == 4 ? 2 : 1; // overlap iff ratio 4 (R:296-298)
    return (Dsv4Cmp){ratio, d, coff, coff * ratio, coff * d, rotate};
}

float dsv4c_bf16(float v) {
    uint32_t u = f2u(v);
    u += 0x7fffu + ((u >> 16) & 1u);
    return u2f(u & 0xffff0000u);
}

float dsv4c_e4m3_round(float v) {
    const float a = fabsf(v);
    float r;
    if (a < 0.015625f) {
        r = nearbyintf(a * 512.0f) * 0.001953125f; // subnormals: steps of 2^-9, ties to even
    } else {
        uint32_t u = f2u(a);
        u += 0x7ffffu + ((u >> 20) & 1u); // keep 3 mantissa bits, ties to even
        r = u2f(u & 0xfff00000u);
    }
    return copysignf(r, v);
}

float dsv4c_e2m1_round(float v) {
    const float a = fabsf(v);
    float r;
    if (a < 2.0f) r = nearbyintf(a * 2.0f) * 0.5f; // 0, .5, 1, 1.5, 2
    else if (a < 4.0f) r = nearbyintf(a);          // 2, 3, 4
    else r = nearbyintf(a * 0.5f) * 2.0f;          // 4, 6
    return copysignf(r, v);
}

// K:22-37: 2^ceil(log2(max(amax, floor) * inv_max)) from the bits of the f32 product; *inv = 1 / s.
static float round_scale(float amax, float floor_, float inv_max, float *inv) {
    const uint32_t b = f2u(fmaxf(amax, floor_) * inv_max);
    const int e = (int)((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0 ? 1 : 0);
    *inv = u2f((uint32_t)(127 - e) << 23);
    return u2f((uint32_t)(127 + e) << 23);
}

void dsv4c_fp8_qdq(float *x, size_t n) {
    for (size_t b = 0; b < n; b += DSV4C_FP8_BLOCK) {
        float amax = 0, inv;
        for (size_t i = 0; i < DSV4C_FP8_BLOCK; i++) amax = fmaxf(amax, fabsf(x[b + i]));
        const float s = round_scale(amax, 1e-4f, (float)(1.0 / 448.0), &inv);
        for (size_t i = 0; i < DSV4C_FP8_BLOCK; i++)
            x[b + i] = dsv4c_e4m3_round(fminf(fmaxf(x[b + i] * inv, -448.0f), 448.0f)) * s;
    }
}

void dsv4c_fp4_qdq(float *x, size_t n) {
    for (size_t b = 0; b < n; b += DSV4C_FP4_BLOCK) {
        float amax = 0, inv;
        for (size_t i = 0; i < DSV4C_FP4_BLOCK; i++) amax = fmaxf(amax, fabsf(x[b + i]));
        const float s = round_scale(amax, 0x1.8p-124f, (float)(1.0 / 6.0), &inv); // floor 6 * 2^-126
        for (size_t i = 0; i < DSV4C_FP4_BLOCK; i++)
            x[b + i] = dsv4c_e2m1_round(fminf(fmaxf(x[b + i] * inv, -6.0f), 6.0f)) * s;
    }
}

void dsv4c_hadamard(float *x, size_t n) {
    for (size_t h = 1; h < n; h *= 2)
        for (size_t i = 0; i < n; i++)
            if ((i & h) == 0) {
                const float a = x[i], b = x[i + h];
                x[i] = a + b;
                x[i + h] = a - b;
            }
    const float scale = (float)(1.0 / sqrt((double)n));
    for (size_t i = 0; i < n; i++) x[i] = dsv4c_bf16(x[i] * scale);
}

void dsv4_cmp_rope_freqs(float freqs[DSV4C_PAIRS]) {
    const double base = 160000.0, factor = 16.0, orig = 65536.0; // C:47-55, 65
    const double dim = 2 * DSV4C_PAIRS;
    // find_correction_range (R:211-217) in double, as Python's math
    const double pi = 3.14159265358979323846;
    const double lo_d = dim * log(orig / (32.0 * 2 * pi)) / (2 * log(base)); // beta_fast 32
    const double hi_d = dim * log(orig / (1.0 * 2 * pi)) / (2 * log(base));  // beta_slow 1
    int lo = (int)floor(lo_d), hi = (int)ceil(hi_d);
    if (lo < 0) lo = 0;
    if (hi > (int)dim - 1) hi = (int)dim - 1;
    const float span = lo == hi ? (float)hi + 0.001f - (float)lo : (float)(hi - lo);
    for (int i = 0; i < DSV4C_PAIRS; i++) {
        const float ex = (float)(2 * i) / (float)dim;
        const float f = 1.0f / (float)pow(base, (double)ex); // R:226, the power correctly rounded
        float ramp = ((float)i - (float)lo) / span;            // R:219-224
        ramp = fminf(fmaxf(ramp, 0.0f), 1.0f);
        const float smooth = 1.0f - ramp;
        freqs[i] = f / (float)factor * (1.0f - smooth) + f * smooth; // R:230
    }
}

void dsv4_cmp_rope_cs(const float freqs[DSV4C_PAIRS], uint32_t pos, float cs[2 * DSV4C_PAIRS]) {
    for (int i = 0; i < DSV4C_PAIRS; i++) {
        const float ang = (float)pos * freqs[i]; // R:232-233
        cs[2 * i] = (float)cos((double)ang);
        cs[2 * i + 1] = (float)sin((double)ang);
    }
}

void dsv4_cmp_state_init(const Dsv4Cmp *k, float *kv_state, float *score_state) {
    for (size_t i = 0; i < (size_t)k->rows * k->cd; i++) {
        kv_state[i] = 0.0f;
        score_state[i] = -INFINITY;
    }
}

float dsv4c_exp(float x) {
    x = fmaxf(x, -87.0f);
    const float n = nearbyintf(x * 1.44269504088896341f);
    const float r = (x - n * 0.693145751953125f) - n * 1.428606765330187045e-06f;
    float p = 1.0f / 5040.0f;
    p = p * r + 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    return p * u2f((uint32_t)((int)n + 127) << 23);
}

// Softmax over n rows and the weighted sum (R:348, 358, 365): kv[r], s[r] at stride; rows with
// score -inf have weight 0 and add nothing.
static float pool(const float *kv, const float *s, size_t n, size_t stride_kv, size_t stride_s) {
    float m = -INFINITY;
    for (size_t r = 0; r < n; r++) m = fmaxf(m, s[r * stride_s]);
    float sum = 0;
    for (size_t r = 0; r < n; r++)
        if (s[r * stride_s] != -INFINITY) sum += dsv4c_exp(s[r * stride_s] - m);
    float acc = 0;
    for (size_t r = 0; r < n; r++)
        if (s[r * stride_s] != -INFINITY) acc += kv[r * stride_kv] * (dsv4c_exp(s[r * stride_s] - m) / sum);
    return acc;
}

// One entry from the 8 (overlap) or ratio rows given row by row as pointers to [cd] rows.
static void pool_rows(const Dsv4Cmp *k, const float *const *kvr, const float *const *scr, float *pooled) {
    float kvv[128], sv[128];
    for (uint32_t c = 0; c < k->d; c++) {
        for (uint32_t r = 0; r < k->rows; r++) {
            const uint32_t ch = (k->coff == 2 && r >= k->ratio) ? k->d + c : c;
            kvv[r] = kvr[r][ch];
            sv[r] = scr[r][ch];
        }
        pooled[c] = pool(kvv, sv, k->rows, 1, 1);
    }
}

bool dsv4_cmp_ref_step(const Dsv4Cmp *k, float *kv_state, float *score_state, const float *kv,
                       const float *score, const float *ape, uint32_t pos, float *pooled) {
    const uint32_t ratio = k->ratio, cd = k->cd, j = pos % ratio;
    const uint32_t row = (k->coff - 1) * ratio + j;
    for (uint32_t i = 0; i < cd; i++) {
        kv_state[row * cd + i] = kv[i];
        score_state[row * cd + i] = score[i] + ape[j * cd + i]; // R:351
    }
    if ((pos + 1) % ratio != 0) return false;
    const float *kvr[128], *scr[128];
    for (uint32_t r = 0; r < k->rows; r++) {
        kvr[r] = kv_state + r * cd;
        scr[r] = score_state + r * cd;
    }
    pool_rows(k, kvr, scr, pooled);
    if (k->coff == 2) { // R:359-360
        memmove(kv_state, kv_state + ratio * cd, (size_t)ratio * cd * 4);
        memmove(score_state, score_state + ratio * cd, (size_t)ratio * cd * 4);
    }
    return true;
}

size_t dsv4_cmp_ref_prefill(const Dsv4Cmp *k, float *kv_state, float *score_state, const float *kv,
                            const float *score, const float *ape, size_t T, float *pooled) {
    const size_t ratio = k->ratio, cd = k->cd, rem = T % ratio, cut = T - rem, n = cut / ratio;
    const bool overlap = k->coff == 2;
    const size_t off = overlap ? ratio : 0;
    if (overlap && cut >= ratio) // R:336-338
        for (size_t r = 0; r < ratio; r++)
            for (size_t i = 0; i < cd; i++) {
                kv_state[r * cd + i] = kv[(cut - ratio + r) * cd + i];
                score_state[r * cd + i] = score[(cut - ratio + r) * cd + i] + ape[r * cd + i];
            }
    for (size_t r = 0; r < rem; r++) // R:339-342
        for (size_t i = 0; i < cd; i++) {
            kv_state[(off + r) * cd + i] = kv[(cut + r) * cd + i];
            score_state[(off + r) * cd + i] = score[(cut + r) * cd + i] + ape[r * cd + i];
        }
    // windows: score + ape (R:344), overlap_transform (R:313-320; kv 0, score -inf before window 0)
    float *sw = malloc(2 * ratio * cd * sizeof(float));
    if (!sw) die("dsv4_cmp_ref_prefill: out of memory\n");
    static const float zero_row[1024] = {0};
    static float ninf_row[1024];
    for (size_t i = 0; i < 1024; i++) ninf_row[i] = -INFINITY;
    for (size_t w = 0; w < n; w++) {
        const float *kvr[256], *scr[256];
        for (size_t r = 0; r < ratio; r++) { // current window
            const size_t t = w * ratio + r;
            for (size_t i = 0; i < cd; i++) sw[(ratio + r) * cd + i] = score[t * cd + i] + ape[r * cd + i];
            kvr[(overlap ? ratio : 0) + r] = kv + t * cd;
            scr[(overlap ? ratio : 0) + r] = sw + (ratio + r) * cd;
        }
        if (overlap)
            for (size_t r = 0; r < ratio; r++) { // previous window
                if (w == 0) {
                    kvr[r] = zero_row;
                    scr[r] = ninf_row;
                } else {
                    const size_t t = (w - 1) * ratio + r;
                    for (size_t i = 0; i < cd; i++) sw[r * cd + i] = score[t * cd + i] + ape[r * cd + i];
                    kvr[r] = kv + t * cd;
                    scr[r] = sw + r * cd;
                }
            }
        pool_rows(k, kvr, scr, pooled + w * k->d);
    }
    free(sw);
    return n;
}

void dsv4_cmp_ref_post(const Dsv4Cmp *k, const float *pooled, const float *normw, const float *cs, float *entry) {
    const uint32_t d = k->d;
    float x[512];
    for (uint32_t c = 0; c < d; c++) x[c] = dsv4c_bf16(pooled[c]); // kv.to(dtype), R:368
    float part[32];
    for (uint32_t j = 0; j < 32; j++) {
        part[j] = 0;
        for (uint32_t i = j; i < d; i += 32) part[j] += x[i] * x[i];
    }
    float ss = 0;
    for (uint32_t j = 0; j < 32; j++) ss += part[j];
    const float r = 1.0f / sqrtf(ss / (float)d + DSV4C_EPS); // R:197-202
    for (uint32_t c = 0; c < d; c++) x[c] = dsv4c_bf16(normw[c] * (x[c] * r));
    float *xr = x + d - DSV4C_ROPE; // R:373, R:238-250
    for (uint32_t i = 0; i < DSV4C_PAIRS; i++) {
        const float re = xr[2 * i], im = xr[2 * i + 1], c = cs[2 * i], s = cs[2 * i + 1];
        xr[2 * i] = dsv4c_bf16(re * c - im * s);
        xr[2 * i + 1] = dsv4c_bf16(re * s + im * c);
    }
    if (k->rotate) { // R:374-376
        dsv4c_hadamard(x, d);
        dsv4c_fp4_qdq(x, d);
    } else {
        dsv4c_fp8_qdq(x, d - DSV4C_ROPE); // R:378
    }
    memcpy(entry, x, d * 4);
}

void dsv4_idx_ref_q(const float *q, const float *cs, float *qx) {
    for (uint32_t h = 0; h < DSV4C_IDX_HEADS; h++) {
        float x[DSV4C_IDX_DIM];
        memcpy(x, q + h * DSV4C_IDX_DIM, sizeof x);
        float *xr = x + DSV4C_IDX_DIM - DSV4C_ROPE; // R:419
        for (uint32_t i = 0; i < DSV4C_PAIRS; i++) {
            const float re = xr[2 * i], im = xr[2 * i + 1], c = cs[2 * i], s = cs[2 * i + 1];
            xr[2 * i] = dsv4c_bf16(re * c - im * s);
            xr[2 * i + 1] = dsv4c_bf16(re * s + im * c);
        }
        dsv4c_hadamard(x, DSV4C_IDX_DIM);         // R:420
        dsv4c_fp4_qdq(x, DSV4C_IDX_DIM);          // R:422
        memcpy(qx + h * DSV4C_IDX_DIM, x, sizeof x);
    }
}

void dsv4_idx_ref_w(const float *wp, float *w) {
    const float scale = (float)(pow(128.0, -0.5) * pow(64.0, -0.5)); // R:401, R:424
    for (uint32_t h = 0; h < DSV4C_IDX_HEADS; h++) w[h] = dsv4c_bf16(wp[h] * scale);
}

void dsv4_idx_ref_scores(const float *qx, const float *w, const float *keys, size_t n, float *scores) {
    for (size_t s = 0; s < n; s++) {
        const float *kk = keys + s * DSV4C_IDX_DIM;
        float acc = 0;
        for (uint32_t h = 0; h < DSV4C_IDX_HEADS; h++) {
            const float *qh = qx + h * DSV4C_IDX_DIM;
            float dot = 0;
            for (uint32_t i = 0; i < DSV4C_IDX_DIM; i++) dot += qh[i] * kk[i];
            const float rl = fmaxf(dsv4c_bf16(dot), 0.0f); // relu_ on the bf16 einsum output
            acc += dsv4c_bf16(rl * w[h]);
        }
        scores[s] = dsv4c_bf16(acc);
    }
}

static const float *topk_scores;
static int topk_cmp(const void *a, const void *b) {
    const int32_t i = *(const int32_t *)a, j = *(const int32_t *)b;
    const float x = topk_scores[i], y = topk_scores[j];
    if (x > y) return -1;
    if (x < y) return 1;
    return i < j ? -1 : (i > j); // ties to the lower index
}

size_t dsv4_idx_ref_topk(const float *scores, size_t n_vis, size_t n_total, int32_t offset, int32_t *idx) {
    const size_t k = n_total < DSV4C_TOPK ? n_total : DSV4C_TOPK;
    int32_t *ord = malloc((n_vis ? n_vis : 1) * sizeof(int32_t));
    if (!ord) die("dsv4_idx_ref_topk: out of memory\n");
    for (size_t i = 0; i < n_vis; i++) ord[i] = (int32_t)i;
    topk_scores = scores;
    qsort(ord, n_vis, sizeof(int32_t), topk_cmp);
    for (size_t r = 0; r < k; r++) idx[r] = r < n_vis ? ord[r] + offset : -1;
    free(ord);
    return k;
}
