// Scalar references for src/kernels/dsv4_hc.metal: DeepSeek-V4-Flash hyper-connections
// (hc_pre, hc_split_sinkhorn, hc_post, hc_head), RMSNorm and the MoE gate, following the
// checkpoint's own inference code (model.py Block.hc_pre/hc_post/hc_head, RMSNorm, Gate;
// kernel.py hc_split_sinkhorn_kernel) in its order of operations.
#include "dsv4_ref_hc.h"

#include <math.h>
#include <string.h>

// dsv4_bf16_round is defined once, in src/dsv4_ref.c.

float dsv4_bf16_val(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float r;
    memcpy(&r, &u, 4);
    return r;
}

static float sigmoid(float z) { return 1.0f / (1.0f + expf(-z)); }

void dsv4_ref_hc_mixes(const float *fn, size_t nmix, const float *x, float *mixes) {
    double ss = 0;
    for (size_t i = 0; i < DSV4_HCD; i++) ss += (double)x[i] * x[i];
    float r = 1.0f / sqrtf((float)(ss / DSV4_HCD) + 1e-6f);
    for (size_t m = 0; m < nmix; m++) {
        double acc = 0;
        for (size_t i = 0; i < DSV4_HCD; i++) acc += (double)fn[m * DSV4_HCD + i] * x[i];
        mixes[m] = (float)acc * r;
    }
}

// Reductions over 4 values as the kernel's lane shuffles: (a0 + a1) + (a2 + a3).
static float sum4(float a, float b, float c, float d) { return (a + b) + (c + d); }

void dsv4_ref_hc_split(const float mixes[DSV4_MIX], const float scale[3], const float base[DSV4_MIX],
                       float pre[DSV4_HC], float post[DSV4_HC], float comb[DSV4_HC * DSV4_HC]) {
    enum { H = DSV4_HC };
    const float eps = DSV4_HC_EPS;
    for (int j = 0; j < H; j++) pre[j] = sigmoid(mixes[j] * scale[0] + base[j]) + eps;
    for (int j = 0; j < H; j++) post[j] = 2 * sigmoid(mixes[j + H] * scale[1] + base[j + H]);
    float c[H][H];
    for (int j = 0; j < H; j++)
        for (int k = 0; k < H; k++) c[j][k] = mixes[2 * H + j * H + k] * scale[2] + base[2 * H + j * H + k];
    // softmax over k, + eps
    for (int j = 0; j < H; j++) {
        float mx = fmaxf(fmaxf(c[j][0], c[j][1]), fmaxf(c[j][2], c[j][3]));
        for (int k = 0; k < H; k++) c[j][k] = expf(c[j][k] - mx);
        float s = sum4(c[j][0], c[j][1], c[j][2], c[j][3]);
        for (int k = 0; k < H; k++) c[j][k] = c[j][k] / s + eps;
    }
    for (int it = 0; it < DSV4_SINKHORN; it++) {
        if (it > 0) { // rows (the first round's row step is the softmax)
            float rs[H];
            for (int j = 0; j < H; j++) rs[j] = sum4(c[j][0], c[j][1], c[j][2], c[j][3]);
            for (int j = 0; j < H; j++)
                for (int k = 0; k < H; k++) c[j][k] = c[j][k] / (rs[j] + eps);
        }
        float cs[H];
        for (int k = 0; k < H; k++) cs[k] = sum4(c[0][k], c[1][k], c[2][k], c[3][k]);
        for (int j = 0; j < H; j++)
            for (int k = 0; k < H; k++) c[j][k] = c[j][k] / (cs[k] + eps);
    }
    memcpy(comb, c, sizeof c);
}

void dsv4_ref_hc_head_pre(const float mixes[DSV4_HC], float scale, const float base[DSV4_HC], float pre[DSV4_HC]) {
    for (int j = 0; j < DSV4_HC; j++) pre[j] = sigmoid(mixes[j] * scale + base[j]) + DSV4_HC_EPS;
}

void dsv4_ref_hc_weighted(const float pre[DSV4_HC], const float *x, float *y) {
    for (size_t d = 0; d < DSV4_D; d++) {
        float a = pre[0] * x[d];
        for (int j = 1; j < DSV4_HC; j++) a = a + pre[j] * x[j * DSV4_D + d];
        y[d] = dsv4_bf16_round(a);
    }
}

void dsv4_ref_hc_post(const float *f, const float *res, const float post[DSV4_HC], const float comb[DSV4_HC * DSV4_HC],
                      float *out) {
    for (int k = 0; k < DSV4_HC; k++)
        for (size_t d = 0; d < DSV4_D; d++) {
            float a = comb[k] * res[d];
            for (int j = 1; j < DSV4_HC; j++) a = a + comb[j * DSV4_HC + k] * res[j * DSV4_D + d];
            out[k * DSV4_D + d] = dsv4_bf16_round(post[k] * f[d] + a);
        }
}

void dsv4_ref_rmsnorm(const float *x, const uint16_t *w, size_t n, float *out) {
    double ss = 0;
    for (size_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float r = 1.0f / sqrtf((float)(ss / (double)n) + DSV4_NORM_EPS);
    for (size_t i = 0; i < n; i++) out[i] = dsv4_bf16_round(dsv4_bf16_val(w[i]) * (x[i] * r));
}

void dsv4_ref_router_logits(const uint16_t *w, const float *x, float *logits) {
    for (size_t e = 0; e < DSV4_NE; e++) {
        double acc = 0;
        for (size_t i = 0; i < DSV4_D; i++) acc += (double)dsv4_bf16_val(w[e * DSV4_D + i]) * x[i];
        logits[e] = (float)acc;
    }
}

// torch.nn.functional.softplus (beta 1, threshold 20), then sqrt.
float dsv4_ref_sqrt_softplus(float l) { return sqrtf(l > 20.0f ? l : log1pf(expf(l))); }

static void normalize(const float s[DSV4_TOPK], float weights[DSV4_TOPK]) {
    float sum = 0;
    for (int k = 0; k < DSV4_TOPK; k++) sum += s[k];
    for (int k = 0; k < DSV4_TOPK; k++) weights[k] = s[k] / sum * DSV4_ROUTE_SCALE;
}

void dsv4_ref_router_select(const float *logits, const float *bias, uint32_t ids[DSV4_TOPK],
                            float weights[DSV4_TOPK]) {
    float s[DSV4_NE], key[DSV4_NE];
    bool taken[DSV4_NE] = {0};
    for (int e = 0; e < DSV4_NE; e++) {
        s[e] = dsv4_ref_sqrt_softplus(logits[e]);
        key[e] = s[e] + bias[e];
    }
    float sel[DSV4_TOPK];
    for (int k = 0; k < DSV4_TOPK; k++) {
        int best = -1;
        for (int e = 0; e < DSV4_NE; e++)
            if (!taken[e] && (best < 0 || key[e] > key[best])) best = e; // strict: ties keep the lower index
        taken[best] = true;
        ids[k] = (uint32_t)best;
        sel[k] = s[best];
    }
    normalize(sel, weights);
}

void dsv4_ref_router_hash(const float *logits, const uint32_t eid[DSV4_TOPK], uint32_t ids[DSV4_TOPK],
                          float weights[DSV4_TOPK]) {
    float sel[DSV4_TOPK];
    for (int k = 0; k < DSV4_TOPK; k++) {
        ids[k] = eid[k];
        sel[k] = dsv4_ref_sqrt_softplus(logits[eid[k]]);
    }
    normalize(sel, weights);
}
