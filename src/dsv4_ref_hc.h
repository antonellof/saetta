// Scalar references for the DeepSeek-V4-Flash hyper-connection, norm and router kernels
// (src/kernels/dsv4_hc.metal). BF16 values are carried as f32 holding BF16 values; BF16
// weights are raw uint16 bit patterns. Large dot products accumulate in f64 (the exact
// value the f32 kernels approximate); the small reductions follow the kernels' order.
#ifndef SAETTA_DSV4_REF_HC_H
#define SAETTA_DSV4_REF_HC_H

#include <stddef.h>
#include <stdint.h>

enum {
    DSV4_D = 4096,
    DSV4_HC = 4,
    DSV4_HCD = DSV4_HC * DSV4_D,
    DSV4_MIX = (2 + DSV4_HC) * DSV4_HC, // 24
    DSV4_SINKHORN = 20,
    DSV4_NE = 256,
    DSV4_TOPK = 6,
    DSV4_HC_SLICES = 8, // dsv4_hc_mix: slices per row (the partials buffer layout)
};
#define DSV4_HC_EPS 1e-6f
#define DSV4_NORM_EPS 1e-6f
#define DSV4_ROUTE_SCALE 1.5f

float dsv4_bf16_round(float v); // to the nearest BF16, ties to even (NaN kept)
float dsv4_bf16_val(uint16_t u);

// mixes[r] = (fn[r] . x) * rsqrt(mean(x^2) + 1e-6), r < nmix; fn [nmix][16384], x [16384].
void dsv4_ref_hc_mixes(const float *fn, size_t nmix, const float *x, float *mixes);
// hc_split_sinkhorn of one token: pre [4], post [4], comb [4][4] (comb[j][k]).
void dsv4_ref_hc_split(const float mixes[DSV4_MIX], const float scale[3], const float base[DSV4_MIX],
                       float pre[DSV4_HC], float post[DSV4_HC], float comb[DSV4_HC * DSV4_HC]);
// hc_head pre: sigmoid(m * scale + base) + eps.
void dsv4_ref_hc_head_pre(const float mixes[DSV4_HC], float scale, const float base[DSV4_HC], float pre[DSV4_HC]);
// y = BF16(sum_j pre[j] x_j), x [4][4096], y [4096].
void dsv4_ref_hc_weighted(const float pre[DSV4_HC], const float *x, float *y);
// out[k] = BF16(post[k] f + sum_j comb[j][k] res_j); f [4096], res and out [4][4096].
void dsv4_ref_hc_post(const float *f, const float *res, const float post[DSV4_HC], const float comb[DSV4_HC * DSV4_HC],
                      float *out);
// out = BF16(w * (x * rsqrt(mean(x^2) + 1e-6))), x [n], w BF16 [n].
void dsv4_ref_rmsnorm(const float *x, const uint16_t *w, size_t n, float *out);

// logits [256] = W (BF16 [256][4096]) . x.
void dsv4_ref_router_logits(const uint16_t *w, const float *x, float *logits);
float dsv4_ref_sqrt_softplus(float l);
// Top-6 of sqrt(softplus(logits)) + bias, descending, ties to the lower index; weights the
// unbiased scores normalized * 1.5.
void dsv4_ref_router_select(const float *logits, const float *bias, uint32_t ids[DSV4_TOPK],
                            float weights[DSV4_TOPK]);
// Hash routing: ids = eid (a tid2eid row), weights from logits at those ids.
void dsv4_ref_router_hash(const float *logits, const uint32_t eid[DSV4_TOPK], uint32_t ids[DSV4_TOPK],
                          float weights[DSV4_TOPK]);

#endif
