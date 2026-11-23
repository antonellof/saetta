// Scalar references for src/kernels/inkling_*.metal (Inkling 2 Mini), in the order of operations
// of transformers be48e7a models/inkling2_loop/modeling_inkling2_loop.py (M), f32 values with
// f64 sums. Used by the unit tests (src/test/inkling.c); the model is checked end to end against a
// numpy reference by `saetta verify`.
#ifndef SAETTA_INKLING_REF_H
#define SAETTA_INKLING_REF_H

#include <stddef.h>
#include <stdint.h>

// Attention residual mix (M:828-853): out [nq][H] from states [n][H] (rows sstride apart) and partial.
void ink_ref_mix(const float *states, size_t sstride, size_t n, const float *partial, const float *query, size_t nq, size_t H,
                 float eps, float *out);
// x * rsqrt(mean(x^2) + eps) * w (w NULL: none).
void ink_ref_rmsnorm(const float *x, const float *w, size_t n, float eps, float *out);
// Message gate (M:79-96); D bf16 [R][2H], F, G [H][R].
void ink_ref_msg_gate(const float *x, const float *m, const uint16_t *D, const float *F, const float *G, size_t H, size_t R,
                      float eps, float *out);
// Query / relative-state prep of one token (M:190-236): q [NH][HD], r [NH][DREL] in place.
void ink_ref_qr(float *q, float *r, const float *qscale, const float *rscale, size_t NH, size_t HD, size_t DREL, size_t pos,
                int logscale, float thr_q, float thr_r, float alpha, float nfloor);
// Key / value conv of T tokens (M:328-455): raw [T][C], state [3][C] (the 3 inputs before),
// w [C][4] lag order, dyn [T][C / HD][4] or NULL; mode 0 key (L2 clip thr), 1 value (RMSNorm eps).
void ink_ref_kv_conv(const float *raw, const float *state, const float *w, const float *dyn, size_t T, size_t C, size_t HD,
                     int mode, float den, float thr, float eps, float *out);
// Post-attention conv + residual: r = (boundary ? 0 : hin) + (conv(a) + a) / den.
void ink_ref_conv_res(const float *a, const float *state, const float *w, const float *hin, size_t T, size_t C, int boundary,
                      float den, float *r);
// Router (M:1152-1172): top K of sigmoid(W x) + bias (first index on ties), weights.
void ink_ref_router(const float *x, const float *W, const float *bias, size_t E, size_t H, size_t K, float scale, uint32_t *idx,
                    float *wts);
// Attention of query position pq over keys at positions lo..pq (K, V [pos][NKV * HD]): q [NH][HD]
// (prepared), r [NH][DREL], proj [DREL][EXT], sink [NH], gate [NH * HD] (pre-sigmoid); out [NH * HD].
void ink_ref_attn(const float *q, const float *r, const float *proj, const float *K, const float *V, const float *sink,
                  const float *gate, size_t NH, size_t NKV, size_t HD, size_t DREL, size_t EXT, size_t pq, size_t win, float *out);
// Dequantized rows (in values) of the expert tensors: NVFP4 (w [rows][in / 2] bytes, s [rows][in / 16],
// gfac) or MXFP8 (w [rows][in], s [rows][in / 32]).
void ink_ref_row_nv(const uint8_t *w, const uint8_t *s, float gfac, size_t in, size_t row, float *out);
void ink_ref_row_mx(const uint8_t *w, const uint8_t *s, size_t in, size_t row, float *out);
float ink_ref_e4m3(uint8_t b);
// Softcapped SwiGLU (M:760-771).
float ink_ref_swiglu(float g, float u, float gcap, float ucap);

#endif
