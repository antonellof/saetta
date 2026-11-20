// DeepSeek-V4-Flash: scalar references for the KV compressor and the indexer
// (src/kernels/dsv4_cmp.metal), used by the unit tests and later by verify. They follow the
// checkpoint's inference/model.py (R) and inference/kernel.py (K).
//
// Compressor (R:285-383), one per compressed layer (ratio 4, coff 2, d 512; ratio 128, coff 1,
// d 512) plus the indexer's own (ratio 4, coff 2, d 128, rotate). Per token: kv = wkv x and
// score = wgate x, [coff d] in f32 (computed elsewhere). The state is R's: kv_state and
// score_state [coff ratio][coff d] f32, initially kv 0 and score -inf (R:309-310).
//   decode (R:349-365): score += ape[pos % ratio]; the token goes to row (coff - 1) ratio +
//     pos % ratio; when (pos + 1) % ratio == 0 an entry is pooled: per channel c a softmax over
//     the rows (ratio 4: rows 0-3 channel c, the previous window's first half, rows 4-7 channel
//     d + c, the current window's second half; ratio 128: the 128 rows), dsv4c_exp(s - max) / sum,
//     then sum of kv * p, all in f32, rows in order; for ratio 4 rows 4-7 then move to 0-3.
//   prefill at start_pos 0 (R:331-348): the same pooled values for every closed window
//     (overlap_transform, R:313-320), and R's state for the tokens after them.
//   post (R:368-378): x = bf16(pooled); RMSNorm in f32 (sum of squares in the order below,
//     x * (1 / sqrt(ss / d + 1e-6)), times the weight, bf16); RoPE on the last 64 dims at the
//     window's first position (interleaved pairs, re c - im s, re s + im c in f32, bf16); then
//     the main compressor quantizes dims [0, d - 64) to FP8 e4m3 and back in blocks of 64
//     (act_quant, K:40-125: scale 2^ceil(log2(max(amax, 1e-4) * f32(1/448))) from the f32 bits,
//     clamp +-448, ties to even); the indexer's compressor applies a Hadamard transform to all
//     128 dims (Sylvester order, butterflies in f32, times f32(128^-0.5), bf16; the ordering of
//     fast_hadamard_transform is assumed, the library is not in the checkpoint) and FP4 e2m1
//     quantize-dequantize in blocks of 32 (K:128-200: amax floored at 6 * 2^-126, scale
//     2^ceil(log2(amax * f32(1/6))), clamp +-6, ties to even).
//   Sum of squares order: 32 partials, partial j the sequential sum over i = j, j + 32, ...;
//   then the partials summed in order (R's mean is a torch reduction of unknown order).
//
// Indexer (R:408-439), one query at position pos:
//   q [64][128] = wq_b(qr) as bf16 values (FP8 linear done elsewhere); RoPE on dims 64-127 at
//   pos, bf16; Hadamard and FP4 qdq per head as for the indexer's keys (R:417-422).
//   w[h] = bf16(wp[h] * f32(128^-0.5 * 64^-0.5)), wp = weights_proj(x) bf16 (R:424).
//   score[s] = bf16(sum_h bf16(relu(bf16(q[h] . k[s])) * w[h])), dot and head sums in f32,
//   in order (R:426-427, BF16 einsum and ops).
//   Entries s < n_vis are visible ((pos + 1) / 4 in decode, R:433; prefill masks the rest,
//   R:430-432). Output: k = min(512, n_total) indices: the visible entries by descending score,
//   ties to the lower index (a stable descending sort), truncated to k, each plus offset;
//   then -1 for the rest (R:433-438). -0 and +0 compare equal.
//
// RoPE for compressed layers (R:205-235): theta 160000, YaRN factor 16, original length 65536,
// beta_fast 32, beta_slow 1; f32 frequencies, angle f32(pos) * freq in f32, cos and sin of it.
// The GPU takes cos and sin from the host (dsv4_cmp_rope_cs), so both sides use the same values.
#ifndef SAETTA_DSV4_REF_CMP_H
#define SAETTA_DSV4_REF_CMP_H

#include "base.h"

#define DSV4C_ROPE 64        // rope dims, the last of each head
#define DSV4C_PAIRS 32       // rope frequency pairs
#define DSV4C_IDX_HEADS 64   // index_n_heads
#define DSV4C_IDX_DIM 128    // index_head_dim
#define DSV4C_TOPK 512       // index_topk
#define DSV4C_FP8_BLOCK 64   // FP8 simulation block of the main compressor
#define DSV4C_FP4_BLOCK 32   // FP4 simulation block of the indexer
#define DSV4C_EPS 1e-6f      // norm_eps

// One compressor's shape.
typedef struct {
    uint32_t ratio;  // 4 or 128
    uint32_t d;      // head dim: 512, or 128 for the indexer's compressor
    uint32_t coff;   // 2 for ratio 4 (overlap), else 1
    uint32_t rows;   // coff * ratio, rows of the state
    uint32_t cd;     // coff * d, width of kv, score, ape and the state
    bool rotate;     // the indexer's compressor: Hadamard + FP4 instead of FP8
} Dsv4Cmp;

Dsv4Cmp dsv4_cmp_shape(uint32_t ratio, uint32_t d, bool rotate);

float dsv4c_bf16(float v);             // round to the nearest bf16, ties to even (finite v)
// exp(x) for x <= 0 (the softmax's exp(s - max)): Cody-Waite reduction by ln 2 and a degree-7
// Taylor polynomial, within 2 ulp of exp; below -87 it returns exp(-87). R uses the CUDA expf;
// this one is written out so the kernel can repeat it operation for operation.
float dsv4c_exp(float x);
float dsv4c_e4m3_round(float v);       // |v| <= 448 to the nearest e4m3, ties to even
float dsv4c_e2m1_round(float v);       // |v| <= 6 to the nearest e2m1, ties to even
void dsv4c_fp8_qdq(float *x, size_t n);    // in place, blocks of DSV4C_FP8_BLOCK
void dsv4c_fp4_qdq(float *x, size_t n);    // in place, blocks of DSV4C_FP4_BLOCK
void dsv4c_hadamard(float *x, size_t n);   // in place: bf16(fwht(x) * f32(n^-0.5)), n a power of 2

// The 32 compressed-layer frequencies, and cos, sin at pos as cs[2 i], cs[2 i + 1].
void dsv4_cmp_rope_freqs(float freqs[DSV4C_PAIRS]);
void dsv4_cmp_rope_cs(const float freqs[DSV4C_PAIRS], uint32_t pos, float cs[2 * DSV4C_PAIRS]);

// R's initial state: kv 0, score -inf, [rows][cd] each.
void dsv4_cmp_state_init(const Dsv4Cmp *k, float *kv_state, float *score_state);
// Decode step at pos > 0 or the first token (R:349-365, which R also runs at pos 0 of a decode
// that starts there): returns true and writes pooled [d] when the window closes.
bool dsv4_cmp_ref_step(const Dsv4Cmp *k, float *kv_state, float *score_state, const float *kv,
                       const float *score, const float *ape, uint32_t pos, float *pooled);
// Prefill at start_pos 0 of T tokens (R:331-348): kv, score [T][cd]; writes pooled
// [T / ratio][d] and returns T / ratio; leaves R's state.
size_t dsv4_cmp_ref_prefill(const Dsv4Cmp *k, float *kv_state, float *score_state, const float *kv,
                            const float *score, const float *ape, size_t T, float *pooled);
// The entry from pooled [d] (R:368-378): normw [d] (bf16 values), cs at the window's first
// position. entry [d] holds bf16 values.
void dsv4_cmp_ref_post(const Dsv4Cmp *k, const float *pooled, const float *normw, const float *cs, float *entry);

// Indexer query: q [64][128] bf16 values in, qx [64][128] out (may equal q); cs at pos.
void dsv4_idx_ref_q(const float *q, const float *cs, float *qx);
// Head weights from weights_proj's bf16 output wp [64].
void dsv4_idx_ref_w(const float *wp, float *w);
// scores [n] of keys [n][128] (bf16 values) against qx and w.
void dsv4_idx_ref_scores(const float *qx, const float *w, const float *keys, size_t n, float *scores);
// The index list (k = min(512, n_total) values, returned); n_vis <= n_total.
size_t dsv4_idx_ref_topk(const float *scores, size_t n_vis, size_t n_total, int32_t offset, int32_t *idx);

#endif
