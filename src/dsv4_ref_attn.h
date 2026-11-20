// Scalar references for the DeepSeek-V4-Flash attention kernels (src/kernels/dsv4_attn.metal):
// RoPE, the query tail, the window KV write, sparse attention with the attention sink and the
// grouped wo_a projection, for one query position at a time (decode, and chunks of up to 8
// rows evaluated as consecutive decode steps).
//
// Written from the checkpoint's reference implementation (DeepSeek-V4-Flash-0731 at 7872f01,
// inference/model.py and inference/kernel.py, MIT, Copyright (c) DeepSeek). BF16 values are
// carried as f32 holding BF16 values; BF16 weights are raw uint16 bit patterns. The large dot
// products accumulate in f64 and are rounded to f32 once (the value an f32 accumulator
// approximates); everything the reference computes elementwise in f32 is done in f32 here,
// in the reference's order.
#ifndef SAETTA_DSV4_REF_ATTN_H
#define SAETTA_DSV4_REF_ATTN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum {
    DSV4A_NH = 64,                    // query heads (one shared KV head)
    DSV4A_HD = 512,                   // head_dim
    DSV4A_RD = 64,                    // rope dims, the last of each head
    DSV4A_NOPE = DSV4A_HD - DSV4A_RD, // 448, FP8-simulated in the KV
    DSV4A_NPAIR = DSV4A_RD / 2,       // 32 rotation pairs
    DSV4A_WIN = 128,                  // window ring slots
    DSV4A_BLK = 64,                   // sparse attention key block (probability rounding)
    DSV4A_QBLK = 64,                  // FP8 block of the KV nope dims
    DSV4A_NG = 8,                     // o_groups
    DSV4A_GD = DSV4A_NH * DSV4A_HD / DSV4A_NG, // 4096 inputs per group
    DSV4A_OL = 1024,                  // o_lora_rank (outputs per group)
    DSV4A_WOA_ROWS = DSV4A_NG * DSV4A_OL, // 8192
    DSV4A_MAX_T = 8,                  // rows per chunk
};
#define DSV4A_EPS 1e-6f
#define DSV4A_SCALE 0.04419417382415922f // f32(512^-0.5), softmax scale

static inline float dsv4a_bf16(float v) { // round to nearest BF16, ties to even (NaN kept)
    uint32_t u;
    memcpy(&u, &v, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return v;
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    memcpy(&v, &u, 4);
    return v;
}
static inline float dsv4a_bf16_val(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float v;
    memcpy(&v, &u, 4);
    return v;
}
static inline uint16_t dsv4a_bf16_bits(float v) { // of a value already rounded, or rounds it
    float r = dsv4a_bf16(v);
    uint32_t u;
    memcpy(&u, &r, 4);
    return (uint16_t)(u >> 16);
}

// precompute_freqs_cis frequencies (32): theta 10000 without YaRN (window-only layers) or
// theta 160000 with YaRN factor 16, original 65536, beta_fast 32, beta_slow 1 (compressed).
void dsv4a_rope_freqs(bool compressed, float freqs[DSV4A_NPAIR]);
// cos and sin of (f32) pos * freq for positions pos0 .. pos0 + n - 1: cs[(p - pos0)][i] =
// {cos, sin}, i.e. cs [n][32][2] floats (the kernels' table).
void dsv4a_rope_table(const float freqs[DSV4A_NPAIR], uint32_t pos0, size_t n, float *cs);
// apply_rotary_emb on 64 BF16 values x (interleaved pairs) with one position's cs [32][2];
// inverse uses the conjugate. Result rounded to BF16.
void dsv4a_ref_rope(float *x, const float *cs, bool inverse);

// Query tail of one row in place: q [64][512] (wq_b output, BF16) gets the unweighted
// per-head RMS norm with every op rounded to BF16, then RoPE.
void dsv4a_ref_q(float *q, const float *cs);
// KV of one row: x [512] (wkv output, BF16), w the kv_norm weight (BF16 bits); kv [512] =
// RMSNorm, RoPE on the last 64, FP8 e4m3 qdq of the first 448 in blocks of 64.
void dsv4a_ref_kv(const float *x, const uint16_t *w, const float *cs, float *kv);
float dsv4a_e4m3_round(float v);   // to the nearest e4m3 value, ties to even (|v| <= 448)
float dsv4a_act_scale(float amax); // 2^ceil(log2(max(amax, 1e-4) * f32(1/448))) from the f32 bits

// The decode window list for a query at position p (get_window_topk_idxs at start_pos p):
// 128 ring slots, -1 for an empty one.
void dsv4a_ref_window_list(uint32_t p, int32_t slots[DSV4A_WIN]);
// sparse_attn for one query position: q [64][512] BF16 values, keys[j] the j-th entry of the
// index list (512 values) or NULL for -1, L entries, sink [64]. o [64][512] BF16.
// Requires a valid key in the first block (always true for the window list).
void dsv4a_ref_sparse_attn(const float *q, const float *const *keys, size_t L, const float *sink, float *o);
// Output de-rotation in place: the last 64 dims of each of the 64 heads of o (BF16).
void dsv4a_ref_derotate(float *o, const float *cs);
// y [8192] = BF16(wo_a[g] o_g) per group g, o [32768] BF16, w [8192][4096] BF16 bits.
void dsv4a_ref_wo_a(const float *o, const uint16_t *w, float *y);

#endif
