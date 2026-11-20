// Scalar references of the DeepSeek-V4-Flash FP8 linears (src/kernels/dsv4_fp8.metal): the
// e4m3 and e8m0 codecs, the activation quantization and the block-scaled FP8 matmul, written
// to follow the checkpoint's reference step by step (f32 throughout, -ffp-contract=off).
// Used by the unit tests and later by verify.
#ifndef SAETTA_DSV4_REF_FP8_H
#define SAETTA_DSV4_REF_FP8_H

#include <stddef.h>
#include <stdint.h>

// Block size of the weight scales (both dimensions) and of the activation scales (along K).
#define DSV4_FP8_BLOCK 128u
// Largest finite e4m3 magnitude.
#define DSV4_FP8_MAX 448.0f

// e4m3fn (bias 7, no infinities): exponent 15 with mantissa 7 (0x7f, 0xff) is NaN.
float dsv4_e4m3_to_f32(uint8_t b);
// e8m0: 2^(v - 127); 0xff is NaN.
float dsv4_e8m0_to_f32(uint8_t v);
// Nearest e4m3, ties to even, saturating at +-448 (the reference clamps before the cast);
// NaN gives 0x7f. The sign of zero is kept.
uint8_t dsv4_f32_to_e4m3(float v);

// The activation scale of a block with absolute maximum amax: 2^ceil(log2(max(amax, 1e-4) /
// 448)), computed from the bits of max(amax, 1e-4) * (1 / 448) in f32.
float dsv4_act_scale(float amax);

// act_quant with power-of-two scales: for each row t < T and block b < K / 128,
// scales[t][b] = dsv4_act_scale(max |x[t][128 b ..]|) and codes[t][k] = e4m3(clamp(x / s,
// -448, 448)). x rows ldx floats apart; codes [T][K]; scales [T][K / 128]. K % 128 == 0.
void dsv4_ref_act_quant(const float *x, size_t T, size_t K, size_t ldx, uint8_t *codes, float *scales);

// fp8_gemm: y[t][n] = sum over K blocks b of (sum_{k in b} code(x[t][k]) code(w[n][k])) *
// (sa[t][b] * sb[n / 128][b]), the partial sum over the block in f32 in k order, then scaled
// and added to an f32 accumulator in block order. w [N][K] e4m3, ws [ceil(N / 128)][K / 128]
// e8m0, codes and scales from dsv4_ref_act_quant, y rows ldy floats apart. K % 128 == 0.
void dsv4_ref_fp8_gemm(const uint8_t *w, const uint8_t *ws, size_t N, size_t K, const uint8_t *codes,
                       const float *scales, size_t T, float *y, size_t ldy);

// The grouped BF16 matmul of wo_a: the weight dequantized as the checkpoint converter does
// (code(w) * sb, exact in BF16), f32 input x [T][groups * K] (rows ldx floats apart), output
// rows [g N / groups, (g + 1) N / groups) reading inputs [g K, (g + 1) K); f32 sum in k order.
void dsv4_ref_grouped(const uint8_t *w, const uint8_t *ws, size_t N, size_t K, size_t groups, const float *x,
                      size_t ldx, size_t T, float *y, size_t ldy);

#endif
