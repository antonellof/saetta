// DeepSeek-V4-Flash: scalar references for the low-precision formats and the routed experts
// (src/kernels/dsv4_moe.metal), used by the unit tests and later by verify.
//
// Formats: FP8 e4m3 (fn: no infinities, max 448), FP4 e2m1 (values 0, .5, 1, 1.5, 2, 3, 4, 6,
// sign in bit 3), E8M0 scales (2^(v - 127)). Activations into the FP8 and FP4 linears are
// quantized and dequantized per block along K with a power-of-two scale
// 2^ceil(log2(max(amax, 1e-4) / 448)) and round-to-nearest-even to e4m3.
//
// Routed expert, per input row x (bf16 values held as f32):
//   xq = act_quant(x, 128)
//   g = bf16(W1 xq), u = bf16(W3 xq): per 32-block an f32 sum of products, times the weight's
//       E8M0 scale, accumulated in f32 over the blocks
//   h = bf16(silu(min(g, 10)) * clamp(u, -10, 10) * route_weight), hq = act_quant(h, 128)
//   d = bf16(W2 hq)
// and the routed output of a row is the f32 sum of its 6 d, in ascending expert id.
#ifndef SAETTA_DSV4_REF_H
#define SAETTA_DSV4_REF_H

#include "base.h"

#define DSV4_DIM 4096
#define DSV4_INTER 2048
#define DSV4_TOPK 6
#define DSV4_ACT_BLOCK 128
#define DSV4_FP4_BLOCK 32
#define DSV4_SWIGLU_LIMIT 10.0f

float dsv4_e8m0(uint8_t v);       // 2^(v - 127); 0 gives 2^-127
float dsv4_e2m1(uint8_t code);    // the low 4 bits
float dsv4_e4m3(uint8_t code);    // NaN for 0x7f and 0xff
float dsv4_e4m3_round(float v);   // v in [-448, 448] to the nearest e4m3 value, ties to even
float dsv4_bf16_round(float v);   // to the nearest bf16, ties to even (NaN kept)
float dsv4_act_scale(float amax); // the power-of-two block scale above
// xq = act_quant(x): n values in blocks of block (n a multiple of block). xq may equal x.
void dsv4_act_quant(const float *x, float *xq, size_t n, size_t block);
// One output of an FP4 linear: row w (k / 2 bytes) with scales s (k / 32 bytes) against xq.
float dsv4_fp4_dot(const uint8_t *w, const uint8_t *s, const float *xq, size_t k);

// One expert's tensors: w1, w3 [DSV4_INTER][DSV4_DIM / 2] with scales [DSV4_INTER][DSV4_DIM / 32],
// w2 [DSV4_DIM][DSV4_INTER / 2] with scales [DSV4_DIM][DSV4_INTER / 32].
typedef struct {
    const uint8_t *w1, *s1, *w2, *s2, *w3, *s3;
} Dsv4Expert;

// h [DSV4_INTER] (bf16 values, before quantization) of one expert on xq [DSV4_DIM].
void dsv4_gate_up_ref(const Dsv4Expert *e, const float *xq, float route_w, float *h);
// d [DSV4_DIM] (bf16 values) of one expert on hq [DSV4_INTER].
void dsv4_down_ref(const Dsv4Expert *e, const float *hq, float *d);
// Routed output of T rows: x [T][DSV4_DIM], experts and rw [T][DSV4_TOPK] (each row's experts in
// ascending id), y [T][DSV4_DIM] in f32. Allocates nothing; scratch holds 2 * DSV4_DIM +
// 2 * DSV4_INTER floats.
void dsv4_moe_ref(const Dsv4Expert *experts, const float *rw, const float *x, size_t T, float *y, float *scratch);

#endif
