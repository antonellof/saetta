// DeepSeek-V4-Flash FP8 dense linears on the GPU (src/kernels/dsv4_fp8.metal): pipelines and
// encode functions. The scalar references are in dsv4_ref_fp8.h.
//
// A model FP8 linear is two steps: dsv4_encode_act_quant once per input (shared by every
// linear that reads the same x, e.g. wq_a and wkv, or shared w1 and w3), then
// dsv4_encode_linear per weight. wo_a (BF16 in the reference) uses dsv4_encode_grouped on the
// raw f32 input with the FP8 weight, which equals the reference's BF16 weight exactly.
#ifndef SAETTA_DSV4_FP8_H
#define SAETTA_DSV4_FP8_H

#include "base.h"
#include "gpu.h"

extern const char dsv4_fp8_src[];

// Largest T of one decode matvec pass; dsv4_encode_linear and dsv4_encode_grouped use the
// matvec up to this many tokens and the accelerator matmul above.
#define DSV4_FP8_SMALL_T 8u

typedef struct {
    sx_pipeline *aq;
    sx_pipeline *mv_q[4], *mv_r[4]; // passes of 1, 2, 4 and 8 tokens
    sx_pipeline *mm_q, *mm_r;
} Dsv4Fp8Pipes;

// Errors: "CompileFailed".
Err dsv4_fp8_init(Gpu *g, Dsv4Fp8Pipes *p);

// One FP8 weight: w [out][in] e4m3 and s [ceil(out / 128)][in / 128] e8m0, bound with byte
// shifts (gpu_tensor_arg). in % 128 == 0, out % 64 == 0.
typedef struct {
    sx_arg w, s;
    uint32_t wshift, sshift;
    uint32_t out, in;
} Dsv4Fp8Lin;

// xc[t][k] (half e4m3 code values, rows K apart) and xs[t][k / 128] (f32 powers of two) from
// x[t][k] (f32, rows ldx_in floats apart, 16-byte aligned), t < T. K % 128 == 0.
void dsv4_encode_act_quant(const Dsv4Fp8Pipes *p, sx_cmd *c, sx_arg x, uint32_t ldx_in, sx_arg xc, sx_arg xs,
                           uint32_t T, uint32_t K);

// y[t][n] = W x[t] for t < T from dsv4_encode_act_quant's xc, xs (K = l->in); y rows ldy apart.
void dsv4_encode_linear(const Dsv4Fp8Pipes *p, sx_cmd *c, const Dsv4Fp8Lin *l, sx_arg xc, sx_arg xs, sx_arg y,
                        uint32_t T, uint32_t ldy);

// wo_a: l->out rows in `groups` equal groups; group g reads inputs [g in, (g + 1) in) of x
// (f32 [T][groups * in], 16-byte aligned). out / groups % 64 == 0.
void dsv4_encode_grouped(const Dsv4Fp8Pipes *p, sx_cmd *c, const Dsv4Fp8Lin *l, uint32_t groups, sx_arg x, sx_arg y,
                         uint32_t T, uint32_t ldy);

// The two paths, for the tests: matvec passes of up to 8 tokens (any T), or the matmul (any T).
// xs is ignored when raw (x f32); ldx is the x row stride in elements, rpg the rows per group.
void dsv4_encode_mv(const Dsv4Fp8Pipes *p, sx_cmd *c, const Dsv4Fp8Lin *l, bool raw, sx_arg x, sx_arg xs,
                    uint32_t ldx, uint32_t rpg, sx_arg y, uint32_t T, uint32_t ldy);
void dsv4_encode_mm(const Dsv4Fp8Pipes *p, sx_cmd *c, const Dsv4Fp8Lin *l, bool raw, sx_arg x, sx_arg xs,
                    uint32_t ldx, uint32_t rpg, sx_arg y, uint32_t T, uint32_t ldy);

#endif
