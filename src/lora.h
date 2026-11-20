// Run-time LoRA kernels (src/kernels/lora.metal): per target linear, after its base matvec,
// t = A x and y += scale * B t, with A f16 [Rt][in] (the A rows of the linears that share x,
// concatenated: q/k/v, gate/up) and B f16 [out][R] applied to the slice t[r0 .. r0+R].
// Here: the pipelines, the encode functions the forward calls, the scalar references the
// unit tests use (src/test/lora.c) and `saetta lora-bench` (decode overhead per token).
#ifndef SAETTA_LORA_H
#define SAETTA_LORA_H

#include "base.h"
#include "gpu.h"

// Kernel source, NUL terminated.
extern const char lora_src[];

// Largest token count of the small (decode, verification) kernels; larger T uses lora_mm.
#define LORA_SMALL_T 8u
// Largest R of one linear.
#define LORA_MAX_R 128u

typedef struct {
    sx_pipeline *down_small, *up_small, *mm;
} LoraPipes;

// Compiles the three kernels. Errors: "CompileFailed".
Err lora_init(Gpu *g, LoraPipes *p);

// t[tok][r] = sum_k A[r][k] * x[tok][k] for tok < T, r < rt. x rows ldx floats apart, t rows rt
// floats apart. Requires in % 32 == 0, ldx % 4 == 0, rt % 8 == 0, x bound at a multiple of
// 16 bytes and A at a multiple of 8 (the kernels read float4 and half4).
void lora_encode_down(const LoraPipes *p, sx_cmd *c, sx_arg A, sx_arg x, sx_arg t, uint32_t T, uint32_t in,
                      uint32_t ldx, uint32_t rt);

// One linear applied to a slice of t: y[tok][o] += scale * sum_r B[o][r] * t[tok][r0 + r].
typedef struct {
    sx_arg B, y;
    uint32_t out, r0, ldy; // ldy: floats between rows of y
} LoraUp;

// Applies n (1..3) linears with the same R that share t (rows rt floats apart). For T <= 8
// they go in one dispatch; otherwise one lora_mm dispatch each. Requires R % 8 == 0,
// R <= 128, r0 + R <= rt, B bound at a multiple of 8 bytes.
void lora_encode_up(const LoraPipes *p, sx_cmd *c, sx_arg t, uint32_t T, uint32_t rt, uint32_t R, float scale,
                    const LoraUp *ups, size_t n);

// Scalar references (f64 accumulation). A and B as f16 bit patterns.
void lora_ref_down(const _Float16 *A, const float *x, float *t, size_t T, size_t in, size_t ldx, size_t rt);
void lora_ref_up(const _Float16 *B, const float *t, float *y, size_t T, size_t rt, size_t r0, size_t R, size_t out,
                 size_t ldy, float scale);

// `saetta lora-bench`: the decode set of one Qwen3.8-27B token (64 layers, 16 with attention,
// R 32), random f16 weights (no model needed): us per call, GB/s, and the per-token total.
Err lora_bench(Arena *a, FILE *out);

#endif
