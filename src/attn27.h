// Qwen3.8-27B prompt-chunk attention on the accelerators (src/kernels/dense_attn.metal):
// the kernel source, the scalar references the unit test uses (src/test/attn27.c) and
// `saetta attn-bench` (the new kernel against attn_full_6h + attn_merge, one layer).
// Shapes: 24 query heads over 4 kv heads, head_dim 256, f16 KV cache rows [pos][4][256],
// query projection rows [T][24][query 256 | gate 256].
#ifndef SAETTA_ATTN27_H
#define SAETTA_ATTN27_H

#include "base.h"
#include "gpu.h"

#define ATTN27_NQ 24
#define ATTN27_NKV 4
#define ATTN27_HD 256

// dense_attn.metal, NUL terminated.
extern const char attn27_src[];

// Kernel parameters (AttnFlP in the kernel).
typedef struct {
    uint32_t pos0, T, tq;
} Attn27P;

// Token stride of the permuted half queries for a block of qb tokens: T rounded up to qb.
static inline uint32_t attn27_tq(size_t T, size_t qb) { return (uint32_t)((T + qb - 1) / qb * qb); }

// Reference output of chunk token t (position pos0 + t) for its 24 heads, out [24][256]:
// softmax(q k / 16) v over positions 0 .. pos0 + t, times sigmoid(gate), in f64.
// q [T][24][256], qg [T][24][512] (gate in the second half of each head), K and V [pos][4][256].
// mirror: round q to half and each probability exp(s - m) to half as the kernel does (m the
// running row max after each 128-key step, the sum taken before rounding); otherwise exact.
void attn27_ref(const float *q, const float *qg, const _Float16 *K, const _Float16 *V, size_t pos0, size_t t,
                bool mirror, float *out);

// `saetta attn-bench`: one layer's prompt attention for a 4,096-token chunk ending at 4K, 16K,
// 32K and 64K tokens of context, attn_full_6h + attn_merge (512-query slices) against
// attn_fl_q6h + attn_fl_6_q8 and attn_fl_6_q16. Prints ms per layer, ms per token over the 16
// attention layers and effective TFLOP/s (causal QK and PV flops).
Err attn27_bench(Arena *a, FILE *out);

#endif
