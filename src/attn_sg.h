// Prompt-chunk attention without tensor ops (src/kernels/attn_sg.metal): simdgroup_matrix
// kernels for GPUs without neural accelerators (M1 to M4), causal GQA for Qwen3.8-27B and QSA
// lists for Qwen3.8-Flash-Next; the kernel source, its loaders, the scalar reference the unit
// test uses (src/test/attn_sg.c) and `saetta attn-sg-bench`.
// Shapes: 24 query heads, head_dim 256; 27B 4 kv heads, Flash-Next 2; f16 KV cache rows
// [pos][nkv][256]; queries [T][24][256] f32; gate [T][24][query 256 | gate 256] f32.
#ifndef SAETTA_ATTN_SG_H
#define SAETTA_ATTN_SG_H

#include "base.h"
#include "gpu.h"
#include "qgen.h"

#define ATTN_SG_NQ 24
#define ATTN_SG_HD 256
#define ATTN_SG_BK 32 // keys per step

// attn_sg.metal, NUL terminated.
extern const char attn_sg_src[];

// Kernel parameters (AttnSgP in the kernel).
typedef struct {
    uint32_t pos0, T, nsplit, budget, ratio, maxsel;
} AttnSgP;

// The 27B causal kernels: direct (gated output) and partial (split-K, merged by attn_merge),
// nsg simdgroups per threadgroup (8 * nsg rows), the widest variant the pipelines allow.
typedef struct {
    sx_pipeline *dir, *part;
    uint32_t nsg;
} AttnSg27;
Err attn_sg_load_27(Gpu *g, AttnSg27 *out);
// The Flash-Next QSA kernel (64 threads per threadgroup).
#define ATTN_SG_FN_THREADS 64
Err attn_sg_load_fn(Gpu *g, sx_pipeline **out);

// Causal row blocks of kv head for a chunk of T tokens, qh query heads per kv head.
static inline size_t attn_sg_blocks(size_t T, size_t qh, size_t nsg) { return (T * qh + 8 * nsg - 1) / (8 * nsg); }
// Key splits for a causal chunk of T tokens whose last token sees nk keys: 1 while the row
// blocks alone give 256 threadgroups or more (or T > max_t, the partial buffer's rows), else up
// to max_split, at least 1024 keys each.
size_t attn_sg_nsplit(size_t T, size_t qh, size_t nkv, size_t nsg, size_t nk, size_t max_t, size_t max_split);

// Reference output of chunk token t for its 24 heads, out [24][256]: softmax(q k / 16) v over
// keys (nkeys entries; NULL: positions 0 .. nkeys - 1), times sigmoid(gate), in f64.
// q [T][24][256], gate [T][24][512] (gate in the second half of each head), K and V
// [pos][nkv][256].
void attn_sg_ref(const float *q, const float *gate, const _Float16 *K, const _Float16 *V, size_t nkv, size_t t,
                 const uint32_t *keys, size_t nkeys, float *out);

// Random QSA lists shaped like the indexer's for chunk tokens t < T at pos0 (test and bench
// data): tokens with n / ratio > budget visible positions get budget distinct random whole
// blocks of ratio positions in increasing order plus the n % ratio tail, at sel + t * maxsel,
// their count in cnt[t]; the others cnt[t] = 0 (dense). r: a qgen_prng state.
void attn_sg_lists(QgenPrng *r, size_t pos0, size_t T, size_t budget, size_t ratio, size_t maxsel, uint32_t *sel,
                   uint32_t *cnt);

// `saetta attn-sg-bench`: one layer's prompt attention for a 4,096-token chunk ending at 4K,
// 16K, 32K and 64K tokens of context: 27B attn_full_6h + attn_merge against attn_sg_27, then
// Flash-Next attn_split_gh + attn_merge against attn_sg_fn (random QSA lists of 512 blocks).
Err attn_sg_bench(Arena *a, FILE *out);

#endif
