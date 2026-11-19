// Run-time quantization kernels (bits in {4, 5, 6, 8}, group in {32, 64, 128} per matrix), for
// packs that mix bit widths: src/kernels/qgen.metal (decode, verification chunks, exact prefill)
// and src/kernels/qgen_mpp.metal (prompt kernels on the accelerators), both after
// qgen_common.metal. Each mirrors a templated kernel (same buffers, grid and threadgroup shape)
// with (bits, group) as one more constant buffer at the end; see the kernel comments.
// Here: the sources, a scalar reference (any bit width, like the model's row dequantization),
// helpers for the unit tests (src/test/qgen.c, random packs at unaligned tensor starts), and
// `saetta qgen-bench` (generic against the 4-bit/32 templated kernels).
#ifndef SAETTA_QGEN_H
#define SAETTA_QGEN_H

#include "base.h"
#include "gpu.h"

// qgen_common.metal + "\n" + qgen.metal, and qgen_common.metal + "\n" + qgen_mpp.metal,
// NUL terminated.
extern const char qgen_src[];
extern const char qgen_mpp_src[];

// MvP of the model with the quantization of each matrix appended (buffer 17 of
// qmv_multi_l*_gen). Layout matches the kernel's constant struct.
typedef struct {
    uint32_t in, nmat, act;
    float act_scale;
    uint32_t rows[4], tg0[5], shift[12], bits[4], group[4];
} MvGP;

// Value i of a little-endian bits-bit stream (MLX affine packing, any bits in 2..8);
// row has row_len bytes.
uint32_t qgen_unpack(const uint8_t *row, size_t row_len, size_t i, size_t bits);

// `saetta qgen-bench`: the generic kernels at 4/32, 4/64 and 5/64 against the templated 4/32
// kernel, on random packs (no model needed). Prints its table to out (flushed per line).
Err qgen_bench(Arena *a, FILE *out);

// ---- helpers for the unit tests and the bench -----------------------------------------------

// std.Random.DefaultPrng (Xoshiro256++ seeded by SplitMix64) and the std.Random helpers
// used, bit for bit, so the random packs equal the Zig engine's.
typedef struct {
    uint64_t s[4];
} QgenPrng;
void qgen_prng_init(QgenPrng *r, uint64_t seed);
uint64_t qgen_prng_next(QgenPrng *r);
void qgen_prng_bytes(QgenPrng *r, uint8_t *buf, size_t n); // Random.bytes
float qgen_prng_float(QgenPrng *r);                         // Random.float(f32), [0, 1)
bool qgen_prng_boolean(QgenPrng *r);                        // Random.boolean

// A packed matrix [E][out][in] in three GPU buffers, each tensor at a byte offset (sh) from
// its buffer start, as pack tensors sit unaligned in the mapped files.
typedef struct {
    size_t E, out, in;
    uint32_t bits, group;
    sx_buffer *buf[3];
    uint32_t sh[3];
} QgenMat;
size_t qgen_mat_w_stride(const QgenMat *m); // bytes of one expert's weights
size_t qgen_mat_s_stride(const QgenMat *m); // bytes of one expert's scales (or biases)
uint8_t *qgen_mat_bytes(const QgenMat *m, int k); // tensor k (0 weight, 1 scales, 2 biases)
// Row r of expert e, dequantized in f32 (w = q * scale + bias); out has in values.
void qgen_mat_row(const QgenMat *m, size_t e, size_t r, float *out);
// The same rounded like the accelerator kernels: half(q) * half(s) + half(b) in half (fma).
void qgen_mat_row_half(const QgenMat *m, size_t e, size_t r, float *out);
void qgen_mat_args(const QgenMat *m, sx_arg out[3]);
void qgen_mat_deinit(const QgenMat *m);

// Distinct unaligned offsets per config and tensor, covering 1..15.
void qgen_offs_for(size_t ci, size_t k, uint32_t out[3]);

typedef struct {
    Arena *arena;
    Gpu g;           // a bare device (no pack files)
    QgenPrng rng;    // seeded 0x5ae77a
    StrMap pipes;    // name -> sx_pipeline * (names copied into the arena)
} QgenCtx;
Err qgen_ctx_init(Arena *a, QgenCtx *x);
// The pipeline for name from source, compiled once.
Err qgen_ctx_pipe(QgenCtx *x, const char *source, const char *name, sx_pipeline **out);
// A buffer of n floats uniform in [lo, hi) (at least 4 bytes).
Err qgen_ctx_floats(QgenCtx *x, size_t n, float lo, float hi, sx_buffer **out);
// A buffer of n halfs uniform in [-1, 1).
Err qgen_ctx_halfs(QgenCtx *x, size_t n, sx_buffer **out);
// Random weights (any bit pattern is a valid stream), scales of either sign around 0.01,
// biases in [-0.1, 0.1]; tensor k at byte offset offs[k]. in must be a multiple of group and 32.
Err qgen_ctx_mat(QgenCtx *x, size_t E, size_t out, size_t in, const uint32_t cfg[2], const uint32_t offs[3],
                 QgenMat *m);
// One dispatch of gx * gy threadgroups of threads threads, committed and waited.
// Errors: "DispatchFailed" (message printed).
Err qgen_ctx_run(QgenCtx *x, sx_pipeline *p, const sx_arg *args, size_t nargs, size_t gx, size_t gy, size_t threads);

#endif
