// Operations, each with a GPU kernel and a scalar C reference used by tests.
#ifndef SAETTA_OPS_H
#define SAETTA_OPS_H

#include "base.h"
#include "gpu.h"
#include "weights.h"

// Kernel sources, NUL terminated (src/kernels/qmatvec.metal, src/kernels/qmm.metal).
extern const char ops_qmatvec_src[];
extern const char ops_qmm_src[];

static inline float ops_bf16_to_f32(uint16_t bits) {
    uint32_t u = (uint32_t)bits << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

// Scalar reference: y = W x for an affine-quantized MLX linear (any bits dividing 32),
// accumulated in f64 per row. x has in_features values, y out_features.
void ops_qmatvec_ref(const Pack *pack, const Linear *lin, const float *x, float *y);

typedef struct {
    sx_pipeline *q4, *q8, *mm4, *mm8;
} Ops;

// Compiles qmatvec_4_32, qmatvec_8_64, qmm_4_32, qmm_8_64. Errors: "CompileFailed".
Err ops_init(Gpu *g, Ops *out);

// GPU seconds for reps calls of qmatvec on the same linear in one command buffer.
// Errors: "DispatchFailed".
Err ops_qmatvec_timed(const Ops *o, Gpu *g, const Linear *lin, sx_buffer *x, sx_buffer *y, size_t reps, double *secs);

// GPU seconds for one qmatvec on each of the n linears in turn, in one command buffer; lanes
// selects the shape-adapted kernel (8, 64 or 256 lanes per row, 4-bit), 0 the default.
Err ops_qmatvec_timed_many(const Ops *o, Gpu *g, const Linear *lins, size_t n, sx_buffer *x, sx_buffer *y,
                           uint32_t lanes, double *secs);

// mv-bench: the verification-chunk kernel (qmv_rows, T input rows) over n linears.
// SAETTA_MV_ROWS starting with 'o' selects one row per simdgroup (qmv_rows1).
Err ops_rows_timed_many(const Ops *o, Gpu *g, const Linear *lins, size_t n, sx_buffer *x, sx_buffer *y, uint32_t T,
                        double *secs);

// Y[T][out] = X[T][in] W^T on the GPU (prefill matmul). Errors: "DispatchFailed".
Err ops_qmm(const Ops *o, Gpu *g, const Linear *lin, sx_buffer *x, sx_buffer *y, uint32_t T);

// y = W x on the GPU. x and y are float buffers of in and out elements.
// Errors: "UnsupportedQuant" (only 4/32 and 8/64), "DispatchFailed".
Err ops_qmatvec(const Ops *o, Gpu *g, const Linear *lin, sx_buffer *x, sx_buffer *y);

// Max relative error between a and b (n values each), relative to the largest |b|.
float ops_max_rel_err(const float *a, const float *b, size_t n);

#endif
