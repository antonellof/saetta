// DeepSeek-V4-Flash KV compressor and indexer on the GPU (src/kernels/dsv4_cmp.metal). Scalar
// reference and the exact semantics in dsv4_ref_cmp.h.
//
// Pipelines (NAME csa: ratio 4, d 512; hca: ratio 128, d 512; idx: the indexer's compressor,
// ratio 4, d 128), buffers in the order of the kernel signatures:
//   dsv4_cmp_step_NAME   (kv, score, ape, kv_state, score_state, normw, cs, cache, pool, P)  1 x d
//   dsv4_cmp_emit_NAME   (same)                                               n_emit x d threads
//   dsv4_cmp_state_NAME  (kv, score, ape, kv_state, score_state, -, -, -, -, P)  coff d / 256 x 256
//   dsv4_idx_q           (q, wp, cs, qx, w)                                   64 x 128 threads
//   dsv4_idx_score       (keys, qx, w, scores, P)                             ceil(n_vis / 32) x 128
//   dsv4_idx_topk        (scores, idx, P)                                     1 x 1024
#ifndef SAETTA_DSV4_CMP_H
#define SAETTA_DSV4_CMP_H

#include "base.h"

// src/kernels/dsv4_cmp.metal, NUL terminated.
extern const char dsv4_cmp_src[];

// The kernels' constant Dsv4CmpP (same layout).
typedef struct {
    uint32_t p0;        // chunk start (step: the token's position)
    uint32_t T;         // tokens in the chunk (step: 1)
    uint32_t e0;        // first entry emitted, p0 / ratio
    uint32_t n_emit;    // entries emitted: (p0 + T) / ratio - p0 / ratio
    uint32_t keep_pool; // also write the pooled values
    uint32_t pad[3];
} Dsv4CmpP;

// The kernels' constant Dsv4IdxP.
typedef struct {
    uint32_t n_vis;   // visible (scored) entries
    uint32_t n_total; // entries in the cache: the list has min(512, n_total) values
    int32_t offset;   // added to each valid index
    uint32_t pad;
} Dsv4IdxP;

#endif
