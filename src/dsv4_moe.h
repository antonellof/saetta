// DeepSeek-V4-Flash routed experts on the GPU (src/kernels/dsv4_moe.metal): decode and
// verification chunks of up to 8 rows. Scalar reference in dsv4_ref.h.
//
// Dispatches, per chunk of T rows with 6 (slot, route weight) pairs per row:
//   dsv4_act_quant    (x, xq, K)                         grid (K / 128, T), 32 threads
//   dsv4_moe_gate_up  (W, S, xq, slots, rw, hq, hraw, P) grid (DSV4_INTER / 128, 6 T), 256 threads
//   dsv4_moe_down     (W, S, hq, slots, y, P)            grid (DSV4_DIM / 16), 256 threads
// W and S are the expert cache's weights and scales regions; each row's pairs are in ascending
// expert id and its slots distinct.
#ifndef SAETTA_DSV4_MOE_H
#define SAETTA_DSV4_MOE_H

#include "base.h"

// src/kernels/dsv4_moe.metal, NUL terminated.
extern const char dsv4_moe_src[];

// The kernels' constant Dsv4MoeP (same layout).
typedef struct {
    uint32_t T;        // rows, 1..8
    uint32_t w_stride; // bytes per slot in W
    uint32_t s_stride; // bytes per slot in S
    uint32_t keep_h;   // gate_up also writes h before quantization to hraw
    uint32_t w_off[4]; // offsets of w1, w2, w3 in a slot of W (16-byte multiples), unused
    uint32_t s_off[4]; // the same in S
} Dsv4MoeP;

// The checkpoint's order inside an expert (w1, w2, w3, as its shards store them).
#define DSV4_W1_OFF 0u
#define DSV4_W2_OFF 4194304u
#define DSV4_W3_OFF 8388608u
#define DSV4_S1_OFF 0u
#define DSV4_S2_OFF 262144u
#define DSV4_S3_OFF 524288u
#define DSV4_EXPERT_W_BYTES 12582912u // three 4 MiB FP4 matrices
#define DSV4_EXPERT_S_BYTES 786432u   // their E8M0 scales

#endif
