// DeepSeek-V4-Flash kernel sources used by the decode pass (src/dsv4.c) that have no host
// helper of their own: mHC, norms and router (src/kernels/dsv4_hc.metal), attention
// (src/kernels/dsv4_attn.metal) and the glue kernels (src/kernels/dsv4_glue.metal: BF16
// matvec, BF16 conversions, shared-expert activation, MoE output sum). NUL terminated.
#ifndef SAETTA_DSV4_GLUE_H
#define SAETTA_DSV4_GLUE_H

#include "base.h"

extern const char dsv4_hc_lib_src[];
extern const char dsv4_attn_lib_src[];
extern const char dsv4_glue_src[];

// The constant GlueMvP of dsv4_bf16_mv.
typedef struct {
    uint32_t K, N, shift, round;
} Dsv4GlueMvP;

#endif
