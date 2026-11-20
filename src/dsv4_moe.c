// DeepSeek-V4-Flash routed-expert kernel source (see dsv4_moe.h).
#include "dsv4_moe.h"

const char dsv4_moe_src[] = {
#embed "kernels/dsv4_moe.metal"
    , 0};

static_assert(sizeof(Dsv4MoeP) == 48, "Dsv4MoeP must match the Metal struct");
