// DeepSeek-V4-Flash compressor and indexer kernel source (see dsv4_cmp.h).
#include "dsv4_cmp.h"

const char dsv4_cmp_src[] = {
#embed "kernels/dsv4_cmp.metal"
    , 0};

static_assert(sizeof(Dsv4CmpP) == 32, "Dsv4CmpP must match the Metal struct");
static_assert(sizeof(Dsv4IdxP) == 16, "Dsv4IdxP must match the Metal struct");
