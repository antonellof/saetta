// DeepSeek-V4-Flash kernel sources (see dsv4_glue.h).
#include "dsv4_glue.h"

const char dsv4_hc_lib_src[] = {
#embed "kernels/dsv4_hc.metal"
    , 0};
const char dsv4_attn_lib_src[] = {
#embed "kernels/dsv4_attn.metal"
    , 0};
const char dsv4_glue_src[] = {
#embed "kernels/dsv4_glue.metal"
    , 0};
