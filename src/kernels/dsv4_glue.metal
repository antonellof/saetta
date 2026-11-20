// DeepSeek-V4-Flash decode glue: the BF16 matvec (compressor and indexer projections,
// weights_proj, the head), BF16 conversions between the other kernels' formats, the shared
// expert's activation and the MoE output sum. BF16 values are carried in f32 (rounded to
// nearest even) or as raw ushort bit patterns where the next kernel reads them so.
//
// The numerics follow the checkpoint's reference (DeepSeek-V4-Flash-0731 at 7872f01,
// inference/model.py; MIT, Copyright (c) DeepSeek): the shared expert computes
// gate = bf16(w1 x), up = bf16(w3 x), up clamped to [-10, 10], gate = min(gate, 10),
// h = bf16(silu(gate) up); the MoE output is bf16(routed sum + bf16(shared output)).
#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)

inline float g_bf16r(float v) {
    uint u = as_type<uint>(v);
    if ((u & 0x7fffffffu) > 0x7f800000u) return v; // NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return as_type<float>(u & 0xffff0000u);
}
inline ushort g_bf16_bits(float v) {
    uint u = as_type<uint>(v);
    if ((u & 0x7fffffffu) > 0x7f800000u) return ushort((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return ushort(u >> 16);
}
inline float2 g_bf16_pair(uint w) { return float2(as_type<float>(w << 16), as_type<float>(w & 0xffff0000u)); }

struct GlueMvP {
    uint K;     // inputs per row, a multiple of 128
    uint N;     // rows
    uint shift; // byte offset of the matrix from the bound address, a multiple of 4
    uint round; // round the outputs to BF16
};

// y[r] = sum_k W[r][k] x[k] in f32 (BF16 weights [N][K], f32 x); one simdgroup per row, 8 rows
// per threadgroup of 256: grid ceil(N / 8).
kernel void dsv4_bf16_mv(device const uchar *w [[buffer(0)]], device const float4 *x [[buffer(1)]],
                         device float *y [[buffer(2)]], constant GlueMvP &p [[buffer(3)]],
                         uint g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    const uint r = g * 8 + sg;
    if (r >= p.N) return;
    device const uint *wr = (device const uint *)(w + p.shift) + (ulong)r * (p.K / 2);
    float acc = 0.0f;
    for (uint i = lane; i < p.K / 4; i += 32) {
        const float2 a = g_bf16_pair(wr[2 * i]), b = g_bf16_pair(wr[2 * i + 1]);
        const float4 xv = x[i];
        acc += ((a.x * xv.x + a.y * xv.y) + b.x * xv.z) + b.y * xv.w;
    }
    const float s = simd_sum(acc);
    if (lane == 0) y[r] = p.round ? g_bf16r(s) : s;
}

// Elementwise kernels: n values, grid ceil(n / 256) x 256.

kernel void dsv4_round_bf16(device float *x [[buffer(0)]], constant uint &n [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    if (i < n) x[i] = g_bf16r(x[i]);
}

kernel void dsv4_to_bf16(device const float *x [[buffer(0)]], device ushort *y [[buffer(1)]],
                         constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = g_bf16_bits(x[i]);
}

kernel void dsv4_from_bf16(device const ushort *x [[buffer(0)]], device float *y [[buffer(1)]],
                           constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = as_type<float>(uint(x[i]) << 16);
}

// Shared expert activation: g, u the w1 and w3 outputs (f32), h the BF16 input of w2.
kernel void dsv4_shared_act(device const float *g [[buffer(0)]], device const float *u [[buffer(1)]],
                            device float *h [[buffer(2)]], constant uint &n [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    const float gate = min(g_bf16r(g[i]), 10.0f);
    const float up = clamp(g_bf16r(u[i]), -10.0f, 10.0f);
    const float sig = precise::divide(1.0f, 1.0f + precise::exp(-gate));
    h[i] = g_bf16r((gate * sig) * up);
}

// MoE output: bf16(routed + bf16(shared)).
kernel void dsv4_moe_add(device const float *yr [[buffer(0)]], device const float *ys [[buffer(1)]],
                         device float *out [[buffer(2)]], constant uint &n [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = g_bf16r(yr[i] + g_bf16r(ys[i]));
}

// dst = src, n float4.
kernel void dsv4_copy4(device const float4 *src [[buffer(0)]], device float4 *dst [[buffer(1)]],
                       constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = src[i];
}
