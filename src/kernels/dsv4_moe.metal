// DeepSeek-V4-Flash routed experts for decode and small verification chunks (T <= 8 rows).
//
// Per row t and each of its 6 selected experts (pair p = t * 6 + k, slot slots[p], route
// weight rw[p]):
//   xq   = act_quant(x[t])            FP8 e4m3 quantize-dequantize per 128, power-of-two scale
//   g, u = bf16(W1 xq), bf16(W3 xq)   FP4 e2m1 weights, one E8M0 scale per 32 along K, f32 sums
//   h    = bf16(silu(min(g, 10)) * clamp(u, -10, 10) * rw[p])
//   hq   = act_quant(h)
//   y[t] = sum over k = 0..5 of bf16(W2 hq), in f32, in pair order
// The caller orders each row's 6 pairs by ascending expert id (the reference sums experts in
// that order) and adds the shared expert.
//
// Three kernels: dsv4_act_quant (x to xq; its output also feeds the shared expert),
// dsv4_moe_gate_up (writes hq, quantized in the epilogue: a threadgroup owns one aligned block
// of 128 channels) and dsv4_moe_down (each threadgroup owns 16 output rows of every row t).
// Both expert kernels read each distinct slot's weights once per chunk: the threadgroups of a
// pair whose slot already appeared at an earlier pair exit, and the first pair's threadgroups
// apply the weights to every row that selected that slot. Slot ids must be distinct within a
// row (6 distinct experts).
//
// Expert cache layout: slot s holds an expert's weights at W + s * w_stride (w1, w2, w3 at the
// byte offsets w_off.xyz) and its scales at S + s * s_stride (s_off.xyz). Weights are
// [out][in / 2] bytes, element 2i in the low nibble; scales [out][in / 32] E8M0 bytes. Every
// weight row must start 16-byte aligned (slots and offsets are multiples of 16).
//
// The e2m1 lookup table in threadgroup memory follows ds4 (metal/moe.metal, MXFP4 matvec),
// MIT, copyright the ds4.c authors, the ggml authors and DeepSeek, commit
// b3f50180ec2c7324892be1f50bc83555ccb8d5e7.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint DSV4_DIM = 4096;   // hidden size, K of w1 and w3, N of w2
constant constexpr uint DSV4_INTER = 2048; // expert intermediate size
constant constexpr uint DSV4_TOPK = 6;
constant constexpr uint DSV4_MAXT = 8;     // rows per chunk
constant constexpr float DSV4_LIMIT = 10.0f;

struct Dsv4MoeP {
    uint T;        // rows in the chunk, 1..8
    uint w_stride; // bytes per slot in the weights buffer
    uint s_stride; // bytes per slot in the scales buffer
    uint keep_h;   // gate_up: also write h (before quantization) to hraw
    uint4 w_off;   // x, y, z: offsets of w1, w2, w3 in a slot's weights
    uint4 s_off;   // the same for the scales
};

constant float dsv4_e2m1_values[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                       -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

inline float dsv4_e8m0(uint e) { return as_type<float>(e == 0 ? 0x00400000u : e << 23); }

// Round to nearest even bf16, kept as f32 (finite inputs).
inline float dsv4_bf16(float v) {
    uint u = as_type<uint>(v);
    u += 0x7fffu + ((u >> 16) & 1u);
    return as_type<float>(u & 0xffff0000u);
}

// v in [-448, 448] rounded to the nearest e4m3 value, ties to even (bit arithmetic, so the
// result does not depend on fast-math).
inline float dsv4_e4m3(float v) {
    const float a = fabs(v);
    float r;
    if (a < 0.015625f) {
        r = rint(a * 512.0f) * 0.001953125f; // subnormals: steps of 2^-9
    } else {
        uint u = as_type<uint>(a);
        u += 0x7ffffu + ((u >> 20) & 1u); // keep 3 mantissa bits
        r = as_type<float>(u & 0xfff00000u);
    }
    return copysign(r, v);
}

// Power-of-two scale of a block with absolute maximum amax: 2^ceil(log2(max(amax, 1e-4) / 448)),
// the exponent taken from the bits of the f32 product as the reference does; inv = 1 / scale.
inline float dsv4_scale(float amax, thread float &inv) {
    const uint b = as_type<uint>(max(amax, 1e-4f) * (1.0f / 448.0f));
    const int e = int((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0 ? 1 : 0);
    inv = as_type<float>(uint(127 - e) << 23);
    return as_type<float>(uint(127 + e) << 23);
}

inline float dsv4_qdq(float v, float s, float inv) { return dsv4_e4m3(clamp(v * inv, -448.0f, 448.0f)) * s; }

// One simdgroup per block of 128: xq = act_quant(x), rows of K floats. Grid (K / 128, rows), 32 threads.
kernel void dsv4_act_quant(device const float *x [[buffer(0)]], device float *xq [[buffer(1)]], constant uint &K [[buffer(2)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const ulong base = ulong(tg.y) * K + tg.x * 128;
    const float4 v = ((device const float4 *)(x + base))[lane];
    const float amax = simd_max(max(max(fabs(v.x), fabs(v.y)), max(fabs(v.z), fabs(v.w))));
    float inv;
    const float s = dsv4_scale(amax, inv);
    ((device float4 *)(xq + base))[lane] = float4(dsv4_qdq(v.x, s, inv), dsv4_qdq(v.y, s, inv), dsv4_qdq(v.z, s, inv), dsv4_qdq(v.w, s, inv));
}

// The 32 weights of one e2m1 block (16 bytes) as 8 float4, element order.
inline void dsv4_fp4_block(uint4 q, threadgroup const float *lut, thread float4 *w) {
    const uint wd[4] = {q.x, q.y, q.z, q.w};
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4; i++) {
        const uint d = wd[i];
        w[2 * i] = float4(lut[d & 15u], lut[(d >> 4) & 15u], lut[(d >> 8) & 15u], lut[(d >> 12) & 15u]);
        w[2 * i + 1] = float4(lut[(d >> 16) & 15u], lut[(d >> 20) & 15u], lut[(d >> 24) & 15u], lut[d >> 28]);
    }
}

inline float dsv4_dot32(thread const float4 *w, device const float *x) {
    device const float4 *xv = (device const float4 *)x;
    float s = 0.0f;
#pragma clang loop unroll(full)
    for (uint j = 0; j < 8; j++) s += dot(w[j], xv[j]);
    return s;
}

// owner[p] = the first pair with the same slot as pair p. Threads < np write; then a barrier.
inline void dsv4_owners(device const uint *slots, uint np, uint tid, threadgroup uint *owner) {
    if (tid < np) {
        const uint s = slots[tid];
        uint o = tid;
        for (uint q = 0; q < tid; q++)
            if (slots[q] == s) { o = q; break; }
        owner[tid] = o;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pairs owned by pair pr, in pair order (at most one per row).
inline uint dsv4_matches(threadgroup const uint *owner, uint pr, uint np, thread uint *mp) {
    uint nm = 0;
    for (uint q = pr; q < np && nm < DSV4_MAXT; q++)
        if (owner[q] == pr) mp[nm++] = q;
    return nm;
}

// Gate and up of 128 channels for one slot, and the activation, route weight and quantization.
// Grid (DSV4_INTER / 128, 6 T), 256 threads (8 simdgroups of 16 channels). xq [T][DSV4_DIM];
// hq (and hraw when keep_h) [6 T][DSV4_INTER].
kernel void dsv4_moe_gate_up(device const uchar *W [[buffer(0)]], device const uchar *S [[buffer(1)]],
                             device const float *xq [[buffer(2)]], device const uint *slots [[buffer(3)]],
                             device const float *rw [[buffer(4)]], device float *hq [[buffer(5)]],
                             device float *hraw [[buffer(6)]], constant Dsv4MoeP &p [[buffer(7)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float lut[16];
    threadgroup uint owner[DSV4_TOPK * DSV4_MAXT];
    threadgroup float hs[DSV4_MAXT][128];
    const uint np = p.T * DSV4_TOPK, pr = tg.y;
    if (tid < 16) lut[tid] = dsv4_e2m1_values[tid];
    dsv4_owners(slots, np, tid, owner);
    if (pr >= np || owner[pr] != pr) return; // uniform over the threadgroup
    uint mp[DSV4_MAXT];
    const uint nm = dsv4_matches(owner, pr, np, mp);

    const ulong slot = slots[pr];
    device const uchar *w1 = W + slot * p.w_stride + p.w_off.x, *w3 = W + slot * p.w_stride + p.w_off.z;
    device const uchar *s1 = S + slot * p.s_stride + p.s_off.x, *s3 = S + slot * p.s_stride + p.s_off.z;
    constexpr uint RB = DSV4_DIM / 2, SB = DSV4_DIM / 32; // bytes per weight row, scales per row

    for (uint i = 0; i < 16; i++) {
        const uint c = sg * 16 + i, r = tg.x * 128 + c;
        float ag[DSV4_MAXT], au[DSV4_MAXT];
#pragma clang loop unroll(full)
        for (uint m = 0; m < DSV4_MAXT; m++) { ag[m] = 0.0f; au[m] = 0.0f; }
        for (uint b = lane; b < SB; b += 32) {
            float4 wg[8], wu[8];
            dsv4_fp4_block(*(device const uint4 *)(w1 + r * RB + b * 16), lut, wg);
            dsv4_fp4_block(*(device const uint4 *)(w3 + r * RB + b * 16), lut, wu);
            const float sg1 = dsv4_e8m0(s1[r * SB + b]), su3 = dsv4_e8m0(s3[r * SB + b]);
#pragma clang loop unroll(full)
            for (uint m = 0; m < DSV4_MAXT; m++) {
                if (m < nm) {
                    device const float *xb = xq + (mp[m] / DSV4_TOPK) * DSV4_DIM + b * 32;
                    ag[m] += dsv4_dot32(wg, xb) * sg1;
                    au[m] += dsv4_dot32(wu, xb) * su3;
                }
            }
        }
#pragma clang loop unroll(full)
        for (uint m = 0; m < DSV4_MAXT; m++) {
            if (m < nm) {
                float g = simd_sum(ag[m]), u = simd_sum(au[m]);
                if (lane == 0) {
                    g = min(dsv4_bf16(g), DSV4_LIMIT);
                    u = clamp(dsv4_bf16(u), -DSV4_LIMIT, DSV4_LIMIT);
                    const float a = precise::divide(g, 1.0f + precise::exp(-g)) * u;
                    hs[m][c] = dsv4_bf16(a * rw[mp[m]]);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // simdgroup m quantizes the block of pair mp[m]: 4 channels per lane
    for (uint m = 0; m < nm; m++) {
        if (m != sg) continue;
        const float4 v = ((threadgroup const float4 *)hs[m])[lane];
        const float amax = simd_max(max(max(fabs(v.x), fabs(v.y)), max(fabs(v.z), fabs(v.w))));
        float inv;
        const float s = dsv4_scale(amax, inv);
        const ulong o = ulong(mp[m]) * DSV4_INTER + tg.x * 128;
        ((device float4 *)(hq + o))[lane] = float4(dsv4_qdq(v.x, s, inv), dsv4_qdq(v.y, s, inv), dsv4_qdq(v.z, s, inv), dsv4_qdq(v.w, s, inv));
        if (p.keep_h) ((device float4 *)(hraw + o))[lane] = v;
    }
}

// Down projections summed over each row's 6 pairs: y[t][r] = sum_k bf16(W2[slot(t, k)][r] . hq[t * 6 + k]).
// Grid (DSV4_DIM / 16), 256 threads (8 simdgroups of 2 rows). y [T][DSV4_DIM].
kernel void dsv4_moe_down(device const uchar *W [[buffer(0)]], device const uchar *S [[buffer(1)]],
                          device const float *hq [[buffer(2)]], device const uint *slots [[buffer(3)]],
                          device float *y [[buffer(4)]], constant Dsv4MoeP &p [[buffer(5)]],
                          uint tgx [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint RPS = 2, RB = DSV4_INTER / 2, SB = DSV4_INTER / 32;
    threadgroup float lut[16];
    threadgroup uint owner[DSV4_TOPK * DSV4_MAXT];
    threadgroup float dv[8][RPS][DSV4_TOPK * DSV4_MAXT];
    const uint np = p.T * DSV4_TOPK;
    if (tid < 16) lut[tid] = dsv4_e2m1_values[tid];
    dsv4_owners(slots, np, tid, owner);
    const uint r0 = tgx * (8 * RPS) + sg * RPS;

    for (uint pr = 0; pr < np; pr++) {
        if (owner[pr] != pr) continue;
        uint mp[DSV4_MAXT];
        const uint nm = dsv4_matches(owner, pr, np, mp);
        const ulong slot = slots[pr];
        device const uchar *w2 = W + slot * p.w_stride + p.w_off.y;
        device const uchar *s2 = S + slot * p.s_stride + p.s_off.y;
        for (uint i = 0; i < RPS; i++) {
            const uint r = r0 + i;
            float acc[DSV4_MAXT];
#pragma clang loop unroll(full)
            for (uint m = 0; m < DSV4_MAXT; m++) acc[m] = 0.0f;
            for (uint b = lane; b < SB; b += 32) {
                float4 w[8];
                dsv4_fp4_block(*(device const uint4 *)(w2 + r * RB + b * 16), lut, w);
                const float sc = dsv4_e8m0(s2[r * SB + b]);
#pragma clang loop unroll(full)
                for (uint m = 0; m < DSV4_MAXT; m++)
                    if (m < nm) acc[m] += dsv4_dot32(w, hq + ulong(mp[m]) * DSV4_INTER + b * 32) * sc;
            }
#pragma clang loop unroll(full)
            for (uint m = 0; m < DSV4_MAXT; m++) {
                if (m < nm) {
                    const float v = simd_sum(acc[m]);
                    if (lane == 0) dv[sg][i][mp[m]] = dsv4_bf16(v);
                }
            }
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < p.T) {
        for (uint i = 0; i < RPS; i++) {
            float s = 0.0f;
            for (uint k = 0; k < DSV4_TOPK; k++) s += dv[sg][i][lane * DSV4_TOPK + k];
            y[lane * DSV4_DIM + r0 + i] = s;
        }
    }
}
