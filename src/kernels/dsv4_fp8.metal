// DeepSeek-V4-Flash FP8 dense linears: W F8_E4M3 [N][K] with one E8M0 scale per 128 x 128
// block (2^(v - 127), [N / 128][K / 128]), y[t][n] = sum_k W[n][k] x[t][k], f32 output.
//
// Quantized inputs (the model's FP8 linears): x goes through dsv4_act_quant first, which
// writes the e4m3 code values as half (exact: every finite e4m3 value is a normal half) and
// one power-of-two scale per 128 inputs, as the reference act_quant does (ue8m0 scale
// 2^ceil(log2(amax / 448)), amax floored at 1e-4, x / s clamped to +-448, rounded to
// nearest even). The products are then code(w) code(x) sa sb, summed in f32. sa and sb are
// powers of two, so scaling before or after a block's partial sum gives the same f32
// results; only the summation order differs from the reference.
//
// Raw inputs (wo_a: dequantized to BF16 and run in BF16 by the reference): x is f32 and the
// weight is used as code(w) sb, which equals that BF16 dequantization exactly; grouped,
// output rows [g rpg, (g + 1) rpg) read inputs [g K, (g + 1) K) of each x row.
//
// Decode (T <= 8 per pass): every weight byte is read once for all TT tokens of the pass.
// Prefill: Metal 4 tensor ops (matmul2d) on half codes, one 128-deep K block per step into
// a fresh f32 tile, scaled by sa sb and added to the f32 accumulator, as the reference
// fp8_gemm does per K block.
//
// Weight and scale tensors may start at any byte (wshift, sshift from the binding); the
// kernels read the weight as 32-bit words, shifted across the misalignment.
// Not handled: the e4m3 NaN codes 0x7f and 0xff decode as +-480, E8M0 0xff as infinity.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// One e4m3 byte (low 8 bits of b) to its value.
inline float dsv4_e4m3(uint b) {
    const uint a = b & 0x7fu;
    const float v = a < 8u ? float(a) * 0x1p-9f : as_type<float>((a << 20) + (120u << 23));
    return as_type<float>(as_type<uint>(v) | ((b & 0x80u) << 24));
}

inline float4 dsv4_e4m3x4(uint w) {
    return float4(dsv4_e4m3(w), dsv4_e4m3(w >> 8), dsv4_e4m3(w >> 16), dsv4_e4m3(w >> 24));
}

inline float dsv4_e8m0(uint v) { return as_type<float>(v << 23); }

// Nearest-even e4m3 value of v, |v| <= 448 (the caller clamps). Bit operations and rint
// only, so fast math cannot change the result.
inline float dsv4_round_e4m3(float v) {
    const float a = fabs(v);
    float q;
    if (a < 0x1p-6f) {
        q = rint(a * 512.0f) * 0x1p-9f; // subnormal codes: multiples of 2^-9
    } else {
        uint u = as_type<uint>(a);
        u += 0x7ffffu + ((u >> 20) & 1u); // 3 mantissa bits, ties to even
        q = as_type<float>(u & ~0xfffffu);
    }
    return copysign(q, v);
}

// The 32-bit word at byte offset off of base, where (base + off) & 3 == sh for every off used.
inline uint dsv4_word(device const uchar *base, ulong off, uint sh) {
    device const uint *q = (device const uint *)(base + off - sh);
    return sh == 0 ? q[0] : ((q[0] >> (8 * sh)) | (q[1] << (32 - 8 * sh)));
}

// ---- activation quantization ---------------------------------------------------------------
// xc[t][k] = e4m3 code value of x[t][k] / s[t][k / 128] (half), xs[t][k / 128] = s.
// One simdgroup per 128-wide block: grid (K / 128, T) x 32. K % 128 == 0, x rows ldx_in floats
// apart (ldx_in % 4 == 0), xc rows K halfs apart, xs rows K / 128 floats apart.
struct Dsv4AqP {
    uint K, ldx_in;
};

kernel void dsv4_act_quant(device const float *x [[buffer(0)]], device half *xc [[buffer(1)]],
                           device float *xs [[buffer(2)]], constant Dsv4AqP &p [[buffer(3)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint kb = tg.x, t = tg.y;
    const float4 v = *(device const float4 *)(x + (ulong)t * p.ldx_in + kb * 128 + lane * 4);
    float amax = simd_max(max(max(fabs(v.x), fabs(v.y)), max(fabs(v.z), fabs(v.w))));
    amax = max(amax, 1e-4f);
    // 2^ceil(log2(amax / 448)) from the bits of amax * (1 / 448), as the reference does
    const uint u = as_type<uint>(amax * (1.0f / 448.0f));
    const int e = int((u >> 23) & 0xffu) - 127 + ((u & 0x7fffffu) != 0u ? 1 : 0);
    const float s = as_type<float>(uint(e + 127) << 23), inv = as_type<float>(uint(127 - e) << 23);
    const float4 q = clamp(v * inv, -448.0f, 448.0f); // v * inv == v / s: s is a power of two
    *(device half4 *)(xc + (ulong)t * p.K + kb * 128 + lane * 4) =
        half4(half(dsv4_round_e4m3(q.x)), half(dsv4_round_e4m3(q.y)), half(dsv4_round_e4m3(q.z)), half(dsv4_round_e4m3(q.w)));
    if (lane == 0) xs[(ulong)t * (p.K / 128) + kb] = s;
}

// ---- decode matvec --------------------------------------------------------------------------
// K: inputs per row; N: rows; ldx: x row stride (elements; K for plain linears, groups x K for
// wo_a); rpg: rows per input group (N when ungrouped); ldy: y row stride.
struct Dsv4MvP {
    uint K, N, T, ldx, rpg, ldy, wshift, sshift;
};

constant constexpr uint MV_R = 4, MV_SG = 4, MV_ROWS = MV_R * MV_SG;

// Rows tg.x * 16 .. +15 (MV_R per simdgroup), tokens tg.y * TT .. +TT-1. Each step of the K loop
// is one 128-wide block: lane l reads 4 weight bytes of each of its rows at 4 l, so the four
// rows' 128 bytes are read once for every token. Requires K % 128 == 0, N % 16 == 0,
// rpg % 16 == 0, ldx % 128 == 0.
template <uint TT, bool QX>
inline void dsv4_mv(device const uchar *w, device const uchar *ws, device const void *xv, device const float *xs,
                    device float *y, constant Dsv4MvP &p, uint2 tg, uint sg, uint lane) {
    const uint row0 = tg.x * MV_ROWS + sg * MV_R;
    const uint t0 = tg.y * TT;
    if (row0 >= p.N || t0 >= p.T) return;
    const uint nt = min(TT, p.T - t0);
    const uint nb = p.K / 128;
    const ulong xoff = (ulong)(row0 / p.rpg) * p.K;
    device const uchar *wb = w + p.wshift;
    const uint sh = uint((ulong)wb & 3u);
    device const uchar *sbp = ws + p.sshift + (ulong)(row0 / 128) * nb;
    float acc[MV_R][TT];
    for (uint r = 0; r < MV_R; r++)
        for (uint t = 0; t < TT; t++) acc[r][t] = 0.0f;
    for (uint kb = 0; kb < nb; kb++) {
        const float sb = dsv4_e8m0(sbp[kb]);
        float4 xq[TT];
        for (uint t = 0; t < TT; t++) {
            xq[t] = float4(0.0f);
            if (t < nt) {
                const ulong xi = (ulong)(t0 + t) * p.ldx + xoff + kb * 128 + lane * 4;
                if (QX) {
                    const float c = xs[(ulong)(t0 + t) * (p.ldx / 128) + xoff / 128 + kb] * sb;
                    xq[t] = float4(*(device const half4 *)((device const half *)xv + xi)) * c;
                } else {
                    xq[t] = *(device const float4 *)((device const float *)xv + xi) * sb;
                }
            }
        }
        for (uint r = 0; r < MV_R; r++) {
            const float4 wv = dsv4_e4m3x4(dsv4_word(wb, (ulong)(row0 + r) * p.K + kb * 128 + lane * 4, sh));
            for (uint t = 0; t < TT; t++) acc[r][t] += dot(wv, xq[t]);
        }
    }
    for (uint r = 0; r < MV_R; r++)
        for (uint t = 0; t < TT; t++) {
            const float v = simd_sum(acc[r][t]);
            if (lane == 0 && t < nt) y[(ulong)(t0 + t) * p.ldy + row0 + r] = v;
        }
}

// Quantized x: xc half codes [T][ldx], xs [T][ldx / 128]. Grid (N / 16, ceil(T / TT)) x 128.
#define DSV4_MV_Q(TT)                                                                                          \
    kernel void dsv4_fp8_mv_q##TT(device const uchar *w [[buffer(0)]], device const uchar *ws [[buffer(1)]],   \
                                  device const half *xc [[buffer(2)]], device const float *xs [[buffer(3)]],  \
                                  device float *y [[buffer(4)]], constant Dsv4MvP &p [[buffer(5)]],          \
                                  uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                                  uint lane [[thread_index_in_simdgroup]]) {                                  \
        dsv4_mv<TT, true>(w, ws, xc, xs, y, p, tg, sg, lane);                                                  \
    }
// Raw f32 x [T][ldx] (wo_a). Buffer 3 is unused.
#define DSV4_MV_R(TT)                                                                                          \
    kernel void dsv4_fp8_mv_r##TT(device const uchar *w [[buffer(0)]], device const uchar *ws [[buffer(1)]],   \
                                  device const float *x [[buffer(2)]], device float *y [[buffer(4)]],          \
                                  constant Dsv4MvP &p [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], \
                                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
        dsv4_mv<TT, false>(w, ws, x, nullptr, y, p, tg, sg, lane);                                             \
    }
DSV4_MV_Q(1)
DSV4_MV_Q(2)
DSV4_MV_Q(4)
DSV4_MV_Q(8)
DSV4_MV_R(1)
DSV4_MV_R(2)
DSV4_MV_R(4)
DSV4_MV_R(8)

// ---- prefill matmul on the neural accelerators ------------------------------------------------
// Tile 64 rows of X (tokens) x 64 rows of W, 4 simdgroups. Per 128-deep K block: the W tile is
// decoded to half codes in threadgroup memory, matmul2d writes code(x) . code(w) into a fresh
// f32 tile (multiply mode), and each element is added to the accumulator times sa[m] sb
// (QX) or sb (raw). Requires K % 128 == 0, N % 64 == 0, rpg % 64 == 0, ldx % 128 == 0.
// After the 64 x 64 x 64 half tile of qmm_mpp.metal (this repo), itself adapted from qwental
// kernels/qmm.metal (MIT, Copyright (c) 2026 Ong Hoe Yuan; b1be701).
struct Dsv4MmP {
    uint K, N, M, ldx, rpg, ldy, wshift, sshift;
};

constant constexpr uint MM_T = 64, MM_KB = 128;

template <bool QX, typename XT>
inline void dsv4_mm(device const uchar *w, device const uchar *ws, device XT *X, device const float *xs,
                    device float *Y, constant Dsv4MmP &p, uint2 tg, uint tid, threadgroup half *Wt,
                    threadgroup float *sc) {
    using T2 = dextents<int32_t, 2>;
    const uint n0 = tg.x * MM_T, m0 = tg.y * MM_T;
    const uint nb = p.K / MM_KB;
    const uint xoff = (n0 / p.rpg) * p.K;
    tensor<device XT, T2, tensor_inline> tX(X, T2(p.ldx, p.M));
    tensor<threadgroup half, T2, tensor_inline> tW(Wt, T2(MM_KB, MM_T));
    constexpr auto desc = matmul2d_descriptor(MM_T, MM_T, MM_KB, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto x0 = tX.slice(0, m0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(x0), decltype(tW), float>();
    auto part = op.template get_destination_cooperative_tensor<decltype(x0), decltype(tW), float>();
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0.0f;
    device const uchar *wb = w + p.wshift;
    const uint sh = uint((ulong)wb & 3u);
    device const uchar *sbp = ws + p.sshift + (ulong)(n0 / 128) * nb;
    const uint wn = tid / 2, wc = (tid % 2) * 64; // this thread decodes W row n0 + wn, bytes wc .. wc+63
    for (uint kb = 0; kb < nb; kb++) {
        {
            threadgroup half4 *dst = (threadgroup half4 *)(Wt + wn * MM_KB + wc);
            const ulong base = (ulong)(n0 + wn) * p.K + kb * MM_KB + wc;
            for (uint i = 0; i < 16; i++) dst[i] = half4(dsv4_e4m3x4(dsv4_word(wb, base + 4 * i, sh)));
        }
        const float sb = dsv4_e8m0(sbp[kb]);
        if (tid < MM_T) {
            float s = sb;
            if (QX) s = m0 + tid < p.M ? xs[(ulong)(m0 + tid) * (p.ldx / 128) + xoff / 128 + kb] * sb : 0.0f;
            sc[tid] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto xk = tX.slice(xoff + kb * MM_KB, m0);
        op.run(xk, tW, part);
#pragma unroll
        for (uint16_t i = 0; i < acc.get_capacity(); ++i)
            if (acc.is_valid_element(i)) acc[i] += part[i] * sc[part.get_multidimensional_index(i)[1]];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) {
            const auto ix = acc.get_multidimensional_index(i);
            const uint n = n0 + ix[0], m = m0 + ix[1];
            if (m < p.M && n < p.N) Y[(ulong)m * p.ldy + n] = acc[i];
        }
}

// Quantized x: xc half codes [M][ldx], xs [M][ldx / 128]. Grid (N / 64, ceil(M / 64)) x 128.
kernel void dsv4_fp8_mm_q(device const uchar *w [[buffer(0)]], device const uchar *ws [[buffer(1)]],
                          device half *xc [[buffer(2)]], device const float *xs [[buffer(3)]],
                          device float *Y [[buffer(4)]], constant Dsv4MmP &p [[buffer(5)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half Wt[MM_T * MM_KB];
    threadgroup float sc[MM_T];
    dsv4_mm<true>(w, ws, xc, xs, Y, p, tg, tid, Wt, sc);
}

// Raw f32 x [M][ldx] (wo_a), f32 x half products. Buffer 3 is unused.
kernel void dsv4_fp8_mm_r(device const uchar *w [[buffer(0)]], device const uchar *ws [[buffer(1)]],
                          device float *x [[buffer(2)]], device float *Y [[buffer(4)]],
                          constant Dsv4MmP &p [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half Wt[MM_T * MM_KB];
    threadgroup float sc[MM_T];
    dsv4_mm<false>(w, ws, x, nullptr, Y, p, tg, tid, Wt, sc);
}
