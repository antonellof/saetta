// Run-time quantization variants of the prompt kernels in qmm_mpp.metal (half inputs on the M5
// neural accelerators, float accumulation), for packs that mix bit widths. Compiled after
// qgen_common.metal (src/qgen.zig: qgen.mpp_src). Same buffers, grid and threadgroup shape as
// the kernel each one mirrors; (bits, group) is one more constant uint2 at the end.
// Not exact, like the kernels they mirror: half weights and inputs.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

constant constexpr uint QG_TN = 64, QG_KC = 64, QG_SG = 4;

// 32 dequantized weights (one chunk of row `row` from column k, k % 32 == 0) as 8 half4.
template <int BITS>
inline void qg_chunk_half(device const uchar *w, device const uchar *scales, device const uchar *biases,
                          uint K, uint gs, uint row, uint k, threadgroup half4 *dst) {
    const uint g = row * (K >> gs) + (k >> gs);
    const half s = half(qg_bf16(scales, g)), b = half(qg_bf16(biases, g));
    float4 qv[8];
    qg_unpack32<BITS>(w + row * (K * BITS / 8) + k * BITS / 8, qv);
#pragma clang loop unroll(full)
    for (int i = 0; i < 8; i++) dst[i] = half4(qv[i]) * s + b;
}

// qmmh_tile of qmm_mpp.metal with the weight-tile dequantization at run-time bits and group.
template <int MTM>
inline void qg_qmmh_tile(uint bits, uint group, device const uchar *w, device const uchar *scales, device const uchar *biases,
                         device half *X, device float *Y, uint K, uint N, uint M, uint m0, uint m_end, uint n0,
                         uint tid, threadgroup half *Wt) {
    using T2 = dextents<int32_t, 2>;
    tensor<device half, T2, tensor_inline> tX(X, T2(K, M));
    tensor<threadgroup half, T2, tensor_inline> tW(Wt, T2(QG_KC, QG_TN));
    constexpr auto desc = matmul2d_descriptor(MTM, QG_TN, QG_KC, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<QG_SG>> op;
    auto x0 = tX.slice(0, m0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(x0), decltype(tW), float>();
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0;
    const uint gs = ctz(group);
    for (uint k0 = 0; k0 < K; k0 += QG_KC) {
        // 128 threads = 64 rows x 2 chunks of 32 values
        {
            const uint n = tid / 2, part = tid % 2, k = k0 + part * 32, row = n0 + n;
            threadgroup half4 *dst = (threadgroup half4 *)(Wt + n * QG_KC + part * 32);
            if (row < N) {
                QG_BITS(bits, qg_chunk_half<B>(w, scales, biases, K, gs, row, k, dst), for (uint i2 = 0; i2 < 8; i2++) dst[i2] = half4(NAN))
            } else {
                for (uint i2 = 0; i2 < 8; i2++) dst[i2] = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto xs = tX.slice(k0, m0);
        op.run(xs, tW, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) {
            auto ix = acc.get_multidimensional_index(i);
            const uint n = n0 + ix[0], m = m0 + ix[1];
            if (m < m_end && n < N) Y[(ulong)m * N + n] = acc[i];
        }
}

// qmmh_4_32: grid (ceil(N / 64), ceil(M / 64)) x 128. dims = (K, N, M). q = (bits, group).
kernel void qmmh_gen(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],
                     device const uchar *biases [[buffer(2)]], device half *X [[buffer(3)]],
                     device float *Y [[buffer(4)]], constant uint3 &dims [[buffer(5)]],
                     constant uint *shift [[buffer(6)]], constant uint2 &q [[buffer(7)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half Wt[QG_TN * QG_KC];
    const uint m0 = tg.y * 64;
    qg_qmmh_tile<64>(q.x, q.y, w + shift[0], scales + shift[1], biases + shift[2], X, Y, dims.x, dims.y, dims.z,
                     m0, min(dims.z, m0 + 64u), tg.x * QG_TN, tid, Wt);
}

// qmmh_g_4_32: grid (N / 64, tiles) x 128. tiles[i] = (expert, first row, end row). q = (bits, group).
kernel void qmmh_g_gen(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],
                       device const uchar *biases [[buffer(2)]], device half *X [[buffer(3)]],
                       device float *Y [[buffer(4)]], constant uint3 &dims [[buffer(5)]],
                       constant uint *shift [[buffer(6)]], device const uint4 *tiles [[buffer(7)]],
                       device const uint *ntiles [[buffer(8)]], constant uint2 &stride [[buffer(9)]],
                       constant uint2 &q [[buffer(10)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (tg.y >= ntiles[0]) return;
    threadgroup half Wt[QG_TN * QG_KC];
    const uint4 t = tiles[tg.y];
    qg_qmmh_tile<64>(q.x, q.y, w + shift[0] + t.x * stride.x, scales + shift[1] + t.x * stride.y, biases + shift[2] + t.x * stride.y,
                     X, Y, dims.x, dims.y, dims.z, t.y, t.z, tg.x * QG_TN, tid, Wt);
}

// dq_f16_4_32: one thread per 8 weights, N * K / 8 threads (any threadgroup size).
// wh [N][K] half. d = (K, N). q = (bits, group).
template <int BITS>
inline void qg_dq8(device const uchar *w, device const uchar *scales, device const uchar *biases, device half4 *wh,
                   uint K, uint gs, uint e) {
    const uint words = K / 8, n = e / words, kw = e % words;
    const uint g = n * (K >> gs) + ((kw * 8) >> gs);
    const half s = half(qg_bf16(scales, g)), b = half(qg_bf16(biases, g));
    float4 qa, qb;
    qg_unpack8<BITS>(w + (ulong)n * (K * BITS / 8) + kw * BITS, qa, qb);
    wh[e * 2] = half4(qa) * s + b;
    wh[e * 2 + 1] = half4(qb) * s + b;
}

kernel void dq_f16_gen(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]], device const uchar *biases [[buffer(2)]],
                       device half4 *wh [[buffer(3)]], constant uint2 &d [[buffer(4)]] /* K, N */, constant uint *shift [[buffer(5)]],
                       constant uint2 &q [[buffer(6)]],
                       uint e [[thread_position_in_grid]]) {
    if (e >= d.y * (d.x / 8)) return;
    QG_BITS(q.x, qg_dq8<B>(w + shift[0], scales + shift[1], biases + shift[2], wh, d.x, ctz(q.y), e),
            wh[e * 2] = half4(NAN); wh[e * 2 + 1] = half4(NAN))
}
