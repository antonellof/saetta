// Prefill matmul on the M5 neural accelerators (Metal 4 tensor ops, matmul2d), half inputs,
// float accumulation and output: Y[m][n] = sum_k X[m][k] * W[n][k], X half [M][K], Y float [M][N].
// W is the MLX affine pack (BITS bits, GROUP-wide groups, unaligned tensors read bytewise).
// Adapted from qwental kernels/qmm.metal (MIT, Copyright (c) 2026 Ong Hoe Yuan; b1be701): the
// dequantize-a-weight-tile-to-threadgroup-half then multiply_accumulate structure and the
// cooperative-tensor write-out. Changed: group 32 or 64, unaligned byte reads, float output,
// shapes at run time. Not exact: half inputs (quality gate in `saetta verify`).
// Tile: TM = 64 rows of X, TN = 64 rows of W, K step KC = 64, 4 simdgroups (128-row tiles were slower).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

inline float bf16_mp(device const uchar *p, uint i) {
    const uint lo = p[2 * i], hi = p[2 * i + 1];
    return as_type<float>(((hi << 8) | lo) << 16);
}

constant constexpr uint MTN = 64, MKC = 64, MSG = 4, MNT = 32 * MSG;

// Rows m0 .. m_end of X (M rows in all) against rows n0 .. n0+63 of W.
template <int BITS, int GROUP, int MTM>
inline void qmmh_tile(device const uchar *w, device const uchar *scales, device const uchar *biases,
                      device half *X, device float *Y, uint K, uint N, uint M, uint m0, uint m_end, uint n0,
                      uint tid, threadgroup half *Wt) {
    using T2 = dextents<int32_t, 2>;
    tensor<device half, T2, tensor_inline> tX(X, T2(K, M));
    tensor<threadgroup half, T2, tensor_inline> tW(Wt, T2(MKC, MTN));
    constexpr auto desc = matmul2d_descriptor(MTM, MTN, MKC, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<MSG>> op;
    auto x0 = tX.slice(0, m0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(x0), decltype(tW), float>();
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0;
    const uint groups = K / GROUP;
    for (uint k0 = 0; k0 < K; k0 += MKC) {
        // dequantize W[n0 .. n0+63][k0 .. k0+63] to half: one group of 32 values per thread
        // (128 threads = 64 rows x 2 groups), the group's 16 bytes read as aligned words and
        // shifted across the tensor's misalignment, scale and bias loaded once (4-bit, group 32)
        static_assert(BITS == 4 && GROUP == 32, "qmmh_tile dequantizes 4-bit, group 32");
        {
            const uint n = tid / 2, part = tid % 2, k = k0 + part * 32, row = n0 + n;
            threadgroup half4 *dst = (threadgroup half4 *)(Wt + n * MKC + part * 32);
            if (row < N) {
                const half s = half(bf16_mp(scales, row * groups + k / 32)), b = half(bf16_mp(biases, row * groups + k / 32));
                const ulong addr = (ulong)(w + row * (K / 2) + k / 2);
                const uint sh = uint(addr & 3);
                device const uint *q = (device const uint *)(addr - sh);
                uint wd[4];
                if (sh == 0) {
                    for (uint i2 = 0; i2 < 4; i2++) wd[i2] = q[i2];
                } else {
                    uint prev = q[0];
                    for (uint i2 = 0; i2 < 4; i2++) { const uint nx = q[i2 + 1]; wd[i2] = (prev >> (8 * sh)) | (nx << (32 - 8 * sh)); prev = nx; }
                }
                for (uint i2 = 0; i2 < 4; i2++) {
                    const uint x = wd[i2];
                    dst[i2 * 2] = half4(half(x & 15u), half((x >> 4) & 15u), half((x >> 8) & 15u), half((x >> 12) & 15u)) * s + b;
                    dst[i2 * 2 + 1] = half4(half((x >> 16) & 15u), half((x >> 20) & 15u), half((x >> 24) & 15u), half(x >> 28)) * s + b;
                }
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

// Dense: grid (ceil(N/64), ceil(M/64)), 128 threads, 64-row tiles. dims = (K, N, M).
kernel void qmmh_4_32(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],
                      device const uchar *biases [[buffer(2)]], device half *X [[buffer(3)]],
                      device float *Y [[buffer(4)]], constant uint3 &dims [[buffer(5)]],
                      constant uint *shift [[buffer(6)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half Wt[MTN * MKC];
    const uint m0 = tg.y * 64;
    qmmh_tile<4, 32, 64>(w + shift[0], scales + shift[1], biases + shift[2], X, Y, dims.x, dims.y, dims.z,
                         m0, min(dims.z, m0 + 64u), tg.x * MTN, tid, Wt);
}

// Grouped routed experts: X rows sorted by expert (M rows in all); tiles[i] = (expert, first
// row, end row), at most 64 rows of one expert. stride: bytes per expert of w and of s/b.
kernel void qmmh_g_4_32(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],
                        device const uchar *biases [[buffer(2)]], device half *X [[buffer(3)]],
                        device float *Y [[buffer(4)]], constant uint3 &dims [[buffer(5)]],
                        constant uint *shift [[buffer(6)]], device const uint4 *tiles [[buffer(7)]],
                        device const uint *ntiles [[buffer(8)]], constant uint2 &stride [[buffer(9)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (tg.y >= ntiles[0]) return;
    threadgroup half Wt[MTN * MKC];
    const uint4 t = tiles[tg.y];
    qmmh_tile<4, 32, 64>(w + shift[0] + ulong(t.x) * stride.x, scales + shift[1] + ulong(t.x) * stride.y, biases + shift[2] + ulong(t.x) * stride.y,
                     X, Y, dims.x, dims.y, dims.z, t.y, t.z, tg.x * MTN, tid, Wt);
}

kernel void to_half(device const float *src [[buffer(0)]], device half *dst [[buffer(1)]], constant uint &n [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = half(src[i]);
}

// ---- QSA attention for prompt chunks on the accelerators ---------------------------------
// Adapted from qwental kernels/qsa_pf.metal qsa_flash_tr (MIT, Copyright (c) 2026 Ong Hoe Yuan; b1be701):
// one threadgroup per (kv head, query token), 8 simdgroups; the 12 query heads
// of a kv head as 16 rows; TR_QCH = 128 visible tokens per step, simdgroup s computes the full
// S [16 x 16] over 256 dims for its 16 tokens (kept in registers), row max and row sum exchanged
// once each, P written as half, then each simdgroup does tot = tot * f + P . V for its 32 dims.
// Selected token ids are double-buffered in threadgroup memory and K / V rows are read straight
// into the right-operand cooperative tensors (half4 per lane: the measured M5 lane layout).
// The sigmoid gate is applied at the write. Changed for saetta: float queries converted to half
// beforehand (qh), the dense / sparse rule and lists of saetta's indexer (all positions while
// n / 4 <= 512, else cnt[t] entries of sel), float output and gate, f16 KV cache layout.
// Not exact: half q, half P (prefill quality gate in `saetta verify`).
constant constexpr int TR_QCH = 128, TR_NSG = 8, TR_GQA = 12, TR_HD = 256, TR_NKV = 2, TR_NQ = 24;

struct QsaTrP { uint pos0, budget_blocks, ratio, maxsel; };

kernel void qsa_flash_tr(device const half *q [[buffer(0)]], device const half *K [[buffer(1)]], device const half *V [[buffer(2)]],
                         device const uint *sel_all [[buffer(3)]], device const uint *cnt [[buffer(4)]],
                         device float *out [[buffer(5)]], constant QsaTrP &a [[buffer(6)]], device const float *qg [[buffer(7)]],
                         uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int NTH = TR_NSG * 32, NS = TR_QCH / TR_NSG, DS = TR_HD / TR_NSG;
    threadgroup half p[16 * TR_QCH];
    threadgroup int cid[2][TR_QCH];
    threadgroup float pm[TR_NSG][16], ps[TR_NSG][16];
    const uint g = tg.x, t = tg.y;
    const uint n = a.pos0 + t + 1;
    const bool dense = n / a.ratio <= a.budget_blocks;
    const uint ntok = dense ? n : cnt[t];
    device const uint *sel = sel_all + (ulong)t * a.maxsel;
    const int nch = (int)((ntok + TR_QCH - 1) / TR_QCH);
    using T2 = dextents<int32_t, 2>;
    auto tok_of = [&](int c) -> int { return c < (int)ntok ? (dense ? c : (int)sel[c]) : -1; };

    constexpr auto dS = matmul2d_descriptor(16, NS, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto dO = matmul2d_descriptor(16, DS, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dS, execution_simdgroup> opS;
    matmul2d<dO, execution_simdgroup> opO;
    tensor<device half, T2, tensor_inline> tQ((device half *)q + ((ulong)t * TR_NQ + g * TR_GQA) * TR_HD, T2(TR_HD, TR_GQA));
    tensor<threadgroup half, T2, tensor_inline> tP(p, T2(TR_QCH, 16));
    auto rv = opO.template get_right_input_cooperative_tensor<half, half, float>();
    auto rk = opS.template get_right_input_cooperative_tensor<half, half, float>();
    auto tPh = tP.template slice<32, 16>(0, 0);
    auto tA0 = tQ.template slice<32, 16>(0, 0);
    auto tot = opO.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tPh)>,
                                                               __tensor_ops_detail::__remove_addrspace_t<decltype(rv)>, float>();
    for (uint16_t i = 0; i < tot.get_capacity(); ++i) tot[i] = 0.0f;
    const int d0 = sg * DS;
    const ulong rs = TR_NKV * TR_HD;
    device const half *Kg = K + g * TR_HD, *Vg = V + g * TR_HD + d0;
    const uint r0 = ((lane >> 1) & 3) + 4 * (lane >> 4);
    const bool wr = (lane & 9) == 0;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    for (int j = tid; j < TR_QCH; j += NTH) cid[0][j] = tok_of(j);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int ch = 0; ch < nch; ++ch) {
        const int c0 = ch * TR_QCH, nc = min(TR_QCH, (int)ntok - c0);
        threadgroup const int *cc_ = cid[ch & 1];
        auto s = opS.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA0)>,
                                                                 __tensor_ops_detail::__remove_addrspace_t<decltype(rk)>, float>();
        for (uint16_t i = 0; i < s.get_capacity(); ++i) s[i] = 0.0f;
        for (int k0 = 0; k0 < TR_HD; k0 += 32) {
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < rk.get_capacity(); i += 4) {
                const auto ix = rk.get_multidimensional_index(i);
                const int id = cc_[sg * NS + ix[1]];
                const half4 w = id >= 0 ? *(device const half4 *)(Kg + (ulong)id * rs + k0 + ix[0]) : half4(0.0h);
                rk[i] = w.x; rk[i + 1] = w.y; rk[i + 2] = w.z; rk[i + 3] = w.w;
            }
            auto tA = tQ.template slice<32, 16>(k0, 0);
            opS.run(tA, rk, s);
        }
        float x0 = -INFINITY, x1 = -INFINITY;
        for (uint16_t i = 0; i < s.get_capacity(); ++i) {
            const auto ix = s.get_multidimensional_index(i);
            const bool vis = s.is_valid_element(i) && ix[1] < TR_GQA && sg * NS + ix[0] < nc;
            const float v = vis ? s[i] * 0.0625f : -INFINITY; // 1 / sqrt(256)
            s[i] = v;
            if (ix[1] < 8) x0 = max(x0, v);
            else x1 = max(x1, v);
        }
        x0 = max(x0, simd_shuffle_xor(x0, 1));
        x0 = max(x0, simd_shuffle_xor(x0, 8));
        x1 = max(x1, simd_shuffle_xor(x1, 1));
        x1 = max(x1, simd_shuffle_xor(x1, 8));
        if (wr) { pm[sg][r0] = x0; pm[sg][r0 + 8] = x1; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ch + 1 < nch)
            for (int j = tid; j < TR_QCH; j += NTH) cid[(ch + 1) & 1][j] = tok_of(c0 + TR_QCH + j);
        float n0 = m0, n1 = m1;
        for (int k = 0; k < TR_NSG; ++k) { n0 = max(n0, pm[k][r0]); n1 = max(n1, pm[k][r0 + 8]); }
        const float f0 = n0 == -INFINITY ? 1.0f : exp(m0 - n0), f1 = n1 == -INFINITY ? 1.0f : exp(m1 - n1);
        float y0 = 0.0f, y1 = 0.0f;
        for (uint16_t i = 0; i < s.get_capacity(); ++i) {
            const auto ix = s.get_multidimensional_index(i);
            const bool lo = ix[1] < 8;
            const float ev = s[i] == -INFINITY ? 0.0f : exp(s[i] - (lo ? n0 : n1));
            if (lo) y0 += ev;
            else y1 += ev;
            if (s.is_valid_element(i)) p[ix[1] * TR_QCH + sg * NS + ix[0]] = half(ev);
        }
        y0 += simd_shuffle_xor(y0, 1);
        y0 += simd_shuffle_xor(y0, 8);
        y1 += simd_shuffle_xor(y1, 1);
        y1 += simd_shuffle_xor(y1, 8);
        if (wr) { ps[sg][r0] = y0; ps[sg][r0 + 8] = y1; }
        m0 = n0;
        m1 = n1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float z0 = 0.0f, z1 = 0.0f;
        for (int k = 0; k < TR_NSG; ++k) { z0 += ps[k][r0]; z1 += ps[k][r0 + 8]; }
        l0 = l0 * f0 + z0;
        l1 = l1 * f1 + z1;
        for (uint16_t i = 0; i < tot.get_capacity(); ++i) tot[i] *= tot.get_multidimensional_index(i)[1] < 8 ? f0 : f1;
        for (int cc = 0; cc < TR_QCH; cc += 32) {
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < rv.get_capacity(); i += 4) {
                const auto ix = rv.get_multidimensional_index(i);
                const int id = cc_[cc + ix[1]];
                const half4 w = id >= 0 ? *(device const half4 *)(Vg + (ulong)id * rs + ix[0]) : half4(0.0h);
                rv[i] = w.x; rv[i + 1] = w.y; rv[i + 2] = w.z; rv[i + 3] = w.w;
            }
            tPh = tP.template slice<32, 16>(cc, 0);
            opO.run(tPh, rv, tot);
        }
    }
    for (uint16_t i = 0; i < tot.get_capacity(); ++i)
        if (tot.is_valid_element(i)) {
            const auto ix = tot.get_multidimensional_index(i);
            const int h = ix[1];
            if (h >= TR_GQA) continue;
            const uint hh = g * TR_GQA + h, d = d0 + ix[0];
            const float o = tot[i] / (h < 8 ? l0 : l1);
            const float gate = qg[((ulong)t * TR_NQ + hh) * 2 * TR_HD + TR_HD + d];
            out[((ulong)t * TR_NQ + hh) * TR_HD + d] = o / (1.0f + exp(-gate));
        }
}

// ---- dense prompt matmuls: weights unpacked to f16 once per chunk, then an f16 GEMM ------
// After Strata metal/kernels/dense_mm.metal (MIT, Copyright (c) 2026 Niko1221 and the Strata contributors; 13eedf55): dequantize each
// dense matrix into its own f16 buffer per chunk, then matmul2d reads both operands from
// device memory (the probe measured 63 TFLOP/s for that shape; the fused tile-dequant kernel
// above reaches about 47). wh [N][K] half; one thread per 8 weights (one 32-bit word).
kernel void dq_f16_4_32(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]], device const uchar *biases [[buffer(2)]],
                        device half4 *wh [[buffer(3)]], constant uint2 &d [[buffer(4)]] /* K, N */, constant uint *shift [[buffer(5)]],
                        uint e [[thread_position_in_grid]]) {
    const uint K = d.x, words = K / 8;
    if (e >= d.y * words) return;
    const uint n = e / words, kw = e % words, g = (kw * 8) / 32, groups = K / 32;
    const ulong addr = (ulong)(w + shift[0] + (ulong)n * (K / 2) + kw * 4);
    const uint sh = uint(addr & 3);
    device const uint *q = (device const uint *)(addr - sh);
    const uint x = sh == 0 ? q[0] : ((q[0] >> (8 * sh)) | (q[1] << (32 - 8 * sh)));
    const half s = half(bf16_mp(scales + shift[1], n * groups + g)), b = half(bf16_mp(biases + shift[2], n * groups + g));
    wh[e * 2] = half4(half(x & 15u), half((x >> 4) & 15u), half((x >> 8) & 15u), half((x >> 12) & 15u)) * s + b;
    wh[e * 2 + 1] = half4(half((x >> 16) & 15u), half((x >> 20) & 15u), half((x >> 24) & 15u), half(x >> 28)) * s + b;
}

// Y[M][N] float = X[M][K] half . Wh[N][K]^T; tile 64 (M) x 32 (N) on one simdgroup (the fast
// shape in the probe). grid (ceil(N / 32), ceil(M / 64)), 32 threads. dims = (K, N, M).
kernel void mmh_dense(device half *X [[buffer(0)]], device half *Wh [[buffer(1)]], device float *Y [[buffer(2)]],
                      constant uint3 &dims [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {
    using T2 = dextents<int32_t, 2>;
    const int K = dims.x, N = dims.y, M = dims.z;
    tensor<device half, T2, tensor_inline> tX(X, T2(K, M));
    tensor<device half, T2, tensor_inline> tW(Wh, T2(K, N));
    tensor<device float, T2, tensor_inline> tY(Y, T2(N, M));
    constexpr auto desc = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<1>> op;
    auto mX = tX.slice(0, tg.y * 64);
    auto mW = tW.slice(0, tg.x * 32);
    auto mY = tY.slice(tg.x * 32, tg.y * 64);
    op.run(mX, mW, mY);
}

// A BF16 matrix [N][K] to f16 (the prompt-reading f16 GEMM reads it): exact for values in the
// f16 normal range (routers). dims = (K, N).
kernel void dq_bf16_f16(device const uchar *w [[buffer(0)]], device half *wh [[buffer(1)]], constant uint2 &d [[buffer(2)]],
                        constant uint &shift [[buffer(3)]], uint e [[thread_position_in_grid]]) {
    if (e >= d.x * d.y) return;
    device const uchar *p = w + shift + (ulong)e * 2; // bytewise: pack tensors may start at odd offsets
    wh[e] = half(as_type<float>((uint(p[0]) | (uint(p[1]) << 8)) << 16));
}
