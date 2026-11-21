// Qwen3.8-27B prompt-chunk attention on the M5 neural accelerators (Metal 4 tensor ops,
// matmul2d): causal GQA over the f16 KV cache, the cached prefix plus the chunk, gated by
// sigmoid(gate) at the write. Compiled alone (src/dense.c, src/attn27.c).
//
// Structure after qsa_flash_tr (src/kernels/qmm_mpp.metal, adapted from qwental
// kernels/qsa_pf.metal, MIT, Copyright (c) 2026 Ong Hoe Yuan; b1be701): S = Q K^T per tile
// on one simdgroup, K and V rows read straight into the right-operand cooperative tensors
// (half4 per lane, the M5 lane layout), the row max exchanged through threadgroup memory once
// per key step (row sums kept per lane, exchanged once at the end), P stored as half, then each simdgroup does
// tot = tot * f + (P V)^T = tot * f + V^T P^T for its dims: V is the left operand, read by the
// matmul from device memory, and P the right one from threadgroup memory (P as the left
// operand with per-lane V loads was 7% slower per layer). Changed for the 27B: a threadgroup serves QB query tokens of one kv head, so its
// 6 * QB query rows (token-major: row = token * 6 + head) share every K / V tile instead of
// one token's heads; dense causal mask; queries permuted to [kv head][token][6][256] half by
// attn_fl_q6h so a row block is one strided tensor.
// Not exact: half q, half P, f32 accumulation (the prefill gate of `saetta verify`).
//
// Shapes are compile-time constants (config_27b.h: 24 q heads, 4 kv heads, head_dim 256),
// like every model kernel; src/dense.c static_asserts them.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int FA_HD = 256, FA_G = 6, FA_NQ = 24, FA_NKV = 4;

// pos0: position of the chunk's first token; T: tokens in the chunk; tq: token stride of a kv
// head's plane in qh (a multiple of QB, at least T).
struct AttnFlP { uint pos0, T, tq; };

// q [T][24][256] f32 -> qh [4][tq][6][256] half, zero for tokens T .. tq-1. Four values per thread.
kernel void attn_fl_q6h(device const float4 *q [[buffer(0)]], device half4 *qh [[buffer(1)]], constant AttnFlP &a [[buffer(2)]],
                        uint e [[thread_position_in_grid]]) {
    constexpr uint D4 = FA_HD / 4;
    if (e >= (uint)FA_NKV * a.tq * FA_G * D4) return;
    const uint d = e % D4, h = (e / D4) % FA_G, t = (e / (D4 * FA_G)) % a.tq, g = e / (D4 * FA_G * a.tq);
    qh[e] = t < a.T ? half4(q[((ulong)t * FA_NQ + g * FA_G + h) * D4 + d]) : half4(0.0h);
}

// Threadgroup (kv head g, block of QB tokens), NSG simdgroups, key steps of BK keys.
// S phase: the NSG simdgroups are KG key groups x NSG / KG row groups; simdgroup sg takes the
// KS = BK / KG keys of group sg % KG against the MS rows of group sg / KG.
// O phase: simdgroup sg takes dims [sg * DS, sg * DS + DS) of all R rows, as the transposed tile
// O^T (DS x R): a lane holds dims r0 + 8 j and runs of four rows, so the rescale factors and the
// row sums of its rows are read from threadgroup memory.
// A destination lane holds rows r0 + 8 j (j < rows / 8) of its tile, r0 below, and runs of four
// consecutive columns: the M5 layout, probed for 48 x 16 and 48 x 32 tiles; lanes xor 1 and
// xor 8 hold the other columns of a row.
// Scores are kept in the log2 domain (the scale folds log2 e), so exp2 replaces exp; steps that
// every row of the block sees in full skip the causal mask and the key bound checks.
// P rows (one per query row) have a stride of PS >= BK halfs: padding them past BK spreads the
// rows of a P^T tile over the threadgroup memory banks (PS 264 against 256: 2.5% per layer).
template <int QB, int NSG, int BK, int KG, int PS>
inline void attn_fl(device const half *q, device const half *K, device const half *V, device const float *qg,
                    device float *out, constant AttnFlP &a, uint2 tg, ushort sg, ushort lane, threadgroup half *p,
                    threadgroup float *pm, threadgroup float *ps, threadgroup float *fs) {
    constexpr int R = FA_G * QB, RG = NSG / KG, MS = R / RG, KS = BK / KG, DS = FA_HD / NSG, J = R / 8, JS = MS / 8;
    static_assert(R % 16 == 0 && NSG % KG == 0 && MS % 16 == 0 && DS % 16 == 0 && KS % 16 == 0 && BK % 32 == 0 && PS >= BK &&
                  PS % 4 == 0);
    using T2 = dextents<int32_t, 2>;
    const uint g = tg.x;
    const uint nblk = (a.T + QB - 1) / QB;
    const uint t0 = (nblk - 1 - tg.y) * QB; // the longest rows first
    const uint nk = a.pos0 + min(a.T, t0 + QB); // keys any row of the block sees
    const int nch = (int)((nk + BK - 1) / BK);
    const int nfull = (int)((a.pos0 + t0 + 1) / BK); // steps every row of the block sees in full
    const int kg = sg % KG, row0 = (sg / KG) * MS, d0 = sg * DS;
    const ulong rs = FA_NKV * FA_HD;
    device const half *Kg = K + g * FA_HD, *Vg = V + g * FA_HD + d0;
    constexpr float SC = 0.0625f * M_LOG2E_F; // 1 / sqrt(256), in log2 units

    constexpr auto dS = matmul2d_descriptor(MS, KS, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    // O^T += V^T P^T: V^T (DS x keys) from the cache rows, transposed; P^T (keys x R) from P, transposed
    constexpr auto dO = matmul2d_descriptor(DS, R, 32, true, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    // the last 32 keys of the chunk, when fewer: K from the tensors, so V rows past nk (any value) are not read
    constexpr auto dOt = matmul2d_descriptor(DS, R, static_cast<int>(dynamic_extent), true, true, false,
                                             matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dS, execution_simdgroup> opS;
    matmul2d<dO, execution_simdgroup> opO;
    matmul2d<dOt, execution_simdgroup> opOt;
    tensor<device half, T2, tensor_inline> tQ((device half *)q + ((ulong)g * a.tq + t0) * FA_G * FA_HD, T2(FA_HD, R));
    tensor<threadgroup half, T2, tensor_inline> tP(p, T2(BK, R), array<int32_t, 2>{1, PS});
    tensor<device half, T2, tensor_inline> tV((device half *)Vg, T2(DS, (int)nk), array<int32_t, 2>{1, (int)rs});
    auto rk = opS.template get_right_input_cooperative_tensor<half, half, float>();
    auto tA0 = tQ.template slice<32, MS>(0, row0);
    auto tP0 = tP.template slice<32, R>(0, 0);
    auto tV0 = tV.template slice<DS, 32>(0, 0);
    auto tot = opO.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tV0)>,
                                                               __tensor_ops_detail::__remove_addrspace_t<decltype(tP0)>, float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < tot.get_capacity(); ++i) tot[i] = 0.0f;
    const uint r0 = ((lane >> 1) & 3) + 4 * (lane >> 4);
    const bool wr = (lane & 9) == 0;
    float m[J], l[J]; // l: this lane's part of the row sum of rows row0 + r0 + 8 j (j < JS) until the end
    int lim[J]; // last visible key of row r0 + 8 j (-1: past the chunk)
#pragma clang loop unroll(full)
    for (int j = 0; j < J; ++j) {
        m[j] = -INFINITY;
        l[j] = 0.0f;
        const uint tl = (r0 + 8 * j) / FA_G;
        lim[j] = t0 + tl < a.T ? (int)(a.pos0 + t0 + tl) : -1;
    }

    for (int ch = 0; ch < nch; ++ch) {
        const int c0 = ch * BK, kbase = c0 + kg * KS;
        const bool full = ch < nfull;
        auto s = opS.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA0)>,
                                                                 __tensor_ops_detail::__remove_addrspace_t<decltype(rk)>, float>();
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < s.get_capacity(); ++i) s[i] = 0.0f;
        for (int k0 = 0; k0 < FA_HD; k0 += 32) {
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < rk.get_capacity(); i += 4) {
                const auto ix = rk.get_multidimensional_index(i);
                const int id = kbase + ix[1];
                const half4 w = full || id < (int)nk ? *(device const half4 *)(Kg + (ulong)id * rs + k0 + ix[0]) : half4(0.0h);
                rk[i] = w.x; rk[i + 1] = w.y; rk[i + 2] = w.z; rk[i + 3] = w.w;
            }
            auto tA = tQ.template slice<32, MS>(k0, row0);
            opS.run(tA, rk, s);
        }
        // scale and causal mask; row max over this simdgroup's KS keys
        float x[JS];
#pragma clang loop unroll(full)
        for (int j = 0; j < JS; ++j) x[j] = -INFINITY;
        if (full) {
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < s.get_capacity(); ++i) {
                const int js = s.get_multidimensional_index(i)[1] / 8;
                const float v = s[i] * SC;
                s[i] = v;
                x[js] = max(x[js], v);
            }
        } else {
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < s.get_capacity(); ++i) {
                const auto ix = s.get_multidimensional_index(i);
                const int js = ix[1] / 8;
                const bool vis = s.is_valid_element(i) && kbase + ix[0] <= lim[row0 / 8 + js];
                const float v = vis ? s[i] * SC : -INFINITY;
                s[i] = v;
                x[js] = max(x[js], v);
            }
        }
#pragma clang loop unroll(full)
        for (int j = 0; j < JS; ++j) {
            float v = max(x[j], simd_shuffle_xor(x[j], 1));
            v = max(v, simd_shuffle_xor(v, 8));
            if (wr) pm[kg * R + row0 + 8 * j + r0] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // new maxima of all R rows (the O phase rescales every row)
        float f[J];
#pragma clang loop unroll(full)
        for (int j = 0; j < J; ++j) {
            float n = m[j];
#pragma clang loop unroll(full)
            for (int k = 0; k < KG; ++k) n = max(n, pm[k * R + 8 * j + r0]);
            f[j] = n == -INFINITY ? 1.0f : exp2(m[j] - n);
            m[j] = n;
            if (sg == 0 && wr) fs[8 * j + r0] = f[j]; // for the O^T rescale; published by the barrier after P
        }
        // P = exp2(s - m) as half, four consecutive keys per store
#pragma clang loop unroll(full)
        for (int j = 0; j < JS; ++j) x[j] = 0.0f;
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < s.get_capacity(); i += 4) {
            const auto ix = s.get_multidimensional_index(i);
            const int js = ix[1] / 8;
            const float mj = m[row0 / 8 + js];
            half4 e;
#pragma clang loop unroll(full)
            for (int u = 0; u < 4; ++u) {
                const float ev = s[i + u] == -INFINITY ? 0.0f : exp2(s[i + u] - mj);
                x[js] += ev;
                e[u] = half(ev);
            }
            if (s.is_valid_element(i)) *(threadgroup half4 *)(p + (row0 + ix[1]) * PS + kg * KS + ix[0]) = e;
        }
        // the row sums stay per lane (one exchange at the end); the barrier publishes P
#pragma clang loop unroll(full)
        for (int j = 0; j < JS; ++j) l[j] = l[j] * f[row0 / 8 + j] + x[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < tot.get_capacity(); ++i) tot[i] *= fs[tot.get_multidimensional_index(i)[0]];
        for (int cc = 0; cc < BK && c0 + cc < (int)nk; cc += 32) {
            if (full || c0 + cc + 32 <= (int)nk) {
                auto tVh = tV.template slice<DS, 32>(0, c0 + cc);
                auto tPh = tP.template slice<32, R>(cc, 0);
                opO.run(tVh, tPh, tot);
            } else {
                const int kl = (int)nk - (c0 + cc);
                tensor<device half, T2, tensor_inline> tVt((device half *)Vg + (ulong)(c0 + cc) * rs, T2(DS, kl),
                                                          array<int32_t, 2>{1, (int)rs});
                tensor<threadgroup half, T2, tensor_inline> tPt(p + cc, T2(kl, R), array<int32_t, 2>{1, PS});
                opOt.run(tVt, tPt, tot);
            }
        }
    }
    // row sums: lanes, then key groups
#pragma clang loop unroll(full)
    for (int j = 0; j < JS; ++j) {
        float v = l[j] + simd_shuffle_xor(l[j], 1);
        v += simd_shuffle_xor(v, 8);
        if (wr) ps[kg * R + row0 + 8 * j + r0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < tot.get_capacity(); ++i)
        if (tot.is_valid_element(i)) {
            const auto ix = tot.get_multidimensional_index(i);
            const uint row = ix[0], t = t0 + row / FA_G;
            if (t >= a.T) continue;
            const uint hh = g * FA_G + row % FA_G, d = d0 + ix[1];
            float z = 0.0f;
#pragma clang loop unroll(full)
            for (int k = 0; k < KG; ++k) z += ps[k * R + row];
            const float o = tot[i] / z;
            const float gate = qg[((ulong)t * FA_NQ + hh) * 2 * FA_HD + d];
            out[((ulong)t * FA_NQ + hh) * FA_HD + d] = o / (1.0f + exp(-gate));
        }
}

// qh: attn_fl_q6h's layout; K, V: the layer's f16 cache planes [pos][4][256]; qg: the gate
// half of the q projection [T][24][query 256 | gate 256] (f32, bound at the first gate); out
// [T][24][256] f32.
// Grid (4, ceil(T / QB)), 32 * NSG threads.
#define ATTN_FL(NAME, QB, NSG, BK, KG, PS)                                                                                  \
    kernel void NAME(device const half *q [[buffer(0)]], device const half *K [[buffer(1)]], device const half *V [[buffer(2)]], \
                     device const float *qg [[buffer(3)]], device float *out [[buffer(4)]], constant AttnFlP &a [[buffer(5)]],  \
                     uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],                    \
                     ushort lane [[thread_index_in_simdgroup]]) {                                                                \
        threadgroup half p[FA_G * QB * PS];                                                                                     \
        threadgroup float pm[KG * FA_G * QB], ps[KG * FA_G * QB], fs[FA_G * QB];                                                \
        attn_fl<QB, NSG, BK, KG, PS>(q, K, V, qg, out, a, tg, sg, lane, p, pm, ps, fs);                                         \
    }
// 8 tokens (48 rows), 8 simdgroups, 256-key steps of 32 keys per simdgroup: S tiles of 48 x 32
// (48 x 16 at 128-key steps: 16% slower); P takes 24.8 KB of the 32 KB of threadgroup memory
ATTN_FL(attn_fl_6_q8, 8, 8, 256, 8, 264)
