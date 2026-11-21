// Qwen3.8-27B prompt-chunk attention on the M5 neural accelerators (Metal 4 tensor ops,
// matmul2d): causal GQA over the f16 KV cache, the cached prefix plus the chunk, gated by
// sigmoid(gate) at the write. Compiled alone (src/dense.c, src/attn27.c).
//
// Structure after qsa_flash_tr (src/kernels/qmm_mpp.metal, adapted from qwental
// kernels/qsa_pf.metal, MIT, Copyright (c) 2026 Ong Hoe Yuan; b1be701): scores per tile on one
// simdgroup, the row max exchanged through threadgroup memory once per key step (row sums kept
// per lane, exchanged once at the end), P stored as half, then each simdgroup does
// tot = tot * f + P V for its dims. Changed for the 27B: a threadgroup serves QB query tokens of
// one kv head, so its 6 * QB query rows (token-major: row = token * 6 + head) share every K / V
// tile instead of one token's heads; dense causal mask; queries permuted to
// [kv head][token][6][256] half by attn_fl_q6h so a row block is one strided tensor; both
// products transposed, S^T = K Q^T and O^T = V^T P^T, so K and V are left operands read by the
// matmul from device memory and the 48 rows are the N side of every tile (against S = Q K^T
// with per-lane K / V loads into right operands: 6% per layer for O^T, 2% more for S^T).
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
// S phase: simdgroup sg takes the KS = BK / KG keys of group sg against all R rows (S^T tile
// KS x R). O phase: simdgroup sg takes dims [sg * DS, sg * DS + DS) of all R rows (O^T tile DS x R).
// Both tiles are 32 x 48, with the M5 lane layout probed for them: element i of a lane is row
// cb + 16 ((i / 8) % 3) + i % 4 (cb = 4 (lane & 1) + 8 ((lane >> 3) & 1)) and column
// r0 + 8 ((i / 4) & 1) + 16 (i / 24) (r0 = ((lane >> 1) & 3) + 4 (lane >> 4)). A lane holds twelve
// rows, slot ((i / 8) % 3) * 4 + i % 4, the same in S^T and O^T, so the row max, rescale factor
// and row sum of a slot stay in registers; lanes xor 2, 4 and 16 hold the other keys of a row.
// P^T is stored [key][row] with a row stride of PT halfs (48: no padding needed, 56 no faster).
// Scores are kept in the log2 domain (the scale folds log2 e), so exp2 replaces exp; steps that
// every row of the block sees in full skip the causal mask and the key bound checks.
template <int QB, int NSG, int BK, int KG, int PT>
inline void attn_fl(device const half *q, device const half *K, device const half *V, device const float *qg,
                    device float *out, constant AttnFlP &a, uint2 tg, ushort sg, ushort lane, threadgroup half *p,
                    threadgroup float *pm, threadgroup float *ps) {
    constexpr int R = FA_G * QB, KS = BK / KG, DS = FA_HD / NSG, NR = 12;
    // the lane layout below is the probed one of 32 x 48 tiles: 48 rows, 32 keys or dims per simdgroup
    static_assert(R == 48 && NSG == KG && KS == 32 && DS == 32 && PT >= R && PT % 4 == 0);
    using T2 = dextents<int32_t, 2>;
    const uint g = tg.x;
    const uint nblk = (a.T + QB - 1) / QB;
    const uint t0 = (nblk - 1 - tg.y) * QB; // the longest rows first
    const uint nk = a.pos0 + min(a.T, t0 + QB); // keys any row of the block sees
    const int nch = (int)((nk + BK - 1) / BK);
    const int nfull = (int)((a.pos0 + t0 + 1) / BK); // steps every row of the block sees in full
    const int kg = sg, d0 = sg * DS;
    const ulong rs = FA_NKV * FA_HD;
    device const half *Kg = K + g * FA_HD, *Vg = V + g * FA_HD + d0;
    constexpr float SC = 0.0625f * M_LOG2E_F; // 1 / sqrt(256), in log2 units

    constexpr auto mac = matmul2d_descriptor::mode::multiply_accumulate;
    // S^T += K Q^T: K (keys x dims) from the cache rows; Q^T from the permuted queries, transposed
    constexpr auto dS = matmul2d_descriptor(KS, R, 32, false, true, false, mac);
    // O^T += V^T P^T: V^T (DS x keys) from the cache rows, transposed; P^T [key][row] from threadgroup memory
    constexpr auto dO = matmul2d_descriptor(DS, R, 32, true, false, false, mac);
    // the last 32 keys of the chunk, when fewer: K from the tensors, so V rows past nk (any value) are not read
    constexpr auto dOt = matmul2d_descriptor(DS, R, static_cast<int>(dynamic_extent), true, false, false, mac);
    matmul2d<dS, execution_simdgroup> opS;
    matmul2d<dO, execution_simdgroup> opO;
    matmul2d<dOt, execution_simdgroup> opOt;
    tensor<device half, T2, tensor_inline> tQ((device half *)q + ((ulong)g * a.tq + t0) * FA_G * FA_HD, T2(FA_HD, R));
    tensor<device half, T2, tensor_inline> tK((device half *)Kg, T2(FA_HD, (int)nk), array<int32_t, 2>{1, (int)rs});
    tensor<threadgroup half, T2, tensor_inline> tP(p, T2(R, BK), array<int32_t, 2>{1, PT});
    tensor<device half, T2, tensor_inline> tV((device half *)Vg, T2(DS, (int)nk), array<int32_t, 2>{1, (int)rs});
    auto tK0 = tK.template slice<32, KS>(0, 0);
    auto tQ0 = tQ.template slice<32, R>(0, 0);
    auto tV0 = tV.template slice<DS, 32>(0, 0);
    auto tP0 = tP.template slice<R, 32>(0, 0);
    auto tot = opO.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tV0)>,
                                                               __tensor_ops_detail::__remove_addrspace_t<decltype(tP0)>, float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < tot.get_capacity(); ++i) tot[i] = 0.0f;
    const uint cb = 4 * (lane & 1) + 8 * ((lane >> 3) & 1);
    const bool wr = (lane & 0x16) == 0; // one lane of the eight that hold rows cb + 16 s + u
    float m[NR], l[NR]; // per slot; l: this lane's part of the row sum (its keys) until the end
#pragma clang loop unroll(full)
    for (int j = 0; j < NR; ++j) {
        m[j] = -INFINITY;
        l[j] = 0.0f;
    }

    for (int ch = 0; ch < nch; ++ch) {
        const int c0 = ch * BK, kbase = c0 + kg * KS;
        const bool full = ch < nfull;
        const int kv = full ? KS : clamp((int)nk - kbase, 0, KS); // keys of this group below nk
        auto s = opS.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tK0)>,
                                                                 __tensor_ops_detail::__remove_addrspace_t<decltype(tQ0)>, float>();
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < s.get_capacity(); ++i) s[i] = 0.0f;
        if (kv == KS) {
            for (int k0 = 0; k0 < FA_HD; k0 += 32) {
                auto tA = tK.template slice<32, KS>(k0, kbase);
                auto tB = tQ.template slice<32, R>(k0, 0);
                opS.run(tA, tB, s);
            }
        } else if (kv > 0) {
            // the key group straddles nk: its keys past nk are masked below, and the M extent keeps
            // the reads below nk
            tensor<device half, T2, tensor_inline> tKt((device half *)Kg + (ulong)kbase * rs, T2(FA_HD, kv),
                                                      array<int32_t, 2>{1, (int)rs});
            for (int k0 = 0; k0 < FA_HD; k0 += 32) {
                auto tA = tKt.template slice<32, dynamic_extent>(k0, 0);
                auto tB = tQ.template slice<32, R>(k0, 0);
                opS.run(tA, tB, s);
            }
        }
        // scale and causal mask; row max over this simdgroup's KS keys
        float x[NR];
#pragma clang loop unroll(full)
        for (int j = 0; j < NR; ++j) x[j] = -INFINITY;
        if (full) {
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < s.get_capacity(); ++i) {
                const int sl = ((i / 8) % 3) * 4 + i % 4;
                const float v = s[i] * SC;
                s[i] = v;
                x[sl] = max(x[sl], v);
            }
        } else {
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < s.get_capacity(); ++i) {
                const int sl = ((i / 8) % 3) * 4 + i % 4;
                const auto ix = s.get_multidimensional_index(i);
                const uint tl = (uint)ix[0] / FA_G;
                const int lim = t0 + tl < a.T ? (int)(a.pos0 + t0 + tl) : -1;
                const bool vis = s.is_valid_element(i) && kbase + ix[1] <= lim;
                const float v = vis ? s[i] * SC : -INFINITY;
                s[i] = v;
                x[sl] = max(x[sl], v);
            }
        }
#pragma clang loop unroll(full)
        for (int j = 0; j < NR; ++j) {
            float v = max(x[j], simd_shuffle_xor(x[j], 2));
            v = max(v, simd_shuffle_xor(v, 4));
            v = max(v, simd_shuffle_xor(v, 16));
            x[j] = v;
        }
        if (wr) {
#pragma clang loop unroll(full)
            for (int sb = 0; sb < 3; ++sb)
                *(threadgroup float4 *)(pm + kg * R + cb + 16 * sb) = float4(x[sb * 4], x[sb * 4 + 1], x[sb * 4 + 2], x[sb * 4 + 3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // new maxima of the lane's rows over the key groups (the O phase rescales every row)
        float f[NR];
#pragma clang loop unroll(full)
        for (int sb = 0; sb < 3; ++sb) {
            float4 n = float4(m[sb * 4], m[sb * 4 + 1], m[sb * 4 + 2], m[sb * 4 + 3]);
#pragma clang loop unroll(full)
            for (int k = 0; k < KG; ++k) n = max(n, *(threadgroup const float4 *)(pm + k * R + cb + 16 * sb));
#pragma clang loop unroll(full)
            for (int u = 0; u < 4; ++u) {
                const int j = sb * 4 + u;
                f[j] = n[u] == -INFINITY ? 1.0f : exp2(m[j] - n[u]);
                m[j] = n[u];
            }
        }
        // P^T = exp2(s - m) as half, four consecutive rows of one key per store
#pragma clang loop unroll(full)
        for (int j = 0; j < NR; ++j) x[j] = 0.0f;
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < s.get_capacity(); i += 4) {
            const auto ix = s.get_multidimensional_index(i);
            const int sl = ((i / 8) % 3) * 4;
            half4 e;
#pragma clang loop unroll(full)
            for (int u = 0; u < 4; ++u) {
                const float ev = s[i + u] == -INFINITY ? 0.0f : exp2(s[i + u] - m[sl + u]);
                x[sl + u] += ev;
                e[u] = half(ev);
            }
            if (s.is_valid_element(i)) *(threadgroup half4 *)(p + (kg * KS + ix[1]) * PT + ix[0]) = e;
        }
#pragma clang loop unroll(full)
        for (int j = 0; j < NR; ++j) l[j] = l[j] * f[j] + x[j];
        // the barrier publishes P^T; O^T has the lane rows of S^T, so f stays in registers
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < tot.get_capacity(); ++i) tot[i] *= f[((i / 8) % 3) * 4 + i % 4];
        for (int cc = 0; cc < BK && c0 + cc < (int)nk; cc += 32) {
            if (full || c0 + cc + 32 <= (int)nk) {
                auto tVh = tV.template slice<DS, 32>(0, c0 + cc);
                auto tPh = tP.template slice<R, 32>(0, cc);
                opO.run(tVh, tPh, tot);
            } else {
                const int kl = (int)nk - (c0 + cc);
                tensor<device half, T2, tensor_inline> tVt((device half *)Vg + (ulong)(c0 + cc) * rs, T2(DS, kl),
                                                          array<int32_t, 2>{1, (int)rs});
                tensor<threadgroup half, T2, tensor_inline> tPt(p + cc * PT, T2(R, kl), array<int32_t, 2>{1, PT});
                opOt.run(tVt, tPt, tot);
            }
        }
    }
    // row sums: lanes, then key groups
#pragma clang loop unroll(full)
    for (int j = 0; j < NR; ++j) {
        float v = l[j] + simd_shuffle_xor(l[j], 2);
        v += simd_shuffle_xor(v, 4);
        v += simd_shuffle_xor(v, 16);
        l[j] = v;
    }
    if (wr) {
#pragma clang loop unroll(full)
        for (int sb = 0; sb < 3; ++sb)
            *(threadgroup float4 *)(ps + kg * R + cb + 16 * sb) = float4(l[sb * 4], l[sb * 4 + 1], l[sb * 4 + 2], l[sb * 4 + 3]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (int sb = 0; sb < 3; ++sb) {
        float4 z = 0.0f;
#pragma clang loop unroll(full)
        for (int k = 0; k < KG; ++k) z += *(threadgroup const float4 *)(ps + k * R + cb + 16 * sb);
#pragma clang loop unroll(full)
        for (int u = 0; u < 4; ++u) l[sb * 4 + u] = z[u];
    }
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < tot.get_capacity(); ++i)
        if (tot.is_valid_element(i)) {
            const auto ix = tot.get_multidimensional_index(i);
            const uint row = ix[0], t = t0 + row / FA_G;
            if (t >= a.T) continue;
            const uint hh = g * FA_G + row % FA_G, d = d0 + ix[1];
            const float o = tot[i] / l[((i / 8) % 3) * 4 + i % 4];
            const float gate = qg[((ulong)t * FA_NQ + hh) * 2 * FA_HD + d];
            out[((ulong)t * FA_NQ + hh) * FA_HD + d] = o / (1.0f + exp(-gate));
        }
}
// qh: attn_fl_q6h's layout; K, V: the layer's f16 cache planes [pos][4][256]; qg: the gate
// half of the q projection [T][24][query 256 | gate 256] (f32, bound at the first gate); out
// [T][24][256] f32.
// Grid (4, ceil(T / QB)), 32 * NSG threads.
#define ATTN_FL(NAME, QB, NSG, BK, KG, PT)                                                                                  \
    kernel void NAME(device const half *q [[buffer(0)]], device const half *K [[buffer(1)]], device const half *V [[buffer(2)]], \
                     device const float *qg [[buffer(3)]], device float *out [[buffer(4)]], constant AttnFlP &a [[buffer(5)]],  \
                     uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],                    \
                     ushort lane [[thread_index_in_simdgroup]]) {                                                                \
        threadgroup half p[BK * PT];                                                                                            \
        threadgroup float pm[KG * FA_G * QB], ps[KG * FA_G * QB];                                                               \
        attn_fl<QB, NSG, BK, KG, PT>(q, K, V, qg, out, a, tg, sg, lane, p, pm, ps);                                             \
    }
// 8 tokens (48 rows), 8 simdgroups, 256-key steps of 32 keys per simdgroup (128-key steps: 9%
// slower); P^T takes 24 KB of the 32 KB of threadgroup memory
ATTN_FL(attn_fl_6_q8, 8, 8, 256, 8, 48)
