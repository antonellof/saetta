// Inkling 2 Mini prompt chunks on the M5 neural accelerators (Metal 4 tensor ops, matmul2d):
// half operands, f32 accumulation. Compiled on its own (src/inkling.c loads it only where tensor
// ops run; elsewhere and with SAETTA_NO_ACCEL=1 the f32 chunk kernels of inkling_moe.metal serve).
// Not exact: half activations (checked by `saetta verify`'s accelerated-prefill gate).
//
// Routed experts, grouped: ink_mpp_plan sorts the chunk's (token, slot) pairs by expert into rows
// padded to 32 per expert and cuts each expert's rows into tiles of at most 128; ink_mpp_gather
// writes the pairs' inputs as half rows; ink_mpp_gu_* (gate_up and the softcapped SwiGLU) and
// ink_mpp_dn_* (down, times the router weight) decode their expert's weights once per 64 x 64
// tile into threadgroup memory as half and run matmul2d on every 32-row sub-tile of the tile.
// The weights in half are exact: an NVFP4 value is e2m1 (2 significant bits) times its E4M3 scale
// (4), at most 2688 in magnitude and at least 2^-10; the per-expert factor global_scale / 2688 is
// applied to the f32 sums. MXFP8: E4M3 times 2^(s - 127), exact where that stays a normal half.
// After Strata (MIT, Copyright (c) 2026 Niko1221 and the Strata contributors; c6f6427,
// engine/src/inkling2/metal/ink_metal_mpp.metal ink_mpp_moe_plan / _gather / _gu_dir / _dn_dir):
// padded per-expert rows, a half gather, decode-once weight tiles shared by several row tiles,
// the activation fused into gate_up. Changed: one tile kind (up to 4 sub-tiles of 32 rows, the
// count per tile at run time) instead of four tile sizes, a stable plan (pairs in token order
// within an expert), the MLX pack's tensors at any byte alignment, the global scale on the sums,
// down written per pair for the existing ordered sum (ink_moe_sum).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr uint IM_TN = 64, IM_TK = 64, IM_TM = 32, IM_NSUB = 4, IM_PAD = 32;

inline float im_sigm(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float im_swiglu(float g, float u, float gcap, float ucap) {
    return tanh(g / gcap) * gcap * im_sigm(g) * (tanh(u / ucap) * ucap);
}

// ---- plan -----------------------------------------------------------------------------------
// n = T * K pairs (pair i: token i / K, slot i % K) with experts idx[i] < E (<= 128). Expert e's
// rows start at off[e] (the padded counts of the experts before it, each rounded up to 32): its
// pairs in token order, then pad rows. srt[row] = pair or -1 (pad), tiles[j] = (e, first row,
// end of the valid rows, end of the padded rows), at most 128 rows a tile; meta = (tiles, rows).
// One threadgroup of 1024 threads: simdgroup s takes pairs [s * blk, (s + 1) * blk).
struct ImPlanP { uint E, n; };
kernel void ink_mpp_plan(device const uint *idx [[buffer(0)]], device int *srt [[buffer(1)]], device uint4 *tiles [[buffer(2)]],
                         device uint *meta [[buffer(3)]], constant ImPlanP &p [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint cnt[32][128];
    threadgroup uint tot[128], eoff[129], toff[129];
    for (uint i = tid; i < 32 * 128; i += 1024) cnt[i / 128][i % 128] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint blk = (p.n + 31) / 32, b0 = sg * blk, b1 = min(p.n, b0 + blk);
    // per simdgroup: its pairs' counts per expert, 32 pairs at a time (the last lane holding an
    // expert adds that expert's count in the group)
    for (uint g0 = b0; g0 < b1; g0 += 32) {
        const uint i = g0 + lane;
        const uint ex = i < b1 ? idx[i] : 0xFFFFFFFFu;
        uint rank = 0;
        bool last = true;
        for (uint j = 0; j < 32; j++) {
            const uint o = simd_shuffle(ex, j);
            rank += j < lane && o == ex;
            last = last && !(j > lane && o == ex);
        }
        if (ex != 0xFFFFFFFFu && last) cnt[sg][ex] += rank + 1;
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < p.E) {
        uint c = 0;
        for (uint s = 0; s < 32; s++) c += cnt[s][tid];
        tot[tid] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        eoff[0] = 0;
        toff[0] = 0;
        for (uint e = 0; e < p.E; e++) {
            const uint pr = (tot[e] + IM_PAD - 1) / IM_PAD * IM_PAD;
            eoff[e + 1] = eoff[e] + pr;
            toff[e + 1] = toff[e] + (pr + IM_TM * IM_NSUB - 1) / (IM_TM * IM_NSUB);
        }
        meta[0] = toff[p.E];
        meta[1] = eoff[p.E];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < p.E) {
        // this expert's running row per simdgroup block, its pad rows and its tiles
        const uint e = tid;
        uint o = eoff[e];
        for (uint s = 0; s < 32; s++) {
            const uint c = cnt[s][e];
            cnt[s][e] = o;
            o += c;
        }
        for (uint r = o; r < eoff[e + 1]; r++) srt[r] = -1;
        const uint vend = o;
        for (uint j = 0, r0 = eoff[e]; r0 < eoff[e + 1]; j++, r0 += IM_TM * IM_NSUB)
            tiles[toff[e] + j] = uint4(e, r0, min(vend, r0 + IM_TM * IM_NSUB), min(eoff[e + 1], r0 + IM_TM * IM_NSUB));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g0 = b0; g0 < b1; g0 += 32) {
        const uint i = g0 + lane;
        const uint ex = i < b1 ? idx[i] : 0xFFFFFFFFu;
        uint rank = 0;
        bool last = true;
        for (uint j = 0; j < 32; j++) {
            const uint o = simd_shuffle(ex, j);
            rank += j < lane && o == ex;
            last = last && !(j > lane && o == ex);
        }
        if (ex != 0xFFFFFFFFu) srt[cnt[sg][ex] + rank] = int(i);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (ex != 0xFFFFFFFFu && last) cnt[sg][ex] += rank + 1;
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// xs[row] = half(x[token of srt[row]]) (zeros for pad rows), H columns. Grid rows bound x 256.
kernel void ink_mpp_gather(device const float *x [[buffer(0)]], device const int *srt [[buffer(1)]], device const uint *meta [[buffer(2)]],
                           device half *xs [[buffer(3)]], constant uint2 &p [[buffer(4)]] /* H, K */,
                           uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (row >= meta[1]) return;
    const int pr = srt[row];
    device half4 *dst = (device half4 *)(xs + (ulong)row * p.x);
    if (pr >= 0) {
        device const float4 *src = (device const float4 *)(x + (ulong)(uint(pr) / p.y) * p.x);
        for (uint i = tid; i < p.x / 4; i += 256) dst[i] = half4(src[i]);
    } else {
        for (uint i = tid; i < p.x / 4; i += 256) dst[i] = half4(0.0h);
    }
}

// ---- weight tiles ---------------------------------------------------------------------------
// 16 bytes at p, whose misalignment p & 3 is the same for every read of the tensor: aligned
// words, funnel-shifted (an aligned word holding a byte of the tensor lies in its mapped page).
inline uint4 im_ld16(device const uchar *p) {
    const uint sh = uint((ulong)p & 3);
    device const uint *q = (device const uint *)(p - sh);
    if (sh == 0) return uint4(q[0], q[1], q[2], q[3]);
    const uint s = 8 * sh, r = 32 - s;
    const uint w0 = q[0], w1 = q[1], w2 = q[2], w3 = q[3], w4 = q[4];
    return uint4((w0 >> s) | (w1 << r), (w1 >> s) | (w2 << r), (w2 >> s) | (w3 << r), (w3 >> s) | (w4 << r));
}
// Value tables in threadgroup memory (one 32-bit word per byte value), filled by im_lut: NVFP4,
// a byte's two E2M1 codes as half2 (element 0 from the low nibble); MXFP8, an E4M3 byte as f32.
// (Half arithmetic on this GPU flushes subnormal halves, so the codes are not decoded through
// subnormal bit patterns; conversions to and from f32 keep them.)
constant float IM_E2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
// E4M3 byte to f32, exact: the 7 magnitude bits placed as a half (exponent bias 15 vs 7), times 2^8.
inline float im_e4m3(uint b) {
    const float v = float(as_type<half>(ushort((b & 0x7Fu) << 7))) * 256.0f;
    return (b & 0x80u) ? -v : v;
}
template <bool MX>
inline void im_lut(threadgroup uint *lut, uint tid) {
    for (uint b = tid; b < 256; b += 128)
        lut[b] = MX ? as_type<uint>(im_e4m3(b)) : as_type<uint>(half2(half(IM_E2M1[b & 15u]), half(IM_E2M1[b >> 4])));
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Row `row` of an expert's matrix, columns k .. k + 31 (k % 32 == 0), into dst (8 half4):
// NVFP4 rows of K / 2 bytes with K / 16 E4M3 scales (no global factor; e2m1 times the scale is
// exact in half); MXFP8 rows of K bytes with K / 32 E8M0 scales (applied in f32, then rounded).
template <bool MX>
inline void im_dec32(device const uchar *W, device const uchar *S, uint K, uint row, uint k, threadgroup const uint *lut,
                     threadgroup half4 *dst) {
    if (MX) {
        device const uchar *wr = W + (ulong)row * K + k;
        const uint s = S[(ulong)row * (K / 32) + k / 32];
        const float f = as_type<float>(s << 23); // 2^(s - 127), as ink_moe_* decode it
        const uint4 a = im_ld16(wr), b = im_ld16(wr + 16);
        const uint w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            const uint x = w[i];
            const float4 v = float4(as_type<float>(lut[x & 0xFFu]), as_type<float>(lut[(x >> 8) & 0xFFu]), as_type<float>(lut[(x >> 16) & 0xFFu]),
                                    as_type<float>(lut[x >> 24]));
            dst[i] = half4(v * f);
        }
    } else {
        device const uchar *wr = W + (ulong)row * (K / 2) + k / 2;
        device const uchar *sr = S + (ulong)row * (K / 16) + k / 16;
        const half2 s0 = half2(half(im_e4m3(sr[0]))), s1 = half2(half(im_e4m3(sr[1])));
        const uint4 a = im_ld16(wr);
        const uint w[4] = {a.x, a.y, a.z, a.w};
#pragma clang loop unroll(full)
        for (uint i = 0; i < 4; i++) {
            const half2 sc = i < 2 ? s0 : s1;
            const uint x = w[i];
            const half2 v0 = as_type<half2>(lut[x & 0xFFu]), v1 = as_type<half2>(lut[(x >> 8) & 0xFFu]);
            const half2 v2 = as_type<half2>(lut[(x >> 16) & 0xFFu]), v3 = as_type<half2>(lut[x >> 24]);
            dst[2 * i] = half4(v0 * sc, v1 * sc);
            dst[2 * i + 1] = half4(v2 * sc, v3 * sc);
        }
    }
}

struct ImMoeP { uint H, I, K; uint sh_w, sh_s; ulong wsE, ssE; float gcap, ucap; };

// C[sub-tile s][32 rows][64] = A[rows][Kd] . W[n0 .. n0 + 63][Kd]^T for the tile's nsub sub-tiles,
// each stored to threadgroup ot (32 x 64 f32, aliasing the weight tile) and handed to out(s).
template <bool MX, typename Out>
inline void im_tile(device const uchar *W, device const uchar *S, uint Kd, uint n0, device half *A, uint row0, uint nsub, uint tid,
                    threadgroup half *Wt, threadgroup uint *lut, thread Out &out) {
    using T2 = dextents<int32_t, 2>;
    tensor<device half, T2, tensor_inline> tA(A, T2(Kd, 1 << 24));
    tensor<threadgroup half, T2, tensor_inline> tW(Wt, T2(IM_TK, IM_TN));
    constexpr auto desc = matmul2d_descriptor(IM_TM, IM_TN, IM_TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto a0 = tA.template slice<IM_TK, IM_TM>(0, row0);
    auto c0 = op.template get_destination_cooperative_tensor<decltype(a0), decltype(tW), float>();
    auto c1 = op.template get_destination_cooperative_tensor<decltype(a0), decltype(tW), float>();
    auto c2 = op.template get_destination_cooperative_tensor<decltype(a0), decltype(tW), float>();
    auto c3 = op.template get_destination_cooperative_tensor<decltype(a0), decltype(tW), float>();
#pragma unroll
    for (uint16_t i = 0; i < c0.get_capacity(); ++i) c0[i] = c1[i] = c2[i] = c3[i] = 0.0f;
    im_lut<MX>(lut, tid);
    const uint r = tid / 2, part = tid % 2;
    threadgroup half4 *dst = (threadgroup half4 *)(Wt + r * IM_TK + part * 32);
    for (uint k0 = 0; k0 < Kd; k0 += IM_TK) {
        im_dec32<MX>(W, S, Kd, n0 + r, k0 + part * 32, lut, dst);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto a0k = tA.template slice<IM_TK, IM_TM>(k0, row0);
        op.run(a0k, tW, c0);
        if (nsub > 1) {
            auto a1k = tA.template slice<IM_TK, IM_TM>(k0, row0 + IM_TM);
            op.run(a1k, tW, c1);
        }
        if (nsub > 2) {
            auto a2k = tA.template slice<IM_TK, IM_TM>(k0, row0 + 2 * IM_TM);
            op.run(a2k, tW, c2);
        }
        if (nsub > 3) {
            auto a3k = tA.template slice<IM_TK, IM_TM>(k0, row0 + 3 * IM_TM);
            op.run(a3k, tW, c3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float *ot = (threadgroup float *)Wt;
    tensor<threadgroup float, T2, tensor_inline> tO(ot, T2(IM_TN, IM_TM));
    for (uint s = 0; s < nsub; s++) {
        if (s == 0) c0.store(tO);
        else if (s == 1) c1.store(tO);
        else if (s == 2) c2.store(tO);
        else c3.store(tO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        out(s, ot);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// gate_up and the activation over a tile: grid (2I / 64, tiles bound) x 128. W rows alternate
// gate and up, so the 64 rows from n0 give the 32 activations from n0 / 2; hs [rows][I] half.
template <bool MX>
struct ImGuOut {
    device half *hs;
    uint row0, I, c0, tid;
    float f, gcap, ucap;
    void operator()(uint s, threadgroup float *ot) thread {
        for (uint q = tid; q < IM_TM * 32; q += 128) {
            const uint m = q / 32, j = q % 32;
            hs[(ulong)(row0 + s * IM_TM + m) * I + c0 + j] = half(im_swiglu(ot[m * IM_TN + 2 * j] * f, ot[m * IM_TN + 2 * j + 1] * f, gcap, ucap));
        }
    }
};
template <bool MX>
inline void im_gu(device const uchar *W, device const uchar *S, device const float *gfac, device half *xs, device const uint4 *tiles,
                  device const uint *meta, device half *hs, constant ImMoeP &p, uint2 tg, uint tid, threadgroup half *Wt, threadgroup uint *lut) {
    if (tg.y >= meta[0]) return;
    const uint4 tl = tiles[tg.y];
    const uint e = tl.x, n0 = tg.x * IM_TN, nsub = (tl.w - tl.y) / IM_TM;
    ImGuOut<MX> out = {hs, tl.y, p.I, n0 / 2, tid, MX ? 1.0f : gfac[e], p.gcap, p.ucap};
    im_tile<MX>(W + p.sh_w + e * p.wsE, S + p.sh_s + e * p.ssE, p.H, n0, xs, tl.y, nsub, tid, Wt, lut, out);
}

// down over a tile, weighted: grid (H / 64, tiles bound) x 128. ys [pairs][H] at the row's pair,
// times its router weight (pad rows skipped).
struct ImDnOut {
    device float *ys;
    device const int *srt;
    device const float *wts;
    uint row0, vend, H, n0, tid;
    float f;
    void operator()(uint s, threadgroup float *ot) thread {
        for (uint q = tid; q < IM_TM * IM_TN; q += 128) {
            const uint m = q / IM_TN, n = q % IM_TN, row = row0 + s * IM_TM + m;
            if (row >= vend) continue;
            const uint pr = uint(srt[row]);
            ys[(ulong)pr * H + n0 + n] = ot[m * IM_TN + n] * f * wts[pr];
        }
    }
};
template <bool MX>
inline void im_dn(device const uchar *W, device const uchar *S, device const float *gfac, device half *hs, device const uint4 *tiles,
                  device const uint *meta, device const int *srt, device const float *wts, device float *ys, constant ImMoeP &p, uint2 tg,
                  uint tid, threadgroup half *Wt, threadgroup uint *lut) {
    if (tg.y >= meta[0]) return;
    const uint4 tl = tiles[tg.y];
    const uint e = tl.x, n0 = tg.x * IM_TN, nsub = (tl.w - tl.y) / IM_TM;
    ImDnOut out = {ys, srt, wts, tl.y, tl.z, p.H, n0, tid, MX ? 1.0f : gfac[e]};
    im_tile<MX>(W + p.sh_w + e * p.wsE, S + p.sh_s + e * p.ssE, p.I, n0, hs, tl.y, nsub, tid, Wt, lut, out);
}

#define IM_GU(NAME, MX)                                                                                                         \
    kernel void NAME(device const uchar *W [[buffer(0)]], device const uchar *S [[buffer(1)]], device const float *gfac [[buffer(2)]], \
                     device half *xs [[buffer(3)]], device const uint4 *tiles [[buffer(4)]], device const uint *meta [[buffer(5)]],      \
                     device half *hs [[buffer(6)]], constant ImMoeP &p [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]],         \
                     uint tid [[thread_index_in_threadgroup]]) {                                                                       \
        threadgroup half Wt[IM_TN * IM_TK];                                                                                       \
        threadgroup uint lut[256];                                                                                                \
        im_gu<MX>(W, S, gfac, xs, tiles, meta, hs, p, tg, tid, Wt, lut);                                                               \
    }
IM_GU(ink_mpp_gu_nv, false)
IM_GU(ink_mpp_gu_mx, true)

#define IM_DN(NAME, MX)                                                                                                         \
    kernel void NAME(device const uchar *W [[buffer(0)]], device const uchar *S [[buffer(1)]], device const float *gfac [[buffer(2)]], \
                     device half *hs [[buffer(3)]], device const uint4 *tiles [[buffer(4)]], device const uint *meta [[buffer(5)]],      \
                     device const int *srt [[buffer(6)]], device const float *wts [[buffer(7)]], device float *ys [[buffer(8)]],         \
                     constant ImMoeP &p [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],                                     \
                     uint tid [[thread_index_in_threadgroup]]) {                                                                       \
        threadgroup half Wt[IM_TN * IM_TK];                                                                                       \
        threadgroup uint lut[256];                                                                                                \
        im_dn<MX>(W, S, gfac, hs, tiles, meta, srt, wts, ys, p, tg, tid, Wt, lut);                                                     \
    }
IM_DN(ink_mpp_dn_nv, false)
IM_DN(ink_mpp_dn_mx, true)
