// Prompt-chunk attention without tensor ops (M1 to M4 GPUs, and SAETTA_NO_ACCEL=1): causal GQA
// over the f16 KV cache with simdgroup_matrix 8x8 products, f32 fragments and an online softmax
// in the log2 domain. Head dim 256. Compiled alone (src/attn_sg.c), without the tensor-op header
// (gpu_pipeline refuses any source that names it when the accelerators are off).
//
// Adapted from sushi's sushi_attn_pd (src/attn_kernels.zig, ATTN_PD_KERNEL_HEADER and
// ATTN_PD_KERNEL_SOURCE; MIT, Copyright (c) 2026 Theinruj Toranavikrai, Copyright (c) 2026 David
// Dalcu; d5e876e), whose fragment layout follows MLX steel BaseMMAFrag (steel/attn/mma.h; MIT,
// Copyright (c) 2023 Apple Inc.; 8f578df): each simdgroup owns 8 rows with Q and O as f32 fragments in registers
// (Q straight from device memory, no threadgroup copy), K staged transposed and V row-major in
// one shared threadgroup buffer per 32-key step, exp2 softmax with the scale folded in, 4
// barriers per step. Changed for saetta: a row is a (query token, query head) pair of one kv
// head, token-major, so every staged K / V tile serves 8 * NSG rows of several tokens and heads
// (GQA sharing) instead of one head's tokens; the causal bound per row from the chunk position;
// the scale folded into Q instead of S; split-K over the visible keys with (max, sum, sum p v)
// partials in attn_merge's layout (src/kernels/decode.metal) for chunks too small to fill the
// GPU; Flash-Next's QSA lists (one token per threadgroup, its 12 heads as 16 rows, gathered key
// rows); f32 queries, gate and output as the engine stores them, the sigmoid gate fused.
// Not exact against f64: f32 products of f16 K / V (no half rounding of q or p).
//
// Shapes are template constants (24 query heads; 27B 4 kv heads, Flash-Next 2), like every model
// kernel: src/dense.c and src/model.c static_assert them.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr int AS_HD = 256;             // head dim
constant constexpr int AS_BK = 32;              // keys per step
constant constexpr int AS_LDK = AS_BK + 8;      // K^T tile row stride (halfs)
constant constexpr int AS_LDV = AS_HD + 8;      // V tile row stride (halfs)
constant constexpr int AS_NW = AS_HD / 8;       // 8-wide fragments over the head dim
constant constexpr int AS_SMEM = AS_LDK * AS_HD > AS_BK * AS_LDV ? AS_LDK * AS_HD : AS_BK * AS_LDV;

// pos0: position of chunk token 0; T: tokens in the chunk; nsplit: key splits (partials when
// the kernel is a _part one); budget, ratio, maxsel: QSA rule (dense while n / ratio <= budget,
// else cnt[t] entries of sel + t * maxsel), unused by the causal kernels.
struct AttnSgP { uint pos0, T, nsplit, budget, ratio, maxsel; };

// Lane (sn, sm): the lane holds row sm, columns sn and sn + 1 of every 8x8 fragment; the four
// lanes of a row differ in lane bits 0 and 3.
inline short2 as_coord(ushort lane) {
    const short qid = lane / 4;
    return short2((qid & 2) * 2 + (lane % 2) * 2, (qid & 4) + (lane / 2) % 4);
}
inline void as_mma(thread float2 &d, float2 a, float2 b) {
    simdgroup_float8x8 D, A, B, C;
    A.thread_elements()[0] = a.x;
    A.thread_elements()[1] = a.y;
    B.thread_elements()[0] = b.x;
    B.thread_elements()[1] = b.y;
    C.thread_elements()[0] = d.x;
    C.thread_elements()[1] = d.y;
    simdgroup_multiply_accumulate(D, A, B, C);
    d.x = D.thread_elements()[0];
    d.y = D.thread_elements()[1];
}
inline float as_rmax(float v) { v = max(v, simd_shuffle_xor(v, ushort(1))); return max(v, simd_shuffle_xor(v, ushort(8))); }
inline float as_rsum(float v) { v += simd_shuffle_xor(v, ushort(1)); return v + simd_shuffle_xor(v, ushort(8)); }

// Threadgroup (kv head tg.x, row block tg.y, split tg.z), NSG simdgroups.
// SEL false (causal): the rows of kv head g are (token t, head h) for t < T, h < QH, numbered
// t * QH + h; block b holds rows [b * R, b * R + R) with R = 8 NSG, the longest blocks first.
// Row (t, h) sees positions 0 .. pos0 + t; the block's keys are split in nsplit ranges of whole
// steps. SEL true: block tg.y is token t alone (rows = its QH heads, R = 16 >= QH), keys its list.
// PART: write partials part[((t * NQ + head) * nsplit + split) * (HD + 2)] = (max, sum, o[HD])
// (natural-log max, -inf when the split saw no key); else out[(t * NQ + head) * HD + d] =
// o / sum * sigmoid(gate[(t * NQ + head) * 2 HD + d]).
template <int NQ, int NKV, int NSG, bool SEL, bool PART>
inline void attn_sg(device const float *q, device const half *K, device const half *V, device const float *gate,
                    device float *out, constant AttnSgP &a, device const uint *sel, device const uint *cnt, uint3 tg,
                    ushort sg, ushort lane, ushort tix, threadgroup half *kv, threadgroup int *ids) {
    constexpr int QH = NQ / NKV, R = 8 * NSG, NT = 32 * NSG;
    constexpr ulong RS = (ulong)NKV * AS_HD; // cache row stride
    static_assert(NT % AS_BK == 0 && (!SEL || (R >= QH && !PART)),"a QSA block is one token's heads, written directly");
    const uint g = tg.x;
    const short2 sc = as_coord(lane);
    const short sn = sc.x, sm = sc.y;
    // the lane's row, its token and head; the block's keys
    uint t, h, nk;
    int t_first; // earliest token of the block (causal: steps below its position need no mask)
    bool list = false;
    device const uint *st = sel;
    if (SEL) {
        t = tg.y;
        h = sg * 8 + sm;
        const uint n = a.pos0 + t + 1;
        list = n / a.ratio > a.budget;
        nk = list ? cnt[t] : n;
        st = sel + (ulong)t * a.maxsel;
        t_first = 0;
    } else {
        const uint nblk = (a.T * QH + R - 1) / R;
        const uint r0 = (nblk - 1 - tg.y) * R;
        const uint r = r0 + sg * 8 + sm;
        t = r / QH;
        h = r % QH;
        t_first = (int)(r0 / QH);
        nk = a.pos0 + min(a.T, (r0 + R - 1) / QH + 1);
    }
    const bool live = SEL ? h < (uint)QH : t < a.T;
    const uint hh = g * QH + h;
    const int row_pos = (int)(a.pos0 + t);
    // this split's keys [k0, k1), whole steps
    const uint nsplit = PART ? a.nsplit : 1;
    const uint per = ((nk + nsplit - 1) / nsplit + AS_BK - 1) / AS_BK * AS_BK;
    const int k0 = (int)min(nk, tg.z * per), k1 = (int)min(nk, (uint)k0 + per);
    const int full_end = SEL ? k1 : min(k1, (int)a.pos0 + t_first + 1); // keys every row sees

    // Q fragments, scaled by 1 / sqrt(256) in log2 units
    constexpr float SC = 0.0625f * M_LOG2E_F;
    float2 qf[AS_NW];
    {
        device const float *qr = q + ((ulong)min(t, a.T - 1) * NQ + g * QH + min(h, (uint)QH - 1)) * AS_HD + sn;
        for (int dd = 0; dd < AS_NW; ++dd) qf[dd] = live ? *(device const float2 *)(qr + dd * 8) * SC : float2(0.0f);
    }
    float2 of[AS_NW];
    for (int dd = 0; dd < AS_NW; ++dd) of[dd] = float2(0.0f);
    float m = -3.0e38f, l = 0.0f; // finite start: exp2(m_old - m_new) is never exp2(nan)

    threadgroup half *Ks = kv, *Vs = kv;
    device const half *Kg = K + g * AS_HD, *Vg = V + g * AS_HD;
    for (int c0 = k0; c0 < k1; c0 += AS_BK) {
        const int nc = min(AS_BK, k1 - c0);
        // K^T tile Ks[d][key]: consecutive threads take consecutive keys (one Ks row per store
        // group); the key ids of the step go to ids for the V stage
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            const int r = tix % AS_BK; // NT is a multiple of AS_BK: the thread's key is fixed
            const int id = r < nc ? (list ? (int)st[c0 + r] : c0 + r) : -1;
            if (tix < AS_BK) ids[r] = id;
            for (int c8 = tix / AS_BK; c8 < AS_NW; c8 += NT / AS_BK) {
                uint4 w = uint4(0);
                if (id >= 0) w = *((device const uint4 *)(Kg + (ulong)id * RS) + c8);
                thread const half *e = (thread const half *)&w;
                const int cb = c8 * 8;
                for (int u = 0; u < 8; ++u) Ks[(cb + u) * AS_LDK + r] = e[u];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for the simdgroup's 8 rows and the step's 32 keys
        float2 s[AS_BK / 8];
        for (int k = 0; k < AS_BK / 8; ++k) s[k] = float2(0.0f);
        for (int dd = 0; dd < AS_NW; ++dd) {
            const int kb = (dd * 8 + sm) * AS_LDK + sn;
            for (int k = 0; k < AS_BK / 8; ++k)
                as_mma(s[k], qf[dd], float2(float(Ks[kb + k * 8]), float(Ks[kb + k * 8 + 1])));
        }
        // mask: keys past the split, and (causal) past the row's position
        if (c0 + AS_BK > full_end) {
            for (int k = 0; k < AS_BK / 8; ++k)
                for (int j = 0; j < 2; ++j) {
                    const int key = c0 + k * 8 + sn + j;
                    if (key >= k1 || (!SEL && key > row_pos)) s[k][j] = -INFINITY;
                }
        }
        // V tile Vs[key][d] in the same memory once every simdgroup is done with Ks
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tix; i < AS_BK * AS_NW; i += NT) {
            const int r = i / AS_NW, c8 = i % AS_NW;
            const int id = ids[r];
            uint4 w = uint4(0);
            if (id >= 0) w = *((device const uint4 *)(Vg + (ulong)id * RS) + c8);
            *((threadgroup uint4 *)(Vs + r * AS_LDV) + c8) = w;
        }
        // online softmax in registers (overlaps the V stage of the other simdgroups)
        float mx = m;
        for (int k = 0; k < AS_BK / 8; ++k) mx = max(mx, max(s[k].x, s[k].y));
        mx = as_rmax(mx);
        const float f = exp2(m - mx);
        float ps = 0.0f;
        for (int k = 0; k < AS_BK / 8; ++k) {
            s[k] = exp2(s[k] - mx); // masked: exp2(-inf) = 0
            ps += s[k].x + s[k].y;
        }
        l = l * f + as_rsum(ps);
        m = mx;
        for (int dd = 0; dd < AS_NW; ++dd) of[dd] *= f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O += P V: P is S in the left-operand layout
        for (int dd = 0; dd < AS_NW; ++dd) {
            const int vb = sm * AS_LDV + dd * 8 + sn;
            for (int k = 0; k < AS_BK / 8; ++k)
                as_mma(of[dd], s[k], float2(float(Vs[vb + k * 8 * AS_LDV]), float(Vs[vb + k * 8 * AS_LDV + 1])));
        }
    }
    if (!live) return;
    if (PART) {
        device float *pt = out + (((ulong)t * NQ + hh) * nsplit + tg.z) * (AS_HD + 2);
        if ((lane & 9) == 0) { // one lane of the row's four
            pt[0] = l > 0.0f ? m * M_LN2_F : -INFINITY;
            pt[1] = l;
        }
        for (int dd = 0; dd < AS_NW; ++dd) *(device float2 *)(pt + 2 + dd * 8 + sn) = of[dd];
    } else {
        const float il = 1.0f / l;
        device const float *gr = gate + ((ulong)t * NQ + hh) * 2 * AS_HD + sn;
        device float *orow = out + ((ulong)t * NQ + hh) * AS_HD + sn;
        for (int dd = 0; dd < AS_NW; ++dd) {
            const float2 gv = *(device const float2 *)(gr + dd * 8);
            *(device float2 *)(orow + dd * 8) = of[dd] * il / (1.0f + exp(-gv));
        }
    }
}

// q [T][24][256] f32 (normed, roped); K, V the layer's f16 cache planes [pos][NKV][256]; gate
// the gate half of the q projection, [T][24][query 256 | gate 256] bound at the first gate;
// out [T][24][256] f32, or the partials. Causal grid (NKV, ceil(T * QH / (8 NSG)), nsplit);
// QSA grid (NKV, T, 1). 32 * NSG threads: the caller checks the pipeline allows that many.
#define ATTN_SG(NAME, NQ, NKV, NSG, SEL, PART)                                                                                 \
    kernel void NAME(device const float *q [[buffer(0)]], device const half *K [[buffer(1)]], device const half *V [[buffer(2)]],  \
                     device const float *gate [[buffer(3)]], device float *out [[buffer(4)]], constant AttnSgP &a [[buffer(5)]], \
                     device const uint *sel [[buffer(6)]], device const uint *cnt [[buffer(7)]],                                  \
                     uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],                     \
                     ushort lane [[thread_index_in_simdgroup]], ushort tix [[thread_index_in_threadgroup]]) {                     \
        threadgroup half kv[AS_SMEM];                                                                                            \
        threadgroup int ids[AS_BK];                                                                                              \
        attn_sg<NQ, NKV, NSG, SEL, PART>(q, K, V, gate, out, a, sel, cnt, tg, sg, lane, tix, kv, ids);                           \
    }
// Qwen3.8-27B: 24 query heads over 4 kv heads; 64 rows (8 simdgroups), or 32 when the pipeline
// allows fewer than 256 threads
ATTN_SG(attn_sg_27, 24, 4, 8, false, false)
ATTN_SG(attn_sg_27_part, 24, 4, 8, false, true)
ATTN_SG(attn_sg_27_s4, 24, 4, 4, false, false)
ATTN_SG(attn_sg_27_s4_part, 24, 4, 4, false, true)
// Qwen3.8-Flash-Next: 24 query heads over 2 kv heads, QSA lists; one token's 12 heads as 16 rows
ATTN_SG(attn_sg_fn, 24, 2, 2, true, false)
