// DeepSeek-V4-Flash hyper-connections (mHC), RMSNorm and MoE router, per token row (T rows
// for prefill). Values that the reference keeps in BF16 (the 4 residual streams, block
// inputs, norm outputs) are carried here as f32 holding BF16 values, rounded to nearest even.
//
// Ideas adapted from ds4 (MIT, Copyright (c) the ds4.c authors, the ggml authors and
// DeepSeek; github.com/antirez/ds4 at b3f50180ec2c7324892be1f50bc83555ccb8d5e7,
// metal/dsv4_hc.metal): the mixes matvec computing the stream RMS in the same pass, and
// splitting the mixer into a matvec dispatch plus a per-token split/weighted-sum dispatch.
//
// The library is compiled with the device's default math mode, so every operation whose
// rounding is part of the model's numerics uses the precise:: form (exp, log, rsqrt, sqrt,
// divide) instead of the fast approximations.
#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)

// Shapes (compile-time; the model has no other).
constant constexpr uint DS_D = 4096;            // hidden size
constant constexpr uint DS_HC = 4;              // streams
constant constexpr uint DS_HCD = DS_HC * DS_D;  // 16384, flattened streams
constant constexpr uint DS_MIX = (2 + DS_HC) * DS_HC; // 24 mixes per layer mixer
constant constexpr uint DS_SINKHORN = 20;       // hc_sinkhorn_iters
constant constexpr float DS_HC_EPS = 1e-6f;     // hc_eps
constant constexpr float DS_NORM_EPS = 1e-6f;   // norm_eps
constant constexpr uint DS_NE = 256;            // routed experts
constant constexpr uint DS_TOPK = 6;            // activated experts
constant constexpr float DS_ROUTE_SCALE = 1.5f; // route_scale
// The mixes matvec splits each 16384-wide row into this many slices, one threadgroup each;
// the split kernel adds the slices in order.
constant constexpr uint DS_SLICES = 8;
constant constexpr uint DS_SLICE = DS_HCD / DS_SLICES; // 2048
constant constexpr uint DS_MIX_THREADS = 128;
constant constexpr uint DS_ROW_THREADS = 256;   // split/head/norm/select threadgroups

inline float bf16_round(float v) {
    uint u = as_type<uint>(v);
    if ((u & 0x7fffffffu) > 0x7f800000u) return v; // NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return as_type<float>(u & 0xffff0000u);
}
inline float4 bf16_round4(float4 v) {
    return float4(bf16_round(v.x), bf16_round(v.y), bf16_round(v.z), bf16_round(v.w));
}
inline float bf16_val(ushort u) { return as_type<float>(uint(u) << 16); }
inline float4 bf16_val4(ushort4 u) { return float4(bf16_val(u.x), bf16_val(u.y), bf16_val(u.z), bf16_val(u.w)); }

// 1 / (1 + exp(-z)), as TileLang's T.sigmoid. Below -87 exp(-z) would overflow to inf and the
// result is 0 (or a denormal); the clamp keeps inf out of the fast-math library.
inline float ds_sigmoid(float z) {
    return precise::divide(1.0f, 1.0f + precise::exp(-max(z, -87.0f)));
}

// Sum over a threadgroup of nt threads (a multiple of 32, at most 1024); every thread gets it.
inline float ds_tg_sum(float v, threadgroup float *sh, uint tid, uint nt) {
    v = simd_sum(v);
    if (tid % 32 == 0) sh[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = (tid % 32 < nt / 32) ? sh[tid % 32] : 0.0f;
    r = simd_sum(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

struct DsMixP {
    uint nmix; // rows of fn: 24 (layer mixer) or 4 (head)
    uint T;
};

// Partial mixes: part[t][s][r] = fn[r][slice s] . x[t][slice s] for r < nmix, and
// part[t][s][nmix] = sum of x^2 over slice s. Grid (nmix, DS_SLICES, T), DS_MIX_THREADS threads.
// fn [nmix][16384] f32, x [T][16384] (the 4 streams of a token, BF16 values).
kernel void dsv4_hc_mix(device const float4 *fn [[buffer(0)]], device const float4 *x [[buffer(1)]],
                        device float *part [[buffer(2)]], constant DsMixP &p [[buffer(3)]],
                        uint3 g [[threadgroup_position_in_grid]], uint3 tid3 [[thread_position_in_threadgroup]]) {
    threadgroup float sh[32];
    const uint tid = tid3.x;
    const uint r = g.x, s = g.y, t = g.z;
    device const float4 *w4 = fn + (size_t)r * (DS_HCD / 4) + s * (DS_SLICE / 4);
    device const float4 *x4 = x + (size_t)t * (DS_HCD / 4) + s * (DS_SLICE / 4);
    float acc = 0, ss = 0;
    for (uint i = tid; i < DS_SLICE / 4; i += DS_MIX_THREADS) {
        float4 xv = x4[i], wv = w4[i];
        acc += dot(wv, xv);
        if (r == 0) ss += dot(xv, xv);
    }
    acc = ds_tg_sum(acc, sh, tid, DS_MIX_THREADS);
    device float *o = part + ((size_t)t * DS_SLICES + s) * (p.nmix + 1);
    if (r == 0) {
        ss = ds_tg_sum(ss, sh, tid, DS_MIX_THREADS);
        if (tid == 0) o[p.nmix] = ss;
    }
    if (tid == 0) o[r] = acc;
}

// Final mixes of token t from the slices: m[r] * rsqrt(mean(x^2) + eps).
inline float ds_mix_final(device const float *pt, uint nmix, uint r) {
    float ss = 0, m = 0;
    for (uint s = 0; s < DS_SLICES; s++) {
        ss += pt[s * (nmix + 1) + nmix];
        m += pt[s * (nmix + 1) + r];
    }
    float rs = precise::rsqrt(ss * (1.0f / float(DS_HCD)) + DS_NORM_EPS);
    return m * rs;
}

// Sum over the 4 lanes of a row (lane = 4 j + k, row j) or of a column (column k), pairwise:
// (a0 + a1) + (a2 + a3), identical in every lane of the group.
inline float ds_row_sum(float v) {
    v = v + simd_shuffle_xor(v, 1);
    return v + simd_shuffle_xor(v, 2);
}
inline float ds_col_sum(float v) {
    v = v + simd_shuffle_xor(v, 4);
    return v + simd_shuffle_xor(v, 8);
}
inline float ds_row_max(float v) {
    v = max(v, simd_shuffle_xor(v, 1));
    return max(v, simd_shuffle_xor(v, 2));
}

struct DsHcP {
    uint T;
};

// y = sum_j pre[j] x_j rounded to BF16, then with NORM the RMSNorm of y with weight nw
// (BF16) rounded to BF16. Each of the DS_ROW_THREADS threads owns 4 float4 of the 4096.
template <bool NORM>
inline void ds_weighted_sum(threadgroup const float *pre, device const float4 *x4, device const ushort4 *nw,
                            device float4 *y4, threadgroup float *sh, uint tid) {
    constexpr uint NV = DS_D / 4 / DS_ROW_THREADS;
    const float p0 = pre[0], p1 = pre[1], p2 = pre[2], p3 = pre[3];
    float4 y[NV];
    float ss = 0;
    for (uint v = 0; v < NV; v++) {
        uint i = tid + v * DS_ROW_THREADS;
        float4 a = p0 * x4[i];
        a = a + p1 * x4[i + DS_D / 4];
        a = a + p2 * x4[i + 2 * DS_D / 4];
        a = a + p3 * x4[i + 3 * DS_D / 4];
        y[v] = bf16_round4(a);
        if (NORM) ss += dot(y[v], y[v]);
    }
    if (NORM) {
        ss = ds_tg_sum(ss, sh, tid, DS_ROW_THREADS);
        float r = precise::rsqrt(ss * (1.0f / float(DS_D)) + DS_NORM_EPS);
        for (uint v = 0; v < NV; v++) {
            uint i = tid + v * DS_ROW_THREADS;
            y4[i] = bf16_round4(bf16_val4(nw[i]) * (y[v] * r));
        }
    } else {
        for (uint v = 0; v < NV; v++) y4[tid + v * DS_ROW_THREADS] = y[v];
    }
}

// Layer mixer split (hc_split_sinkhorn) and block input. One threadgroup of DS_ROW_THREADS
// per token. part from dsv4_hc_mix (nmix 24), scale [3], base [24]; x [T][4][4096].
// Writes y [T][4096] (the BF16 block input, or with NORM its RMSNorm with weight nw),
// post [T][4] and comb [T][4][4] (comb[j][k], j the source stream) for dsv4_hc_post.
template <bool NORM>
inline void ds_hc_split(device const float *part, device const float *scale, device const float *base,
                        device const float4 *x, device const ushort4 *nw, device float4 *y, device float *post,
                        device float *comb, threadgroup float *sh, threadgroup float *pre, uint t, uint tid) {
    if (tid < 32) {
        device const float *pt = part + (size_t)t * DS_SLICES * (DS_MIX + 1);
        const uint l = tid;
        // lanes 0-3 pre, 4-7 post, 0-15 comb (lane 4 j + k)
        if (l < 2 * DS_HC) {
            float m = ds_mix_final(pt, DS_MIX, l);
            if (l < DS_HC) pre[l] = ds_sigmoid(m * scale[0] + base[l]) + DS_HC_EPS;
            else post[t * DS_HC + (l - DS_HC)] = 2.0f * ds_sigmoid(m * scale[1] + base[l]);
        }
        float c = 0;
        if (l < DS_HC * DS_HC) c = ds_mix_final(pt, DS_MIX, 2 * DS_HC + l) * scale[2] + base[2 * DS_HC + l];
        // comb = softmax over k + eps
        float mx = ds_row_max(c);
        c = precise::exp(c - mx);
        c = precise::divide(c, ds_row_sum(c)) + DS_HC_EPS;
        // comb = comb / (column sum + eps)
        c = precise::divide(c, ds_col_sum(c) + DS_HC_EPS);
        for (uint it = 0; it < DS_SINKHORN - 1; it++) {
            c = precise::divide(c, ds_row_sum(c) + DS_HC_EPS);
            c = precise::divide(c, ds_col_sum(c) + DS_HC_EPS);
        }
        if (l < DS_HC * DS_HC) comb[t * DS_HC * DS_HC + l] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ds_weighted_sum<NORM>(pre, x + (size_t)t * (DS_HCD / 4), nw, y + (size_t)t * (DS_D / 4), sh, tid);
}

kernel void dsv4_hc_split(device const float *part [[buffer(0)]], device const float *scale [[buffer(1)]],
                          device const float *base [[buffer(2)]], device const float4 *x [[buffer(3)]],
                          device float4 *y [[buffer(4)]], device float *post [[buffer(5)]],
                          device float *comb [[buffer(6)]], uint t [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float sh[32];
    threadgroup float pre[DS_HC];
    ds_hc_split<false>(part, scale, base, x, nullptr, y, post, comb, sh, pre, t, tid);
}

// The same with the block's RMSNorm (attn_norm or ffn_norm, BF16 weight [4096]) fused.
kernel void dsv4_hc_split_norm(device const float *part [[buffer(0)]], device const float *scale [[buffer(1)]],
                               device const float *base [[buffer(2)]], device const float4 *x [[buffer(3)]],
                               device float4 *y [[buffer(4)]], device float *post [[buffer(5)]],
                               device float *comb [[buffer(6)]], device const ushort4 *nw [[buffer(7)]],
                               uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float sh[32];
    threadgroup float pre[DS_HC];
    ds_hc_split<true>(part, scale, base, x, nw, y, post, comb, sh, pre, t, tid);
}

// Final collapse (hc_head): pre = sigmoid(m * scale + base) + eps, no Sinkhorn; part from
// dsv4_hc_mix with nmix 4, scale [1], base [4]. y as dsv4_hc_split; with NORM the final norm.
template <bool NORM>
inline void ds_hc_head(device const float *part, device const float *scale, device const float *base,
                       device const float4 *x, device const ushort4 *nw, device float4 *y, threadgroup float *sh,
                       threadgroup float *pre, uint t, uint tid) {
    if (tid < DS_HC) {
        device const float *pt = part + (size_t)t * DS_SLICES * (DS_HC + 1);
        pre[tid] = ds_sigmoid(ds_mix_final(pt, DS_HC, tid) * scale[0] + base[tid]) + DS_HC_EPS;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ds_weighted_sum<NORM>(pre, x + (size_t)t * (DS_HCD / 4), nw, y + (size_t)t * (DS_D / 4), sh, tid);
}

kernel void dsv4_hc_head(device const float *part [[buffer(0)]], device const float *scale [[buffer(1)]],
                         device const float *base [[buffer(2)]], device const float4 *x [[buffer(3)]],
                         device float4 *y [[buffer(4)]], uint t [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float sh[32];
    threadgroup float pre[DS_HC];
    ds_hc_head<false>(part, scale, base, x, nullptr, y, sh, pre, t, tid);
}

kernel void dsv4_hc_head_norm(device const float *part [[buffer(0)]], device const float *scale [[buffer(1)]],
                              device const float *base [[buffer(2)]], device const float4 *x [[buffer(3)]],
                              device float4 *y [[buffer(4)]], device const ushort4 *nw [[buffer(5)]],
                              uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float sh[32];
    threadgroup float pre[DS_HC];
    ds_hc_head<true>(part, scale, base, x, nw, y, sh, pre, t, tid);
}

// hc_post: out[t][k] = BF16(post[k] f + sum_j comb[j][k] res[t][j]), the sum over j in order.
// f [T][4096] (block output), res and out [T][4][4096]; out may alias res. One thread per
// float4 of a token's 4096: grid T * 1024 threads.
kernel void dsv4_hc_post(device const float4 *f [[buffer(0)]], device const float4 *res [[buffer(1)]],
                         device const float *post [[buffer(2)]], device const float *comb [[buffer(3)]],
                         device float4 *out [[buffer(4)]], constant DsHcP &p [[buffer(5)]],
                         uint gid [[thread_position_in_grid]]) {
    const uint t = gid / (DS_D / 4), i = gid % (DS_D / 4);
    if (t >= p.T) return;
    device const float *c = comb + t * DS_HC * DS_HC;
    device const float4 *r = res + (size_t)t * (DS_HCD / 4) + i;
    const float4 fv = f[(size_t)t * (DS_D / 4) + i];
    const float4 r0 = r[0], r1 = r[DS_D / 4], r2 = r[2 * DS_D / 4], r3 = r[3 * DS_D / 4];
    float4 o[DS_HC];
    for (uint k = 0; k < DS_HC; k++) {
        float4 a = c[k] * r0;
        a = a + c[DS_HC + k] * r1;
        a = a + c[2 * DS_HC + k] * r2;
        a = a + c[3 * DS_HC + k] * r3;
        o[k] = bf16_round4(post[t * DS_HC + k] * fv + a);
    }
    device float4 *ot = out + (size_t)t * (DS_HCD / 4) + i;
    for (uint k = 0; k < DS_HC; k++) ot[k * (DS_D / 4)] = o[k];
}

struct DsNormP {
    uint n; // row length, a multiple of 4, at most 4 * 4 * DS_ROW_THREADS
    uint T;
};

// RMSNorm: out = BF16(w * (x * rsqrt(mean(x^2) + 1e-6))), x [T][n] f32, w BF16 [n].
// One threadgroup of DS_ROW_THREADS per row.
kernel void dsv4_rmsnorm(device const float4 *x [[buffer(0)]], device const ushort4 *w [[buffer(1)]],
                         device float4 *out [[buffer(2)]], constant DsNormP &p [[buffer(3)]],
                         uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float sh[32];
    const uint n4 = p.n / 4;
    device const float4 *xr = x + (size_t)t * n4;
    float ss = 0;
    for (uint i = tid; i < n4; i += DS_ROW_THREADS) ss += dot(xr[i], xr[i]);
    ss = ds_tg_sum(ss, sh, tid, DS_ROW_THREADS);
    const float r = precise::rsqrt(precise::divide(ss, float(p.n)) + DS_NORM_EPS);
    for (uint i = tid; i < n4; i += DS_ROW_THREADS) out[(size_t)t * n4 + i] = bf16_round4(bf16_val4(w[i]) * (xr[i] * r));
}

// ---- router ------------------------------------------------------------------------------

// sqrt(softplus(l)) with torch's softplus (beta 1, threshold 20: l above 20 is returned as is).
// log1p(u) as log(1 + u) * u / ((1 + u) - 1), exact to a few ulp (no log1p in the library).
inline float ds_sqrt_softplus(float l) {
    float sp;
    if (l > 20.0f) {
        sp = l;
    } else {
        float u = precise::exp(l);
        float w = 1.0f + u;
        sp = (w == 1.0f) ? u : precise::divide(precise::log(w) * u, w - 1.0f);
    }
    return precise::sqrt(sp);
}

// Dot product of a BF16 row [4096] with x [4096] over one simdgroup; lane 0 gets it.
inline float ds_row_dot(device const ushort4 *w, device const float4 *x, uint lane) {
    float acc = 0;
    for (uint i = lane; i < DS_D / 4; i += 32) acc += dot(bf16_val4(w[i]), x[i]);
    return simd_sum(acc);
}

struct DsRouterP {
    uint T;
};

// Gate logits [T][256] = W (BF16 [256][4096]) . x[t] in f32. Grid (256 / 8, T), 256 threads:
// one simdgroup per expert row.
kernel void dsv4_router_logits(device const ushort4 *w [[buffer(0)]], device const float4 *x [[buffer(1)]],
                               device float *logits [[buffer(2)]], uint2 g [[threadgroup_position_in_grid]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint e = g.x * 8 + sg, t = g.y;
    float v = ds_row_dot(w + (size_t)e * (DS_D / 4), x + (size_t)t * (DS_D / 4), lane);
    if (lane == 0) logits[t * DS_NE + e] = v;
}

// (value, index) is preferred to (v2, i2): greater value, or equal value and lower index.
inline bool ds_better(float v, uint i, float v2, uint i2) { return v > v2 || (v == v2 && i < i2); }

// Score routing: s = sqrt(softplus(logits)), the 6 experts of largest s + bias (ties to the
// lower index, in descending order), weights = s at those / their sum * 1.5.
// One threadgroup of 256 threads per token (thread e owns expert e).
kernel void dsv4_router_select(device const float *logits [[buffer(0)]], device const float *bias [[buffer(1)]],
                               device uint *ids [[buffer(2)]], device float *weights [[buffer(3)]],
                               uint t [[threadgroup_position_in_grid]], uint e [[thread_position_in_threadgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sv[DS_NE];
    threadgroup float bv[8];
    threadgroup uint bi[8];
    threadgroup uint sel[DS_TOPK];
    const float s = ds_sqrt_softplus(logits[t * DS_NE + e]);
    sv[e] = s;
    const float key = s + bias[e];
    bool taken = false;
    for (uint k = 0; k < DS_TOPK; k++) {
        float v = taken ? -INFINITY : key;
        uint i = taken ? 0xffffffffu : e;
        for (uint off = 16; off > 0; off >>= 1) {
            float v2 = simd_shuffle_xor(v, off);
            uint i2 = simd_shuffle_xor(i, off);
            if (ds_better(v2, i2, v, i)) { v = v2; i = i2; }
        }
        if (lane == 0) { bv[sg] = v; bi[sg] = i; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (e == 0) {
            float bvv = bv[0];
            uint bii = bi[0];
            for (uint j = 1; j < 8; j++)
                if (ds_better(bv[j], bi[j], bvv, bii)) { bvv = bv[j]; bii = bi[j]; }
            sel[k] = bii;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sel[k] == e) taken = true;
    }
    if (e == 0) {
        float sum = 0;
        for (uint k = 0; k < DS_TOPK; k++) sum += sv[sel[k]];
        for (uint k = 0; k < DS_TOPK; k++) {
            ids[t * DS_TOPK + k] = sel[k];
            weights[t * DS_TOPK + k] = precise::divide(sv[sel[k]], sum) * DS_ROUTE_SCALE;
        }
    }
}

// Hash routing (layers 0-2): ids = tid2eid[token] (a uint32 copy of the pack's int64
// [129280][6]); only those 6 gate rows are computed: s = sqrt(softplus(W[id] . x)), weights
// = s / their sum * 1.5, in tid2eid order. One threadgroup of 6 simdgroups per token.
kernel void dsv4_router_hash(device const ushort4 *w [[buffer(0)]], device const float4 *x [[buffer(1)]],
                             device const uint *tid2eid [[buffer(2)]], device const uint *tokens [[buffer(3)]],
                             device uint *ids [[buffer(4)]], device float *weights [[buffer(5)]],
                             uint t [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sv[DS_TOPK];
    const uint id = tid2eid[(size_t)tokens[t] * DS_TOPK + sg];
    float l = ds_row_dot(w + (size_t)id * (DS_D / 4), x + (size_t)t * (DS_D / 4), lane);
    if (lane == 0) sv[sg] = ds_sqrt_softplus(l);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float sum = 0;
        for (uint k = 0; k < DS_TOPK; k++) sum += sv[k];
        for (uint k = 0; k < DS_TOPK; k++) {
            ids[t * DS_TOPK + k] = tid2eid[(size_t)tokens[t] * DS_TOPK + k];
            weights[t * DS_TOPK + k] = precise::divide(sv[k], sum) * DS_ROUTE_SCALE;
        }
    }
}
