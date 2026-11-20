// DeepSeek-V4-Flash KV compressor and indexer (the formulas and their sources are in the
// scalar reference's header, src/dsv4_ref_cmp.h).
//
// Compressor kernels, instantiated for (ratio, head dim, rotate) = (4, 512, no) CSA,
// (128, 512, no) HCA and (4, 128, yes) the indexer's compressor:
//   dsv4_cmp_step_*   one decode token at P.p0: writes its state row, and when the window closes
//                     pools, post-processes and writes entry P.p0 / ratio. Grid 1 x D threads.
//   dsv4_cmp_emit_*   a chunk of P.T tokens from P.p0 (any p0; R's prefill is p0 = 0): the
//                     P.n_emit entries from P.e0 = p0 / ratio whose window closes in the chunk,
//                     rows read from the chunk or, for tokens before p0, from the state.
//                     Grid n_emit x D threads.
//   dsv4_cmp_state_*  then the state after the chunk, as T decode steps leave it. Grid
//                     coff D / 256 x 256 threads; encode it after emit (emit reads the old state).
// Buffers: kv, score [T][coff D] f32 (the wkv and wgate outputs), ape [ratio][coff D] f32,
// kv_state, score_state [coff ratio][coff D] f32, norm weight [D] bf16, cs [n_emit][32] float2
// (cos, sin at each entry's first position, from the host), cache [entries][D] bf16,
// pool [n_emit][D] f32 (pooled values before post, written when P.keep_pool).
//
// Indexer kernels:
//   dsv4_idx_q      q [64][128] (bf16 values in f32) to the rotated, FP4-simulated query, and
//                   the head weights from weights_proj's output. Grid 64 x 128 threads.
//   dsv4_idx_score  scores of the first P.n_vis cached keys. Grid ceil(n_vis / 32) x 128.
//   dsv4_idx_topk   the index list of min(512, P.n_total) values. Grid 1 x 1024.
//
// Arithmetic that decides a rounding is done in a fixed order the reference repeats, with
// contraction and reassociation off, the precise:: divide and sqrt, and the softmax exp written
// out (cmp_exp, the reference's dsv4c_exp), so the compressor is bit-exact; values that are infinite in R (the score of an
// empty state row) are never computed on: those rows are skipped by position.
#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

constant constexpr uint CMP_ROPE = 64;
constant constexpr uint CMP_PAIRS = 32;
constant constexpr float CMP_EPS = 1e-6f;
constant constexpr float CMP_HSCALE = 0.08838834764831845f; // 128^-0.5
constant constexpr uint IDX_H = 64;
constant constexpr uint IDX_D = 128;
constant constexpr uint IDX_TOPK = 512;
constant constexpr float IDX_WSCALE = 0.011048543456039806f; // 128^-0.5 * 64^-0.5
constant constexpr uint IDX_TILE = 32;                       // keys per score threadgroup

struct Dsv4CmpP {
    uint p0;        // first token position of the chunk (step: the token)
    uint T;         // tokens in the chunk
    uint e0;        // first entry emitted (p0 / ratio)
    uint n_emit;    // entries emitted
    uint keep_pool; // also write the pooled values
    uint pad0, pad1, pad2;
};

struct Dsv4IdxP {
    uint n_vis;   // visible entries, scored
    uint n_total; // entries in the cache (k = min(512, n_total))
    int offset;   // added to every valid index
    uint pad;
};

inline float bf16r(float v) {
    uint u = as_type<uint>(v);
    u += 0x7fffu + ((u >> 16) & 1u);
    return as_type<float>(u & 0xffff0000u);
}
inline float bf16_load(ushort b) { return as_type<float>(uint(b) << 16); }
inline ushort bf16_bits(float v) { return ushort(as_type<uint>(v) >> 16); } // v already bf16

inline float e4m3_round(float v) {
    const float a = fabs(v);
    float r;
    if (a < 0.015625f) {
        r = rint(a * 512.0f) * 0.001953125f;
    } else {
        uint u = as_type<uint>(a);
        u += 0x7ffffu + ((u >> 20) & 1u);
        r = as_type<float>(u & 0xfff00000u);
    }
    return copysign(r, v);
}

inline float e2m1_round(float v) {
    const float a = fabs(v);
    float r;
    if (a < 2.0f) r = rint(a * 2.0f) * 0.5f;
    else if (a < 4.0f) r = rint(a);
    else r = rint(a * 0.5f) * 2.0f;
    return copysign(r, v);
}

inline float round_scale(float amax, float floor_, float inv_max, thread float &inv) {
    const uint b = as_type<uint>(max(amax, floor_) * inv_max);
    const int e = int((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0 ? 1 : 0);
    inv = as_type<float>(uint(127 - e) << 23);
    return as_type<float>(uint(127 + e) << 23);
}

// RoPE of x[D - 64, D) in place (interleaved pairs), bf16; every thread of the group calls it.
template <uint D>
inline void tg_rope(threadgroup float *x, uint c, device const float2 *cs) {
    float v = x[c];
    if (c >= D - CMP_ROPE) {
        const float re = x[c & ~1u], im = x[c | 1u];
        const float2 t = cs[(c - (D - CMP_ROPE)) >> 1];
        v = (c & 1u) ? bf16r(re * t.y + im * t.x) : bf16r(re * t.x - im * t.y);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    x[c] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Hadamard (Sylvester butterflies in f32, times 128^-0.5, bf16) then FP4 qdq per 32 (one
// simdgroup per block); returns this thread's value. D threads, D a power of two.
template <uint D>
inline float tg_rotate_fp4(threadgroup float *x, uint c) {
    for (uint h = 1; h < D; h <<= 1) {
        if ((c & h) == 0) {
            const float a = x[c], b = x[c + h];
            x[c] = a + b;
            x[c + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float v = bf16r(x[c] * CMP_HSCALE);
    const float amax = simd_max(fabs(v));
    float inv;
    const float s = round_scale(amax, 0x1.8p-124f, 1.0f / 6.0f, inv);
    return e2m1_round(clamp(v * inv, -6.0f, 6.0f)) * s;
}

// R:368-378 for one entry: D threads, c the channel.
template <uint D, bool ROT>
inline void cmp_post(float pooled, uint c, threadgroup float *x, threadgroup float *red,
                     device const ushort *normw, device const float2 *cs, device ushort *out) {
    x[c] = bf16r(pooled);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c < 32) {
        float p = 0.0f;
        for (uint i = c; i < D; i += 32) p += x[i] * x[i];
        red[c] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c == 0) {
        float ss = 0.0f;
        for (uint j = 0; j < 32; j++) ss += red[j];
        red[32] = precise::divide(1.0f, precise::sqrt(ss * (1.0f / float(D)) + CMP_EPS));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float y = bf16r(bf16_load(normw[c]) * (x[c] * red[32]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    x[c] = y;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tg_rope<D>(x, c, cs);
    float v;
    if (ROT) {
        v = tg_rotate_fp4<D>(x, c);
    } else {
        v = x[c];
        const float m = simd_max(fabs(v));
        if ((c & 31u) == 0) red[40 + (c >> 5)] = m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (c < D - CMP_ROPE) {
            const uint b = c >> 6;
            float inv;
            const float s = round_scale(max(red[40 + 2 * b], red[41 + 2 * b]), 1e-4f, 1.0f / 448.0f, inv);
            v = e4m3_round(clamp(v * inv, -448.0f, 448.0f)) * s;
        }
    }
    out[c] = bf16_bits(v);
}

// exp(x) for x <= 0 by range reduction and a degree-7 polynomial, the same operations as
// dsv4c_exp in the reference (bit-identical on both sides); below -87 it is exp(-87).
inline float cmp_exp(float x) {
    x = max(x, -87.0f);
    const float n = rint(x * 1.44269504088896341f);
    const float r = (x - n * 0.693145751953125f) - n * 1.428606765330187045e-06f;
    float p = 1.0f / 5040.0f;
    p = p * r + 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    return p * as_type<float>(uint(int(n) + 127) << 23);
}

// Softmax over the valid rows and the weighted sum, rows in order (R:358, 365, 348).
template <uint ROWS, typename Src>
inline float cmp_pool(thread const Src &src) {
    float m = 0.0f, k, s;
    bool any = false;
    for (uint r = 0; r < ROWS; r++)
        if (src.get(r, k, s)) {
            m = any ? max(m, s) : s;
            any = true;
        }
    float sum = 0.0f;
    for (uint r = 0; r < ROWS; r++)
        if (src.get(r, k, s)) sum += cmp_exp(s - m);
    float acc = 0.0f;
    for (uint r = 0; r < ROWS; r++)
        if (src.get(r, k, s)) acc += k * precise::divide(cmp_exp(s - m), sum);
    return acc;
}

// Rows of entry w as the decode step sees them: all in the state.
template <uint RATIO, uint D>
struct StateRows {
    static constant constexpr uint COFF = RATIO == 4 ? 2 : 1;
    static constant constexpr uint CD = COFF * D;
    device const float *kvs, *scs;
    uint c;
    bool first; // window 0: the previous-window rows are empty
    bool get(uint r, thread float &k, thread float &s) const {
        if (COFF == 2 && r < RATIO && first) return false;
        const uint ch = (COFF == 2 && r >= RATIO) ? D + c : c;
        k = kvs[r * CD + ch];
        s = scs[r * CD + ch];
        return true;
    }
};

// Rows of entry w in the chunk form: token t from the chunk if t >= p0, else from the state.
template <uint RATIO, uint D>
struct ChunkRows {
    static constant constexpr uint COFF = RATIO == 4 ? 2 : 1;
    static constant constexpr uint CD = COFF * D;
    device const float *kv, *score, *ape, *kvs, *scs;
    uint c, w, p0;
    bool get(uint r, thread float &k, thread float &s) const {
        int t;
        uint ch = c;
        if (COFF == 2) {
            if (r < RATIO) t = int(RATIO * w) - int(RATIO) + int(r);
            else {
                t = int(RATIO * w) + int(r - RATIO);
                ch = D + c;
            }
        } else {
            t = int(RATIO * w + r);
        }
        if (t < 0) return false;
        const uint tu = uint(t);
        if (tu >= p0) {
            const ulong o = ulong(tu - p0) * CD + ch;
            k = kv[o];
            s = score[o] + ape[(tu % RATIO) * CD + ch];
        } else {
            const uint row = COFF == 2 ? ((tu / RATIO == p0 / RATIO) ? RATIO + tu % RATIO : tu % RATIO) : tu % RATIO;
            k = kvs[row * CD + ch];
            s = scs[row * CD + ch];
        }
        return true;
    }
};

template <uint RATIO, uint D, bool ROT>
kernel void dsv4_cmp_step(device const float *kv [[buffer(0)]], device const float *score [[buffer(1)]],
                          device const float *ape [[buffer(2)]], device float *kvs [[buffer(3)]],
                          device float *scs [[buffer(4)]], device const ushort *normw [[buffer(5)]],
                          device const float2 *cs [[buffer(6)]], device ushort *cache [[buffer(7)]],
                          device float *pool [[buffer(8)]], constant Dsv4CmpP &P [[buffer(9)]],
                          uint c [[thread_position_in_threadgroup]]) {
    constexpr uint COFF = RATIO == 4 ? 2 : 1, CD = COFF * D;
    threadgroup float x[D];
    threadgroup float red[64];
    const uint pos = P.p0, j = pos % RATIO, row = (COFF - 1) * RATIO + j;
    for (uint h = 0; h < COFF; h++) { // R:351-354, R:362-363
        const uint ch = h * D + c;
        kvs[row * CD + ch] = kv[ch];
        scs[row * CD + ch] = score[ch] + ape[j * CD + ch];
    }
    if ((pos + 1) % RATIO != 0) return;
    StateRows<RATIO, D> src{kvs, scs, c, pos < RATIO};
    const float pooled = cmp_pool<COFF * RATIO>(src);
    if (COFF == 2) // R:359-360
        for (uint h = 0; h < 2; h++)
            for (uint r = 0; r < RATIO; r++) {
                const uint ch = h * D + c;
                kvs[r * CD + ch] = kvs[(RATIO + r) * CD + ch];
                scs[r * CD + ch] = scs[(RATIO + r) * CD + ch];
            }
    if (P.keep_pool) pool[c] = pooled;
    cmp_post<D, ROT>(pooled, c, x, red, normw, cs, cache + ulong(pos / RATIO) * D);
}

template <uint RATIO, uint D, bool ROT>
kernel void dsv4_cmp_emit(device const float *kv [[buffer(0)]], device const float *score [[buffer(1)]],
                          device const float *ape [[buffer(2)]], device const float *kvs [[buffer(3)]],
                          device const float *scs [[buffer(4)]], device const ushort *normw [[buffer(5)]],
                          device const float2 *cs [[buffer(6)]], device ushort *cache [[buffer(7)]],
                          device float *pool [[buffer(8)]], constant Dsv4CmpP &P [[buffer(9)]],
                          uint e [[threadgroup_position_in_grid]], uint c [[thread_position_in_threadgroup]]) {
    threadgroup float x[D];
    threadgroup float red[64];
    const uint w = P.e0 + e;
    ChunkRows<RATIO, D> src{kv, score, ape, kvs, scs, c, w, P.p0};
    const float pooled = cmp_pool<(RATIO == 4 ? 2 : 1) * RATIO>(src);
    if (P.keep_pool) pool[ulong(e) * D + c] = pooled;
    cmp_post<D, ROT>(pooled, c, x, red, normw, cs + e * CMP_PAIRS, cache + ulong(w) * D);
}

template <uint RATIO, uint D>
kernel void dsv4_cmp_state(device const float *kv [[buffer(0)]], device const float *score [[buffer(1)]],
                           device const float *ape [[buffer(2)]], device float *kvs [[buffer(3)]],
                           device float *scs [[buffer(4)]], constant Dsv4CmpP &P [[buffer(9)]],
                           uint ch [[thread_position_in_grid]]) {
    constexpr uint COFF = RATIO == 4 ? 2 : 1, CD = COFF * D;
    const int p0 = int(P.p0), pe = int(P.p0 + P.T) - 1;
    if (COFF == 2) {
        float ko[2 * RATIO], so[2 * RATIO];
        for (uint r = 0; r < 2 * RATIO; r++) {
            ko[r] = kvs[r * CD + ch];
            so[r] = scs[r * CD + ch];
        }
        const int closed_before = p0 / int(RATIO), closed_after = (pe + 1) / int(RATIO);
        for (uint r = 0; r < 2 * RATIO; r++) {
            const int j = int(r % RATIO);
            int t; // the token decode would hold in this row, or -1
            if (r >= RATIO) t = pe - (((pe - j) % int(RATIO)) + int(RATIO)) % int(RATIO);
            else t = closed_after > closed_before ? int(RATIO) * (closed_after - 1) + j : -1;
            float k, s;
            if (t >= p0) {
                const ulong o = ulong(t - p0) * CD + ch;
                k = kv[o];
                s = score[o] + ape[uint(j) * CD + ch];
            } else if (r < RATIO && t >= 0) { // closed in the chunk, token from before it
                k = ko[RATIO + r];
                s = so[RATIO + r];
            } else {
                k = ko[r];
                s = so[r];
            }
            kvs[r * CD + ch] = k;
            scs[r * CD + ch] = s;
        }
    } else {
        for (uint r = 0; r < RATIO; r++) {
            const int t = pe - (((pe - int(r)) % int(RATIO)) + int(RATIO)) % int(RATIO);
            if (t >= p0) {
                const ulong o = ulong(t - p0) * CD + ch;
                kvs[r * CD + ch] = kv[o];
                scs[r * CD + ch] = score[o] + ape[r * CD + ch];
            }
        }
    }
}

#define CMP_INST(R, D, ROT, NAME)                                                                       \
    template [[host_name("dsv4_cmp_step_" NAME)]] kernel void dsv4_cmp_step<R, D, ROT>(                 \
        device const float *, device const float *, device const float *, device float *, device float *, \
        device const ushort *, device const float2 *, device ushort *, device float *, constant Dsv4CmpP &, uint); \
    template [[host_name("dsv4_cmp_emit_" NAME)]] kernel void dsv4_cmp_emit<R, D, ROT>(                 \
        device const float *, device const float *, device const float *, device const float *,           \
        device const float *, device const ushort *, device const float2 *, device ushort *, device float *, \
        constant Dsv4CmpP &, uint, uint);                                                                \
    template [[host_name("dsv4_cmp_state_" NAME)]] kernel void dsv4_cmp_state<R, D>(                    \
        device const float *, device const float *, device const float *, device float *, device float *, \
        constant Dsv4CmpP &, uint);
CMP_INST(4, 512, false, "csa")
CMP_INST(128, 512, false, "hca")
CMP_INST(4, 128, true, "idx")

// ---------------------------------------------------------------------------- indexer -----

kernel void dsv4_idx_q(device const float *q [[buffer(0)]], device const float *wp [[buffer(1)]],
                       device const float2 *cs [[buffer(2)]], device float *qx [[buffer(3)]],
                       device float *w [[buffer(4)]], uint h [[threadgroup_position_in_grid]],
                       uint c [[thread_position_in_threadgroup]]) {
    threadgroup float x[IDX_D];
    x[c] = q[h * IDX_D + c];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tg_rope<IDX_D>(x, c, cs);                // R:419
    qx[h * IDX_D + c] = tg_rotate_fp4<IDX_D>(x, c); // R:420-422
    if (c == 0) w[h] = bf16r(wp[h] * IDX_WSCALE); // R:424
}

// One threadgroup of 4 simdgroups per 32 keys: S[h][s] = q[h] . k[s] with 8x8 f32 matrices
// (inputs are FP4 values times powers of two, so the f32 products are exact), then per key
// bf16(sum_h bf16(relu(bf16(S)) * w[h])) with heads in order.
kernel void dsv4_idx_score(device const ushort *keys [[buffer(0)]], device const float *qx [[buffer(1)]],
                           device const float *w [[buffer(2)]], device float *scores [[buffer(3)]],
                           constant Dsv4IdxP &P [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                           uint t [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float ks[IDX_TILE * IDX_D];
    threadgroup float S[IDX_H * IDX_TILE];
    const uint s0 = tg * IDX_TILE;
    for (uint i = t; i < IDX_TILE * IDX_D; i += 128) {
        const uint s = s0 + i / IDX_D;
        ks[i] = s < P.n_vis ? bf16_load(keys[ulong(s) * IDX_D + i % IDX_D]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 acc[IDX_H / 8];
    for (uint hb = 0; hb < IDX_H / 8; hb++) acc[hb] = simdgroup_float8x8(0.0f);
    for (uint kk = 0; kk < IDX_D / 8; kk++) {
        simdgroup_float8x8 B;
        simdgroup_load(B, ks + sg * 8 * IDX_D + kk * 8, IDX_D, ulong2(0, 0), true);
        for (uint hb = 0; hb < IDX_H / 8; hb++) {
            simdgroup_float8x8 A;
            simdgroup_load(A, qx + hb * 8 * IDX_D + kk * 8, IDX_D);
            simdgroup_multiply_accumulate(acc[hb], A, B, acc[hb]);
        }
    }
    for (uint hb = 0; hb < IDX_H / 8; hb++) simdgroup_store(acc[hb], S + hb * 8 * IDX_TILE + sg * 8, IDX_TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < IDX_TILE && s0 + t < P.n_vis) {
        float sum = 0.0f;
        for (uint h = 0; h < IDX_H; h++) sum += bf16r(max(bf16r(S[h * IDX_TILE + t]), 0.0f) * w[h]);
        scores[s0 + t] = bf16r(sum);
    }
}

// Order key of a bf16 score: larger score, larger key; -0 as +0.
inline uint score_key(float v) {
    uint u = as_type<uint>(v) >> 16;
    if (u == 0x8000u) u = 0;
    return (u & 0x8000u) ? (~u & 0xffffu) : (u | 0x8000u);
}
inline ulong pack_key(uint key, uint i) { return (ulong(key) << 32) | ulong(0xffffffffu - i); }

// Top min(512, n_total) of the n_vis scores, descending, ties to the lower index: a two-pass
// radix select on the 16-bit keys finds the threshold key, every key above it is taken and the
// threshold ties in index order until the list is full; then a bitonic sort of (key, -index).
kernel void dsv4_idx_topk(device const float *scores [[buffer(0)]], device int *idx [[buffer(1)]],
                          constant Dsv4IdxP &P [[buffer(2)]], uint t [[thread_position_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup atomic_uint cnt;
    threadgroup ulong comp[IDX_TOPK];
    threadgroup uint sh[64];
    const uint n = P.n_vis, k = min(IDX_TOPK, P.n_total), m = min(k, n);
    for (uint i = t; i < IDX_TOPK; i += 1024) comp[i] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (n <= k) {
        for (uint i = t; i < n; i += 1024) comp[i] = pack_key(score_key(scores[i]), i);
    } else {
        if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = t; i < n; i += 1024) atomic_fetch_add_explicit(&hist[score_key(scores[i]) >> 8], 1u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            uint cum = 0, b = 255;
            for (;; b--) {
                const uint hc = atomic_load_explicit(&hist[b], memory_order_relaxed);
                if (cum + hc >= m || b == 0) break;
                cum += hc;
            }
            sh[0] = b;
            sh[1] = cum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint b1 = sh[0];
        if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = t; i < n; i += 1024) {
            const uint key = score_key(scores[i]);
            if ((key >> 8) == b1) atomic_fetch_add_explicit(&hist[key & 255u], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            uint cum = sh[1], b = 255;
            for (;; b--) {
                const uint hc = atomic_load_explicit(&hist[b], memory_order_relaxed);
                if (cum + hc >= m || b == 0) break;
                cum += hc;
            }
            sh[2] = (b1 << 8) | b;
            sh[3] = cum;
            atomic_store_explicit(&cnt, 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint kt = sh[2], above = sh[3], need = m - above;
        for (uint i = t; i < n; i += 1024) {
            const uint key = score_key(scores[i]);
            if (key > kt) comp[atomic_fetch_add_explicit(&cnt, 1u, memory_order_relaxed)] = pack_key(key, i);
        }
        uint base = 0;
        for (uint i0 = 0; i0 < n && base < need; i0 += 1024) {
            const uint i = i0 + t;
            const uint f = (i < n && score_key(scores[i]) == kt) ? 1u : 0u;
            const uint pre = simd_prefix_exclusive_sum(f), tot = simd_sum(f);
            if (lane == 0) sh[8 + sg] = tot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint off = 0, all = 0;
            for (uint g = 0; g < 32; g++) {
                const uint v = sh[8 + g];
                off += g < sg ? v : 0u;
                all += v;
            }
            const uint slot = base + off + pre;
            if (f && slot < need) comp[above + slot] = pack_key(kt, i);
            base += all;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 2; kk <= IDX_TOPK; kk <<= 1)
        for (uint j = kk >> 1; j > 0; j >>= 1) {
            if (t < IDX_TOPK) {
                const uint l = t ^ j;
                if (l > t) {
                    const ulong a = comp[t], b = comp[l];
                    if (((t & kk) == 0) ? (a < b) : (a > b)) {
                        comp[t] = b;
                        comp[l] = a;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    for (uint r = t; r < k; r += 1024) idx[r] = r < m ? int(0xffffffffu - uint(comp[r])) + P.offset : -1;
}
