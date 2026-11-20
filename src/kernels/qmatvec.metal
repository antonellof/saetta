// Affine-quantized matrix-vector product, MLX layout.
// W is [out, in], BITS bits per value, little endian, lowest bits first.
// Per group of GROUP values along `in`: w = q * scale + bias (bf16 scale, bias).
// y[o] = sum_i w[o, i] * x[i]. One simdgroup (32 threads) per output row; each
// lane takes whole groups and the lanes are summed with simd_sum.
// Naive and correct first; no tiling, no fusion.
//
// The MLX pack stores tensors without alignment padding, so weights, scales and
// biases are read through byte pointers. shift[0..2] are the byte offsets of
// w, scales and biases from their 16-byte aligned bindings.
#include <metal_stdlib>
using namespace metal;

inline float bf16_at(device const uchar *p, uint i) {
    const uint lo = p[2 * i], hi = p[2 * i + 1];
    return as_type<float>(((hi << 8) | lo) << 16);
}

template <int BITS, int GROUP>
inline void qmatvec_row(device const uchar *w, device const uchar *scales, device const uchar *biases,
                        device const float *x, device float *y, uint in, uint row, uint lane) {
    // One lane per group: its BPG weight bytes are read as aligned 32-bit words and
    // shifted across the tensor's misalignment (the pack has no alignment padding).
    constexpr uint BPG = GROUP * BITS / 8;  // bytes per group: 16 (4/32), 64 (8/64)
    constexpr uint NW = BPG / 4;
    constexpr uint PER = 32 / BITS;         // values per word
    constexpr uint MASK = (1u << BITS) - 1u;
    const uint groups = in / GROUP;
    device const uchar *wr = w + row * (in * BITS / 8);
    float acc = 0.0f;
    for (uint g = lane; g < groups; g += 32) {
        const ulong addr = (ulong)(wr + g * BPG);
        const uint sh = uint(addr & 3);
        device const uint *q = (device const uint *)(addr - sh);
        uint words[NW];
        if (sh == 0) {
            for (uint i = 0; i < NW; i++) words[i] = q[i];
        } else {
            uint prev = q[0];
            for (uint i = 0; i < NW; i++) {
                const uint nx = q[i + 1];
                words[i] = (prev >> (8 * sh)) | (nx << (32 - 8 * sh));
                prev = nx;
            }
        }
        device const float4 *xg = (device const float4 *)(x + g * GROUP);
        float dsum = 0.0f, xsum = 0.0f;
        for (uint i = 0; i < NW; i++) {
            const uint wd = words[i];
            for (uint j = 0; j < PER; j += 4) {
                const float4 xv = xg[(i * PER + j) / 4];
                const float4 qv = float4(float((wd >> (BITS * j)) & MASK), float((wd >> (BITS * (j + 1))) & MASK),
                                         float((wd >> (BITS * (j + 2))) & MASK), float((wd >> (BITS * (j + 3))) & MASK));
                dsum += dot(qv, xv);
                xsum += xv.x + xv.y + xv.z + xv.w;
            }
        }
        acc += bf16_at(scales, row * groups + g) * dsum + bf16_at(biases, row * groups + g) * xsum;
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc;
}

#define QMATVEC(NAME, BITS, GROUP)                                                                          \
kernel void NAME(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],             \
                 device const uchar *biases [[buffer(2)]], device const float *x [[buffer(3)]],             \
                 device float *y [[buffer(4)]], constant uint &in [[buffer(5)]], constant uint &out [[buffer(6)]], \
                 constant uint *shift [[buffer(7)]],                                                         \
                 uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {       \
    if (row < out) qmatvec_row<BITS, GROUP>(w + shift[0], scales + shift[1], biases + shift[2], x, y, in, row, lane); \
}

QMATVEC(qmatvec_4_32, 4, 32)
QMATVEC(qmatvec_8_64, 8, 64)

// Routed experts: weights are rank-3 [E, out, in]. Threadgroup (row, slot) computes row
// `row` of expert idx[slot] on input x + (slot / stride.w) * stride.z, into y + slot * out.
// stride.x: bytes per expert of w; stride.y: bytes per expert of scales and biases.
#define QMATVEC_EXPERT(NAME, BITS, GROUP)                                                                   \
kernel void NAME(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],             \
                 device const uchar *biases [[buffer(2)]], device const float *x [[buffer(3)]],             \
                 device float *y [[buffer(4)]], constant uint &in [[buffer(5)]], constant uint &out [[buffer(6)]], \
                 constant uint *shift [[buffer(7)]], device const uint *idx [[buffer(8)]],                   \
                 constant uint4 &stride [[buffer(9)]],                                                       \
                 uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {       \
    if (tg.x >= out) return;                                                                                 \
    const uint e = idx[tg.y];                                                                                \
    qmatvec_row<BITS, GROUP>(w + shift[0] + ulong(e) * stride.x, scales + shift[1] + ulong(e) * stride.y,                  \
                             biases + shift[2] + ulong(e) * stride.y, x + (tg.y / stride.w) * stride.z, y + tg.y * out, in, tg.x, lane); \
}

QMATVEC_EXPERT(qmatvec_expert_4_32, 4, 32)

// Shape-adapted matvec for matrices the one-simdgroup-per-row kernel serves badly
// (mv-bench: hyper-connection down 320 x 10240 ran at 53 GB/s, up 10240 x 320 at 129 GB/s,
// against about 500 GB/s for large matrices). L lanes per row, 256 threads per threadgroup:
// L = 256 splits a long row over 8 simdgroups; L = 8 packs 4 short rows per simdgroup.
template <int BITS, int GROUP, int L>
inline void qmv_l(device const uchar *w, device const uchar *scales, device const uchar *biases,
                  device const float *x, device float *y, uint in, uint out, uint tg, uint tid, threadgroup float *red) {
    constexpr uint BPG = GROUP * BITS / 8, NW = BPG / 4, PER = 32 / BITS, MASK = (1u << BITS) - 1u;
    constexpr uint RPT = 256 / L;
    const uint row = tg * RPT + tid / L, sub = tid % L, groups = in / GROUP;
    float acc = 0.0f;
    if (row < out) {
        device const uchar *wr = w + row * (in * BITS / 8);
        for (uint g = sub; g < groups; g += L) {
            const ulong addr = (ulong)(wr + g * BPG);
            const uint sh = uint(addr & 3);
            device const uint *q = (device const uint *)(addr - sh);
            uint words[NW];
            if (sh == 0) {
                for (uint i = 0; i < NW; i++) words[i] = q[i];
            } else {
                uint prev = q[0];
                for (uint i = 0; i < NW; i++) { const uint nx = q[i + 1]; words[i] = (prev >> (8 * sh)) | (nx << (32 - 8 * sh)); prev = nx; }
            }
            device const float4 *xg = (device const float4 *)(x + g * GROUP);
            float dsum = 0.0f, xsum = 0.0f;
            for (uint i = 0; i < NW; i++) {
                const uint wd = words[i];
                for (uint j = 0; j < PER; j += 4) {
                    const float4 xv = xg[(i * PER + j) / 4];
                    const float4 qv = float4(float((wd >> (BITS * j)) & MASK), float((wd >> (BITS * (j + 1))) & MASK),
                                             float((wd >> (BITS * (j + 2))) & MASK), float((wd >> (BITS * (j + 3))) & MASK));
                    dsum += dot(qv, xv);
                    xsum += xv.x + xv.y + xv.z + xv.w;
                }
            }
            acc += bf16_at(scales, row * groups + g) * dsum + bf16_at(biases, row * groups + g) * xsum;
        }
    }
    if (L <= 32) {
        for (uint o = L / 2; o > 0; o /= 2) acc += simd_shuffle_xor(acc, o);
        if (sub == 0 && row < out) y[row] = acc;
    } else {
        acc = simd_sum(acc);
        if (tid % 32 == 0) red[tid / 32] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sub == 0 && row < out) {
            float t = 0.0f;
            for (uint k = 0; k < L / 32; k++) t += red[tid / 32 + k];
            y[row] = t;
        }
    }
}

#define QMV_L(NAME, BITS, GROUP, L)                                                                         \
kernel void NAME(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],             \
                 device const uchar *biases [[buffer(2)]], device const float *x [[buffer(3)]],             \
                 device float *y [[buffer(4)]], constant uint &in [[buffer(5)]], constant uint &out [[buffer(6)]], \
                 constant uint *shift [[buffer(7)]],                                                         \
                 uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {    \
    threadgroup float red[8];                                                                                \
    qmv_l<BITS, GROUP, L>(w + shift[0], scales + shift[1], biases + shift[2], x, y, in, out, tg, tid, red);   \
}

QMV_L(qmv4_l8, 4, 32, 8)
QMV_L(qmv4_l64, 4, 32, 64)
QMV_L(qmv4_l256, 4, 32, 256)

// ---- fused decode matvecs ------------------------------------------------------------

// Dot product of row `row` of a 4-bit, group-32 matrix with x, split over L lanes (sub = lane
// index within the row); ACT: input transform x -> silu(x * act_scale) as it is read.
template <int L, bool ACT>
inline float qdot_l(device const uchar *w, device const uchar *scales, device const uchar *biases,
                    device const float *x, uint in, uint row, uint sub, float act_scale) {
    const uint groups = in / 32;
    device const uchar *wr = w + row * (in / 2);
    float acc = 0.0f;
    for (uint g = sub; g < groups; g += L) {
        const ulong addr = (ulong)(wr + g * 16);
        const uint sh = uint(addr & 3);
        device const uint *q = (device const uint *)(addr - sh);
        uint words[4];
        if (sh == 0) {
            for (uint i = 0; i < 4; i++) words[i] = q[i];
        } else {
            uint prev = q[0];
            for (uint i = 0; i < 4; i++) { const uint nx = q[i + 1]; words[i] = (prev >> (8 * sh)) | (nx << (32 - 8 * sh)); prev = nx; }
        }
        device const float4 *xg = (device const float4 *)(x + g * 32);
        float dsum = 0.0f, xsum = 0.0f;
        for (uint i = 0; i < 4; i++) {
            const uint wd = words[i];
            for (uint j = 0; j < 8; j += 4) {
                float4 xv = xg[(i * 8 + j) / 4];
                if (ACT) xv = (xv * act_scale) / (1.0f + exp(-(xv * act_scale)));
                const float4 qv = float4(float((wd >> (4 * j)) & 15u), float((wd >> (4 * (j + 1))) & 15u),
                                         float((wd >> (4 * (j + 2))) & 15u), float((wd >> (4 * (j + 3))) & 15u));
                dsum += dot(qv, xv);
                xsum += xv.x + xv.y + xv.z + xv.w;
            }
        }
        acc += bf16_at(scales, row * groups + g) * dsum + bf16_at(biases, row * groups + g) * xsum;
    }
    return acc;
}

// Sum over the L lanes of a row (L <= 32: shuffles; L > 32: through red, all threads call).
template <int L>
inline float lanes_sum(float acc, uint tid, threadgroup float *red) {
    if (L <= 32) {
        for (uint o = L / 2; o > 0; o /= 2) acc += simd_shuffle_xor(acc, o);
        return acc;
    }
    acc = simd_sum(acc);
    if (tid % 32 == 0) red[tid / 32] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    const uint first = (tid / L) * (L / 32);
    for (uint k = 0; k < L / 32; k++) t += red[first + k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

struct MvP { uint in, nmat, act; float act_scale; uint rows[4]; uint tg0[5]; uint shift[12]; };

// Up to 4 matrices (4-bit, group 32) on the same input, one dispatch; matrix k owns
// threadgroups tg0[k] .. tg0[k+1] and writes y_k. 256 threads, L lanes per row.
template <int L>
inline void qmv_multi(device const uchar *const *W, device float *const *Y, device const float *x, constant MvP &p,
                      uint tg, uint t, uint tid, threadgroup float *red) {
    x += t * p.in; // token t of a small chunk (grid y); decode has t = 0
    uint k = 0;
    while (k + 1 < p.nmat && tg >= p.tg0[k + 1]) k++;
    constexpr uint RPT = 256 / L;
    const uint row = (tg - p.tg0[k]) * RPT + tid / L, sub = tid % L;
    const bool valid = row < p.rows[k];
    float acc = 0.0f;
    if (valid) acc = p.act ? qdot_l<L, true>(W[3 * k] + p.shift[3 * k], W[3 * k + 1] + p.shift[3 * k + 1], W[3 * k + 2] + p.shift[3 * k + 2], x, p.in, row, sub, p.act_scale)
                           : qdot_l<L, false>(W[3 * k] + p.shift[3 * k], W[3 * k + 1] + p.shift[3 * k + 1], W[3 * k + 2] + p.shift[3 * k + 2], x, p.in, row, sub, p.act_scale);
    acc = lanes_sum<L>(acc, tid, red);
    if (valid && sub == 0) Y[k][t * p.rows[k] + row] = acc;
}

#define QMV_MULTI(NAME, L)                                                                                  \
kernel void NAME(device const uchar *w0 [[buffer(0)]], device const uchar *s0 [[buffer(1)]], device const uchar *b0 [[buffer(2)]], \
                 device const uchar *w1 [[buffer(3)]], device const uchar *s1 [[buffer(4)]], device const uchar *b1 [[buffer(5)]], \
                 device const uchar *w2 [[buffer(6)]], device const uchar *s2 [[buffer(7)]], device const uchar *b2 [[buffer(8)]], \
                 device const uchar *w3 [[buffer(9)]], device const uchar *s3 [[buffer(10)]], device const uchar *b3 [[buffer(11)]], \
                 device const float *x [[buffer(12)]], device float *y0 [[buffer(13)]], device float *y1 [[buffer(14)]], \
                 device float *y2 [[buffer(15)]], device float *y3 [[buffer(16)]], constant MvP &p [[buffer(17)]], \
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {      \
    threadgroup float red[8];                                                                                \
    device const uchar *W[12] = {w0, s0, b0, w1, s1, b1, w2, s2, b2, w3, s3, b3};                            \
    device float *Y[4] = {y0, y1, y2, y3};                                                                   \
    qmv_multi<L>(W, Y, x, p, tg.x, tg.y, tid, red);                                                          \
}

QMV_MULTI(qmv_multi_l8, 8)
QMV_MULTI(qmv_multi_l32, 32)
QMV_MULTI(qmv_multi_l256, 256)

// Gate and up of an expert (or the shared expert) in one pass: h = silu(gate . x) * (up . x).
// Threadgroup (row, slot), 32 threads; expert idx[slot] (stride bytes per expert; 0 = shared),
// input x + (slot / div) * xstride, output h + slot * out.
kernel void qmv_gate_up(device const uchar *wg [[buffer(0)]], device const uchar *sg [[buffer(1)]], device const uchar *bg [[buffer(2)]],
                        device const uchar *wu [[buffer(3)]], device const uchar *su [[buffer(4)]], device const uchar *bu [[buffer(5)]],
                        device const float *x [[buffer(6)]], device float *h [[buffer(7)]], device const uint *idx [[buffer(8)]],
                        constant uint4 &d [[buffer(9)]] /* in, out, xstride, div */, constant uint4 &st [[buffer(10)]] /* w, s/b bytes per expert, use idx, - */,
                        constant uint *shift [[buffer(11)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (tg.x >= d.y) return;
    const uint e = st.z ? idx[tg.y] : 0;
    device const float *xs = x + (tg.y / d.w) * d.z;
    float g = qdot_l<32, false>(wg + shift[0] + ulong(e) * st.x, sg + shift[1] + ulong(e) * st.y, bg + shift[2] + ulong(e) * st.y, xs, d.x, tg.x, lane, 1.0f);
    float u = qdot_l<32, false>(wu + shift[3] + ulong(e) * st.x, su + shift[4] + ulong(e) * st.y, bu + shift[5] + ulong(e) * st.y, xs, d.x, tg.x, lane, 1.0f);
    g = simd_sum(g);
    u = simd_sum(u);
    if (lane == 0) h[tg.y * d.y + tg.x] = g / (1.0f + exp(-g)) * u;
}

// ---- small chunks (speculative verification): matvec over up to 8 input rows ------------

// Y[t][row] = W[row] . X[t] for t < T <= 8: each group's weights are read and dequantized
// once and applied to every row of X (a verification chunk reads the weights once, like
// decode, instead of the prefill matmul's per-tile dequantization). One simdgroup per row.
// One row per simdgroup (32-thread threadgroups): best for T <= 2 and small matrices, where the
// many threadgroups hide memory latency; the shared-input layout below wins for T >= 3 on large
// matrices .
template <int BITS, int GROUP, int T>
inline void qmv_rows_row(device const uchar *w, device const uchar *scales, device const uchar *biases,
                         device const float *X, device float *Y, uint in, uint out, uint row, uint lane) {
    // T is a template parameter so the T accumulators stay in registers (a runtime T spilled
    // them: 2-token verification chunks cost 23 ms against 14 for a decode step)
    constexpr uint BPG = GROUP * BITS / 8, NW = BPG / 4, PER = 32 / BITS, MASK = (1u << BITS) - 1u;
    const uint groups = in / GROUP;
    device const uchar *wr = w + row * (in * BITS / 8);
    float acc[T];
    for (int t = 0; t < T; t++) acc[t] = 0.0f;
    for (uint g = lane; g < groups; g += 32) {
        const ulong addr = (ulong)(wr + g * BPG);
        const uint sh = uint(addr & 3);
        device const uint *q = (device const uint *)(addr - sh);
        uint words[NW];
        if (sh == 0) {
            for (uint i = 0; i < NW; i++) words[i] = q[i];
        } else {
            uint prev = q[0];
            for (uint i = 0; i < NW; i++) { const uint nx = q[i + 1]; words[i] = (prev >> (8 * sh)) | (nx << (32 - 8 * sh)); prev = nx; }
        }
        const float s = bf16_at(scales, row * groups + g), b = bf16_at(biases, row * groups + g);
        if (BITS == 8) {
            // 8-bit groups of 64: decode word by word and apply it to every row at once (a
            // whole decoded group is 64 floats of registers; that version ran the Coder's 8-bit
            // layers 3x slower in verification chunks than the one-row matvec)
            float dsum[T], xsum[T];
#pragma clang loop unroll(full)
            for (int t = 0; t < T; t++) { dsum[t] = 0.0f; xsum[t] = 0.0f; }
            for (uint i = 0; i < NW; i++) {
                const uint wd = words[i];
                const float4 qv = float4(float(wd & 255u), float((wd >> 8) & 255u), float((wd >> 16) & 255u), float(wd >> 24));
#pragma clang loop unroll(full)
                for (int t = 0; t < T; t++) {
                    const float4 xv = ((device const float4 *)(X + t * in + g * GROUP))[i];
                    dsum[t] += dot(qv, xv);
                    xsum[t] += xv.x + xv.y + xv.z + xv.w;
                }
            }
#pragma clang loop unroll(full)
            for (int t = 0; t < T; t++) acc[t] += s * dsum[t] + b * xsum[t];
            continue;
        }
        float4 qv[GROUP / 4];
        for (uint i = 0; i < NW; i++) {
            const uint wd = words[i];
            for (uint j = 0; j < PER; j += 4)
                qv[(i * PER + j) / 4] = float4(float((wd >> (BITS * j)) & MASK), float((wd >> (BITS * (j + 1))) & MASK),
                                               float((wd >> (BITS * (j + 2))) & MASK), float((wd >> (BITS * (j + 3))) & MASK));
        }
#pragma clang loop unroll(full)
        for (int t = 0; t < T; t++) {
            device const float4 *xg = (device const float4 *)(X + t * in + g * GROUP);
            float dsum = 0.0f, xsum = 0.0f;
            for (uint i = 0; i < GROUP / 4; i++) {
                const float4 xv = xg[i];
                dsum += dot(qv[i], xv);
                xsum += xv.x + xv.y + xv.z + xv.w;
            }
            acc[t] += s * dsum + b * xsum;
        }
    }
#pragma clang loop unroll(full)
    for (int t = 0; t < T; t++) {
        const float v = simd_sum(acc[t]);
        if (lane == 0) Y[t * out + row] = v;
    }
}

#define QMV_ROWS1(NAME, BITS, GROUP, T)                                                                      \
kernel void NAME(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],             \
                 device const uchar *biases [[buffer(2)]], device const float *X [[buffer(3)]],             \
                 device float *Y [[buffer(4)]], constant uint3 &d [[buffer(5)]] /* in, out, T */,           \
                 constant uint *shift [[buffer(6)]],                                                         \
                 uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {       \
    if (row < d.y) qmv_rows_row<BITS, GROUP, T>(w + shift[0], scales + shift[1], biases + shift[2], X, Y, d.x, d.y, row, lane); \
}

QMV_ROWS1(qmv_rows1_4_32_1, 4, 32, 1)
QMV_ROWS1(qmv_rows1_4_32_2, 4, 32, 2)
QMV_ROWS1(qmv_rows1_4_32_3, 4, 32, 3)
QMV_ROWS1(qmv_rows1_4_32_4, 4, 32, 4)
QMV_ROWS1(qmv_rows1_4_32_5, 4, 32, 5)
QMV_ROWS1(qmv_rows1_4_32_6, 4, 32, 6)
QMV_ROWS1(qmv_rows1_4_32_7, 4, 32, 7)
QMV_ROWS1(qmv_rows1_4_32_8, 4, 32, 8)
QMV_ROWS1(qmv_rows1_8_64_1, 8, 64, 1)
QMV_ROWS1(qmv_rows1_8_64_2, 8, 64, 2)
QMV_ROWS1(qmv_rows1_8_64_3, 8, 64, 3)
QMV_ROWS1(qmv_rows1_8_64_4, 8, 64, 4)
QMV_ROWS1(qmv_rows1_8_64_5, 8, 64, 5)
QMV_ROWS1(qmv_rows1_8_64_6, 8, 64, 6)
QMV_ROWS1(qmv_rows1_8_64_7, 8, 64, 7)
QMV_ROWS1(qmv_rows1_8_64_8, 8, 64, 8)

constant constexpr int ROWS_R = 4; // rows per simdgroup in the chunk kernel

// Y[t][row] = W[row] . X[t] for t < T <= 8 (verification chunks). Layout after the idea of MLX's
// qmv kernel (ml-explore/mlx, MIT; not vendored here, no code taken): each lane owns 8 consecutive inputs, holds them
// for all T rows of X in registers, and applies them to R output rows read as packed words, so
// the input traffic from cache is shared by R rows (one row per simdgroup re-read all T inputs
// per row and was bound by cache bandwidth: in_proj_qkv 519 GB/s at T = 1, 213 at T = 5). The
// bias term uses the lane's partial input sum (bias x sum over the group = sum over its lanes).
// T and R are template parameters so everything stays in registers. 256 threads, R rows per
// simdgroup, 8 * R rows per threadgroup.
template <int BITS, int GROUP, int T, int R>
inline void qmv_rows_mlx(device const uchar *w, device const uchar *scales, device const uchar *biases,
                         device const float *X, device float *Y, uint in, uint out, uint row0, uint lane) {
    constexpr uint WPL = 8 * BITS / 32;          // packed words per lane step (8 inputs): 1 (4-bit) or 2 (8-bit)
    constexpr uint MASK = (1u << BITS) - 1u;
    const uint groups = in / GROUP;
    const uint row_bytes = in * BITS / 8;
    const uint sh = uint((ulong)w & 3);          // constant per tensor: row_bytes is a multiple of 4
    device const uint *wbase = (device const uint *)(w - sh);
    float acc[R][T];
    for (int r = 0; r < R; r++) for (int t = 0; t < T; t++) acc[r][t] = 0.0f;
    for (uint k0 = lane * 8; k0 < in; k0 += 256) {
        float4 xa[T], xb[T];
        float xs[T];
#pragma clang loop unroll(full)
        for (int t = 0; t < T; t++) {
            device const float4 *xp = (device const float4 *)(X + t * in + k0);
            xa[t] = xp[0];
            xb[t] = xp[1];
            const float4 u = xa[t] + xb[t];
            xs[t] = u.x + u.y + u.z + u.w;
        }
        const uint g = k0 / GROUP;
#pragma clang loop unroll(full)
        for (int r = 0; r < R; r++) {
            const uint row = min(row0 + r, out - 1);
            const uint wi = (row * row_bytes + k0 * BITS / 8) / 4;
            uint wd[WPL];
            if (sh == 0) {
                for (uint i = 0; i < WPL; i++) wd[i] = wbase[wi + i];
            } else {
                for (uint i = 0; i < WPL; i++) wd[i] = (wbase[wi + i] >> (8 * sh)) | (wbase[wi + i + 1] << (32 - 8 * sh));
            }
            float4 qa, qb;
            if (BITS == 4) {
                qa = float4(float(wd[0] & MASK), float((wd[0] >> 4) & MASK), float((wd[0] >> 8) & MASK), float((wd[0] >> 12) & MASK));
                qb = float4(float((wd[0] >> 16) & MASK), float((wd[0] >> 20) & MASK), float((wd[0] >> 24) & MASK), float(wd[0] >> 28));
            } else {
                qa = float4(float(wd[0] & MASK), float((wd[0] >> 8) & MASK), float((wd[0] >> 16) & MASK), float(wd[0] >> 24));
                qb = float4(float(wd[WPL - 1] & MASK), float((wd[WPL - 1] >> 8) & MASK), float((wd[WPL - 1] >> 16) & MASK), float(wd[WPL - 1] >> 24));
            }
            const float sc = bf16_at(scales, row * groups + g), bi = bf16_at(biases, row * groups + g);
#pragma clang loop unroll(full)
            for (int t = 0; t < T; t++) acc[r][t] += sc * (dot(qa, xa[t]) + dot(qb, xb[t])) + bi * xs[t];
        }
    }
#pragma clang loop unroll(full)
    for (int r = 0; r < R; r++)
#pragma clang loop unroll(full)
        for (int t = 0; t < T; t++) {
            const float v = simd_sum(acc[r][t]);
            if (lane == 0 && row0 + r < out) Y[t * out + row0 + r] = v;
        }
}

#define QMV_ROWS(NAME, BITS, GROUP, T, R)                                                                   \
kernel void NAME(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],             \
                 device const uchar *biases [[buffer(2)]], device const float *X [[buffer(3)]],             \
                 device float *Y [[buffer(4)]], constant uint3 &d [[buffer(5)]] /* in, out, T */,           \
                 constant uint *shift [[buffer(6)]],                                                         \
                 uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],      \
                 uint lane [[thread_index_in_simdgroup]]) {                                                  \
    const uint row0 = (tg * 2 + sg) * R;                                                                     \
    if (row0 < d.y) qmv_rows_mlx<BITS, GROUP, T, R>(w + shift[0], scales + shift[1], biases + shift[2], X, Y, d.x, d.y, row0, lane); \
}

QMV_ROWS(qmv_rows_4_32_1, 4, 32, 1, ROWS_R)
QMV_ROWS(qmv_rows_4_32_2, 4, 32, 2, ROWS_R)
QMV_ROWS(qmv_rows_4_32_3, 4, 32, 3, ROWS_R)
QMV_ROWS(qmv_rows_4_32_4, 4, 32, 4, ROWS_R)
QMV_ROWS(qmv_rows_4_32_5, 4, 32, 5, ROWS_R)
QMV_ROWS(qmv_rows_4_32_6, 4, 32, 6, ROWS_R)
QMV_ROWS(qmv_rows_4_32_7, 4, 32, 7, ROWS_R)
QMV_ROWS(qmv_rows_4_32_8, 4, 32, 8, ROWS_R)
QMV_ROWS(qmv_rows_8_64_1, 8, 64, 1, ROWS_R)
QMV_ROWS(qmv_rows_8_64_2, 8, 64, 2, ROWS_R)
QMV_ROWS(qmv_rows_8_64_3, 8, 64, 3, ROWS_R)
QMV_ROWS(qmv_rows_8_64_4, 8, 64, 4, ROWS_R)
QMV_ROWS(qmv_rows_8_64_5, 8, 64, 5, ROWS_R)
QMV_ROWS(qmv_rows_8_64_6, 8, 64, 6, ROWS_R)
QMV_ROWS(qmv_rows_8_64_7, 8, 64, 7, ROWS_R)
QMV_ROWS(qmv_rows_8_64_8, 8, 64, 8, ROWS_R)

// Hyper-connection up projection fused with the mix: threadgroup g computes the rows
// k*H + j for the 4 streams k and 8 positions j in [8g, 8g+8), input silu(low / HC) read
// on load, then a[j] = mean_k sigmoid(row(k, j)) * xn[k*H + j]. Same values and summation
// order as up matvec + hc_mix, without the mix buffer. 256 threads, 8 lanes per row.
kernel void qmv_up_mix(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]], device const uchar *biases [[buffer(2)]],
                       device const float *low [[buffer(3)]], device const float *xn [[buffer(4)]], device float *a [[buffer(5)]],
                       constant uint4 &p [[buffer(6)]] /* in, H, HC, - */, constant float &act_scale [[buffer(7)]],
                       constant uint *shift [[buffer(8)]],
                       uint2 gt [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float m[32];
    threadgroup float red[8];
    const uint g = gt.x, t = gt.y; // token t of a small chunk (grid y)
    low += t * p.x;
    xn += t * p.z * p.y;
    a += t * p.y;
    const uint rl = tid / 8, sub = tid % 8, k = rl / 8, j = g * 8 + rl % 8, row = k * p.y + j;
    float acc = qdot_l<8, true>(w + shift[0], scales + shift[1], biases + shift[2], low, p.x, row, sub, act_scale);
    acc = lanes_sum<8>(acc, tid, red);
    if (sub == 0) m[rl] = (1.0f / (1.0f + exp(-acc))) * xn[row];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) a[g * 8 + tid] = (((m[tid] + m[8 + tid]) + m[16 + tid]) + m[24 + tid]) / float(p.z);
}

// ---- decode MoE with the shared expert folded into the routed launches (after sushi:
// "shared expert folded into the routed launches", docs/qwen4-perf.md, MIT): slot < TOPK is
// routed expert idx[slot], slot == TOPK the shared expert (its own weights, same width: the
// shared intermediate size equals the routed one). One token (decode).
kernel void qmv_gate_up_sh(device const uchar *wg [[buffer(0)]], device const uchar *sg [[buffer(1)]], device const uchar *bg [[buffer(2)]],
                           device const uchar *wu [[buffer(3)]], device const uchar *su [[buffer(4)]], device const uchar *bu [[buffer(5)]],
                           device const float *x [[buffer(6)]], device float *h [[buffer(7)]], device const uint *idx [[buffer(8)]],
                           constant uint4 &d [[buffer(9)]] /* in, out, topk, - */, constant uint4 &st [[buffer(10)]] /* w, s/b bytes per expert */,
                           constant uint *shift [[buffer(11)]],
                           device const uchar *xg [[buffer(12)]], device const uchar *xsg [[buffer(13)]], device const uchar *xbg [[buffer(14)]],
                           device const uchar *xu [[buffer(15)]], device const uchar *xsu [[buffer(16)]], device const uchar *xbu [[buffer(17)]],
                           constant uint *shift2 [[buffer(18)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (tg.x >= d.y) return;
    float g, u;
    if (tg.y < d.z) {
        const uint e = idx[tg.y];
        g = qdot_l<32, false>(wg + shift[0] + ulong(e) * st.x, sg + shift[1] + ulong(e) * st.y, bg + shift[2] + ulong(e) * st.y, x, d.x, tg.x, lane, 1.0f);
        u = qdot_l<32, false>(wu + shift[3] + ulong(e) * st.x, su + shift[4] + ulong(e) * st.y, bu + shift[5] + ulong(e) * st.y, x, d.x, tg.x, lane, 1.0f);
    } else {
        g = qdot_l<32, false>(xg + shift2[0], xsg + shift2[1], xbg + shift2[2], x, d.x, tg.x, lane, 1.0f);
        u = qdot_l<32, false>(xu + shift2[3], xsu + shift2[4], xbu + shift2[5], x, d.x, tg.x, lane, 1.0f);
    }
    g = simd_sum(g);
    u = simd_sum(u);
    if (lane == 0) h[tg.y * d.y + tg.x] = g / (1.0f + exp(-g)) * u;
}

// Down projections of the TOPK routed slots and the shared expert (slot TOPK): row tg.x of
// slot tg.y from h + slot * in into y + slot * out.
kernel void qmv_down_sh(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]], device const uchar *biases [[buffer(2)]],
                        device const float *x [[buffer(3)]], device float *y [[buffer(4)]], constant uint4 &d [[buffer(5)]] /* in, out, topk, - */,
                        constant uint *shift [[buffer(6)]], device const uint *idx [[buffer(7)]], constant uint2 &st [[buffer(8)]],
                        device const uchar *xw [[buffer(9)]], device const uchar *xs [[buffer(10)]], device const uchar *xb [[buffer(11)]],
                        constant uint *shift2 [[buffer(12)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (tg.x >= d.y) return;
    if (tg.y < d.z) {
        const uint e = idx[tg.y];
        qmatvec_row<4, 32>(w + shift[0] + ulong(e) * st.x, scales + shift[1] + ulong(e) * st.y, biases + shift[2] + ulong(e) * st.y, x + tg.y * d.x, y + tg.y * d.y, d.x, tg.x, lane);
    } else {
        qmatvec_row<4, 32>(xw + shift2[0], xs + shift2[1], xb + shift2[2], x + tg.y * d.x, y + tg.y * d.y, d.x, tg.x, lane);
    }
}

// Router and shared-expert gate of one token in one dispatch: rows < n0 from the router into y0,
// row n0 the shared gate (both 8-bit group 64, same input) into y1[0].
kernel void qmv8_router_sg(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]], device const uchar *biases [[buffer(2)]],
                           device const float *x [[buffer(3)]], device float *y0 [[buffer(4)]], constant uint2 &d [[buffer(5)]] /* in, router rows */,
                           constant uint *shift [[buffer(6)]],
                           device const uchar *gw [[buffer(7)]], device const uchar *gs [[buffer(8)]], device const uchar *gb [[buffer(9)]],
                           device float *y1 [[buffer(10)]], constant uint *shift2 [[buffer(11)]],
                           uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (row < d.y) qmatvec_row<8, 64>(w + shift[0], scales + shift[1], biases + shift[2], x, y0, d.x, row, lane);
    else if (row == d.y) qmatvec_row<8, 64>(gw + shift2[0], gs + shift2[1], gb + shift2[2], x, y1, d.x, 0, lane);
}
