// Affine-quantized matrix times activations, MLX layout: Y[t][r] = sum_k X[t][k] * W[r][k],
// X is [T][in] f32, Y is [T][out] f32, W is [out][in] at BITS bits with GROUP-wide groups.
// Prefill: each weight tile is dequantized once per 32 tokens.
// f32 simdgroup matrices first, so the chunked path stays within the oracle tolerance;
// fp16 on the M5 matrix units comes later behind a quality gate.
// Tile: 32 tokens x 32 output rows per threadgroup of 4 simdgroups, K step 32.
// shift[0..2]: byte offsets of w, scales, biases from their 16-byte aligned bindings.
#include <metal_stdlib>
using namespace metal;

inline float bf16_ld(device const uchar *p, uint i) {
    const uint lo = p[2 * i], hi = p[2 * i + 1];
    return as_type<float>(((hi << 8) | lo) << 16);
}

template <int BITS, int GROUP>
inline void qmm_tile(device const uchar *w, device const uchar *scales, device const uchar *biases,
                     device const float *X, device float *Y, uint in, uint out, uint T,
                     uint2 tg, uint sgid, uint tid,
                     threadgroup float (*xs)[33], threadgroup float (*ws)[33], threadgroup float (*ys)[33]) {
    constexpr uint BM = 32, BN = 32, BK = 32;
    constexpr uint MASK = (1u << BITS) - 1u;
    const uint r0 = tg.x * BN, t0 = tg.y * BM;
    const uint groups = in / GROUP;
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) acc[i][j] = simdgroup_float8x8(0.0f);
    const uint sm = sgid / 2, sn = sgid % 2;
    for (uint k0 = 0; k0 < in; k0 += BK) {
        // activations: 32 x 32 floats, 8 per thread
        for (uint e = tid; e < BM * BK; e += 128) {
            const uint t = e / BK, k = e % BK;
            xs[t][k] = (t0 + t < T) ? X[(t0 + t) * in + k0 + k] : 0.0f;
        }
        // weights: row tid/4, 8 values from column (tid%4)*8
        {
            const uint rr = tid / 4, part = tid % 4, r = r0 + rr;
            const uint kk = k0 + part * 8;
            if (r < out) {
                const uint g = kk / GROUP;
                const float s = bf16_ld(scales, r * groups + g), b = bf16_ld(biases, r * groups + g);
                device const uchar *wb = w + r * (in * BITS / 8) + kk * BITS / 8;
                for (uint v = 0; v < 8; v++) {
                    const uint bit = v * BITS;
                    const uint q = (uint(wb[bit / 8]) >> (bit % 8)) & MASK;
                    ws[rr][part * 8 + v] = float(q) * s + b;
                }
            } else {
                for (uint v = 0; v < 8; v++) ws[rr][part * 8 + v] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < BK; k += 8) {
            simdgroup_float8x8 a[2], bm[2];
            for (uint i = 0; i < 2; i++) simdgroup_load(a[i], &xs[sm * 16 + i * 8][k], 33);
            for (uint j = 0; j < 2; j++) simdgroup_load(bm[j], &ws[sn * 16 + j * 8][k], 33, ulong2(0, 0), true);
            for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], bm[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) simdgroup_store(acc[i][j], &ys[sm * 16 + i * 8][sn * 16 + j * 8], 33);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < BM * BN; e += 128) {
        const uint t = e / BN, r = e % BN;
        if (t0 + t < T && r0 + r < out) Y[(t0 + t) * out + r0 + r] = ys[t][r];
    }
}

#define QMM(NAME, BITS, GROUP)                                                                              \
kernel void NAME(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],             \
                 device const uchar *biases [[buffer(2)]], device const float *X [[buffer(3)]],             \
                 device float *Y [[buffer(4)]], constant uint3 &dims [[buffer(5)]] /* in, out, T */,        \
                 constant uint *shift [[buffer(6)]],                                                         \
                 uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],   \
                 uint tid [[thread_index_in_threadgroup]]) {                                                \
    threadgroup float xs[32][33];                                                                            \
    threadgroup float ws[32][33];                                                                            \
    threadgroup float ys[32][33];                                                                            \
    qmm_tile<BITS, GROUP>(w + shift[0], scales + shift[1], biases + shift[2], X, Y, dims.x, dims.y, dims.z,  \
                          tg, sgid, tid, xs, ws, ys);                                                        \
}

QMM(qmm_4_32, 4, 32)
QMM(qmm_8_64, 8, 64)

// Routed experts in prefill: rows of X are token-slots grouped by expert (moe_group);
// tile i = (expert, first row, end row) covers at most 32 rows of one expert. Each
// expert's weights are read once per tile instead of once per token.
#define QMM_GROUPED(NAME, BITS, GROUP)                                                                      \
kernel void NAME(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],             \
                 device const uchar *biases [[buffer(2)]], device const float *X [[buffer(3)]],             \
                 device float *Y [[buffer(4)]], constant uint3 &dims [[buffer(5)]] /* in, out, - */,        \
                 constant uint *shift [[buffer(6)]], device const uint4 *tiles [[buffer(7)]],               \
                 device const uint *ntiles [[buffer(8)]], constant uint2 &stride [[buffer(9)]] /* w, s/b bytes per expert */, \
                 uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],   \
                 uint tid [[thread_index_in_threadgroup]]) {                                                \
    if (tg.y >= ntiles[0]) return;                                                                          \
    threadgroup float xs[32][33];                                                                            \
    threadgroup float ws[32][33];                                                                            \
    threadgroup float ys[32][33];                                                                            \
    const uint4 t = tiles[tg.y];                                                                             \
    qmm_tile<BITS, GROUP>(w + shift[0] + ulong(t.x) * stride.x, scales + shift[1] + ulong(t.x) * stride.y,                 \
                          biases + shift[2] + ulong(t.x) * stride.y, X + t.y * dims.x, Y + t.y * dims.y,            \
                          dims.x, dims.y, t.z - t.y, uint2(tg.x, 0), sgid, tid, xs, ws, ys);                 \
}

QMM_GROUPED(qmm_grouped_4_32, 4, 32)
