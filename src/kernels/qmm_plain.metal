// Prompt matmuls for GPUs without tensor ops (M1 to M4, or SAETTA_NO_ACCEL=1): MLX affine packs at
// run-time (bits, group) as in qgen.metal, Y[t][r] = sum_k X[t][k] * W[r][k] with X [T][in] f32 and
// Y [T][out] f32. Weights are dequantized in f32 (w = q * scale + bias, one rounding) and stored
// in threadgroup memory as S (half, or float for the float-fragment variant), activations are
// converted to S on load; simdgroup 8x8 products accumulate in f32.
// Concatenated after qgen_common.metal (qg_bf16, qg_val, QG_BITS); helper names carry a qp_ prefix.
// in % 32 == 0 (every group is 32, 64 or 128 wide). Ragged T and out: loads clamp to the last
// valid row or token (llama.cpp's trick, no branches in the loop) and stores are masked.
//
// Two tilings, both 4 simdgroups (128 threads) and a K step of 32:
// - L, adapted from llama.cpp kernel_mul_mm (ggml/src/ggml-metal/kernels/mul_mm.metal at 9197945,
//   MIT, Copyright (c) 2023-2026 The ggml authors): 64 rows x 32 tokens per threadgroup, the tiles
//   pre-blocked as 8x8 blocks in threadgroup memory (stride-8 loads), each simdgroup 32 rows x 16
//   tokens (8 accumulators), full tiles stored straight to device memory.
// - M, adapted from MLX qmm_t / QuantizedBlockLoader / steel BlockMMA (mlx/backend/metal/kernels/
//   quantized.h at 8f578df, MIT, Copyright (c) 2023 Apple Inc.): 32 rows x 32 tokens, row-major
//   tiles padded by 16 bytes, each simdgroup 16 x 16 (4 accumulators). The 4-bit and 8-bit
//   dequantization uses MLX's scale-absorbed masks (no shifts).
// DB: double buffering. The next K step's device reads are issued into registers before the
// current step's products, then stored into the other threadgroup buffer: one barrier per step.
#include <metal_stdlib>
using namespace metal;

// The 2*B bytes of 16 B-bit values from byte p (any alignment), as words with value 0 at bit 0.
// Only the aligned words holding those bytes are read.
#define QP_NW(B) ((2 * (B) + 3) / 4) // words of the shifted window
template <int B>
struct QpWin {
    uint wd[QP_NW(B)];
};

template <int B>
inline QpWin<B> qp_fetch16(device const uchar *p) {
    QpWin<B> r;
    const ulong addr = (ulong)p;
    const uint sh = uint(addr & 3);
    device const uint *q = (device const uint *)(addr - sh);
    constexpr int NW = QP_NW(B);
    const uint nread = (sh + 2 * B + 3) / 4; // NW or NW + 1
    uint raw[NW + 1];
#pragma clang loop unroll(full)
    for (int i = 0; i < NW; i++) raw[i] = q[i];
    raw[NW] = nread > uint(NW) ? q[NW] : 0u;
    if (sh == 0) {
#pragma clang loop unroll(full)
        for (int i = 0; i < NW; i++) r.wd[i] = raw[i];
    } else {
#pragma clang loop unroll(full)
        for (int i = 0; i < NW; i++) r.wd[i] = (raw[i] >> (8 * sh)) | (raw[i + 1] << (32 - 8 * sh));
    }
    return r;
}

// Value j of the window (j a constant after unrolling).
template <int B>
inline uint qp_val(thread const uint *wd, int j) {
    const uint bit = uint(j * B), wi = bit / 32, off = bit % 32;
    uint x = wd[wi] >> off;
    if (off + B > 32) x |= wd[wi + 1] << (32 - off);
    return x & ((1u << B) - 1u);
}

// The 16 dequantized values of a window: v[j] = q_j * s + b in f32.
template <int B>
inline void qp_deq16(thread const QpWin<B> &win, float s, float b, thread float *v) {
    if (B == 4) {
        // MLX: each 16-bit half holds 4 nibbles; mask in place and absorb the shift in the scale
        const float s1 = s / 16.0f, s2 = s / 256.0f, s3 = s / 4096.0f;
#pragma clang loop unroll(full)
        for (int i = 0; i < 4; i++) {
            const uint h = (win.wd[i / 2] >> (16 * (i % 2))) & 0xffffu;
            v[4 * i + 0] = float(h & 0x000fu) * s + b;
            v[4 * i + 1] = float(h & 0x00f0u) * s1 + b;
            v[4 * i + 2] = float(h & 0x0f00u) * s2 + b;
            v[4 * i + 3] = float(h & 0xf000u) * s3 + b;
        }
    } else if (B == 8) {
        const float s1 = s / 256.0f;
#pragma clang loop unroll(full)
        for (int i = 0; i < 8; i++) {
            const uint h = (win.wd[i / 2] >> (16 * (i % 2))) & 0xffffu;
            v[2 * i + 0] = float(h & 0x00ffu) * s + b;
            v[2 * i + 1] = float(h & 0xff00u) * s1 + b;
        }
    } else {
#pragma clang loop unroll(full)
        for (int j = 0; j < 16; j++) v[j] = float(qp_val<B>(win.wd, j)) * s + b;
    }
}

// What one thread brings from device memory for one K step: 16 weights of one row (window,
// scale, bias as raw bf16 bits) and 8 activations of one token.
template <int B>
struct QpRegs {
    QpWin<B> w;
    ushort s, b;
    float4 x0, x1;
};

inline float qp_bf16_bits(ushort u) { return as_type<float>(uint(u) << 16); }

// ---- tiling L (llama.cpp): 64 rows x 32 tokens -------------------------------------------------

constexpr constant int QP_L_SA = 64 * 32; // weight tile elements
constexpr constant int QP_L_SB = 32 * 32; // activation tile elements

template <int B, typename S, bool DB>
inline void qp_tile_l(device const uchar *w, device const uchar *scales, device const uchar *biases,
                      device const float *X, device float *Y, uint in, uint out, uint T, uint group,
                      uint2 tg, ushort sg, ushort tid, threadgroup char *shmem) {
    typedef simdgroup_matrix<S, 8, 8> S8;
    const uint r0 = tg.x * 64, t0 = tg.y * 32;
    const uint nr0 = min(64u, out - r0), nr1 = min(32u, T - t0);
    // weights: row lr0, 16 values from k = 16 * il; activations: token lr1, 8 values from 8 * part
    const ushort lr0 = ushort(min(uint(tid / 2), nr0 - 1)), il = tid % 2;
    const ushort lr1 = ushort(min(uint(tid / 4), nr1 - 1)), part = tid % 4;
    const uint rowb = in * B / 8, gs = ctz(group), groups = in >> gs;
    device const uchar *wrow = w + ulong(r0 + lr0) * rowb + il * (2 * B);
    const ulong g0 = ulong(r0 + lr0) * groups;
    device const float *xrow = X + ulong(t0 + lr1) * in + 8 * part;

    threadgroup S *sa0 = (threadgroup S *)shmem;
    threadgroup S *sb0 = sa0 + QP_L_SA;
    threadgroup S *sa1 = DB ? sb0 + QP_L_SB : sa0;
    threadgroup S *sb1 = DB ? sa1 + QP_L_SA : sb0;

    auto fetch = [&](uint k0, thread QpRegs<B> &r) {
        r.w = qp_fetch16<B>(wrow + k0 * B / 8);
        const ulong g = g0 + ((k0 + 16 * il) >> gs);
        r.s = ushort(scales[2 * g]) | ushort(ushort(scales[2 * g + 1]) << 8);
        r.b = ushort(biases[2 * g]) | ushort(ushort(biases[2 * g + 1]) << 8);
        r.x0 = float4(*(device const packed_float4 *)(xrow + k0)); // scratch offsets are 4-byte aligned only
        r.x1 = float4(*(device const packed_float4 *)(xrow + k0 + 4));
    };
    auto store = [&](thread const QpRegs<B> &r, threadgroup S *sa, threadgroup S *sb) {
        float v[16];
        qp_deq16<B>(r.w, qp_bf16_bits(r.s), qp_bf16_bits(r.b), v);
        // element (row, k) at 64 * (8 * (k / 8) + row / 8) + 8 * (k % 8) + row % 8
        const short sy = lr0 / 8, lx = lr0 % 8;
#pragma clang loop unroll(full)
        for (short i = 0; i < 16; i++) {
            const short sx = 2 * il + i / 8, ly = i % 8;
            sa[64 * (8 * sx + sy) + 8 * ly + lx] = S(v[i]);
        }
        // element (token, k) at 64 * (4 * (k / 8) + token / 8) + 8 * (token % 8) + k % 8
        threadgroup S *d = sb + 64 * (4 * part + lr1 / 8) + 8 * (lr1 % 8);
        *(threadgroup vec<S, 4> *)d = vec<S, 4>(r.x0);
        *(threadgroup vec<S, 4> *)(d + 4) = vec<S, 4>(r.x1);
    };

    simdgroup_float8x8 mc[8];
#pragma clang loop unroll(full)
    for (short i = 0; i < 8; i++) mc[i] = simdgroup_float8x8(0.0f);
    auto mma = [&](threadgroup const S *sa, threadgroup const S *sb) {
        threadgroup const S *lsma = sa + 4 * 64 * (sg % 2);
        threadgroup const S *lsmb = sb + 2 * 64 * (sg / 2);
#pragma clang loop unroll(full)
        for (short ik = 0; ik < 4; ik++) {
            S8 ma[4], mb[2];
#pragma clang loop unroll(full)
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
#pragma clang loop unroll(full)
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
#pragma clang loop unroll(full)
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    };

    QpRegs<B> r;
    if (DB) {
        fetch(0, r);
        store(r, sa0, sb0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint step = 0;
        for (uint k0 = 0; k0 < in; k0 += 32, step++) {
            const bool next = k0 + 32 < in;
            if (next) fetch(k0 + 32, r);
            mma((step & 1) ? sa1 : sa0, (step & 1) ? sb1 : sb0);
            if (next) store(r, (step & 1) ? sa0 : sa1, (step & 1) ? sb0 : sb1);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (uint k0 = 0; k0 < in; k0 += 32) {
            fetch(k0, r);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            store(r, sa0, sb0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            mma(sa0, sb0);
        }
    }

    if (nr0 == 64 && nr1 == 32) {
        device float *C = Y + ulong(t0 + 16 * (sg >> 1)) * out + r0 + 32 * (sg & 1);
#pragma clang loop unroll(full)
        for (short i = 0; i < 8; i++) simdgroup_store(mc[i], C + 8 * (i % 4) + ulong(8 * (i / 4)) * out, out, 0, false);
    } else {
        // ragged tile: through threadgroup memory [32 tokens][64 rows] (8 KB), masked stores
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float *tmp = (threadgroup float *)shmem;
        threadgroup float *ts = tmp + 32 * (sg & 1) + 16 * (sg >> 1) * 64;
#pragma clang loop unroll(full)
        for (short i = 0; i < 8; i++) simdgroup_store(mc[i], ts + 8 * (i % 4) + 8 * 64 * (i / 4), 64, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < 32 * 64; e += 128) {
            const uint t = e / 64, rr = e % 64;
            if (t < nr1 && rr < nr0) Y[ulong(t0 + t) * out + r0 + rr] = tmp[e];
        }
    }
}

// ---- tiling M (MLX): 32 rows x 32 tokens, padded row-major tiles ---------------------------------

constexpr constant int QP_M_LD = 40; // 32 + 16 bytes of half padding (MLX BK_padded)

template <int B, bool DB>
inline void qp_tile_m(device const uchar *w, device const uchar *scales, device const uchar *biases,
                      device const float *X, device float *Y, uint in, uint out, uint T, uint group,
                      uint2 tg, ushort sg, ushort tid, threadgroup char *shmem) {
    const uint r0 = tg.x * 32, t0 = tg.y * 32;
    const uint nr0 = min(32u, out - r0), nr1 = min(32u, T - t0);
    // weights: row lr0, 8 values from 8 * part (half of a 16-value window: B bytes)
    const ushort lr0 = ushort(min(uint(tid / 4), nr0 - 1)), part = tid % 4;
    const ushort lr1 = ushort(min(uint(tid / 4), nr1 - 1));
    const uint rowb = in * B / 8, gs = ctz(group), groups = in >> gs;
    device const uchar *wrow = w + ulong(r0 + lr0) * rowb + part * B;
    const ulong g0 = ulong(r0 + lr0) * groups;
    device const float *xrow = X + ulong(t0 + lr1) * in + 8 * part;
    threadgroup half *xs0 = (threadgroup half *)shmem;
    threadgroup half *ws0 = xs0 + 32 * QP_M_LD;
    threadgroup half *xs1 = DB ? ws0 + 32 * QP_M_LD : xs0;
    threadgroup half *ws1 = DB ? xs1 + 32 * QP_M_LD : ws0;

    struct Regs {
        uint wd[2];
        ushort s, b;
        float4 x0, x1;
    };
    auto fetch = [&](uint k0, thread Regs &r) {
        // 8 values = B bytes: the 16-value window helper reads 2B; read only what is needed
        device const uchar *p = wrow + k0 * B / 8;
        const ulong addr = (ulong)p;
        const uint sh = uint(addr & 3);
        device const uint *q = (device const uint *)(addr - sh);
        const uint nread = (sh + B + 3) / 4;
        const uint a0 = q[0], a1 = nread > 1 ? q[1] : 0u, a2 = nread > 2 ? q[2] : 0u;
        r.wd[0] = sh == 0 ? a0 : (a0 >> (8 * sh)) | (a1 << (32 - 8 * sh));
        r.wd[1] = sh == 0 ? a1 : (a1 >> (8 * sh)) | (a2 << (32 - 8 * sh));
        const ulong g = g0 + ((k0 + 8 * part) >> gs);
        r.s = ushort(scales[2 * g]) | ushort(ushort(scales[2 * g + 1]) << 8);
        r.b = ushort(biases[2 * g]) | ushort(ushort(biases[2 * g + 1]) << 8);
        r.x0 = float4(*(device const packed_float4 *)(xrow + k0)); // scratch offsets are 4-byte aligned only
        r.x1 = float4(*(device const packed_float4 *)(xrow + k0 + 4));
    };
    auto store = [&](thread const Regs &r, threadgroup half *xs, threadgroup half *ws) {
        const float s = qp_bf16_bits(r.s), b = qp_bf16_bits(r.b);
        float v[8];
        if (B == 4) {
            const float s1 = s / 16.0f, s2 = s / 256.0f, s3 = s / 4096.0f;
#pragma clang loop unroll(full)
            for (int i = 0; i < 2; i++) {
                const uint h = (r.wd[0] >> (16 * i)) & 0xffffu;
                v[4 * i + 0] = float(h & 0x000fu) * s + b;
                v[4 * i + 1] = float(h & 0x00f0u) * s1 + b;
                v[4 * i + 2] = float(h & 0x0f00u) * s2 + b;
                v[4 * i + 3] = float(h & 0xf000u) * s3 + b;
            }
        } else if (B == 8) {
            const float s1 = s / 256.0f;
#pragma clang loop unroll(full)
            for (int i = 0; i < 4; i++) {
                const uint h = (r.wd[i / 2] >> (16 * (i % 2))) & 0xffffu;
                v[2 * i + 0] = float(h & 0x00ffu) * s + b;
                v[2 * i + 1] = float(h & 0xff00u) * s1 + b;
            }
        } else {
#pragma clang loop unroll(full)
            for (int j = 0; j < 8; j++) v[j] = float(qp_val<B>(r.wd, j)) * s + b;
        }
        threadgroup half *wd = ws + lr0 * QP_M_LD + 8 * part;
        *(threadgroup half4 *)wd = half4(v[0], v[1], v[2], v[3]);
        *(threadgroup half4 *)(wd + 4) = half4(v[4], v[5], v[6], v[7]);
        threadgroup half *xd = xs + lr1 * QP_M_LD + 8 * part;
        *(threadgroup half4 *)xd = half4(r.x0);
        *(threadgroup half4 *)(xd + 4) = half4(r.x1);
    };
    const ushort sm = sg / 2, sn = sg % 2;
    simdgroup_float8x8 acc[2][2];
#pragma clang loop unroll(full)
    for (int i = 0; i < 2; i++)
#pragma clang loop unroll(full)
        for (int j = 0; j < 2; j++) acc[i][j] = simdgroup_float8x8(0.0f);
    auto mma = [&](threadgroup const half *xs, threadgroup const half *ws) {
#pragma clang loop unroll(full)
        for (int k = 0; k < 32; k += 8) {
            simdgroup_half8x8 a[2], bm[2];
#pragma clang loop unroll(full)
            for (int i = 0; i < 2; i++) simdgroup_load(a[i], xs + (sm * 16 + i * 8) * QP_M_LD + k, QP_M_LD);
#pragma clang loop unroll(full)
            for (int j = 0; j < 2; j++) simdgroup_load(bm[j], ws + (sn * 16 + j * 8) * QP_M_LD + k, QP_M_LD, ulong2(0, 0), true);
#pragma clang loop unroll(full)
            for (int i = 0; i < 2; i++)
#pragma clang loop unroll(full)
                for (int j = 0; j < 2; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], bm[j], acc[i][j]);
        }
    };

    Regs r;
    if (DB) {
        fetch(0, r);
        store(r, xs0, ws0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint step = 0;
        for (uint k0 = 0; k0 < in; k0 += 32, step++) {
            const bool next = k0 + 32 < in;
            if (next) fetch(k0 + 32, r);
            mma((step & 1) ? xs1 : xs0, (step & 1) ? ws1 : ws0);
            if (next) store(r, (step & 1) ? xs0 : xs1, (step & 1) ? ws0 : ws1);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (uint k0 = 0; k0 < in; k0 += 32) {
            fetch(k0, r);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            store(r, xs0, ws0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            mma(xs0, ws0);
        }
    }

    if (nr0 == 32 && nr1 == 32) {
        device float *C = Y + ulong(t0 + sm * 16) * out + r0 + sn * 16;
#pragma clang loop unroll(full)
        for (int i = 0; i < 2; i++)
#pragma clang loop unroll(full)
            for (int j = 0; j < 2; j++) simdgroup_store(acc[i][j], C + ulong(i * 8) * out + j * 8, out);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float *tmp = (threadgroup float *)shmem; // [32 tokens][32 rows]
#pragma clang loop unroll(full)
        for (int i = 0; i < 2; i++)
#pragma clang loop unroll(full)
            for (int j = 0; j < 2; j++) simdgroup_store(acc[i][j], tmp + (sm * 16 + i * 8) * 32 + sn * 16 + j * 8, 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < 32 * 32; e += 128) {
            const uint t = e / 32, rr = e % 32;
            if (t < nr1 && rr < nr0) Y[ulong(t0 + t) * out + r0 + rr] = tmp[e];
        }
    }
}

// ---- kernels -----------------------------------------------------------------------------------
// Dense: the buffers of qmm_gen (qgen.metal), grid (ceil(out / R), ceil(T / 32)) x 128 with R = 64
// (L) or 32 (M). Grouped: the buffers of qmm_grouped_gen, grid (ceil(out / R), tiles) x 128, tiles
// of at most 32 rows (moe_group's 32-row table). Unsupported bits write NaN.

// threadgroup bytes: L half 6 KB (12 DB), L float 12 KB (24 DB), M 5 KB (10 DB); at least the
// 8 KB (L) or 4 KB (M) of the ragged-tile store
#define QP_SHMEM_L(S, DB) (((DB) ? 2 : 1) * (QP_L_SA + QP_L_SB) * sizeof(S) > 8192 ? ((DB) ? 2 : 1) * (QP_L_SA + QP_L_SB) * sizeof(S) : 8192)
#define QP_SHMEM_M(DB) (((DB) ? 2 : 1) * 2 * 32 * QP_M_LD * 2 > 4096 ? ((DB) ? 2 : 1) * 2 * 32 * QP_M_LD * 2 : 4096)

#define QP_NAN_DENSE(R)                                                                                     \
    if (tid == 0)                                                                                           \
        for (uint t = tg.y * 32; t < min(d.z, tg.y * 32 + 32); t++)                                         \
            for (uint rr = tg.x * R; rr < min(d.y, tg.x * R + R); rr++) Y[ulong(t) * d.y + rr] = NAN

#define QP_DENSE(NAME, TILE, R, SHMEM)                                                                       \
kernel void NAME(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],              \
                 device const uchar *biases [[buffer(2)]], device const float *X [[buffer(3)]],              \
                 device float *Y [[buffer(4)]], constant uint3 &d [[buffer(5)]] /* in, out, T */,            \
                 constant uint *shift [[buffer(6)]], constant uint2 &q [[buffer(7)]],                         \
                 uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],    \
                 ushort tid [[thread_index_in_threadgroup]]) {                                               \
    threadgroup char shmem[SHMEM] __attribute__((aligned(16)));                                              \
    QG_BITS(q.x, (TILE(w + shift[0], scales + shift[1], biases + shift[2], X, Y, d.x, d.y, d.z, q.y, tg, sg, tid, shmem)), \
            QP_NAN_DENSE(R))                                                                                 \
}

#define QP_GROUPED(NAME, TILE, R, SHMEM)                                                                     \
kernel void NAME(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],              \
                 device const uchar *biases [[buffer(2)]], device const float *X [[buffer(3)]],              \
                 device float *Y [[buffer(4)]], constant uint3 &d [[buffer(5)]] /* in, out, - */,            \
                 constant uint *shift [[buffer(6)]], device const uint4 *tiles [[buffer(7)]],                \
                 device const uint *ntiles [[buffer(8)]], constant uint2 &stride [[buffer(9)]],               \
                 constant uint2 &q [[buffer(10)]],                                                            \
                 uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],    \
                 ushort tid [[thread_index_in_threadgroup]]) {                                               \
    if (tg.y >= ntiles[0]) return;                                                                           \
    threadgroup char shmem[SHMEM] __attribute__((aligned(16)));                                              \
    const uint4 t = tiles[tg.y];                                                                             \
    device const uchar *we = w + shift[0] + ulong(t.x) * stride.x;                                           \
    device const uchar *se = scales + shift[1] + ulong(t.x) * stride.y;                                      \
    device const uchar *be = biases + shift[2] + ulong(t.x) * stride.y;                                      \
    device const float *Xe = X + ulong(t.y) * d.x;                                                           \
    device float *Ye = Y + ulong(t.y) * d.y;                                                                 \
    const uint Te = t.z - t.y;                                                                               \
    QG_BITS(q.x, (TILE(we, se, be, Xe, Ye, d.x, d.y, Te, q.y, uint2(tg.x, 0), sg, tid, shmem)),               \
            if (tid == 0) for (uint i = 0; i < Te; i++) for (uint rr = tg.x * R; rr < min(d.y, tg.x * R + R); rr++) \
                Ye[ulong(i) * d.y + rr] = NAN)                                                               \
}

#define QP_L_H(...) qp_tile_l<B, half, false>(__VA_ARGS__)
#define QP_L_H_DB(...) qp_tile_l<B, half, true>(__VA_ARGS__)
#define QP_L_F(...) qp_tile_l<B, float, false>(__VA_ARGS__)
#define QP_L_F_DB(...) qp_tile_l<B, float, true>(__VA_ARGS__)
#define QP_M_H(...) qp_tile_m<B, false>(__VA_ARGS__)
#define QP_M_H_DB(...) qp_tile_m<B, true>(__VA_ARGS__)

QP_DENSE(qmmp_l_gen, QP_L_H, 64, QP_SHMEM_L(half, false))
QP_DENSE(qmmp_ldb_gen, QP_L_H_DB, 64, QP_SHMEM_L(half, true))
QP_DENSE(qmmp_lf_gen, QP_L_F, 64, QP_SHMEM_L(float, false))
QP_DENSE(qmmp_lfdb_gen, QP_L_F_DB, 64, QP_SHMEM_L(float, true))
QP_DENSE(qmmp_m_gen, QP_M_H, 32, QP_SHMEM_M(false))
QP_DENSE(qmmp_mdb_gen, QP_M_H_DB, 32, QP_SHMEM_M(true))

QP_GROUPED(qmmp_g_l_gen, QP_L_H, 64, QP_SHMEM_L(half, false))
QP_GROUPED(qmmp_g_ldb_gen, QP_L_H_DB, 64, QP_SHMEM_L(half, true))
QP_GROUPED(qmmp_g_lf_gen, QP_L_F, 64, QP_SHMEM_L(float, false))
QP_GROUPED(qmmp_g_lfdb_gen, QP_L_F_DB, 64, QP_SHMEM_L(float, true))
QP_GROUPED(qmmp_g_m_gen, QP_M_H, 32, QP_SHMEM_M(false))
QP_GROUPED(qmmp_g_mdb_gen, QP_M_H_DB, 32, QP_SHMEM_M(true))
