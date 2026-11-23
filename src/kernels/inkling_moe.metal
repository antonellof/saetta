// Inkling 2 Mini routed experts, decode (W4A32 / W8A32: f32 activations against the weights
// dequantized exactly as the MLX pack defines them), compiled after inkling_ops.metal.
// gate_up rows alternate gate and up (the checkpoint's w13; transformers deinterleaves them,
// conversion_mapping.py:2665); the activation is the softcapped SwiGLU
// tanh(g / 4) 4 * sigmoid(g) * tanh(u / 25) 25 (modeling_inkling2_loop.py:751-771).
// NVFP4 (layers 0-19): two E2M1 codes per byte, even element in the low nibble, one E4M3 scale per
// 16 values, w = e2m1 * e4m3 * gfac[e] (gfac = global_scale / 2688, mx.dequantize's f32 result).
// MXFP8 (layers 20-23): E4M3 values, one E8M0 scale per 32, w = v * 2^(s - 127).
// Scalar references: src/inkling_ref.c (ink_ref_row_*, ink_ref_swiglu). Tensors sit at any byte
// alignment in the pack: 8-byte chunks are read as aligned words with a funnel shift.
// The decode kernels' work split (a gate row and its up row in the two halves of one simdgroup,
// 16 lanes per row; 8 lanes per down row) is adapted from Strata (MIT, Copyright (c) 2026 Niko1221
// and the Strata contributors; c6f6427, engine/src/inkling2/metal/ink_metal_codecs.metal
// ink_moe_gu_sf4 / ink_moe_dn_sf4), here on the MLX pack's layout.

constant float IK_E2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// E4M3 byte to float, exact: the 7 magnitude bits placed as a half (exponent bias 15 vs 7) times 2^8.
inline float ik_e4m3(uint b) {
    const float v = float(as_type<half>(ushort((b & 0x7Fu) << 7))) * 256.0f;
    return (b & 0x80u) ? -v : v;
}
// 2^(s - 127) for an E8M0 byte s in 1..254.
inline float ik_e8m0(uint s) { return as_type<float>(s << 23); }

// 8 bytes at any byte address, as two little-endian words: aligned words for the first ones, the
// bytes of the last partial word one by one (nothing read past p + 8, the end of a tensor).
inline uint2 ik_load8(device const uchar *p) {
    const ulong a = (ulong)p;
    const uint sh = uint(a & 3);
    device const uint *w = (device const uint *)(a - sh);
    if (sh == 0) return uint2(w[0], w[1]);
    uint w2 = 0;
    for (uint k = 0; k < sh; k++) w2 |= uint(p[8 - sh + k]) << (8 * k);
    const uint w0 = w[0], w1 = w[1];
    return uint2((w0 >> (8 * sh)) | (w1 << (32 - 8 * sh)), (w1 >> (8 * sh)) | (w2 << (32 - 8 * sh)));
}

inline float ik_dot_nv16(uint2 cw, device const float *x) {
    float s = 0.0f;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8; i++) {
        const uint b = ((i < 4 ? cw.x : cw.y) >> (8 * (i & 3))) & 0xFFu;
        s += IK_E2M1[b & 15u] * x[2 * i] + IK_E2M1[b >> 4] * x[2 * i + 1];
    }
    return s;
}
inline float ik_dot_mx8(uint2 cw, device const float *x) {
    float s = 0.0f;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8; i++) s += ik_e4m3(((i < 4 ? cw.x : cw.y) >> (8 * (i & 3))) & 0xFFu) * x[i];
    return s;
}

inline float ik_swiglu(float g, float u, float gcap, float ucap) {
    return tanh(g / gcap) * gcap * ik_sigm(g) * (tanh(u / ucap) * ucap);
}

// Byte strides per expert (weights, scales) and the tensors' starts within their bindings.
struct MoeP { uint H, I, K; uint sh_w, sh_s; ulong wsE, ssE; float gcap, ucap; };

// gate_up + activation: grid (I / 8, K, T) x 256; simdgroup sg takes pair j = 8 tg.x + sg, lanes
// 0-15 its gate row 2j, lanes 16-31 its up row 2j + 1. h [T][K][I].
kernel void ink_moe_gu_nv(device const uchar *W [[buffer(0)]], device const uchar *S [[buffer(1)]], device const float *gfac [[buffer(2)]],
                          device const float *x [[buffer(3)]], device const uint *idx [[buffer(4)]], device float *h [[buffer(5)]],
                          constant MoeP &p [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint t = tg.z, k = tg.y, j = tg.x * 8 + sg;
    const uint e = idx[t * p.K + k], row = 2 * j + (lane >> 4), l16 = lane & 15;
    device const uchar *wr = W + p.sh_w + ulong(e) * p.wsE + ulong(row) * (p.H / 2);
    device const uchar *sr = S + p.sh_s + ulong(e) * p.ssE + ulong(row) * (p.H / 16);
    device const float *xt = x + t * p.H;
    float acc = 0.0f;
    for (uint g = l16; g < p.H / 16; g += 16) acc += ik_e4m3(sr[g]) * ik_dot_nv16(ik_load8(wr + g * 8), xt + g * 16);
    acc += simd_shuffle_xor(acc, 8);
    acc += simd_shuffle_xor(acc, 4);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 1);
    acc *= gfac[e];
    const float gv = simd_shuffle(acc, 0), uv = simd_shuffle(acc, 16);
    if (lane == 0) h[(t * p.K + k) * p.I + j] = ik_swiglu(gv, uv, p.gcap, p.ucap);
}

// down, weighted by the router: grid (H / 32, K, T) x 256; 8 lanes per row, 4 rows per simdgroup.
// y [T][K][H] = (W_e h) * wts.
kernel void ink_moe_dn_nv(device const uchar *W [[buffer(0)]], device const uchar *S [[buffer(1)]], device const float *gfac [[buffer(2)]],
                          device const float *h [[buffer(3)]], device const uint *idx [[buffer(4)]], device const float *wts [[buffer(5)]],
                          device float *y [[buffer(6)]], constant MoeP &p [[buffer(7)]], uint3 tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint t = tg.z, k = tg.y, r = tg.x * 32 + sg * 4 + (lane >> 3), l8 = lane & 7;
    const uint e = idx[t * p.K + k];
    device const uchar *wr = W + p.sh_w + ulong(e) * p.wsE + ulong(r) * (p.I / 2);
    device const uchar *sr = S + p.sh_s + ulong(e) * p.ssE + ulong(r) * (p.I / 16);
    device const float *ht = h + (t * p.K + k) * p.I;
    float acc = 0.0f;
    for (uint g = l8; g < p.I / 16; g += 8) acc += ik_e4m3(sr[g]) * ik_dot_nv16(ik_load8(wr + g * 8), ht + g * 16);
    acc += simd_shuffle_xor(acc, 4);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 1);
    if (l8 == 0) y[(t * p.K + k) * p.H + r] = acc * gfac[e] * wts[t * p.K + k];
}

// MXFP8 forms: rows of H (gate_up) or I (down) bytes in 8-byte chunks, scale of chunk c at c / 4.
kernel void ink_moe_gu_mx(device const uchar *W [[buffer(0)]], device const uchar *S [[buffer(1)]], device const float *gfac [[buffer(2)]],
                          device const float *x [[buffer(3)]], device const uint *idx [[buffer(4)]], device float *h [[buffer(5)]],
                          constant MoeP &p [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint t = tg.z, k = tg.y, j = tg.x * 8 + sg;
    const uint e = idx[t * p.K + k], row = 2 * j + (lane >> 4), l16 = lane & 15;
    device const uchar *wr = W + p.sh_w + ulong(e) * p.wsE + ulong(row) * p.H;
    device const uchar *sr = S + p.sh_s + ulong(e) * p.ssE + ulong(row) * (p.H / 32);
    device const float *xt = x + t * p.H;
    float acc = 0.0f;
    for (uint c = l16; c < p.H / 8; c += 16) acc += ik_e8m0(sr[c / 4]) * ik_dot_mx8(ik_load8(wr + c * 8), xt + c * 8);
    acc += simd_shuffle_xor(acc, 8);
    acc += simd_shuffle_xor(acc, 4);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 1);
    const float gv = simd_shuffle(acc, 0), uv = simd_shuffle(acc, 16);
    if (lane == 0) h[(t * p.K + k) * p.I + j] = ik_swiglu(gv, uv, p.gcap, p.ucap);
}

kernel void ink_moe_dn_mx(device const uchar *W [[buffer(0)]], device const uchar *S [[buffer(1)]], device const float *gfac [[buffer(2)]],
                          device const float *h [[buffer(3)]], device const uint *idx [[buffer(4)]], device const float *wts [[buffer(5)]],
                          device float *y [[buffer(6)]], constant MoeP &p [[buffer(7)]], uint3 tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint t = tg.z, k = tg.y, r = tg.x * 32 + sg * 4 + (lane >> 3), l8 = lane & 7;
    const uint e = idx[t * p.K + k];
    device const uchar *wr = W + p.sh_w + ulong(e) * p.wsE + ulong(r) * p.I;
    device const uchar *sr = S + p.sh_s + ulong(e) * p.ssE + ulong(r) * (p.I / 32);
    device const float *ht = h + (t * p.K + k) * p.I;
    float acc = 0.0f;
    for (uint c = l8; c < p.I / 8; c += 8) acc += ik_e8m0(sr[c / 4]) * ik_dot_mx8(ik_load8(wr + c * 8), ht + c * 8);
    acc += simd_shuffle_xor(acc, 4);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 1);
    if (l8 == 0) y[(t * p.K + k) * p.H + r] = acc * wts[t * p.K + k];
}

// ---- chunks: experts grouped (prompt chunks and verification chunks, T > 1) -------------------
// ink_moe_plan sorts the chunk's (token, slot) pairs by expert (stable in token order) and cuts
// each expert's run into tiles of at most 32 pairs: tiles[i] = (expert, first pair, end pair).
// One threadgroup of E (<= 128) threads... any size >= E.
struct PlanP { uint E, K, T; };
kernel void ink_moe_plan(device const uint *idx [[buffer(0)]], device uint *pairs [[buffer(1)]], device uint4 *tiles [[buffer(2)]],
                         device uint *ntiles [[buffer(3)]], constant PlanP &p [[buffer(4)]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint cnt[128], off[129], toff[129];
    const uint n = p.T * p.K;
    if (tid < p.E) {
        uint c = 0;
        for (uint i = 0; i < n; i++) c += idx[i] == tid;
        cnt[tid] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        off[0] = 0;
        toff[0] = 0;
        for (uint e = 0; e < p.E; e++) {
            off[e + 1] = off[e] + cnt[e];
            toff[e + 1] = toff[e] + (cnt[e] + 31) / 32;
        }
        ntiles[0] = toff[p.E];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < p.E) {
        uint o = off[tid];
        for (uint i = 0; i < n; i++)
            if (idx[i] == tid) pairs[o++] = i;
        for (uint s = 0; s < (cnt[tid] + 31) / 32; s++)
            tiles[toff[tid] + s] = uint4(tid, off[tid] + 32 * s, min(off[tid] + 32 * (s + 1), off[tid] + cnt[tid]), 0);
    }
}

// One 32-wide K step of the weight tile [64 rows][32] (f32, scale applied, no gfac) from row n0.
template <bool MX>
inline void ik_wtile(device const uchar *W, device const uchar *S, ulong wrow_bytes, ulong srow_bytes, uint n0, uint k0, uint tid,
                     threadgroup float (*ws)[33]) {
    const uint r = tid / 2, half_ = tid % 2; // 64 rows x 2 halves of 16 values
    const uint kk = k0 + half_ * 16;
    device const uchar *wr = W + ulong(n0 + r) * wrow_bytes;
    device const uchar *sr = S + ulong(n0 + r) * srow_bytes;
    if (MX) {
        const float sc = ik_e8m0(sr[k0 / 32]);
        const uint2 a = ik_load8(wr + kk), b = ik_load8(wr + kk + 8);
        for (uint i = 0; i < 8; i++) {
            ws[r][half_ * 16 + i] = ik_e4m3(((i < 4 ? a.x : a.y) >> (8 * (i & 3))) & 0xFFu) * sc;
            ws[r][half_ * 16 + 8 + i] = ik_e4m3(((i < 4 ? b.x : b.y) >> (8 * (i & 3))) & 0xFFu) * sc;
        }
    } else {
        const float sc = ik_e4m3(sr[kk / 16]);
        const uint2 cw = ik_load8(wr + kk / 2);
        for (uint i = 0; i < 8; i++) {
            const uint bt = ((i < 4 ? cw.x : cw.y) >> (8 * (i & 3))) & 0xFFu;
            ws[r][half_ * 16 + 2 * i] = IK_E2M1[bt & 15u] * sc;
            ws[r][half_ * 16 + 2 * i + 1] = IK_E2M1[bt >> 4] * sc;
        }
    }
}

// C[32 pairs][64 rows] = X[pairs][Kd] . W[n0 .. n0 + 64][Kd]^T for the tile's pairs; X row of pair
// r: xrow(r). 4 simdgroups, each 32 x 16 (4 x 2 blocks of 8 x 8, f32). The result goes to cs.
template <bool MX>
inline void ik_tile_mm(device const uchar *W, device const uchar *S, ulong wrow_bytes, ulong srow_bytes, uint n0, uint Kd,
                       device const float *X, thread const uint *xoff, uint np, uint tid, uint sg,
                       threadgroup float (*xs)[33], threadgroup float (*ws)[33], threadgroup float (*cs)[65]) {
    simdgroup_float8x8 acc[4][2];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 2; j++) acc[i][j] = simdgroup_float8x8(0.0f);
    for (uint k0 = 0; k0 < Kd; k0 += 32) {
        // X tile: 32 rows x 32, 8 values per thread
        for (uint q = 0; q < 8; q++) {
            const uint e = tid * 8 + q, r = e / 32, c = e % 32;
            xs[r][c] = r < np ? X[xoff[r] + k0 + c] : 0.0f;
        }
        ik_wtile<MX>(W, S, wrow_bytes, srow_bytes, n0, k0, tid, ws);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_float8x8 a[4], b[2];
            for (uint i = 0; i < 4; i++) simdgroup_load(a[i], &xs[i * 8][kk], 33);
            for (uint j = 0; j < 2; j++) simdgroup_load(b[j], &ws[sg * 16 + j * 8][kk], 33, ulong2(0, 0), true);
            for (uint i = 0; i < 4; i++)
                for (uint j = 0; j < 2; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 2; j++) simdgroup_store(acc[i][j], &cs[i * 8][sg * 16 + j * 8], 65);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

struct MgP { uint H, I, K; uint sh_w, sh_s; ulong wsE, ssE; float gcap, ucap; };

// gate_up + activation over a tile: grid (2I / 64, max tiles) x 128. x [T][H]; the activations of
// sorted pair p go to h[p][I].
template <bool MX>
inline void ik_gu_g(device const uchar *W, device const uchar *S, device const float *gfac, device const float *x,
                    device const uint *pairs, device const uint4 *tiles, device const uint *ntiles, device float *h,
                    constant MgP &p, uint2 tg, uint sg, uint tid, threadgroup float (*xs)[33], threadgroup float (*ws)[33],
                    threadgroup float (*cs)[65]) {
    if (tg.y >= ntiles[0]) return;
    const uint4 tl = tiles[tg.y];
    const uint e = tl.x, p0 = tl.y, np = tl.z - tl.y, n0 = tg.x * 64;
    uint xoff[32];
    for (uint r = 0; r < 32; r++) xoff[r] = r < np ? (pairs[p0 + r] / p.K) * p.H : 0;
    const ulong wrow = MX ? p.H : p.H / 2, srow = MX ? p.H / 32 : p.H / 16;
    ik_tile_mm<MX>(W + p.sh_w + ulong(e) * p.wsE, S + p.sh_s + ulong(e) * p.ssE, wrow, srow, n0, p.H, x, xoff, np, tid, sg, xs, ws, cs);
    const float f = MX ? 1.0f : gfac[e];
    for (uint q = tid; q < 32 * 32; q += 128) {
        const uint r = q / 32, j = q % 32;
        if (r < np) h[(p0 + r) * p.I + tg.x * 32 + j] = ik_swiglu(cs[r][2 * j] * f, cs[r][2 * j + 1] * f, p.gcap, p.ucap);
    }
}
kernel void ink_moe_gu_nv_g(device const uchar *W [[buffer(0)]], device const uchar *S [[buffer(1)]], device const float *gfac [[buffer(2)]],
                            device const float *x [[buffer(3)]], device const uint *pairs [[buffer(4)]], device const uint4 *tiles [[buffer(5)]],
                            device const uint *ntiles [[buffer(6)]], device float *h [[buffer(7)]], constant MgP &p [[buffer(8)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float xs[32][33], ws[64][33], cs[32][65];
    ik_gu_g<false>(W, S, gfac, x, pairs, tiles, ntiles, h, p, tg, sg, tid, xs, ws, cs);
}
kernel void ink_moe_gu_mx_g(device const uchar *W [[buffer(0)]], device const uchar *S [[buffer(1)]], device const float *gfac [[buffer(2)]],
                            device const float *x [[buffer(3)]], device const uint *pairs [[buffer(4)]], device const uint4 *tiles [[buffer(5)]],
                            device const uint *ntiles [[buffer(6)]], device float *h [[buffer(7)]], constant MgP &p [[buffer(8)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float xs[32][33], ws[64][33], cs[32][65];
    ik_gu_g<true>(W, S, gfac, x, pairs, tiles, ntiles, h, p, tg, sg, tid, xs, ws, cs);
}

// down over a tile, weighted: grid (H / 64, max tiles) x 128. y [T][K][H] at the pair's (token,
// slot), times its router weight.
template <bool MX>
inline void ik_dn_g(device const uchar *W, device const uchar *S, device const float *gfac, device const float *h,
                    device const uint *pairs, device const uint4 *tiles, device const uint *ntiles, device const float *wts,
                    device float *y, constant MgP &p, uint2 tg, uint sg, uint tid, threadgroup float (*xs)[33],
                    threadgroup float (*ws)[33], threadgroup float (*cs)[65]) {
    if (tg.y >= ntiles[0]) return;
    const uint4 tl = tiles[tg.y];
    const uint e = tl.x, p0 = tl.y, np = tl.z - tl.y, n0 = tg.x * 64;
    uint xoff[32];
    for (uint r = 0; r < 32; r++) xoff[r] = (p0 + min(r, np - 1)) * p.I;
    const ulong wrow = MX ? p.I : p.I / 2, srow = MX ? p.I / 32 : p.I / 16;
    ik_tile_mm<MX>(W + p.sh_w + ulong(e) * p.wsE, S + p.sh_s + ulong(e) * p.ssE, wrow, srow, n0, p.I, h, xoff, np, tid, sg, xs, ws, cs);
    const float f = MX ? 1.0f : gfac[e];
    for (uint q = tid; q < 32 * 64; q += 128) {
        const uint r = q / 64, n = q % 64;
        if (r < np) {
            const uint tk = pairs[p0 + r];
            y[tk * p.H + n0 + n] = cs[r][n] * f * wts[tk];
        }
    }
}
kernel void ink_moe_dn_nv_g(device const uchar *W [[buffer(0)]], device const uchar *S [[buffer(1)]], device const float *gfac [[buffer(2)]],
                            device const float *h [[buffer(3)]], device const uint *pairs [[buffer(4)]], device const uint4 *tiles [[buffer(5)]],
                            device const uint *ntiles [[buffer(6)]], device const float *wts [[buffer(7)]], device float *y [[buffer(8)]],
                            constant MgP &p [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float xs[32][33], ws[64][33], cs[32][65];
    ik_dn_g<false>(W, S, gfac, h, pairs, tiles, ntiles, wts, y, p, tg, sg, tid, xs, ws, cs);
}
kernel void ink_moe_dn_mx_g(device const uchar *W [[buffer(0)]], device const uchar *S [[buffer(1)]], device const float *gfac [[buffer(2)]],
                            device const float *h [[buffer(3)]], device const uint *pairs [[buffer(4)]], device const uint4 *tiles [[buffer(5)]],
                            device const uint *ntiles [[buffer(6)]], device const float *wts [[buffer(7)]], device float *y [[buffer(8)]],
                            constant MgP &p [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float xs[32][33], ws[64][33], cs[32][65];
    ik_dn_g<true>(W, S, gfac, h, pairs, tiles, ntiles, wts, y, p, tg, sg, tid, xs, ws, cs);
}
