// Run-time quantization variants of the decode, verification-chunk and exact prefill kernels
// (qmatvec.metal, qmm.metal), for packs that mix bit widths (bits in {4, 5, 6, 8}, group in
// {32, 64, 128}). Compiled after qgen_common.metal (src/qgen.zig: qgen.src).
// Each kernel mirrors one templated kernel: same buffers at the same indices, same grid and
// threadgroup shape, same outputs; the quantization of each matrix is one more constant buffer
// at the end of the argument list (uint2 = bits, group; uint4 = two such pairs; MvGP for
// qmv_multi). The templated kernels stay the fast path for the 4-bit/32 pack.
// Work split: a lane takes whole 32-value chunks (BITS words) instead of whole groups, so every
// group size keeps the same lane load; scale and bias are those of the chunk's group.

// Dot product of row `row` with x over lanes sub, sub + L, ...; ACT: x -> silu(x * act_scale).
template <int BITS, bool ACT>
inline float qg_dot(device const uchar *w, device const uchar *scales, device const uchar *biases,
                    device const float *x, uint in, uint group, uint row, uint sub, uint L, float act_scale) {
    const uint chunks = in / 32, gs = ctz(group), groups = in >> gs;
    device const uchar *wr = w + row * (in * BITS / 8);
    float acc = 0.0f;
    for (uint c = sub; c < chunks; c += L) {
        float4 qv[8];
        qg_unpack32<BITS>(wr + c * (4 * BITS), qv);
        device const float4 *xg = (device const float4 *)(x + c * 32);
        float dsum = 0.0f, xsum = 0.0f;
#pragma clang loop unroll(full)
        for (int i = 0; i < 8; i++) {
            float4 xv = xg[i];
            if (ACT) xv = (xv * act_scale) / (1.0f + exp(-(xv * act_scale)));
            dsum += dot(qv[i], xv);
            xsum += xv.x + xv.y + xv.z + xv.w;
        }
        const uint g = row * groups + ((c * 32) >> gs);
        acc += qg_bf16(scales, g) * dsum + qg_bf16(biases, g) * xsum;
    }
    return acc;
}

inline float qg_dot_rt(uint bits, device const uchar *w, device const uchar *s, device const uchar *b,
                       device const float *x, uint in, uint group, uint row, uint sub, uint L) {
    float r = 0.0f;
    QG_BITS(bits, r = (qg_dot<B, false>(w, s, b, x, in, group, row, sub, L, 1.0f)), r = NAN)
    return r;
}

inline float qg_dot_act_rt(uint bits, bool act, device const uchar *w, device const uchar *s, device const uchar *b,
                           device const float *x, uint in, uint group, uint row, uint sub, uint L, float act_scale) {
    float r = 0.0f;
    if (act) {
        QG_BITS(bits, r = (qg_dot<B, true>(w, s, b, x, in, group, row, sub, L, act_scale)), r = NAN)
    } else {
        QG_BITS(bits, r = (qg_dot<B, false>(w, s, b, x, in, group, row, sub, L, act_scale)), r = NAN)
    }
    return r;
}

// One simdgroup per row (qmatvec_row): y[row] = W[row] . x.
inline void qg_row(uint bits, uint group, device const uchar *w, device const uchar *s, device const uchar *b,
                   device const float *x, device float *y, uint in, uint row, uint lane) {
    const float acc = simd_sum(qg_dot_rt(bits, w, s, b, x, in, group, row, lane, 32));
    if (lane == 0) y[row] = acc;
}

// qmatvec_4_32: grid out x 32 threads. q = (bits, group).
kernel void qmatvec_gen(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],
                        device const uchar *biases [[buffer(2)]], device const float *x [[buffer(3)]],
                        device float *y [[buffer(4)]], constant uint &in [[buffer(5)]], constant uint &out [[buffer(6)]],
                        constant uint *shift [[buffer(7)]], constant uint2 &q [[buffer(8)]],
                        uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (row < out) qg_row(q.x, q.y, w + shift[0], scales + shift[1], biases + shift[2], x, y, in, row, lane);
}

// qmatvec_expert_4_32: grid (out, slots) x 32 threads. q = (bits, group).
kernel void qmatvec_expert_gen(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],
                               device const uchar *biases [[buffer(2)]], device const float *x [[buffer(3)]],
                               device float *y [[buffer(4)]], constant uint &in [[buffer(5)]], constant uint &out [[buffer(6)]],
                               constant uint *shift [[buffer(7)]], device const uint *idx [[buffer(8)]],
                               constant uint4 &stride [[buffer(9)]], constant uint2 &q [[buffer(10)]],
                               uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (tg.x >= out) return;
    const uint e = idx[tg.y];
    qg_row(q.x, q.y, w + shift[0] + ulong(e) * stride.x, scales + shift[1] + ulong(e) * stride.y, biases + shift[2] + ulong(e) * stride.y,
           x + (tg.y / stride.w) * stride.z, y + tg.y * out, in, tg.x, lane);
}

// qmv_down_sh: grid (out, topk + 1) x 32. q = (routed bits, routed group, shared bits, shared group).
kernel void qmv_down_sh_gen(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]], device const uchar *biases [[buffer(2)]],
                            device const float *x [[buffer(3)]], device float *y [[buffer(4)]], constant uint4 &d [[buffer(5)]] /* in, out, topk, - */,
                            constant uint *shift [[buffer(6)]], device const uint *idx [[buffer(7)]], constant uint2 &st [[buffer(8)]],
                            device const uchar *xw [[buffer(9)]], device const uchar *xs [[buffer(10)]], device const uchar *xb [[buffer(11)]],
                            constant uint *shift2 [[buffer(12)]], constant uint4 &q [[buffer(13)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (tg.x >= d.y) return;
    if (tg.y < d.z) {
        const uint e = idx[tg.y];
        qg_row(q.x, q.y, w + shift[0] + ulong(e) * st.x, scales + shift[1] + ulong(e) * st.y, biases + shift[2] + ulong(e) * st.y, x + tg.y * d.x, y + tg.y * d.y, d.x, tg.x, lane);
    } else {
        qg_row(q.z, q.w, xw + shift2[0], xs + shift2[1], xb + shift2[2], x + tg.y * d.x, y + tg.y * d.y, d.x, tg.x, lane);
    }
}

// qmv8_router_sg: grid (router rows + 1) x 32. q = (router bits, group, shared-gate bits, group).
kernel void qmv_router_sg_gen(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]], device const uchar *biases [[buffer(2)]],
                              device const float *x [[buffer(3)]], device float *y0 [[buffer(4)]], constant uint2 &d [[buffer(5)]] /* in, router rows */,
                              constant uint *shift [[buffer(6)]],
                              device const uchar *gw [[buffer(7)]], device const uchar *gs [[buffer(8)]], device const uchar *gb [[buffer(9)]],
                              device float *y1 [[buffer(10)]], constant uint *shift2 [[buffer(11)]], constant uint4 &q [[buffer(12)]],
                              uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (row < d.y) qg_row(q.x, q.y, w + shift[0], scales + shift[1], biases + shift[2], x, y0, d.x, row, lane);
    else if (row == d.y) qg_row(q.z, q.w, gw + shift2[0], gs + shift2[1], gb + shift2[2], x, y1, d.x, 0, lane);
}

// ---- lanes-per-row family (qdot_l) -------------------------------------------------------

template <int L>
inline float qg_lanes_sum(float acc, uint tid, threadgroup float *red) {
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

// MvP of qmv_multi with the quantization of each matrix appended (bits[k], group[k]).
struct MvGP { uint in, nmat, act; float act_scale; uint rows[4]; uint tg0[5]; uint shift[12]; uint bits[4]; uint group[4]; };

template <int L>
inline void qg_multi(device const uchar *const *W, device float *const *Y, device const float *x, constant MvGP &p,
                     uint tg, uint t, uint tid, threadgroup float *red) {
    x += t * p.in;
    uint k = 0;
    while (k + 1 < p.nmat && tg >= p.tg0[k + 1]) k++;
    constexpr uint RPT = 256 / L;
    const uint row = (tg - p.tg0[k]) * RPT + tid / L, sub = tid % L;
    const bool valid = row < p.rows[k];
    float acc = 0.0f;
    if (valid) acc = qg_dot_act_rt(p.bits[k], p.act != 0, W[3 * k] + p.shift[3 * k], W[3 * k + 1] + p.shift[3 * k + 1], W[3 * k + 2] + p.shift[3 * k + 2],
                                   x, p.in, p.group[k], row, sub, L, p.act_scale);
    acc = qg_lanes_sum<L>(acc, tid, red);
    if (valid && sub == 0) Y[k][t * p.rows[k] + row] = acc;
}

// qmv_multi_l*: grid (tg0[nmat], T) x 256; buffer 17 is MvGP instead of MvP.
#define QG_MULTI(NAME, L)                                                                                   \
kernel void NAME(device const uchar *w0 [[buffer(0)]], device const uchar *s0 [[buffer(1)]], device const uchar *b0 [[buffer(2)]], \
                 device const uchar *w1 [[buffer(3)]], device const uchar *s1 [[buffer(4)]], device const uchar *b1 [[buffer(5)]], \
                 device const uchar *w2 [[buffer(6)]], device const uchar *s2 [[buffer(7)]], device const uchar *b2 [[buffer(8)]], \
                 device const uchar *w3 [[buffer(9)]], device const uchar *s3 [[buffer(10)]], device const uchar *b3 [[buffer(11)]], \
                 device const float *x [[buffer(12)]], device float *y0 [[buffer(13)]], device float *y1 [[buffer(14)]], \
                 device float *y2 [[buffer(15)]], device float *y3 [[buffer(16)]], constant MvGP &p [[buffer(17)]], \
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {      \
    threadgroup float red[8];                                                                                \
    device const uchar *W[12] = {w0, s0, b0, w1, s1, b1, w2, s2, b2, w3, s3, b3};                            \
    device float *Y[4] = {y0, y1, y2, y3};                                                                   \
    qg_multi<L>(W, Y, x, p, tg.x, tg.y, tid, red);                                                           \
}

QG_MULTI(qmv_multi_l8_gen, 8)
QG_MULTI(qmv_multi_l32_gen, 32)
QG_MULTI(qmv_multi_l256_gen, 256)

// qmv_gate_up: grid (out, slots) x 32. q = (bits, group) of gate and up.
kernel void qmv_gate_up_gen(device const uchar *wg [[buffer(0)]], device const uchar *sg [[buffer(1)]], device const uchar *bg [[buffer(2)]],
                            device const uchar *wu [[buffer(3)]], device const uchar *su [[buffer(4)]], device const uchar *bu [[buffer(5)]],
                            device const float *x [[buffer(6)]], device float *h [[buffer(7)]], device const uint *idx [[buffer(8)]],
                            constant uint4 &d [[buffer(9)]] /* in, out, xstride, div */, constant uint4 &st [[buffer(10)]] /* w, s/b bytes per expert, use idx, - */,
                            constant uint *shift [[buffer(11)]], constant uint2 &q [[buffer(12)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (tg.x >= d.y) return;
    const uint e = st.z ? idx[tg.y] : 0;
    device const float *xs = x + (tg.y / d.w) * d.z;
    float g = qg_dot_rt(q.x, wg + shift[0] + ulong(e) * st.x, sg + shift[1] + ulong(e) * st.y, bg + shift[2] + ulong(e) * st.y, xs, d.x, q.y, tg.x, lane, 32);
    float u = qg_dot_rt(q.x, wu + shift[3] + ulong(e) * st.x, su + shift[4] + ulong(e) * st.y, bu + shift[5] + ulong(e) * st.y, xs, d.x, q.y, tg.x, lane, 32);
    g = simd_sum(g);
    u = simd_sum(u);
    if (lane == 0) h[tg.y * d.y + tg.x] = g / (1.0f + exp(-g)) * u;
}

// qmv_gate_up_sh: grid (out, topk + 1) x 32. q = (routed bits, group, shared bits, group).
kernel void qmv_gate_up_sh_gen(device const uchar *wg [[buffer(0)]], device const uchar *sg [[buffer(1)]], device const uchar *bg [[buffer(2)]],
                               device const uchar *wu [[buffer(3)]], device const uchar *su [[buffer(4)]], device const uchar *bu [[buffer(5)]],
                               device const float *x [[buffer(6)]], device float *h [[buffer(7)]], device const uint *idx [[buffer(8)]],
                               constant uint4 &d [[buffer(9)]] /* in, out, topk, - */, constant uint4 &st [[buffer(10)]] /* w, s/b bytes per expert */,
                               constant uint *shift [[buffer(11)]],
                               device const uchar *xg [[buffer(12)]], device const uchar *xsg [[buffer(13)]], device const uchar *xbg [[buffer(14)]],
                               device const uchar *xu [[buffer(15)]], device const uchar *xsu [[buffer(16)]], device const uchar *xbu [[buffer(17)]],
                               constant uint *shift2 [[buffer(18)]], constant uint4 &q [[buffer(19)]],
                               uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (tg.x >= d.y) return;
    float g, u;
    if (tg.y < d.z) {
        const uint e = idx[tg.y];
        g = qg_dot_rt(q.x, wg + shift[0] + ulong(e) * st.x, sg + shift[1] + ulong(e) * st.y, bg + shift[2] + ulong(e) * st.y, x, d.x, q.y, tg.x, lane, 32);
        u = qg_dot_rt(q.x, wu + shift[3] + ulong(e) * st.x, su + shift[4] + ulong(e) * st.y, bu + shift[5] + ulong(e) * st.y, x, d.x, q.y, tg.x, lane, 32);
    } else {
        g = qg_dot_rt(q.z, xg + shift2[0], xsg + shift2[1], xbg + shift2[2], x, d.x, q.w, tg.x, lane, 32);
        u = qg_dot_rt(q.z, xu + shift2[3], xsu + shift2[4], xbu + shift2[5], x, d.x, q.w, tg.x, lane, 32);
    }
    g = simd_sum(g);
    u = simd_sum(u);
    if (lane == 0) h[tg.y * d.y + tg.x] = g / (1.0f + exp(-g)) * u;
}

// qmv_up_mix: grid (H / 8, T) x 256. q = (bits, group).
kernel void qmv_up_mix_gen(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]], device const uchar *biases [[buffer(2)]],
                           device const float *low [[buffer(3)]], device const float *xn [[buffer(4)]], device float *a [[buffer(5)]],
                           constant uint4 &p [[buffer(6)]] /* in, H, HC, - */, constant float &act_scale [[buffer(7)]],
                           constant uint *shift [[buffer(8)]], constant uint2 &q [[buffer(9)]],
                           uint2 gt [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float m[32];
    threadgroup float red[8];
    const uint g = gt.x, t = gt.y;
    low += t * p.x;
    xn += t * p.z * p.y;
    a += t * p.y;
    const uint rl = tid / 8, sub = tid % 8, k = rl / 8, j = g * 8 + rl % 8, row = k * p.y + j;
    float acc = qg_dot_act_rt(q.x, true, w + shift[0], scales + shift[1], biases + shift[2], low, p.x, q.y, row, sub, 8, act_scale);
    acc = qg_lanes_sum<8>(acc, tid, red);
    if (sub == 0) m[rl] = (1.0f / (1.0f + exp(-acc))) * xn[row];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) a[g * 8 + tid] = (((m[tid] + m[8 + tid]) + m[16 + tid]) + m[24 + tid]) / float(p.z);
}

// ---- verification chunks: up to 8 input rows ----------------------------------------------

// qmv_rows1 (one row per simdgroup): every chunk is unpacked once and applied to the T rows of X.
template <int BITS, int T>
inline void qg_rows_row(device const uchar *w, device const uchar *scales, device const uchar *biases,
                        device const float *X, device float *Y, uint in, uint out, uint group, uint row, uint lane) {
    const uint chunks = in / 32, gs = ctz(group), groups = in >> gs;
    device const uchar *wr = w + row * (in * BITS / 8);
    float acc[T];
    for (int t = 0; t < T; t++) acc[t] = 0.0f;
    for (uint c = lane; c < chunks; c += 32) {
        float4 qv[8];
        qg_unpack32<BITS>(wr + c * (4 * BITS), qv);
        const uint g = row * groups + ((c * 32) >> gs);
        const float s = qg_bf16(scales, g), b = qg_bf16(biases, g);
#pragma clang loop unroll(full)
        for (int t = 0; t < T; t++) {
            device const float4 *xg = (device const float4 *)(X + t * in + c * 32);
            float dsum = 0.0f, xsum = 0.0f;
#pragma clang loop unroll(full)
            for (int i = 0; i < 8; i++) {
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

// qmv_rows1_<b>_<g>_<T>: grid out x 32. q = (bits, group).
#define QG_ROWS1(NAME, T)                                                                                    \
kernel void NAME(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],             \
                 device const uchar *biases [[buffer(2)]], device const float *X [[buffer(3)]],             \
                 device float *Y [[buffer(4)]], constant uint3 &d [[buffer(5)]] /* in, out, T */,           \
                 constant uint *shift [[buffer(6)]], constant uint2 &q [[buffer(7)]],                        \
                 uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {       \
    if (row >= d.y) return;                                                                                  \
    QG_BITS(q.x, (qg_rows_row<B, T>(w + shift[0], scales + shift[1], biases + shift[2], X, Y, d.x, d.y, q.y, row, lane)), \
            if (lane == 0) for (int t = 0; t < T; t++) Y[t * d.y + row] = NAN)                               \
}

QG_ROWS1(qmv_rows1_gen_1, 1)
QG_ROWS1(qmv_rows1_gen_2, 2)
QG_ROWS1(qmv_rows1_gen_3, 3)
QG_ROWS1(qmv_rows1_gen_4, 4)
QG_ROWS1(qmv_rows1_gen_5, 5)
QG_ROWS1(qmv_rows1_gen_6, 6)
QG_ROWS1(qmv_rows1_gen_7, 7)
QG_ROWS1(qmv_rows1_gen_8, 8)

// qmv_rows (shared-input layout of qmv_rows_mlx): each lane owns 8 consecutive inputs for all T
// rows of X and applies them to R output rows; the 8 values of a row are BITS bytes at a byte
// offset that is word-aligned only for 4 and 8 bits (qg_unpack8 reads the window either way).
template <int BITS, int T, int R>
inline void qg_rows_mlx(device const uchar *w, device const uchar *scales, device const uchar *biases,
                        device const float *X, device float *Y, uint in, uint out, uint group, uint row0, uint lane) {
    const uint gs = ctz(group), groups = in >> gs;
    const uint row_bytes = in * BITS / 8;
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
        const uint g = k0 >> gs;
#pragma clang loop unroll(full)
        for (int r = 0; r < R; r++) {
            const uint row = min(row0 + r, out - 1);
            float4 qa, qb;
            qg_unpack8<BITS>(w + row * row_bytes + k0 * BITS / 8, qa, qb);
            const float sc = qg_bf16(scales, row * groups + g), bi = qg_bf16(biases, row * groups + g);
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

constant constexpr int QG_ROWS_R = 4; // = ROWS_R of qmatvec.metal

// qmv_rows_<b>_<g>_<T>: grid ceil(out / 8) x 64. q = (bits, group).
#define QG_ROWS(NAME, T)                                                                                     \
kernel void NAME(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],             \
                 device const uchar *biases [[buffer(2)]], device const float *X [[buffer(3)]],             \
                 device float *Y [[buffer(4)]], constant uint3 &d [[buffer(5)]] /* in, out, T */,           \
                 constant uint *shift [[buffer(6)]], constant uint2 &q [[buffer(7)]],                        \
                 uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],      \
                 uint lane [[thread_index_in_simdgroup]]) {                                                  \
    const uint row0 = (tg * 2 + sg) * QG_ROWS_R;                                                             \
    if (row0 >= d.y) return;                                                                                 \
    QG_BITS(q.x, (qg_rows_mlx<B, T, QG_ROWS_R>(w + shift[0], scales + shift[1], biases + shift[2], X, Y, d.x, d.y, q.y, row0, lane)), \
            if (lane == 0) for (int r = 0; r < QG_ROWS_R; r++) if (row0 + r < d.y) for (int t = 0; t < T; t++) Y[t * d.y + row0 + r] = NAN) \
}

QG_ROWS(qmv_rows_gen_1, 1)
QG_ROWS(qmv_rows_gen_2, 2)
QG_ROWS(qmv_rows_gen_3, 3)
QG_ROWS(qmv_rows_gen_4, 4)
QG_ROWS(qmv_rows_gen_5, 5)
QG_ROWS(qmv_rows_gen_6, 6)
QG_ROWS(qmv_rows_gen_7, 7)
QG_ROWS(qmv_rows_gen_8, 8)

// ---- exact prefill matmul (qmm.metal, f32 simdgroup matrices) -----------------------------

// 8 weights of row r from column kk (kk % 8 == 0, inside one group) into dst, dequantized.
inline void qg_fill8(uint bits, uint group, device const uchar *w, device const uchar *scales, device const uchar *biases,
                     uint in, uint r, uint kk, threadgroup float *dst) {
    const uint gs = ctz(group), groups = in >> gs, g = r * groups + (kk >> gs);
    const float s = qg_bf16(scales, g), b = qg_bf16(biases, g);
    float4 qa = float4(NAN), qb = float4(NAN);
    QG_BITS(bits, qg_unpack8<B>(w + r * (in * B / 8) + kk * B / 8, qa, qb), )
    qa = qa * s + b;
    qb = qb * s + b;
    for (uint v = 0; v < 4; v++) { dst[v] = qa[v]; dst[4 + v] = qb[v]; }
}

inline void qg_qmm_tile(uint bits, uint group, device const uchar *w, device const uchar *scales, device const uchar *biases,
                        device const float *X, device float *Y, uint in, uint out, uint T,
                        uint2 tg, uint sgid, uint tid,
                        threadgroup float (*xs)[33], threadgroup float (*ws)[33], threadgroup float (*ys)[33]) {
    constexpr uint BM = 32, BN = 32, BK = 32;
    const uint r0 = tg.x * BN, t0 = tg.y * BM;
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) acc[i][j] = simdgroup_float8x8(0.0f);
    const uint sm = sgid / 2, sn = sgid % 2;
    for (uint k0 = 0; k0 < in; k0 += BK) {
        for (uint e = tid; e < BM * BK; e += 128) {
            const uint t = e / BK, k = e % BK;
            xs[t][k] = (t0 + t < T) ? X[(t0 + t) * in + k0 + k] : 0.0f;
        }
        {
            const uint rr = tid / 4, part = tid % 4, r = r0 + rr;
            if (r < out) qg_fill8(bits, group, w, scales, biases, in, r, k0 + part * 8, &ws[rr][part * 8]);
            else for (uint v = 0; v < 8; v++) ws[rr][part * 8 + v] = 0.0f;
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

// qmm_4_32 / qmm_8_64: grid (ceil(out / 32), ceil(T / 32)) x 128. q = (bits, group).
kernel void qmm_gen(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],
                    device const uchar *biases [[buffer(2)]], device const float *X [[buffer(3)]],
                    device float *Y [[buffer(4)]], constant uint3 &dims [[buffer(5)]] /* in, out, T */,
                    constant uint *shift [[buffer(6)]], constant uint2 &q [[buffer(7)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
                    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float xs[32][33];
    threadgroup float ws[32][33];
    threadgroup float ys[32][33];
    qg_qmm_tile(q.x, q.y, w + shift[0], scales + shift[1], biases + shift[2], X, Y, dims.x, dims.y, dims.z, tg, sgid, tid, xs, ws, ys);
}

// qmm_grouped_4_32: grid (ceil(out / 32), tiles) x 128. q = (bits, group).
kernel void qmm_grouped_gen(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],
                            device const uchar *biases [[buffer(2)]], device const float *X [[buffer(3)]],
                            device float *Y [[buffer(4)]], constant uint3 &dims [[buffer(5)]] /* in, out, - */,
                            constant uint *shift [[buffer(6)]], device const uint4 *tiles [[buffer(7)]],
                            device const uint *ntiles [[buffer(8)]], constant uint2 &stride [[buffer(9)]],
                            constant uint2 &q [[buffer(10)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
                            uint tid [[thread_index_in_threadgroup]]) {
    if (tg.y >= ntiles[0]) return;
    threadgroup float xs[32][33];
    threadgroup float ws[32][33];
    threadgroup float ys[32][33];
    const uint4 t = tiles[tg.y];
    qg_qmm_tile(q.x, q.y, w + shift[0] + ulong(t.x) * stride.x, scales + shift[1] + ulong(t.x) * stride.y, biases + shift[2] + ulong(t.x) * stride.y,
                X + t.y * dims.x, Y + t.y * dims.y, dims.x, dims.y, t.z - t.y, uint2(tg.x, 0), sgid, tid, xs, ws, ys);
}
