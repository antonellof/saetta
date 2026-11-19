// Per-token small ops of the Flash-Next forward pass, all f32.
// Each kernel is the direct GPU form of the CPU bring-up code it replaced; `zig build
// verify` checks the whole pass against a numpy reference. Naive first: no fusion beyond
// what saves a dispatch for free.
#include <metal_stdlib>
using namespace metal;

// Sum over a threadgroup of up to 1024 threads. Every thread gets the result.
inline float tg_sum(float v, threadgroup float *sh, uint tid, uint nt) {
    v = simd_sum(v);
    if (tid % 32 == 0) sh[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = (tid % 32 < (nt + 31) / 32) ? sh[tid % 32] : 0.0f;
    r = simd_sum(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}
inline float tg_max(float v, threadgroup float *sh, uint tid, uint nt) {
    v = simd_max(v);
    if (tid % 32 == 0) sh[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = (tid % 32 < (nt + 31) / 32) ? sh[tid % 32] : -INFINITY;
    r = simd_max(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}
inline float sigm(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float silu(float x) { return x / (1.0f + exp(-x)); }

struct NormP { uint n, xs, os, ws, plus1, wmod; float eps; };

// Group g: out[g*os + i] = x[g*xs + i] * rsqrt(mean(x^2) + eps) * (plus1 ? 1 + w : w)[g*ws + i].
// In place is fine (x == out).
kernel void rmsnorm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                    device float *out [[buffer(2)]], constant NormP &p [[buffer(3)]],
                    uint g [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                    uint nt [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    device const float *xg = x + g * p.xs;
    device float *og = out + g * p.os;
    device const float *wg = w + (p.wmod ? g % p.wmod : g) * p.ws;
    float ss = 0;
    for (uint i = tid; i < p.n; i += nt) { const float v = xg[i]; ss += v * v; }
    ss = tg_sum(ss, sh, tid, nt);
    const float r = 1.0f / sqrt(ss / float(p.n) + p.eps);
    for (uint i = tid; i < p.n; i += nt) { const float wv = wg[i]; og[i] = xg[i] * r * (p.plus1 ? 1.0f + wv : wv); }
}

// x = silu(x * scale)
kernel void silu_scale(device float *x [[buffer(0)]], constant uint &n [[buffer(1)]], constant float &scale [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < n) x[i] = silu(x[i] * scale);
}

// a[j] = mean_k sigmoid(mix[k*H + j]) * xn[k*H + j]
kernel void hc_mix(device const float *mix [[buffer(0)]], device const float *xn [[buffer(1)]], device float *a [[buffer(2)]],
                   constant uint3 &p [[buffer(3)]] /* H, HC, T */, uint i [[thread_position_in_grid]]) {
    if (i >= p.x * p.z) return;
    const uint t = i / p.x, j = i % p.x, row = t * p.x * p.y;
    float s = 0;
    for (uint k = 0; k < p.y; k++) s += sigm(mix[row + k * p.x + j]) * xn[row + k * p.x + j];
    a[i] = s / float(p.y);
}

// X[k*H + j] += 2 * sigmoid(r[k] / HC) * y[j]
kernel void hc_add(device float *X [[buffer(0)]], device const float *y [[buffer(1)]], device const float *r [[buffer(2)]],
                   constant uint3 &p [[buffer(3)]] /* H, HC, T; r is [T][HC] */, uint i [[thread_position_in_grid]]) {
    const uint w = p.x * p.y;
    if (i >= w * p.z) return;
    const uint t = i / w, rem = i % w, k = rem / p.x;
    X[i] += 2.0f * sigm(r[t * p.y + k] / float(p.y)) * y[t * p.x + rem % p.x];
}

// out = silu(depthwise causal conv over the last 3 inputs and this one); state shifts.
// state is [3][n], oldest first; w is [n][4].
kernel void gdn_conv(device const float *qkv [[buffer(0)]], device float *state [[buffer(1)]], device const float *w [[buffer(2)]],
                     device float *out [[buffer(3)]], constant uint4 &p [[buffer(4)]] /* n, T, save, slot */,
                     device float *hist [[buffer(5)]], device const float *src [[buffer(6)]], uint c [[thread_position_in_grid]]) {
    // src: the state to start from (state itself, or a speculation slot after a rollback; each
    // thread reads its own elements before any write, so src may alias hist)
    const uint n = p.x;
    if (c >= n) return;
    float s0 = src[c], s1 = src[n + c], s2 = src[2 * n + c];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (uint t = 0; t < p.y; t++) {
        const float x = qkv[t * n + c];
        out[t * n + c] = silu(x * w3 + s0 * w0 + s1 * w1 + s2 * w2);
        s0 = s1; s1 = s2; s2 = x;
        if (p.z) { device float *h = hist + t * p.w; h[c] = s0; h[n + c] = s1; h[2 * n + c] = s2; }
    }
    state[c] = s0; state[n + c] = s1; state[2 * n + c] = s2;
}

// L2 norm of the 16 q heads and 16 k heads (eps 1e-6 on the sum), q also / sqrt(dk).
kernel void gdn_qknorm(device float *conv [[buffer(0)]], constant uint4 &p [[buffer(1)]] /* nk, dk, key_dim, conv_dim */,
                       uint g0 [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                       uint nt [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    const uint t = g0 / (2 * p.x), g = g0 % (2 * p.x);
    const bool is_q = g < p.x;
    device float *v = conv + t * p.w + (is_q ? g * p.y : p.z + (g - p.x) * p.y);
    float ss = 0;
    for (uint i = tid; i < p.y; i += nt) ss += v[i] * v[i];
    ss = tg_sum(ss, sh, tid, nt);
    const float r = 1.0f / sqrt(ss + 1e-6f) * (is_q ? 1.0f / sqrt(float(p.y)) : 1.0f);
    for (uint i = tid; i < p.y; i += nt) v[i] *= r;
}

struct GdnP { uint nk, nv, dk, dv, key_dim, conv_dim, T; float eps; uint save, slot; };

// Gated delta rule, one threadgroup per value head, one thread per value column j,
// then the gated RMSNorm (plain weight, sigmoid(z) gate).
// Gated delta rule over T tokens in order. Threadgroup (column block cb, value head hv) owns
// CB = 16 state columns (state layout [hv][j][i], dk contiguous); each column belongs to 4
// adjacent lanes holding 32 rows each in registers for the whole chunk. Columns are
// independent, so heads split over 8 threadgroups (384 in all instead of 48: the recurrence
// was latency bound). Writes the output before the gated norm (gdn_gnorm). Threads: 64.
kernel void gdn_step(device const float *conv [[buffer(0)]], device const float *ga [[buffer(1)]], device const float *gb [[buffer(2)]],
                     device const float *a_log [[buffer(3)]], device const float *dt_bias [[buffer(4)]], device float *state [[buffer(5)]],
                     device float *out [[buffer(6)]], constant GdnP &p [[buffer(7)]], device float *hist [[buffer(8)]],
                     device const float *src [[buffer(9)]] /* start state, as gdn_conv */,
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint CB = 16;
    threadgroup float kk[128];
    threadgroup float qq[128];
    const uint hv = tg.y, j = tg.x * CB + tid / 4, qb = (tid % 4) * 32;  // column, first row of this lane
    const uint kh = hv / (p.nv / p.nk);
    const float alog = exp(a_log[hv]), dtb = dt_bias[hv];
    device float *S = state + (hv * p.dv + j) * p.dk + qb;
    const uint vd = p.nv * p.dv;
    float st[32];
    device const float *S0 = src + (hv * p.dv + j) * p.dk + qb;
    for (uint r = 0; r < 32; r++) st[r] = S0[r];
    for (uint t = 0; t < p.T; t++) {
        device const float *ct = conv + t * p.conv_dim;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < p.dk; i += 4 * CB) { qq[i] = ct[kh * p.dk + i]; kk[i] = ct[p.key_dim + kh * p.dk + i]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float beta = sigm(gb[t * p.nv + hv]);
        const float xa = ga[t * p.nv + hv] + dtb;
        const float decay = exp(-alog * (xa > 20.0f ? xa : log(1.0f + exp(xa))));
        float kv = 0;
        for (uint r = 0; r < 32; r++) { st[r] *= decay; kv += st[r] * kk[qb + r]; }
        kv += simd_shuffle_xor(kv, 1);
        kv += simd_shuffle_xor(kv, 2);
        const float delta = (ct[2 * p.key_dim + hv * p.dv + j] - kv) * beta;
        float o = 0;
        for (uint r = 0; r < 32; r++) { st[r] += kk[qb + r] * delta; o += st[r] * qq[qb + r]; }
        o += simd_shuffle_xor(o, 1);
        o += simd_shuffle_xor(o, 2);
        if (tid % 4 == 0) out[t * vd + hv * p.dv + j] = o;
        // speculative verification: the state after each token, slot t (p.slot floats apart)
        if (p.save) { device float *Hs = hist + t * p.slot + (hv * p.dv + j) * p.dk + qb; for (uint r = 0; r < 32; r++) Hs[r] = st[r]; }
    }
    for (uint r = 0; r < 32; r++) S[r] = st[r];
}

// Gated RMSNorm of the GDN output, in place: group g = (token, value head) of dv values,
// o * rsqrt(mean(o^2) + eps) * w * sigmoid(z). Threads: dv.
kernel void gdn_gnorm(device float *o [[buffer(0)]], device const float *z [[buffer(1)]], device const float *w [[buffer(2)]],
                      constant GdnP &p [[buffer(3)]], uint g [[threadgroup_position_in_grid]], uint j [[thread_position_in_threadgroup]],
                      uint nt [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    const float v = o[g * p.dv + j];
    const float ss = tg_sum(v * v, sh, j, nt);
    o[g * p.dv + j] = v / sqrt(ss / float(p.dv) + p.eps) * w[j] * sigm(z[g * p.dv + j]);
}

// rotate_half RoPE on the first 2*half dims of each head; cs holds (cos, sin) per pair,
// computed by the host in double precision for this position.
kernel void rope(device float *x [[buffer(0)]], constant uint3 &p [[buffer(1)]] /* head stride, half, heads per token */,
                 constant float2 *cs [[buffer(2)]] /* [T][half] */, uint h [[threadgroup_position_in_grid]], uint i [[thread_position_in_threadgroup]]) {
    if (i >= p.y) return;
    device float *v = x + h * p.x;
    const float2 c = cs[(h / p.z) * p.y + i];
    const float a = v[i], b = v[i + p.y];
    v[i] = a * c.x - b * c.y;
    v[i + p.y] = b * c.x + a * c.y;
}

// dst[r * ds + i] = src[r * ss + i] for i < n
kernel void copy_rows(device const float *src [[buffer(0)]], device float *dst [[buffer(1)]], constant uint4 &p [[buffer(2)]] /* n, rows, ss, ds */,
                      uint e [[thread_position_in_grid]]) {
    if (e >= p.x * p.y) return;
    const uint r = e / p.x, i = e % p.x;
    dst[r * p.w + i] = src[r * p.z + i];
}

kernel void copy_f32(device const float *src [[buffer(0)]], device float *dst [[buffer(1)]], constant uint &n [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = src[i];
}

struct AttnP { uint nsel, dense, head_dim, kv_stride, q_per_kv, gate_stride; float scale; };

// One threadgroup per query head over the selected tokens (at most budget + ratio - 1).
// dense: tokens 0..nsel-1; else the token list `sel` of length cnt[0] (GPU selection).
// Output gated by sigmoid(gate).
kernel void attn_decode(device const float *q [[buffer(0)]], device const float *kc [[buffer(1)]], device const float *vc [[buffer(2)]],
                        device const uint *sel [[buffer(3)]], device const float *gate [[buffer(4)]], device float *out [[buffer(5)]],
                        constant AttnP &p [[buffer(6)]], device const uint *cnt [[buffer(7)]],
                        uint h [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    threadgroup float pr[2560];
    const uint kvh = h / p.q_per_kv;
    const uint nsel = p.dense ? p.nsel : cnt[0];
    // scores: one simdgroup per token, 8 dims per lane (head_dim 256), coalesced key reads
    device const float4 *qh = (device const float4 *)(q + h * p.head_dim);
    const uint lane = tid % 32, sg = tid / 32, nsg = nt / 32;
    const float4 q0 = qh[lane * 2], q1 = qh[lane * 2 + 1];
    for (uint i = sg; i < nsel; i += nsg) {
        const uint t = p.dense ? i : sel[i];
        device const float4 *kt = (device const float4 *)(kc + t * p.kv_stride + kvh * p.head_dim);
        const float d = simd_sum(dot(q0, kt[lane * 2]) + dot(q1, kt[lane * 2 + 1]));
        if (lane == 0) pr[i] = d * p.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -INFINITY;
    for (uint i = tid; i < nsel; i += nt) mx = max(mx, pr[i]);
    mx = tg_max(mx, sh, tid, nt);
    float sum = 0;
    for (uint i = tid; i < nsel; i += nt) { pr[i] = exp(pr[i] - mx); sum += pr[i]; }
    sum = tg_sum(sum, sh, tid, nt);
    for (uint e = tid; e < p.head_dim; e += nt) {
        float acc = 0;
        for (uint i = 0; i < nsel; i++) {
            const uint t = p.dense ? i : sel[i];
            acc += pr[i] * vc[t * p.kv_stride + kvh * p.head_dim + e];
        }
        out[h * p.head_dim + e] = acc / sum * sigm(gate[h * p.gate_stride + e]);
    }
}

inline uint tg_min_u(uint v, threadgroup uint *sh, uint tid, uint nt) {
    v = simd_min(v);
    if (tid % 32 == 0) sh[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r = (tid % 32 < (nt + 31) / 32) ? sh[tid % 32] : 0xffffffffu;
    r = simd_min(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

// Softmax over the router logits, top-k by probability (lowest index on ties),
// renormalized. One thread per expert (threads = experts, at most 1024).
kernel void moe_topk(device const float *logits0 [[buffer(0)]], device uint *idx0 [[buffer(1)]], device float *wts0 [[buffer(2)]],
                     constant uint2 &p [[buffer(3)]] /* experts, k */, uint t [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    threadgroup uint shi[32];
    device const float *logits = logits0 + t * p.x;
    device uint *idx = idx0 + t * p.y;
    device float *wts = wts0 + t * p.y;
    const float l = tid < p.x ? logits[tid] : -INFINITY;
    const float mx = tg_max(l, sh, tid, nt);
    const float ex = tid < p.x ? exp(l - mx) : 0.0f;
    const float v = ex / tg_sum(ex, sh, tid, nt);
    bool taken = tid >= p.x;
    float wsum = 0;
    for (uint r = 0; r < p.y; r++) {
        const float cand = taken ? -1.0f : v;
        const float best = tg_max(cand, sh, tid, nt);
        const uint bi = tg_min_u(cand == best ? tid : 0xffffffffu, shi, tid, nt);
        if (tid == bi) taken = true;
        if (tid == 0) { idx[r] = bi; wts[r] = best; }
        wsum += best;
    }
    if (tid == 0) for (uint r = 0; r < p.y; r++) wts[r] /= wsum;
}

// h = silu(g) * u
kernel void swiglu(device const float *g [[buffer(0)]], device const float *u [[buffer(1)]], device float *h [[buffer(2)]],
                   constant uint &n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < n) h[i] = silu(g[i]) * u[i];
}

// y[j] = sum_s wts[s] * yd[s*H + j] + sigmoid(sg) * sh[j]
kernel void moe_combine(device const float *yd [[buffer(0)]], device const float *wts [[buffer(1)]], device const float *shared_y [[buffer(2)]],
                        device const float *sg [[buffer(3)]], device float *y [[buffer(4)]], constant uint3 &p [[buffer(5)]] /* H, k, T */,
                        uint i [[thread_position_in_grid]]) {
    if (i >= p.x * p.z) return;
    const uint t = i / p.x, j = i % p.x;
    float acc = 0;
    for (uint s = 0; s < p.y; s++) acc += wts[t * p.y + s] * yd[(t * p.y + s) * p.x + j];
    y[i] = acc + sigm(sg[t]) * shared_y[i];
}

// PLE gate per stream k: g = <kn_k, qn_k> / sqrt(H), signed sqrt, gv_k = sigmoid(g) * val.
kernel void ple_gate(device const float *kn [[buffer(0)]], device const float *qn [[buffer(1)]], device const float *val0 [[buffer(2)]],
                     device float *gv [[buffer(3)]], constant uint2 &hp [[buffer(4)]] /* H, HC */,
                     uint k [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    const uint H = hp.x;
    device const float *val = val0 + (k / hp.y) * H;  // k runs over T * HC streams
    float d = 0;
    for (uint j = tid; j < H; j += nt) d += kn[k * H + j] * qn[k * H + j];
    d = tg_sum(d, sh, tid, nt);
    float g = d / sqrt(float(H));
    g = sign(g) * sqrt(max(abs(g), 1e-6f));
    const float s = sigm(g);
    for (uint j = tid; j < H; j += nt) gv[k * H + j] = s * val[j];
}

// X += gv + silu(dilated causal conv over taps t-9, t-6, t-3, t of gvn); history shifts.
// hist is [9][n], oldest first; w is [n][4].
kernel void ple_conv(device const float *gv [[buffer(0)]], device const float *gvn [[buffer(1)]], device float *hist [[buffer(2)]],
                     device const float *w [[buffer(3)]], device float *X [[buffer(4)]], constant uint4 &p [[buffer(5)]] /* n, T, save, - */,
                     device float *hsave [[buffer(6)]], device const float *src [[buffer(7)]] /* start, as gdn_conv */,
                     uint c [[thread_position_in_grid]]) {
    const uint n = p.x;
    if (c >= n) return;
    float h[9];
    for (uint t = 0; t < 9; t++) h[t] = src[t * n + c];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (uint t = 0; t < p.y; t++) {
        const float x = gvn[t * n + c];
        X[t * n + c] += gv[t * n + c] + silu(x * w3 + h[0] * w0 + h[3] * w1 + h[6] * w2);
        for (uint u = 0; u < 8; u++) h[u] = h[u + 1];
        h[8] = x;
        if (p.z) for (uint u = 0; u < 9; u++) hsave[(t * 9 + u) * n + c] = h[u];
    }
    for (uint t = 0; t < 9; t++) hist[t * n + c] = h[t];
}

// ---- QSA indexer on the GPU (M:690-773) ------------------------------------------------

// Pooled key of blocks b0 .. b0+nb-1: mean of the block's raw indexer keys, RMSNorm (1+w),
// RoPE at the block's first position (table rows [pos][half]). One threadgroup per block,
// dim threads.
kernel void ix_block(device const float *rawk [[buffer(0)]], device const float *w [[buffer(1)]], constant float2 *tab [[buffer(2)]],
                     device float *bk [[buffer(3)]], constant uint4 &p [[buffer(4)]] /* b0, dim, ratio, half */,
                     constant float &eps [[buffer(5)]],
                     uint g [[threadgroup_position_in_grid]], uint d [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    threadgroup float v[256];
    const uint b = p.x + g, dim = p.y, cr = p.z, hf = p.w;
    float m = 0;
    for (uint i = 0; i < cr; i++) m += rawk[(b * cr + i) * dim + d];
    m /= float(cr);
    const float ss = tg_sum(m * m, sh, d, nt);
    v[d] = m / sqrt(ss / float(dim) + eps) * (1.0f + w[d]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o = v[d];
    if (d < 2 * hf) {
        const uint i = d % hf;
        const float2 c = tab[(b * cr) * hf + i];
        o = d < hf ? v[i] * c.x - v[i + hf] * c.y : v[i + hf] * c.x + v[i] * c.y;
    }
    bk[b * dim + d] = o;
}

// scores[b] = sum_h relu(<q_h, bk_b>) / sqrt(dim), for b < nb. One thread per block.
kernel void ix_score(device const float *q [[buffer(0)]], device const float *bk [[buffer(1)]], device float *scores [[buffer(2)]],
                     constant uint3 &p [[buffer(3)]] /* nb, heads, dim */, uint b [[thread_position_in_grid]]) {
    if (b >= p.x) return;
    device const float *k = bk + b * p.z;
    float sc = 0;
    for (uint h = 0; h < p.y; h++) {
        float d = 0;
        for (uint e = 0; e < p.z; e++) d += q[h * p.z + e] * k[e];
        sc += max(d, 0.0f);
    }
    scores[b] = sc / sqrt(float(p.z));
}

inline uint ordered(float f) { const uint u = as_type<uint>(f); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }

inline uint tg_sum_u(uint v, threadgroup uint *sh, uint tid, uint nt) {
    v = simd_sum(v);
    if (tid % 32 == 0) sh[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r = (tid % 32 < (nt + 31) / 32) ? sh[tid % 32] : 0u;
    r = simd_sum(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

// Exclusive prefix sum over the threadgroup (1024 threads at most).
inline uint tg_scan_u(uint v, threadgroup uint *sh, uint tid, uint nt) {
    const uint incl = simd_prefix_inclusive_sum(v);
    if (tid % 32 == 31) sh[tid / 32] = incl;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        const uint x = tid < (nt + 31) / 32 ? sh[tid] : 0u;
        sh[tid] = simd_prefix_exclusive_sum(x);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint r = sh[tid / 32] + incl - v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

// Top-k blocks by score (lowest index on ties), expanded to their tokens, then the
// tail tokens past the last complete block. Writes the token list and its length.
// One threadgroup; each thread owns a contiguous run of blocks.
inline void ix_select_row(device const float *scores, device uint *sel, device uint *cnt, uint nb, uint K, uint cr, uint n,
                          uint tid, uint nt, threadgroup uint *sh) {
    const uint per = (nb + nt - 1) / nt, lo = min(nb, tid * per), hi = min(nb, lo + per);
    uint thr = 0;
    for (int bit = 31; bit >= 0; bit--) {
        const uint cand = thr | (1u << bit);
        uint c = 0;
        for (uint b = lo; b < hi; b++) c += ordered(scores[b]) >= cand;
        if (tg_sum_u(c, sh, tid, nt) >= K) thr = cand;
    }
    uint gt = 0, eq = 0;
    for (uint b = lo; b < hi; b++) { const uint u = ordered(scores[b]); gt += u > thr; eq += u == thr; }
    const uint need_eq = K - tg_sum_u(gt, sh, tid, nt);
    uint eq_before = tg_scan_u(eq, sh, tid, nt);
    uint mine = 0;
    {
        uint r = eq_before;
        for (uint b = lo; b < hi; b++) { const uint u = ordered(scores[b]); if (u > thr || (u == thr && r++ < need_eq)) mine++; }
    }
    uint slot = tg_scan_u(mine, sh, tid, nt) * cr;
    uint r = eq_before;
    for (uint b = lo; b < hi; b++) {
        const uint u = ordered(scores[b]);
        if (u > thr || (u == thr && r++ < need_eq)) for (uint i = 0; i < cr; i++) sel[slot++] = b * cr + i;
    }
    if (tid == 0) {
        uint s = K * cr;
        for (uint t = nb * cr; t < n; t++) sel[s++] = t;
        cnt[0] = s;
    }
}

kernel void ix_select(device const float *scores [[buffer(0)]], device uint *sel [[buffer(1)]], device uint *cnt [[buffer(2)]],
                      constant uint4 &p [[buffer(3)]] /* nb, k, ratio, n tokens */,
                      uint tid [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
    threadgroup uint sh[32];
    ix_select_row(scores, sel, cnt, p.x, p.y, p.z, p.w, tid, nt, sh);
}

// One threadgroup per token t of a prefill chunk at positions pos0..: its scores row
// (stride), its list sel + t*maxsel and count cnt[t]; tokens within the budget are dense
// and skipped.
kernel void ix_select_t(device const float *scores [[buffer(0)]], device uint *sel [[buffer(1)]], device uint *cnt [[buffer(2)]],
                        constant uint4 &p [[buffer(3)]] /* pos0, k, ratio, row stride */, constant uint &maxsel [[buffer(4)]],
                        uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
    threadgroup uint sh[32];
    const uint n = p.x + t + 1, nb = n / p.z;
    if (nb <= p.y) return;
    ix_select_row(scores + t * p.w, sel + t * maxsel, cnt + t, nb, p.y, p.z, n, tid, nt, sh);
}

// ---- MoE grouping for prefill ----------------------------------------------------------

// Groups the n = T * k token-slots by expert. Writes tok[r] (the token of grouped row r),
// inv[slot] (its grouped row), the tile table (expert, first row, end row) of at most tr
// rows each (p.w), and the tile count. One threadgroup of 1024 threads, experts <= 1024.
kernel void moe_group(device const uint *topi [[buffer(0)]], device uint *tok [[buffer(1)]], device uint *inv [[buffer(2)]],
                      device uint4 *tiles [[buffer(3)]], device uint *ntiles [[buffer(4)]],
                      constant uint4 &p [[buffer(5)]] /* n slots, k, experts, rows per tile */,
                      uint tid [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
    threadgroup atomic_uint cur[1024];
    threadgroup uint cnt[1024];
    threadgroup uint sh[32];
    const uint n = p.x, k = p.y, E = p.z, tr = p.w;
    if (tid < E) atomic_store_explicit(&cur[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < n; i += nt) atomic_fetch_add_explicit(&cur[topi[i]], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint c = tid < E ? atomic_load_explicit(&cur[tid], memory_order_relaxed) : 0u;
    const uint off = tg_scan_u(c, sh, tid, nt);
    const uint nti = (c + tr - 1) / tr;
    const uint toff = tg_scan_u(nti, sh, tid, nt);
    const uint total = tg_sum_u(nti, sh, tid, nt);
    if (tid < E) {
        cnt[tid] = off;
        atomic_store_explicit(&cur[tid], 0u, memory_order_relaxed);
        for (uint j = 0; j < nti; j++) tiles[toff + j] = uint4(tid, off + j * tr, off + min(c, (j + 1) * tr), 0);
    }
    if (tid == 0) ntiles[0] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < n; i += nt) {
        const uint e = topi[i];
        const uint r = cnt[e] + atomic_fetch_add_explicit(&cur[e], 1u, memory_order_relaxed);
        tok[r] = i / k;
        inv[i] = r;
    }
}

// dst[r][j] = src[idx[r]][j], rows of n floats
kernel void gather_rows(device const float *src [[buffer(0)]], device const uint *idx [[buffer(1)]], device float *dst [[buffer(2)]],
                        constant uint2 &p [[buffer(3)]] /* n, rows */, uint e [[thread_position_in_grid]]) {
    if (e >= p.x * p.y) return;
    const uint r = e / p.x, j = e % p.x;
    dst[e] = src[idx[r] * p.x + j];
}

// As moe_combine, the slot outputs read through inv (grouped row of each token-slot).
kernel void moe_combine_g(device const float *yd [[buffer(0)]], device const float *wts [[buffer(1)]], device const float *shared_y [[buffer(2)]],
                          device const float *sg [[buffer(3)]], device float *y [[buffer(4)]], constant uint3 &p [[buffer(5)]] /* H, k, T */,
                          device const uint *inv [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.x * p.z) return;
    const uint t = i / p.x, j = i % p.x;
    float acc = 0;
    for (uint s = 0; s < p.y; s++) acc += wts[t * p.y + s] * yd[inv[t * p.y + s] * p.x + j];
    y[i] = acc + sigm(sg[t]) * shared_y[i];
}

// ---- attention, split over the selected tokens (flash-decoding) -----------------------

struct AttnSP { uint pos0, budget_blocks, ratio, head_dim, kv_stride, q_per_kv, gate_stride, nq, nsplit, maxsel; float scale; };
constant uint SPL = 64;  // selected tokens per split (64: 66 threadgroups per decode attention, 128 gave 34)

// Threadgroup (h, split, t): query head h of token t over its selected tokens
// [split*SPL, (split+1)*SPL). Token t sees n = pos0+t+1 tokens; dense when n/ratio <= budget,
// else the list sel + t*maxsel of length cnt[t]. Writes (max, sum, sum p*v) partials.
kernel void attn_split(device const float *q [[buffer(0)]], device const float *kc [[buffer(1)]], device const float *vc [[buffer(2)]],
                       device const uint *sel [[buffer(3)]], device const uint *cnt [[buffer(4)]], device float *part [[buffer(5)]],
                       constant AttnSP &p [[buffer(6)]],
                       uint3 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[32];
    threadgroup float pr[SPL];
    const uint h = g.x, sp = g.y, t = g.z, nt = 256;
    const uint n = p.pos0 + t + 1;
    const bool dense = n / p.ratio <= p.budget_blocks;
    const uint nsel = dense ? n : cnt[t];
    device const uint *st = sel + t * p.maxsel;
    const uint i0 = sp * SPL, i1 = min(nsel, i0 + SPL);
    device float *out = part + ((t * p.nq + h) * p.nsplit + sp) * (p.head_dim + 2);
    if (i0 >= i1) { if (tid == 0) { out[0] = -INFINITY; out[1] = 0.0f; } for (uint e = tid; e < p.head_dim; e += nt) out[2 + e] = 0.0f; return; }
    const uint kvh = h / p.q_per_kv;
    device const float4 *qh = (device const float4 *)(q + (t * p.nq + h) * p.head_dim);
    const uint lane = tid % 32, sg = tid / 32;
    const float4 q0 = qh[lane * 2], q1 = qh[lane * 2 + 1];
    for (uint i = i0 + sg; i < i1; i += 8) {
        const uint tok = dense ? i : st[i];
        device const float4 *kt = (device const float4 *)(kc + tok * p.kv_stride + kvh * p.head_dim);
        const float d = simd_sum(dot(q0, kt[lane * 2]) + dot(q1, kt[lane * 2 + 1]));
        if (lane == 0) pr[i - i0] = d * p.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -INFINITY;
    for (uint i = tid; i < i1 - i0; i += nt) mx = max(mx, pr[i]);
    mx = tg_max(mx, sh, tid, nt);
    float sum = 0;
    for (uint i = tid; i < i1 - i0; i += nt) { pr[i] = exp(pr[i] - mx); sum += pr[i]; }
    sum = tg_sum(sum, sh, tid, nt);
    for (uint e = tid; e < p.head_dim; e += nt) {
        float acc = 0;
        for (uint i = i0; i < i1; i++) {
            const uint tok = dense ? i : st[i];
            acc += pr[i - i0] * vc[tok * p.kv_stride + kvh * p.head_dim + e];
        }
        out[2 + e] = acc;
    }
    if (tid == 0) { out[0] = mx; out[1] = sum; }
}

// Threadgroup (h, t): merges the splits of head h, token t, gated by sigmoid(gate).
kernel void attn_merge(device const float *part [[buffer(0)]], device const float *gate [[buffer(1)]], device float *att [[buffer(2)]],
                       constant AttnSP &p [[buffer(3)]], uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint h = g.x, t = g.y;
    device const float *pt = part + (t * p.nq + h) * p.nsplit * (p.head_dim + 2);
    float M = -INFINITY;
    for (uint s = 0; s < p.nsplit; s++) M = max(M, pt[s * (p.head_dim + 2)]);
    float L = 0;
    for (uint s = 0; s < p.nsplit; s++) { const float m = pt[s * (p.head_dim + 2)]; if (m > -INFINITY) L += pt[s * (p.head_dim + 2) + 1] * exp(m - M); }
    for (uint e = tid; e < p.head_dim; e += 256) {
        float acc = 0;
        for (uint s = 0; s < p.nsplit; s++) { const float m = pt[s * (p.head_dim + 2)]; if (m > -INFINITY) acc += pt[s * (p.head_dim + 2) + 2 + e] * exp(m - M); }
        att[(t * p.nq + h) * p.head_dim + e] = acc / L * sigm(gate[t * p.nq * p.gate_stride + h * p.gate_stride + e]);
    }
}

// GQA-grouped form of attn_split: threadgroup (kv head, split, t) serves the q_per_kv (12)
// query heads that share kv head g.x, so each selected key and value row is read once,
// not once per query head. Same partial layout as attn_split (merged by attn_merge).
// KV is the cache element type: float, or half (the default;).
template <typename KV>
inline void attn_split_gq(device const float *q, device const KV *kc, device const KV *vc,
                          device const uint *sel, device const uint *cnt, device float *part,
                          constant AttnSP &p, uint3 g, uint tid,
                          threadgroup float (*qs)[256], threadgroup float (*pr)[SPL], threadgroup float (*stat)[2]) {
    constexpr uint QH = 12;  // query heads per kv head (24 / 2)
    const uint kvh = g.x, sp = g.y, t = g.z, nt = 256;
    const uint n = p.pos0 + t + 1;
    const bool dense = n / p.ratio <= p.budget_blocks;
    const uint nsel = dense ? n : cnt[t];
    device const uint *st = sel + t * p.maxsel;
    const uint i0 = sp * SPL, i1 = min(nsel, i0 + SPL), cnt_i = i1 > i0 ? i1 - i0 : 0;
    const uint h0 = kvh * QH;
    if (cnt_i == 0) {
        for (uint h = 0; h < QH; h++) {
            device float *out = part + ((t * p.nq + h0 + h) * p.nsplit + sp) * (p.head_dim + 2);
            if (tid == 0) { out[0] = -INFINITY; out[1] = 0.0f; }
            out[2 + tid] = 0.0f;
        }
        return;
    }
    for (uint h = 0; h < QH; h++) qs[h][tid] = q[(t * p.nq + h0 + h) * p.head_dim + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // scores: one simdgroup per token, 8 dims per lane, all 12 heads from one key read
    const uint lane = tid % 32, sg = tid / 32;
    for (uint i = sg; i < cnt_i; i += 8) {
        const uint tok = dense ? i0 + i : st[i0 + i];
        device const vec<KV, 4> *kt = (device const vec<KV, 4> *)(kc + tok * p.kv_stride + kvh * p.head_dim);
        const float4 k0 = float4(kt[lane * 2]), k1 = float4(kt[lane * 2 + 1]);
        for (uint h = 0; h < QH; h++) {
            threadgroup const float4 *qh = (threadgroup const float4 *)qs[h];
            const float d = simd_sum(dot(qh[lane * 2], k0) + dot(qh[lane * 2 + 1], k1));
            if (lane == 0) pr[h][i] = d * p.scale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // softmax statistics per head: simdgroup sg handles heads sg, sg + 8
    for (uint h = sg; h < QH; h += 8) {
        float mx = -INFINITY;
        for (uint i = lane; i < cnt_i; i += 32) mx = max(mx, pr[h][i]);
        mx = simd_max(mx);
        float sum = 0;
        for (uint i = lane; i < cnt_i; i += 32) { const float e = exp(pr[h][i] - mx); pr[h][i] = e; sum += e; }
        sum = simd_sum(sum);
        if (lane == 0) { stat[h][0] = mx; stat[h][1] = sum; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // values: thread = dimension, 12 accumulators, each value row read once
    float acc[QH];
    for (uint h = 0; h < QH; h++) acc[h] = 0.0f;
    for (uint i = 0; i < cnt_i; i++) {
        const uint tok = dense ? i0 + i : st[i0 + i];
        const float v = float(vc[tok * p.kv_stride + kvh * p.head_dim + tid]);
        for (uint h = 0; h < QH; h++) acc[h] += pr[h][i] * v;
    }
    for (uint h = 0; h < QH; h++) {
        device float *out = part + ((t * p.nq + h0 + h) * p.nsplit + sp) * (p.head_dim + 2);
        out[2 + tid] = acc[h];
        if (tid == 0) { out[0] = stat[h][0]; out[1] = stat[h][1]; }
    }
}

kernel void attn_split_g(device const float *q [[buffer(0)]], device const float *kc [[buffer(1)]], device const float *vc [[buffer(2)]],
                         device const uint *sel [[buffer(3)]], device const uint *cnt [[buffer(4)]], device float *part [[buffer(5)]],
                         constant AttnSP &p [[buffer(6)]],
                         uint3 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float qs[12][256];
    threadgroup float pr[12][SPL];
    threadgroup float stat[12][2];
    attn_split_gq<float>(q, kc, vc, sel, cnt, part, p, g, tid, qs, pr, stat);
}

kernel void attn_split_gh(device const float *q [[buffer(0)]], device const half *kc [[buffer(1)]], device const half *vc [[buffer(2)]],
                          device const uint *sel [[buffer(3)]], device const uint *cnt [[buffer(4)]], device float *part [[buffer(5)]],
                          constant AttnSP &p [[buffer(6)]],
                          uint3 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float qs[12][256];
    threadgroup float pr[12][SPL];
    threadgroup float stat[12][2];
    attn_split_gq<half>(q, kc, vc, sel, cnt, part, p, g, tid, qs, pr, stat);
}

// dst[r * ds + i] = half(src[r * ss + i]) for i < n (KV cache writes)
kernel void copy_rows_h(device const float *src [[buffer(0)]], device half *dst [[buffer(1)]], constant uint4 &p [[buffer(2)]] /* n, rows, ss, ds */,
                        uint e [[thread_position_in_grid]]) {
    if (e >= p.x * p.y) return;
    const uint r = e / p.x, i = e % p.x;
    dst[r * p.w + i] = half(src[r * p.z + i]);
}

// ---- QSA indexer, batched over a prefill chunk -----------------------------------------

// scores[t][b] = sum_h relu(<q[t][h], bk[b]>) / sqrt(128) for b < nbs (blocks complete at the
// chunk's last token; a token only reads its own visible prefix). f32 simdgroup matrices:
// tile of 32 tokens x 32 blocks, the 4 heads accumulated separately, ReLU'd, then summed.
// q is [T][4][128] (normed, roped), bk is [blocks][128]. Grid (ceil(nbs/32), ceil(T/32)), 128 threads.
kernel void ix_score_t(device const float *q [[buffer(0)]], device const float *bk [[buffer(1)]], device float *scores [[buffer(2)]],
                       constant uint3 &p [[buffer(3)]] /* T, nbs, row stride of scores */,
                       uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
                       uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float qs[32][33];
    threadgroup float ks[32][33];
    threadgroup float ys[32][33];
    threadgroup float tt[32][33];
    const uint b0 = tg.x * 32, t0 = tg.y * 32, T = p.x, nbs = p.y;
    const uint sm = sgid / 2, sn = sgid % 2;
    for (uint e = tid; e < 32 * 32; e += 128) tt[e / 32][e % 32] = 0.0f;
    for (uint h = 0; h < 4; h++) {
        simdgroup_float8x8 acc[2][2];
        for (uint i = 0; i < 2; i++) for (uint jj = 0; jj < 2; jj++) acc[i][jj] = simdgroup_float8x8(0.0f);
        for (uint k0 = 0; k0 < 128; k0 += 32) {
            for (uint e = tid; e < 32 * 32; e += 128) {
                const uint r = e / 32, k = e % 32;
                qs[r][k] = (t0 + r < T) ? q[((t0 + r) * 4 + h) * 128 + k0 + k] : 0.0f;
                ks[r][k] = (b0 + r < nbs) ? bk[(b0 + r) * 128 + k0 + k] : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < 32; k += 8) {
                simdgroup_float8x8 a[2], bm[2];
                for (uint i = 0; i < 2; i++) simdgroup_load(a[i], &qs[sm * 16 + i * 8][k], 33);
                for (uint jj = 0; jj < 2; jj++) simdgroup_load(bm[jj], &ks[sn * 16 + jj * 8][k], 33, ulong2(0, 0), true);
                for (uint i = 0; i < 2; i++) for (uint jj = 0; jj < 2; jj++) simdgroup_multiply_accumulate(acc[i][jj], a[i], bm[jj], acc[i][jj]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // relu per head, then add into the total (through threadgroup memory)
        for (uint i = 0; i < 2; i++) for (uint jj = 0; jj < 2; jj++) simdgroup_store(acc[i][jj], &ys[sm * 16 + i * 8][sn * 16 + jj * 8], 33);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < 32 * 32; e += 128) tt[e / 32][e % 32] += max(ys[e / 32][e % 32], 0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = 1.0f / sqrt(128.0f);
    for (uint e = tid; e < 32 * 32; e += 128) {
        const uint r = e / 32, b = e % 32;
        if (t0 + r < T && b0 + b < nbs) scores[(t0 + r) * p.z + b0 + b] = tt[r][b] * inv;
    }
}

// Residual add fused with the next block's norm: per stream k (one threadgroup each),
// X[k*H + j] += 2 * sigmoid(r[k] / HC) * y[j], then xn = X * rsqrt(mean(X^2) + eps) * (1 + w).
kernel void hc_add_norm(device float *X [[buffer(0)]], device const float *y [[buffer(1)]], device const float *r [[buffer(2)]],
                        device const float *w [[buffer(3)]], device float *xn [[buffer(4)]], constant uint2 &p [[buffer(5)]] /* H, HC */,
                        constant float &eps [[buffer(6)]],
                        uint2 kt [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 ntv [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    const uint nt = ntv.x;
    const uint H = p.x, k = kt.x, t = kt.y; // stream k of token t (grid y: a small chunk)
    X += t * p.y * H;
    y += t * H;
    r += t * p.y;
    xn += t * p.y * H;
    const float gk = 2.0f * sigm(r[k] / float(p.y));
    float ss = 0;
    for (uint j = tid; j < H; j += nt) { const float v = X[k * H + j] + gk * y[j]; X[k * H + j] = v; ss += v * v; }
    ss = tg_sum(ss, sh, tid, nt);
    const float rr = 1.0f / sqrt(ss / float(H) + eps);
    for (uint j = tid; j < H; j += nt) xn[k * H + j] = X[k * H + j] * rr * (1.0f + w[k * H + j]);
}

// MTP input: X[t][k][j] += e[t][j] for the 4 hyper-connection streams (n = T * HC * H).
kernel void add_streams(device float *X [[buffer(0)]], device const float *e [[buffer(1)]], constant uint3 &p [[buffer(2)]] /* n, H, HC */,
                        uint i [[thread_position_in_grid]]) {
    if (i >= p.x) return;
    X[i] += e[(i / (p.y * p.z)) * p.y + i % p.y];
}

// Prefill accelerator path: the expert-grouped input and the SwiGLU output written directly in
// half (the matmul's input format), instead of float followed by a to_half pass.
kernel void gather_rows_h(device const float *src [[buffer(0)]], device const uint *idx [[buffer(1)]], device half *dst [[buffer(2)]],
                          constant uint2 &p [[buffer(3)]] /* n, rows */, uint e [[thread_position_in_grid]]) {
    if (e >= p.x * p.y) return;
    const uint r = e / p.x, j = e % p.x;
    dst[e] = half(src[idx[r] * p.x + j]);
}
kernel void swiglu_h(device const float *g [[buffer(0)]], device const float *u [[buffer(1)]], device half *h [[buffer(2)]],
                     constant uint &n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < n) h[i] = half(silu(g[i]) * u[i]);
}

// Prefill input: X[t][k][j] = e[t][j] for the 4 hyper-connection streams (n = T * HC * H).
kernel void bcast_streams(device float *X [[buffer(0)]], device const float *e [[buffer(1)]], constant uint3 &p [[buffer(2)]] /* n, H, HC */,
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.x) return;
    X[i] = e[(i / (p.y * p.z)) * p.y + i % p.y];
}

// Decode (one token): gdn_conv and gdn_qknorm in one dispatch. Threadgroup g < 2 * nk is q or
// k head g (dk channels: conv, then its L2 norm, q also / sqrt(dk)); the others cover dk value
// channels each (conv only). Threads: dk.
kernel void gdn_conv_norm(device const float *qkv [[buffer(0)]], device float *state [[buffer(1)]], device const float *w [[buffer(2)]],
                          device float *out [[buffer(3)]], constant uint4 &p [[buffer(4)]] /* nk, dk, key_dim, conv_dim */,
                          device const float *src [[buffer(5)]] /* start, as gdn_conv */,
                          uint g [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    const uint n = p.w, c = g * p.y + tid;
    const float s0 = src[c], s1 = src[n + c], s2 = src[2 * n + c];
    const float x = qkv[c];
    float v = silu(x * w[c * 4 + 3] + s0 * w[c * 4] + s1 * w[c * 4 + 1] + s2 * w[c * 4 + 2]);
    state[c] = s1; state[n + c] = s2; state[2 * n + c] = x;
    if (g < 2 * p.x) {
        const float ss = tg_sum(v * v, sh, tid, nt);
        v *= 1.0f / sqrt(ss + 1e-6f) * (g < p.x ? 1.0f / sqrt(float(p.y)) : 1.0f);
    }
    out[c] = v;
}

// Decode attention, after the projections, in one dispatch (f16 KV): threadgroup h < nq: query
// head h, RMS norm (1 + w) then rotate_half RoPE on the first 2 * half dims, into qn; nq + k
// (k < nkv): key head k likewise, stored as half into the KV cache row of this position; the
// last threadgroup stores the values (half) and copies the raw indexer key. Threads: head_dim.
struct AttnPrepP { uint nq, nkv, hd, half_rot, ixd, ix_off; float eps; };
kernel void attn_prep_dec(device const float *qg [[buffer(0)]], device const float *kx [[buffer(1)]], device const float *vx [[buffer(2)]],
                          device const float *ix [[buffer(3)]], device const float *qw [[buffer(4)]], device const float *kw [[buffer(5)]],
                          constant float2 *cs [[buffer(6)]], device float *qn [[buffer(7)]], device half *kc [[buffer(8)]], device half *vc [[buffer(9)]],
                          device float *ixk [[buffer(10)]], constant AttnPrepP &p [[buffer(11)]],
                          uint g [[threadgroup_position_in_grid]], uint d [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    threadgroup float row[256];
    if (g == p.nq + p.nkv) {
        for (uint i = d; i < p.nkv * p.hd; i += nt) vc[i] = half(vx[i]);
        for (uint i = d; i < p.ixd; i += nt) ixk[i] = ix[p.ix_off + i];
        return;
    }
    const bool is_q = g < p.nq;
    const float x = is_q ? qg[g * 2 * p.hd + d] : kx[(g - p.nq) * p.hd + d];
    const float ss = tg_sum(x * x, sh, d, nt);
    const float w = is_q ? qw[d] : kw[d];
    row[d] = x / sqrt(ss / float(p.hd) + p.eps) * (1.0f + w);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float y = row[d];
    if (d < p.half_rot) { const float2 c = cs[d]; y = row[d] * c.x - row[d + p.half_rot] * c.y; }
    else if (d < 2 * p.half_rot) { const float2 c = cs[d - p.half_rot]; y = row[d] * c.x + row[d - p.half_rot] * c.y; }
    if (is_q) qn[g * p.hd + d] = y;
    else kc[(g - p.nq) * p.hd + d] = half(y);
}

// BF16 matrices used as stored (the Coder pack's routers, copied 16-byte aligned at load):
// y[t][row] = W[row] . x[t], one simdgroup per (row, token), 8 values per lane per load.
// dims = (in, out, T), in % 8 == 0; grid (out, T) x 32.
kernel void mv_bf16(device const uchar *w [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]],
                    constant uint3 &d [[buffer(3)]], constant uint &shift [[buffer(4)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg.x, t = tg.y;
    if (row >= d.y) return;
    device const uint4 *wr = (device const uint4 *)(w + shift + (ulong)row * d.x * 2);
    device const float4 *xt = (device const float4 *)(x + (ulong)t * d.x);
    float acc = 0.0f;
    for (uint i = lane; i < d.x / 8; i += 32) {
        const uint4 q = wr[i];
        const float4 x0 = xt[2 * i], x1 = xt[2 * i + 1];
        acc += as_type<float>(q.x << 16) * x0.x + as_type<float>(q.x & 0xffff0000u) * x0.y
             + as_type<float>(q.y << 16) * x0.z + as_type<float>(q.y & 0xffff0000u) * x0.w
             + as_type<float>(q.z << 16) * x1.x + as_type<float>(q.z & 0xffff0000u) * x1.y
             + as_type<float>(q.w << 16) * x1.z + as_type<float>(q.w & 0xffff0000u) * x1.w;
    }
    acc = simd_sum(acc);
    if (lane == 0) y[(ulong)t * d.y + row] = acc;
}

// ---- decode QSA indexer, faster forms (same results as ix_score / ix_select) ------------
// Scores with one simdgroup per block (lanes over the 128 dims: coalesced key reads), same
// sum per head of relu(q . k) / sqrt(dim). Grid nb x 32 (dim = 128: 4 per lane).
kernel void ix_score_sg(device const float *q [[buffer(0)]], device const float *bk [[buffer(1)]], device float *scores [[buffer(2)]],
                        constant uint3 &p [[buffer(3)]] /* nb, heads, dim */,
                        uint b [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (b >= p.x) return;
    const float4 k = ((device const float4 *)(bk + b * p.z))[lane];
    float sc = 0;
    for (uint h = 0; h < p.y; h++) {
        const float d = simd_sum(dot(((device const float4 *)(q + h * p.z))[lane], k));
        sc += max(d, 0.0f);
    }
    if (lane == 0) scores[b] = sc / sqrt(float(p.z));
}

// Top-K selection with a 4-pass radix select (8 bits per pass, 256-bin threadgroup
// histogram) over each thread's scores held in registers, then the same emit as
// ix_select_row: the K-th largest value is the same threshold the 32-pass bitwise search
// finds, so the selected set and its order are identical. nb <= 1024 * RS_MAX.
constant constexpr uint RS_MAX = 64;
kernel void ix_select_rx(device const float *scores [[buffer(0)]], device uint *sel [[buffer(1)]], device uint *cnt [[buffer(2)]],
                         constant uint4 &p [[buffer(3)]] /* nb, k, ratio, n tokens */,
                         uint tid [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
    threadgroup uint sh[32];
    threadgroup atomic_uint hist[256];
    threadgroup uint pick[2];
    const uint nb = p.x, K = p.y, cr = p.z, n = p.w;
    const uint per = (nb + nt - 1) / nt, lo = min(nb, tid * per), hi = min(nb, lo + per);
    uint v[RS_MAX];
    for (uint i = 0; i < hi - lo; i++) v[i] = ordered(scores[lo + i]);
    uint prefix = 0, mask = 0, need = K;
    for (int pass = 0; pass < 4; pass++) {
        const uint shift = 24 - 8 * pass;
        for (uint i = tid; i < 256; i += nt) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < hi - lo; i++)
            if ((v[i] & mask) == prefix) atomic_fetch_add_explicit(&hist[(v[i] >> shift) & 255u], 1u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint cum = 0, d = 0;
            for (int bin = 255; bin >= 0; bin--) {
                const uint h = atomic_load_explicit(&hist[bin], memory_order_relaxed);
                if (cum + h >= need) { d = uint(bin); break; }
                cum += h;
            }
            pick[0] = d;
            pick[1] = need - cum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix |= pick[0] << shift;
        mask |= 255u << shift;
        need = pick[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint thr = prefix;
    uint gt = 0, eq = 0;
    for (uint i = 0; i < hi - lo; i++) { gt += v[i] > thr; eq += v[i] == thr; }
    const uint need_eq = K - tg_sum_u(gt, sh, tid, nt);
    const uint eq_before = tg_scan_u(eq, sh, tid, nt);
    uint mine = 0;
    {
        uint r = eq_before;
        for (uint i = 0; i < hi - lo; i++) if (v[i] > thr || (v[i] == thr && r++ < need_eq)) mine++;
    }
    uint slot = tg_scan_u(mine, sh, tid, nt) * cr;
    uint r = eq_before;
    for (uint i = 0; i < hi - lo; i++)
        if (v[i] > thr || (v[i] == thr && r++ < need_eq)) for (uint j = 0; j < cr; j++) sel[slot++] = (lo + i) * cr + j;
    if (tid == 0) {
        uint s = K * cr;
        for (uint t = nb * cr; t < n; t++) sel[s++] = t;
        cnt[0] = s;
    }
}
