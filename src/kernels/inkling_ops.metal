// Inkling 2 Mini (model_type inkling2_loop): the small per-token ops of src/inkling.c, all f32.
// Formulas: transformers be48e7a models/inkling2_loop/modeling_inkling2_loop.py (M below);
// scalar references in src/inkling_ref.c, unit tests in src/test/inkling.c.
// Every kernel takes T token rows (decode: T = 1) on the y (or only) grid axis.
#include <metal_stdlib>
using namespace metal;

// Sum over a threadgroup of up to 1024 threads; every thread gets the result.
inline float ik_sum(float v, threadgroup float *sh, uint tid, uint nt) {
    v = simd_sum(v);
    if (tid % 32 == 0) sh[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = (tid % 32 < (nt + 31) / 32) ? sh[tid % 32] : 0.0f;
    r = simd_sum(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}
inline float ik_sigm(float x) { return 1.0f / (1.0f + exp(-x)); }

// ---- attention residual mix (M:828-853, the message input M:145-173) ---------------------------
// Softmax over the RMS-normalized stacked states [states 0..n-1, partial] with nq learned queries,
// scale 1/sqrt(H); the output of query q is the softmax-weighted sum of the normalized states.
// State j of token t: states[j * sstride + t * H]; partial of token t: partial[t * pstride];
// out of token t, query q: out[t * ostride + q * H]. Grid T, any threads (1024).
struct MixP { uint n, nq, H, sstride, pstride, ostride; float eps; };
#define IK_MAXS 6
#define IK_MAXQ 4
kernel void ink_mix(device const float *states [[buffer(0)]], device const float *partial [[buffer(1)]],
                    device const float *query [[buffer(2)]], device float *out [[buffer(3)]], constant MixP &p [[buffer(4)]],
                    uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint nsg_ [[simdgroups_per_threadgroup]]) {
    const uint nt = nsg_ * 32;
    threadgroup float sh[32];
    const uint ns = p.n + 1;
    float ss[IK_MAXS], dq[IK_MAXQ][IK_MAXS];
    for (uint j = 0; j < IK_MAXS; j++) {
        ss[j] = 0.0f;
        for (uint q = 0; q < IK_MAXQ; q++) dq[q][j] = 0.0f;
    }
    device const float *pt = partial + t * p.pstride;
    for (uint i = tid; i < p.H; i += nt) {
        for (uint j = 0; j < ns; j++) {
            const float x = j < p.n ? states[j * p.sstride + t * p.H + i] : pt[i];
            ss[j] += x * x;
            for (uint q = 0; q < p.nq; q++) dq[q][j] += query[q * p.H + i] * x;
        }
    }
    float r[IK_MAXS], pr[IK_MAXQ][IK_MAXS];
    for (uint j = 0; j < ns; j++) r[j] = 1.0f / sqrt(ik_sum(ss[j], sh, tid, nt) / float(p.H) + p.eps);
    const float isq = 1.0f / sqrt(float(p.H));
    for (uint q = 0; q < p.nq; q++) {
        float mx = -INFINITY;
        for (uint j = 0; j < ns; j++) {
            pr[q][j] = ik_sum(dq[q][j], sh, tid, nt) * r[j] * isq;
            mx = max(mx, pr[q][j]);
        }
        float z = 0.0f;
        for (uint j = 0; j < ns; j++) {
            pr[q][j] = exp(pr[q][j] - mx);
            z += pr[q][j];
        }
        for (uint j = 0; j < ns; j++) pr[q][j] = pr[q][j] / z * r[j];
    }
    for (uint i = tid; i < p.H; i += nt) {
        float acc[IK_MAXQ] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint j = 0; j < ns; j++) {
            const float x = j < p.n ? states[j * p.sstride + t * p.H + i] : pt[i];
            for (uint q = 0; q < p.nq; q++) acc[q] += pr[q][j] * x;
        }
        for (uint q = 0; q < p.nq; q++) out[t * p.ostride + q * p.H + i] = acc[q];
    }
}

// ---- RMSNorm with weight (M:730-745) / without (M:67-76) ----------------------------------------
// out[t] = x[t] * rsqrt(mean(x^2) + eps) * w (w NULL: no weight). Grid T.
struct NormP { uint n, xs, os, hasw; float eps; };
kernel void ink_rmsnorm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]], device float *out [[buffer(2)]],
                        constant NormP &p [[buffer(3)]], uint t [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint nsg_ [[simdgroups_per_threadgroup]]) {
    const uint nt = nsg_ * 32;
    threadgroup float sh[32];
    device const float *xr = x + t * p.xs;
    float ss = 0.0f;
    for (uint i = tid; i < p.n; i += nt) ss += xr[i] * xr[i];
    const float r = 1.0f / sqrt(ik_sum(ss, sh, tid, nt) / float(p.n) + p.eps);
    for (uint i = tid; i < p.n; i += nt) out[t * p.os + i] = p.hasw ? xr[i] * r * w[i] : xr[i] * r;
}

// ---- message gate (M:79-96) ---------------------------------------------------------------------
// d[t] = D [norm(x[t]), norm(m[t])], D bf16 [R][2H]. Grid (R / 8, T) x 256: every threadgroup
// takes the two norms, then simdgroup sg row 8 tg.x + sg.
struct MsgP { uint H, R, xs, ms; float eps; };
kernel void ink_msg_down(device const float *x [[buffer(0)]], device const float *m [[buffer(1)]],
                         device const ushort *D [[buffer(2)]], device float *d [[buffer(3)]], constant MsgP &p [[buffer(4)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint nsg_ [[simdgroups_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    const uint nt = nsg_ * 32, t = tg.y;
    threadgroup float sh[32];
    device const float *xr = x + t * p.xs, *mr = m + t * p.ms;
    float sx = 0.0f, sm = 0.0f;
    for (uint i = tid; i < p.H; i += nt) {
        sx += xr[i] * xr[i];
        sm += mr[i] * mr[i];
    }
    const float rx = 1.0f / sqrt(ik_sum(sx, sh, tid, nt) / float(p.H) + p.eps);
    const float rm = 1.0f / sqrt(ik_sum(sm, sh, tid, nt) / float(p.H) + p.eps);
    const uint r = tg.x * nsg_ + sg;
    if (r >= p.R) return;
    device const ushort *row = D + r * 2 * p.H;
    float acc = 0.0f;
    for (uint i = lane; i < p.H; i += 32) {
        acc += as_type<float>(uint(row[i]) << 16) * (xr[i] * rx);
        acc += as_type<float>(uint(row[p.H + i]) << 16) * (mr[i] * rm);
    }
    acc = simd_sum(acc);
    if (lane == 0) d[t * p.R + r] = acc;
}

// out[t][c] = (1 - tanh(F[c] . d[t])) x[t][c] + tanh(G[c] . d[t]) m[t][c]; F, G f32 [H][R].
// Grid (H / 256, T) x 256.
kernel void ink_msg_out(device const float *x [[buffer(0)]], device const float *m [[buffer(1)]], device const float *d [[buffer(2)]],
                        device const float *F [[buffer(3)]], device const float *G [[buffer(4)]], device float *out [[buffer(5)]],
                        constant MsgP &p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    const uint c = tg.x * 256 + tid, t = tg.y;
    if (c >= p.H) return;
    device const float *dr = d + t * p.R;
    float f = 0.0f, g = 0.0f;
    for (uint k = 0; k < p.R; k++) {
        f += F[c * p.R + k] * dr[k];
        g += G[c * p.R + k] * dr[k];
    }
    out[t * p.H + c] = (1.0f - tanh(f)) * x[t * p.xs + c] + tanh(g) * m[t * p.ms + c];
}

// ---- query and relative states (M:190-236, 458-475, 239-264) -------------------------------------
// Per head: v = x * scale, v *= min(thr / (|v| + 1e-6), 1), then times tau (log scaling: the full
// layers), tau = 1 + alpha log(max((pos + 1) / floor, 1)). q rows [T][NH * HD], r rows
// [T][NH * DREL]. Grid (NH, T) x HD threads.
struct QrP { uint NH, HD, DREL, pos0, logscale; float thr_q, thr_r, alpha, nfloor; };
kernel void ink_qr_prep(device float *q [[buffer(0)]], device float *r [[buffer(1)]], device const float *qscale [[buffer(2)]],
                        device const float *rscale [[buffer(3)]], constant QrP &p [[buffer(4)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint nsg_ [[simdgroups_per_threadgroup]]) {
    const uint nt = nsg_ * 32;
    threadgroup float sh[32];
    const uint h = tg.x, t = tg.y;
    const float tau = p.logscale ? 1.0f + p.alpha * log(max(float(p.pos0 + t + 1) / p.nfloor, 1.0f)) : 1.0f;
    device float *qh = q + (t * p.NH + h) * p.HD;
    const float v = qh[tid] * qscale[h * p.HD + tid];
    const float nq = sqrt(ik_sum(v * v, sh, tid, nt));
    qh[tid] = v * min(p.thr_q / (nq + 1e-6f), 1.0f) * tau;
    device float *rh = r + (t * p.NH + h) * p.DREL;
    const float u = tid < p.DREL ? rh[tid] * rscale[h * p.DREL + tid] : 0.0f;
    const float nr = sqrt(ik_sum(u * u, sh, tid, nt));
    if (tid < p.DREL) rh[tid] = u * min(p.thr_r / (nr + 1e-6f), 1.0f) * tau;
}

// ---- short convolutions (M:328-455) -------------------------------------------------------------
// The input of token t at lag l (0: itself): chunk row t - l, or state row 3 + (t - l) for the
// tokens before the chunk (state rows: the 3 inputs before the chunk, oldest first).
inline float ik_lag(device const float *raw, device const float *state, uint C, uint c, int t, int l) {
    const int s = t - l;
    return s >= 0 ? raw[uint(s) * C + c] : state[uint(3 + s) * C + c];
}

// Keys and values: y = (sum_l (w[c][l] + dyn[t][head][l]) x_{t-l} + x_t) / den, then per head
// (HD channels) the key L2 clip (mode 0) or the value's unweighted RMSNorm (mode 1). w [C][4]
// in lag order (static convs converted at load; the dynamic convs' static taps M:432);
// dyn [T][NKV * 4] (weight_proj of the k/v input, M:430-432), NULL-equivalent when dyn == 0.
// update (T == 1 only): the state shifts in this token's input. Grid (NKV, T) x HD.
struct KvP { uint C, HD, dyn, mode, update; float den, thr, eps; };
kernel void ink_kv_conv(device const float *raw [[buffer(0)]], device float *state [[buffer(1)]], device const float *w [[buffer(2)]],
                        device const float *dynw [[buffer(3)]], device float *out [[buffer(4)]], constant KvP &p [[buffer(5)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint nsg_ [[simdgroups_per_threadgroup]]) {
    const uint nt = nsg_ * 32;
    threadgroup float sh[32];
    const uint h = tg.x, t = tg.y, c = h * p.HD + tid;
    const uint nkv = p.C / p.HD;
    float y = 0.0f;
    for (int l = 0; l < 4; l++) {
        const float wl = w[c * 4 + l] + (p.dyn ? dynw[t * nkv * 4 + h * 4 + l] : 0.0f);
        y += wl * ik_lag(raw, state, p.C, c, int(t), l);
    }
    const float x0 = raw[t * p.C + c];
    y = (y + x0) / p.den;
    const float n2 = ik_sum(y * y, sh, tid, nt);
    if (p.mode == 0) y *= min(p.thr / (sqrt(n2) + 1e-6f), 1.0f);
    else y *= 1.0f / sqrt(n2 / float(p.HD) + p.eps);
    out[t * p.C + c] = y;
    if (p.update) {
        state[c] = state[p.C + c];
        state[p.C + c] = state[2 * p.C + c];
        state[2 * p.C + c] = x0;
    }
}

// Post-attention / post-MLP conv with the residual (M:899-910): y = (sum_l w[c][l] x_{t-l} + x_t)
// / den; boundary: bs[t] = hin[t] (the block state the layer appends) and r = y; else r = hin + y.
// update (T == 1 only) shifts the state. Grid (C / 256, T) x 256.
struct CrP { uint C, boundary, update; float den; };
kernel void ink_conv_res(device const float *a [[buffer(0)]], device float *state [[buffer(1)]], device const float *w [[buffer(2)]],
                         device const float *hin [[buffer(3)]], device float *r [[buffer(4)]], device float *bs [[buffer(5)]],
                         constant CrP &p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    const uint c = tg.x * 256 + tid, t = tg.y;
    if (c >= p.C) return;
    float y = 0.0f;
    for (int l = 0; l < 4; l++) y += w[c * 4 + l] * ik_lag(a, state, p.C, c, int(t), l);
    const float x0 = a[t * p.C + c];
    y = (y + x0) / p.den;
    const float hv = hin[t * p.C + c];
    if (p.boundary) {
        bs[t * p.C + c] = hv;
        r[t * p.C + c] = y;
    } else
        r[t * p.C + c] = hv + y;
    if (p.update) {
        state[c] = state[p.C + c];
        state[p.C + c] = state[2 * p.C + c];
        state[2 * p.C + c] = x0;
    }
}

// The conv state after a chunk of T rows (T >= 1): the last 3 of [state, raw]. Grid C / 256.
struct CsP { uint C, T; };
kernel void ink_conv_state(device const float *raw [[buffer(0)]], device float *state [[buffer(1)]], constant CsP &p [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= p.C) return;
    for (uint k = 0; k < 3; k++) {
        const uint s = p.T + k; // index into [state 0..2, raw 0..T-1]
        state[k * p.C + i] = s < 3 ? state[s * p.C + i] : raw[(s - 3) * p.C + i];
    }
}

// ---- router (M:1152-1172) -----------------------------------------------------------------------
// logits = W x (f32): grid (E / 8, T) x 256, simdgroup sg row 8 tg.x + sg.
kernel void ink_router_logits(device const float *x [[buffer(0)]], device const float *W [[buffer(1)]], device float *lg [[buffer(2)]],
                              constant uint2 &p [[buffer(3)]] /* E, H */, uint2 tg [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint e = tg.x * 8 + sg, t = tg.y;
    if (e >= p.x) return;
    device const float4 *xr = (device const float4 *)(x + t * p.y), *wr = (device const float4 *)(W + e * p.y);
    float acc = 0.0f;
    for (uint i = lane; i < p.y / 4; i += 32) acc += dot(wr[i], xr[i]);
    acc = simd_sum(acc);
    if (lane == 0) lg[t * p.x + e] = acc;
}

// scores = sigmoid(logits), top-K by score + bias (first index on ties), weights the chosen
// scores / (sum + 1e-20) * scale. With a reference (verify): ref[t * K ..] are its experts and
// margin[t] its 8th - 9th score gap; forced uses them instead of the selection, and check
// compares the selection (as a set) with them: a difference counts in stats[0] where the
// reference's margin exceeds thr, else in stats[1], and stats[2] keeps the first such position.
// Grid T x E threads (E <= 128, a multiple of 32).
struct RtP { uint E, H, K; float scale; uint forced, check, pos0; float thr; };
kernel void ink_router(device const float *lg [[buffer(0)]], device const float *bias [[buffer(1)]], device uint *idx [[buffer(2)]],
                       device float *wts [[buffer(3)]], constant RtP &p [[buffer(4)]], device const uint *ref [[buffer(5)]],
                       device const float *margin [[buffer(6)]], device atomic_uint *stats [[buffer(7)]],
                       uint t [[threadgroup_position_in_grid]], uint e [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sc[128], bv[4];
    threadgroup uint bi[4], sel[16];
    const float s = ik_sigm(lg[t * p.E + e]);
    sc[e] = s;
    float b = s + bias[e];
    const uint nsg = p.E / 32;
    if (!p.forced || p.check)
        for (uint k = 0; k < p.K; k++) {
            // the first largest: max value, then the smallest index holding it
            const float mv = simd_max(b);
            const uint mi = simd_min(b == mv ? e : 0xFFFFFFFFu);
            if (lane == 0) {
                bv[sg] = mv;
                bi[sg] = mi;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint best = bi[0];
            float bb = bv[0];
            for (uint g = 1; g < nsg; g++)
                if (bv[g] > bb) {
                    bb = bv[g];
                    best = bi[g];
                }
            if (e == best) b = -INFINITY;
            if (e == 0) sel[k] = best;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e != 0) return;
    if (p.check) {
        bool same = true;
        for (uint k = 0; k < p.K; k++) {
            bool found = false;
            for (uint j = 0; j < p.K; j++) found = found || ref[t * p.K + j] == sel[k];
            same = same && found;
        }
        if (!same) {
            atomic_fetch_add_explicit(&stats[margin[t] > p.thr ? 0 : 1], 1u, memory_order_relaxed);
            atomic_fetch_min_explicit(&stats[2], p.pos0 + t, memory_order_relaxed);
        }
    }
    if (p.forced)
        for (uint k = 0; k < p.K; k++) sel[k] = ref[t * p.K + k];
    float sum = 0.0f;
    for (uint k = 0; k < p.K; k++) sum += sc[sel[k]];
    for (uint k = 0; k < p.K; k++) {
        idx[t * p.K + k] = sel[k];
        wts[t * p.K + k] = sc[sel[k]] / (sum + 1e-20f) * p.scale;
    }
}

// ys [T][K][C] summed over K. Grid (C / 256, T) x 256.
struct SumP { uint C, K; };
kernel void ink_moe_sum(device const float *ys [[buffer(0)]], device float *out [[buffer(1)]], constant SumP &p [[buffer(2)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint c = tg.x * 256 + tid, t = tg.y;
    if (c >= p.C) return;
    float s = 0.0f;
    for (uint k = 0; k < p.K; k++) s += ys[(t * p.K + k) * p.C + c];
    out[t * p.C + c] = s;
}

// logits = tanh(z / cap) * cap (M:751-757, 1535), in place. Grid n / 256.
struct CapP { uint n; float cap; };
kernel void ink_softcap(device float *z [[buffer(0)]], constant CapP &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) z[i] = tanh(z[i] / p.cap) * p.cap;
}

// dst[i] = src[i]. Grid n / 256.
kernel void ink_copy(device const float *src [[buffer(0)]], device float *dst [[buffer(1)]], constant uint &n [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = src[i];
}

// KV store after attention: chunk rows [T][C] to cache rows (ring: (pos0 + t) % ring, else pos0 + t;
// in a ring only the chunk's last `ring` rows, which are the ones it keeps). Grid (C / 256, T) x 256.
struct StP { uint C, pos0, ring, T; };
kernel void ink_kv_store(device const float *kn [[buffer(0)]], device const float *vn [[buffer(1)]], device float *kc [[buffer(2)]],
                         device float *vc [[buffer(3)]], constant StP &p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    const uint c = tg.x * 256 + tid, t = tg.y;
    if (c >= p.C || (p.ring && t + p.ring < p.T)) return;
    const uint pos = p.pos0 + t, row = p.ring ? pos % p.ring : pos;
    kc[row * p.C + c] = kn[t * p.C + c];
    vc[row * p.C + c] = vn[t * p.C + c];
}
