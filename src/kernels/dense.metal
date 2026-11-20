// Kernels of the Qwen3.8-27B forward (src/dense.c), appended to qmatvec.metal, decode.metal
// and qmm.metal, whose helpers (tg_sum, sigm, GdnP, AttnSP, SPL) they use.

// x[g] += y[g] (n values per group), then out[g] = rms(x[g]) * w: the residual add of one
// block fused with the plain RMSNorm of the next (the MLX pack stores 1 + w already,
// vendor/mlx-lm/mlx_lm/models/qwen3_5.py sanitize). Threadgroup per group (token).
kernel void add_rmsnorm(device float *x [[buffer(0)]], device const float *y [[buffer(1)]], device const float *w [[buffer(2)]],
                        device float *out [[buffer(3)]], constant uint &n [[buffer(4)]], constant float &eps [[buffer(5)]],
                        uint g [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                        uint nt [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    device float *xg = x + g * n;
    device const float *yg = y + g * n;
    float ss = 0;
    for (uint i = tid; i < n; i += nt) { const float v = xg[i] + yg[i]; xg[i] = v; ss += v * v; }
    ss = tg_sum(ss, sh, tid, nt);
    const float r = 1.0f / sqrt(ss / float(n) + eps);
    for (uint i = tid; i < n; i += nt) out[g * n + i] = xg[i] * r * w[i];
}

// Gated RMSNorm of the GDN output with a SiLU gate (27B: Qwen3_5RMSNormGated, activation silu,
// vendor/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py:202-220), in place:
// o * rsqrt(mean(o^2) + eps) * w * silu(z). Threads: dv.
kernel void gdn_gnorm_silu(device float *o [[buffer(0)]], device const float *z [[buffer(1)]], device const float *w [[buffer(2)]],
                           constant GdnP &p [[buffer(3)]], uint g [[threadgroup_position_in_grid]], uint j [[thread_position_in_threadgroup]],
                           uint nt [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    const float v = o[g * p.dv + j];
    const float ss = tg_sum(v * v, sh, j, nt);
    const float zz = z[g * p.dv + j];
    o[g * p.dv + j] = v / sqrt(ss / float(p.dv) + p.eps) * w[j] * (zz * sigm(zz));
}

// ---- dense causal attention, split over the visible tokens (flash-decoding) ------------
//
// Threadgroup (kv head, split, t): the QH query heads of kv head g.x for query token t, which
// sees n = pos0 + t + 1 tokens, over its share [sp * per, (sp + 1) * per) with per =
// ceil(n / nsplit). Tokens are taken SPL at a time with an online softmax, so a split may cover
// any number of tokens. Writes (max, sum, sum p * v) partials in attn_split's layout, merged by
// attn_merge. 256 threads = head_dim. K and V are the cache element type.
template <typename KV, uint QH>
inline void attn_full_q(device const float *q, device const KV *kc, device const KV *vc, device float *part,
                        constant AttnSP &p, uint3 g, uint tid,
                        threadgroup float (*qs)[256], threadgroup float (*pr)[SPL], threadgroup float (*stat)[3]) {
    const uint kvh = g.x, sp = g.y, t = g.z;
    const uint n = p.pos0 + t + 1;
    const uint per = (n + p.nsplit - 1) / p.nsplit;
    const uint i0 = min(n, sp * per), i1 = min(n, i0 + per);
    const uint h0 = kvh * QH;
    if (i0 >= i1) {
        for (uint h = 0; h < QH; h++) {
            device float *out = part + ((t * p.nq + h0 + h) * p.nsplit + sp) * (p.head_dim + 2);
            if (tid == 0) { out[0] = -INFINITY; out[1] = 0.0f; }
            out[2 + tid] = 0.0f;
        }
        return;
    }
    for (uint h = 0; h < QH; h++) qs[h][tid] = q[(t * p.nq + h0 + h) * p.head_dim + tid];
    if (tid < QH) { stat[tid][0] = -INFINITY; stat[tid][1] = 0.0f; stat[tid][2] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint lane = tid % 32, sg = tid / 32;
    float acc[QH];
    for (uint h = 0; h < QH; h++) acc[h] = 0.0f;
    for (uint c0 = i0; c0 < i1; c0 += SPL) {
        const uint cn = min(SPL, i1 - c0);
        // scores: one simdgroup per token, 8 dims per lane, every head from one key read
        for (uint i = sg; i < cn; i += 8) {
            device const vec<KV, 4> *kt = (device const vec<KV, 4> *)(kc + (c0 + i) * p.kv_stride + kvh * p.head_dim);
            const float4 k0 = float4(kt[lane * 2]), k1 = float4(kt[lane * 2 + 1]);
            for (uint h = 0; h < QH; h++) {
                threadgroup const float4 *qh = (threadgroup const float4 *)qs[h];
                const float d = simd_sum(dot(qh[lane * 2], k0) + dot(qh[lane * 2 + 1], k1));
                if (lane == 0) pr[h][i] = d * p.scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax per head: simdgroup sg handles heads sg, sg + 8, ...
        for (uint h = sg; h < QH; h += 8) {
            float mx = -INFINITY;
            for (uint i = lane; i < cn; i += 32) mx = max(mx, pr[h][i]);
            mx = simd_max(mx);
            const float m_old = stat[h][0];
            const float m_new = max(m_old, mx);
            float sum = 0;
            for (uint i = lane; i < cn; i += 32) { const float e = exp(pr[h][i] - m_new); pr[h][i] = e; sum += e; }
            sum = simd_sum(sum);
            if (lane == 0) {
                const float corr = exp(m_old - m_new); // 0 on the first chunk (m_old = -inf)
                stat[h][0] = m_new;
                stat[h][1] = stat[h][1] * corr + sum;
                stat[h][2] = corr;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // values: thread = dimension, QH accumulators, each value row read once
        for (uint h = 0; h < QH; h++) acc[h] *= stat[h][2];
        for (uint i = 0; i < cn; i++) {
            const float v = float(vc[(c0 + i) * p.kv_stride + kvh * p.head_dim + tid]);
            for (uint h = 0; h < QH; h++) acc[h] += pr[h][i] * v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // pr is rewritten by the next chunk
    }
    for (uint h = 0; h < QH; h++) {
        device float *out = part + ((t * p.nq + h0 + h) * p.nsplit + sp) * (p.head_dim + 2);
        out[2 + tid] = acc[h];
        if (tid == 0) { out[0] = stat[h][0]; out[1] = stat[h][1]; }
    }
}

// 27B: 24 query heads over 4 kv heads (6 per kv head).
kernel void attn_full_6(device const float *q [[buffer(0)]], device const float *kc [[buffer(1)]], device const float *vc [[buffer(2)]],
                        device float *part [[buffer(3)]], constant AttnSP &p [[buffer(4)]],
                        uint3 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float qs[6][256];
    threadgroup float pr[6][SPL];
    threadgroup float stat[6][3];
    attn_full_q<float, 6>(q, kc, vc, part, p, g, tid, qs, pr, stat);
}

kernel void attn_full_6h(device const float *q [[buffer(0)]], device const half *kc [[buffer(1)]], device const half *vc [[buffer(2)]],
                         device float *part [[buffer(3)]], constant AttnSP &p [[buffer(4)]],
                         uint3 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float qs[6][256];
    threadgroup float pr[6][SPL];
    threadgroup float stat[6][3];
    attn_full_q<half, 6>(q, kc, vc, part, p, g, tid, qs, pr, stat);
}
