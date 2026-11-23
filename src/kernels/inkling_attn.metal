// Inkling 2 Mini attention (M:546-727 of transformers be48e7a modeling_inkling2_loop.py), f32:
// scores q.k / sqrt(HD) + the relative bias rs . proj[:, distance] (0 at distances >= EXT,
// M:239-270), causal, sliding layers masked to distances < win, one sink logit per head with a
// zero value (M:132-135), output times sigmoid(gate) (M:531-543). Compiled after
// inkling_ops.metal. Scalar reference: src/inkling_ref.c (ink_ref_attn).
//
// Keys of positions below pos0 come from the cache (a ring of `ring` rows on sliding layers: row
// position % ring, else row position), those from pos0 on from the chunk's own rows kn / vn:
// the chunk is written to the ring only after its attention (ink_kv_store), so a chunk longer
// than the window never reads a row a later token of the same chunk overwrote.
//
// ink_attn_part: grid (NH, nsplit, T) x 128 (4 simdgroups; lane l holds dims 4l..4l+3). Split s
// of query t covers keys [lo + s * span, lo + (s + 1) * span) of its range [lo, pos]; writes
// (acc[HD], max, sum) of its online softmax to part[((t * NH + h) * nsplit + s) * (HD + 2)].
struct AttP { uint NH, NKV, HD, DREL, EXT, pos0, win, ring, nsplit, span; float scale; };
kernel void ink_attn_part(device const float *q [[buffer(0)]], device const float *rs [[buffer(1)]],
                          device const float *proj [[buffer(2)]], device const float *kc [[buffer(3)]],
                          device const float *vc [[buffer(4)]], device const float *kn [[buffer(5)]],
                          device const float *vn [[buffer(6)]], device float *part [[buffer(7)]], constant AttP &p [[buffer(8)]],
                          uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tm[4], tl[4];
    threadgroup float4 tacc[4][32];
    const uint h = tg.x, s = tg.y, t = tg.z;
    const uint kvh = h / (p.NH / p.NKV), ld = p.NKV * p.HD;
    const int pq = int(p.pos0 + t);
    const int lo = p.win ? max(0, pq - int(p.win) + 1) : 0;
    const int k0 = lo + int(s * p.span), k1 = min(k0 + int(p.span), pq + 1);
    const float4 qv = ((device const float4 *)(q + (t * p.NH + h) * p.HD))[lane] * p.scale;
    const float rl = lane < p.DREL ? rs[(t * p.NH + h) * p.DREL + lane] : 0.0f;
    float m = -INFINITY, l = 0.0f;
    float4 acc = 0.0f;
    for (int j = k0 + int(sg); j < k1; j += 4) {
        device const float *kp, *vp;
        if (j < int(p.pos0)) {
            const uint row = p.ring ? uint(j) % p.ring : uint(j);
            kp = kc + row * ld;
            vp = vc + row * ld;
        } else {
            kp = kn + uint(j - int(p.pos0)) * ld;
            vp = vn + uint(j - int(p.pos0)) * ld;
        }
        const uint d = uint(pq - j);
        float part_s = dot(qv, ((device const float4 *)(kp + kvh * p.HD))[lane]);
        if (lane < p.DREL && d < p.EXT) part_s += rl * proj[lane * p.EXT + d];
        const float sc = simd_sum(part_s);
        const float mn = max(m, sc);
        const float c = exp(m - mn), e = exp(sc - mn);
        l = l * c + e;
        acc = acc * c + e * ((device const float4 *)(vp + kvh * p.HD))[lane];
        m = mn;
    }
    if (lane == 0) {
        tm[sg] = m;
        tl[sg] = l;
    }
    tacc[sg][lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float M = -INFINITY;
    for (uint i = 0; i < 4; i++) M = max(M, tm[i]);
    float L = 0.0f, A = 0.0f;
    if (M > -INFINITY)
        for (uint i = 0; i < 4; i++) {
            if (tm[i] == -INFINITY) continue;
            const float f = exp(tm[i] - M);
            L += tl[i] * f;
            A += ((threadgroup float *)tacc[i])[tid] * f;
        }
    device float *o = part + ((t * p.NH + h) * p.nsplit + s) * (p.HD + 2);
    o[tid] = A;
    if (tid == 0) {
        o[p.HD] = M;
        o[p.HD + 1] = L;
    }
}

// The splits of each (query, head) with its sink, gated: out[t][h] = sigmoid(gate) * sum_s acc_s
// e^{m_s - M} / (e^{sink - M} + sum_s l_s e^{m_s - M}). Grid (NH, T) x HD.
kernel void ink_attn_merge(device const float *part [[buffer(0)]], device const float *sink [[buffer(1)]],
                           device const float *gate [[buffer(2)]], device float *out [[buffer(3)]], constant AttP &p [[buffer(4)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint h = tg.x, t = tg.y;
    device const float *pp = part + (t * p.NH + h) * p.nsplit * (p.HD + 2);
    const float sk = sink[t * p.NH + h];
    float M = sk;
    for (uint s = 0; s < p.nsplit; s++) M = max(M, pp[s * (p.HD + 2) + p.HD]);
    float L = exp(sk - M), A = 0.0f;
    for (uint s = 0; s < p.nsplit; s++) {
        device const float *ps = pp + s * (p.HD + 2);
        if (ps[p.HD] == -INFINITY) continue;
        const float f = exp(ps[p.HD] - M);
        L += ps[p.HD + 1] * f;
        A += ps[tid] * f;
    }
    const uint o = (t * p.NH + h) * p.HD + tid;
    out[o] = A / L * ik_sigm(gate[o]);
}
