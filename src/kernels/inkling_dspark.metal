// Inkling 2 Mini's DSpark block drafter (thinkingmachines' mtp.safetensors; the forward as vLLM
// ef687f6 vllm/models/inkling2/nvidia/dspark.py runs it, D below): f32 activations, BF16 weights
// (16-byte aligned rows in the model's own buffer). Compiled after inkling_ops.metal and
// inkling_attn.metal.

// y[t][r] = W[r][0..in) . x[t] for t < T <= 8 (W rows ldw values apart, starting at column col0):
// one simdgroup per output row reads it once for every token. Grid (ceil(out / 8), ceil(T / 8))
// x 256; the second axis takes blocks of 8 tokens.
struct DsMvP { uint in, out, T, ldw, col0, xs, ys; };
kernel void ds_mv(device const ushort *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]],
                  constant DsMvP &p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint r = tg.x * 8 + sg, t0 = tg.y * 8, nt = min(8u, p.T - t0);
    if (r >= p.out) return;
    device const ushort *wr = W + ulong(r) * p.ldw + p.col0;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint i = lane * 4; i < p.in; i += 128) {
        const ushort4 q = *(device const ushort4 *)(wr + i);
        const float4 w4 = float4(as_type<float>(uint(q.x) << 16), as_type<float>(uint(q.y) << 16), as_type<float>(uint(q.z) << 16),
                                 as_type<float>(uint(q.w) << 16));
        for (uint t = 0; t < nt; t++) acc[t] += dot(w4, *(device const float4 *)(x + (t0 + t) * p.xs + i));
    }
    for (uint t = 0; t < nt; t++) {
        const float a = simd_sum(acc[t]);
        if (lane == 0) y[(t0 + t) * p.ys + r] = a;
    }
}

// out[t] = RMSNorm(sum_k alpha[k] tap_k[t]) * w (D:403-408, ops mix_rmsnorm): taps [4][tstride].
// Grid T x 256.
struct DsMixP { uint H, tstride; float eps; };
kernel void ds_mix_norm(device const float *taps [[buffer(0)]], device const float *alpha [[buffer(1)]], device const float *w [[buffer(2)]],
                        device float *out [[buffer(3)]], constant DsMixP &p [[buffer(4)]], uint t [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint nsg_ [[simdgroups_per_threadgroup]]) {
    threadgroup float sh[32];
    const uint nt = nsg_ * 32;
    float ss = 0.0f;
    for (uint i = tid; i < p.H; i += nt) {
        float v = 0.0f;
        for (uint k = 0; k < 4; k++) v += taps[k * p.tstride + t * p.H + i] * alpha[k];
        out[t * p.H + i] = v;
        ss += v * v;
    }
    const float r = 1.0f / sqrt(ik_sum(ss, sh, tid, nt) / float(p.H) + p.eps);
    for (uint i = tid; i < p.H; i += nt) out[t * p.H + i] = out[t * p.H + i] * r * w[i];
}

// Per-head RMSNorm with weight (when w is given) then NeoX RoPE (halves rotated) with the
// host's cos / sin rows cs[t][HD / 2][2] (double precision), in place on x[t][heads][HD].
// Grid (heads, T) x 64.
struct DsRopeP { uint heads, HD, xs; uint norm; float eps; };
kernel void ds_norm_rope(device float *x [[buffer(0)]], device const float *w [[buffer(1)]], device const float *cs [[buffer(2)]],
                         constant DsRopeP &p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                         uint i [[thread_index_in_threadgroup]]) {
    threadgroup float sh[32];
    const uint h = tg.x, t = tg.y, half_ = p.HD / 2;
    device float *row = x + t * p.xs + h * p.HD;
    float a = row[i], b = row[i + half_];
    if (p.norm) {
        const float r = 1.0f / sqrt(ik_sum(a * a + b * b, sh, i, half_) / float(p.HD) + p.eps);
        a = a * r * w[i];
        b = b * r * w[i + half_];
    }
    const float c = cs[(t * half_ + i) * 2], s = cs[(t * half_ + i) * 2 + 1];
    row[i] = a * c - b * s;
    row[i + half_] = b * c + a * s;
}

// Block attention (D:195-246): the 8 block rows (positions p0 .. p0 + 7) see every context key in
// [lo, p0) from the cache (ring of `ring` rows on the local layers) and all 8 block keys (not
// causal), scale 1/sqrt(HD), no bias, no sink; the 3 query heads of a KV head together, split-K
// like ink_attn_part (its partials, merged by ink_attn_merge with a sink of -inf and gate 1).
// Grid (NKV, nsplit, 8) x 128.
struct DsAttP { uint NH, NKV, HD, p0, lo, ring, nsplit, span; float scale; };
template <typename KT>
inline void ds_attn_part_t(device const float *q, device const KT *kc, device const KT *vc, device const float *kb,
                           device const float *vb, device float *part, constant DsAttP &p, uint3 tg, uint sg, uint lane, uint tid,
                           threadgroup float (*tm)[IK_QPK], threadgroup float (*tl)[IK_QPK], threadgroup float4 (*tacc)[IK_QPK][32]) {
    const uint g = tg.x, s = tg.y, t = tg.z, h0 = g * IK_QPK, ld = p.NKV * p.HD;
    const int end = int(p.p0) + 8;
    const int k0 = int(p.lo) + int(s * p.span), k1 = min(k0 + int(p.span), end);
    float4 qv[IK_QPK], acc[IK_QPK];
    float m[IK_QPK], l[IK_QPK];
    for (uint i = 0; i < IK_QPK; i++) {
        qv[i] = ((device const float4 *)(q + (t * p.NH + h0 + i) * p.HD))[lane] * p.scale;
        m[i] = -INFINITY;
        l[i] = 0.0f;
        acc[i] = 0.0f;
    }
    for (int j = k0 + int(sg); j < k1; j += 4) {
        float4 kv4, vv4;
        if (j >= int(p.p0)) {
            const uint o = uint(j - int(p.p0)) * ld + g * p.HD + lane * 4;
            kv4 = ik_ld4(kb + o);
            vv4 = ik_ld4(vb + o);
        } else {
            const uint o = (p.ring ? uint(j) % p.ring : uint(j)) * ld + g * p.HD + lane * 4;
            kv4 = ik_ld4(kc + o);
            vv4 = ik_ld4(vc + o);
        }
        for (uint i = 0; i < IK_QPK; i++) {
            const float sc = simd_sum(dot(qv[i], kv4));
            const float mn = max(m[i], sc);
            const float c = exp(m[i] - mn), e = exp(sc - mn);
            l[i] = l[i] * c + e;
            acc[i] = acc[i] * c + e * vv4;
            m[i] = mn;
        }
    }
    for (uint i = 0; i < IK_QPK; i++) {
        if (lane == 0) {
            tm[sg][i] = m[i];
            tl[sg][i] = l[i];
        }
        tacc[sg][i][lane] = acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < IK_QPK; i++) {
        float M = -INFINITY;
        for (uint w = 0; w < 4; w++) M = max(M, tm[w][i]);
        float L = 0.0f, A = 0.0f;
        if (M > -INFINITY)
            for (uint w = 0; w < 4; w++) {
                if (tm[w][i] == -INFINITY) continue;
                const float f = exp(tm[w][i] - M);
                L += tl[w][i] * f;
                A += ((threadgroup float *)tacc[w][i])[tid] * f;
            }
        device float *o = part + ((t * p.NH + h0 + i) * p.nsplit + s) * (p.HD + 2);
        o[tid] = A;
        if (tid == 0) {
            o[p.HD] = M;
            o[p.HD + 1] = L;
        }
    }
}
#define DS_ATTN_KERNEL(NAME, KT)                                                                                           \
    kernel void NAME(device const float *q [[buffer(0)]], device const KT *kc [[buffer(1)]], device const KT *vc [[buffer(2)]], \
                     device const float *kb [[buffer(3)]], device const float *vb [[buffer(4)]], device float *part [[buffer(5)]], \
                     constant DsAttP &p [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],                             \
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                   \
                     uint tid [[thread_index_in_threadgroup]]) {                                                            \
        threadgroup float tm[4][IK_QPK], tl[4][IK_QPK];                                                                     \
        threadgroup float4 tacc[4][IK_QPK][32];                                                                             \
        ds_attn_part_t<KT>(q, kc, vc, kb, vb, part, p, tg, sg, lane, tid, tm, tl, tacc);                                    \
    }
DS_ATTN_KERNEL(ds_attn_part, float)
DS_ATTN_KERNEL(ds_attn_part_h, half)

// x += y (n values); then out = RMSNorm(x) * w per row of H (rows = n / H). Grid rows x 256.
struct DsAddP { uint H; float eps; };
kernel void ds_add_norm(device float *x [[buffer(0)]], device const float *y [[buffer(1)]], device const float *w [[buffer(2)]],
                        device float *out [[buffer(3)]], constant DsAddP &p [[buffer(4)]], uint t [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint nsg_ [[simdgroups_per_threadgroup]]) {
    threadgroup float sh[32];
    const uint nt = nsg_ * 32;
    float ss = 0.0f;
    for (uint i = tid; i < p.H; i += nt) {
        const float v = x[t * p.H + i] + y[t * p.H + i];
        x[t * p.H + i] = v;
        ss += v * v;
    }
    const float r = 1.0f / sqrt(ik_sum(ss, sh, tid, nt) / float(p.H) + p.eps);
    for (uint i = tid; i < p.H; i += nt) out[t * p.H + i] = x[t * p.H + i] * r * w[i];
}

// h = silu(g) * u (fp32, D ops silu_and_mul); g, u rows of n. Grid n * T / 256.
kernel void ds_silu_mul(device const float *g [[buffer(0)]], device const float *u [[buffer(1)]], device float *h [[buffer(2)]],
                        constant uint &n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < n) h[i] = g[i] / (1.0f + exp(-g[i])) * u[i];
}

// One step of the Markov head (D:330-385, ops markov_gate): raw = ctx + bias + R[:, 0:256] state +
// R[:, 256:512] w1[prev] (R the rnn_proj weight, rows of ldr = 2 * 256 + 2 * hidden), g, c, o = raw's thirds; state =
// sigmoid(g) state + (1 - sigmoid(g)) tanh(c); latent = tanh(o). One threadgroup of 256.
kernel void ds_markov_step(device const float *ctx [[buffer(0)]], device const float *bias [[buffer(1)]], device float *state [[buffer(2)]],
                           device const uint *prev [[buffer(3)]], device const ushort *w1 [[buffer(4)]], device const ushort *R [[buffer(5)]],
                           device float *latent [[buffer(6)]], constant uint &ldr [[buffer(7)]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float st[256], pe[256], raw[768];
    const uint pv = prev[0];
    st[tid] = state[tid];
    pe[tid] = as_type<float>(uint(w1[ulong(pv) * 256 + tid]) << 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = tid; r < 768; r += 256) {
        device const ushort *wr = R + ulong(r) * ldr;
        float a = ctx[r] + bias[r];
        for (uint i = 0; i < 256; i++) a += as_type<float>(uint(wr[i]) << 16) * st[i];
        for (uint i = 0; i < 256; i++) a += as_type<float>(uint(wr[256 + i]) << 16) * pe[i];
        raw[r] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gs = ik_sigm(raw[tid]);
    state[tid] = gs * st[tid] + (1.0f - gs) * tanh(raw[256 + tid]);
    latent[tid] = tanh(raw[512 + tid]);
}

// logits of a draft step, as argmax candidates: base[v] + fb[v] + F[v] . latent (F f16
// [vocab][256], the lm_head folded with hidden_out); the softcap is monotone and left out. Each
// threadgroup writes its first largest (value, index). Grid ceil(V / 256) x 256 (8 simdgroups, a
// row per simdgroup at a time).
struct DsFoldP { uint V; };
kernel void ds_fold_argmax(device const float *base [[buffer(0)]], device const float *fb [[buffer(1)]], device const half *F [[buffer(2)]],
                           device const float *latent [[buffer(3)]], device float2 *partial [[buffer(4)]], constant DsFoldP &p [[buffer(5)]],
                           uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float bv[8];
    threadgroup uint bi[8];
    const float4 l0 = ((device const float4 *)latent)[lane * 2], l1 = ((device const float4 *)latent)[lane * 2 + 1];
    float best = -INFINITY;
    uint bidx = 0xFFFFFFFFu;
    for (uint k = 0; k < 32; k++) {
        const uint v = tg * 256 + sg * 32 + k;
        if (v >= p.V) break;
        device const half4 *fr = (device const half4 *)(F + ulong(v) * 256 + lane * 8);
        const float a = simd_sum(dot(float4(fr[0]), l0) + dot(float4(fr[1]), l1));
        const float z = base[v] + fb[v] + a;
        if (z > best) {
            best = z;
            bidx = v;
        }
    }
    if (lane == 0) {
        bv[sg] = best;
        bi[sg] = bidx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float b = bv[0];
        uint i = bi[0];
        for (uint s = 1; s < 8; s++)
            if (bv[s] > b) {
                b = bv[s];
                i = bi[s];
            }
        partial[tg] = float2(b, as_type<float>(i));
    }
}

// The step's draft: the first largest of the partials (NaN as -inf), to draft[step] and prev.
// One threadgroup of 256.
kernel void ds_argmax_final(device const float2 *partial [[buffer(0)]], constant uint &n [[buffer(1)]], device uint *draft [[buffer(2)]],
                            device uint *prev [[buffer(3)]], constant uint &step [[buffer(4)]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float bv[256];
    threadgroup uint bi[256];
    float b = -INFINITY;
    uint i = 0xFFFFFFFFu;
    for (uint k = tid; k < n; k += 256) {
        const float2 e = partial[k];
        const float v = isnan(e.x) ? -INFINITY : e.x;
        if (v > b) {
            b = v;
            i = as_type<uint>(e.y);
        }
    }
    bv[tid] = b;
    bi[tid] = i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0) return;
    for (uint k = 1; k < 256; k++)
        if (bv[k] > b || (bv[k] == b && bi[k] < i)) {
            b = bv[k];
            i = bi[k];
        }
    draft[step] = i;
    prev[0] = i;
}

// out[v][r] = half(in[r][v]): the fold (lm_head . hidden_out) from the [R][V] product. Grid
// (ceil(V / 256), R) x 256.
kernel void ds_fold_t(device const float *in [[buffer(0)]], device half *out [[buffer(1)]], constant uint2 &p [[buffer(2)]] /* V, R */,
                      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint v = tg.x * 256 + tid, r = tg.y;
    if (v < p.x) out[ulong(v) * p.y + r] = half(in[ulong(r) * p.x + v]);
}

// Rows i < T of the sliding visits' rings at positions pos0 + i (row (pos0 + i) % ring), to or from
// a backup [planes][2][KMAX][row words]: before a verification chunk's commit, and back for
// session_roll_to. Grid (row words / 256, T, planes * 2) x 256.
struct DsRingP { uint words, pos0, ring, kmax, dir; };
kernel void ds_ring_copy(device uint *rings [[buffer(0)]], device uint *bak [[buffer(1)]], constant DsRingP &p [[buffer(2)]],
                         uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint w = tg.x * 256 + tid, i = tg.y, plane = tg.z;
    if (w >= p.words) return;
    const ulong r = ulong(plane) * p.ring + (p.pos0 + i) % p.ring, b = ulong(plane) * p.kmax + i;
    if (p.dir == 0) bak[b * p.words + w] = rings[r * p.words + w];
    else rings[r * p.words + w] = bak[b * p.words + w];
}
