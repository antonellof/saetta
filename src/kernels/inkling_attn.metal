// Inkling 2 Mini attention (M:546-727 of transformers be48e7a modeling_inkling2_loop.py):
// scores q.k / sqrt(HD) + the relative bias rs . proj[:, distance] (0 at distances >= EXT,
// M:239-270), causal, sliding layers masked to distances < win, one sink logit per head with a
// zero value (M:132-135), output times sigmoid(gate) (M:531-543). Compiled after
// inkling_ops.metal. Scalar reference: src/inkling_ref.c (ink_ref_attn).
//
// Keys of positions below pos0 come from the cache (a ring of `ring` rows on sliding layers: row
// position % ring, else row position), those from pos0 on from the chunk's own rows kn / vn
// (f32): the chunk is written to the cache only after its attention (ink_kv_store), so a chunk
// longer than the window never reads a row a later token of the same chunk overwrote. The cache
// is f32 or f16 (KT; the _h kernels). Element offsets stay below 2^32: at most 2^20 positions of
// NKV * HD = 1024 values (static_assert in src/inkling.c).
//
// Each threadgroup serves one KV head and its NH / NKV (3) query heads, so the head's keys and
// values are read once for the three.

struct AttP { uint NH, NKV, HD, DREL, EXT, pos0, win, ring, nsplit, span, T; float scale; };

inline float4 ik_ld4(device const float *p) { return *(device const float4 *)p; }
inline float4 ik_ld4(device const half *p) { return float4(*(device const half4 *)p); }

// ---- decode and verification chunks: split-K ---------------------------------------------------
// Grid (NKV, nsplit, T) x 128 (4 simdgroups; lane l holds dims 4l..4l+3). Split s of query t
// covers keys [lo + s * span, lo + (s + 1) * span) of its range [lo, pos]; for each of the 3 query
// heads it writes (acc[HD], max, sum) of its online softmax to part[((t * NH + h) * nsplit + s) * (HD + 2)].
#define IK_QPK 3
template <typename KT>
inline void ik_attn_part(device const float *q, device const float *rs, device const float *proj, device const KT *kc,
                         device const KT *vc, device const float *kn, device const float *vn, device float *part,
                         constant AttP &p, uint3 tg, uint sg, uint lane, uint tid, threadgroup float (*tm)[IK_QPK],
                         threadgroup float (*tl)[IK_QPK], threadgroup float4 (*tacc)[IK_QPK][32]) {
    const uint g = tg.x, s = tg.y, t = tg.z, h0 = g * IK_QPK;
    const int pq = int(p.pos0 + t);
    const int lo = p.win ? max(0, pq - int(p.win) + 1) : 0;
    // this query's own splits, as a decode of it alone takes them (256 keys a split, at most
    // p.nsplit); the grid's extra splits stay empty
    const uint n_q = uint(pq - lo + 1), ns_q = min(p.nsplit, max(1u, (n_q + 255) / 256)), span_q = (n_q + ns_q - 1) / ns_q;
    const int k0 = s < ns_q ? lo + int(s * span_q) : pq + 1, k1 = min(k0 + int(span_q), pq + 1);
    float4 qv[IK_QPK], acc[IK_QPK];
    float rl[IK_QPK], m[IK_QPK], l[IK_QPK];
    for (uint i = 0; i < IK_QPK; i++) {
        qv[i] = ((device const float4 *)(q + (t * p.NH + h0 + i) * p.HD))[lane] * p.scale;
        rl[i] = lane < p.DREL ? rs[(t * p.NH + h0 + i) * p.DREL + lane] : 0.0f;
        m[i] = -INFINITY;
        l[i] = 0.0f;
        acc[i] = 0.0f;
    }
    const uint ld = p.NKV * p.HD;
    for (int j = k0 + int(sg); j < k1; j += 4) {
        float4 kv4, vv4;
        if (j >= int(p.pos0)) {
            // the chunk's own rows; earlier ones rounded to the cache's type, as a decode of this
            // query would read them from the cache
            const uint o = uint(j - int(p.pos0)) * ld + g * p.HD + lane * 4;
            kv4 = ik_ld4(kn + o);
            vv4 = ik_ld4(vn + o);
            if (j < pq) {
                kv4 = float4(vec<KT, 4>(kv4));
                vv4 = float4(vec<KT, 4>(vv4));
            }
        } else {
            const uint o = (p.ring ? uint(j) % p.ring : uint(j)) * ld + g * p.HD + lane * 4;
            kv4 = ik_ld4(kc + o);
            vv4 = ik_ld4(vc + o);
        }
        const uint d = uint(pq - j);
        const float pr = lane < p.DREL && d < p.EXT ? proj[lane * p.EXT + d] : 0.0f;
        for (uint i = 0; i < IK_QPK; i++) {
            const float sc = simd_sum(dot(qv[i], kv4) + rl[i] * pr);
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

#define IK_PART_KERNEL(NAME, KT)                                                                                          \
    kernel void NAME(device const float *q [[buffer(0)]], device const float *rs [[buffer(1)]],                              \
                     device const float *proj [[buffer(2)]], device const KT *kc [[buffer(3)]], device const KT *vc [[buffer(4)]], \
                     device const float *kn [[buffer(5)]], device const float *vn [[buffer(6)]], device float *part [[buffer(7)]], \
                     constant AttP &p [[buffer(8)]], uint3 tg [[threadgroup_position_in_grid]],                              \
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                    \
                     uint tid [[thread_index_in_threadgroup]]) {                                                             \
        threadgroup float tm[4][IK_QPK], tl[4][IK_QPK];                                                                      \
        threadgroup float4 tacc[4][IK_QPK][32];                                                                              \
        ik_attn_part<KT>(q, rs, proj, kc, vc, kn, vn, part, p, tg, sg, lane, tid, tm, tl, tacc);                             \
    }
IK_PART_KERNEL(ink_attn_part, float)
IK_PART_KERNEL(ink_attn_part_h, half)

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

// ---- prompt chunks: tiles of keys shared by a block of queries (flash style) -------------------
// Grid (NKV, ceil(T / IK_BQ)) x 256 (8 simdgroups). A threadgroup takes IK_BQ queries of one KV
// head, so IK_BQ * 3 rows (query, head), each simdgroup IK_ROWS of them; it walks the keys of
// the block's range [lo(first query), last query] in tiles of 32 (K transposed and V in
// threadgroup memory, read once for every row). In a tile lane l holds key l for the scores and
// dims 4l..4l+3 for the output; the rows keep their online softmax in registers. The sink and the
// gate are applied at the end; the output goes to out[t][h][HD].
#define IK_BQ 16
#define IK_BK 32
#define IK_ROWS (IK_BQ * IK_QPK / 8)
template <typename KT>
inline void ik_attn_flash(device const float *q, device const float *rs, device const float *proj, device const KT *kc,
                          device const KT *vc, device const float *kn, device const float *vn, device const float *sink,
                          device const float *gate, device float *out, constant AttP &p, uint2 tg, uint sg, uint lane,
                          uint tid, threadgroup float (*kt)[IK_BK], threadgroup float4 (*vt)[32]) {
    const uint g = tg.x, q0 = tg.y * IK_BQ, ld = p.NKV * p.HD;
    const uint nq = min(uint(IK_BQ), p.T - q0);
    const int pfirst = int(p.pos0 + q0), plast = int(p.pos0 + q0 + nq - 1);
    const int lo = p.win ? max(0, pfirst - int(p.win) + 1) : 0;
    float m[IK_ROWS], l[IK_ROWS];
    float4 acc[IK_ROWS];
    for (uint r = 0; r < IK_ROWS; r++) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
        acc[r] = 0.0f;
    }
    for (int j0 = lo; j0 <= plast; j0 += IK_BK) {
        // the tile: 32 keys x 128 dims of K (transposed) and V, 16 values each per thread
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < IK_BK * 32; e += 256) {
            const uint kk = e / 32, d4 = e % 32;
            const int j = j0 + int(kk);
            float4 kv4 = 0.0f, vv4 = 0.0f;
            if (j <= plast) {
                if (j >= int(p.pos0)) {
                    const uint o = uint(j - int(p.pos0)) * ld + g * p.HD + d4 * 4;
                    kv4 = ik_ld4(kn + o);
                    vv4 = ik_ld4(vn + o);
                } else {
                    const uint o = (p.ring ? uint(j) % p.ring : uint(j)) * ld + g * p.HD + d4 * 4;
                    kv4 = ik_ld4(kc + o);
                    vv4 = ik_ld4(vc + o);
                }
            }
            kt[d4 * 4 + 0][kk] = kv4.x;
            kt[d4 * 4 + 1][kk] = kv4.y;
            kt[d4 * 4 + 2][kk] = kv4.z;
            kt[d4 * 4 + 3][kk] = kv4.w;
            vt[kk][d4] = vv4;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int j = j0 + int(lane);
        for (uint r = 0; r < IK_ROWS; r++) {
            const uint row = sg * IK_ROWS + r, qi = row / IK_QPK, h = g * IK_QPK + row % IK_QPK;
            if (qi >= nq) continue;
            const int pq = int(p.pos0 + q0 + qi);
            const int qlo = p.win ? max(0, pq - int(p.win) + 1) : 0;
            if (j0 > pq || j0 + IK_BK <= qlo) continue; // the whole tile outside this row's range
            const uint t = q0 + qi;
            device const float *qr = q + (t * p.NH + h) * p.HD;
            float s = -INFINITY;
            if (j <= pq && j >= qlo) {
                float a = 0.0f;
                for (uint d = 0; d < 128; d++) a += qr[d] * kt[d][lane];
                a *= p.scale;
                const uint dist = uint(pq - j);
                if (dist < p.EXT) {
                    device const float *rr = rs + (t * p.NH + h) * p.DREL;
                    for (uint i = 0; i < p.DREL; i++) a += rr[i] * proj[i * p.EXT + dist];
                }
                s = a;
            }
            const float mt = simd_max(s);
            if (mt == -INFINITY) continue;
            const float mn = max(m[r], mt);
            const float c = exp(m[r] - mn), e = s == -INFINITY ? 0.0f : exp(s - mn);
            l[r] = l[r] * c + simd_sum(e);
            float4 a4 = acc[r] * c;
            for (uint kk = 0; kk < IK_BK; kk++) a4 += simd_shuffle(e, kk) * vt[kk][lane];
            acc[r] = a4;
            m[r] = mn;
        }
    }
    for (uint r = 0; r < IK_ROWS; r++) {
        const uint row = sg * IK_ROWS + r, qi = row / IK_QPK, h = g * IK_QPK + row % IK_QPK;
        if (qi >= nq) continue;
        const uint t = q0 + qi;
        const float sk = sink[t * p.NH + h];
        const float M = max(m[r], sk);
        const float L = l[r] * exp(m[r] - M) + exp(sk - M);
        const float4 o = acc[r] * (exp(m[r] - M) / L);
        const uint ob = (t * p.NH + h) * p.HD + lane * 4;
        const float4 gt = *(device const float4 *)(gate + ob);
        *(device float4 *)(out + ob) = o * float4(ik_sigm(gt.x), ik_sigm(gt.y), ik_sigm(gt.z), ik_sigm(gt.w));
    }
}

#define IK_FLASH_KERNEL(NAME, KT)                                                                                          \
    kernel void NAME(device const float *q [[buffer(0)]], device const float *rs [[buffer(1)]],                               \
                     device const float *proj [[buffer(2)]], device const KT *kc [[buffer(3)]], device const KT *vc [[buffer(4)]], \
                     device const float *kn [[buffer(5)]], device const float *vn [[buffer(6)]], device const float *sink [[buffer(7)]], \
                     device const float *gate [[buffer(8)]], device float *out [[buffer(9)]], constant AttP &p [[buffer(10)]],     \
                     uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                    \
                     uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {                      \
        threadgroup float kt[128][IK_BK];                                                                                     \
        threadgroup float4 vt[IK_BK][32];                                                                                     \
        ik_attn_flash<KT>(q, rs, proj, kc, vc, kn, vn, sink, gate, out, p, tg, sg, lane, tid, kt, vt);                        \
    }
IK_FLASH_KERNEL(ink_attn_flash, float)

// ---- prompt chunks on an f16 cache: the same tiles with simdgroup matrices ----------------------
// Grid (NKV, ceil(T / 16)) x 192 (6 simdgroups). The threadgroup's 48 rows (16 queries x 3 heads,
// row = query * 3 + head) are 6 row blocks of 8, one per simdgroup; Q (times the scale) is held
// as half 8x8 matrices, the scores of a tile of 32 keys come out of S = Q K^T in f32, take the
// bias and the mask, and the online softmax runs per row (4 lanes a row, 8 columns each); P goes
// back as half for O += P V (f32 accumulators, rescaled through a diagonal matrix). K and V tiles
// are half in threadgroup memory. Half Q, K, V and P: the accelerated path, checked by KL.
#define IKM_BQ 16
#define IKM_BK 32
kernel void ink_attn_flash_mma(device const float *q [[buffer(0)]], device const float *rs [[buffer(1)]],
                               device const float *proj [[buffer(2)]], device const half *kc [[buffer(3)]],
                               device const half *vc [[buffer(4)]], device const float *kn [[buffer(5)]],
                               device const float *vn [[buffer(6)]], device const float *sink [[buffer(7)]],
                               device const float *gate [[buffer(8)]], device float *out [[buffer(9)]], constant AttP &p [[buffer(10)]],
                               uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half kt[IKM_BK][128];      // K tile, rows = keys
    threadgroup half vt[IKM_BK][128];      // V tile
    threadgroup float sbuf[6][8][IKM_BK];  // per simdgroup: its scores
    threadgroup half pbuf[6][8][IKM_BK];   // per simdgroup: its probabilities
    threadgroup float dbuf[6][8][8];       // per simdgroup: diag(rescale)
    const uint g = tg.x, q0 = tg.y * IKM_BQ, ld = p.NKV * p.HD;
    const uint nq = min(uint(IKM_BQ), p.T - q0);
    const int pfirst = int(p.pos0 + q0), plast = int(p.pos0 + q0 + nq - 1);
    const int lo = p.win ? max(0, pfirst - int(p.win) + 1) : 0;
    // this lane's row and columns in the softmax layout
    const uint lr = lane / 4, lc0 = (lane % 4) * 8;
    const uint row = sg * 8 + lr, qi = row / 3, hh = row % 3, h = g * 3 + hh, t = q0 + qi;
    const bool rvalid = qi < nq;
    const int pq = int(p.pos0 + t);
    const int qlo = p.win ? max(0, pq - int(p.win) + 1) : 0;
    float rr[16];
    for (uint i = 0; i < 16; i++) rr[i] = rvalid && i < p.DREL ? rs[(t * p.NH + h) * p.DREL + i] : 0.0f;
    // Q of the row block, scaled, as 16 half matrices: staged through kt, 24 rows at a time
    simdgroup_half8x8 qm[16];
    for (uint pass = 0; pass < 2; pass++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < 24 * 128; e += 192) {
            const uint r = pass * 24 + e / 128, d = e % 128, rq = r / 3, rh = r % 3;
            const float v = rq < nq ? q[((q0 + rq) * p.NH + g * 3 + rh) * p.HD + d] * p.scale : 0.0f;
            kt[e / 128][d] = half(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg / 3 == pass)
            for (uint kb = 0; kb < 16; kb++) simdgroup_load(qm[kb], &kt[(sg % 3) * 8][kb * 8], 128);
    }
    float m = -INFINITY, l = 0.0f;
    simdgroup_float8x8 om[16];
    for (uint cb = 0; cb < 16; cb++) om[cb] = simdgroup_float8x8(0.0f);
    for (int j0 = lo; j0 <= plast; j0 += IKM_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < IKM_BK * 32; e += 192) {
            const uint kk = e / 32, d4 = e % 32;
            const int j = j0 + int(kk);
            half4 kv4 = half4(0.0h), vv4 = half4(0.0h);
            if (j <= plast) {
                if (j >= int(p.pos0)) {
                    const uint o = uint(j - int(p.pos0)) * ld + g * p.HD + d4 * 4;
                    kv4 = half4(*(device const float4 *)(kn + o));
                    vv4 = half4(*(device const float4 *)(vn + o));
                } else {
                    const uint o = (p.ring ? uint(j) % p.ring : uint(j)) * ld + g * p.HD + d4 * 4;
                    kv4 = *(device const half4 *)(kc + o);
                    vv4 = *(device const half4 *)(vc + o);
                }
            }
            *(threadgroup half4 *)&kt[kk][d4 * 4] = kv4;
            *(threadgroup half4 *)&vt[kk][d4 * 4] = vv4;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for this row block: 4 column blocks of 8 keys
        for (uint cb = 0; cb < 4; cb++) {
            simdgroup_float8x8 sm = simdgroup_float8x8(0.0f);
            for (uint kb = 0; kb < 16; kb++) {
                simdgroup_half8x8 km;
                simdgroup_load(km, &kt[cb * 8][kb * 8], 128, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sm, qm[kb], km, sm);
            }
            simdgroup_store(sm, &sbuf[sg][0][cb * 8], IKM_BK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // bias, mask, online softmax: lane (lr, lc0..lc0+7)
        float sv[8], mt = -INFINITY;
        for (uint c = 0; c < 8; c++) {
            const int j = j0 + int(lc0 + c);
            float s = -INFINITY;
            if (rvalid && j <= pq && j >= qlo) {
                s = sbuf[sg][lr][lc0 + c];
                const uint dist = uint(pq - j);
                if (dist < p.EXT)
                    for (uint i = 0; i < p.DREL; i++) s += rr[i] * proj[i * p.EXT + dist];
            }
            sv[c] = s;
            mt = max(mt, s);
        }
        mt = max(mt, simd_shuffle_xor(mt, 1));
        mt = max(mt, simd_shuffle_xor(mt, 2));
        const float mn = max(m, mt);
        const float cfac = mn == -INFINITY ? 1.0f : exp(m - mn);
        float ps = 0.0f;
        for (uint c = 0; c < 8; c++) {
            const float e = sv[c] == -INFINITY ? 0.0f : exp(sv[c] - mn);
            ps += e;
            pbuf[sg][lr][lc0 + c] = half(e);
        }
        ps += simd_shuffle_xor(ps, 1);
        ps += simd_shuffle_xor(ps, 2);
        l = l * cfac + ps;
        m = mn;
        // diag(cfac) for the rescale of O
        if (lane % 4 == 0)
            for (uint c = 0; c < 8; c++) dbuf[sg][lr][c] = c == lr ? cfac : 0.0f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 dm;
        simdgroup_load(dm, &dbuf[sg][0][0], 8);
        simdgroup_half8x8 pm[4];
        for (uint kb = 0; kb < 4; kb++) simdgroup_load(pm[kb], &pbuf[sg][0][kb * 8], IKM_BK);
        for (uint cb = 0; cb < 16; cb++) {
            simdgroup_multiply(om[cb], dm, om[cb]);
            for (uint kb = 0; kb < 4; kb++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, &vt[kb * 8][cb * 8], 128);
                simdgroup_multiply_accumulate(om[cb], pm[kb], vm, om[cb]);
            }
        }
    }
    // the sink and the gate, through sbuf in four parts of 32 columns
    const float sk = rvalid ? sink[t * p.NH + h] : 0.0f;
    const float M = max(m, sk);
    const float fin = rvalid ? (m == -INFINITY ? 0.0f : exp(m - M)) / (l * (m == -INFINITY ? 0.0f : exp(m - M)) + exp(sk - M)) : 0.0f;
    for (uint part_ = 0; part_ < 4; part_++) {
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint cb = 0; cb < 4; cb++) simdgroup_store(om[part_ * 4 + cb], &sbuf[sg][0][cb * 8], IKM_BK);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (rvalid)
            for (uint c = lc0; c < lc0 + 8; c++) {
                const uint o = (t * p.NH + h) * p.HD + part_ * 32 + c;
                out[o] = sbuf[sg][lr][c] * fin * ik_sigm(gate[o]);
            }
    }
}
