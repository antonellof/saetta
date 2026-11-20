// GGUF import (see gguf.h). Formats from llama.cpp 9197945 (MIT, the ggml authors):
// ggml/include/gguf.h (file layout), ggml/src/ggml-common.h (blocks), ggml/src/ggml-quants.c
// dequantize_row_q8_0 / q4_K / q5_K / q6_K and get_scale_min_k4, conversion/qwen.py (the
// Qwen3.5 transforms: _reorder_v_heads, A_log, conv1d, norm +1, merges), gguf-py tensor names.
#include "gguf.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <unistd.h>

#include "json.h"
#include "metal/shim.h"

// ---- GGUF file ------------------------------------------------------------------------------

enum { GT_F32 = 0, GT_F16 = 1, GT_Q8_0 = 8, GT_Q4_K = 12, GT_Q5_K = 13, GT_Q6_K = 14, GT_BF16 = 30 };
#define QK_K 256

typedef struct {
    Str name;
    uint32_t nd;
    uint64_t ne[4]; // ne[0] innermost (the Hugging Face last dim)
    uint32_t type;
    const uint8_t *data;
} GTensor;

typedef struct {
    const uint8_t *map;
    size_t len;
    GTensor *t;
    size_t nt;
    Str arch;
    uint64_t block_count, embed, nextn;
} Gguf;

typedef struct {
    const uint8_t *p, *end;
    bool bad;
} Rd;

static uint64_t rd_n(Rd *r, size_t n) {
    if (r->bad || (size_t)(r->end - r->p) < n) {
        r->bad = true;
        return 0;
    }
    uint64_t v = 0;
    memcpy(&v, r->p, n); // little-endian host
    r->p += n;
    return v;
}
static Str rd_str(Rd *r) {
    uint64_t n = rd_n(r, 8);
    if (r->bad || (uint64_t)(r->end - r->p) < n) {
        r->bad = true;
        return str_n("", 0);
    }
    Str s = str_n((const char *)r->p, (size_t)n);
    r->p += n;
    return s;
}
static const size_t scalar_size[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};

// Skips (or reads, for the integer and string keys we need) one value of type t.
static void rd_skip(Rd *r, uint32_t t, uint64_t *as_int, Str *as_str) {
    if (t == 8) {
        Str s = rd_str(r);
        if (as_str) *as_str = s;
        return;
    }
    if (t == 9) {
        uint32_t et = (uint32_t)rd_n(r, 4);
        uint64_t n = rd_n(r, 8);
        if (et < 13 && et != 8 && et != 9) {
            uint64_t bytes = n * scalar_size[et];
            if ((uint64_t)(r->end - r->p) < bytes) r->bad = true;
            else r->p += bytes;
            return;
        }
        if (et != 8) { // arrays of arrays are not used by any GGUF writer: refuse (no recursion)
            r->bad = true;
            return;
        }
        for (uint64_t i = 0; i < n && !r->bad; i++) rd_str(r);
        return;
    }
    if (t >= 13 || scalar_size[t] == 0) {
        r->bad = true;
        return;
    }
    uint64_t v = rd_n(r, scalar_size[t]);
    if (as_int) *as_int = v;
}

static uint64_t row_bytes(uint32_t type, uint64_t n) {
    switch (type) {
    case GT_F32: return n * 4;
    case GT_F16:
    case GT_BF16: return n * 2;
    case GT_Q8_0: return n / 32 * 34;
    case GT_Q4_K: return n / QK_K * 144;
    case GT_Q5_K: return n / QK_K * 176;
    case GT_Q6_K: return n / QK_K * 210;
    default: return 0;
    }
}
static uint64_t blk_len(uint32_t type) {
    return type == GT_Q8_0 ? 32 : (type == GT_Q4_K || type == GT_Q5_K || type == GT_Q6_K) ? QK_K : 1;
}

static Err gguf_open(Arena *a, const char *path, Gguf *g) {
    *g = (Gguf){0};
    g->map = sx_map_file(path, &g->len);
    if (!g->map) return "MapFailed";
    Rd r = {g->map, g->map + g->len, false};
    if (g->len < 24 || memcmp(g->map, "GGUF", 4) != 0) return "GgufBadFile";
    r.p += 4;
    uint32_t ver = (uint32_t)rd_n(&r, 4);
    if (ver < 2) return "GgufBadFile";
    uint64_t nt = rd_n(&r, 8), nkv = rd_n(&r, 8);
    uint64_t align = 32;
    for (uint64_t i = 0; i < nkv && !r.bad; i++) {
        Str k = rd_str(&r);
        uint32_t t = (uint32_t)rd_n(&r, 4);
        uint64_t iv = 0;
        Str sv = {0};
        rd_skip(&r, t, &iv, &sv);
        if (str_eqz(k, "general.architecture")) g->arch = sv;
        else if (str_eqz(k, "general.alignment")) align = iv;
        else if (str_ends(k, S(".block_count"))) g->block_count = iv;
        else if (str_ends(k, S(".embedding_length"))) g->embed = iv;
        else if (str_ends(k, S(".nextn_predict_layers"))) g->nextn = iv;
    }
    if (r.bad || nt > 100000 || align == 0) return "GgufBadFile";
    g->t = ARENA_ARRAY(a, GTensor, nt ? nt : 1);
    g->nt = nt;
    uint64_t *offs = ARENA_ARRAY(a, uint64_t, nt ? nt : 1);
    for (uint64_t i = 0; i < nt && !r.bad; i++) {
        GTensor *t = &g->t[i];
        t->name = rd_str(&r);
        t->nd = (uint32_t)rd_n(&r, 4);
        if (t->nd == 0 || t->nd > 4) r.bad = true;
        for (uint32_t d = 0; d < 4; d++) t->ne[d] = d < t->nd ? rd_n(&r, 8) : 1;
        t->type = (uint32_t)rd_n(&r, 4);
        offs[i] = rd_n(&r, 8);
    }
    if (r.bad) return "GgufBadFile";
    if (align > 4096) return "GgufBadFile";
    uint64_t data = ALIGN_UP((uint64_t)(r.p - g->map), align);
    if (data > g->len) return "GgufBadFile";
    for (uint64_t i = 0; i < nt; i++) {
        GTensor *t = &g->t[i];
        t->data = NULL; // tensors of other types stay unreadable; the plan refuses them
        if (row_bytes(t->type, 1 << 8) == 0) continue;
        uint64_t rows = 1, total;
        for (uint32_t d = 0; d < 4; d++)
            if (t->ne[d] == 0 || t->ne[d] > (1ull << 32)) return "GgufBadFile";
        if (__builtin_mul_overflow(t->ne[1], t->ne[2], &rows) || __builtin_mul_overflow(rows, t->ne[3], &rows))
            return "GgufBadFile";
        if (t->ne[0] % blk_len(t->type) != 0) return "GgufBadFile";
        uint64_t rb = row_bytes(t->type, t->ne[0]);
        if (__builtin_mul_overflow(rows, rb, &total)) return "GgufBadFile";
        if (offs[i] > g->len - data || total > g->len - data - offs[i]) return "GgufBadFile";
        t->data = g->map + data + offs[i];
    }
    return OK;
}

static const GTensor *gget(const Gguf *g, const char *name) {
    for (size_t i = 0; i < g->nt; i++)
        if (str_eqz(g->t[i].name, name)) return &g->t[i];
    return NULL;
}

// ---- numbers ----------------------------------------------------------------------------------

static inline float f16_to_f32(const uint8_t *p) {
    uint16_t u;
    memcpy(&u, p, 2);
    _Float16 h;
    memcpy(&h, &u, 2);
    return (float)h;
}
static inline float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static inline uint16_t f32_to_bf16(float f) { // round to nearest even
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

static void scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

// One row of an exact-grid type as groups of 32: q[n], scale[n/32], bias[n/32].
static void row_groups(uint32_t type, const uint8_t *src, uint64_t n, uint8_t *q, float *s, float *b) {
    if (type == GT_Q8_0) {
        for (uint64_t k = 0; k < n / 32; k++, src += 34) {
            float d = f16_to_f32(src);
            for (int l = 0; l < 32; l++) q[k * 32 + l] = (uint8_t)((int)(int8_t)src[2 + l] + 128);
            s[k] = d;
            b[k] = -128.0f * d;
        }
        return;
    }
    bool five = type == GT_Q5_K;
    size_t bs = five ? 176 : 144;
    for (uint64_t k = 0; k < n / QK_K; k++, src += bs) {
        float d = f16_to_f32(src), dmin = f16_to_f32(src + 2);
        const uint8_t *sc = src + 4, *qh = src + 16, *qs = src + (five ? 48 : 16);
        for (int j = 0; j < 4; j++) {
            for (int h = 0; h < 2; h++) {
                int gi = 2 * j + h;
                uint8_t scv, mv;
                scale_min_k4(gi, sc, &scv, &mv);
                s[k * 8 + gi] = d * scv;
                b[k * 8 + gi] = -(dmin * mv);
                uint8_t *o = q + k * QK_K + gi * 32;
                for (int l = 0; l < 32; l++) {
                    uint8_t v = h ? qs[32 * j + l] >> 4 : qs[32 * j + l] & 0xF;
                    if (five && (qh[l] >> gi) & 1) v += 16;
                    o[l] = v;
                }
            }
        }
    }
}

// One row of any other type as f32 values.
static void row_values(uint32_t type, const uint8_t *src, uint64_t n, float *y) {
    switch (type) {
    case GT_F32: memcpy(y, src, n * 4); return;
    case GT_F16:
        for (uint64_t i = 0; i < n; i++) y[i] = f16_to_f32(src + 2 * i);
        return;
    case GT_BF16:
        for (uint64_t i = 0; i < n; i++) {
            uint16_t u;
            memcpy(&u, src + 2 * i, 2);
            y[i] = bf16_to_f32(u);
        }
        return;
    case GT_Q6_K:
        for (uint64_t k = 0; k < n / QK_K; k++, src += 210) {
            const uint8_t *ql = src, *qh = src + 128;
            const int8_t *sc = (const int8_t *)(src + 192);
            float d = f16_to_f32(src + 208);
            float *o = y + k * QK_K;
            for (int part = 0; part < 2; part++) {
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    int q1 = (int)(int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int)(int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int)(int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int)(int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    o[l] = d * sc[is] * q1;
                    o[l + 32] = d * sc[is + 2] * q2;
                    o[l + 64] = d * sc[is + 4] * q3;
                    o[l + 96] = d * sc[is + 6] * q4;
                }
                o += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        return;
    default: memset(y, 0, n * 4); return;
    }
}

// Min-max affine quantization of n values at (bits, group), scale and bias rounded to bf16
// first so q is chosen against the stored values.
static void quantize_values(const float *v, uint64_t n, uint32_t bits, uint32_t group, uint8_t *q, uint16_t *s,
                            uint16_t *b) {
    float qmax = (float)((1u << bits) - 1);
    for (uint64_t g = 0; g < n / group; g++) {
        const float *x = v + g * group;
        float lo = x[0], hi = x[0];
        for (uint32_t j = 1; j < group; j++) {
            lo = fminf(lo, x[j]);
            hi = fmaxf(hi, x[j]);
        }
        float sc = (hi - lo) / qmax;
        if (!(sc > 0)) sc = 1.0f;
        uint16_t sb = f32_to_bf16(sc), bb = f32_to_bf16(lo);
        float sr = bf16_to_f32(sb), br = bf16_to_f32(bb);
        if (!(sr > 0)) sr = 1.0f, sb = f32_to_bf16(1.0f);
        for (uint32_t j = 0; j < group; j++) {
            float t = rintf((x[j] - br) / sr);
            q[g * group + j] = (uint8_t)fminf(fmaxf(t, 0.0f), qmax);
        }
        s[g] = sb;
        b[g] = bb;
    }
}

// MLX packing: a little-endian bitstream per row, value j at bit j * bits.
static void pack_bits(const uint8_t *q, uint64_t n, uint32_t bits, uint32_t *w) {
    memset(w, 0, n * bits / 8);
    for (uint64_t j = 0; j < n; j++) {
        uint64_t bit = j * bits;
        uint32_t v = q[j], sh = (uint32_t)(bit % 32);
        w[bit / 32] |= v << sh;
        if (sh + bits > 32) w[bit / 32 + 1] |= v >> (32 - sh);
    }
}

// ---- the tensor plan ------------------------------------------------------------------------

typedef enum { P_NONE, P_VROWS, P_QKV_ROWS, P_HEADS, P_OUT_COLS } Perm;
typedef enum { V_COPY, V_ALOG, V_CONV, V_HEADS } VecOp;

typedef struct {
    const char *name; // module (quantized) or tensor (vector) name in the pack
    bool quant;
    const GTensor *src;
    uint64_t row0;      // first source row (split merged tensors)
    uint64_t out, in;   // HF shape [out, in]
    Perm perm;
    VecOp vop;
    uint32_t bits, group;
    bool exact;         // the source grid is kept (Q8_0, Q4_K, Q5_K)
    uint64_t off_w, off_s, off_b, nb_w, nb_s; // offsets in the data section
    const char *shape;  // vector shape, JSON
} Out;

typedef struct {
    uint64_t nk, nv, dk, dv; // linear attention heads (config.json)
} Heads;

// HF row r of a value-head-ordered axis -> GGUF row. HF order is grouped [nk][nv/nk][head],
// llama.cpp's tiled [nv/nk][nk][head] (conversion/qwen.py _reorder_v_heads).
static uint64_t vhead_src(const Heads *h, uint64_t r, uint64_t head) {
    uint64_t per = h->nv / h->nk;
    uint64_t hd = r / head, d = r % head;
    uint64_t k = hd / per, v = hd % per;
    return (v * h->nk + k) * head + d;
}

static uint64_t src_row(const Heads *h, const Out *o, uint64_t r) {
    switch (o->perm) {
    case P_VROWS: return vhead_src(h, r, h->dv);
    case P_HEADS: return vhead_src(h, r, 1);
    case P_QKV_ROWS: {
        uint64_t qk = 2 * h->nk * h->dk;
        return r < qk ? r : qk + vhead_src(h, r - qk, h->dv);
    }
    default: return r;
    }
}

typedef struct {
    const Gguf *g;
    const Heads *h;
    Out *outs;
    size_t n;
    int fd;
    uint64_t data0;
    _Atomic size_t next;
    _Atomic uint64_t done_bytes;
    _Atomic int failed;
} Job;

static Err pwrite_all(int fd, const void *p, size_t n, uint64_t off) {
    const uint8_t *c = p;
    while (n > 0) {
        ssize_t w = pwrite(fd, c, n, (off_t)off);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return err_errno(w < 0 ? errno : EIO);
        c += w;
        n -= (size_t)w;
        off += (uint64_t)w;
    }
    return OK;
}

#define CHUNK_ROWS 64

static Err convert_quant(Job *j, const Out *o, uint8_t *q, float *vals, float *cv, uint32_t *wbuf, uint16_t *sbuf,
                         uint16_t *bbuf, float *fs, float *fb) {
    const GTensor *t = o->src;
    uint64_t in = o->in, rb = row_bytes(t->type, t->ne[0]);
    uint64_t groups = in / o->group, wrow = in * o->bits / 8;
    for (uint64_t r0 = 0; r0 < o->out; r0 += CHUNK_ROWS) {
        uint64_t nr = MIN((uint64_t)CHUNK_ROWS, o->out - r0);
        for (uint64_t i = 0; i < nr; i++) {
            uint64_t r = r0 + i;
            const uint8_t *src = t->data + (o->row0 + src_row(j->h, o, r)) * rb;
            uint8_t *qr = q + i * in;
            uint16_t *sr = sbuf + i * groups, *br = bbuf + i * groups;
            if (o->exact) {
                row_groups(t->type, src, in, qr, fs, fb);
                if (o->perm == P_OUT_COLS) { // move whole 32-groups: HF column c <- GGUF column src(c)
                    memcpy(vals, qr, in);    // vals used as byte scratch here
                    uint8_t *tmp = (uint8_t *)vals;
                    float *ts = cv, *tb = cv + groups;
                    for (uint64_t g = 0; g < groups; g++) {
                        uint64_t sg = vhead_src(j->h, g * 32, j->h->dv) / 32;
                        memcpy(qr + g * 32, tmp + sg * 32, 32);
                        ts[g] = fs[sg];
                        tb[g] = fb[sg];
                    }
                    memcpy(fs, ts, groups * 4);
                    memcpy(fb, tb, groups * 4);
                }
                for (uint64_t g = 0; g < groups; g++) {
                    sr[g] = f32_to_bf16(fs[g]);
                    br[g] = f32_to_bf16(fb[g]);
                }
            } else {
                row_values(t->type, src, in, vals);
                const float *v = vals;
                if (o->perm == P_OUT_COLS) {
                    for (uint64_t c = 0; c < in; c++) cv[c] = vals[vhead_src(j->h, c, j->h->dv)];
                    v = cv;
                }
                quantize_values(v, in, o->bits, o->group, qr, sr, br);
            }
            pack_bits(qr, in, o->bits, wbuf + i * (wrow / 4));
        }
        TRY(pwrite_all(j->fd, wbuf, nr * wrow, j->data0 + o->off_w + r0 * wrow));
        TRY(pwrite_all(j->fd, sbuf, nr * groups * 2, j->data0 + o->off_s + r0 * groups * 2));
        TRY(pwrite_all(j->fd, bbuf, nr * groups * 2, j->data0 + o->off_b + r0 * groups * 2));
        atomic_fetch_add_explicit(&j->done_bytes, nr * rb, memory_order_relaxed);
    }
    return OK;
}

static Err convert_vec(Job *j, const Out *o, float *vals, float *cv) {
    const GTensor *t = o->src;
    uint64_t n = o->out * o->in; // total values
    uint64_t rows = t->ne[1] * t->ne[2] * t->ne[3], rlen = t->ne[0];
    for (uint64_t r = 0; r < rows; r++) row_values(t->type, t->data + r * row_bytes(t->type, rlen), rlen, vals + r * rlen);
    const Heads *h = j->h;
    switch (o->vop) {
    case V_COPY: memcpy(cv, vals, n * 4); break;
    case V_ALOG: // ssm_a = -exp(A_log), heads tiled
        for (uint64_t i = 0; i < n; i++) cv[i] = logf(-vals[vhead_src(h, i, 1)]);
        break;
    case V_HEADS:
        for (uint64_t i = 0; i < n; i++) cv[i] = vals[vhead_src(h, i, 1)];
        break;
    case V_CONV: { // [channels][taps]; value channels (after q and k) tiled
        uint64_t taps = o->in, qk = 2 * h->nk * h->dk;
        for (uint64_t c = 0; c < o->out; c++) {
            uint64_t sc = c < qk ? c : qk + vhead_src(h, c - qk, h->dv);
            memcpy(cv + c * taps, vals + sc * taps, taps * 4);
        }
        break;
    }
    }
    TRY(pwrite_all(j->fd, cv, n * 4, j->data0 + o->off_w));
    atomic_fetch_add_explicit(&j->done_bytes, n * 4, memory_order_relaxed);
    return OK;
}

static void *worker(void *p) {
    Job *j = p;
    uint64_t maxin = 0, maxv = 0;
    for (size_t i = 0; i < j->n; i++) {
        maxin = MAX(maxin, j->outs[i].in);
        if (!j->outs[i].quant) maxv = MAX(maxv, j->outs[i].out * j->outs[i].in);
    }
    uint64_t vlen = MAX(maxin * CHUNK_ROWS, maxv) + 64;
    uint8_t *q = xmalloc(maxin * CHUNK_ROWS);
    float *vals = xmalloc(vlen * 4), *cv = xmalloc(vlen * 4);
    uint32_t *w = xmalloc(maxin * CHUNK_ROWS + 64);
    uint16_t *s = xmalloc(maxin / 32 * CHUNK_ROWS * 2 + 64), *b = xmalloc(maxin / 32 * CHUNK_ROWS * 2 + 64);
    float *fs = xmalloc(maxin / 32 * 4 + 64), *fb = xmalloc(maxin / 32 * 4 + 64);
    for (;;) {
        size_t k = atomic_fetch_add_explicit(&j->next, 1, memory_order_acq_rel);
        if (k >= j->n || atomic_load_explicit(&j->failed, memory_order_acquire)) break;
        const Out *o = &j->outs[k];
        Err e = o->quant ? convert_quant(j, o, q, vals, cv, w, s, b, fs, fb) : convert_vec(j, o, vals, cv);
        if (e) {
            dbg("\nsaetta import: %s: %s\n", o->name, e);
            atomic_store_explicit(&j->failed, 1, memory_order_release);
        }
    }
    free(q);
    free(vals);
    free(cv);
    free(w);
    free(s);
    free(b);
    free(fs);
    free(fb);
    return NULL;
}

// ---- mapping GGUF names to the pack ---------------------------------------------------------

typedef struct {
    Arena *a;
    const Gguf *g;
    Out *v;
    size_t n, cap;
    Err err;
} Plan;

static Out *plan_add(Plan *p) {
    if (p->n == p->cap) {
        size_t cap = p->cap ? p->cap * 2 : 1024;
        Out *nv = ARENA_ARRAY(p->a, Out, cap);
        if (p->n) memcpy(nv, p->v, p->n * sizeof(Out));
        p->v = nv;
        p->cap = cap;
    }
    Out *o = &p->v[p->n++];
    *o = (Out){0};
    return o;
}

static const GTensor *need(Plan *p, const char *name) {
    const GTensor *t = gget(p->g, name);
    if (!t && !p->err) {
        dbg("saetta import: missing tensor %s\n", name);
        p->err = "GgufMissingTensor";
    }
    return t;
}

// A matrix: GGUF tensor t (rows [row0, row0 + out)) as module `mod`.
static void add_quant(Plan *p, const char *mod, const GTensor *t, uint64_t row0, uint64_t out, Perm perm) {
    if (!t || p->err) return;
    Out *o = plan_add(p);
    o->name = mod;
    o->quant = true;
    o->src = t;
    o->row0 = row0;
    o->in = t->ne[0];
    o->out = out ? out : t->ne[1];
    o->perm = perm;
    switch (t->type) {
    case GT_Q8_0: o->bits = 8, o->group = 32, o->exact = true; break;
    case GT_Q4_K: o->bits = 4, o->group = 32, o->exact = true; break;
    case GT_Q5_K: o->bits = 5, o->group = 32, o->exact = true; break;
    case GT_Q6_K: o->bits = 8, o->group = 32; break;
    case GT_F32:
    case GT_F16:
    case GT_BF16: o->bits = 8, o->group = 64; break;
    default:
        dbg("saetta import: %.*s has ggml type %u; supported: Q8_0, Q4_K, Q5_K, Q6_K, F16, BF16, F32\n",
            (int)t->name.len, t->name.ptr, t->type);
        p->err = "GgufUnsupportedType";
    }
    if (o->in % o->group != 0 && !p->err) p->err = "GgufNotQwen35";
}

static void add_vec(Plan *p, const char *name, const GTensor *t, VecOp op, uint64_t out, uint64_t in, const char *shape) {
    if (!t || p->err) return;
    if (t->type != GT_F32 && t->type != GT_F16 && t->type != GT_BF16) {
        p->err = "GgufUnsupportedType";
        return;
    }
    Out *o = plan_add(p);
    o->name = name;
    o->src = t;
    o->vop = op;
    o->out = out;
    o->in = in;
    o->shape = shape;
}

static uint64_t cfg_u(const JValue *tc, const char *k, Err *err) {
    const JValue *v = json_get(tc, k);
    if (!v || v->kind != J_INTEGER) {
        dbg("saetta import: config.json has no text_config.%s\n", k);
        *err = "GgufNotQwen35";
        return 0;
    }
    return (uint64_t)v->i;
}

static Err make_plan(Arena *a, const Gguf *g, const JValue *tc, Heads *h, Plan *p) {
    Err e = OK;
    uint64_t L = cfg_u(tc, "num_hidden_layers", &e), H = cfg_u(tc, "hidden_size", &e);
    uint64_t interval = cfg_u(tc, "full_attention_interval", &e);
    h->nk = cfg_u(tc, "linear_num_key_heads", &e);
    h->nv = cfg_u(tc, "linear_num_value_heads", &e);
    h->dk = cfg_u(tc, "linear_key_head_dim", &e);
    h->dv = cfg_u(tc, "linear_value_head_dim", &e);
    uint64_t taps = cfg_u(tc, "linear_conv_kernel_dim", &e);
    TRY(e);
    if (!str_eqz(g->arch, "qwen35") || g->embed != H || g->block_count != L + g->nextn || g->nextn > 1 || interval == 0 ||
        h->nk == 0 || h->nv % h->nk != 0) {
        dbg("saetta import: the GGUF is '%.*s' with %llu blocks of width %llu; expected qwen35, %llu layers (+ MTP) of "
            "width %llu\n",
            (int)g->arch.len, g->arch.ptr, (unsigned long long)g->block_count, (unsigned long long)g->embed,
            (unsigned long long)L, (unsigned long long)H);
        return "GgufNotQwen35";
    }
    *p = (Plan){.a = a, .g = g};
    const char *M = "language_model.model.";
#define GN(...) arena_printf(a, __VA_ARGS__)
    add_quant(p, GN("%sembed_tokens", M), need(p, "token_embd.weight"), 0, 0, P_NONE);
    add_quant(p, "language_model.lm_head", need(p, "output.weight"), 0, 0, P_NONE);
    add_vec(p, GN("%snorm.weight", M), need(p, "output_norm.weight"), V_COPY, 1, H, GN("[%llu]", (unsigned long long)H));
    uint64_t qkv_rows = 2 * h->nk * h->dk + h->nv * h->dv, z_rows = h->nv * h->dv;
    for (uint64_t i = 0; i < L + g->nextn && !p->err; i++) {
        bool mtp = i >= L;
        const char *pre = mtp ? "mtp.layers.0." : GN("%slayers.%llu.", M, (unsigned long long)i);
        bool full = mtp || (i + 1) % interval == 0;
#define B(s) GN("blk.%llu.%s", (unsigned long long)i, s)
#define T(s) need(p, B(s))
        add_vec(p, GN("%sinput_layernorm.weight", pre), T("attn_norm.weight"), V_COPY, 1, H, GN("[%llu]", (unsigned long long)H));
        add_vec(p, GN("%spost_attention_layernorm.weight", pre), T("post_attention_norm.weight"), V_COPY, 1, H,
                GN("[%llu]", (unsigned long long)H));
        if (full) {
            add_quant(p, GN("%sself_attn.q_proj", pre), T("attn_q.weight"), 0, 0, P_NONE);
            add_quant(p, GN("%sself_attn.k_proj", pre), T("attn_k.weight"), 0, 0, P_NONE);
            add_quant(p, GN("%sself_attn.v_proj", pre), T("attn_v.weight"), 0, 0, P_NONE);
            add_quant(p, GN("%sself_attn.o_proj", pre), T("attn_output.weight"), 0, 0, P_NONE);
            const GTensor *qn = T("attn_q_norm.weight"), *kn = T("attn_k_norm.weight");
            if (qn) add_vec(p, GN("%sself_attn.q_norm.weight", pre), qn, V_COPY, 1, qn->ne[0], GN("[%llu]", (unsigned long long)qn->ne[0]));
            if (kn) add_vec(p, GN("%sself_attn.k_norm.weight", pre), kn, V_COPY, 1, kn->ne[0], GN("[%llu]", (unsigned long long)kn->ne[0]));
        } else {
            const char *la = GN("%slinear_attn.", pre);
            const GTensor *qkvz = gget(g, B("attn_qkv_gate.weight"));
            if (qkvz) { // merged [qkv; z] rows
                add_quant(p, GN("%sin_proj_qkv", la), qkvz, 0, qkv_rows, P_QKV_ROWS);
                add_quant(p, GN("%sin_proj_z", la), qkvz, qkv_rows, z_rows, P_VROWS);
            } else {
                add_quant(p, GN("%sin_proj_qkv", la), T("attn_qkv.weight"), 0, 0, P_QKV_ROWS);
                add_quant(p, GN("%sin_proj_z", la), T("attn_gate.weight"), 0, 0, P_VROWS);
            }
            const GTensor *ba = gget(g, B("ssm_beta_alpha.weight"));
            if (ba) { // merged [beta; alpha] rows
                add_quant(p, GN("%sin_proj_b", la), ba, 0, h->nv, P_HEADS);
                add_quant(p, GN("%sin_proj_a", la), ba, h->nv, h->nv, P_HEADS);
            } else {
                add_quant(p, GN("%sin_proj_a", la), T("ssm_alpha.weight"), 0, 0, P_HEADS);
                add_quant(p, GN("%sin_proj_b", la), T("ssm_beta.weight"), 0, 0, P_HEADS);
            }
            add_quant(p, GN("%sout_proj", la), T("ssm_out.weight"), 0, 0, P_OUT_COLS);
            add_vec(p, GN("%sA_log", la), T("ssm_a"), V_ALOG, 1, h->nv, GN("[%llu]", (unsigned long long)h->nv));
            add_vec(p, GN("%sdt_bias", la), T("ssm_dt.bias"), V_HEADS, 1, h->nv, GN("[%llu]", (unsigned long long)h->nv));
            add_vec(p, GN("%sconv1d.weight", la), T("ssm_conv1d.weight"), V_CONV, qkv_rows, taps,
                    GN("[%llu,%llu,1]", (unsigned long long)qkv_rows, (unsigned long long)taps));
            add_vec(p, GN("%snorm.weight", la), T("ssm_norm.weight"), V_COPY, 1, h->dv, GN("[%llu]", (unsigned long long)h->dv));
        }
        add_quant(p, GN("%smlp.gate_proj", pre), T("ffn_gate.weight"), 0, 0, P_NONE);
        add_quant(p, GN("%smlp.up_proj", pre), T("ffn_up.weight"), 0, 0, P_NONE);
        add_quant(p, GN("%smlp.down_proj", pre), T("ffn_down.weight"), 0, 0, P_NONE);
        if (mtp) {
            add_quant(p, "mtp.fc", T("nextn.eh_proj.weight"), 0, 0, P_NONE);
            add_vec(p, "mtp.pre_fc_norm_embedding.weight", T("nextn.enorm.weight"), V_COPY, 1, H, GN("[%llu]", (unsigned long long)H));
            add_vec(p, "mtp.pre_fc_norm_hidden.weight", T("nextn.hnorm.weight"), V_COPY, 1, H, GN("[%llu]", (unsigned long long)H));
            add_vec(p, "mtp.norm.weight", T("nextn.shared_head_norm.weight"), V_COPY, 1, H, GN("[%llu]", (unsigned long long)H));
        }
#undef T
#undef B
    }
#undef GN
    if (p->err) return p->err;
    // shapes: every vector must hold out * in values
    for (size_t i = 0; i < p->n; i++) {
        Out *o = &p->v[i];
        uint64_t have = o->src->ne[0] * o->src->ne[1] * o->src->ne[2] * o->src->ne[3];
        if (!o->quant && have != o->out * o->in) return "GgufNotQwen35";
        if (!o->src->data) return "GgufUnsupportedType";
        if (!o->quant) continue;
        if (o->row0 + o->out > o->src->ne[1] || o->src->ne[2] != 1 || o->src->ne[3] != 1) return "GgufNotQwen35";
        // the permuted axes must have exactly the config's head layout
        uint64_t vd = h->nv * h->dv;
        if ((o->perm == P_QKV_ROWS && o->out != qkv_rows) || (o->perm == P_VROWS && o->out != vd) ||
            (o->perm == P_HEADS && o->out != h->nv) || (o->perm == P_OUT_COLS && (o->in != vd || h->dv % 32 != 0)))
            return "GgufNotQwen35";
    }
    return OK;
}

// ---- output ---------------------------------------------------------------------------------

static void hdr_entry(Buf *b, bool *first, const char *name, const char *dtype, const char *shape, uint64_t a, uint64_t z) {
    buf_adds(b, *first ? "{" : ",");
    *first = false;
    json_write_string(b, str(name));
    buf_printf(b, ":{\"dtype\":\"%s\",\"shape\":%s,\"data_offsets\":[%llu,%llu]}", dtype, shape, (unsigned long long)a,
               (unsigned long long)z);
}

static Err write_config(Arena *a, const char *like_dir, const char *out_dir, const Plan *p) {
    char *text;
    size_t len;
    const char *src = path_join(a, like_dir, "config.json");
    Err e = read_file(src, 1 << 20, &text, &len);
    if (e) {
        dbg("saetta import: cannot read %s\n", src);
        return e;
    }
    JValue *root;
    e = json_parse(a, str_n(text, len), false, &root);
    free(text);
    TRY(e);
    if (root->kind != J_OBJECT) return "GgufNotQwen35";
    JValue *q = json_new(a, J_OBJECT);
    json_put(a, q, S("group_size"), json_int(a, 32));
    json_put(a, q, S("bits"), json_int(a, 4));
    json_put(a, q, S("mode"), json_string(a, S("affine")));
    for (size_t i = 0; i < p->n; i++) {
        const Out *o = &p->v[i];
        if (!o->quant) continue;
        JValue *m = json_new(a, J_OBJECT);
        json_put(a, m, S("group_size"), json_int(a, o->group));
        json_put(a, m, S("bits"), json_int(a, o->bits));
        json_put(a, q, str(o->name), m);
    }
    json_put(a, root, S("quantization"), q);
    json_put(a, root, S("quantization_config"), q);
    Buf b = {0};
    json_write(&b, root);
    buf_addc(&b, '\n');
    e = write_file(path_join(a, out_dir, "config.json"), b.ptr, b.len);
    buf_free(&b);
    return e;
}

Err gguf_import(const char *gguf_path, const char *like_dir, const char *out_dir) {
    Arena a = {0};
    Err e = OK;
    Gguf g = {0};
    int fd = -1;
    const char *part = NULL;
    e = gguf_open(&a, gguf_path, &g);
    if (e) goto done;
    char *ctext;
    size_t clen;
    e = read_file(path_join(&a, like_dir, "config.json"), 1 << 20, &ctext, &clen);
    if (e) {
        dbg("saetta import: %s/config.json: %s\n", like_dir, e);
        goto done;
    }
    JValue *cfg;
    e = json_parse(&a, str_n(ctext, clen), true, &cfg);
    free(ctext);
    if (e) goto done;
    JValue *tc = json_get(cfg, "text_config");
    if (!tc) tc = cfg;
    Heads h;
    Plan p;
    e = make_plan(&a, &g, tc, &h, &p);
    if (e) goto done;

    // layout: weight (U32), scales and biases (BF16) per module, F32 vectors; contiguous
    uint64_t off = 0, in_bytes = 0;
    for (size_t i = 0; i < p.n; i++) {
        Out *o = &p.v[i];
        if (o->quant) {
            o->nb_w = o->out * o->in * o->bits / 8;
            o->nb_s = o->out * (o->in / o->group) * 2;
            o->off_w = off;
            o->off_s = off + o->nb_w;
            o->off_b = o->off_s + o->nb_s;
            off = o->off_b + o->nb_s;
            in_bytes += o->out * row_bytes(o->src->type, o->src->ne[0]);
        } else {
            o->nb_w = o->out * o->in * 4;
            o->off_w = off;
            off += o->nb_w;
            in_bytes += o->nb_w;
        }
    }
    Buf hb = {0};
    bool first = true;
    for (size_t i = 0; i < p.n; i++) {
        const Out *o = &p.v[i];
        if (o->quant) {
            char sw[64], ss[64];
            snprintf(sw, sizeof sw, "[%llu,%llu]", (unsigned long long)o->out, (unsigned long long)(o->in * o->bits / 32));
            snprintf(ss, sizeof ss, "[%llu,%llu]", (unsigned long long)o->out, (unsigned long long)(o->in / o->group));
            hdr_entry(&hb, &first, arena_printf(&a, "%s.weight", o->name), "U32", sw, o->off_w, o->off_w + o->nb_w);
            hdr_entry(&hb, &first, arena_printf(&a, "%s.scales", o->name), "BF16", ss, o->off_s, o->off_s + o->nb_s);
            hdr_entry(&hb, &first, arena_printf(&a, "%s.biases", o->name), "BF16", ss, o->off_b, o->off_b + o->nb_s);
        } else {
            hdr_entry(&hb, &first, o->name, "F32", o->shape, o->off_w, o->off_w + o->nb_w);
        }
    }
    buf_adds(&hb, ",\"__metadata__\":{\"format\":\"mlx\"}}");
    while ((8 + hb.len) % 64 != 0) buf_addc(&hb, ' '); // data starts 64-byte aligned
    e = mkdir_p(out_dir);
    if (e) {
        buf_free(&hb);
        goto done;
    }
    const char *st_path = path_join(&a, out_dir, "model.safetensors");
    part = arena_printf(&a, "%s.part", st_path);
    fd = open(part, O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) {
        e = err_errno(errno);
        buf_free(&hb);
        goto done;
    }
    uint64_t hl = hb.len;
    e = pwrite_all(fd, &hl, 8, 0);
    if (!e) e = pwrite_all(fd, hb.ptr, hb.len, 8);
    uint64_t data0 = 8 + hb.len;
    buf_free(&hb);
    if (e) goto done;
    if (ftruncate(fd, (off_t)(data0 + off)) != 0) {
        e = err_errno(errno);
        goto done;
    }

    dbg("saetta import: %zu tensors, %.1f GB -> %.1f GB\n", p.n, (double)in_bytes / 1e9, (double)off / 1e9);
    Job job = {.g = &g, .h = &h, .outs = p.v, .n = p.n, .fd = fd, .data0 = data0};
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    size_t nth = (size_t)CLAMP(ncpu > 0 ? ncpu : 4, 1, 12);
    pthread_t th[12];
    for (size_t i = 0; i < nth; i++)
        if (pthread_create(&th[i], NULL, worker, &job) != 0) die("saetta import: cannot start a thread\n");
    uint64_t t0 = now_ns();
    for (;;) {
        bool all = true;
        if (atomic_load_explicit(&job.next, memory_order_acquire) < p.n + nth &&
            !atomic_load_explicit(&job.failed, memory_order_acquire))
            all = false;
        uint64_t done = atomic_load_explicit(&job.done_bytes, memory_order_relaxed);
        double s = (double)(now_ns() - t0) / 1e9;
        dbg("\r  converted %.1f / %.1f GB  (%.0f s)", (double)done / 1e9, (double)in_bytes / 1e9, s);
        if (all) break;
        usleep(500000);
    }
    for (size_t i = 0; i < nth; i++) pthread_join(th[i], NULL);
    dbg("\n");
    if (atomic_load_explicit(&job.failed, memory_order_acquire)) {
        e = "ImportFailed";
        goto done;
    }
    if (fcntl(fd, F_FULLFSYNC) != 0 && fsync(fd) != 0) {
        e = err_errno(errno);
        goto done;
    }
    close(fd);
    fd = -1;
    {
        Buf ib = {0};
        buf_adds(&ib, "{\"metadata\":{\"total_size\":");
        buf_printf(&ib, "%llu", (unsigned long long)off);
        buf_adds(&ib, "},\"weight_map\":{");
        bool f2 = true;
        for (size_t i = 0; i < p.n; i++) {
            const Out *o = &p.v[i];
            const char *suf[3] = {".weight", ".scales", ".biases"};
            for (int k = 0; k < (o->quant ? 3 : 1); k++) {
                if (!f2) buf_addc(&ib, ',');
                f2 = false;
                json_write_string(&ib, str(o->quant ? arena_printf(&a, "%s%s", o->name, suf[k]) : o->name));
                buf_adds(&ib, ":\"model.safetensors\"");
            }
        }
        buf_adds(&ib, "}}\n");
        e = write_file(path_join(&a, out_dir, "model.safetensors.index.json"), ib.ptr, ib.len);
        buf_free(&ib);
        if (e) goto done;
    }
    e = write_config(&a, like_dir, out_dir, &p);
    // last: model.safetensors in place marks a complete import (saetta download checks it)
    if (!e && rename(part, st_path) != 0) e = err_errno(errno);
done:
    if (fd >= 0) close(fd);
    if (e && part) unlink(part);
    if (g.map) sx_unmap_file(g.map, g.len);
    arena_reset(&a);
    return e;
}

Err gguf_import_run(const char *const *args, size_t n) {
    const char *in = NULL, *out = NULL, *like = NULL;
    for (size_t i = 0; i < n; i++) {
        if (strcmp(args[i], "--like") == 0 && i + 1 < n) like = args[++i];
        else if (!in) in = args[i];
        else if (!out) out = args[i];
        else return "Usage";
    }
    if (!in || !out) {
        dbg("usage: saetta import <file.gguf> <out dir> [--like <dir with the model's config.json>]\n"
            "  Qwen3.8-27B GGUF (llama.cpp qwen35; Q8_0, Q4_K, Q5_K, Q6_K, F16, BF16) -> a saetta pack.\n"
            "  config.json comes from --like (default: the out dir); copy the tokenizer files there too.\n");
        return "Usage";
    }
    return gguf_import(in, like ? like : out, out);
}
