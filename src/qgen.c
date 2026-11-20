// Run-time quantization kernels: sources, reference helpers and qgen-bench (see qgen.h).
#include "qgen.h"

// Zig's {d:.N} for the bench lines (ties round as Zig does, not as printf).
static const char *fx(char *buf, size_t cap, double x, int prec) {
    fmt_fixed(buf, cap, x, prec);
    return buf;
}
#define FX(x, prec) fx((char[64]){0}, 64, (x), (prec))

#include <assert.h>
#include <math.h>

#include "attn27.h"
#include "dsv4_moe.h"
#include "ops.h"

const char qgen_src[] = {
#embed "kernels/qgen_common.metal"
    , '\n',
#embed "kernels/qgen.metal"
    , 0};
const char qgen_mpp_src[] = {
#embed "kernels/qgen_common.metal"
    , '\n',
#embed "kernels/qgen_mpp.metal"
    , 0};
static const char qmatvec_src[] = {
#embed "kernels/qmatvec.metal"
    , 0};
static const char qmm_mpp_src[] = {
#embed "kernels/qmm_mpp.metal"
    , 0};

uint32_t qgen_unpack(const uint8_t *row, size_t row_len, size_t i, size_t bits) {
    size_t bit = i * bits;
    uint32_t lo = row[bit / 8];
    uint32_t hi = bit / 8 + 1 < row_len ? row[bit / 8 + 1] : 0;
    return ((lo | (hi << 8)) >> (bit % 8)) & ((1u << bits) - 1);
}

static inline float bf16(const uint8_t *b, size_t i) {
    return ops_bf16_to_f32((uint16_t)(b[i * 2] | (b[i * 2 + 1] << 8)));
}

// ---- PRNG ------------------------------------------------------------------------------------

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

void qgen_prng_init(QgenPrng *r, uint64_t seed) {
    uint64_t s = seed; // SplitMix64
    for (int i = 0; i < 4; i++) {
        s += 0x9e3779b97f4a7c15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        r->s[i] = z ^ (z >> 31);
    }
}

uint64_t qgen_prng_next(QgenPrng *r) {
    uint64_t *s = r->s;
    uint64_t res = rotl(s[0] + s[3], 23) + s[0];
    uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return res;
}

void qgen_prng_bytes(QgenPrng *r, uint8_t *buf, size_t n) {
    size_t i = 0, aligned = n - (n & 7);
    for (; i < aligned; i += 8) {
        uint64_t v = qgen_prng_next(r);
        for (int j = 0; j < 8; j++, v >>= 8) buf[i + j] = (uint8_t)v;
    }
    if (i != n) { // the rest cuts the stream
        uint64_t v = qgen_prng_next(r);
        for (; i < n; i++, v >>= 8) buf[i] = (uint8_t)v;
    }
}

static inline unsigned clz64(uint64_t x) { return x ? (unsigned)__builtin_clzll(x) : 64; }
static inline unsigned clz32(uint32_t x) { return x ? (unsigned)__builtin_clz(x) : 32; }

float qgen_prng_float(QgenPrng *r) {
    uint64_t rand = qgen_prng_next(r);
    unsigned lz = clz64(rand);
    if (lz >= 41) {
        lz = 41 + clz64(qgen_prng_next(r));
        if (lz == 41 + 64) lz += clz32((uint32_t)qgen_prng_next(r) | 0x7FF);
    }
    uint32_t bits = ((uint32_t)(126 - lz) << 23) | (uint32_t)(rand & 0x7FFFFF);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

bool qgen_prng_boolean(QgenPrng *r) { return (qgen_prng_next(r) & 1) != 0; }

// ---- packed matrices -------------------------------------------------------------------------

size_t qgen_mat_w_stride(const QgenMat *m) { return m->out * m->in * m->bits / 8; }
size_t qgen_mat_s_stride(const QgenMat *m) { return m->out * (m->in / m->group) * 2; }

uint8_t *qgen_mat_bytes(const QgenMat *m, int k) { return (uint8_t *)sx_buffer_contents(m->buf[k]) + m->sh[k]; }

void qgen_mat_row(const QgenMat *m, size_t e, size_t r, float *out) {
    size_t rb = m->in * m->bits / 8;
    const uint8_t *w = qgen_mat_bytes(m, 0) + e * qgen_mat_w_stride(m) + r * rb;
    size_t groups = m->in / m->group;
    const uint8_t *s = qgen_mat_bytes(m, 1) + e * qgen_mat_s_stride(m);
    const uint8_t *b = qgen_mat_bytes(m, 2) + e * qgen_mat_s_stride(m);
    for (size_t i = 0; i < m->in; i++) {
        size_t g = r * groups + i / m->group;
        out[i] = (float)qgen_unpack(w, rb, i, m->bits) * bf16(s, g) + bf16(b, g);
    }
}

void qgen_mat_row_half(const QgenMat *m, size_t e, size_t r, float *out) {
    size_t rb = m->in * m->bits / 8;
    const uint8_t *w = qgen_mat_bytes(m, 0) + e * qgen_mat_w_stride(m) + r * rb;
    size_t groups = m->in / m->group;
    const uint8_t *s = qgen_mat_bytes(m, 1) + e * qgen_mat_s_stride(m);
    const uint8_t *b = qgen_mat_bytes(m, 2) + e * qgen_mat_s_stride(m);
    for (size_t i = 0; i < m->in; i++) {
        size_t g = r * groups + i / m->group;
        _Float16 q = (_Float16)qgen_unpack(w, rb, i, m->bits);
        _Float16 sh = (_Float16)bf16(s, g);
        _Float16 bh = (_Float16)bf16(b, g);
        out[i] = (float)__builtin_fmaf16(q, sh, bh);
    }
}

void qgen_mat_args(const QgenMat *m, sx_arg out[3]) {
    for (int k = 0; k < 3; k++) out[k] = gpu_buf_arg(m->buf[k]);
}

void qgen_mat_deinit(const QgenMat *m) {
    for (int k = 0; k < 3; k++) sx_buffer_release(m->buf[k]);
}

// f32 to bf16 by truncation (the random scales and biases).
static uint16_t to_bf16_trunc(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}

static inline void wr16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }

static Err new_mat(Gpu *g, QgenPrng *rng, size_t E, size_t out, size_t in, uint32_t bits, uint32_t group,
                   const uint32_t offs[3], QgenMat *mo) {
    assert(in % group == 0 && in % 32 == 0);
    QgenMat m = {.E = E, .out = out, .in = in, .bits = bits, .group = group, .sh = {offs[0], offs[1], offs[2]}};
    for (int k = 0; k < 3; k++) {
        size_t n = E * (k == 0 ? qgen_mat_w_stride(&m) : qgen_mat_s_stride(&m));
        Err e = gpu_new_buffer(g, offs[k] + n + 64, &m.buf[k]); // + over-read of the funnel shift
        if (e) {
            for (int j = 0; j < k; j++) sx_buffer_release(m.buf[j]);
            return e;
        }
    }
    qgen_prng_bytes(rng, qgen_mat_bytes(&m, 0), E * qgen_mat_w_stride(&m));
    uint8_t *s = qgen_mat_bytes(&m, 1);
    uint8_t *b = qgen_mat_bytes(&m, 2);
    size_t ns = E * qgen_mat_s_stride(&m) / 2;
    for (size_t i = 0; i < ns; i++) {
        float f = qgen_prng_float(rng); // drawn in the Zig order: float, then sign, then bias
        float sign = qgen_prng_boolean(rng) ? 1.0f : -1.0f;
        float sv = (0.5f + f) * 0.01f * sign;
        wr16(s + i * 2, to_bf16_trunc(sv));
        float bv = (qgen_prng_float(rng) * 2 - 1) * 0.1f;
        wr16(b + i * 2, to_bf16_trunc(bv));
    }
    *mo = m;
    return OK;
}

void qgen_offs_for(size_t ci, size_t k, uint32_t out[3]) {
    uint32_t base = (uint32_t)((ci * 5 + k * 7) % 15);
    out[0] = 1 + base;
    out[1] = 1 + (base + 4) % 15;
    out[2] = 1 + (base + 9) % 15;
}

// ---- context ---------------------------------------------------------------------------------

Err qgen_ctx_init(Arena *a, QgenCtx *x) {
    Pack empty = {0};
    *x = (QgenCtx){.arena = a};
    TRY(gpu_init(a, &empty, &x->g));
    qgen_prng_init(&x->rng, 0x5ae77a);
    return OK;
}

Err qgen_ctx_pipe(QgenCtx *x, const char *source, const char *name, sx_pipeline **out) {
    void **v = strmap_get(&x->pipes, str(name));
    if (v) {
        *out = *v;
        return OK;
    }
    TRY(gpu_pipeline(&x->g, source, name, out));
    strmap_put(&x->pipes, str(arena_strdup(x->arena, name)), *out);
    return OK;
}

Err qgen_ctx_floats(QgenCtx *x, size_t n, float lo, float hi, sx_buffer **out) {
    TRY(gpu_new_buffer(&x->g, MAX(n, (size_t)1) * 4, out));
    float *p = gpu_floats(*out);
    for (size_t i = 0; i < n; i++) p[i] = lo + (hi - lo) * qgen_prng_float(&x->rng);
    return OK;
}

Err qgen_ctx_halfs(QgenCtx *x, size_t n, sx_buffer **out) {
    TRY(gpu_new_buffer(&x->g, n * 2, out));
    _Float16 *p = sx_buffer_contents(*out);
    for (size_t i = 0; i < n; i++) p[i] = (_Float16)(qgen_prng_float(&x->rng) * 2 - 1);
    return OK;
}

Err qgen_ctx_mat(QgenCtx *x, size_t E, size_t out, size_t in, const uint32_t cfg[2], const uint32_t offs[3],
                 QgenMat *m) {
    return new_mat(&x->g, &x->rng, E, out, in, cfg[0], cfg[1], offs, m);
}

Err qgen_ctx_run(QgenCtx *x, sx_pipeline *p, const sx_arg *args, size_t nargs, size_t gx, size_t gy, size_t threads) {
    sx_cmd *cmd = sx_cmd_begin(x->g.dev);
    if (!cmd) return "DispatchFailed";
    sx_cmd_dispatch(cmd, p, args, nargs, gx, gy, threads);
    char err[512];
    if (sx_cmd_finish(cmd, err, sizeof err) != 0) {
        dbg("dispatch: %s\n", err);
        return "DispatchFailed";
    }
    return OK;
}

// ---- saetta qgen-bench -------------------------------------------------------------------

typedef void (*run_fn)(const void *ctx, sx_cmd *cmd);

// GPU seconds of one command buffer running run: about 0.3 s of the same work first (the
// first bursts after idle run at a low clock), then the best of 20.
static Err timed(Gpu *g, const void *ctx, run_fn run, double *best_out) {
    double warm = 0;
    char err[256];
    while (warm < 0.3) {
        sx_cmd *cmd = sx_cmd_begin(g->dev);
        if (!cmd) return "DispatchFailed";
        for (int i = 0; i < 8; i++) run(ctx, cmd);
        double secs = 0;
        if (sx_cmd_finish_timed(cmd, &secs, err, sizeof err) != 0) return "DispatchFailed";
        warm += secs;
    }
    double best = INFINITY;
    for (int i = 0; i < 20; i++) {
        sx_cmd *cmd = sx_cmd_begin(g->dev);
        if (!cmd) return "DispatchFailed";
        run(ctx, cmd);
        double secs = 0;
        if (sx_cmd_finish_timed(cmd, &secs, err, sizeof err) != 0) return "DispatchFailed";
        best = fmin(best, secs);
    }
    *best_out = best;
    return OK;
}

typedef struct {
    sx_pipeline *p;
    const QgenMat *mats;
    size_t nmats;
    sx_buffer *xb, *yb;
    bool gen;
} MvRun;

static void mv_run(const void *ctx, sx_cmd *cmd) {
    const MvRun *r = ctx;
    for (size_t i = 0; i < r->nmats; i++) {
        const QgenMat *m = &r->mats[i];
        sx_arg a[3];
        qgen_mat_args(m, a);
        uint32_t d[2] = {(uint32_t)m->in, (uint32_t)m->out};
        uint32_t q[2] = {m->bits, m->group};
        sx_arg args[9] = {a[0], a[1], a[2], gpu_buf_arg(r->xb), gpu_buf_arg(r->yb), gpu_u32_arg(&d[0]),
                          gpu_u32_arg(&d[1]), gpu_bytes_arg(m->sh, 12), gpu_bytes_arg(q, 8)};
        sx_cmd_dispatch(cmd, r->p, args, r->gen ? 9 : 8, m->out, 1, 32);
    }
}

typedef struct {
    sx_pipeline *p;
    const QgenMat *m;
    sx_buffer *xb, *yb, *idx;
    size_t slots, reps;
    bool gen;
} ExRun;

static void ex_run(const void *ctx, sx_cmd *cmd) {
    const ExRun *r = ctx;
    sx_arg a[3];
    qgen_mat_args(r->m, a);
    uint32_t d[2] = {(uint32_t)r->m->in, (uint32_t)r->m->out};
    uint32_t st[4] = {(uint32_t)qgen_mat_w_stride(r->m), (uint32_t)qgen_mat_s_stride(r->m), (uint32_t)r->m->in, 1};
    uint32_t q[2] = {r->m->bits, r->m->group};
    for (size_t i = 0; i < r->reps; i++) {
        // each rep reads other experts (idx advances by slots), as the layers of a token do
        sx_arg args[11] = {a[0], a[1], a[2], gpu_buf_arg(r->xb), gpu_buf_arg(r->yb), gpu_u32_arg(&d[0]),
                           gpu_u32_arg(&d[1]), gpu_bytes_arg(r->m->sh, 12), {r->idx, i * r->slots * 4, NULL, 0},
                           gpu_bytes_arg(st, 16), gpu_bytes_arg(q, 8)};
        sx_cmd_dispatch(cmd, r->p, args, r->gen ? 11 : 10, r->m->out, r->slots, 32);
    }
}

typedef struct {
    sx_pipeline *p;
    const QgenMat *m;
    sx_buffer *xb, *yb, *tb, *nt;
    size_t ntiles, M;
    bool gen;
} GrRun;

static void gr_run(const void *ctx, sx_cmd *cmd) {
    const GrRun *r = ctx;
    sx_arg a[3];
    qgen_mat_args(r->m, a);
    uint32_t d[3] = {(uint32_t)r->m->in, (uint32_t)r->m->out, (uint32_t)r->M};
    uint32_t st[2] = {(uint32_t)qgen_mat_w_stride(r->m), (uint32_t)qgen_mat_s_stride(r->m)};
    uint32_t q[2] = {r->m->bits, r->m->group};
    sx_arg args[11] = {a[0], a[1], a[2], gpu_buf_arg(r->xb), gpu_buf_arg(r->yb), gpu_bytes_arg(d, 12),
                       gpu_bytes_arg(r->m->sh, 12), gpu_buf_arg(r->tb), gpu_buf_arg(r->nt), gpu_bytes_arg(st, 8),
                       gpu_bytes_arg(q, 8)};
    sx_cmd_dispatch(cmd, r->p, args, r->gen ? 11 : 10, r->m->out / 64, r->ntiles, 128);
}

// DeepSeek-V4-Flash routed experts (src/kernels/dsv4_moe.metal): act_quant, gate + up and the
// summed down projection of T rows, 6 distinct experts per row, reps per command buffer, each
// rep on other slots of a cache larger than the system cache.
typedef struct {
    sx_pipeline *aq, *gu, *dn;
    sx_buffer *W, *S, *x, *xq, *slots, *rw, *hq, *y;
    uint32_t T, nslot, w_stride, s_stride;
    size_t reps;
} DsRun;

static void ds_run(const void *ctx, sx_cmd *cmd) {
    const DsRun *r = ctx;
    const Dsv4MoeP P = {r->T, r->w_stride, r->s_stride, 0, {DSV4_W1_OFF, DSV4_W2_OFF, DSV4_W3_OFF, 0},
                        {DSV4_S1_OFF, DSV4_S2_OFF, DSV4_S3_OFF, 0}};
    static const uint32_t K = 4096;
    const size_t np = (size_t)r->T * 6;
    for (size_t i = 0; i < r->reps; i++) {
        const sx_arg sl = {r->slots, i * np * 4, NULL, 0};
        sx_arg a0[3] = {gpu_buf_arg(r->x), gpu_buf_arg(r->xq), gpu_u32_arg(&K)};
        sx_cmd_dispatch(cmd, r->aq, a0, 3, 4096 / 128, r->T, 32);
        sx_arg a1[8] = {gpu_buf_arg(r->W), gpu_buf_arg(r->S), gpu_buf_arg(r->xq), sl, gpu_buf_arg(r->rw),
                        gpu_buf_arg(r->hq), gpu_buf_arg(r->hq), gpu_bytes_arg(&P, sizeof P)};
        sx_cmd_dispatch(cmd, r->gu, a1, 8, 2048 / 128, np, 256);
        sx_arg a2[6] = {gpu_buf_arg(r->W), gpu_buf_arg(r->S), gpu_buf_arg(r->hq), sl, gpu_buf_arg(r->y),
                        gpu_bytes_arg(&P, sizeof P)};
        sx_cmd_dispatch(cmd, r->dn, a2, 6, 4096 / 16, 1, 256);
    }
}

static Err dsv4_bench(QgenCtx *x, FILE *out) {
    enum { NSLOT = 40, REPS = 24 };
    const uint32_t ws = DSV4_EXPERT_W_BYTES, ss = DSV4_EXPERT_S_BYTES;
    DsRun r = {.nslot = NSLOT, .w_stride = ws, .s_stride = ss, .reps = REPS};
    TRY(qgen_ctx_pipe(x, dsv4_moe_src, "dsv4_act_quant", &r.aq));
    TRY(qgen_ctx_pipe(x, dsv4_moe_src, "dsv4_moe_gate_up", &r.gu));
    TRY(qgen_ctx_pipe(x, dsv4_moe_src, "dsv4_moe_down", &r.dn));
    TRY(gpu_new_buffer(&x->g, (size_t)NSLOT * ws, &r.W));
    TRY(gpu_new_buffer(&x->g, (size_t)NSLOT * ss, &r.S));
    qgen_prng_bytes(&x->rng, sx_buffer_contents(r.W), (size_t)NSLOT * ws);
    uint8_t *sp = sx_buffer_contents(r.S);
    for (size_t i = 0; i < (size_t)NSLOT * ss; i++) sp[i] = (uint8_t)(116 + qgen_prng_next(&x->rng) % 6);
    TRY(qgen_ctx_floats(x, 8 * 4096, -2, 2, &r.x));
    TRY(gpu_new_buffer(&x->g, 8 * 4096 * 4, &r.xq));
    TRY(gpu_new_buffer(&x->g, (size_t)REPS * 48 * 4, &r.slots));
    TRY(qgen_ctx_floats(x, 48, 0.1f, 0.4f, &r.rw));
    TRY(gpu_new_buffer(&x->g, 48 * 2048 * 4, &r.hq));
    TRY(gpu_new_buffer(&x->g, 8 * 4096 * 4, &r.y));
    static const uint32_t Ts[2] = {1, 4};
    for (size_t ti = 0; ti < 2; ti++) {
        r.T = Ts[ti];
        const size_t np = (size_t)r.T * 6;
        uint32_t *ip = sx_buffer_contents(r.slots);
        for (size_t i = 0; i < REPS * np; i++) ip[i] = (uint32_t)(i % NSLOT); // np <= NSLOT: distinct in a rep
        double s;
        TRY(timed(&x->g, &r, ds_run, &s));
        const double by = (double)REPS * (double)np * (ws + ss);
        fprintf(out, "dsv4 experts T=%u x6 4096x2048 fp4 act_quant+gate_up+down %7s us  %6s GB/s\n", r.T,
                FX(s * 1e6 / REPS, 2), FX(by / s / 1e9, 1));
        fflush(out);
    }
    sx_buffer *bufs[] = {r.W, r.S, r.x, r.xq, r.slots, r.rw, r.hq, r.y};
    for (size_t i = 0; i < COUNT(bufs); i++) sx_buffer_release(bufs[i]);
    return OK;
}

// Decode matvecs: 24 distinct copies of the matrix per command buffer (more than the system
// cache), GB/s of weights + scales + biases. Expert matvec: 10 slots x 24 reps over 240
// distinct experts [640][2560]. Prompt grouped expert matmul (accelerators): 2048 rows over
// 32 experts [640][2560], TFLOP/s.
Err qgen_bench(Arena *a, FILE *out) {
    QgenCtx x;
    TRY(qgen_ctx_init(a, &x));
    static const uint32_t cfgs[3][2] = {{4, 32}, {4, 64}, {5, 64}};
    static const size_t shapes[2][2] = {{6144, 2560}, {2560, 10240}}; // out x in
    sx_buffer *xb, *yb;
    TRY(qgen_ctx_floats(&x, 10240 * 16, -1, 1, &xb));
    TRY(gpu_new_buffer(&x.g, 10 * 10240 * 4 * 16, &yb));
    sx_pipeline *p;
    double s;
    for (size_t si = 0; si < 2; si++) {
        const size_t *shp = shapes[si];
        for (size_t ci = 0; ci < 3; ci++) {
            const uint32_t *cfg = cfgs[ci];
            QgenMat mats[24];
            for (size_t i = 0; i < 24; i++) {
                uint32_t offs[3];
                qgen_offs_for(i, 0, offs);
                Err e = qgen_ctx_mat(&x, 1, shp[0], shp[1], cfg, offs, &mats[i]);
                if (e) {
                    for (size_t j = 0; j < i; j++) qgen_mat_deinit(&mats[j]);
                    return e;
                }
            }
            double by = 0;
            for (size_t i = 0; i < 24; i++) by += (double)(qgen_mat_w_stride(&mats[i]) + 2 * qgen_mat_s_stride(&mats[i]));
            Err e = OK;
            if (ci == 0) {
                e = qgen_ctx_pipe(&x, qmatvec_src, "qmatvec_4_32", &p);
                MvRun r = {p, mats, 24, xb, yb, false};
                if (!e) e = timed(&x.g, &r, mv_run, &s);
                if (!e)
                    fprintf(out, "matvec %5zux%5zu qmatvec_4_32       4/32  %7s us  %6s GB/s\n", shp[0], shp[1],
                            FX(s * 1e6 / 24, 2), FX(by / s / 1e9, 1));
            }
            if (!e) e = qgen_ctx_pipe(&x, qgen_src, "qmatvec_gen", &p);
            if (!e) {
                MvRun r = {p, mats, 24, xb, yb, true};
                e = timed(&x.g, &r, mv_run, &s);
            }
            if (!e) {
                fprintf(out, "matvec %5zux%5zu qmatvec_gen        %u/%-3u %7s us  %6s GB/s\n", shp[0], shp[1], cfg[0],
                        cfg[1], FX(s * 1e6 / 24, 2), FX(by / s / 1e9, 1));
                fflush(out);
            }
            for (size_t i = 0; i < 24; i++) qgen_mat_deinit(&mats[i]);
            TRY(e);
        }
    }
    { // decode routed experts
        enum { E = 240, slots = 10, reps = 24 };
        sx_buffer *idx;
        TRY(gpu_new_buffer(&x.g, E * 4, &idx));
        uint32_t *ip = sx_buffer_contents(idx);
        for (size_t i = 0; i < E; i++) ip[i] = (uint32_t)((i * 97) % E);
        for (size_t ci = 0; ci < 3; ci++) {
            const uint32_t *cfg = cfgs[ci];
            QgenMat m;
            static const uint32_t offs[3] = {3, 5, 7};
            TRY(qgen_ctx_mat(&x, E, 640, 2560, cfg, offs, &m));
            double by = (double)(slots * reps * (qgen_mat_w_stride(&m) + 2 * qgen_mat_s_stride(&m)));
            Err e = OK;
            if (ci == 0) {
                e = qgen_ctx_pipe(&x, qmatvec_src, "qmatvec_expert_4_32", &p);
                ExRun r = {p, &m, xb, yb, idx, slots, reps, false};
                if (!e) e = timed(&x.g, &r, ex_run, &s);
                if (!e)
                    fprintf(out, "expert  640x 2560 x10 qmatvec_expert_4_32 4/32 %7s us  %6s GB/s\n", FX(s * 1e6 / reps, 2),
                            FX(by / s / 1e9, 1));
            }
            if (!e) e = qgen_ctx_pipe(&x, qgen_src, "qmatvec_expert_gen", &p);
            if (!e) {
                ExRun r = {p, &m, xb, yb, idx, slots, reps, true};
                e = timed(&x.g, &r, ex_run, &s);
            }
            if (!e) {
                fprintf(out, "expert  640x 2560 x10 qmatvec_expert_gen  %u/%-3u %7s us  %6s GB/s\n", cfg[0], cfg[1],
                        FX(s * 1e6 / reps, 2), FX(by / s / 1e9, 1));
                fflush(out);
            }
            qgen_mat_deinit(&m);
            TRY(e);
        }
    }
    { // prompt grouped experts on the accelerators
        enum { E = 32, M = 2048, K = 2560, N = 640 };
        sx_buffer *xh, *yh, *tb, *nt;
        TRY(qgen_ctx_halfs(&x, (size_t)M * K, &xh));
        TRY(gpu_new_buffer(&x.g, (size_t)M * N * 4, &yh));
        uint32_t tiles[E][4];
        for (uint32_t i = 0; i < E; i++) {
            tiles[i][0] = i;
            tiles[i][1] = i * 64;
            tiles[i][2] = i * 64 + 64;
            tiles[i][3] = 0;
        }
        TRY(gpu_new_buffer(&x.g, sizeof tiles, &tb));
        memcpy(sx_buffer_contents(tb), tiles, sizeof tiles);
        TRY(gpu_new_buffer(&x.g, 4, &nt));
        *(uint32_t *)sx_buffer_contents(nt) = E;
        double flop = 2.0 * M * K * N;
        for (size_t ci = 0; ci < 3; ci++) {
            const uint32_t *cfg = cfgs[ci];
            QgenMat m;
            static const uint32_t offs[3] = {3, 5, 7};
            TRY(qgen_ctx_mat(&x, E, N, K, cfg, offs, &m));
            Err e = OK;
            if (ci == 0) {
                e = qgen_ctx_pipe(&x, qmm_mpp_src, "qmmh_g_4_32", &p);
                GrRun r = {p, &m, xh, yh, tb, nt, E, M, false};
                if (!e) e = timed(&x.g, &r, gr_run, &s);
                if (!e)
                    fprintf(out, "prompt experts 2048x640x2560 qmmh_g_4_32 4/32  %7s us  %6s TFLOP/s\n", FX(s * 1e6, 1),
                            FX(flop / s / 1e12, 1));
            }
            if (!e) e = qgen_ctx_pipe(&x, qgen_mpp_src, "qmmh_g_gen", &p);
            if (!e) {
                GrRun r = {p, &m, xh, yh, tb, nt, E, M, true};
                e = timed(&x.g, &r, gr_run, &s);
            }
            if (!e) {
                fprintf(out, "prompt experts 2048x640x2560 qmmh_g_gen  %u/%-3u %7s us  %6s TFLOP/s\n", cfg[0], cfg[1],
                        FX(s * 1e6, 1), FX(flop / s / 1e12, 1));
                fflush(out);
            }
            qgen_mat_deinit(&m);
            TRY(e);
        }
    }
    TRY(dsv4_bench(&x, out));
    TRY(attn27_bench(a, out)); // Qwen3.8-27B prompt attention (src/attn27.h)
    return OK;
}
