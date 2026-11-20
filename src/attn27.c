// Qwen3.8-27B prompt-chunk attention: references and bench (see attn27.h).
#include "attn27.h"

#include <math.h>

#include "qgen.h"

const char attn27_src[] = {
#embed "kernels/dense_attn.metal"
    , 0};

// The 27B forward's own kernel source, for attn_full_6h and attn_merge (as src/dense.c builds it).
static const char old_src[] = {
#embed "kernels/qmatvec.metal"
    , '\n',
#embed "kernels/decode.metal"
    , '\n',
#embed "kernels/qmm.metal"
    , '\n',
#embed "kernels/dense.metal"
    , 0};

#define NQ ATTN27_NQ
#define NKV ATTN27_NKV
#define HD ATTN27_HD
#define G (NQ / NKV)
#define BK 128 // the kernel's key step (FA_BK)

void attn27_ref(const float *q, const float *qg, const _Float16 *K, const _Float16 *V, size_t pos0, size_t t,
                bool mirror, float *out) {
    const size_t n = pos0 + t + 1;
    double s[BK], o[HD];
    for (size_t h = 0; h < NQ; h++) {
        const size_t g = h / G;
        double qv[HD];
        for (size_t d = 0; d < HD; d++) {
            const float x = q[(t * NQ + h) * HD + d];
            qv[d] = mirror ? (double)(_Float16)x : (double)x;
        }
        double m = -INFINITY, l = 0;
        for (size_t d = 0; d < HD; d++) o[d] = 0;
        // 128-key steps from key 0, as the kernel: the max is the running one after each step
        for (size_t c0 = 0; c0 < n; c0 += BK) {
            const size_t cn = MIN((size_t)BK, n - c0);
            double mx = m;
            for (size_t i = 0; i < cn; i++) {
                const _Float16 *k = K + (c0 + i) * NKV * HD + g * HD;
                double dot = 0;
                for (size_t d = 0; d < HD; d++) dot += qv[d] * (double)k[d];
                s[i] = dot / 16.0; // 1 / sqrt(256)
                mx = fmax(mx, s[i]);
            }
            const double f = m == -INFINITY ? 1.0 : exp(m - mx);
            l *= f;
            for (size_t d = 0; d < HD; d++) o[d] *= f;
            for (size_t i = 0; i < cn; i++) {
                const double e = exp(s[i] - mx);
                l += e;
                const double pe = mirror ? (double)(_Float16)e : e;
                const _Float16 *v = V + (c0 + i) * NKV * HD + g * HD;
                for (size_t d = 0; d < HD; d++) o[d] += pe * (double)v[d];
            }
            m = mx;
        }
        for (size_t d = 0; d < HD; d++) {
            const double gate = qg[(t * NQ + h) * 2 * HD + HD + d];
            out[h * HD + d] = (float)(o[d] / l / (1.0 + exp(-gate)));
        }
    }
}

// ---- saetta attn-bench ------------------------------------------------------------------

// AttnSP of decode.metal.
typedef struct {
    uint32_t pos0, budget_blocks, ratio, head_dim, kv_stride, q_per_kv, gate_stride, nq, nsplit, maxsel;
    float scale;
} AttnSP;

typedef struct {
    sx_pipeline *full, *merge, *q6h, *fl;
    sx_buffer *q, *qg, *kv, *part, *qh, *att;
    size_t pos0, T, cap, qb;
    int which; // 0 old, 1 new
} Run;

#define AB 512
#define NSPLIT_PRE 8

static void run_old(const Run *r, sx_cmd *c) {
    const size_t T = r->T;
    AttnSP p = {(uint32_t)r->pos0, 0, 1, HD, NKV * HD, G, 2 * HD, NQ, NSPLIT_PRE, 0, 1.0f / 16.0f};
    const sx_arg kc = {r->kv, 0, NULL, 0}, vc = {r->kv, r->cap * NKV * HD * 2, NULL, 0};
    for (size_t q0 = 0; q0 < T; q0 += AB) {
        const size_t tb = MIN((size_t)AB, T - q0);
        AttnSP pq = p;
        pq.pos0 = (uint32_t)(r->pos0 + q0);
        sx_arg a[] = {{r->q, q0 * NQ * HD * 4, NULL, 0}, kc, vc, gpu_buf_arg(r->part), gpu_bytes_arg(&pq, sizeof pq)};
        sx_cmd_dispatch3(c, r->full, a, COUNT(a), NKV, NSPLIT_PRE, tb, 256);
        sx_arg b[] = {gpu_buf_arg(r->part), {r->qg, (q0 * NQ * 2 * HD + HD) * 4, NULL, 0}, {r->att, q0 * NQ * HD * 4, NULL, 0},
                      gpu_bytes_arg(&pq, sizeof pq)};
        sx_cmd_dispatch(c, r->merge, b, COUNT(b), NQ, tb, 256);
    }
}

static void run_new(const Run *r, sx_cmd *c) {
    const Attn27P p = {(uint32_t)r->pos0, (uint32_t)r->T, attn27_tq(r->T, r->qb)};
    const size_t n4 = (size_t)NKV * p.tq * G * HD / 4;
    sx_arg a[] = {gpu_buf_arg(r->q), gpu_buf_arg(r->qh), gpu_bytes_arg(&p, sizeof p)};
    sx_cmd_dispatch(c, r->q6h, a, COUNT(a), (n4 + 255) / 256, 1, 256);
    sx_arg b[] = {gpu_buf_arg(r->qh), {r->kv, 0, NULL, 0}, {r->kv, r->cap * NKV * HD * 2, NULL, 0}, {r->qg, HD * 4, NULL, 0},
                  gpu_buf_arg(r->att), gpu_bytes_arg(&p, sizeof p)};
    sx_cmd_dispatch(c, r->fl, b, COUNT(b), NKV, p.tq / r->qb, r->qb == 8 ? 256 : 512);
}

// Best GPU seconds of reps command buffers of one call each, after one warm-up call.
static Err time_run(Gpu *g, const Run *r, int reps, double *best) {
    char err[256];
    *best = INFINITY;
    for (int i = 0; i <= reps; i++) {
        sx_cmd *c = sx_cmd_begin(g->dev);
        if (!c) return "DispatchFailed";
        if (r->which == 0) run_old(r, c);
        else run_new(r, c);
        double s = 0;
        if (sx_cmd_finish_timed(c, &s, err, sizeof err) != 0) {
            dbg("attn-bench: %s\n", err);
            return "DispatchFailed";
        }
        if (i > 0) *best = fmin(*best, s);
    }
    return OK;
}

Err attn27_bench(Arena *a, FILE *out) {
    QgenCtx x;
    TRY(qgen_ctx_init(a, &x));
    enum { T = 4096 };
    static const size_t ctxs[4] = {4096, 16384, 32768, 65536};
    const size_t cap = 65536;
    Run r = {.T = T, .cap = cap};
    TRY(qgen_ctx_pipe(&x, old_src, "attn_full_6h", &r.full));
    TRY(qgen_ctx_pipe(&x, old_src, "attn_merge", &r.merge));
    TRY(qgen_ctx_pipe(&x, attn27_src, "attn_fl_q6h", &r.q6h));
    sx_pipeline *fl8, *fl16;
    TRY(qgen_ctx_pipe(&x, attn27_src, "attn_fl_6_q8", &fl8));
    TRY(qgen_ctx_pipe(&x, attn27_src, "attn_fl_6_q16", &fl16));
    TRY(qgen_ctx_floats(&x, (size_t)T * NQ * HD, -2, 2, &r.q));
    TRY(qgen_ctx_floats(&x, (size_t)T * NQ * 2 * HD, -2, 2, &r.qg));
    TRY(qgen_ctx_halfs(&x, 2 * cap * NKV * HD, &r.kv));
    TRY(gpu_new_buffer(&x.g, (size_t)AB * NQ * NSPLIT_PRE * (HD + 2) * 4, &r.part));
    TRY(gpu_new_buffer(&x.g, (size_t)NKV * (T + 16) * G * HD * 2, &r.qh));
    TRY(gpu_new_buffer(&x.g, (size_t)T * NQ * HD * 4, &r.att));
    fprintf(out, "attn-bench: Qwen3.8-27B prompt attention, one layer, chunk of %d tokens, f16 KV, best GPU time\n", T);
    fflush(out);
    for (size_t ci = 0; ci < 4; ci++) {
        r.pos0 = ctxs[ci] - T;
        double fl = 0; // causal flops: QK and PV, 2 each per head, dim and visible key
        for (size_t t = 0; t < T; t++) fl += 4.0 * NQ * HD * (double)(r.pos0 + t + 1);
        for (int k = 0; k < 3; k++) {
            r.which = k > 0;
            r.fl = k == 1 ? fl8 : fl16;
            r.qb = k == 1 ? 8 : 16;
            double s;
            TRY(time_run(&x.g, &r, ctxs[ci] > 16384 && k == 0 ? 2 : 5, &s));
            static const char *names[3] = {"attn_full_6h+merge", "attn_fl_6_q8", "attn_fl_6_q16"};
            fprintf(out, "ctx %6zu pos0 %6zu %-20s %9.2f ms/layer %7.3f ms/token (16 layers) %6.2f TFLOP/s\n", ctxs[ci],
                    r.pos0, names[k], s * 1e3, s * 1e3 * 16 / T, fl / s / 1e12);
            fflush(out);
        }
    }
    sx_buffer *bufs[] = {r.q, r.qg, r.kv, r.part, r.qh, r.att};
    for (size_t i = 0; i < COUNT(bufs); i++) sx_buffer_release(bufs[i]);
    return OK;
}
