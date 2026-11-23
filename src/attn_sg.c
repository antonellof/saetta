// Prompt-chunk attention without tensor ops: loaders, reference and bench (see attn_sg.h).
#include "attn_sg.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

const char attn_sg_src[] = {
#embed "kernels/attn_sg.metal"
    , 0};

#define NQ ATTN_SG_NQ
#define HD ATTN_SG_HD

// The pipeline for name if it compiles and allows threads threads per threadgroup, else NULL.
static Err pipe_fit(Gpu *g, const char *name, uint32_t threads, sx_pipeline **out) {
    TRY(gpu_pipeline(g, attn_sg_src, name, out));
    if (sx_pipeline_max_threads(*out) < threads) {
        dbg("attn_sg: %s allows %u threads per threadgroup, needs %u\n", name, sx_pipeline_max_threads(*out), threads);
        sx_pipeline_release(*out);
        *out = NULL;
        return "TooManyThreads";
    }
    return OK;
}

Err attn_sg_load_27(Gpu *g, AttnSg27 *out) {
    static const struct {
        const char *dir, *part;
        uint32_t nsg;
    } vs[] = {{"attn_sg_27", "attn_sg_27_part", 8}, {"attn_sg_27_s4", "attn_sg_27_s4_part", 4}};
    *out = (AttnSg27){0};
    Err e = "NoVariant";
    for (size_t i = 0; i < COUNT(vs); i++) {
        sx_pipeline *d = NULL, *p = NULL;
        e = pipe_fit(g, vs[i].dir, vs[i].nsg * 32, &d);
        if (!e) e = pipe_fit(g, vs[i].part, vs[i].nsg * 32, &p);
        if (!e) {
            *out = (AttnSg27){d, p, vs[i].nsg};
            return OK;
        }
        sx_pipeline_release(d);
    }
    return e;
}

Err attn_sg_load_fn(Gpu *g, sx_pipeline **out) { return pipe_fit(g, "attn_sg_fn", ATTN_SG_FN_THREADS, out); }

size_t attn_sg_nsplit(size_t T, size_t qh, size_t nkv, size_t nsg, size_t nk, size_t max_t, size_t max_split) {
    const size_t groups = nkv * attn_sg_blocks(T, qh, nsg);
    if (groups >= 256 || T > max_t) return 1;
    size_t ns = (256 + groups - 1) / groups;
    ns = MIN(ns, max_split);
    ns = MIN(ns, MAX((size_t)1, nk / 1024));
    return ns;
}

void attn_sg_ref(const float *q, const float *gate, const _Float16 *K, const _Float16 *V, size_t nkv, size_t t,
                 const uint32_t *keys, size_t nkeys, float *out) {
    const size_t qh = NQ / nkv;
    double *s = malloc(nkeys * sizeof *s);
    if (!s) die("attn_sg_ref: out of memory\n");
    for (size_t h = 0; h < NQ; h++) {
        const size_t g = h / qh;
        const float *qv = q + (t * NQ + h) * HD;
        double mx = -INFINITY;
        for (size_t i = 0; i < nkeys; i++) {
            const _Float16 *k = K + (size_t)(keys ? keys[i] : i) * nkv * HD + g * HD;
            double dot = 0;
            for (size_t d = 0; d < HD; d++) dot += (double)qv[d] * (double)k[d];
            s[i] = dot / 16.0; // 1 / sqrt(256)
            mx = fmax(mx, s[i]);
        }
        double o[HD] = {0}, l = 0;
        for (size_t i = 0; i < nkeys; i++) {
            const double e = exp(s[i] - mx);
            l += e;
            const _Float16 *v = V + (size_t)(keys ? keys[i] : i) * nkv * HD + g * HD;
            for (size_t d = 0; d < HD; d++) o[d] += e * (double)v[d];
        }
        for (size_t d = 0; d < HD; d++) {
            const double gv = gate[(t * NQ + h) * 2 * HD + HD + d];
            out[h * HD + d] = (float)(o[d] / l / (1.0 + exp(-gv)));
        }
    }
    free(s);
}

void attn_sg_lists(QgenPrng *r, size_t pos0, size_t T, size_t budget, size_t ratio, size_t maxsel, uint32_t *sel,
                   uint32_t *cnt) {
    const size_t nbmax = (pos0 + T) / ratio;
    uint8_t *mark = calloc(nbmax + 1, 1);
    if (!mark) die("attn_sg_lists: out of memory\n");
    for (size_t t = 0; t < T; t++) {
        const size_t n = pos0 + t + 1, nb = n / ratio;
        if (nb <= budget) {
            cnt[t] = 0;
            continue;
        }
        // Floyd's sampling of budget distinct blocks of nb
        memset(mark, 0, nb);
        for (size_t j = nb - budget; j < nb; j++) {
            const size_t x = (size_t)(qgen_prng_next(r) % (j + 1));
            mark[mark[x] ? j : x] = 1;
        }
        uint32_t *st = sel + t * maxsel;
        size_t c = 0;
        for (size_t b = 0; b < nb; b++)
            if (mark[b])
                for (size_t u = 0; u < ratio; u++) st[c++] = (uint32_t)(b * ratio + u);
        for (size_t p = nb * ratio; p < n; p++) st[c++] = (uint32_t)p;
        if (c > maxsel) die("attn_sg_lists: %zu entries over %zu\n", c, maxsel);
        cnt[t] = (uint32_t)c;
    }
    free(mark);
}

// ---- saetta attn-sg-bench -----------------------------------------------------------------

// The models' own kernel sources, for attn_full_6h, attn_split_gh and attn_merge.
static const char old_src[] = {
#embed "kernels/qmatvec.metal"
    , '\n',
#embed "kernels/decode.metal"
    , '\n',
#embed "kernels/qmm.metal"
    , '\n',
#embed "kernels/dense.metal"
    , 0};

// AttnSP of decode.metal.
typedef struct {
    uint32_t pos0, budget_blocks, ratio, head_dim, kv_stride, q_per_kv, gate_stride, nq, nsplit, maxsel;
    float scale;
} AttnSP;

#define AB 512
#define NSPLIT_27 8                     // dense.c NSPLIT_PRE
#define FN_BUDGET 512                   // config_flash.h: indexer budget / ratio
#define FN_RATIO 4
#define FN_MAXSEL (FN_BUDGET * FN_RATIO + FN_RATIO - 1 + 13) // model.c's list stride, MAXSEL + 13
#define FN_NSPLIT ((FN_BUDGET * FN_RATIO + FN_RATIO - 1 + 63) / 64)

typedef struct {
    sx_pipeline *old, *merge, *sg;
    sx_buffer *q, *qg, *kv, *part, *att, *sel, *cnt;
    size_t pos0, T, cap, nkv;
    bool fn, use_old;
    AttnSg27 a27;
} Run;

static void run_old(const Run *r, sx_cmd *c) {
    const size_t T = r->T, qh = NQ / r->nkv;
    const size_t nsplit = r->fn ? FN_NSPLIT : NSPLIT_27;
    AttnSP p = {(uint32_t)r->pos0, r->fn ? FN_BUDGET : 0, r->fn ? FN_RATIO : 1, HD, (uint32_t)(r->nkv * HD), (uint32_t)qh,
                2 * HD, NQ, (uint32_t)nsplit, FN_MAXSEL, 1.0f / 16.0f};
    const sx_arg kc = {r->kv, 0, NULL, 0}, vc = {r->kv, r->cap * r->nkv * HD * 2, NULL, 0};
    for (size_t q0 = 0; q0 < T; q0 += AB) {
        const size_t tb = MIN((size_t)AB, T - q0);
        AttnSP pq = p;
        pq.pos0 = (uint32_t)(r->pos0 + q0);
        const sx_arg qa = {r->q, q0 * NQ * HD * 4, NULL, 0};
        if (r->fn) {
            sx_arg a[] = {qa, kc, vc, {r->sel, q0 * FN_MAXSEL * 4, NULL, 0}, {r->cnt, q0 * 4, NULL, 0}, gpu_buf_arg(r->part),
                          gpu_bytes_arg(&pq, sizeof pq)};
            sx_cmd_dispatch3(c, r->old, a, COUNT(a), r->nkv, nsplit, tb, 256);
        } else {
            sx_arg a[] = {qa, kc, vc, gpu_buf_arg(r->part), gpu_bytes_arg(&pq, sizeof pq)};
            sx_cmd_dispatch3(c, r->old, a, COUNT(a), r->nkv, nsplit, tb, 256);
        }
        sx_arg b[] = {gpu_buf_arg(r->part), {r->qg, (q0 * NQ * 2 * HD + HD) * 4, NULL, 0}, {r->att, q0 * NQ * HD * 4, NULL, 0},
                      gpu_bytes_arg(&pq, sizeof pq)};
        sx_cmd_dispatch(c, r->merge, b, COUNT(b), NQ, tb, 256);
    }
}

static void run_sg(const Run *r, sx_cmd *c) {
    const AttnSgP p = {(uint32_t)r->pos0, (uint32_t)r->T, 1, FN_BUDGET, FN_RATIO, FN_MAXSEL};
    sx_arg a[] = {gpu_buf_arg(r->q), {r->kv, 0, NULL, 0}, {r->kv, r->cap * r->nkv * HD * 2, NULL, 0}, {r->qg, HD * 4, NULL, 0},
                  gpu_buf_arg(r->att), gpu_bytes_arg(&p, sizeof p), gpu_buf_arg(r->sel), gpu_buf_arg(r->cnt)};
    if (r->fn)
        sx_cmd_dispatch3(c, r->sg, a, COUNT(a), r->nkv, r->T, 1, ATTN_SG_FN_THREADS);
    else
        sx_cmd_dispatch3(c, r->a27.dir, a, COUNT(a), r->nkv, attn_sg_blocks(r->T, NQ / r->nkv, r->a27.nsg), 1,
                         r->a27.nsg * 32);
}

// Best GPU seconds of reps command buffers of one call each, after one warm-up call.
static Err time_run(Gpu *g, const Run *r, int reps, double *best) {
    char err[256];
    *best = INFINITY;
    for (int i = 0; i <= reps; i++) {
        sx_cmd *c = sx_cmd_begin(g->dev);
        if (!c) return "DispatchFailed";
        if (r->use_old) run_old(r, c);
        else run_sg(r, c);
        double s = 0;
        if (sx_cmd_finish_timed(c, &s, err, sizeof err) != 0) {
            dbg("attn-sg-bench: %s\n", err);
            return "DispatchFailed";
        }
        if (i > 0) *best = fmin(*best, s);
    }
    return OK;
}

Err attn_sg_bench(Arena *a, FILE *out) {
    QgenCtx x;
    TRY(qgen_ctx_init(a, &x));
    enum { T = 4096 };
    static const size_t ctxs[4] = {4096, 16384, 32768, 65536};
    const size_t cap = 65536;
    Run r = {.T = T, .cap = cap};
    TRY(qgen_ctx_pipe(&x, old_src, "attn_merge", &r.merge));
    TRY(qgen_ctx_floats(&x, (size_t)T * NQ * HD, -2, 2, &r.q));
    TRY(qgen_ctx_floats(&x, (size_t)T * NQ * 2 * HD, -2, 2, &r.qg));
    TRY(qgen_ctx_halfs(&x, 2 * cap * 4 * HD, &r.kv));
    TRY(gpu_new_buffer(&x.g, (size_t)AB * NQ * FN_NSPLIT * (HD + 2) * 4, &r.part));
    TRY(gpu_new_buffer(&x.g, (size_t)T * NQ * HD * 4, &r.att));
    TRY(gpu_new_buffer(&x.g, (size_t)T * FN_MAXSEL * 4, &r.sel));
    TRY(gpu_new_buffer(&x.g, (size_t)T * 4, &r.cnt));
    for (int model = 0; model < 2; model++) {
        r.fn = model == 1;
        r.nkv = r.fn ? 2 : 4;
        if (r.fn) {
            TRY(qgen_ctx_pipe(&x, old_src, "attn_split_gh", &r.old));
            TRY(attn_sg_load_fn(&x.g, &r.sg));
        } else {
            TRY(qgen_ctx_pipe(&x, old_src, "attn_full_6h", &r.old));
            TRY(attn_sg_load_27(&x.g, &r.a27));
        }
        fprintf(out, "attn-sg-bench: %s prompt attention, one layer, chunk of %d tokens, f16 KV, best GPU time%s\n",
                r.fn ? "Qwen3.8-Flash-Next (QSA lists of 2,048 + tail)" : "Qwen3.8-27B", T,
                r.fn ? "" : (r.a27.nsg == 8 ? " (attn_sg_27, 8 simdgroups)" : " (attn_sg_27_s4, 4 simdgroups)"));
        fflush(out);
        for (size_t ci = 0; ci < 4; ci++) {
            r.pos0 = ctxs[ci] - T;
            if (r.fn) {
                QgenPrng rng;
                qgen_prng_init(&rng, 0x5ae77a + ci);
                attn_sg_lists(&rng, r.pos0, T, FN_BUDGET, FN_RATIO, FN_MAXSEL, sx_buffer_contents(r.sel),
                              sx_buffer_contents(r.cnt));
            }
            double fl = 0; // flops: QK and PV, 2 each per head, dim and attended key
            for (size_t t = 0; t < T; t++) {
                const size_t n = r.pos0 + t + 1;
                const size_t nk = r.fn && n / FN_RATIO > FN_BUDGET ? FN_BUDGET * FN_RATIO + n % FN_RATIO : n;
                fl += 4.0 * NQ * HD * (double)nk;
            }
            for (int k = 0; k < 2; k++) {
                r.use_old = k == 0;
                double s;
                TRY(time_run(&x.g, &r, ctxs[ci] > 16384 && k == 0 ? 2 : 5, &s));
                const char *nm = r.fn ? (k == 0 ? "attn_split_gh+merge" : "attn_sg_fn") : (k == 0 ? "attn_full_6h+merge" : "attn_sg_27");
                const int nl = r.fn ? 12 : 16;
                fprintf(out, "ctx %6zu pos0 %6zu %-20s %9.2f ms/layer %7.3f ms/token (%d layers) %6.2f TFLOP/s\n", ctxs[ci],
                        r.pos0, nm, s * 1e3, s * 1e3 * nl / T, nl, fl / s / 1e12);
                fflush(out);
            }
        }
    }
    sx_buffer *bufs[] = {r.q, r.qg, r.kv, r.part, r.att, r.sel, r.cnt};
    for (size_t i = 0; i < COUNT(bufs); i++) sx_buffer_release(bufs[i]);
    return OK;
}
