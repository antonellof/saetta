// Run-time LoRA kernels: sources, encode functions, scalar references and lora-bench (see lora.h).
#include "lora.h"

#include <math.h>

#include "qgen.h"

const char lora_src[] = {
#embed "kernels/lora.metal"
    , 0};

// Constant structs of lora.metal.
typedef struct {
    uint32_t in, rt, T, ldx;
} LoraDownP;
typedef struct {
    uint32_t T, rt, R, n;
    uint32_t out[3], r0[3], ldy[3];
    uint32_t tg0[4];
    float scale;
} LoraUpP;
typedef struct {
    uint32_t M, N, K, ldx, ldc, add;
    float alpha;
} LoraMmP;

enum { DN_TG = 256, UP_TG = 64, MM_TG = 128, MM_TILE = 32 };

Err lora_init(Gpu *g, LoraPipes *p) {
    TRY(gpu_pipeline(g, lora_src, "lora_down_small", &p->down_small));
    TRY(gpu_pipeline(g, lora_src, "lora_up_small", &p->up_small));
    TRY(gpu_pipeline(g, lora_src, "lora_mm", &p->mm));
    return OK;
}

static size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

static void encode_mm(const LoraPipes *p, sx_cmd *c, sx_arg W, sx_arg X, sx_arg C, const LoraMmP *mp) {
    sx_arg args[4] = {W, X, C, gpu_bytes_arg(mp, sizeof *mp)};
    sx_cmd_dispatch(c, p->mm, args, 4, cdiv(mp->N, MM_TILE), cdiv(mp->M, MM_TILE), MM_TG);
}

void lora_encode_down(const LoraPipes *p, sx_cmd *c, sx_arg A, sx_arg x, sx_arg t, uint32_t T, uint32_t in,
                      uint32_t ldx, uint32_t rt) {
    if (T == 0) return;
    if (T <= LORA_SMALL_T) {
        LoraDownP dp = {in, rt, T, ldx};
        sx_arg args[4] = {A, x, t, gpu_bytes_arg(&dp, sizeof dp)};
        sx_cmd_dispatch(c, p->down_small, args, 4, rt, 1, DN_TG);
        return;
    }
    LoraMmP mp = {T, rt, in, ldx, rt, 0, 1.0f};
    encode_mm(p, c, A, x, t, &mp);
}

void lora_encode_up(const LoraPipes *p, sx_cmd *c, sx_arg t, uint32_t T, uint32_t rt, uint32_t R, float scale,
                    const LoraUp *ups, size_t n) {
    if (T == 0 || n == 0) return;
    if (T <= LORA_SMALL_T) {
        LoraUpP up = {.T = T, .rt = rt, .R = R, .n = (uint32_t)n, .scale = scale};
        sx_arg args[8];
        uint32_t g = 0;
        for (size_t j = 0; j < 3; j++) {
            const LoraUp *u = &ups[j < n ? j : 0];
            up.tg0[j] = g;
            if (j < n) {
                up.out[j] = u->out;
                up.r0[j] = u->r0;
                up.ldy[j] = u->ldy;
                g += (uint32_t)cdiv(u->out, UP_TG);
            }
            args[j] = u->B;
            args[3 + j] = u->y;
        }
        for (size_t j = n; j < 4; j++) up.tg0[j] = g; // targets past n start at the end
        args[6] = t;
        args[7] = gpu_bytes_arg(&up, sizeof up);
        sx_cmd_dispatch(c, p->up_small, args, 8, g, 1, UP_TG);
        return;
    }
    for (size_t j = 0; j < n; j++) {
        sx_arg ts = t;
        ts.offset += (size_t)ups[j].r0 * 4;
        LoraMmP mp = {T, ups[j].out, R, rt, ups[j].ldy, 1, scale};
        encode_mm(p, c, ups[j].B, ts, ups[j].y, &mp);
    }
}

void lora_ref_down(const _Float16 *A, const float *x, float *t, size_t T, size_t in, size_t ldx, size_t rt) {
    for (size_t tok = 0; tok < T; tok++)
        for (size_t r = 0; r < rt; r++) {
            double acc = 0;
            for (size_t k = 0; k < in; k++) acc += (double)A[r * in + k] * (double)x[tok * ldx + k];
            t[tok * rt + r] = (float)acc;
        }
}

void lora_ref_up(const _Float16 *B, const float *t, float *y, size_t T, size_t rt, size_t r0, size_t R, size_t out,
                 size_t ldy, float scale) {
    for (size_t tok = 0; tok < T; tok++)
        for (size_t o = 0; o < out; o++) {
            double acc = 0;
            for (size_t r = 0; r < R; r++) acc += (double)B[o * R + r] * (double)t[tok * rt + r0 + r];
            y[tok * ldy + o] = (float)((double)y[tok * ldy + o] + (double)scale * acc);
        }
}

// ---- saetta lora-bench -----------------------------------------------------------------------

typedef void (*run_fn)(const void *ctx, sx_cmd *cmd);

// GPU seconds of one command buffer running run: about 0.3 s of the same work first (the
// first bursts after idle run at a low clock), then the best of 20.
static Err timed(Gpu *g, const void *ctx, run_fn run, double *best_out) {
    double warm = 0;
    char err[256];
    while (warm < 0.3) {
        sx_cmd *cmd = sx_cmd_begin(g->dev);
        if (!cmd) return "DispatchFailed";
        run(ctx, cmd);
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

// Qwen3.8-27B decode set (docs/dense-lora.md): R 32 everywhere.
enum { BR = 32, NLAYERS = 64, NATTN = 16, HID = 5120, QOUT = 12288, KVOUT = 1024, OIN = 6144, FF = 17408 };

// The adapter weights of all layers, one buffer per kind, layer l at l * (bytes per layer).
enum { W_QKV, W_GU, W_O, W_D, W_BQ, W_BK, W_BV, W_BG, W_BU, W_BO, W_BD, NW };
static const size_t w_rows[NW] = {3 * BR, 2 * BR, BR, BR, QOUT, KVOUT, KVOUT, FF, FF, HID, HID};
static const size_t w_cols[NW] = {HID, HID, OIN, FF, BR, BR, BR, BR, BR, BR, BR};
static const size_t w_layers[NW] = {NATTN, NLAYERS, NATTN, NLAYERS, NATTN, NATTN, NATTN, NLAYERS, NLAYERS, NATTN, NLAYERS};

typedef struct {
    const LoraPipes *p;
    sx_buffer *w[NW];
    sx_buffer *x5, *x6, *x17, *t, *yq, *yk, *yv, *yg, *yu, *yo, *yd;
    int kind;   // which op the run encodes, over all its layers; -1 the whole token
    bool fused; // up of q/k/v and gate/up in one dispatch
} Bench;

static size_t wbytes(int k) { return w_rows[k] * w_cols[k] * 2; }
static sx_arg warg(const Bench *b, int k, size_t l) { return (sx_arg){b->w[k], l * wbytes(k), NULL, 0}; }

enum { K_DQKV, K_UQKV, K_DGU, K_UGU, K_DO, K_UO, K_DD, K_UD, NK };
static const char *const k_name[NK] = {"down q/k/v   in  5120 Rt 96", "up   q/k/v   out 12288+1024+1024",
                                       "down gate/up in  5120 Rt 64", "up   gate/up out 17408+17408",
                                       "down o_proj  in  6144 Rt 32", "up   o_proj  out 5120",
                                       "down down    in 17408 Rt 32", "up   down    out 5120"};

static void up_n(const Bench *b, sx_cmd *c, const LoraUp *u, size_t n, uint32_t rt) {
    if (b->fused) {
        lora_encode_up(b->p, c, gpu_buf_arg(b->t), 1, rt, BR, 2.0f, u, n);
    } else {
        for (size_t j = 0; j < n; j++) lora_encode_up(b->p, c, gpu_buf_arg(b->t), 1, rt, BR, 2.0f, &u[j], 1);
    }
}

static void op(const Bench *b, sx_cmd *c, int k, size_t l) {
    switch (k) {
    case K_DQKV:
        lora_encode_down(b->p, c, warg(b, W_QKV, l), gpu_buf_arg(b->x5), gpu_buf_arg(b->t), 1, HID, HID, 3 * BR);
        break;
    case K_UQKV: {
        LoraUp u[3] = {{warg(b, W_BQ, l), gpu_buf_arg(b->yq), QOUT, 0, QOUT},
                       {warg(b, W_BK, l), gpu_buf_arg(b->yk), KVOUT, BR, KVOUT},
                       {warg(b, W_BV, l), gpu_buf_arg(b->yv), KVOUT, 2 * BR, KVOUT}};
        up_n(b, c, u, 3, 3 * BR);
        break;
    }
    case K_DGU:
        lora_encode_down(b->p, c, warg(b, W_GU, l), gpu_buf_arg(b->x5), gpu_buf_arg(b->t), 1, HID, HID, 2 * BR);
        break;
    case K_UGU: {
        LoraUp u[2] = {{warg(b, W_BG, l), gpu_buf_arg(b->yg), FF, 0, FF}, {warg(b, W_BU, l), gpu_buf_arg(b->yu), FF, BR, FF}};
        up_n(b, c, u, 2, 2 * BR);
        break;
    }
    case K_DO:
        lora_encode_down(b->p, c, warg(b, W_O, l), gpu_buf_arg(b->x6), gpu_buf_arg(b->t), 1, OIN, OIN, BR);
        break;
    case K_UO: {
        LoraUp u = {warg(b, W_BO, l), gpu_buf_arg(b->yo), HID, 0, HID};
        up_n(b, c, &u, 1, BR);
        break;
    }
    case K_DD:
        lora_encode_down(b->p, c, warg(b, W_D, l), gpu_buf_arg(b->x17), gpu_buf_arg(b->t), 1, FF, FF, BR);
        break;
    case K_UD: {
        LoraUp u = {warg(b, W_BD, l), gpu_buf_arg(b->yd), HID, 0, HID};
        up_n(b, c, &u, 1, BR);
        break;
    }
    }
}

static bool k_attn(int k) { return k == K_DQKV || k == K_UQKV || k == K_DO || k == K_UO; }

// Weight bytes read by op k on one layer.
static double k_bytes(int k) {
    switch (k) {
    case K_DQKV: return (double)wbytes(W_QKV);
    case K_UQKV: return (double)(wbytes(W_BQ) + wbytes(W_BK) + wbytes(W_BV));
    case K_DGU: return (double)wbytes(W_GU);
    case K_UGU: return (double)(wbytes(W_BG) + wbytes(W_BU));
    case K_DO: return (double)wbytes(W_O);
    case K_UO: return (double)wbytes(W_BO);
    case K_DD: return (double)wbytes(W_D);
    default: return (double)wbytes(W_BD);
    }
}

static void bench_run(const void *ctx, sx_cmd *c) {
    const Bench *b = ctx;
    if (b->kind >= 0) {
        size_t n = k_attn(b->kind) ? NATTN : NLAYERS;
        for (size_t l = 0; l < n; l++) op(b, c, b->kind, l);
        return;
    }
    size_t a = 0;
    for (size_t l = 0; l < NLAYERS; l++) { // every 4th layer has full attention
        bool attn = l % 4 == 3;
        if (attn) {
            op(b, c, K_DQKV, a);
            op(b, c, K_UQKV, a);
            op(b, c, K_DO, a);
            op(b, c, K_UO, a);
            a++;
        }
        op(b, c, K_DGU, l);
        op(b, c, K_UGU, l);
        op(b, c, K_DD, l);
        op(b, c, K_UD, l);
    }
}

Err lora_bench(Arena *a, FILE *out) {
    QgenCtx x;
    TRY(qgen_ctx_init(a, &x));
    LoraPipes p;
    TRY(lora_init(&x.g, &p));
    Bench b = {.p = &p};
    double total_bytes = 0;
    for (int k = 0; k < NW; k++) {
        TRY(qgen_ctx_halfs(&x, w_rows[k] * w_cols[k] * w_layers[k], &b.w[k]));
        total_bytes += (double)(wbytes(k) * w_layers[k]);
    }
    TRY(qgen_ctx_floats(&x, HID, -1, 1, &b.x5));
    TRY(qgen_ctx_floats(&x, OIN, -1, 1, &b.x6));
    TRY(qgen_ctx_floats(&x, FF, -1, 1, &b.x17));
    TRY(qgen_ctx_floats(&x, 3 * BR, -1, 1, &b.t));
    sx_buffer **ys[7] = {&b.yq, &b.yk, &b.yv, &b.yg, &b.yu, &b.yo, &b.yd};
    const size_t yn[7] = {QOUT, KVOUT, KVOUT, FF, FF, HID, HID};
    for (int i = 0; i < 7; i++) TRY(gpu_new_buffer(&x.g, yn[i] * 4, ys[i]));
    fprintf(out, "lora-bench: Qwen3.8-27B decode (T 1, R 32), distinct f16 weights per layer, %.1f MB per token\n",
            total_bytes / 1e6);
    double s;
    for (int k = 0; k < NK; k++) {
        size_t n = k_attn(k) ? NATTN : NLAYERS;
        for (int f = 1; f >= 0; f--) {
            if (!f && k != K_UQKV && k != K_UGU) continue;
            b.kind = k;
            b.fused = f;
            TRY(timed(&x.g, &b, bench_run, &s));
            fprintf(out, "%-34s %s x%2zu %7.2f us/call %6.1f GB/s\n", k_name[k], f ? "fused  " : "1 by 1 ", n,
                    s * 1e6 / (double)n, k_bytes(k) * (double)n / s / 1e9);
            fflush(out);
        }
    }
    for (int f = 1; f >= 0; f--) {
        b.kind = -1;
        b.fused = f;
        TRY(timed(&x.g, &b, bench_run, &s));
        size_t calls = f ? NATTN * 4 + NLAYERS * 4 : NATTN * 6 + NLAYERS * 5;
        fprintf(out, "token: 64 layers (16 attention), %s, %zu dispatches: %7.1f us/token %6.1f GB/s\n",
                f ? "fused ups" : "ups 1 by 1", calls, s * 1e6, total_bytes / s / 1e9);
        fflush(out);
    }
    for (int k = 0; k < NW; k++) sx_buffer_release(b.w[k]);
    sx_buffer *bufs[11] = {b.x5, b.x6, b.x17, b.t, b.yq, b.yk, b.yv, b.yg, b.yu, b.yo, b.yd};
    for (int i = 0; i < 11; i++) sx_buffer_release(bufs[i]);
    return OK;
}
