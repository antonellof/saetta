// Run-time LoRA kernels (src/lora.h) against the scalar f64 references, on random f16 weights.
// No model pack needed.
#include <math.h>
#include <stdlib.h>

#include "../lora.h"
#include "../ops.h"
#include "../qgen.h"
#include "../test.h"
#include "../tolerances.h"

static const size_t Ts[] = {1, 3, 8, 40, 512}; // 40: partial mm tiles

static Err run(QgenCtx *x, void (*enc)(const void *, sx_cmd *), const void *ctx) {
    sx_cmd *c = sx_cmd_begin(x->g.dev);
    if (!c) return "DispatchFailed";
    enc(ctx, c);
    char err[512];
    if (sx_cmd_finish(c, err, sizeof err) != 0) {
        dbg("dispatch: %s\n", err);
        return "DispatchFailed";
    }
    return OK;
}

typedef struct {
    const LoraPipes *p;
    sx_buffer *A, *x, *t;
    uint32_t T, in, ldx, rt;
} DownCtx;

static void enc_down(const void *ctx, sx_cmd *c) {
    const DownCtx *d = ctx;
    lora_encode_down(d->p, c, gpu_buf_arg(d->A), gpu_buf_arg(d->x), gpu_buf_arg(d->t), d->T, d->in, d->ldx, d->rt);
}

TEST(lora_down_matches_the_scalar_reference) {
    Arena arena = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&arena, &x));
    LoraPipes p;
    EXPECT_OK(lora_init(&x.g, &p));
    static const size_t ins[] = {5120, 6144, 17408};
    static const size_t rts[] = {32, 40, 96}; // 40: a partial tile of A rows
    float worst_small = 0, worst_mm = 0;
    for (size_t ii = 0; ii < COUNT(ins); ii++)
        for (size_t ri = 0; ri < COUNT(rts); ri++)
            for (size_t ti = 0; ti < COUNT(Ts); ti++) {
                const size_t in = ins[ii], rt = rts[ri], T = Ts[ti];
                const size_t ldx = in + (ti % 2) * 4; // padded rows on half the cases
                sx_buffer *A, *xb, *tb;
                EXPECT_OK(qgen_ctx_halfs(&x, rt * in, &A));
                EXPECT_OK(qgen_ctx_floats(&x, T * ldx, -1, 1, &xb));
                EXPECT_OK(gpu_new_buffer(&x.g, T * rt * 4, &tb));
                DownCtx d = {&p, A, xb, tb, (uint32_t)T, (uint32_t)in, (uint32_t)ldx, (uint32_t)rt};
                EXPECT_OK(run(&x, enc_down, &d));
                float *ref = ARENA_ARRAY(&arena, float, T * rt);
                lora_ref_down(sx_buffer_contents(A), gpu_floats(xb), ref, T, in, ldx, rt);
                float e = ops_max_rel_err(gpu_floats(tb), ref, T * rt);
                if (!(e < TOL_LORA_REL)) {
                    dbg("lora_down in %zu Rt %zu T %zu: max rel err %g (bound %g)\n", in, rt, T, (double)e,
                        (double)TOL_LORA_REL);
                    *failed_ = 1;
                    return;
                }
                if (T <= LORA_SMALL_T) worst_small = fmaxf(worst_small, e);
                else worst_mm = fmaxf(worst_mm, e);
                sx_buffer_release(A);
                sx_buffer_release(xb);
                sx_buffer_release(tb);
            }
    dbg("lora_down: worst max rel err %g (small), %g (mm)\n", (double)worst_small, (double)worst_mm);
    arena_reset(&arena);
}

typedef struct {
    const LoraPipes *p;
    sx_buffer *t;
    uint32_t T, rt, R;
    LoraUp u[3];
    size_t n;
} UpCtx;

static void enc_up(const void *ctx, sx_cmd *c) {
    const UpCtx *d = ctx;
    lora_encode_up(d->p, c, gpu_buf_arg(d->t), d->T, d->rt, d->R, 2.0f, d->u, d->n);
}

// out sizes per target, t of rt values per row, slices r0[j] .. r0[j] + R; y_j rows ldy_j apart
// (out + pad), bound at byte offset yoff.
static bool up_case(QgenCtx *x, Arena *arena, const LoraPipes *p, size_t T, size_t rt, size_t R, const size_t *outs,
                    const size_t *r0s, size_t n, size_t pad, size_t yoff, float *worst) {
    sx_buffer *tb, *B[3], *y[3];
    float *y0[3];
    if (qgen_ctx_floats(x, T * rt, -1, 1, &tb)) return false;
    UpCtx d = {p, tb, (uint32_t)T, (uint32_t)rt, (uint32_t)R, {{0}}, n};
    for (size_t j = 0; j < n; j++) {
        const size_t ldy = outs[j] + pad;
        if (qgen_ctx_halfs(x, outs[j] * R, &B[j])) return false;
        if (qgen_ctx_floats(x, yoff / 4 + T * ldy, -1, 1, &y[j])) return false;
        y0[j] = ARENA_ARRAY(arena, float, T * ldy);
        memcpy(y0[j], gpu_floats(y[j]) + yoff / 4, T * ldy * 4);
        d.u[j] = (LoraUp){gpu_buf_arg(B[j]), {y[j], yoff, NULL, 0}, (uint32_t)outs[j], (uint32_t)r0s[j], (uint32_t)ldy};
    }
    if (run(x, enc_up, &d)) return false;
    bool ok = true;
    for (size_t j = 0; j < n; j++) {
        const size_t ldy = outs[j] + pad;
        lora_ref_up(sx_buffer_contents(B[j]), gpu_floats(tb), y0[j], T, rt, r0s[j], R, outs[j], ldy, 2.0f);
        float e = ops_max_rel_err(gpu_floats(y[j]) + yoff / 4, y0[j], T * ldy);
        if (!(e < TOL_LORA_REL)) {
            dbg("lora_up out %zu R %zu r0 %zu Rt %zu T %zu (target %zu of %zu): max rel err %g (bound %g)\n", outs[j], R,
                r0s[j], rt, T, j, n, (double)e, (double)TOL_LORA_REL);
            ok = false;
        }
        *worst = fmaxf(*worst, e);
        sx_buffer_release(B[j]);
        sx_buffer_release(y[j]);
    }
    sx_buffer_release(tb);
    return ok;
}

TEST(lora_up_add_matches_the_scalar_reference) {
    Arena arena = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&arena, &x));
    LoraPipes p;
    EXPECT_OK(lora_init(&x.g, &p));
    static const size_t outs[] = {1024, 5120, 12288, 17408};
    float worst_small = 0, worst_mm = 0;
    for (size_t oi = 0; oi < COUNT(outs); oi++)
        for (size_t ti = 0; ti < COUNT(Ts); ti++) {
            const size_t T = Ts[ti];
            float *w = T <= LORA_SMALL_T ? &worst_small : &worst_mm;
            // R 32 slices of a q/k/v t row (Rt 96) at r0 0, 32, 64; R 96 at r0 0
            for (size_t r0 = 0; r0 < 96; r0 += 32)
                EXPECT(up_case(&x, &arena, &p, T, 96, 32, &outs[oi], &r0, 1, (r0 / 32) * 4, (r0 / 32) * 16, w));
            static const size_t z = 0;
            EXPECT(up_case(&x, &arena, &p, T, 96, 96, &outs[oi], &z, 1, 0, 0, w));
        }
    // three targets in one dispatch (q/k/v), and two (gate/up)
    static const size_t qkv[3] = {12288, 1024, 1024}, qkv_r0[3] = {0, 32, 64};
    static const size_t gu[2] = {17408, 17408}, gu_r0[2] = {0, 32};
    for (size_t ti = 0; ti < COUNT(Ts); ti++) {
        const size_t T = Ts[ti];
        float *w = T <= LORA_SMALL_T ? &worst_small : &worst_mm;
        EXPECT(up_case(&x, &arena, &p, T, 96, 32, qkv, qkv_r0, 3, 0, 0, w));
        EXPECT(up_case(&x, &arena, &p, T, 64, 32, gu, gu_r0, 2, 4, 16, w));
    }
    dbg("lora_up_add: worst max rel err %g (small), %g (mm)\n", (double)worst_small, (double)worst_mm);
    arena_reset(&arena);
}
