// DeepSeek-V4-Flash FP8 dense linears: kernel source, pipelines and encode functions
// (see dsv4_fp8.h).
#include "dsv4_fp8.h"

const char dsv4_fp8_src[] = {
#embed "kernels/dsv4_fp8.metal"
    , 0};

// Constant structs of dsv4_fp8.metal.
typedef struct {
    uint32_t K, ldx_in;
} Dsv4AqP;
typedef struct {
    uint32_t K, N, T, ldx, rpg, ldy, wshift, sshift;
} Dsv4MvP;
typedef struct {
    uint32_t K, N, M, ldx, rpg, ldy, wshift, sshift;
} Dsv4MmP;

enum { MV_ROWS = 16, MV_TG = 128, MM_TILE = 64, MM_TG = 128 };

Err dsv4_fp8_init(Gpu *g, Dsv4Fp8Pipes *p) {
    static const char *const mvq[4] = {"dsv4_fp8_mv_q1", "dsv4_fp8_mv_q2", "dsv4_fp8_mv_q4", "dsv4_fp8_mv_q8"};
    static const char *const mvr[4] = {"dsv4_fp8_mv_r1", "dsv4_fp8_mv_r2", "dsv4_fp8_mv_r4", "dsv4_fp8_mv_r8"};
    TRY(gpu_pipeline(g, dsv4_fp8_src, "dsv4_act_quant", &p->aq));
    for (size_t i = 0; i < 4; i++) {
        TRY(gpu_pipeline(g, dsv4_fp8_src, mvq[i], &p->mv_q[i]));
        TRY(gpu_pipeline(g, dsv4_fp8_src, mvr[i], &p->mv_r[i]));
    }
    TRY(gpu_pipeline(g, dsv4_fp8_src, "dsv4_fp8_mm_q", &p->mm_q));
    TRY(gpu_pipeline(g, dsv4_fp8_src, "dsv4_fp8_mm_r", &p->mm_r));
    return OK;
}

static size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

void dsv4_encode_act_quant(const Dsv4Fp8Pipes *p, sx_cmd *c, sx_arg x, uint32_t ldx_in, sx_arg xc, sx_arg xs,
                           uint32_t T, uint32_t K) {
    if (T == 0) return;
    Dsv4AqP ap = {K, ldx_in};
    sx_arg a[4] = {x, xc, xs, gpu_bytes_arg(&ap, sizeof ap)};
    sx_cmd_dispatch(c, p->aq, a, 4, K / 128, T, 32);
}

void dsv4_encode_mv(const Dsv4Fp8Pipes *p, sx_cmd *c, const Dsv4Fp8Lin *l, bool raw, sx_arg x, sx_arg xs,
                    uint32_t ldx, uint32_t rpg, sx_arg y, uint32_t T, uint32_t ldy) {
    if (T == 0) return;
    // the smallest pass covering T, passes of 8 above
    const size_t pi = T <= 1 ? 0 : T <= 2 ? 1 : T <= 4 ? 2 : 3;
    const uint32_t tt = 1u << pi;
    Dsv4MvP mp = {l->in, l->out, T, ldx, rpg, ldy, l->wshift, l->sshift};
    sx_arg a[6] = {l->w, l->s, x, raw ? x : xs, y, gpu_bytes_arg(&mp, sizeof mp)};
    sx_cmd_dispatch(c, raw ? p->mv_r[pi] : p->mv_q[pi], a, 6, l->out / MV_ROWS, cdiv(T, tt), MV_TG);
}

void dsv4_encode_mm(const Dsv4Fp8Pipes *p, sx_cmd *c, const Dsv4Fp8Lin *l, bool raw, sx_arg x, sx_arg xs,
                    uint32_t ldx, uint32_t rpg, sx_arg y, uint32_t T, uint32_t ldy) {
    if (T == 0) return;
    Dsv4MmP mp = {l->in, l->out, T, ldx, rpg, ldy, l->wshift, l->sshift};
    sx_arg a[6] = {l->w, l->s, x, raw ? x : xs, y, gpu_bytes_arg(&mp, sizeof mp)};
    sx_cmd_dispatch(c, raw ? p->mm_r : p->mm_q, a, 6, l->out / MM_TILE, cdiv(T, MM_TILE), MM_TG);
}

void dsv4_encode_linear(const Dsv4Fp8Pipes *p, sx_cmd *c, const Dsv4Fp8Lin *l, sx_arg xc, sx_arg xs, sx_arg y,
                        uint32_t T, uint32_t ldy) {
    if (T <= DSV4_FP8_SMALL_T) dsv4_encode_mv(p, c, l, false, xc, xs, l->in, l->out, y, T, ldy);
    else dsv4_encode_mm(p, c, l, false, xc, xs, l->in, l->out, y, T, ldy);
}

void dsv4_encode_grouped(const Dsv4Fp8Pipes *p, sx_cmd *c, const Dsv4Fp8Lin *l, uint32_t groups, sx_arg x, sx_arg y,
                         uint32_t T, uint32_t ldy) {
    const uint32_t ldx = groups * l->in, rpg = l->out / groups;
    if (T <= DSV4_FP8_SMALL_T) dsv4_encode_mv(p, c, l, true, x, x, ldx, rpg, y, T, ldy);
    else dsv4_encode_mm(p, c, l, true, x, x, ldx, rpg, y, T, ldy);
}
