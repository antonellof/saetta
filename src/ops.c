// Operations (see ops.h).
#include "ops.h"

#include <math.h>
#include <stdlib.h>

const char ops_qmatvec_src[] = {
#embed "kernels/qmatvec.metal"
    , 0};
const char ops_qmm_src[] = {
#embed "kernels/qmm.metal"
    , 0};

static inline float bf16_at(const uint8_t *bytes, size_t i) {
    return ops_bf16_to_f32((uint16_t)(bytes[i * 2] | (bytes[i * 2 + 1] << 8)));
}

void ops_qmatvec_ref(const Pack *pack, const Linear *lin, const float *x, float *y) {
    const uint8_t *w = pack_data(pack, &lin->weight);
    const uint8_t *s = pack_data(pack, &lin->scales);
    const uint8_t *b = pack_data(pack, &lin->biases);
    uint32_t per_word = 32 / lin->bits;
    uint32_t mask = (1u << lin->bits) - 1;
    size_t in = (size_t)lin->in_features;
    size_t groups = in / lin->group_size;
    size_t words_per_row = in / per_word;
    for (size_t o = 0; o < (size_t)lin->out_features; o++) {
        double acc = 0;
        for (size_t i = 0; i < in; i++) {
            uint32_t word;
            memcpy(&word, w + (o * words_per_row + i / per_word) * 4, 4); // little-endian host
            uint32_t q = (word >> (lin->bits * (i % per_word))) & mask;
            size_t g = o * groups + i / lin->group_size;
            float wv = (float)q * bf16_at(s, g) + bf16_at(b, g);
            acc += (double)wv * (double)x[i];
        }
        y[o] = (float)acc;
    }
}

Err ops_init(Gpu *g, Ops *out) {
    TRY(gpu_pipeline(g, ops_qmatvec_src, "qmatvec_4_32", &out->q4));
    TRY(gpu_pipeline(g, ops_qmatvec_src, "qmatvec_8_64", &out->q8));
    TRY(gpu_pipeline(g, ops_qmm_src, "qmm_4_32", &out->mm4));
    TRY(gpu_pipeline(g, ops_qmm_src, "qmm_8_64", &out->mm8));
    return OK;
}

// The 8 arguments of the qmatvec kernels; in, out and sh must outlive the dispatch.
static void mv_args(const Gpu *g, const Linear *lin, sx_buffer *x, sx_buffer *y, const uint32_t *in,
                    const uint32_t *out, uint32_t sh[3], sx_arg args[8]) {
    args[0] = gpu_tensor_arg(g, &lin->weight, &sh[0]);
    args[1] = gpu_tensor_arg(g, &lin->scales, &sh[1]);
    args[2] = gpu_tensor_arg(g, &lin->biases, &sh[2]);
    args[3] = gpu_buf_arg(x);
    args[4] = gpu_buf_arg(y);
    args[5] = gpu_u32_arg(in);
    args[6] = gpu_u32_arg(out);
    args[7] = gpu_bytes_arg(sh, 12);
}

// The 7 arguments of the qmm / rows kernels (dims = in, out, T).
static void mm_args(const Gpu *g, const Linear *lin, sx_buffer *x, sx_buffer *y, const uint32_t dims[3],
                    uint32_t sh[3], sx_arg args[7]) {
    args[0] = gpu_tensor_arg(g, &lin->weight, &sh[0]);
    args[1] = gpu_tensor_arg(g, &lin->scales, &sh[1]);
    args[2] = gpu_tensor_arg(g, &lin->biases, &sh[2]);
    args[3] = gpu_buf_arg(x);
    args[4] = gpu_buf_arg(y);
    args[5] = gpu_bytes_arg(dims, 12);
    args[6] = gpu_bytes_arg(sh, 12);
}

Err ops_qmatvec_timed(const Ops *o, Gpu *g, const Linear *lin, sx_buffer *x, sx_buffer *y, size_t reps, double *secs) {
    sx_pipeline *p = lin->bits == 8 ? o->q8 : o->q4;
    uint32_t in = (uint32_t)lin->in_features, out = (uint32_t)lin->out_features;
    uint32_t sh[3];
    sx_arg args[8];
    mv_args(g, lin, x, y, &in, &out, sh, args);
    sx_cmd *cmd = sx_cmd_begin(g->dev);
    if (!cmd) return "DispatchFailed";
    for (size_t i = 0; i < reps; i++) sx_cmd_dispatch(cmd, p, args, 8, out, 1, 32);
    *secs = 0;
    char err[256];
    if (sx_cmd_finish_timed(cmd, secs, err, sizeof err) != 0) return "DispatchFailed";
    return OK;
}

Err ops_qmatvec_timed_many(const Ops *o, Gpu *g, const Linear *lins, size_t n, sx_buffer *x, sx_buffer *y,
                           uint32_t lanes, double *secs) {
    sx_pipeline *lp = NULL;
    if (lanes != 0)
        TRY(gpu_pipeline(g, ops_qmatvec_src, lanes == 8 ? "qmv4_l8" : lanes == 64 ? "qmv4_l64" : "qmv4_l256", &lp));
    sx_cmd *cmd = sx_cmd_begin(g->dev);
    if (!cmd) return "DispatchFailed";
    for (size_t k = 0; k < n; k++) {
        const Linear *lin = &lins[k];
        sx_pipeline *p = lp ? lp : lin->bits == 8 ? o->q8 : o->q4;
        uint32_t in = (uint32_t)lin->in_features, out = (uint32_t)lin->out_features;
        uint32_t sh[3];
        sx_arg args[8];
        mv_args(g, lin, x, y, &in, &out, sh, args);
        if (lp) sx_cmd_dispatch(cmd, p, args, 8, (out + 256 / lanes - 1) / (256 / lanes), 1, 256);
        else sx_cmd_dispatch(cmd, p, args, 8, out, 1, 32);
    }
    *secs = 0;
    char err[256];
    int rc = sx_cmd_finish_timed(cmd, secs, err, sizeof err);
    if (lp) sx_pipeline_release(lp);
    return rc != 0 ? "DispatchFailed" : OK;
}

Err ops_rows_timed_many(const Ops *o, Gpu *g, const Linear *lins, size_t n, sx_buffer *x, sx_buffer *y, uint32_t T,
                        double *secs) {
    // SAETTA_MV_ROWS=1: the shared-input layout (T >= 3 in decode); =one: one row per
    // simdgroup (qmv_rows1, the T <= 2 kernel)
    const char *v = getenv("SAETTA_MV_ROWS");
    bool one = v && v[0] == 'o';
    char nb[32];
    int len = snprintf(nb, sizeof nb, "qmv_rows%s_%u_%u_%u", one ? "1" : "", lins[0].bits, lins[0].group_size, T);
    if (len < 0 || (size_t)len >= sizeof nb) return "NoSpaceLeft";
    sx_pipeline *p;
    TRY(gpu_pipeline(g, ops_qmatvec_src, nb, &p));
    sx_cmd *cmd = sx_cmd_begin(g->dev);
    if (!cmd) return "DispatchFailed";
    for (size_t k = 0; k < n; k++) {
        const Linear *lin = &lins[k];
        uint32_t dims[3] = {(uint32_t)lin->in_features, (uint32_t)lin->out_features, T};
        uint32_t sh[3];
        sx_arg args[7];
        mm_args(g, lin, x, y, dims, sh, args);
        if (one) sx_cmd_dispatch(cmd, p, args, 7, dims[1], 1, 32);
        else sx_cmd_dispatch(cmd, p, args, 7, (dims[1] + 7) / 8, 1, 64);
    }
    *secs = 0;
    char err[256];
    int rc = sx_cmd_finish_timed(cmd, secs, err, sizeof err);
    sx_pipeline_release(p);
    return rc != 0 ? "DispatchFailed" : OK;
}

Err ops_qmm(const Ops *o, Gpu *g, const Linear *lin, sx_buffer *x, sx_buffer *y, uint32_t T) {
    sx_pipeline *p = lin->bits == 8 ? o->mm8 : o->mm4;
    uint32_t dims[3] = {(uint32_t)lin->in_features, (uint32_t)lin->out_features, T};
    uint32_t sh[3];
    sx_arg args[7];
    mm_args(g, lin, x, y, dims, sh, args);
    sx_cmd *cmd = sx_cmd_begin(g->dev);
    if (!cmd) return "DispatchFailed";
    sx_cmd_dispatch(cmd, p, args, 7, (dims[1] + 31) / 32, (T + 31) / 32, 128);
    char err[512];
    if (sx_cmd_finish(cmd, err, sizeof err) != 0) {
        dbg("qmm: %s\n", err);
        return "DispatchFailed";
    }
    return OK;
}

Err ops_qmatvec(const Ops *o, Gpu *g, const Linear *lin, sx_buffer *x, sx_buffer *y) {
    sx_pipeline *p;
    if (lin->bits == 4 && lin->group_size == 32) p = o->q4;
    else if (lin->bits == 8 && lin->group_size == 64) p = o->q8;
    else return "UnsupportedQuant";
    uint32_t in = (uint32_t)lin->in_features, out = (uint32_t)lin->out_features;
    uint32_t sh[3];
    sx_arg args[8];
    mv_args(g, lin, x, y, &in, &out, sh, args);
    char err[512];
    if (sx_dispatch_sync(g->dev, p, args, 8, out, 32, err, sizeof err) != 0) {
        dbg("qmatvec: %s\n", err);
        return "DispatchFailed";
    }
    return OK;
}

float ops_max_rel_err(const float *a, const float *b, size_t n) {
    float m = 0, scale = 1e-30f;
    for (size_t i = 0; i < n; i++) scale = fmaxf(scale, fabsf(b[i]));
    for (size_t i = 0; i < n; i++) m = fmaxf(m, fabsf(a[i] - b[i]));
    return m / scale;
}
