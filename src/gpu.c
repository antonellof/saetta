// GPU context (see gpu.h).
#include "gpu.h"

Err gpu_init(Arena *a, const Pack *pack, Gpu *out) {
    sx_device *dev = sx_device_create();
    if (!dev) return "NoMetalDevice";
    size_t page = sx_page_size();
    sx_buffer **bufs = ARENA_ARRAY(a, sx_buffer *, pack->nfiles ? pack->nfiles : 1);
    for (size_t i = 0; i < pack->nfiles; i++) {
        const MappedFile *f = &pack->files[i];
        bufs[i] = sx_buffer_wrap(dev, f->bytes, ALIGN_UP(f->len, page));
        if (!bufs[i]) return "WrapFailed";
    }
    *out = (Gpu){dev, bufs, pack->nfiles};
    return OK;
}

void gpu_deinit(Gpu *g) {
    for (size_t i = 0; i < g->nfile_buffers; i++) sx_buffer_release(g->file_buffers[i]);
    sx_device_destroy(g->dev);
}

sx_arg gpu_tensor_arg(const Gpu *g, const TensorInfo *t, uint32_t *shift) {
    uint64_t base = t->offset & ~(uint64_t)15;
    *shift = (uint32_t)(t->offset - base);
    return (sx_arg){g->file_buffers[t->file], base, NULL, 0};
}

Err gpu_warm(Gpu *g) { return gpu_warm_except(g, NULL, ""); }

static const char touch_src[] =
    "#include <metal_stdlib>\n"
    "kernel void touch(device const uint *b [[buffer(0)]], device metal::atomic_uint *sink [[buffer(1)]],\n"
    "                  constant uint2 &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n"
    "    if (i < p.x && b[(ulong)i * p.y] == 0x9e3779b9u) atomic_fetch_add_explicit(sink, 1u, metal::memory_order_relaxed);\n"
    "}";

Err gpu_warm_except(Gpu *g, const Pack *pack, const char *skip) {
    sx_pipeline *pl;
    TRY(gpu_pipeline(g, touch_src, "touch", &pl));
    sx_buffer *sink;
    Err e = gpu_new_buffer(g, 16, &sink);
    if (e) {
        sx_pipeline_release(pl);
        return e;
    }
    size_t page = sx_page_size();
    sx_cmd *cmd = sx_cmd_begin(g->dev);
    if (!cmd) {
        sx_buffer_release(sink);
        sx_pipeline_release(pl);
        return "GpuFailed";
    }
    if (pack) {
        // per tensor, page aligned, skipping names that contain skip
        size_t skip_len = strlen(skip);
        Str key;
        void *val;
        for (size_t it = 0; strmap_next(&pack->tensors, &it, &key, &val);) {
            if (skip_len > 0 && str_find(key, str_n(skip, skip_len), 0) >= 0) continue;
            const TensorInfo *t = val;
            uint64_t first = t->offset / page * page;
            uint32_t pages = (uint32_t)((t->offset + t->nbytes - first + page - 1) / page);
            uint32_t p[2] = {pages, (uint32_t)(page / 4)};
            sx_arg args[3] = {{g->file_buffers[t->file], first, NULL, 0}, gpu_buf_arg(sink), gpu_bytes_arg(p, 8)};
            sx_cmd_dispatch(cmd, pl, args, 3, (pages + 255) / 256, 1, 256);
        }
    } else {
        for (size_t i = 0; i < g->nfile_buffers; i++) {
            sx_buffer *b = g->file_buffers[i];
            uint32_t pages = (uint32_t)(sx_buffer_length(b) / page);
            uint32_t p[2] = {pages, (uint32_t)(page / 4)};
            sx_arg args[3] = {gpu_buf_arg(b), gpu_buf_arg(sink), gpu_bytes_arg(p, 8)};
            sx_cmd_dispatch(cmd, pl, args, 3, (pages + 255) / 256, 1, 256);
        }
    }
    char err[512];
    int rc = sx_cmd_finish(cmd, err, sizeof err);
    sx_buffer_release(sink);
    sx_pipeline_release(pl);
    if (rc != 0) {
        dbg("warm: %s\n", err);
        return "GpuFailed";
    }
    return OK;
}

static const char nop_src[] =
    "kernel void nop(device float *b [[buffer(0)]], uint i [[thread_position_in_grid]]) { if (i == 0) b[0] += 1.0f; }";

Err gpu_dispatch_cost(Gpu *g, size_t n, size_t serial_threads, double *secs) {
    sx_pipeline *pl;
    TRY(gpu_pipeline(g, nop_src, "nop", &pl));
    sx_buffer *buf;
    Err e = gpu_new_buffer(g, 4096, &buf);
    if (e) {
        sx_pipeline_release(pl);
        return e;
    }
    sx_cmd *cmd = sx_cmd_begin(g->dev);
    int rc = 1;
    char err[256];
    *secs = 0;
    if (cmd) {
        sx_arg args[1] = {gpu_buf_arg(buf)};
        for (size_t i = 0; i < n; i++) sx_cmd_dispatch(cmd, pl, args, 1, (serial_threads + 255) / 256, 1, 256);
        rc = sx_cmd_finish_timed(cmd, secs, err, sizeof err);
    }
    sx_buffer_release(buf);
    sx_pipeline_release(pl);
    return rc != 0 ? "GpuFailed" : OK;
}

Err gpu_pipeline(Gpu *g, const char *source, const char *name, sx_pipeline **out) {
    char err[2048];
    err[0] = 0;
    *out = sx_pipeline_from_source(g->dev, source, name, err, sizeof err);
    if (!*out) {
        dbg("kernel %s: %s\n", name, err);
        return "CompileFailed";
    }
    return OK;
}

Err gpu_add_file(Gpu *g, Arena *a, sx_buffer *b, uint16_t *index) {
    sx_buffer **n = ARENA_ARRAY(a, sx_buffer *, g->nfile_buffers + 1);
    if (g->nfile_buffers) memcpy(n, g->file_buffers, g->nfile_buffers * sizeof *n);
    n[g->nfile_buffers] = b;
    g->file_buffers = n;
    g->nfile_buffers++;
    *index = (uint16_t)(g->nfile_buffers - 1);
    return OK;
}

Err gpu_new_buffer(Gpu *g, size_t len, sx_buffer **out) {
    *out = sx_buffer_new(g->dev, len);
    return *out ? OK : "AllocFailed";
}
