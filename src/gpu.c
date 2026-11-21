// GPU context (see gpu.h).
#include "gpu.h"

#include <stdlib.h>
#include <string.h>

#include "xcache.h"

// A trunk routed-expert tensor, read from the files by the expert cache when streaming:
// Flash-Next's stacked mlp.switch_mlp.* tensors, or DeepSeek-V4's per-expert
// layers.N.ffn.experts.E.* (not its shared experts, not the draft head's mtp.*).
bool gpu_streamed_tensor(Str name) {
    if (str_find(name, S("mtp."), 0) == 0) return false;
    return str_find(name, S("mlp.switch_mlp."), 0) >= 0 ||
           (str_starts(name, S("layers.")) && str_find(name, S(".ffn.experts."), 0) >= 0);
}
#define streamed_tensor gpu_streamed_tensor

bool gpu_pack_always_streams(const Pack *pack) {
    return pack_get(pack, S("layers.0.ffn.experts.0.w1.weight")) != NULL;
}

typedef struct {
    uint64_t a, b;
} Hole;

static int hole_cmp(const void *x, const void *y) {
    const Hole *p = x, *q = y;
    return (p->a > q->a) - (p->a < q->a);
}

// Wraps file i as the segments between its streamed tensors (each hole shrunk to whole pages).
static Err wrap_segments(Arena *a, sx_device *dev, const Pack *pack, size_t i, GpuSeg **out, uint32_t *nout) {
    const MappedFile *f = &pack->files[i];
    size_t page = sx_page_size();
    size_t nh = 0;
    Str key;
    void *val;
    for (size_t it = 0; strmap_next(&pack->tensors, &it, &key, &val);) {
        const TensorInfo *t = val;
        if (t->file == i && streamed_tensor(key)) nh++;
    }
    Hole *h = ARENA_ARRAY(a, Hole, nh ? nh : 1);
    size_t k = 0;
    for (size_t it = 0; strmap_next(&pack->tensors, &it, &key, &val);) {
        const TensorInfo *t = val;
        if (t->file != i || !streamed_tensor(key)) continue;
        uint64_t s = ALIGN_UP(t->offset, page), e = (t->offset + t->nbytes) / page * page;
        if (e > s) h[k++] = (Hole){s, e};
    }
    nh = k;
    qsort(h, nh, sizeof *h, hole_cmp);
    GpuSeg *seg = ARENA_ARRAY(a, GpuSeg, nh + 1);
    uint32_t ns = 0;
    uint64_t cur = 0, flen = ALIGN_UP(f->len, page);
    for (size_t j = 0; j <= nh; j++) {
        uint64_t end = j < nh ? h[j].a : flen;
        if (end > cur) {
            sx_buffer *b = sx_buffer_wrap(dev, f->bytes + cur, (size_t)(end - cur));
            if (!b) return "WrapFailed";
            seg[ns++] = (GpuSeg){cur, end, b};
        }
        if (j < nh && h[j].b > cur) cur = h[j].b;
    }
    *out = seg;
    *nout = ns;
    return OK;
}

Err gpu_init(Arena *a, const Pack *pack, Gpu *out) {
    sx_device *dev = sx_device_create();
    if (!dev) return "NoMetalDevice";
    size_t page = sx_page_size();
    size_t nf = pack->nfiles ? pack->nfiles : 1;
    sx_buffer **bufs = ARENA_ARRAY(a, sx_buffer *, nf);
    bool stream = xc_requested_bytes() != 0 || gpu_pack_always_streams(pack);
    GpuSeg **segs = stream ? ARENA_ARRAY(a, GpuSeg *, nf) : NULL;
    uint32_t *nseg = stream ? ARENA_ARRAY(a, uint32_t, nf) : NULL;
    for (size_t i = 0; i < pack->nfiles; i++) {
        const MappedFile *f = &pack->files[i];
        if (stream) {
            TRY(wrap_segments(a, dev, pack, i, &segs[i], &nseg[i]));
            bufs[i] = nseg[i] ? segs[i][0].buf : NULL;
            continue;
        }
        bufs[i] = sx_buffer_wrap(dev, f->bytes, ALIGN_UP(f->len, page));
        if (!bufs[i]) return "WrapFailed";
    }
    *out = (Gpu){.dev = dev, .file_buffers = bufs, .nfile_buffers = pack->nfiles, .segs = segs, .nseg = nseg,
                 .nseg_files = stream ? pack->nfiles : 0};
    return OK;
}

// The buffer holding byte off of file `file` and the offset into it.
static sx_arg file_arg(const Gpu *g, size_t file, uint64_t off) {
    if (file < g->nseg_files) {
        const GpuSeg *s = g->segs[file];
        uint32_t lo = 0, hi = g->nseg[file];
        while (hi - lo > 1) {
            uint32_t mid = (lo + hi) / 2;
            if (s[mid].start <= off) lo = mid;
            else hi = mid;
        }
        return (sx_arg){s[lo].buf, off - s[lo].start, NULL, 0};
    }
    return (sx_arg){g->file_buffers[file], off, NULL, 0};
}

void gpu_deinit(Gpu *g) {
    for (size_t i = 0; i < g->nfile_buffers; i++)
        if (i >= g->nseg_files) sx_buffer_release(g->file_buffers[i]);
    for (size_t i = 0; i < g->nseg_files; i++)
        for (uint32_t j = 0; j < g->nseg[i]; j++) sx_buffer_release(g->segs[i][j].buf);
    sx_device_destroy(g->dev);
}

sx_arg gpu_tensor_arg(const Gpu *g, const TensorInfo *t, uint32_t *shift) {
    uint64_t base = t->offset & ~(uint64_t)15;
    *shift = (uint32_t)(t->offset - base);
    return file_arg(g, t->file, base);
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
        const bool always = gpu_pack_always_streams(pack);
        for (size_t it = 0; strmap_next(&pack->tensors, &it, &key, &val);) {
            if (skip_len > 0 && str_find(key, str_n(skip, skip_len), 0) >= 0) continue;
            // streamed experts are read from the files on demand, not touched here
            if (g->segs && streamed_tensor(key)) continue;
            // a pack that always streams (DeepSeek-V4-Flash) does not run its draft head yet:
            // touching it would read 10.9 GB at every start for nothing
            if (always && str_starts(key, S("mtp."))) continue;
            const TensorInfo *t = val;
            uint64_t first = t->offset / page * page;
            uint32_t pages = (uint32_t)((t->offset + t->nbytes - first + page - 1) / page);
            uint32_t p[2] = {pages, (uint32_t)(page / 4)};
            sx_arg args[3] = {file_arg(g, t->file, first), gpu_buf_arg(sink), gpu_bytes_arg(p, 8)};
            sx_cmd_dispatch(cmd, pl, args, 3, (pages + 255) / 256, 1, 256);
        }
    } else {
        for (size_t i = 0; i < g->nfile_buffers; i++) {
            uint32_t nb = i < g->nseg_files ? g->nseg[i] : 1;
            for (uint32_t j = 0; j < nb; j++) {
                sx_buffer *b = i < g->nseg_files ? g->segs[i][j].buf : g->file_buffers[i];
                uint32_t pages = (uint32_t)(sx_buffer_length(b) / page);
                uint32_t p[2] = {pages, (uint32_t)(page / 4)};
                sx_arg args[3] = {gpu_buf_arg(b), gpu_buf_arg(sink), gpu_bytes_arg(p, 8)};
                sx_cmd_dispatch(cmd, pl, args, 3, (pages + 255) / 256, 1, 256);
            }
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

Err gpu_touch_buffers(Gpu *g, sx_buffer *const *bufs, size_t n) {
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
    for (size_t i = 0; i < n; i++) {
        uint32_t pages = (uint32_t)(sx_buffer_length(bufs[i]) / page);
        uint32_t p[2] = {pages, (uint32_t)(page / 4)};
        sx_arg args[3] = {gpu_buf_arg(bufs[i]), gpu_buf_arg(sink), gpu_bytes_arg(p, 8)};
        sx_cmd_dispatch(cmd, pl, args, 3, (pages + 255) / 256, 1, 256);
    }
    char err[512];
    int rc = sx_cmd_finish(cmd, err, sizeof err);
    sx_buffer_release(sink);
    sx_pipeline_release(pl);
    if (rc != 0) {
        dbg("touch: %s\n", err);
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
    // SAETTA_NO_ACCEL=1: no tensor-op (neural accelerator) kernels, as when they do not compile;
    // every model then takes its simdgroup paths (checks the path older GPUs may take)
    if (strstr(source, "<metal_tensor>") != NULL && getenv("SAETTA_NO_ACCEL") != NULL &&
        strcmp(getenv("SAETTA_NO_ACCEL"), "0") != 0) {
        *out = NULL;
        return "AccelDisabled";
    }
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
