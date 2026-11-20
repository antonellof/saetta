// GPU context: the Metal device, the pack's files wrapped once as no-copy
// buffers, and compiled pipelines. All Metal access goes through shim.h.
#ifndef SAETTA_GPU_H
#define SAETTA_GPU_H

#include "base.h"
#include "metal/shim.h"
#include "safetensors.h"
#include "weights.h"

// A wrapped piece [start, end) of a pack file (streamed experts: the file minus its routed
// expert tensors, so binding a dense tensor does not make the experts resident).
typedef struct {
    uint64_t start, end;
    sx_buffer *buf;
} GpuSeg;

typedef struct {
    sx_device *dev;
    sx_buffer **file_buffers; // one per pack file, then the files added with gpu_add_file
    size_t nfile_buffers;
    // Streamed packs only: per pack file, its segments in file order (NULL otherwise, and
    // then file_buffers[i] wraps the whole file).
    GpuSeg **segs;
    uint32_t *nseg;
    size_t nseg_files;
} Gpu;

// Whether a tensor is a trunk routed expert, which a streamed pack reads through the expert
// cache (src/xcache.h) instead of binding it.
bool gpu_streamed_tensor(Str name);
// Packs whose model always streams its routed experts (DeepSeek-V4-Flash: they cannot fit).
bool gpu_pack_always_streams(const Pack *pack);

// Creates the device and wraps every pack file; with streaming (SAETTA_EXPERT_CACHE_GB set, or
// a pack that always streams) as segments that leave out the routed experts.
// Errors: "NoMetalDevice", "WrapFailed".
Err gpu_init(Arena *a, const Pack *pack, Gpu *out);
void gpu_deinit(Gpu *g);

// A kernel argument pointing at a tensor inside the mapped pack. The MLX pack
// stores tensors without alignment padding, so the binding is rounded down to
// 16 bytes and the remainder is written to *shift; kernels read such
// tensors through byte pointers. Cannot fail.
sx_arg gpu_tensor_arg(const Gpu *g, const TensorInfo *t, uint32_t *shift);

// Touches every page of every mapped pack file from the GPU once, so the first-touch
// cost is paid at load instead of on the first tokens. Errors: "CompileFailed",
// "AllocFailed", "GpuFailed".
Err gpu_warm(Gpu *g);
// The same per tensor of pack (page aligned); tensors whose name contains skip are not
// touched (the n-gram table is read 16 rows per token on the host; warming its 32 GB only
// costs RAM). pack NULL warms every file buffer, as gpu_warm.
Err gpu_warm_except(Gpu *g, const Pack *pack, const char *skip);
// Reads one word per page of each buffer from the GPU, so its pages are mapped for the GPU
// before the first real use (the streamed-expert slabs: their first GPU touch is slow).
Err gpu_touch_buffers(Gpu *g, sx_buffer *const *bufs, size_t n);

// Seconds of GPU time for n dispatches of a one-thread kernel in one serial
// command buffer: the fixed cost each dispatch adds to a token. Errors: "GpuFailed".
Err gpu_dispatch_cost(Gpu *g, size_t n, size_t serial_threads, double *secs);

// Compiles source (NUL terminated) and returns the pipeline for function name; on failure
// prints "kernel <name>: <message>" to stderr and returns "CompileFailed".
Err gpu_pipeline(Gpu *g, const char *source, const char *name, sx_pipeline **out);

// Registers a GPU buffer as one more "file", so tensors built in it at load (the MTP
// head's weights requantized from BF16) bind through gpu_tensor_arg like pack tensors.
// *index gets its file index. The old array is left in the arena.
Err gpu_add_file(Gpu *g, Arena *a, sx_buffer *b, uint16_t *index);

// A new zero-filled shared buffer. Errors: "AllocFailed".
Err gpu_new_buffer(Gpu *g, size_t len, sx_buffer **out);

static inline sx_arg gpu_buf_arg(sx_buffer *b) { return (sx_arg){b, 0, NULL, 0}; }
static inline sx_arg gpu_u32_arg(const uint32_t *v) { return (sx_arg){NULL, 0, v, 4}; }
// An inline-bytes argument of n bytes at p.
static inline sx_arg gpu_bytes_arg(const void *p, size_t n) { return (sx_arg){NULL, 0, p, n}; }
// The buffer contents as floats; gpu_nfloats is its length in floats (bytes / 4).
static inline float *gpu_floats(sx_buffer *b) { return (float *)sx_buffer_contents(b); }
static inline size_t gpu_nfloats(const sx_buffer *b) { return sx_buffer_length(b) / 4; }

#endif
