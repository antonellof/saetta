// The only interface between the C engine and the platform (Metal, Foundation, libc mmap).
// Plain C, opaque handles. Implemented in shim.m (Objective-C with ARC).
#ifndef SAETTA_SHIM_H
#define SAETTA_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sx_device sx_device;      // MTLDevice + one command queue
typedef struct sx_buffer sx_buffer;      // MTLBuffer
typedef struct sx_pipeline sx_pipeline;  // MTLComputePipelineState

// --- files ---------------------------------------------------------------
// Maps a whole file read-only. Returns NULL on failure (errno set). *len gets
// the file size; the mapping is page aligned and spans round_up(len, page).
const void *sx_map_file(const char *path, size_t *len);
void sx_unmap_file(const void *ptr, size_t len);
size_t sx_page_size(void);
// Asks the kernel to start reading the pages of [addr, addr+len) (madvise MADV_WILLNEED),
// so n-gram rows of a token or chunk fault in parallel instead of one after another.
void sx_prefetch(const void *addr, size_t len);

// Unicode NFC of UTF-8 text (Foundation). Writes up to cap bytes to out and returns the
// full NFC length; returns (size_t)-1 when the input is not valid UTF-8.
size_t sx_nfc(const char *in, size_t len, char *out, size_t cap);

// Thermal state of the machine (NSProcessInfo.thermalState): 0 nominal, 1 fair, 2 serious,
// 3 critical.
int sx_thermal_state(void);

// --- device --------------------------------------------------------------
sx_device *sx_device_create(void);
void sx_device_destroy(sx_device *d);
// Writes the device name into buf (NUL terminated, truncated to cap).
void sx_device_name(const sx_device *d, char *buf, size_t cap);
uint64_t sx_device_recommended_working_set(const sx_device *d);
// Bytes the device has allocated now (MTLDevice.currentAllocatedSize); safe from any thread.
uint64_t sx_device_allocated_size(const sx_device *d);
int sx_device_supports_apple10(const sx_device *d);

// --- buffers -------------------------------------------------------------
// Wraps existing memory without copying. ptr must be page aligned; len is
// rounded up to the page size by the caller. The memory must outlive the buffer.
sx_buffer *sx_buffer_wrap(sx_device *d, const void *ptr, size_t len);
// Allocates a new shared buffer, zero filled.
sx_buffer *sx_buffer_new(sx_device *d, size_t len);
void sx_buffer_release(sx_buffer *b);
void *sx_buffer_contents(sx_buffer *b);
size_t sx_buffer_length(const sx_buffer *b);

// --- kernels -------------------------------------------------------------
// Compiles Metal source and returns a pipeline for function `name`. On failure
// returns NULL and writes the compiler message into err (NUL terminated).
sx_pipeline *sx_pipeline_from_source(sx_device *d, const char *source, const char *name, char *err, size_t err_cap);
void sx_pipeline_release(sx_pipeline *p);
uint32_t sx_pipeline_thread_width(const sx_pipeline *p);

typedef struct {
    sx_buffer *buffer;  // NULL for an inline-bytes argument
    size_t offset;
    const void *bytes;  // used when buffer is NULL
    size_t bytes_len;
} sx_arg;

// Encodes one dispatch of `groups` threadgroups of `threads` threads (1-D),
// commits and waits. Returns 0 on success, nonzero with err set on GPU error.
int sx_dispatch_sync(sx_device *d, sx_pipeline *p, const sx_arg *args, size_t nargs,
                     size_t groups, size_t threads, char *err, size_t err_cap);

// --- command buffers -------------------------------------------------------
// One command buffer with one serial compute encoder: dispatches run in the order
// encoded, each seeing the writes of the ones before (Metal hazard tracking).
typedef struct sx_cmd sx_cmd;
sx_cmd *sx_cmd_begin(sx_device *d);
// Encodes groups_x * groups_y threadgroups of `threads` threads.
void sx_cmd_dispatch(sx_cmd *c, sx_pipeline *p, const sx_arg *args, size_t nargs,
                     size_t groups_x, size_t groups_y, size_t threads);
void sx_cmd_dispatch3(sx_cmd *c, sx_pipeline *p, const sx_arg *args, size_t nargs,
                      size_t groups_x, size_t groups_y, size_t groups_z, size_t threads);
// Ends encoding, commits, waits and frees c. Returns 0, or nonzero with err set.
int sx_cmd_finish(sx_cmd *c, char *err, size_t err_cap);
// Same, and writes the GPU execution time of the command buffer in seconds.
int sx_cmd_finish_timed(sx_cmd *c, double *gpu_seconds, char *err, size_t err_cap);
// Total GPU execution time of the command buffers finished with sx_cmd_finish, in seconds.
double sx_gpu_seconds(void);
// Split finish: commit ends encoding and commits without waiting (c stays valid until wait,
// which blocks or spins as sx_cmd_finish, then frees c); discard frees a command buffer
// that was encoded but never committed.
void sx_cmd_commit(sx_cmd *c);
int sx_cmd_wait(sx_cmd *c, char *err, size_t err_cap);
void sx_cmd_discard(sx_cmd *c);

// --- HTTP client -------------------------------------------------------------
// One HTTP(S) request through NSURLSession (redirects followed; no caching, no cookies,
// no compression: Accept-Encoding identity is sent unless a header overrides it).
// Headers are extra request headers (name, value), e.g. user-agent, authorization, range.
// on_data(ctx, status, p, n) receives the response body in order as it arrives (status is
// the final HTTP status, after redirects); it returns 0 to go on, nonzero to cancel.
// Blocks until the response is complete. Returns 0 with *status set (any HTTP status), or
// nonzero on a transport error, timeout or cancel, with err set.
// timeout_s: idle timeout between bytes (NSURLSessionConfiguration.timeoutIntervalForRequest).
typedef struct {
    const char *name;
    const char *value;
} sx_header;
typedef int (*sx_http_sink)(void *ctx, int status, const void *p, size_t n);
int sx_http(const char *method, const char *url, const sx_header *headers, size_t nheaders,
            const void *body, size_t body_len, double timeout_s, sx_http_sink on_data, void *ctx,
            int *status, char *err, size_t err_cap);

#ifdef __cplusplus
}
#endif
#endif
