// Streamed experts in a pass: the stops of a command buffer and their host service, shared by
// the models that stream (src/model.c, src/dsv4.c).
//
// A stop is a point where the command buffer waits until the host has turned one layer's
// expert ids (written by the GPU before the stop) into cache slots: the GPU signals sig once
// every dispatch before the stop has completed, the host maps the ids (xc_map, reading the
// missing experts) and sets sig + 1, which releases the GPU. The model encodes its stops with
// xs_stop while it encodes the pass, commits, then calls xs_service, which serves them in order
// through the model's map function (where the ids are and how they become slots is the
// model's business). Nothing here allocates.
#ifndef SAETTA_XSTREAM_H
#define SAETTA_XSTREAM_H

#include "base.h"
#include "gpu.h"
#include "metal/shim.h"
#include "xcache.h"

// One stop: the model's kind, its cache layer, the number of ids, the event value it signals.
typedef struct {
    uint8_t kind;
    uint32_t layer, n;
    uint64_t sig;
} XSvc;

// The cache, the event the command buffers stop on and its last value handed out.
typedef struct {
    XCache *xc;
    sx_event *ev;
    uint64_t val;
    bool fault; // a stop failed (GPU fault or timeout): streamed work is refused from then on
} XStream;

// Whether a cache of xbytes fits next to keep resident bytes: their sum plus 8 GB within the
// device's recommended working set. Errors: "CacheTooLarge" (with a dbg line).
Err xs_fits(Gpu *g, uint64_t xbytes, uint64_t keep);
// The largest cache size xs_fits accepts, less margin bytes (0 when none).
uint64_t xs_fit_bytes(Gpu *g, uint64_t keep, uint64_t margin);

// Opens the cache (xc_open with the default loader threads), touches its regions from the GPU
// and creates the event. xs must be zeroed. Errors: xc_open's, gpu_touch_buffers', "GpuFailed".
Err xs_open(Arena *a, const Pack *pack, Gpu *g, const XcLayer *layers, uint32_t nlayers, uint32_t nt,
            uint64_t bytes, XStream *xs);

// Encodes a stop in cmd and appends it to list[0..*n) (room for cap). Returns false, encoding
// nothing, when the list is full or a stop failed before: the caller must not commit cmd.
bool xs_stop(XStream *xs, sx_cmd *cmd, XSvc *list, uint32_t *n, uint32_t cap, uint8_t kind, uint32_t layer,
             uint32_t nids);

// Maps the ids of stop s to slots in place; on an error it must leave valid slots (0) behind.
typedef Err (*XsMapFn)(void *ctx, XCache *xc, const XSvc *s);

// After cmd is committed, before waiting for it: serves list[0..*n) in order, then sets *n to 0.
// Each stop is released even when its mapping fails, so the GPU never hangs; the first error
// is returned. If the GPU never reaches a stop (fault or timeout), every later stop is released,
// xs->fault is set and "GpuFailed" or "GpuTimeout" returned.
Err xs_service(XStream *xs, const sx_cmd *cmd, const XSvc *list, uint32_t *n, XsMapFn map, void *ctx);

// The expert lines of model_prof_report for the cache c (hits, reads, host time at the stops).
void xs_report(const XCache *c, size_t tokens);

#endif
