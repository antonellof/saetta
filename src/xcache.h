// Expert cache for streamed packs: routed experts live in a fixed pool of GPU-visible slots and
// are read from the pack files on demand when the router picks one that is not cached.
//
// A layer's experts are stored by the pack as nt stacked tensors (for Flash-Next 9: gate, up
// and down, each weight, scales and biases), expert e being one contiguous range of
// nbytes / ne bytes in each. The cache keeps one region per tensor index, every region holding
// n_slots slots at a fixed stride (the largest per-expert size of that tensor over all layers,
// rounded up to 16 KB). A slot holds one (layer, expert) in all nt regions at once. Kernels
// bind region r at offset 0 with stride xc_stride(r) and read slot ids where they read expert
// ids, so the arithmetic is the same as on the pack.
//
// Eviction is least recently used over all layers. The caller guarantees that the GPU does not
// read any slot while xc_map runs (the command buffer is stopped at an event, and every
// earlier layer has finished), so a slot can be overwritten as soon as it is not one of the
// slots the current call returns.
//
// Reads are pread with F_NOCACHE on a pool of loader threads, started at open. Nothing is
// allocated after xc_open.
#ifndef SAETTA_XCACHE_H
#define SAETTA_XCACHE_H

#include "base.h"
#include "metal/shim.h"
#include "weights.h"

#define XC_MAX_TENSORS 9

typedef struct {
    const TensorInfo *t[XC_MAX_TENSORS]; // stacked [ne, ...] tensors of this layer's experts
    uint32_t ne;                         // experts in this layer
} XcLayer;

typedef struct {
    uint64_t requests, hits, misses, bytes_read;
    double read_seconds; // wall time spent waiting for reads inside xc_map
} XcStats;

typedef struct XCache XCache;

// Opens the cache for nlayers layers of nt tensors each (the same nt for every layer), with as
// many slots as fit in cache_bytes, at least the largest ne (so one layer's whole expert set
// fits, which a prompt chunk can need). threads: loader threads (0 picks 4).
// Errors: "BadExpertLayout" (an nbytes not divisible by ne, or nt mismatch), "CacheTooSmall",
// "OpenFailed", "OutOfMemory".
Err xc_open(Arena *a, const Pack *p, sx_device *d, const XcLayer *layers, uint32_t nlayers, uint32_t nt,
            uint64_t cache_bytes, uint32_t threads, XCache **out);
void xc_close(XCache *c);

uint32_t xc_slots(const XCache *c);
sx_buffer *xc_region(const XCache *c, uint32_t r);
uint32_t xc_stride(const XCache *c, uint32_t r); // bytes per slot in region r
uint64_t xc_bytes(const XCache *c);              // total slab bytes

// Maps n expert ids of layer to slots (ids may repeat; equal ids get the same slot), reading
// the missing experts first. slots_out[i] is the slot of ids[i]. Errors: "BadExpertId",
// "ReadFailed", "CacheTooSmall" (more distinct ids than slots).
Err xc_map(XCache *c, uint32_t layer, const uint32_t *ids, uint32_t n, uint32_t *slots_out);

// Cache size asked for with SAETTA_EXPERT_CACHE_GB=<GB> (decimal GB, fractions allowed), or 0
// when unset: every expert resident.
uint64_t xc_requested_bytes(void);

XcStats xc_stats(const XCache *c);
void xc_reset_stats(XCache *c);

#endif
