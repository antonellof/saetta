// Streamed-expert stops and their host service (see xstream.h).
#include "xstream.h"

// Host time in the service loop: waiting for the GPU to reach a stop, and mapping (with reads).
static double xs_wait_s, xs_map_s;
static uint64_t xs_stops;
// the same wait split by whether the previous stop read from the SSD (diagnostic)
static double xs_wait_after_miss_s, xs_wait_after_hit_s;
static uint64_t xs_after_miss, xs_after_hit, xs_prev_misses;
static bool xs_prev_had_miss;

Err xs_fits(Gpu *g, uint64_t xbytes, uint64_t keep) {
    uint64_t ws = sx_device_recommended_working_set(g->dev);
    if (xbytes + keep + 8000000000ull > ws) {
        dbg("model: expert cache of %.1f GB does not fit: %.1f GB stay resident, working set %.1f GB\n",
            (double)xbytes / 1e9, (double)keep / 1e9, (double)ws / 1e9);
        return "CacheTooLarge";
    }
    return OK;
}

uint64_t xs_fit_bytes(Gpu *g, uint64_t keep, uint64_t margin) {
    uint64_t ws = sx_device_recommended_working_set(g->dev), need = keep + 8000000000ull + margin;
    return ws > need ? ws - need : 0;
}

Err xs_open(Arena *a, const Pack *pack, Gpu *g, const XcLayer *layers, uint32_t nlayers, uint32_t nt,
            uint64_t bytes, XStream *xs) {
    TRY(xc_open(a, pack, g->dev, layers, nlayers, nt, bytes, 0, &xs->xc));
    {
        sx_buffer *rb[XC_MAX_TENSORS];
        for (uint32_t r = 0; r < nt; r++) rb[r] = xc_region(xs->xc, r);
        TRY(gpu_touch_buffers(g, rb, nt));
    }
    xs->ev = sx_event_new(g->dev);
    if (!xs->ev) return "GpuFailed";
    return OK;
}

bool xs_stop(XStream *xs, sx_cmd *cmd, XSvc *list, uint32_t *n, uint32_t cap, uint8_t kind, uint32_t layer,
             uint32_t nids) {
    if (*n >= cap || xs->fault) return false;
    uint64_t sig = xs->val + 1;
    xs->val += 2;
    sx_cmd_signal(cmd, xs->ev, sig);
    sx_cmd_wait_event(cmd, xs->ev, sig + 1);
    list[(*n)++] = (XSvc){.kind = kind, .layer = layer, .n = nids, .sig = sig};
    return true;
}

Err xs_service(XStream *xs, const sx_cmd *cmd, const XSvc *list, uint32_t *n, XsMapFn map, void *ctx) {
    Err first = OK;
    const uint32_t ns = *n;
    for (uint32_t i = 0; i < ns; i++) {
        const XSvc *s = &list[i];
        double t0 = now_s();
        int wrc = sx_event_wait_cmd(xs->ev, s->sig, cmd);
        double t1 = now_s();
        xs_wait_s += t1 - t0;
        xs_stops++;
        if (i > 0) {
            if (xs_prev_had_miss) { xs_wait_after_miss_s += t1 - t0; xs_after_miss++; }
            else { xs_wait_after_hit_s += t1 - t0; xs_after_hit++; }
        }
        if (wrc != 0) {
            // the GPU never reached this stop (fault or timeout): release every later one so
            // nothing waits forever, and refuse further streamed work, since the event's
            // values can no longer be trusted
            sx_event_set(xs->ev, list[ns - 1].sig + 1);
            *n = 0;
            xs->fault = true;
            return wrc == 2 ? "GpuFailed" : "GpuTimeout";
        }
        Err err = map(ctx, xs->xc, s);
        if (err && !first) first = err;
        xs_map_s += now_s() - t1;
        uint64_t mnow = xc_stats(xs->xc).misses;
        xs_prev_had_miss = mnow != xs_prev_misses;
        xs_prev_misses = mnow;
        sx_event_set(xs->ev, s->sig + 1);
    }
    *n = 0;
    return first;
}

void xs_report(const XCache *c, size_t tokens) {
    XcStats st = xc_stats(c);
    double hit = st.requests ? 100.0 * (double)st.hits / (double)st.requests : 0;
    dbg("experts: %llu requests, %.1f%% hits, %llu misses, %.2f GB read in %.1f ms (%.2f GB/s), %.2f MB/token\n",
        (unsigned long long)st.requests, hit, (unsigned long long)st.misses, (double)st.bytes_read / 1e9,
        st.read_seconds * 1e3, st.read_seconds > 0 ? (double)st.bytes_read / 1e9 / st.read_seconds : 0,
        tokens ? (double)st.bytes_read / 1e6 / (double)tokens : 0);
    dbg("experts: %llu stops, host waiting for the GPU %.1f ms, mapping %.1f ms (%.1f us per stop)\n",
        (unsigned long long)xs_stops, xs_wait_s * 1e3, xs_map_s * 1e3,
        xs_stops ? (xs_wait_s + xs_map_s) * 1e6 / (double)xs_stops : 0);
    dbg("experts: wait after a stop that read %.1f us (%llu), after an all-hit stop %.1f us (%llu)\n",
        xs_after_miss ? xs_wait_after_miss_s * 1e6 / (double)xs_after_miss : 0, (unsigned long long)xs_after_miss,
        xs_after_hit ? xs_wait_after_hit_s * 1e6 / (double)xs_after_hit : 0, (unsigned long long)xs_after_hit);
}
