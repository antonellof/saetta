// `saetta monitor [--once]`: a live view of the saetta server on this machine and the Mac
// under it. Read-only: it reads the server's snapshot (~/.saetta/run/metrics.json, written
// twice a second by src/stats.c) and samples the system itself: GPU load from the
// IOAccelerator statistics (`ioreg`), memory from `vm_stat` and sysctl, the thermal state from
// Foundation. q quits and leaves the server running. --once prints one frame and exits.
//
// The frame is drawn by monitor_frame from a snapshot and a system sample, for a given size:
// `saetta monitor` and Saetta Code's /monitor (src/code_monitor.c) draw the same view.
#ifndef SAETTA_MONITOR_H
#define SAETTA_MONITOR_H

#include "base.h"
#include "instance.h"
#include "json.h"

// What the panel under the cards shows: r requests, c conversations, l server log.
typedef enum { MONITOR_REQUESTS, MONITOR_CONVERSATIONS, MONITOR_LOG } MonitorView;

// The Mac under the server.
typedef struct {
    bool has_gpu;
    double gpu_util;
    uint64_t ram_total, ram_used, swap_used;
    int thermal;
} MonitorSys;

// Recent values for the sparklines, oldest first.
typedef struct {
    double v[64];
    size_t n;
} MonitorHist;

typedef struct {
    MonitorHist tok, gpu;
} MonitorHists;

// What one frame shows.
typedef struct {
    const JValue *m;  // the server's snapshot (stats.h stats_json), NULL: no server
    MonitorSys sys;
    const MonitorHists *hist;
    size_t w, h;      // the terminal's size
    MonitorView view;
    const Client *sessions; // the agent sessions using the server
    size_t n_sessions;
    bool inproc;      // the model runs in the viewer's process: no address, no server log
    // with no server: the header's text and hint, and the line under the panel (NULL: none)
    const char *offline, *offline_hint, *waiting;
    const char *keys; // the key hints of the last line
    // set: the view is not available, and the frame is this message and the keys only
    const char *unavailable, *unavailable_hint;
} MonitorFrame;

// Samples the GPU load, memory and thermal state (runs ioreg and vm_stat).
MonitorSys monitor_sample(void);
// The snapshot at path (metrics.json) when the server that wrote it still runs, else NULL.
JValue *monitor_read_snapshot(Arena *a, const char *path);
// One sample more in the sparklines: the server's speed (0 with no server) and the GPU load.
void monitor_push(MonitorHists *h, const JValue *m, const MonitorSys *sys);
// The frame's text: lines ending in '\n', with ANSI colors, about f->w columns wide.
void monitor_frame(Arena *a, Buf *o, const MonitorFrame *f);

// args: the arguments after "monitor". Sets *code (0, or 64 on a usage error). Errors:
// "NoHome", "WriteFailed".
Err monitor_run(const char *const *args, size_t n, int *code);

#endif
