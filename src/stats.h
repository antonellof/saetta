// Live server statistics for `saetta monitor`. The request path only stores into atomics
// (no lock, no syscall per token); finished requests go into a small ring under a spin
// lock, once per request. A background thread writes a JSON snapshot to
// ~/.saetta/run/metrics.json every 500 ms (write to a temporary file, then rename), so the
// monitor never waits behind a request on the single-threaded HTTP loop. The same thread is
// the watchdog of a server that `saetta start` launched: it exits once no registered session
// has been alive for 20 s (the launcher may have been killed without cleaning up, e.g. by
// closing its terminal).
#ifndef SAETTA_STATS_H
#define SAETTA_STATS_H

#include <stdatomic.h>

#include "base.h"
#include "metal/shim.h"

// Values of Stats.state.
typedef enum { STATE_IDLE = 0, STATE_READING = 1, STATE_GENERATING = 2 } State;

// One conversation the server keeps (for the monitor's table).
typedef struct {
    uint32_t tokens, cap;
    int64_t last_s;
    bool active;
} ConvInfo;

// A finished request. Zero initialize, then set the fields; finish is NUL padded.
typedef struct {
    int64_t end_s; // unix seconds (set by stats_finish)
    bool anthropic;
    uint32_t prompt, reused, output, passes, prefill_ms, decode_ms;
    char finish[12];
} Done;

typedef struct Stats {
    // set once (before stats_spawn)
    const char *model;
    const char *pack;
    uint16_t port;
    size_t ctx;
    int64_t started_s;
    sx_device *dev; // may be NULL
    // live, the current request: the server stores into these with memory_order_release
    _Atomic uint8_t state; // State
    _Atomic bool anthropic;
    _Atomic uint32_t prompt_total, prompt_done, reused, gen, passes, max_tokens;
    _Atomic int64_t req_start_ns, decode_start_ns; // stats_now_ns()
    // totals since start
    _Atomic uint64_t requests, prompt_tokens, reused_tokens, output_tokens, output_passes;
    // restore points of the session (token counts), updated at request boundaries
    _Atomic uint32_t ck0, ck1;
    _Atomic uint32_t convs; // 1 after stats_init
    // A request finished: the snapshot is written at the next 100 ms tick, not the 500 ms one.
    _Atomic bool dirty;
    // finished requests, newest last, and conversations, both under mu (a spin lock)
    _Atomic bool mu;
    Done ring[16];
    size_t ring_n;
    ConvInfo conv[8];
    size_t conv_n;
} Stats;

// Zeroes st and sets the defaults (convs = 1); then set model, pack, port, ctx, started_s, dev.
void stats_init(Stats *st);
void stats_set_convs(Stats *st, const ConvInfo *cv, size_t n);
// The awake (monotonic) clock in ns, as stored in req_start_ns and decode_start_ns.
int64_t stats_now_ns(void);
void stats_begin(Stats *st, bool anthropic, size_t prompt, size_t reused, size_t max_tokens);
void stats_finish(Stats *st, Done d);
// The snapshot as JSON (what the thread writes to metrics.json), appended to o. Reads the
// atomics and takes the spin lock as the thread does; any thread may call it.
void stats_json(Stats *st, Buf *o);
// Starts the snapshot (and, for an auto-started server, watchdog) thread; detached, it runs
// for the life of the process. Does nothing without $HOME.
Err stats_spawn(Stats *st);

#endif
