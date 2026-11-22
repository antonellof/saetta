// Saetta Code's /monitor: `saetta monitor`'s view (src/monitor.h, the same frame) over the whole
// screen, refreshed twice a second, until Esc closes it. pi has no such command.
//
// Its numbers come from where the model runs: the in-process engine's statistics (read in this
// process, server_engine_stats and stats_json), the snapshot of the saetta server on this Mac
// when the model's base URL is that server (~/.saetta/run/metrics.json, what `saetta monitor`
// reads), and nothing for any other provider (a short "not available" message; the "saetta"
// provider with no saetta server running gets the view's no-data state).
#ifndef SAETTA_CODE_MONITOR_H
#define SAETTA_CODE_MONITOR_H

#include "code_tui.h"

struct Engine;

// The current model (code_main.c, at start and on each switch): its in-process engine (NULL
// over HTTP), provider key and base URL. Strings are copied; any thread.
void code_monitor_set_model(struct Engine *engine, const char *provider, const char *base_url);

// /monitor: opens the view (UI thread, as the calls below).
void code_monitor_open(void);
bool code_monitor_is_open(void);
// A key while the view is open: Esc, Ctrl+C or q close it, r / c / l switch the panel, space
// freezes it. Returns false (the key not taken) when the view is closed.
bool code_monitor_key(const Key *k);
// Milliseconds until the next sample, -1 when closed (the UI's poll timeout).
int code_monitor_wait_ms(void);
// Samples when one is due; true when the view changed (redraw).
bool code_monitor_tick(void);
// The whole screen while the view is open: height rows of width columns at most.
void code_monitor_render(int width, int height, Lines *out);

#endif
