// `saetta monitor [--once]`: a live view of the saetta server on this machine and the Mac
// under it. Read-only: it reads the server's snapshot (~/.saetta/run/metrics.json, written
// twice a second by src/stats.c) and samples the system itself: GPU load from the
// IOAccelerator statistics (`ioreg`), memory from `vm_stat` and sysctl, the thermal state from
// Foundation. q quits and leaves the server running. --once prints one frame and exits.
#ifndef SAETTA_MONITOR_H
#define SAETTA_MONITOR_H

#include "base.h"

// What the panel under the cards shows: r requests, c conversations, l server log.
typedef enum { VIEW_REQUESTS, VIEW_CONVERSATIONS, VIEW_LOG } View;

// args: the arguments after "monitor". Sets *code (0, or 64 on a usage error). Errors:
// "NoHome", "WriteFailed".
Err monitor_run(const char *const *args, size_t n, int *code);

#endif
