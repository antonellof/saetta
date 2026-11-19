// One model server per machine, shared by every agent session.
//   ~/.saetta/run/serve.lock   held (exclusive advisory lock) by the running `saetta serve` for
//                              its whole life; it records "pid port mode". The lock dies with
//                              the process, so a crashed server never blocks the next one.
//   ~/.saetta/run/clients/<pid> one file per running `saetta start`. A server that `start`
//                              launched (mode "auto") is stopped by the last session to exit;
//                              a server started by hand (mode "manual") is never stopped.
// Paths come from $HOME (getenv); errors are "NoHome" or the errno names of base.h.
#ifndef SAETTA_INSTANCE_H
#define SAETTA_INSTANCE_H

#include "base.h"

// autostop: mode "auto" (started by `saetta start`, stopped by its last session).
typedef struct {
    int32_t pid;
    uint16_t port;
    bool autostop;
} Server;

// Takes the machine-wide server lock for this process, or reports the server holding it
// (message on stderr, error "AlreadyRunning"). *fd is the lock file: keep it open for the
// life of the server (serve() never closes it).
Err instance_acquire(uint16_t port, int *fd);

// The running server, if one holds the lock (false otherwise or on any error).
bool instance_running(Server *out);

// Registers this `saetta start` process as a user of the server; the file holds the agent's
// name (for `saetta monitor`).
Err instance_register(const char *agent);

// `saetta start --keep` leaves this marker so the server's watchdog keeps it running with no
// session; the next session without --keep removes it.
Err instance_keep_path(Arena *a, char **out);
bool instance_kept(void);
void instance_set_keep(bool on);

typedef struct {
    int32_t pid;
    const char *agent; // NUL terminated, in the arena ("agent" when the file is empty)
} Client;

// Live registered sessions into *out (arena owned), their count returned. Stale entries of
// dead processes are removed.
size_t instance_clients(Arena *a, Client **out);
size_t instance_count_clients(void);

// Unregisters this process; returns how many other live sessions still use the server
// (stale entries of dead processes are removed on the way).
size_t instance_unregister(void);

#endif
