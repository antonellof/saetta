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
// The in-process engine's lock (~/.saetta/run/inproc.lock), held for the life of a Saetta Code
// that runs the model in its own process; taking it also makes this process the holder of a new
// identity and token and writes the record (loading, the pack folder dir). A /model restart hands
// the lock to the next image (instance_inproc_restarting). Errors: "EngineBusy" (another holds it).
Err instance_inproc_acquire(const char *dir, int *fd);
// True when a Saetta Code holds the model in its process (*pid its pid, may be NULL): its record
// names a live process with the start time it had (no lock is probed).
bool instance_inproc_running(int *pid);

// Sharing that engine with other Saetta Codes (code_share.c). The holder's record
// (~/.saetta/run/inproc.rec, 0600, replaced atomically) gives its state, the context, its identity
// and start time, the token every request must carry, the model id, the pack folder and the Unix
// socket it serves the server's HTTP API on. Each session that uses it registers in
// ~/.saetta/run/inproc-clients/<pid> (0700 folder) with the holder's identity.
typedef enum { INPROC_LOADING, INPROC_SERVING, INPROC_PRIVATE, INPROC_CLOSING, INPROC_RESTARTING } InprocState;
typedef struct {
    int32_t pid;
    InprocState state; // CLOSING: no record (the holder is exiting); RESTARTING: a /model restart
    uint32_t ctx;
    char holder[40], token[65], model[256], dir[1024], sock[1024];
} Inproc;
Err instance_inproc_publish(InprocState st, uint32_t ctx, const char *model, const char *sock);
const char *instance_inproc_token(void); // this holder's token, NULL when not the holder
// The holder's record if its holder is alive (false otherwise).
bool instance_inproc_info(Inproc *out);
// A /model restart of the holder: the record says restarting (dir: the pack it restarts on, may
// be empty), and the lock stays held across the exec (SAETTA_INPROC_LOCK_FD, which the next
// image's instance_inproc_acquire takes). instance_inproc_release_inherited: an image that does
// not load the model gives the inherited lock (and the record) up.
Err instance_inproc_restarting(const char *dir);
void instance_inproc_release_inherited(void);
// Takes SAETTA_INPROC_LOCK_FD out of the environment (always) and keeps the lock it names when it
// is the real one (code_run calls it first, so it never reaches a child).
void instance_inproc_take_inherited(void);
// Registers this process with holder h, under the registration lock, if h still serves: false
// when it does not (it is closing, or another holder took its place).
bool instance_inproc_join(const Inproc *h);
void instance_inproc_leave(void);
// The holder: the live sessions registered with it (stale entries are removed).
size_t instance_inproc_clients(void);
// The holder: under the registration lock, the count of its sessions; when none (or force), the
// record is removed in the same step, so no session can join after it. Returns the count.
size_t instance_inproc_withdraw(bool force);

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
