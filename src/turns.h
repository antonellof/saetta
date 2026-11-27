// Turns: one holder at a time, in arrival order (a ticket lock whose waiters can give their turn
// up). The engine's requests take turns with it (src/server.c), as do the relayed requests of
// Saetta Code's test stand-in for a shared engine (src/code_share.c).
#ifndef SAETTA_TURNS_H
#define SAETTA_TURNS_H

#include "base.h"

#include <pthread.h>

#define TURNS_MAX 256 // waiters at most

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    uint64_t next, serving;
    bool gone[TURNS_MAX];
} Turns;

void turns_init(Turns *t);
uint64_t turns_take(Turns *t);
// True when it is turn n's (within ms; ms < 0 waits for it).
bool turns_wait(Turns *t, uint64_t n, int ms);
void turns_done(Turns *t);              // the turn being served ends
void turns_leave(Turns *t, uint64_t n); // turn n is given up (served now: ended)
// True when a ticket after the one being served was taken (someone waits for a turn, or took one
// and gave it up since): what the holder of a turn that yields checks (server.c prefill ahead).
bool turns_waiting(Turns *t);

#endif
