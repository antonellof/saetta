// Turns (see turns.h).
#include "turns.h"

#include <time.h>

void turns_init(Turns *t) {
    *t = (Turns){0};
    pthread_mutex_init(&t->mu, NULL);
    pthread_cond_init(&t->cv, NULL);
}

uint64_t turns_take(Turns *t) {
    pthread_mutex_lock(&t->mu);
    uint64_t n = t->next++;
    pthread_mutex_unlock(&t->mu);
    return n;
}

bool turns_wait(Turns *t, uint64_t n, int ms) {
    pthread_mutex_lock(&t->mu);
    struct timespec dl;
    clock_gettime(CLOCK_REALTIME, &dl);
    if (ms > 0) {
        dl.tv_sec += ms / 1000;
        dl.tv_nsec += (long)(ms % 1000) * 1000000L;
        if (dl.tv_nsec >= 1000000000L) {
            dl.tv_sec += 1;
            dl.tv_nsec -= 1000000000L;
        }
    }
    while (t->serving != n) {
        if (ms < 0) pthread_cond_wait(&t->cv, &t->mu);
        else if (pthread_cond_timedwait(&t->cv, &t->mu, &dl) != 0) break;
    }
    bool mine = t->serving == n;
    pthread_mutex_unlock(&t->mu);
    return mine;
}

// The next turn: past the ones given up.
static void advance_l(Turns *t) {
    t->serving++;
    while (t->serving < t->next && t->gone[t->serving % TURNS_MAX]) {
        t->gone[t->serving % TURNS_MAX] = false;
        t->serving++;
    }
    pthread_cond_broadcast(&t->cv);
}

void turns_done(Turns *t) {
    pthread_mutex_lock(&t->mu);
    advance_l(t);
    pthread_mutex_unlock(&t->mu);
}

void turns_leave(Turns *t, uint64_t n) {
    pthread_mutex_lock(&t->mu);
    if (t->serving == n) advance_l(t);
    else t->gone[n % TURNS_MAX] = true;
    pthread_mutex_unlock(&t->mu);
}

bool turns_waiting(Turns *t) {
    pthread_mutex_lock(&t->mu);
    bool w = t->next > t->serving + 1;
    pthread_mutex_unlock(&t->mu);
    return w;
}
