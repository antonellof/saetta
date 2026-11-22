// Live server statistics and the snapshot / watchdog thread (see stats.h).
#include "stats.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "instance.h"

static void lock(Stats *st) {
    for (;;) {
        bool want = false;
        if (atomic_compare_exchange_strong_explicit(&st->mu, &want, true, memory_order_acquire, memory_order_relaxed))
            return;
        __builtin_arm_yield();
    }
}

static void unlock(Stats *st) { atomic_store_explicit(&st->mu, false, memory_order_release); }

void stats_init(Stats *st) {
    memset(st, 0, sizeof *st);
    st->model = "";
    st->pack = "";
    atomic_store(&st->convs, 1);
}

void stats_set_convs(Stats *st, const ConvInfo *cv, size_t n) {
    lock(st);
    n = MIN(n, COUNT(st->conv));
    memcpy(st->conv, cv, n * sizeof *cv);
    st->conv_n = n;
    unlock(st);
}

int64_t stats_now_ns(void) { return (int64_t)now_ns(); }

static int64_t unix_s(void) { return (int64_t)wall_ns() / 1000000000; }

#define REL memory_order_release
#define ACQ memory_order_acquire
#define AR memory_order_acq_rel

void stats_begin(Stats *st, bool anthropic, size_t prompt, size_t reused, size_t max_tokens) {
    atomic_store_explicit(&st->anthropic, anthropic, REL);
    atomic_store_explicit(&st->prompt_total, (uint32_t)prompt, REL);
    atomic_store_explicit(&st->prompt_done, (uint32_t)reused, REL);
    atomic_store_explicit(&st->reused, (uint32_t)reused, REL);
    atomic_store_explicit(&st->gen, 0, REL);
    atomic_store_explicit(&st->passes, 0, REL);
    atomic_store_explicit(&st->max_tokens, (uint32_t)MIN(max_tokens, (size_t)UINT32_MAX), REL);
    atomic_store_explicit(&st->req_start_ns, stats_now_ns(), REL);
    atomic_store_explicit(&st->state, STATE_READING, REL);
}

void stats_finish(Stats *st, Done d) {
    atomic_fetch_add_explicit(&st->requests, 1, AR);
    atomic_fetch_add_explicit(&st->prompt_tokens, d.prompt, AR);
    atomic_fetch_add_explicit(&st->reused_tokens, d.reused, AR);
    atomic_fetch_add_explicit(&st->output_tokens, d.output, AR);
    atomic_fetch_add_explicit(&st->output_passes, d.passes, AR);
    lock(st);
    d.end_s = unix_s();
    if (st->ring_n == COUNT(st->ring)) {
        memmove(st->ring, st->ring + 1, (COUNT(st->ring) - 1) * sizeof(Done));
        st->ring_n -= 1;
    }
    st->ring[st->ring_n++] = d;
    atomic_store_explicit(&st->state, STATE_IDLE, REL);
    atomic_store_explicit(&st->dirty, true, REL);
    unlock(st);
}

static const char *state_name(uint8_t s) {
    switch (s) {
    case STATE_IDLE: return "idle";
    case STATE_READING: return "reading";
    default: return "generating";
    }
}

void stats_json(Stats *st, Buf *o) {
    int64_t now = stats_now_ns();
    uint8_t state = atomic_load_explicit(&st->state, ACQ);
    uint32_t gen = atomic_load_explicit(&st->gen, ACQ);
    int64_t ds = atomic_load_explicit(&st->decode_start_ns, ACQ);
    double dec_s = state == STATE_GENERATING && ds > 0 ? (double)(now - ds) / 1e9 : 0;
    double rate = dec_s > 0.05 ? (double)gen / dec_s : 0;
    buf_printf(o, "{\"pid\":%d,\"port\":%u,\"model\":\"%s\",\"pack\":\"%s\",\"ctx\":%zu,\"started\":%lld,\"now\":%lld",
               (int)getpid(), st->port, st->model, st->pack, st->ctx, (long long)st->started_s, (long long)unix_s());
    if (st->dev)
        buf_printf(o, ",\"gpu_allocated\":%llu,\"gpu_working_set\":%llu",
                   (unsigned long long)sx_device_allocated_size(st->dev),
                   (unsigned long long)sx_device_recommended_working_set(st->dev));
    char el[64], ts[64];
    fmt_fixed(el, sizeof el, (double)(now - atomic_load_explicit(&st->req_start_ns, ACQ)) / 1e9, 2);
    fmt_fixed(ts, sizeof ts, rate, 1);
    buf_printf(o,
               ",\"state\":\"%s\",\"api\":\"%s\",\"prompt_total\":%u,\"prompt_done\":%u,\"reused\":%u,\"gen\":%u,"
               "\"passes\":%u,\"max_tokens\":%u,\"elapsed_s\":%s,\"tok_s\":%s",
               state_name(state), atomic_load_explicit(&st->anthropic, ACQ) ? "anthropic" : "openai",
               atomic_load_explicit(&st->prompt_total, ACQ), atomic_load_explicit(&st->prompt_done, ACQ),
               atomic_load_explicit(&st->reused, ACQ), gen, atomic_load_explicit(&st->passes, ACQ),
               atomic_load_explicit(&st->max_tokens, ACQ), el, ts);
    buf_printf(o,
               ",\"requests\":%llu,\"prompt_tokens\":%llu,\"reused_tokens\":%llu,\"output_tokens\":%llu,"
               "\"output_passes\":%llu,\"ck\":[%u,%u],\"convs\":%u,\"recent\":[",
               (unsigned long long)atomic_load_explicit(&st->requests, ACQ),
               (unsigned long long)atomic_load_explicit(&st->prompt_tokens, ACQ),
               (unsigned long long)atomic_load_explicit(&st->reused_tokens, ACQ),
               (unsigned long long)atomic_load_explicit(&st->output_tokens, ACQ),
               (unsigned long long)atomic_load_explicit(&st->output_passes, ACQ),
               atomic_load_explicit(&st->ck0, ACQ), atomic_load_explicit(&st->ck1, ACQ),
               atomic_load_explicit(&st->convs, ACQ));
    lock(st);
    for (size_t i = 0; i < st->ring_n; i++) {
        const Done *d = &st->ring[i];
        if (i > 0) buf_addc(o, ',');
        buf_printf(o,
                   "{\"end\":%lld,\"api\":\"%s\",\"prompt\":%u,\"reused\":%u,\"output\":%u,\"passes\":%u,"
                   "\"prefill_ms\":%u,\"decode_ms\":%u,\"finish\":\"%.*s\"}",
                   (long long)d->end_s, d->anthropic ? "anthropic" : "openai", d->prompt, d->reused, d->output,
                   d->passes, d->prefill_ms, d->decode_ms, (int)strnlen(d->finish, sizeof d->finish), d->finish);
    }
    unlock(st);
    buf_adds(o, "],\"conversations\":[");
    lock(st);
    for (size_t i = 0; i < st->conv_n; i++) {
        const ConvInfo *cv = &st->conv[i];
        if (i > 0) buf_addc(o, ',');
        buf_printf(o, "{\"tokens\":%u,\"cap\":%u,\"last\":%lld,\"active\":%s}", cv->tokens, cv->cap,
                   (long long)cv->last_s, cv->active ? "true" : "false");
    }
    unlock(st);
    buf_adds(o, "]}\n");
}

static Err write_snapshot(Stats *st, const char *dir) {
    Buf o = {0};
    stats_json(st, &o);
    char tmp[1024], dst[1024];
    snprintf(tmp, sizeof tmp, "%s/metrics.json.tmp", dir);
    snprintf(dst, sizeof dst, "%s/metrics.json", dir);
    Err e = write_file(tmp, o.ptr, o.len);
    if (!e && rename(tmp, dst) != 0) e = err_errno(errno);
    buf_free(&o);
    return e;
}

typedef struct {
    Stats *st;
    char *dir;
    bool autostop;
} LoopArgs;

static void *loop(void *p) {
    LoopArgs *la = p;
    Stats *st = la->st;
    uint32_t lonely_ticks = 0;
    for (uint32_t tick = 0;; tick++) {
        // every 100 ms: the snapshot when a request just finished, else every 500 ms
        if (tick % 5 == 0 || atomic_exchange_explicit(&st->dirty, false, AR)) (void)write_snapshot(st, la->dir);
        if (la->autostop && tick % 20 == 0) {
            // sessions alive? (stale entries of dead processes are removed on the way)
            size_t live = instance_count_clients() + (instance_kept() ? 1 : 0);
            bool busy = atomic_load_explicit(&st->state, ACQ) != STATE_IDLE;
            lonely_ticks = live == 0 && !busy ? lonely_ticks + 1 : 0;
            if (lonely_ticks >= 10) { // 10 checks, 2 s apart
                dbg("saetta: no agent session has been alive for 20 s; stopping\n");
                exit(0);
            }
        }
        struct timespec ts = {0, 100 * 1000000};
        nanosleep(&ts, NULL);
    }
    return NULL;
}

Err stats_spawn(Stats *st) {
    const char *home = getenv("HOME");
    if (!home) return OK;
    LoopArgs *la = xmalloc(sizeof *la);
    size_t n = strlen(home) + 16;
    la->dir = xmalloc(n);
    snprintf(la->dir, n, "%s/.saetta/run", home);
    const char *v = getenv("SAETTA_AUTOSTOP");
    la->autostop = v && strcmp(v, "1") == 0;
    la->st = st;
    pthread_t t;
    int rc = pthread_create(&t, NULL, loop, la);
    if (rc != 0) {
        free(la->dir);
        free(la);
        return err_errno(rc);
    }
    pthread_detach(t);
    return OK;
}
