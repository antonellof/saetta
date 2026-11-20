// Expert cache for streamed packs (xcache.h). The tiered placement (hot experts resident,
// cold ones read from the SSD when the router asks, a loader pool reading while compute
// waits) follows the approach of colibri (github.com/JustVugg/colibri, Apache-2.0); no code
// is taken from it.
#include "xcache.h"

#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define XC_ALIGN 16384u

typedef struct {
    int fd;
    uint64_t off, len;
    uint8_t *dst;
} XcJob;

struct XCache {
    uint32_t nlayers, nt, max_ne, n_slots;
    uint32_t stride[XC_MAX_TENSORS];
    sx_buffer *region[XC_MAX_TENSORS];
    uint8_t *base[XC_MAX_TENSORS];
    // per layer and tensor: file descriptor, offset of expert 0 and bytes per expert
    int *fd;           // [nlayers * nt], stacked layers
    uint64_t *off0;    // [nlayers * nt]
    uint64_t *per;     // [nlayers * nt]
    const XcRange **ranges; // [nlayers], explicit layers (NULL for stacked)
    uint32_t *ne;      // [nlayers]
    int32_t *slot_of;  // [nlayers * max_ne], -1 when not cached
    uint32_t *key;     // [n_slots], layer * max_ne + expert, UINT32_MAX when empty
    uint32_t *prev, *next; // LRU list over slots; head is most recent
    uint32_t head, tail;
    uint64_t *mark;    // [n_slots], epoch of the xc_map call that returned it
    uint64_t epoch;
    int *files;        // one F_NOCACHE fd per pack file
    size_t nfiles;
    // loader pool
    XcJob *jobs;       // [max_ne * nt]
    uint32_t njobs;
    _Atomic uint32_t next_job, done_jobs, failed;
    // Workers that finished the current batch. run_jobs returns only when every loader has
    // acknowledged it, so no worker can still be reading jobs[] or njobs when the next
    // xc_map refills them (a worker done with its last job used to race with the refill).
    _Atomic uint32_t acked;
    pthread_t *threads;
    uint32_t nthreads;
    // Per worker (the loader threads, then the calling thread): a page-aligned staging buffer
    // the read lands in before it is copied to its slot. Reading straight into a slot with
    // F_NOCACHE makes the kernel remap the slot's pages for the transfer, and the GPU then
    // faults them back in on its next read (about 3 ms per layer of Flash-Next experts).
    uint8_t **stage;
    uint64_t stage_len;
    _Atomic uint32_t next_worker;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    uint64_t gen;
    bool stop;
    XcStats stats;
};

#define NIL UINT32_MAX

static void lru_unlink(XCache *c, uint32_t s) {
    if (c->prev[s] != NIL) c->next[c->prev[s]] = c->next[s]; else c->head = c->next[s];
    if (c->next[s] != NIL) c->prev[c->next[s]] = c->prev[s]; else c->tail = c->prev[s];
    c->prev[s] = c->next[s] = NIL;
}

static void lru_front(XCache *c, uint32_t s) {
    c->prev[s] = NIL;
    c->next[s] = c->head;
    if (c->head != NIL) c->prev[c->head] = s;
    c->head = s;
    if (c->tail == NIL) c->tail = s;
}

static bool run_job(const XcJob *j, uint8_t *stage, uint64_t stage_len) {
    uint64_t done = 0;
    while (done < j->len) {
        uint64_t want = j->len - done < stage_len ? j->len - done : stage_len;
        uint64_t got_all = 0;
        while (got_all < want) {
            ssize_t got = pread(j->fd, stage + got_all, (size_t)(want - got_all), (off_t)(j->off + done + got_all));
            if (got <= 0) return false;
            got_all += (uint64_t)got;
        }
        memcpy(j->dst + done, stage, (size_t)want);
        done += want;
    }
    return true;
}

static void work(XCache *c, uint32_t w) {
    for (;;) {
        uint32_t i = atomic_fetch_add(&c->next_job, 1);
        if (i >= c->njobs) return;
        if (!run_job(&c->jobs[i], c->stage[w], c->stage_len)) atomic_store(&c->failed, 1);
        atomic_fetch_add(&c->done_jobs, 1);
    }
}

static void *loader(void *arg) {
    XCache *c = arg;
    uint32_t w = atomic_fetch_add(&c->next_worker, 1);
    uint64_t seen = 0;
    for (;;) {
        pthread_mutex_lock(&c->mu);
        while (!c->stop && c->gen == seen) pthread_cond_wait(&c->cv, &c->mu);
        if (c->stop) {
            pthread_mutex_unlock(&c->mu);
            return NULL;
        }
        seen = c->gen;
        pthread_mutex_unlock(&c->mu);
        work(c, w);
        atomic_fetch_add_explicit(&c->acked, 1, memory_order_release);
    }
}

// Runs the queued jobs on the pool and the calling thread; returns false if a read failed.
static bool run_jobs(XCache *c) {
    if (c->njobs == 0) return true;
    atomic_store(&c->next_job, 0);
    atomic_store(&c->done_jobs, 0);
    atomic_store(&c->failed, 0);
    atomic_store(&c->acked, 0);
    pthread_mutex_lock(&c->mu); // publishes jobs[] and njobs to the workers it wakes
    c->gen++;
    pthread_cond_broadcast(&c->cv);
    pthread_mutex_unlock(&c->mu);
    work(c, c->nthreads);
    // every job finished and every loader out of work(): only then may the caller refill
    while (atomic_load_explicit(&c->acked, memory_order_acquire) < c->nthreads) {
    }
    return atomic_load(&c->done_jobs) == c->njobs && atomic_load(&c->failed) == 0;
}

static Err open_impl(Arena *a, const Pack *p, sx_device *d, const XcLayer *layers, uint32_t nlayers, uint32_t nt,
                     uint64_t cache_bytes, uint32_t threads, XCache **out, XCache **partial);

// One F_NOCACHE descriptor per pack file, opened on first use.
static Err open_file(Arena *a, const Pack *p, XCache *c, uint16_t f) {
    if (c->files[f] >= 0) return OK;
    char *path = arena_printf(a, "%s/%s", p->dir, p->files[f].name);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return "OpenFailed";
    fcntl(fd, F_NOCACHE, 1);
    fcntl(fd, F_RDAHEAD, 0);
    c->files[f] = fd;
    return OK;
}

// Releases what a failed open_impl acquired (files, slab regions, staging); no thread runs yet.
static void open_cleanup(XCache *c) {
    if (!c) return;
    for (uint32_t r = 0; r < XC_MAX_TENSORS; r++)
        if (c->region[r]) sx_buffer_release(c->region[r]);
    if (c->files)
        for (size_t f = 0; f < c->nfiles; f++)
            if (c->files[f] >= 0) close(c->files[f]);
    if (c->stage)
        for (uint32_t i = 0; i <= c->nthreads; i++) free(c->stage[i]);
}

Err xc_open(Arena *a, const Pack *p, sx_device *d, const XcLayer *layers, uint32_t nlayers, uint32_t nt,
            uint64_t cache_bytes, uint32_t threads, XCache **out) {
    XCache *partial = NULL;
    Err e = open_impl(a, p, d, layers, nlayers, nt, cache_bytes, threads, out, &partial);
    if (e) open_cleanup(partial);
    return e;
}

static Err open_impl(Arena *a, const Pack *p, sx_device *d, const XcLayer *layers, uint32_t nlayers, uint32_t nt,
                     uint64_t cache_bytes, uint32_t threads, XCache **out, XCache **partial) {
    if (nt == 0 || nt > XC_MAX_TENSORS || nlayers == 0) return "BadExpertLayout";
    XCache *c = ARENA_NEW(a, XCache);
    if (!c) return "OutOfMemory";
    *partial = c;
    c->nlayers = nlayers;
    c->nt = nt;
    c->fd = ARENA_ARRAY(a, int, nlayers * nt);
    c->off0 = ARENA_ARRAY(a, uint64_t, nlayers * nt);
    c->per = ARENA_ARRAY(a, uint64_t, nlayers * nt);
    c->ne = ARENA_ARRAY(a, uint32_t, nlayers);
    c->ranges = ARENA_ARRAY(a, const XcRange *, nlayers);
    c->files = ARENA_ARRAY(a, int, p->nfiles ? p->nfiles : 1);
    if (!c->fd || !c->off0 || !c->per || !c->ne || !c->ranges || !c->files) return "OutOfMemory";
    for (size_t f = 0; f < p->nfiles; f++) c->files[f] = -1;
    c->nfiles = p->nfiles; // only now: cleanup closes files[0..nfiles), which must be -1 or open
    for (uint32_t l = 0; l < nlayers; l++) {
        uint32_t ne = layers[l].ne;
        if (ne == 0) return "BadExpertLayout";
        c->ne[l] = ne;
        if (ne > c->max_ne) c->max_ne = ne;
        if (layers[l].ranges) {
            c->ranges[l] = layers[l].ranges;
            for (uint32_t e = 0; e < ne; e++)
                for (uint32_t r = 0; r < nt; r++) {
                    const XcRange *g = &layers[l].ranges[(size_t)e * nt + r];
                    if (g->len == 0 || g->file >= p->nfiles || g->off + g->len > p->files[g->file].len)
                        return "BadExpertLayout";
                    uint64_t st = (g->len + XC_ALIGN - 1) / XC_ALIGN * XC_ALIGN;
                    if (st > UINT32_MAX) return "BadExpertLayout";
                    if (st > c->stride[r]) c->stride[r] = (uint32_t)st;
                    TRY(open_file(a, p, c, g->file));
                }
            continue;
        }
        for (uint32_t r = 0; r < nt; r++) {
            const TensorInfo *t = layers[l].t[r];
            if (!t || t->nbytes % ne != 0 || t->file >= p->nfiles) return "BadExpertLayout";
            uint64_t per = t->nbytes / ne;
            c->per[l * nt + r] = per;
            c->off0[l * nt + r] = t->offset;
            uint64_t st = (per + XC_ALIGN - 1) / XC_ALIGN * XC_ALIGN;
            if (st > UINT32_MAX) return "BadExpertLayout";
            if (st > c->stride[r]) c->stride[r] = (uint32_t)st;
            TRY(open_file(a, p, c, t->file));
            c->fd[l * nt + r] = c->files[t->file];
        }
    }
    uint64_t slot_bytes = 0;
    for (uint32_t r = 0; r < nt; r++) slot_bytes += c->stride[r];
    uint64_t n = cache_bytes / slot_bytes;
    if (n < c->max_ne) return "CacheTooSmall";
    if (n > (uint64_t)nlayers * c->max_ne) n = (uint64_t)nlayers * c->max_ne;
    c->n_slots = (uint32_t)n;
    for (uint32_t r = 0; r < nt; r++) {
        c->region[r] = sx_buffer_new(d, (size_t)n * c->stride[r]);
        if (!c->region[r]) return "OutOfMemory";
        c->base[r] = sx_buffer_contents(c->region[r]);
    }
    c->slot_of = ARENA_ARRAY(a, int32_t, (size_t)nlayers * c->max_ne);
    c->key = ARENA_ARRAY(a, uint32_t, n);
    c->prev = ARENA_ARRAY(a, uint32_t, n);
    c->next = ARENA_ARRAY(a, uint32_t, n);
    c->mark = ARENA_ARRAY(a, uint64_t, n);
    c->jobs = ARENA_ARRAY(a, XcJob, (size_t)c->max_ne * nt);
    if (!c->slot_of || !c->key || !c->prev || !c->next || !c->mark || !c->jobs) return "OutOfMemory";
    for (size_t i = 0; i < (size_t)nlayers * c->max_ne; i++) c->slot_of[i] = -1;
    c->head = c->tail = NIL;
    for (uint32_t s = 0; s < n; s++) {
        c->key[s] = NIL;
        c->prev[s] = c->next[s] = NIL;
        lru_front(c, s);
    }
    pthread_mutex_init(&c->mu, NULL);
    pthread_cond_init(&c->cv, NULL);
    c->nthreads = threads ? threads : 4;
    c->threads = ARENA_ARRAY(a, pthread_t, c->nthreads);
    c->stage = ARENA_ARRAY(a, uint8_t *, c->nthreads + 1);
    if (!c->threads || !c->stage) return "OutOfMemory";
    c->stage_len = 4u << 20;
    for (uint32_t i = 0; i <= c->nthreads; i++) {
        void *p = NULL;
        if (posix_memalign(&p, XC_ALIGN, (size_t)c->stage_len)) return "OutOfMemory";
        c->stage[i] = p;
    }
    for (uint32_t i = 0; i < c->nthreads; i++)
        if (pthread_create(&c->threads[i], NULL, loader, c)) { c->nthreads = i; break; }
    *out = c;
    return OK;
}

void xc_close(XCache *c) {
    if (!c) return;
    pthread_mutex_lock(&c->mu);
    c->stop = true;
    pthread_cond_broadcast(&c->cv);
    pthread_mutex_unlock(&c->mu);
    for (uint32_t i = 0; i < c->nthreads; i++) pthread_join(c->threads[i], NULL);
    for (uint32_t r = 0; r < c->nt; r++) sx_buffer_release(c->region[r]);
    for (uint32_t i = 0; i <= c->nthreads; i++) free(c->stage[i]);
    for (size_t f = 0; f < c->nfiles; f++)
        if (c->files[f] >= 0) close(c->files[f]);
}

uint32_t xc_slots(const XCache *c) { return c->n_slots; }
sx_buffer *xc_region(const XCache *c, uint32_t r) { return c->region[r]; }
uint32_t xc_stride(const XCache *c, uint32_t r) { return c->stride[r]; }
uint64_t xc_bytes(const XCache *c) {
    uint64_t b = 0;
    for (uint32_t r = 0; r < c->nt; r++) b += (uint64_t)c->n_slots * c->stride[r];
    return b;
}
XcStats xc_stats(const XCache *c) { return c->stats; }
uint64_t xc_requested_bytes(void) {
    const char *v = getenv("SAETTA_EXPERT_CACHE_GB");
    if (!v || !*v) return 0;
    double gb = atof(v);
    return gb > 0 ? (uint64_t)(gb * 1e9) : 0;
}
void xc_reset_stats(XCache *c) { memset(&c->stats, 0, sizeof c->stats); }

Err xc_map(XCache *c, uint32_t layer, const uint32_t *ids, uint32_t n, uint32_t *slots_out) {
    if (layer >= c->nlayers) return "BadExpertId";
    // reject bad ids before any state changes: a miss is recorded in slot_of before its read
    // runs, so returning half way would leave slots that look cached but hold other bytes
    for (uint32_t i = 0; i < n; i++)
        if (ids[i] >= c->ne[layer]) return "BadExpertId";
    const uint64_t ep = ++c->epoch;
    Err err = OK;
    c->njobs = 0;
    uint64_t missed = 0, bytes = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t e = ids[i];
        size_t k = (size_t)layer * c->max_ne + e;
        int32_t s = c->slot_of[k];
        c->stats.requests++;
        if (s >= 0) {
            if (c->mark[s] != ep) c->stats.hits++; // a repeat within the call is not a new hit
            else c->stats.requests--;
            lru_unlink(c, (uint32_t)s);
            lru_front(c, (uint32_t)s);
            c->mark[s] = ep;
            slots_out[i] = (uint32_t)s;
            continue;
        }
        // victim: least recently used slot not returned by this call
        uint32_t v = c->tail;
        while (v != NIL && c->mark[v] == ep) v = c->prev[v];
        if (v == NIL) {
            // more distinct experts than slots (not reachable: slots >= ne); still read what
            // was queued so every recorded slot holds its expert
            err = "CacheTooSmall";
            break;
        }
        if (c->key[v] != NIL) c->slot_of[c->key[v]] = -1;
        c->key[v] = (uint32_t)k;
        c->slot_of[k] = (int32_t)v;
        c->mark[v] = ep;
        lru_unlink(c, v);
        lru_front(c, v);
        for (uint32_t r = 0; r < c->nt; r++) {
            XcJob *j = &c->jobs[c->njobs++];
            if (c->ranges[layer]) {
                const XcRange *g = &c->ranges[layer][(size_t)e * c->nt + r];
                j->fd = c->files[g->file];
                j->off = g->off;
                j->len = g->len;
            } else {
                size_t lr = (size_t)layer * c->nt + r;
                j->fd = c->fd[lr];
                j->off = c->off0[lr] + (uint64_t)e * c->per[lr];
                j->len = c->per[lr];
            }
            j->dst = c->base[r] + (size_t)v * c->stride[r];
            bytes += j->len;
        }
        missed++;
        slots_out[i] = v;
    }
    double t0 = now_s();
    bool ok = run_jobs(c);
    c->stats.read_seconds += now_s() - t0;
    c->stats.misses += missed;
    c->stats.bytes_read += bytes;
    if (!ok) {
        // drop every slot this call filled: their contents are unknown
        for (uint32_t i = 0; i < c->njobs; i += c->nt) {
            uint32_t v = (uint32_t)((c->jobs[i].dst - c->base[0]) / c->stride[0]);
            if (c->key[v] != NIL) c->slot_of[c->key[v]] = -1;
            c->key[v] = NIL;
        }
        return "ReadFailed";
    }
    return err;
}
