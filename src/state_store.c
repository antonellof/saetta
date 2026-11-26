// Conversation state on the SSD (see state_store.h).
#include "state_store.h"

#include <arm_acle.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include "hash.h"

#define ST_THREADS 4
// tokens hashed for the lookup's first filter (a state always has more)
#define ST_FIRST 64
// free disk kept beyond a state's size
#define ST_DISK_FLOOR ((uint64_t)8 << 30)

// ---- CRC32C ------------------------------------------------------------------------------------

uint32_t crc32c(uint32_t crc, const void *p, size_t n) {
    const uint8_t *b = p;
    crc = ~crc;
    while (n && ((uintptr_t)b & 7)) {
        crc = __crc32cb(crc, *b++);
        n--;
    }
    while (n >= 8) {
        uint64_t v;
        memcpy(&v, b, 8);
        crc = __crc32cd(crc, v);
        b += 8;
        n -= 8;
    }
    while (n--) crc = __crc32cb(crc, *b++);
    return ~crc;
}

// Four chains in one loop: the CRC instruction's latency is hidden by the other three.
void crc32c_quad(const void *p, size_t n, uint32_t out[4]) {
    const uint8_t *b = p;
    size_t q = (n / 4) & ~(size_t)7;
    const uint8_t *p0 = b, *p1 = b + q, *p2 = b + 2 * q, *p3 = b + 3 * q;
    uint32_t c0 = ~0u, c1 = ~0u, c2 = ~0u, c3 = ~0u;
    for (size_t i = 0; i < q; i += 8) {
        uint64_t v0, v1, v2, v3;
        memcpy(&v0, p0 + i, 8);
        memcpy(&v1, p1 + i, 8);
        memcpy(&v2, p2 + i, 8);
        memcpy(&v3, p3 + i, 8);
        c0 = __crc32cd(c0, v0);
        c1 = __crc32cd(c1, v1);
        c2 = __crc32cd(c2, v2);
        c3 = __crc32cd(c3, v3);
    }
    out[0] = ~c0;
    out[1] = ~c1;
    out[2] = ~c2;
    out[3] = crc32c(~c3, p3 + q, n - 4 * q);
}

// ---- file formats ------------------------------------------------------------------------------

typedef struct {
    uint64_t off, len;
    uint32_t crc[4];
} StSlice;

// Rows [r0, r1) of every plane of every region, plane after plane, region after region.
typedef struct {
    uint64_t r0, r1;
    uint32_t nslices, pad;
    StSlice s[ST_MAX_SLICES];
} RowBlock;

typedef struct {
    char magic[8]; // "SAETTACK"
    uint32_t format, crc;
    uint8_t shape[32];
    uint8_t key[32]; // sha256 of the lineage that wrote it and the ids before r1
    RowBlock rb;
} ChunkHead;

typedef struct {
    char magic[8]; // "SAETTAST"
    uint32_t format, crc;
    uint8_t shape[32];
    uint8_t name[32];    // sha256 of the lineage and the ids (the file's name)
    uint8_t ids_hash[32]; // sha256 of the ids
    uint8_t first[32];   // sha256 of the first ST_FIRST ids
    uint8_t lineage[16]; // the lineage its own chunks were written by
    uint64_t n, nchunks;
    int64_t created_s;
    StSlice ids, keys;
    RowBlock tail;
    uint32_t nrec, pad;
    StSlice rec[ST_MAX_REC];
    uint32_t prev[2];
    uint64_t since_eos;
    uint64_t extra_len;
    uint8_t extra[ST_EXTRA_MAX];
} StateHead;

_Static_assert(sizeof(ChunkHead) <= ST_PAGE, "chunk header page");
_Static_assert(sizeof(StateHead) <= ST_PAGE, "state header page");

static size_t align_page(size_t x) { return (x + ST_PAGE - 1) / ST_PAGE * ST_PAGE; }

// The crc of a header page, with its crc field (at offset 12) taken as zero.
static uint32_t page_crc(const uint8_t *page) {
    static const uint8_t zero[4];
    uint32_t c = crc32c(0, page, 12);
    c = crc32c(c, zero, 4);
    return crc32c(c, page + 16, ST_PAGE - 16);
}

static bool in_file(const StSlice *s, uint64_t fsize) { return s->off <= fsize && s->len <= fsize - s->off; }

static size_t layout_slices(const StLayout *lay) {
    size_t k = 0;
    for (size_t i = 0; i < lay->nreg; i++) k += lay->reg[i].planes;
    return k;
}

// Slice k of a row block over rows [r0, r1): its memory in lay and its length.
static uint8_t *slice_mem(const StLayout *lay, size_t k, uint64_t r0, uint64_t r1, size_t *len) {
    for (size_t i = 0; i < lay->nreg; i++) {
        const StRegion *g = &lay->reg[i];
        if (k < g->planes) {
            size_t a = (size_t)r0 / g->per_row, b = (size_t)r1 / g->per_row;
            *len = (b - a) * g->row_bytes;
            return g->base + k * g->plane_stride + a * g->row_bytes;
        }
        k -= g->planes;
    }
    *len = 0;
    return NULL;
}

// The table of a row block over rows [r0, r1) placed from file offset *off, its crcs from memory.
static void rowblock_plan(const StLayout *lay, uint64_t r0, uint64_t r1, uint64_t *off, RowBlock *rb) {
    memset(rb, 0, sizeof *rb);
    rb->r0 = r0;
    rb->r1 = r1;
    rb->nslices = (uint32_t)layout_slices(lay);
    for (size_t k = 0; k < rb->nslices; k++) {
        size_t len;
        const uint8_t *m = slice_mem(lay, k, r0, r1, &len);
        rb->s[k].off = *off;
        rb->s[k].len = len;
        crc32c_quad(m, len, rb->s[k].crc);
        *off = align_page(*off + len);
    }
}

static bool rowblock_in_file(const RowBlock *rb, uint64_t fsize) {
    if (rb->nslices > ST_MAX_SLICES || rb->r1 < rb->r0) return false;
    for (size_t k = 0; k < rb->nslices; k++)
        if (!in_file(&rb->s[k], fsize)) return false;
    return true;
}

// ---- I/O ---------------------------------------------------------------------------------------

static Err pwrite_all(int fd, const void *p, size_t n, uint64_t off) {
    const uint8_t *b = p;
    while (n) {
        size_t step = n > ((size_t)64 << 20) ? ((size_t)64 << 20) : n;
        ssize_t w = pwrite(fd, b, step, (off_t)off);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return "WriteFailed";
        b += w;
        n -= (size_t)w;
        off += (uint64_t)w;
    }
    return OK;
}

static Err pread_all(int fd, void *p, size_t n, uint64_t off) {
    uint8_t *b = p;
    while (n) {
        size_t step = n > ((size_t)64 << 20) ? ((size_t)64 << 20) : n;
        ssize_t r = pread(fd, b, step, (off_t)off);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return "ReadFailed";
        b += r;
        n -= (size_t)r;
        off += (uint64_t)r;
    }
    return OK;
}

static uint64_t fd_size(int fd) {
    struct stat sb;
    return fstat(fd, &sb) == 0 && sb.st_size > 0 ? (uint64_t)sb.st_size : 0;
}

static void hex16(const uint8_t *d, char out[33]) { hash_hex(d, 16, out); }

// little endian u32 in memory on this machine
static void ids_hash(const uint32_t *ids, size_t n, uint8_t out[32]) { sha256(ids, n * sizeof(uint32_t), out); }

// sha256 of a lineage and ids[0..n): a chunk's key (n a multiple of ST_CHUNK) or a state's name.
static void lineage_hash(const uint8_t lineage[16], const uint32_t *ids, size_t n, uint8_t out[32]) {
    Sha256 h;
    sha256_init(&h);
    sha256_update(&h, lineage, 16);
    sha256_update(&h, ids, n * sizeof(uint32_t));
    sha256_final(&h, out);
}

// ---- the store ---------------------------------------------------------------------------------

typedef struct {
    char name[33];
    uint64_t n, bytes;
    uint8_t first[32];
    int64_t mtime;
    bool skip;   // could not be removed
    bool parsed; // its header and chunk keys were read (keys[0..nkeys))
    uint8_t (*keys)[16];
    size_t nkeys;
} StEntry;

struct StateStore {
    char dir[1024];
    uint8_t shape[32];
    int lock_fd;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_t thr;
    bool quit;
    // the write job (job: queued or running)
    bool job;
    StLayout lay;
    uint32_t *ids;
    size_t ids_cap, n;
    StMeta meta;
    uint8_t (*inh)[32]; // the inherited chunks' keys of the job
    size_t ninh, inh_cap;
    uint8_t *rec;
    size_t rec_cap;
    void *hold[ST_MAX_REGIONS];
    size_t nhold;
    void (*release)(void *);
    // the states on disk (scan)
    StEntry *ents;
    size_t nents;
    uint64_t total; // bytes of states and chunks
    char root[1024], pack_dir[1100];
};

static char *pathf(char *buf, size_t cap, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
static char *pathf(char *buf, size_t cap, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, cap, fmt, ap);
    va_end(ap);
    return buf;
}

static bool ends_with(const char *s, const char *suf) {
    size_t a = strlen(s), b = strlen(suf);
    return a >= b && !strcmp(s + a - b, suf);
}

const char *state_store_dir(const StateStore *st) { return st->dir; }

// Reads and checks a state's header page: its format, crc and shape, and that every section it
// names lies inside the file (so nothing below allocates or reads from an unchecked length).
// shape: the one expected, or NULL with shape_hex, the store folder's name (another store's).
static bool head_ok(int fd, StateHead *h, uint8_t *page, const uint8_t *shape, const char *shape_hex) {
    uint64_t fsize = fd_size(fd);
    if (fsize < ST_PAGE || pread_all(fd, page, ST_PAGE, 0)) return false;
    memcpy(h, page, sizeof *h);
    if (memcmp(h->magic, "SAETTAST", 8) || h->format != ST_FORMAT || h->crc != page_crc(page)) return false;
    if (shape && memcmp(h->shape, shape, 32)) return false;
    if (!shape) {
        char hx[17];
        hash_hex(h->shape, 8, hx);
        if (strcmp(hx, shape_hex)) return false;
    }
    if (h->n < ST_FIRST || h->n > fsize / 4 || h->nchunks != h->n / ST_CHUNK) return false;
    if (h->ids.len != h->n * 4 || !in_file(&h->ids, fsize)) return false;
    if (h->keys.len != h->nchunks * 32 || !in_file(&h->keys, fsize)) return false;
    if (h->nrec > ST_MAX_REC || h->extra_len > ST_EXTRA_MAX) return false;
    for (size_t i = 0; i < h->nrec; i++)
        if (!in_file(&h->rec[i], fsize)) return false;
    return rowblock_in_file(&h->tail, fsize) && h->tail.r0 == h->nchunks * ST_CHUNK && h->tail.r1 == h->n;
}

// Reads and checks chunk c's header against its expected key.
static bool read_state_head(const StateStore *st, int fd, StateHead *h, uint8_t *page) {
    return head_ok(fd, h, page, st->shape, NULL);
}

static bool read_chunk_head(const StateStore *st, int fd, const uint8_t key[32], size_t c, ChunkHead *h, uint8_t *page) {
    uint64_t fsize = fd_size(fd);
    if (fsize < ST_PAGE || pread_all(fd, page, ST_PAGE, 0)) return false;
    memcpy(h, page, sizeof *h);
    return !memcmp(h->magic, "SAETTACK", 8) && h->format == ST_FORMAT && h->crc == page_crc(page) &&
           !memcmp(h->shape, st->shape, 32) && !memcmp(h->key, key, 32) && h->rb.r0 == c * ST_CHUNK &&
           h->rb.r1 == (c + 1) * ST_CHUNK && rowblock_in_file(&h->rb, fsize);
}

static int cmp_key(const void *a, const void *b) { return memcmp(a, b, 16); }

// Removes a state file unless a load holds it: a load keeps a shared flock on the state file while
// it reads (in any process), so an exclusive non-blocking flock fails then and the state stays.
// True when it is gone.
static bool remove_state(const char *path) {
    int fd = open(path, O_RDONLY | O_NOFOLLOW);
    if (fd < 0) return errno == ENOENT;
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        close(fd);
        return false;
    }
    bool ok = unlink(path) == 0 || errno == ENOENT;
    close(fd);
    return ok;
}

// A temporary file of a process that is gone (name: <file>.tmp.<pid>.<random>).
static bool stale_tmp(const char *name) {
    const char *t = strstr(name, ".tmp.");
    if (!t) return false;
    long pid = strtol(t + 5, NULL, 10);
    if (pid <= 0) return true;
    return kill((pid_t)pid, 0) != 0 && errno == ESRCH;
}

typedef struct {
    uint8_t key[16];
    uint64_t bytes;
    bool skip;
} Ck;

// One store folder as a scan sees it (this store's own, or another pack's or shape's).
typedef struct {
    char dir[1100];
    char shape_hex[17];
    bool own, same_pack;
    StEntry *ents;
    size_t nents, ecap;
    Ck *cks;
    size_t ncks, ccap;
    uint64_t bytes;
    size_t unread; // states whose chunk keys are unknown (not readable now, or of another format)
} View;

// Reads a state's chunk keys (h checked) into e.
static bool read_keys(int fd, const StateHead *h, StEntry *e) {
    uint8_t *keys = xmalloc(h->keys.len + 1);
    bool ok = !pread_all(fd, keys, h->keys.len, h->keys.off) && crc32c(0, keys, h->keys.len) == h->keys.crc[0];
    if (ok) {
        e->keys = xmalloc((h->nchunks + 1) * 16);
        for (size_t c = 0; c < h->nchunks; c++) memcpy(e->keys[c], keys + c * 32, 16);
        e->nkeys = h->nchunks;
        e->parsed = true;
    }
    free(keys);
    return ok;
}

static void entry_free(StEntry *e) {
    free(e->keys);
    e->keys = NULL;
}

// Scans one store folder: dead processes' temporary files and states past the age limit go; this
// store's states that fail their checks go (they are damaged); another store's that this build
// cannot read stay, with their chunks (their keys unknown).
static void scan_view(StateStore *st, View *v, int64_t max_age_s, int64_t now, bool sweep_tmp, uint8_t *page) {
    char p[1200];
    DIR *d = opendir(pathf(p, sizeof p, "%s/states", v->dir));
    struct dirent *de;
    while (d && (de = readdir(d))) {
        if (de->d_name[0] == '.') continue;
        char f[1300];
        pathf(f, sizeof f, "%s/states/%s", v->dir, de->d_name);
        if (strstr(de->d_name, ".tmp.")) {
            if (sweep_tmp && stale_tmp(de->d_name)) unlink(f);
            continue;
        }
        if (!ends_with(de->d_name, ".st") || strlen(de->d_name) != 35) continue;
        struct stat sb;
        if (lstat(f, &sb) || !S_ISREG(sb.st_mode)) continue;
        if (now - (int64_t)sb.st_mtime > max_age_s && remove_state(f)) continue;
        StEntry e = {0};
        memcpy(e.name, de->d_name, 32);
        e.bytes = (uint64_t)sb.st_size;
        e.mtime = (int64_t)sb.st_mtime;
        int fd = open(f, O_RDONLY | O_NOFOLLOW);
        if (fd >= 0) {
            StateHead h;
            if (head_ok(fd, &h, page, v->own ? st->shape : NULL, v->shape_hex) && read_keys(fd, &h, &e)) {
                e.n = h.n;
                memcpy(e.first, h.first, 32);
            } else if (v->own) {
                close(fd);
                if (remove_state(f)) continue;
                fd = -1;
            }
            if (fd >= 0) close(fd);
        }
        if (!e.parsed) v->unread++;
        if (v->nents == v->ecap) v->ents = xrealloc(v->ents, (v->ecap = v->ecap ? v->ecap * 2 : 64) * sizeof *v->ents);
        v->ents[v->nents++] = e;
        v->bytes += e.bytes;
    }
    if (d) closedir(d);
    d = opendir(pathf(p, sizeof p, "%s/chunks", v->dir));
    while (d && (de = readdir(d))) {
        if (de->d_name[0] == '.') continue;
        char f[1300];
        pathf(f, sizeof f, "%s/chunks/%s", v->dir, de->d_name);
        if (strstr(de->d_name, ".tmp.")) {
            if (sweep_tmp && stale_tmp(de->d_name)) unlink(f);
            continue;
        }
        if (!ends_with(de->d_name, ".kv") || strlen(de->d_name) != 35) continue;
        struct stat sb;
        if (lstat(f, &sb) || !S_ISREG(sb.st_mode)) continue;
        if (v->ncks == v->ccap) v->cks = xrealloc(v->cks, (v->ccap = v->ccap ? v->ccap * 2 : 64) * sizeof *v->cks);
        Ck *c = &v->cks[v->ncks++];
        for (size_t i = 0; i < 16; i++) {
            unsigned x;
            sscanf(de->d_name + 2 * i, "%2x", &x);
            c->key[i] = (uint8_t)x;
        }
        c->bytes = (uint64_t)sb.st_size;
        c->skip = false;
        v->bytes += c->bytes;
    }
    if (d) closedir(d);
}

// Removes the view's chunks no state uses; nothing while a state's keys are unknown (its chunks
// would look unused). Returns the bytes removed.
static uint64_t sweep_view(View *v) {
    if (v->unread) return 0;
    size_t nrefs = 0;
    for (size_t i = 0; i < v->nents; i++) nrefs += v->ents[i].nkeys;
    uint8_t (*refs)[16] = xmalloc((nrefs + 1) * 16);
    nrefs = 0;
    for (size_t i = 0; i < v->nents; i++)
        for (size_t c = 0; c < v->ents[i].nkeys; c++) memcpy(refs[nrefs++], v->ents[i].keys[c], 16);
    if (nrefs) qsort(refs, nrefs, 16, cmp_key);
    uint64_t freed = 0;
    for (size_t i = 0; i < v->ncks;) {
        if (v->cks[i].skip || (nrefs && bsearch(v->cks[i].key, refs, nrefs, 16, cmp_key))) {
            i++;
            continue;
        }
        char f[1300], hx[33];
        hex16(v->cks[i].key, hx);
        if (unlink(pathf(f, sizeof f, "%s/chunks/%s.kv", v->dir, hx)) != 0 && errno != ENOENT) {
            v->cks[i++].skip = true;
            continue;
        }
        freed += v->cks[i].bytes;
        v->cks[i] = v->cks[--v->ncks];
    }
    free(refs);
    v->bytes -= freed;
    return freed;
}

// Removes the view's state i (unless a load holds it: then it is skipped from now on), then the
// chunks only it used. Returns the bytes removed.
static uint64_t evict_entry(View *v, size_t i) {
    char f[1300];
    StEntry *e = &v->ents[i];
    if (!remove_state(pathf(f, sizeof f, "%s/states/%s.st", v->dir, e->name))) {
        e->skip = true;
        return 0;
    }
    uint64_t freed = e->bytes;
    v->bytes -= e->bytes;
    if (!e->parsed) v->unread--;
    entry_free(e);
    v->ents[i] = v->ents[--v->nents];
    return freed + sweep_view(v);
}

// The oldest removable state among views[0..nv) (only those of this pack's other shapes when
// same_pack_only): its view and index, or false.
static bool oldest(View *views, size_t nv, bool same_pack_only, size_t *vi, size_t *ei) {
    bool found = false;
    for (size_t k = 0; k < nv; k++) {
        if (same_pack_only && (views[k].own || !views[k].same_pack)) continue;
        for (size_t i = 0; i < views[k].nents; i++) {
            const StEntry *e = &views[k].ents[i];
            if (e->skip || (found && e->mtime >= views[*vi].ents[*ei].mtime)) continue;
            *vi = k;
            *ei = i;
            found = true;
        }
    }
    return found;
}

// The store tree's lock (root/.lock): writes, sweeps and evictions of every store on the machine
// run under it.
static void lock_store(StateStore *st) {
    if (st->lock_fd >= 0) flock(st->lock_fd, LOCK_EX);
}
static void unlock_store(StateStore *st) {
    if (st->lock_fd >= 0) flock(st->lock_fd, LOCK_UN);
}

// Removes the other stores' emptied folders (rmdir fails on a folder that still holds files).
static void prune_empty(StateStore *st) {
    DIR *r = opendir(st->root);
    struct dirent *pe;
    while (r && (pe = readdir(r))) {
        if (pe->d_name[0] == '.') continue;
        char pdir[1100];
        pathf(pdir, sizeof pdir, "%s/%s", st->root, pe->d_name);
        DIR *pd = opendir(pdir);
        struct dirent *se;
        while (pd && (se = readdir(pd))) {
            if (se->d_name[0] == '.') continue;
            char sdir[1150], t[1200];
            pathf(sdir, sizeof sdir, "%s/%s", pdir, se->d_name);
            if (!strcmp(sdir, st->dir)) continue;
            rmdir(pathf(t, sizeof t, "%s/states", sdir));
            rmdir(pathf(t, sizeof t, "%s/chunks", sdir));
            rmdir(sdir);
        }
        if (pd) closedir(pd);
        if (strcmp(pdir, st->pack_dir)) rmdir(pdir);
    }
    if (r) closedir(r);
}

// Scan and eviction over the whole tree (every pack's and shape's store), under its lock. Past
// the age limit states go, in every store; a store's chunks go when no state of it uses them
// (none while a state's keys are unknown). Over the cap (every store's bytes together): this
// pack's other shapes first (older engine builds or layouts, unreadable by this one: an engine id
// change makes them eligible at once), oldest state first, then the oldest of the other packs'
// and this store's states, each with the chunks only it used.
static void gc_locked(StateStore *st, uint64_t max_bytes, int64_t max_age_s, int64_t now, bool sweep_tmp) {
    uint8_t *page = xmalloc(ST_PAGE);
    View *views = xcalloc(1, sizeof *views);
    size_t nv = 1, vcap = 1;
    pathf(views[0].dir, sizeof views[0].dir, "%s", st->dir);
    hash_hex(st->shape, 8, views[0].shape_hex);
    views[0].own = views[0].same_pack = true;
    DIR *r = opendir(st->root);
    struct dirent *pe;
    while (r && (pe = readdir(r))) {
        if (pe->d_name[0] == '.') continue;
        char pdir[1100];
        pathf(pdir, sizeof pdir, "%s/%s", st->root, pe->d_name);
        DIR *pd = opendir(pdir);
        struct dirent *se;
        while (pd && (se = readdir(pd))) {
            if (se->d_name[0] == '.' || strlen(se->d_name) != 16) continue;
            View v = {.same_pack = !strcmp(pdir, st->pack_dir)};
            pathf(v.dir, sizeof v.dir, "%s/%s", pdir, se->d_name);
            if (!strcmp(v.dir, st->dir)) continue;
            memcpy(v.shape_hex, se->d_name, 17);
            if (nv == vcap) views = xrealloc(views, (vcap *= 2) * sizeof *views);
            views[nv++] = v;
        }
        if (pd) closedir(pd);
    }
    if (r) closedir(r);
    uint64_t total = 0;
    for (size_t k = 0; k < nv; k++) {
        scan_view(st, &views[k], max_age_s, now, sweep_tmp, page);
        sweep_view(&views[k]);
        total += views[k].bytes;
    }
    size_t all = 0;
    for (size_t k = 0; k < nv; k++) all += views[k].nents;
    size_t vi = 0, ei = 0;
    for (size_t pass = 0; total > max_bytes && pass <= all && oldest(views, nv, true, &vi, &ei); pass++)
        total -= evict_entry(&views[vi], ei);
    for (size_t pass = 0; total > max_bytes && pass <= all && oldest(views, nv, false, &vi, &ei); pass++)
        total -= evict_entry(&views[vi], ei);
    for (size_t k = 1; k < nv; k++) {
        for (size_t i = 0; i < views[k].nents; i++) entry_free(&views[k].ents[i]);
        free(views[k].ents);
        free(views[k].cks);
    }
    prune_empty(st);
    pthread_mutex_lock(&st->mu);
    for (size_t i = 0; i < st->nents; i++) entry_free(&st->ents[i]);
    free(st->ents);
    st->ents = views[0].ents;
    st->nents = views[0].nents;
    st->total = total;
    pthread_mutex_unlock(&st->mu);
    free(views[0].cks);
    free(views);
    free(page);
}

static void gc(StateStore *st, uint64_t max_bytes, int64_t max_age_s, int64_t now, bool sweep_tmp) {
    lock_store(st);
    gc_locked(st, max_bytes, max_age_s, now, sweep_tmp);
    unlock_store(st);
}

void state_store_gc(StateStore *st, uint64_t max_bytes, int64_t max_age_s, int64_t now) {
    state_store_wait(st);
    gc(st, max_bytes, max_age_s, now, false);
}

void state_store_usage(StateStore *st, uint64_t *bytes, size_t *states) {
    pthread_mutex_lock(&st->mu);
    *bytes = st->total;
    *states = st->nents;
    pthread_mutex_unlock(&st->mu);
}

// ---- writing -----------------------------------------------------------------------------------

// Creates a new temporary file beside path (O_EXCL, O_NOFOLLOW, 0600) for an uncached write.
static int open_tmp(const char *path, char *tmp, size_t cap) {
    for (int attempt = 0; attempt < 4; attempt++) {
        pathf(tmp, cap, "%s.tmp.%d.%08x", path, (int)getpid(), arc4random());
        int fd = open(tmp, O_CREAT | O_EXCL | O_NOFOLLOW | O_WRONLY | O_CLOEXEC, 0600);
        if (fd >= 0) {
            fcntl(fd, F_NOCACHE, 1);
            return fd;
        }
        if (errno != EEXIST) break;
    }
    return -1;
}

// Syncs, closes and renames a temporary file into place (removed on failure), then syncs the
// folder so the rename itself is on disk.
static Err commit_tmp(int fd, const char *tmp, const char *path, Err err) {
    if (!err && fcntl(fd, F_FULLFSYNC) != 0 && fsync(fd) != 0) err = "WriteFailed";
    if (close(fd) != 0 && !err) err = "WriteFailed";
    if (!err && rename(tmp, path) != 0) err = "WriteFailed";
    if (err) unlink(tmp);
    if (!err) {
        char dir[1300];
        snprintf(dir, sizeof dir, "%s", path);
        char *slash = strrchr(dir, '/');
        if (slash) *slash = 0;
        int dfd = open(dir, O_RDONLY | O_DIRECTORY);
        if (dfd >= 0) {
            if (fsync(dfd) != 0) err = "WriteFailed";
            close(dfd);
        }
    }
    return err;
}

static Err write_rowblock(int fd, const StLayout *lay, const RowBlock *rb) {
    for (size_t k = 0; k < rb->nslices; k++) {
        size_t len;
        const uint8_t *m = slice_mem(lay, k, rb->r0, rb->r1, &len);
        if (len) TRY(pwrite_all(fd, m, len, rb->s[k].off));
    }
    return OK;
}

static Err write_chunk(StateStore *st, size_t c, const uint8_t key[32], const char *path, uint64_t *written) {
    char tmp[1300];
    int fd = open_tmp(path, tmp, sizeof tmp);
    if (fd < 0) return "WriteFailed";
    uint8_t *page = xcalloc(1, ST_PAGE);
    ChunkHead *h = (ChunkHead *)page;
    memcpy(h->magic, "SAETTACK", 8);
    h->format = ST_FORMAT;
    memcpy(h->shape, st->shape, 32);
    memcpy(h->key, key, 32);
    uint64_t off = ST_PAGE;
    rowblock_plan(&st->lay, c * ST_CHUNK, (c + 1) * ST_CHUNK, &off, &h->rb);
    h->crc = page_crc(page);
    Err err = write_rowblock(fd, &st->lay, &h->rb);
    if (!err) err = pwrite_all(fd, page, ST_PAGE, 0);
    // the file's length covers its last slice even when that slice is empty
    if (!err && ftruncate(fd, (off_t)off) != 0) err = "WriteFailed";
    free(page);
    *written += off;
    return commit_tmp(fd, tmp, path, err);
}

static Err write_state(StateStore *st, const uint8_t *keys, size_t nchunks, const uint8_t name_hash[32], const char *path,
                       uint64_t *written) {
    char tmp[1300];
    int fd = open_tmp(path, tmp, sizeof tmp);
    if (fd < 0) return "WriteFailed";
    uint8_t *page = xcalloc(1, ST_PAGE);
    StateHead *h = (StateHead *)page;
    memcpy(h->magic, "SAETTAST", 8);
    h->format = ST_FORMAT;
    memcpy(h->shape, st->shape, 32);
    memcpy(h->name, name_hash, 32);
    ids_hash(st->ids, st->n, h->ids_hash);
    ids_hash(st->ids, ST_FIRST, h->first);
    memcpy(h->lineage, st->meta.lineage, 16);
    h->n = st->n;
    h->nchunks = nchunks;
    h->created_s = (int64_t)(wall_ns() / 1000000000ull);
    uint64_t off = ST_PAGE;
    h->ids = (StSlice){off, st->n * 4, {crc32c(0, st->ids, st->n * 4)}};
    off = align_page(off + h->ids.len);
    h->keys = (StSlice){off, nchunks * 32, {crc32c(0, keys, nchunks * 32)}};
    off = align_page(off + h->keys.len);
    rowblock_plan(&st->lay, nchunks * ST_CHUNK, st->n, &off, &h->tail);
    h->nrec = (uint32_t)st->lay.nrec;
    size_t ro = 0;
    for (size_t i = 0; i < st->lay.nrec; i++) {
        h->rec[i].off = off;
        h->rec[i].len = st->lay.rec_bytes[i];
        crc32c_quad(st->rec + ro, st->lay.rec_bytes[i], h->rec[i].crc);
        off = align_page(off + st->lay.rec_bytes[i]);
        ro += st->lay.rec_bytes[i];
    }
    h->prev[0] = st->meta.prev[0];
    h->prev[1] = st->meta.prev[1];
    h->since_eos = st->meta.since_eos;
    h->extra_len = st->meta.extra_len;
    memcpy(h->extra, st->meta.extra, st->meta.extra_len);
    h->crc = page_crc(page);
    Err err = pwrite_all(fd, st->ids, h->ids.len, h->ids.off);
    if (!err && nchunks) err = pwrite_all(fd, keys, h->keys.len, h->keys.off);
    if (!err) err = write_rowblock(fd, &st->lay, &h->tail);
    ro = 0;
    for (size_t i = 0; i < st->lay.nrec && !err; i++) {
        err = pwrite_all(fd, st->rec + ro, h->rec[i].len, h->rec[i].off);
        ro += st->lay.rec_bytes[i];
    }
    if (!err) err = pwrite_all(fd, page, ST_PAGE, 0);
    if (!err && ftruncate(fd, (off_t)off) != 0) err = "WriteFailed";
    free(page);
    *written += off;
    return commit_tmp(fd, tmp, path, err);
}

// Whether the chunk file at path is chunk c of this shape with this key (else it is rewritten).
static bool chunk_sound(StateStore *st, const char *path, const uint8_t key[32], size_t c) {
    int fd = open(path, O_RDONLY | O_NOFOLLOW);
    if (fd < 0) return false;
    uint8_t *page = xmalloc(ST_PAGE);
    ChunkHead h;
    bool ok = read_chunk_head(st, fd, key, c, &h, page);
    free(page);
    close(fd);
    return ok;
}

// The write, under the store's flock (with the eviction after it).
static void do_write(StateStore *st) {
    uint64_t t0 = now_ns();
    lock_store(st);
    size_t n = st->n, nchunks = n / ST_CHUNK;
    uint8_t *keys = xmalloc(nchunks * 32 + 32);
    // the keys: the inherited chunks' own (the state this conversation was restored from), then
    // sha256 of this conversation's lineage and the ids before each chunk's end, in one pass
    Sha256 h;
    sha256_init(&h);
    sha256_update(&h, st->meta.lineage, 16);
    for (size_t c = 0; c < nchunks; c++) {
        sha256_update(&h, st->ids + c * ST_CHUNK, ST_CHUNK * 4);
        if (c < st->ninh) memcpy(keys + c * 32, st->inh[c], 32);
        else {
            Sha256 t = h;
            sha256_final(&t, keys + c * 32);
        }
    }
    sha256_update(&h, st->ids + nchunks * ST_CHUNK, (n - nchunks * ST_CHUNK) * 4);
    uint8_t name_hash[32];
    sha256_final(&h, name_hash);
    uint64_t written = 0;
    size_t fresh = 0;
    Err err = OK;
    char p[1200], hx[33];
    for (size_t c = 0; c < nchunks && !err; c++) {
        hex16(keys + c * 32, hx);
        pathf(p, sizeof p, "%s/chunks/%s.kv", st->dir, hx);
        if (chunk_sound(st, p, keys + c * 32, c)) {
            utimes(p, NULL);
            continue;
        }
        err = write_chunk(st, c, keys + c * 32, p, &written);
        fresh++;
    }
    if (!err) {
        hex16(name_hash, hx);
        err = write_state(st, keys, nchunks, name_hash, pathf(p, sizeof p, "%s/states/%s.st", st->dir, hx), &written);
    }
    free(keys);
    double ms = (double)(now_ns() - t0) / 1e6;
    if (err) dbg("state: save failed (%s) at %zu tokens\n", err, n);
    else
        dbg("state: saved %zu tokens (%zu new of %zu chunks, %.2f GB written) in %.0f ms (%.2f GB/s)\n", n, fresh, nchunks,
            (double)written / 1e9, ms, (double)written / 1e6 / (ms > 0 ? ms : 1));
    gc_locked(st, ST_MAX_BYTES, ST_MAX_AGE_S, (int64_t)(wall_ns() / 1000000000ull), false);
    unlock_store(st);
}

static void *writer(void *arg) {
    StateStore *st = arg;
    pthread_mutex_lock(&st->mu);
    for (;;) {
        while (!st->job && !st->quit) pthread_cond_wait(&st->cv, &st->mu);
        if (!st->job) break;
        pthread_mutex_unlock(&st->mu);
        do_write(st);
        for (size_t i = 0; i < st->nhold; i++) st->release(st->hold[i]);
        pthread_mutex_lock(&st->mu);
        st->nhold = 0;
        st->job = false;
        pthread_cond_broadcast(&st->cv);
    }
    pthread_mutex_unlock(&st->mu);
    return NULL;
}

// mkdir -p, then the folder owner-only.
static bool private_dir(const char *path) { return !mkdir_p(path) && chmod(path, 0700) == 0; }

Err state_store_open(const char *root, const uint8_t ident[32], const uint8_t shape[32], StateStore **out) {
    StateStore *st = xcalloc(1, sizeof *st);
    char a[33], b[33], p[1200];
    hash_hex(ident, 8, a);
    hash_hex(shape, 8, b);
    pathf(st->dir, sizeof st->dir, "%s/%s/%s", root, a, b);
    pathf(st->root, sizeof st->root, "%s", root);
    pathf(st->pack_dir, sizeof st->pack_dir, "%s/%s", root, a);
    bool ok = private_dir(root) && private_dir(pathf(p, sizeof p, "%s/%s", root, a)) && private_dir(st->dir) &&
              private_dir(pathf(p, sizeof p, "%s/chunks", st->dir)) && private_dir(pathf(p, sizeof p, "%s/states", st->dir));
    if (!ok) {
        free(st);
        return "StateDirFailed";
    }
    st->lock_fd = open(pathf(p, sizeof p, "%s/.lock", root), O_CREAT | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (st->lock_fd < 0) dbg("state: cannot open %s (%s): the store works without its lock\n", p, strerror(errno));
    memcpy(st->shape, shape, 32);
    pthread_mutex_init(&st->mu, NULL);
    pthread_cond_init(&st->cv, NULL);
    gc(st, ST_MAX_BYTES, ST_MAX_AGE_S, (int64_t)(wall_ns() / 1000000000ull), true);
    if (pthread_create(&st->thr, NULL, writer, st) != 0) {
        if (st->lock_fd >= 0) close(st->lock_fd);
        for (size_t i = 0; i < st->nents; i++) entry_free(&st->ents[i]);
        free(st->ents);
        pthread_mutex_destroy(&st->mu);
        pthread_cond_destroy(&st->cv);
        free(st);
        return "StateWriterFailed";
    }
    *out = st;
    return OK;
}

void state_store_wait(StateStore *st) {
    pthread_mutex_lock(&st->mu);
    while (st->job) pthread_cond_wait(&st->cv, &st->mu);
    pthread_mutex_unlock(&st->mu);
}

void state_store_close(StateStore *st) {
    pthread_mutex_lock(&st->mu);
    st->quit = true;
    pthread_cond_broadcast(&st->cv);
    pthread_mutex_unlock(&st->mu);
    pthread_join(st->thr, NULL);
    if (st->lock_fd >= 0) close(st->lock_fd);
    free(st->ids);
    free(st->inh);
    free(st->rec);
    for (size_t i = 0; i < st->nents; i++) entry_free(&st->ents[i]);
    free(st->ents);
    pthread_mutex_destroy(&st->mu);
    pthread_cond_destroy(&st->cv);
    free(st);
}

void state_store_fence(StateStore *st, const void *base, size_t from) {
    pthread_mutex_lock(&st->mu);
    while (st->job && from < st->n) {
        bool uses = false;
        for (size_t i = 0; i < st->lay.nreg; i++) uses |= st->lay.reg[i].base == base;
        if (!uses) break;
        pthread_cond_wait(&st->cv, &st->mu);
    }
    pthread_mutex_unlock(&st->mu);
}

static uint64_t state_bytes(const StLayout *lay, size_t n) {
    uint64_t b = ST_PAGE + n * 4;
    for (size_t i = 0; i < lay->nreg; i++) b += (uint64_t)lay->reg[i].planes * (n / lay->reg[i].per_row) * lay->reg[i].row_bytes;
    for (size_t i = 0; i < lay->nrec; i++) b += lay->rec_bytes[i];
    return b;
}

// Whether the state file at path is this state, sound (header, shape, ids hash, lineage).
static bool state_sound(StateStore *st, const char *path, const uint32_t *ids, size_t n, const uint8_t lineage[16]) {
    int fd = open(path, O_RDONLY | O_NOFOLLOW);
    if (fd < 0) return false;
    uint8_t *page = xmalloc(ST_PAGE);
    StateHead h;
    uint8_t ih[32];
    ids_hash(ids, n, ih);
    bool ok = read_state_head(st, fd, &h, page) && h.n == n && !memcmp(h.ids_hash, ih, 32) && !memcmp(h.lineage, lineage, 16);
    free(page);
    close(fd);
    return ok;
}

bool state_store_save(StateStore *st, const StLayout *lay, const uint32_t *ids, size_t n, const StMeta *meta,
                      const uint8_t (*inh)[32], size_t ninh, void *const *hold, size_t nhold, void (*release)(void *)) {
    if (ninh > n / ST_CHUNK) ninh = n / ST_CHUNK;
    const char *why = NULL;
    if (n < ST_MIN_TOKENS) why = "under the minimum";
    else if (layout_slices(lay) > ST_MAX_SLICES || lay->nrec > ST_MAX_REC || nhold > ST_MAX_REGIONS ||
             meta->extra_len > ST_EXTRA_MAX)
        why = "layout too large";
    state_store_wait(st);
    uint8_t h[32];
    char hx[33], p[1200];
    if (!why) {
        lineage_hash(meta->lineage, ids, n, h);
        hex16(h, hx);
        if (state_sound(st, pathf(p, sizeof p, "%s/states/%s.st", st->dir, hx), ids, n, meta->lineage)) {
            utimes(p, NULL);
            why = "already saved";
        }
    }
    if (!why) {
        struct statfs fs;
        if (statfs(st->dir, &fs) == 0 && (uint64_t)fs.f_bavail * fs.f_bsize < state_bytes(lay, n) + ST_DISK_FLOOR) why = "disk short";
        else if (state_bytes(lay, n) > ST_MAX_BYTES) why = "larger than the store";
    }
    if (why) {
        if (strcmp(why, "already saved")) dbg("state: skipped %zu tokens (%s)\n", n, why);
        for (size_t i = 0; i < nhold; i++) release(hold[i]);
        return false;
    }
    if (st->ids_cap < n) st->ids = xrealloc(st->ids, (st->ids_cap = n + n / 2) * sizeof(uint32_t));
    memcpy(st->ids, ids, n * sizeof(uint32_t));
    size_t rb = 0;
    for (size_t i = 0; i < lay->nrec; i++) rb += lay->rec_bytes[i];
    if (st->rec_cap < rb) {
        free(st->rec);
        st->rec = xmalloc(rb);
        st->rec_cap = rb;
    }
    size_t ro = 0;
    for (size_t i = 0; i < lay->nrec; i++) {
        memcpy(st->rec + ro, lay->rec[i], lay->rec_bytes[i]);
        ro += lay->rec_bytes[i];
    }
    if (st->inh_cap < ninh) st->inh = xrealloc(st->inh, (st->inh_cap = ninh) * 32);
    if (ninh) memcpy(st->inh, inh, ninh * 32);
    pthread_mutex_lock(&st->mu);
    st->lay = *lay;
    st->n = n;
    st->ninh = ninh;
    st->meta = *meta;
    memcpy(st->hold, hold, nhold * sizeof *hold);
    st->nhold = nhold;
    st->release = release;
    st->job = true;
    pthread_cond_broadcast(&st->cv);
    pthread_mutex_unlock(&st->mu);
    return true;
}

// ---- finding and loading -----------------------------------------------------------------------

static int by_n_desc(const void *a, const void *b) {
    const StEntry *x = a, *y = b;
    return x->n < y->n ? 1 : x->n > y->n ? -1 : 0;
}

// Whether the ids section of the state open at fd (header h checked) is prompt[0..h.n).
static bool ids_match(int fd, const StateHead *h, const uint32_t *prompt, size_t np) {
    if (h->n >= np) return false;
    uint32_t *ids = xmalloc(h->ids.len + 4);
    bool ok = !pread_all(fd, ids, h->ids.len, h->ids.off) && crc32c(0, ids, h->ids.len) == h->ids.crc[0] &&
              !memcmp(ids, prompt, h->ids.len);
    free(ids);
    return ok;
}

size_t state_store_find(StateStore *st, const uint32_t *prompt, size_t np, size_t min_n, char name[33]) {
    if (np <= ST_FIRST) return 0;
    uint8_t first[32];
    ids_hash(prompt, ST_FIRST, first);
    pthread_mutex_lock(&st->mu);
    StEntry *c = xmalloc((st->nents + 1) * sizeof *c);
    size_t nc = 0;
    for (size_t i = 0; i < st->nents; i++) {
        const StEntry *e = &st->ents[i];
        if (e->n > min_n && e->n < np && !memcmp(e->first, first, 32)) c[nc++] = *e;
    }
    pthread_mutex_unlock(&st->mu);
    qsort(c, nc, sizeof *c, by_n_desc);
    size_t found = 0;
    uint8_t *page = xmalloc(ST_PAGE);
    for (size_t i = 0; i < nc && !found; i++) {
        char p[1200];
        int fd = open(pathf(p, sizeof p, "%s/states/%s.st", st->dir, c[i].name), O_RDONLY | O_NOFOLLOW);
        if (fd < 0) continue;
        StateHead h;
        if (read_state_head(st, fd, &h, page) && h.n == c[i].n && ids_match(fd, &h, prompt, np)) {
            found = h.n;
            memcpy(name, c[i].name, 33);
        }
        close(fd);
    }
    free(page);
    free(c);
    return found;
}

typedef struct {
    int fd;
    uint64_t off, len;
    uint8_t *dst;
    uint32_t crc[4];
    int file; // index in the load's files (0: the state)
} ReadItem;

typedef struct {
    ReadItem *items;
    size_t n;
    atomic_size_t next;
    atomic_int bad; // 1 + the file of the first failure
} ReadJob;

static void *read_worker(void *arg) {
    ReadJob *j = arg;
    for (;;) {
        size_t i = atomic_fetch_add(&j->next, 1);
        if (i >= j->n || atomic_load(&j->bad)) break;
        ReadItem *it = &j->items[i];
        uint32_t c[4];
        if (pread_all(it->fd, it->dst, it->len, it->off)) {
            atomic_store(&j->bad, it->file + 1);
            break;
        }
        crc32c_quad(it->dst, it->len, c);
        if (memcmp(c, it->crc, sizeof c)) {
            atomic_store(&j->bad, it->file + 1);
            break;
        }
    }
    return NULL;
}

// Adds a row block's slices (checked against lay's shapes) to the reads.
static bool add_rowblock(const StLayout *lay, const RowBlock *rb, int fd, int file, ReadItem *items, size_t *ni) {
    if (rb->nslices != layout_slices(lay)) return false;
    for (size_t k = 0; k < rb->nslices; k++) {
        size_t len;
        uint8_t *m = slice_mem(lay, k, rb->r0, rb->r1, &len);
        if (rb->s[k].len != len) return false;
        if (!len) continue;
        items[*ni] = (ReadItem){fd, rb->s[k].off, len, m, {rb->s[k].crc[0], rb->s[k].crc[1], rb->s[k].crc[2], rb->s[k].crc[3]}, file};
        *ni += 1;
    }
    return true;
}

Err state_store_load(StateStore *st, const char *name, const uint32_t *prompt, size_t np, const StLayout *lay, size_t *n_out,
                     StMeta *meta, uint8_t (**keys_out)[32], size_t *nkeys, uint64_t *bytes, bool *touched) {
    char p[1200];
    *touched = false;
    *keys_out = NULL;
    *nkeys = 0;
    pathf(p, sizeof p, "%s/states/%s.st", st->dir, name);
    int sfd = open(p, O_RDONLY | O_NOFOLLOW);
    if (sfd < 0) return "StateMissing";
    // held while it reads: an eviction in any process skips this state (remove_state), and so its
    // chunks; a state removed between the open and the lock is no longer at its path
    struct stat a, b;
    if (flock(sfd, LOCK_SH) != 0 || fstat(sfd, &a) != 0 || stat(p, &b) != 0 || a.st_ino != b.st_ino || a.st_dev != b.st_dev) {
        close(sfd);
        return "StateMissing";
    }
    fcntl(sfd, F_NOCACHE, 1);
    uint8_t *page = xmalloc(ST_PAGE);
    StateHead h;
    Err err = OK;
    uint8_t *keys = NULL;
    int *fds = NULL;
    char (*names)[33] = NULL;
    ReadItem *items = NULL;
    size_t nfiles = 0;
    int bad_file = -1; // a chunk that failed a check (deleted with the state)
    if (!read_state_head(st, sfd, &h, page)) err = "StateCorrupt";
    else if (h.nrec != lay->nrec) err = "StateMismatch";
    else if (!ids_match(sfd, &h, prompt, np)) err = "StateMismatch";
    for (size_t i = 0; !err && i < lay->nreg; i++)
        if (h.n / lay->reg[i].per_row > lay->reg[i].plane_stride / lay->reg[i].row_bytes) err = "StateTooLarge";
    for (size_t i = 0; !err && i < lay->nrec; i++)
        if (h.rec[i].len != lay->rec_bytes[i]) err = "StateMismatch";
    if (!err) {
        keys = xmalloc(h.keys.len + 1);
        if (pread_all(sfd, keys, h.keys.len, h.keys.off) || crc32c(0, keys, h.keys.len) != h.keys.crc[0]) err = "StateCorrupt";
    }
    size_t per = layout_slices(lay), ni = 0;
    if (!err) {
        items = xmalloc(((h.nchunks + 1) * per + ST_MAX_REC) * sizeof *items);
        fds = xmalloc((h.nchunks + 1) * sizeof *fds);
        names = xmalloc((h.nchunks + 1) * sizeof *names);
        fds[nfiles] = sfd;
        memcpy(names[nfiles], name, 33);
        nfiles++;
    }
    for (size_t c = 0; !err && c < h.nchunks; c++) {
        char hx[33];
        hex16(keys + c * 32, hx);
        int fd = open(pathf(p, sizeof p, "%s/chunks/%s.kv", st->dir, hx), O_RDONLY | O_NOFOLLOW);
        if (fd < 0) {
            err = "StateMissing";
            break;
        }
        fcntl(fd, F_NOCACHE, 1);
        fds[nfiles] = fd;
        memcpy(names[nfiles], hx, 33);
        int file = (int)nfiles++;
        ChunkHead ch;
        if (!read_chunk_head(st, fd, keys + c * 32, c, &ch, page) || !add_rowblock(lay, &ch.rb, fd, file, items, &ni)) {
            err = "StateCorrupt";
            bad_file = file;
        }
    }
    if (!err && !add_rowblock(lay, &h.tail, sfd, 0, items, &ni)) err = "StateCorrupt";
    for (size_t i = 0; !err && i < lay->nrec; i++)
        items[ni++] = (ReadItem){sfd, h.rec[i].off, h.rec[i].len, lay->rec[i],
                                 {h.rec[i].crc[0], h.rec[i].crc[1], h.rec[i].crc[2], h.rec[i].crc[3]}, 0};
    if (!err) {
        *touched = true;
        ReadJob j = {.items = items, .n = ni};
        atomic_init(&j.next, 0);
        atomic_init(&j.bad, 0);
        pthread_t t[ST_THREADS];
        size_t nt = 0;
        for (; nt < ST_THREADS; nt++)
            if (pthread_create(&t[nt], NULL, read_worker, &j) != 0) break;
        if (nt == 0) read_worker(&j);
        for (size_t i = 0; i < nt; i++) pthread_join(t[i], NULL);
        int b = atomic_load(&j.bad);
        if (b) {
            err = "StateCorrupt";
            bad_file = b - 1;
        }
        uint64_t tot = 0;
        for (size_t i = 0; i < ni; i++) tot += items[i].len;
        *bytes = tot;
    }
    for (size_t i = 1; i < nfiles; i++) {
        if (!err) futimes(fds[i], NULL); // a chunk read is a chunk in use
        close(fds[i]);
    }
    if (err && strcmp(err, "StateTooLarge")) {
        // a state that fails a check is dropped, and a chunk that failed its header or crc
        unlink(pathf(p, sizeof p, "%s/states/%s.st", st->dir, name));
        if (bad_file > 0) unlink(pathf(p, sizeof p, "%s/chunks/%s.kv", st->dir, names[bad_file]));
        pthread_mutex_lock(&st->mu);
        for (size_t i = 0; i < st->nents; i++)
            if (!strcmp(st->ents[i].name, name)) {
                entry_free(&st->ents[i]);
                st->ents[i] = st->ents[--st->nents];
                break;
            }
        pthread_mutex_unlock(&st->mu);
    }
    if (!err) {
        *n_out = h.n;
        memset(meta, 0, sizeof *meta);
        memcpy(meta->lineage, h.lineage, 16);
        meta->prev[0] = h.prev[0];
        meta->prev[1] = h.prev[1];
        meta->since_eos = h.since_eos;
        meta->extra_len = h.extra_len;
        memcpy(meta->extra, h.extra, h.extra_len);
        futimes(sfd, NULL);
        *keys_out = (uint8_t (*)[32])keys;
        *nkeys = h.nchunks;
        keys = NULL;
    }
    close(sfd); // and its lock
    free(items);
    free(fds);
    free(names);
    free(keys);
    free(page);
    return err;
}
