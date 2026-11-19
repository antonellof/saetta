// `saetta download` (see download.h).
#include "download.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "hash.h"
#include "json.h"
#include "metal/shim.h"
#include "stats.h"

const DownloadPack download_packs[DOWNLOAD_NPACKS] = {
    {.name = "coder", .repo = "mbrown-uno/Qwen3.8-Flash-Next-GSQ-RCO-Coder-MLX", .rev = "50fa68c5e9cebf53276d8eae8af594fa8b3de9e4", .note = "256 experts, pruned for code, 76 GB"},
    {.name = "full", .repo = "mlx-community/Qwen3.8-Flash-Next-4bit-mtp", .rev = "825df623302aeebc434f44ca9122827b04f9a0f1", .note = "512 experts, the original model, 105 GB"},
};

// Idle timeout of a transfer (between bytes); a stalled connection fails and is retried.
#define IDLE_TIMEOUT_S 300.0

const DownloadPack *download_find(const char *name) {
    for (size_t i = 0; i < DOWNLOAD_NPACKS; i++)
        if (strcmp(download_packs[i].name, name) == 0) return &download_packs[i];
    return NULL;
}

Err download_models_dir(Arena *a, char **out) {
    const char *m = getenv("SAETTA_MODELS");
    if (m && m[0]) {
        *out = arena_strdup(a, m);
        return OK;
    }
    const char *home = getenv("HOME");
    if (!home) return "NoHome";
    *out = path_join(a, home, "Models");
    return OK;
}

Err download_pack_dir(Arena *a, const DownloadPack *p, char **out) {
    char *d;
    TRY(download_models_dir(a, &d));
    *out = path_join(a, d, p->repo);
    return OK;
}

const char *download_usage(void) {
    return "usage: saetta download [coder|full|all] [--jobs N]   (default coder; into $SAETTA_MODELS, default ~/Models)\n";
}

typedef struct {
    const char *path; // relative to the repo
    uint64_t size;
    const char *sha256; // LFS files, else NULL
    const char *git_oid; // git blob SHA-1 otherwise
} Item;

typedef struct {
    const DownloadPack *pack;
    const char *root;
    const char *token;
    const Item *items;
    size_t n_items;
    _Atomic size_t next;
    _Atomic uint64_t done_bytes;
    _Atomic uint32_t failed;
    _Atomic uint32_t files_done;
} Shared;

// An HTTP transport error message, returned as the Err (per thread, valid until the next).
static _Thread_local char http_err[256];

static Err http_error(const char *msg) {
    snprintf(http_err, sizeof http_err, "%s", msg[0] ? msg : "HttpFailed");
    return http_err;
}

static const char *hf_token(Arena *a) {
    const char *t = getenv("HF_TOKEN");
    if (t && t[0]) return t;
    const char *home = getenv("HOME");
    if (!home) return NULL;
    char *path = path_join(a, path_join(a, path_join(a, home, ".cache"), "huggingface"), "token");
    char *text;
    size_t len;
    if (read_file(path, 4096, &text, &len)) return NULL;
    Str s = str_trim(str_n(text, len));
    char *r = s.len > 0 ? str_z(a, s) : NULL;
    free(text);
    return r;
}

static double gb(uint64_t b) { return (double)b / 1e9; }

static void fixed(char *out, double x, int prec) { fmt_fixed(out, 64, x, prec); }

static void progress(Shared *sh, uint64_t need, uint64_t t0, bool final) {
    uint64_t done = atomic_load_explicit(&sh->done_bytes, memory_order_acquire);
    double s = (double)(now_ns() - t0) / 1e9;
    double rate = s > 0 ? (double)done / s : 0;
    double left = rate > 0 ? (double)(need > done ? need - done : 0) / rate : 0;
    char a[64], b[64], c[64];
    fixed(a, gb(done), 1);
    fixed(b, gb(need), 1);
    fixed(c, rate / 1e6, 0);
    dbg("\r  %s / %s GB  %s MB/s  %u/%zu files  ETA %llum%02llus   %s", a, b, c,
        atomic_load_explicit(&sh->files_done, memory_order_acquire), sh->n_items,
        (unsigned long long)(left / 60), (unsigned long long)fmod(left, 60), final ? "\n" : "");
}

static int sink_buf(void *ctx, int status, const void *p, size_t n) {
    buf_add(ctx, p, n);
    return 0;
}

// The revision's files, from https://huggingface.co/api/models/<repo>/tree/<rev>?recursive=true
static Err list_files(Arena *a, const DownloadPack *p, const char *token, Item **out, size_t *n_out) {
    char *url = arena_printf(a, "https://huggingface.co/api/models/%s/tree/%s?recursive=true", p->repo, p->rev);
    sx_header h[3] = {{"accept-encoding", "identity"}, {"user-agent", "saetta"}, {0}};
    size_t nh = 2;
    if (token) h[nh++] = (sx_header){"authorization", arena_printf(a, "Bearer %s", token)};
    Buf body = {0};
    int status = 0;
    char err[256] = "";
    if (sx_http("GET", url, h, nh, NULL, 0, IDLE_TIMEOUT_S, sink_buf, &body, &status, err, sizeof err) != 0) {
        buf_free(&body);
        return http_error(err);
    }
    if (status != 200) {
        dbg("saetta download: listing %s failed: HTTP %d\n", p->repo, status);
        buf_free(&body);
        return "ListFailed";
    }
    Str text = str_n(arena_dup(a, body.ptr ? body.ptr : "", body.len), body.len);
    buf_free(&body);
    JValue *v;
    TRY(json_parse(a, text, true, &v));
    if (!json_is(v, J_ARRAY)) return "BadListing";
    VEC(Item) items = {0};
    for (size_t i = 0; i < v->arr.len; i++) {
        JValue *e = v->arr.items[i];
        if (!json_is(e, J_OBJECT)) continue;
        JValue *ty = json_get(e, "type");
        if (!ty) continue;
        if (!json_is(ty, J_STRING) || !str_eqz(ty->s, "file")) continue;
        JValue *path = json_get(e, "path"), *size = json_get(e, "size"), *oid = json_get(e, "oid");
        if (!json_is(path, J_STRING) || !json_is(size, J_INTEGER) || !json_is(oid, J_STRING)) continue;
        const char *sha = NULL;
        JValue *l = json_get(e, "lfs");
        if (json_is(l, J_OBJECT)) {
            JValue *o = json_get(l, "oid");
            if (json_is(o, J_STRING)) sha = o->s.ptr;
        }
        // a path must stay inside the pack folder
        if (str_find(path->s, S(".."), 0) >= 0 || (path->s.len > 0 && path->s.ptr[0] == '/')) {
            vec_free(&items);
            return "BadListing";
        }
        vec_push(&items, ((Item){.path = path->s.ptr, .size = (uint64_t)size->i, .sha256 = sha, .git_oid = oid->s.ptr}));
    }
    if (items.len == 0) {
        vec_free(&items);
        return "BadListing";
    }
    *out = arena_dup(a, items.ptr, items.len * sizeof(Item));
    *n_out = items.len;
    vec_free(&items);
    return OK;
}

typedef struct {
    bool is_sha256;
    Sha256 s256;
    Sha1 s1;
} Hasher;

static void hasher_init(Hasher *h, const Item *it) {
    h->is_sha256 = it->sha256 != NULL;
    if (h->is_sha256) {
        sha256_init(&h->s256);
    } else {
        sha1_init(&h->s1);
        char hdr[64];
        int n = snprintf(hdr, sizeof hdr, "blob %llu", (unsigned long long)it->size);
        sha1_update(&h->s1, hdr, (size_t)n + 1); // with the NUL
    }
}

static void hasher_update(Hasher *h, const void *p, size_t n) {
    if (h->is_sha256) sha256_update(&h->s256, p, n);
    else sha1_update(&h->s1, p, n);
}

static Err pwrite_all(int fd, const void *p, size_t n, off_t off) {
    const char *c = p;
    while (n) {
        ssize_t w = pwrite(fd, c, n, off);
        if (w < 0) {
            if (errno == EINTR) continue;
            return err_errno(errno);
        }
        c += w;
        n -= (size_t)w;
        off += w;
    }
    return OK;
}

// Body sink of one file transfer.
typedef struct {
    Shared *sh;
    const Item *it;
    Hasher *h;
    int fd;
    uint64_t off;
    uint64_t *counted;
    bool started;
    bool full; // the expected size is in: the rest of the body is not read (as Zig)
    Err err;
} Fetch;

// The status check done once the response head is in.
static Err on_status(Fetch *f, int status) {
    if (status == 206) {
        if (f->off == 0) return "UnexpectedPartial";
    } else if (status == 200) {
        if (f->off > 0) {
            // the server ignored the range: start over
            atomic_fetch_sub_explicit(&f->sh->done_bytes, *f->counted, memory_order_acq_rel);
            *f->counted = 0;
            f->off = 0;
            hasher_init(f->h, f->it);
            if (ftruncate(f->fd, 0) != 0) return err_errno(errno);
        }
    } else return "HttpStatus";
    return OK;
}

static int sink_file(void *ctx, int status, const void *p, size_t n) {
    Fetch *f = ctx;
    if (!f->started) {
        f->started = true;
        f->err = on_status(f, status);
        if (f->err) return 1;
    }
    if (n == 0) return 0;
    // Zig read while off < size: bytes past the expected size are never written or hashed
    if (n > f->it->size - f->off) n = (size_t)(f->it->size - f->off);
    f->err = pwrite_all(f->fd, p, n, (off_t)f->off);
    if (f->err) return 1;
    hasher_update(f->h, p, n);
    f->off += n;
    *f->counted += n;
    atomic_fetch_add_explicit(&f->sh->done_bytes, n, memory_order_acq_rel);
    if (f->off == f->it->size) {
        f->full = true;
        return 1; // stop the transfer
    }
    return 0;
}

static Err fetch_body(Shared *sh, const Item *it, int fd, Hasher *h, uint64_t *off, uint64_t *counted) {
    char url[1024], range[64], auth[256];
    if ((size_t)snprintf(url, sizeof url, "https://huggingface.co/%s/resolve/%s/%s", sh->pack->repo, sh->pack->rev, it->path) >= sizeof url)
        return "NoSpaceLeft";
    snprintf(range, sizeof range, "bytes=%llu-", (unsigned long long)*off);
    sx_header hd[4] = {{"accept-encoding", "identity"}, {"user-agent", "saetta"}, {0}, {0}};
    size_t nh = 2;
    if (*off > 0) hd[nh++] = (sx_header){"range", range};
    if (sh->token) {
        if ((size_t)snprintf(auth, sizeof auth, "Bearer %s", sh->token) >= sizeof auth) return "NoSpaceLeft";
        hd[nh++] = (sx_header){"authorization", auth};
    }
    Fetch f = {.sh = sh, .it = it, .h = h, .fd = fd, .off = *off, .counted = counted};
    int status = 0;
    char err[256] = "";
    int rc = sx_http("GET", url, hd, nh, NULL, 0, IDLE_TIMEOUT_S, sink_file, &f, &status, err, sizeof err);
    *off = f.off;
    if (f.err) return f.err;
    if (rc != 0 && !f.full) return f.started ? "ReadFailed" : http_error(err);
    if (!f.started) TRY(on_status(&f, status)); // an empty body
    *off = f.off;
    if (*off != it->size) return "ShortBody";
    return OK;
}

static Err fetch_one(Shared *sh, const Item *it) {
    char dest[512], part[512];
    if ((size_t)snprintf(dest, sizeof dest, "%s/%s", sh->root, it->path) >= sizeof dest) return "NoSpaceLeft";
    if ((size_t)snprintf(part, sizeof part, "%s.part", dest) >= sizeof part) return "NoSpaceLeft";
    char *slash = strrchr(dest, '/');
    if (slash && slash != dest) {
        *slash = 0;
        Err e = mkdir_p(dest);
        *slash = '/';
        TRY(e);
    }
    int fd = open(part, O_RDWR | O_CREAT | O_CLOEXEC, 0666);
    if (fd < 0) return err_errno(errno);
    Err e = OK;
    uint8_t *chunk = NULL;
    uint64_t counted = 0; // bytes of this file added to done_bytes
    struct stat st;
    if (fstat(fd, &st) != 0) {
        e = err_errno(errno);
        goto out;
    }
    uint64_t off = (uint64_t)st.st_size;
    if (off > it->size) {
        // a longer .part than the file: start over from an empty one
        off = 0;
        if (ftruncate(fd, 0) != 0) {
            e = err_errno(errno);
            goto out;
        }
    }

    Hasher h;
    hasher_init(&h, it);
    const size_t chunk_len = 4 << 20;
    chunk = xmalloc(chunk_len);
    // rehash what a previous run already wrote
    uint64_t pos = 0;
    while (pos < off) {
        size_t want = (size_t)MIN((uint64_t)chunk_len, off - pos);
        size_t got = 0;
        while (got < want) {
            ssize_t r = pread(fd, chunk + got, want - got, (off_t)(pos + got));
            if (r < 0 && errno == EINTR) continue;
            if (r < 0) {
                e = err_errno(errno);
                goto out;
            }
            if (r == 0) break;
            got += (size_t)r;
        }
        if (got == 0) break;
        hasher_update(&h, chunk, got);
        pos += got;
    }
    off = pos;
    atomic_fetch_add_explicit(&sh->done_bytes, off, memory_order_acq_rel);
    counted = off;

    if (off < it->size) {
        e = fetch_body(sh, it, fd, &h, &off, &counted);
        if (e) goto fail;
    }

    // verify, then move into place
    char got_hex[2 * SHA256_LEN + 1];
    if (h.is_sha256) {
        uint8_t d[SHA256_LEN];
        sha256_final(&h.s256, d);
        hash_hex(d, sizeof d, got_hex);
    } else {
        uint8_t d[SHA1_LEN];
        sha1_final(&h.s1, d);
        hash_hex(d, sizeof d, got_hex);
    }
    const char *want_hex = it->sha256 ? it->sha256 : it->git_oid;
    if (strlen(got_hex) != strlen(want_hex) || strcasecmp(got_hex, want_hex) != 0) {
        unlink(part);
        e = "HashMismatch";
        goto fail;
    }
    if (rename(part, dest) != 0) {
        e = err_errno(errno);
        goto fail;
    }
    goto out;
fail:
    atomic_fetch_sub_explicit(&sh->done_bytes, counted, memory_order_acq_rel);
out:
    free(chunk);
    close(fd);
    return e;
}

static void *worker(void *p) {
    Shared *sh = p;
    for (;;) {
        size_t k = atomic_fetch_add_explicit(&sh->next, 1, memory_order_acq_rel);
        if (k >= sh->n_items) return NULL;
        const Item *it = &sh->items[k];
        for (uint32_t attempt = 0;; attempt++) {
            Err e = fetch_one(sh, it);
            if (!e) {
                atomic_fetch_add_explicit(&sh->files_done, 1, memory_order_acq_rel);
                break;
            }
            if (strcmp(e, "HashMismatch") == 0 || attempt >= 5) {
                dbg("\n  %s: %s\n", it->path, e);
                atomic_fetch_add_explicit(&sh->failed, 1, memory_order_acq_rel);
                break;
            }
            sleep(2u << attempt);
        }
    }
}

// std.mem.sort (stable): largest first.
static void sort_by_size_desc(Item *v, size_t n) {
    for (size_t i = 1; i < n; i++) {
        Item x = v[i];
        size_t j = i;
        while (j > 0 && x.size > v[j - 1].size) {
            v[j] = v[j - 1];
            j--;
        }
        v[j] = x;
    }
}

static Err one(Arena *a, const DownloadPack *p, const char *token, size_t jobs) {
    char *root;
    TRY(download_pack_dir(a, p, &root));
    dbg("saetta download: %s (%s)\n  %s at %.7s\n  into %s\n", p->name, p->note, p->repo, p->rev, root);
    Item *all;
    size_t n_all;
    TRY(list_files(a, p, token, &all, &n_all));

    // keep files already present at their full size
    Item *todo = ARENA_ARRAY(a, Item, n_all);
    size_t n_todo = 0;
    uint64_t total = 0, have = 0;
    for (size_t i = 0; i < n_all; i++) {
        const Item *it = &all[i];
        total += it->size;
        char *dest = path_join(a, root, it->path);
        struct stat st;
        if (stat(dest, &st) != 0) {
            todo[n_todo++] = *it;
            continue;
        }
        if ((uint64_t)st.st_size == it->size) have += it->size;
        else todo[n_todo++] = *it;
    }
    char x[64], y[64];
    if (n_todo == 0) {
        fixed(x, gb(total), 1);
        dbg("  all %zu files present (%s GB)\n", n_all, x);
        return OK;
    }
    // largest first, so the long transfers overlap
    sort_by_size_desc(todo, n_todo);
    uint64_t need = 0;
    for (size_t i = 0; i < n_todo; i++) need += todo[i].size;
    fixed(x, gb(need), 1);
    fixed(y, gb(have), 1);
    dbg("  %zu of %zu files to fetch, %s GB (%s GB present)\n", n_todo, n_all, x, y);

    Shared sh = {.pack = p, .root = root, .token = token, .items = todo, .n_items = n_todo};
    size_t n = MIN(jobs, n_todo);
    pthread_t *threads = ARENA_ARRAY(a, pthread_t, n);
    for (size_t i = 0; i < n; i++) {
        int rc = pthread_create(&threads[i], NULL, worker, &sh);
        if (rc != 0) die("saetta download: cannot start a thread: %s\n", err_errno(rc));
    }

    uint64_t t0 = now_ns();
    while (atomic_load_explicit(&sh.files_done, memory_order_acquire) + atomic_load_explicit(&sh.failed, memory_order_acquire) < n_todo) {
        sleep(1);
        progress(&sh, need, t0, false);
    }
    for (size_t i = 0; i < n; i++) pthread_join(threads[i], NULL);
    progress(&sh, need, t0, true);
    uint32_t failed = atomic_load_explicit(&sh.failed, memory_order_acquire);
    if (failed > 0) {
        dbg("saetta download: %u file(s) failed; run the same command again to resume\n", failed);
        return "DownloadFailed";
    }
    dbg("  done: %s\n", root);
    return OK;
}

static Err run_in(Arena *a, const char *const *args, size_t n) {
    const char *which = "coder";
    size_t jobs = 4;
    for (size_t i = 0; i < n; i++) {
        if (strcmp(args[i], "--jobs") == 0 && i + 1 < n) {
            i += 1;
            uint64_t j;
            TRY(parse_u64(str(args[i]), &j));
            jobs = (size_t)CLAMP(j, 1, 16);
        } else which = args[i];
    }
    const char *token = hf_token(a);
    if (strcmp(which, "all") == 0) {
        for (size_t i = 0; i < DOWNLOAD_NPACKS; i++) TRY(one(a, &download_packs[i], token, jobs));
        return OK;
    }
    const DownloadPack *p = download_find(which);
    if (!p) {
        dbg("saetta download: unknown pack '%s'\n%s", which, download_usage());
        return "UnknownPack";
    }
    return one(a, p, token, jobs);
}

Err download_run(const char *const *args, size_t n) {
    Arena a = {0};
    Err e = run_in(&a, args, n);
    arena_reset(&a);
    return e;
}
