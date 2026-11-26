// `saetta download` (see download.h).
#include "download.h"
#include "fit.h"

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

#include "gguf.h"
#include "hash.h"
#include "json.h"
#include "metal/shim.h"
#include "stats.h"

// DavidAU's TURBO fine-tune of Qwen3.8-27B: config and tokenizer from the BF16 repo, weights
// from the GGUF repo (its NEO-CODER imatrix quants with the MTP layer), both Apache-2.0.
#define TURBO_REPO "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU"
#define TURBO_REV "079b2b268c637b77f2fb4ab6a5ca1edc47d462a5"
#define TURBO_GGUF_REPO "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF"
#define TURBO_GGUF_REV "36116c48f4e81b4c66a9c63da13a72959999a5f6"
static const char *const turbo_files[] = {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                                          "generation_config.json", "vocab.json", NULL};

// nerkyor's Coder390 (SFT + RLOO on coding over Qwen3.8-27B): config and tokenizer from the
// BF16 folder of the main repo, weights from the GGUF repo (imatrix quants with the MTP
// layer), both Apache-2.0.
#define C390_REPO "nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2"
#define C390_REV "7eced947d7132dffbc2c0ea4db332081360afbf8"
#define C390_GGUF_REPO "nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-GGUF-NInfer"
#define C390_GGUF_REV "635a4b3d48fd634850d41ade45b4578ad9211e47"
// khazarai's Fable-5 coding distill of Qwen3.8-27B: config and tokenizer from its BF16 repo,
// weights from mradermacher's GGUF quants (with the MTP layer), both Apache-2.0.
#define FABLE_REPO "khazarai/Qwen3.8-27B-Fable-5-Coding-Distilled"
#define FABLE_REV "dbf6da43b26174c26d36c6d85adba5e29280f63e"
static const char *const hf_files[] = {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                                       "generation_config.json", NULL};

const DownloadPack download_packs[DOWNLOAD_NPACKS] = {
    {.name = "coder", .title = "Qwen3.8-Flash-Next Coder", .disk_gb = 80, .min_ram_gb = 96, .repo = "mbrown-uno/Qwen3.8-Flash-Next-GSQ-RCO-Coder-MLX", .rev = "50fa68c5e9cebf53276d8eae8af594fa8b3de9e4", .note = "256 experts, pruned for code, 76 GB"},
    {.name = "full", .title = "Qwen3.8-Flash-Next", .disk_gb = 110, .min_ram_gb = 128, .repo = "mlx-community/Qwen3.8-Flash-Next-4bit-mtp", .rev = "825df623302aeebc434f44ca9122827b04f9a0f1", .note = "512 experts, the original model, 105 GB"},
    {.name = "27b-8bit", .title = "Qwen3.8-27B 8-bit", .disk_gb = 30, .min_ram_gb = 48, .repo = "mlx-community/Qwen3.8-27B-8bit", .rev = "1159008f24a3f97d7afd366fcfee9a79f3bd2b94", .note = "Qwen3.8-27B dense, 8-bit, 29 GB (for LoRA adapters)"},
    {.name = "27b-4bit", .title = "Qwen3.8-27B 4-bit", .disk_gb = 17, .min_ram_gb = 32, .repo = "mlx-community/Qwen3.8-27B-4bit", .rev = "b935172e4bb5dcd0543774a34748fb92422e3027", .note = "Qwen3.8-27B dense, 4-bit, 16 GB"},
    {.name = "27b-pi", .title = "Qwen3.8-27B pi adapter", .disk_gb = 1, .repo = "harness-hub/Qwen3.8-27B-pi-adapted", .rev = "80d05c9051dfd2ae464e1daf4329cb15524caf60", .note = "LoRA adapter for the pi agent on Qwen3.8-27B, 0.6 GB"},
    {.name = "27b-turbo", .title = "Qwen3.8-27B TURBO", .disk_gb = 40, .min_ram_gb = 36, .repo = TURBO_REPO, .rev = TURBO_REV, .only = turbo_files, .gguf_repo = TURBO_GGUF_REPO, .gguf_rev = TURBO_GGUF_REV,
     .gguf_file = "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf",
     .dir = "DavidAU/Qwen3.8-27B-TURBO-NEO-CODER-MAX-MTP-Q4_K_M",
     .note = "Qwen3.8-27B TURBO Fable Cold Fusion fine-tune (DavidAU), Q4_K_M GGUF with MTP, imported, 20 GB"},
    {.name = "27b-turbo-8bit", .title = "Qwen3.8-27B TURBO 8-bit", .disk_gb = 60, .min_ram_gb = 48, .repo = TURBO_REPO, .rev = TURBO_REV, .only = turbo_files, .gguf_repo = TURBO_GGUF_REPO, .gguf_rev = TURBO_GGUF_REV,
     .gguf_file = "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q8_0.gguf",
     .dir = "DavidAU/Qwen3.8-27B-TURBO-NEO-CODER-MAX-MTP-Q8_0",
     .note = "the same fine-tune, Q8_0 GGUF with MTP, imported, 30 GB"},
    {.name = "27b-coder390", .title = "Qwen3.8-27B Coder390", .disk_gb = 45, .min_ram_gb = 48, .repo = C390_REPO, .rev = C390_REV, .subdir = "BF16", .only = hf_files,
     .gguf_repo = C390_GGUF_REPO, .gguf_rev = C390_GGUF_REV, .gguf_subdir = "GGUF",
     .gguf_file = "Qwen3.8-27B-Coder390-EfficientThink-Q4LynnStyle-Q8MTP.gguf",
     .dir = "nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Q4LynnStyle-Q8MTP",
     .note = "Qwen3.8-27B Coder390 fine-tune (nerkyor), Q4 LynnStyle GGUF with Q8 MTP, imported, 23 GB"},
    {.name = "27b-coder390-8bit", .title = "Qwen3.8-27B Coder390 8-bit", .disk_gb = 62, .min_ram_gb = 48, .repo = C390_REPO, .rev = C390_REV, .subdir = "BF16", .only = hf_files,
     .gguf_repo = C390_GGUF_REPO, .gguf_rev = C390_GGUF_REV, .gguf_subdir = "GGUF",
     .gguf_file = "Qwen3.8-27B-Coder390-EfficientThink-Q8_0-MTP.gguf",
     .dir = "nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Q8_0-MTP",
     .note = "the same fine-tune, Q8_0 GGUF with MTP, imported, 31 GB"},
    {.name = "27b-fable5", .title = "Qwen3.8-27B Fable-5", .disk_gb = 40, .min_ram_gb = 36, .repo = FABLE_REPO, .rev = FABLE_REV, .only = hf_files,
     .gguf_repo = "mradermacher/Qwen3.8-27B-Fable-5-Coding-Distilled-i1-GGUF", .gguf_rev = "55f3b23db4b7aab28f6e68e8a49be5bc2e076a6c",
     .gguf_file = "Qwen3.8-27B-Fable-5-Coding-Distilled.i1-Q4_K_M.gguf",
     .dir = "khazarai/Qwen3.8-27B-Fable-5-Coding-Distilled-i1-Q4_K_M",
     .note = "Qwen3.8-27B Fable-5 coding distill (khazarai), imatrix Q4_K_M GGUF with MTP, imported, 20 GB"},
    {.name = "27b-fable5-8bit", .title = "Qwen3.8-27B Fable-5 8-bit", .disk_gb = 62, .min_ram_gb = 48, .repo = FABLE_REPO, .rev = FABLE_REV, .only = hf_files,
     .gguf_repo = "mradermacher/Qwen3.8-27B-Fable-5-Coding-Distilled-GGUF", .gguf_rev = "e8eeb91df17f4b8c107dbaaf3edeeff9f9bfb928",
     .gguf_file = "Qwen3.8-27B-Fable-5-Coding-Distilled.Q8_0.gguf",
     .dir = "khazarai/Qwen3.8-27B-Fable-5-Coding-Distilled-Q8_0",
     .note = "the same fine-tune, Q8_0 GGUF with MTP, imported, 31 GB"},
    // Jackrong's Qwopus3.8-27B-Coder-Flash (Apache-2.0), a coding fine-tune of Qwen3.8-27B
    // distilled from Qwen3.8-Flash-Max: mlx-community's conversions, the 27B's exact tensors
    // and quantization, no MTP head
    {.name = "27b-qwopus", .title = "Qwopus3.8-27B Coder-Flash 4-bit", .disk_gb = 17, .min_ram_gb = 32, .repo = "mlx-community/Qwopus3.8-27B-Coder-Flash-4bit",
     .rev = "15aa5286a501c8f0407c1076b5b130702e48a71d", .note = "Qwopus3.8-27B Coder-Flash fine-tune (Jackrong), dense, 4-bit, 16 GB"},
    {.name = "27b-qwopus-8bit", .title = "Qwopus3.8-27B Coder-Flash 8-bit", .disk_gb = 30, .min_ram_gb = 48, .repo = "mlx-community/Qwopus3.8-27B-Coder-Flash-8bit",
     .rev = "9f17e9c1dee159d7d13e62700a0d9ef817bd7143", .note = "the same fine-tune, 8-bit, 29 GB"},
    // Inkling 2 Mini (thinkingmachines, Apache-2.0): the MLX conversion, dense affine 4-bit
    // with the official NVFP4 / MXFP8 experts; 17.3 GB of weights, so 32 GB Macs (fit.h)
    {.name = "inkling", .title = "Inkling 2 Mini", .disk_gb = 18, .min_ram_gb = 32, .repo = "mlx-community/Inkling-2-Mini-4bit",
     .rev = "9c0e2960609ba7cb40daf92c6f7bf8ed56e1dd62", .note = "Inkling 2 Mini, 26B MoE, 3.4B active, 17 GB"},
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
    *out = path_join(a, d, p->dir ? p->dir : p->repo);
    return OK;
}

const char *download_usage(void) {
    return "usage: saetta download [coder|full|all|27b-8bit|27b-4bit|27b-pi|27b-turbo|27b-turbo-8bit|27b-coder390|27b-coder390-8bit|27b-fable5|27b-fable5-8bit|27b-qwopus|27b-qwopus-8bit|inkling] [--jobs N]   (default: coder with 96 GB of memory or more, 27b-turbo with 36, 27b-4bit with 32; all = coder and "
           "full; into $SAETTA_MODELS, default ~/Models)\n";
}

typedef struct {
    const char *path;  // relative to the repo
    const char *local; // relative to the pack folder (path without the pack's subdir)
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
        // a subdir's files land at the pack folder's root; the others are not fetched
        Str local = path->s;
        if (p->subdir) {
            Str pre = str(p->subdir);
            if (local.len <= pre.len + 1 || memcmp(local.ptr, pre.ptr, pre.len) != 0 || local.ptr[pre.len] != '/') continue;
            local = str_n(local.ptr + pre.len + 1, local.len - pre.len - 1);
        }
        if (p->only) {
            bool keep = false;
            for (const char *const *f = p->only; *f && !keep; f++) keep = str_eqz(local, *f);
            if (!keep) continue;
        }
        vec_push(&items, ((Item){.path = path->s.ptr, .local = local.ptr, .size = (uint64_t)size->i, .sha256 = sha, .git_oid = oid->s.ptr}));
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
    if ((size_t)snprintf(dest, sizeof dest, "%s/%s", sh->root, it->local) >= sizeof dest) return "NoSpaceLeft";
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

// Fetches the files of p->repo at p->rev (those in p->only, if set) into root.
static Err fetch_repo(Arena *a, const DownloadPack *p, const char *root, const char *token, size_t jobs) {
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
        char *dest = path_join(a, root, it->local);
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

static Err one(Arena *a, const DownloadPack *p, const char *token, size_t jobs) {
    char *root;
    TRY(download_pack_dir(a, p, &root));
    dbg("saetta download: %s (%s)\n  %s at %.7s\n  into %s\n", p->name, p->note, p->repo, p->rev, root);
    // an imported pack is complete once model.safetensors exists (gguf_import renames it into
    // place last); its config.json differs from the repo's, so the repo is not fetched again
    const char *st = path_join(a, root, "model.safetensors");
    if (p->gguf_file && file_exists(st)) {
        dbg("  imported pack present: %s\n", st);
        return OK;
    }
    TRY(fetch_repo(a, p, root, token, jobs));
    if (!p->gguf_file) return OK;
    DownloadPack g = {.name = p->name, .repo = p->gguf_repo, .rev = p->gguf_rev, .subdir = p->gguf_subdir,
                      .only = (const char *const[]){p->gguf_file, NULL}};
    dbg("  %s at %.7s: %s%s%s\n", g.repo, g.rev, p->gguf_subdir ? p->gguf_subdir : "", p->gguf_subdir ? "/" : "", p->gguf_file);
    TRY(fetch_repo(a, &g, root, token, jobs));
    const char *gf = path_join(a, root, p->gguf_file);
    dbg("  importing the GGUF into a saetta pack\n");
    TRY(gguf_import(gf, root, root));
    if (unlink(gf) != 0) dbg("  could not delete %s: %s\n", gf, strerror(errno));
    else dbg("  deleted the GGUF (%s)\n", p->gguf_file);
    return OK;
}

static Err run_in(Arena *a, const char *const *args, size_t n) {
    // no name: the pack this Mac's memory suits (fit.h)
    const char *which = NULL;
    size_t jobs = 4;
    for (size_t i = 0; i < n; i++) {
        if (strcmp(args[i], "--jobs") == 0 && i + 1 < n) {
            i += 1;
            uint64_t j;
            TRY(parse_u64(str(args[i]), &j));
            jobs = (size_t)CLAMP(j, 1, 16);
        } else which = args[i];
    }
    if (!which) which = fit_default_pack();
    if (!which) {
        dbg("saetta download: this Mac has %llu GB of memory.\n", (unsigned long long)(fit_ram_bytes() >> 30));
        fit_suggest(fit_ram_bytes());
        return "NotEnoughMemory";
    }
    const char *token = hf_token(a);
    const uint64_t ram = fit_ram_bytes();
    const char *ig = getenv("SAETTA_IGNORE_MEMORY");
    const bool ignore = ig && strcmp(ig, "0") != 0;
    if (strcmp(which, "all") == 0) {
        size_t got = 0;
        for (size_t i = 0; i < DOWNLOAD_NFLASH; i++) {
            const DownloadPack *p = &download_packs[i];
            if (ram && !ignore && ((uint64_t)p->min_ram_gb << 30) > ram) {
                dbg("saetta download: %s skipped: it needs a Mac with %u GB of memory or more; this one has %llu GB.\n", p->name,
                    p->min_ram_gb, (unsigned long long)(ram >> 30));
                continue;
            }
            TRY(one(a, p, token, jobs));
            got++;
        }
        return got ? OK : "NotEnoughMemory";
    }
    const DownloadPack *p = download_find(which);
    if (!p) {
        dbg("saetta download: unknown pack '%s'\n%s", which, download_usage());
        return "UnknownPack";
    }
    // a pack that cannot run here is not downloaded (SAETTA_IGNORE_MEMORY=1 fetches it anyway)
    if (ram && p->min_ram_gb && ((uint64_t)p->min_ram_gb << 30) > ram && !ignore) {
        dbg("saetta download: %s needs a Mac with %u GB of memory or more; this one has %llu GB.\n", p->name, p->min_ram_gb,
            (unsigned long long)(ram >> 30));
        fit_suggest(ram);
        return "NotEnoughMemory";
    }
    return one(a, p, token, jobs);
}

Err download_run(const char *const *args, size_t n) {
    Arena a = {0};
    Err e = run_in(&a, args, n);
    arena_reset(&a);
    return e;
}
