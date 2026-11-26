// Tests for src/state_store.c: CRC32C against a bitwise reference, and the store on a temporary
// folder with a made-up layout (no GPU): round trip into buffers of another capacity, chunks shared
// within a lineage and not across lineages, the longest prefix, corruption, crash leftovers,
// bounds of a forged header, eviction, permissions, another shape in its own folder.
#include "../hash.h"
#include "../state_store.h"
#include "../test.h"

#include <dirent.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

// bit by bit, reflected polynomial 0x82F63B78
static uint32_t crc_ref(uint32_t crc, const uint8_t *p, size_t n) {
    crc = ~crc;
    for (size_t i = 0; i < n; i++) {
        crc ^= p[i];
        for (int k = 0; k < 8; k++) crc = (crc >> 1) ^ (0x82F63B78u & (0u - (crc & 1)));
    }
    return ~crc;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static void fill(uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t)rnd();
}

TEST(state_crc32c) {
    EXPECT_EQ(crc32c(0, "123456789", 9), 0xE3069283u);
    EXPECT_EQ(crc32c(0, "", 0), 0);
    uint8_t *buf = malloc(70000);
    fill(buf, 70000);
    static const size_t lens[] = {0, 1, 7, 8, 9, 31, 32, 33, 100, 4095, 4096, 65536, 69990};
    for (size_t li = 0; li < COUNT(lens); li++)
        for (size_t off = 0; off < 9; off++) {
            size_t n = lens[li];
            EXPECT_EQ(crc32c(0, buf + off, n), crc_ref(0, buf + off, n));
            size_t h = n / 3;
            EXPECT_EQ(crc32c(crc32c(0, buf + off, h), buf + off + h, n - h), crc_ref(0, buf + off, n));
            uint32_t q[4];
            crc32c_quad(buf + off, n, q);
            size_t part = (n / 4) & ~(size_t)7;
            for (size_t k = 0; k < 3; k++) EXPECT_EQ(q[k], crc_ref(0, buf + off + k * part, part));
            EXPECT_EQ(q[3], crc_ref(0, buf + off + 3 * part, n - 3 * part));
        }
    free(buf);
}

// A layout like Flash-Next's, small: KV (4 planes of 64-byte rows), pooled keys (2 planes, one
// 32-byte row per 4 tokens), two recurrent buffers.
typedef struct {
    size_t cap;
    uint8_t *kv, *bk, *r0, *r1;
    StLayout lay;
} Fake;

static void fake_new(Fake *f, size_t cap, bool random) {
    f->cap = cap;
    f->kv = calloc(4, cap * 64);
    f->bk = calloc(2, cap / 4 * 32);
    f->r0 = calloc(1, 100000);
    f->r1 = calloc(1, 777);
    if (random) {
        fill(f->kv, 4 * cap * 64);
        fill(f->bk, 2 * cap / 4 * 32);
        fill(f->r0, 100000);
        fill(f->r1, 777);
    }
    f->lay = (StLayout){.nreg = 2, .nrec = 2,
                        .reg = {{f->kv, 4, cap * 64, 64, 1}, {f->bk, 2, cap / 4 * 32, 32, 4}},
                        .rec = {f->r0, f->r1}, .rec_bytes = {100000, 777}};
}

static void fake_free(Fake *f) {
    free(f->kv);
    free(f->bk);
    free(f->r0);
    free(f->r1);
}

// rows [0, n) of every plane, and the recurrent buffers, equal
static bool fake_same(const Fake *a, const Fake *b, size_t n) {
    for (size_t p = 0; p < 4; p++)
        if (memcmp(a->kv + p * a->cap * 64, b->kv + p * b->cap * 64, n * 64)) return false;
    for (size_t p = 0; p < 2; p++)
        if (memcmp(a->bk + p * (a->cap / 4) * 32, b->bk + p * (b->cap / 4) * 32, (n / 4) * 32)) return false;
    return !memcmp(a->r0, b->r0, 100000) && !memcmp(a->r1, b->r1, 777);
}

static size_t count_files(const char *dir, const char *suffix) {
    DIR *d = opendir(dir);
    size_t k = 0;
    struct dirent *de;
    while (d && (de = readdir(d))) {
        size_t l = strlen(de->d_name), s = strlen(suffix);
        if (l > s && !strcmp(de->d_name + l - s, suffix)) k++;
    }
    if (d) closedir(d);
    return k;
}

static bool first_file(const char *dir, const char *suffix, char *out, size_t cap) {
    DIR *d = opendir(dir);
    struct dirent *de;
    bool found = false;
    while (d && (de = readdir(d)) && !found) {
        size_t l = strlen(de->d_name), s = strlen(suffix);
        if (l > s && !strcmp(de->d_name + l - s, suffix)) {
            snprintf(out, cap, "%s/%s", dir, de->d_name);
            found = true;
        }
    }
    if (d) closedir(d);
    return found;
}

static int mode_of(const char *p) {
    struct stat sb;
    return stat(p, &sb) ? -1 : (int)(sb.st_mode & 0777);
}

static void noop_release(void *p) { (void)p; }

static StMeta meta_of(uint8_t lin, uint32_t a, uint32_t b, uint64_t since) {
    StMeta m = {.nsnap = 1, .snap = {{.prev = {a, b}, .since_eos = since}}, .extra_len = 5};
    memset(m.lineage, lin, 16);
    memcpy(m.extra, "spec!", 5);
    return m;
}

// the chunk file name of chunk c of lineage lin over ids
static void chunk_path(const char *chunks, uint8_t lin, const uint32_t *ids, size_t c, char *out, size_t cap) {
    uint8_t l[16], h[32];
    memset(l, lin, 16);
    Sha256 s;
    sha256_init(&s);
    sha256_update(&s, l, 16);
    sha256_update(&s, ids, (c + 1) * ST_CHUNK * 4);
    sha256_final(&s, h);
    char hx[33];
    hash_hex(h, 16, hx);
    snprintf(out, cap, "%s/%s.kv", chunks, hx);
}

static Err copy_file(const char *from, const char *to) {
    char *t;
    size_t n;
    TRY(read_file(from, (size_t)1 << 30, &t, &n));
    Err e = write_file(to, t, n);
    free(t);
    return e;
}

typedef struct {
    char root[64], dir[256], chunks[300], states[300];
} Dirs;

static void dirs_for(Dirs *d, const char *root, const uint8_t ident[32], const uint8_t shape[32]) {
    char a[33], b[33];
    hash_hex(ident, 8, a);
    hash_hex(shape, 8, b);
    snprintf(d->root, sizeof d->root, "%s", root);
    snprintf(d->dir, sizeof d->dir, "%s/%s/%s", root, a, b);
    snprintf(d->chunks, sizeof d->chunks, "%s/chunks", d->dir);
    snprintf(d->states, sizeof d->states, "%s/states", d->dir);
}

TEST(state_store_roundtrip) {
    char root[] = "/tmp/saetta-state-test-XXXXXX";
    EXPECT(mkdtemp(root) != NULL);
    char sroot[128];
    snprintf(sroot, sizeof sroot, "%s/state", root);
    uint8_t ident[32] = {1, 2, 3}, shape[32] = {9};
    Dirs D;
    dirs_for(&D, sroot, ident, shape);
    StateStore *st;
    EXPECT_OK(state_store_open(sroot, ident, shape, &st));
    EXPECT_STR(str(state_store_dir(st)), str(D.dir));
    EXPECT_EQ(mode_of(sroot), 0700);
    EXPECT_EQ(mode_of(D.dir), 0700);
    EXPECT_EQ(mode_of(D.chunks), 0700);

    size_t n = 10000;
    uint32_t *ids = malloc(20000 * 4);
    for (size_t i = 0; i < 20000; i++) ids[i] = (uint32_t)(rnd() % 248000);
    Fake a;
    fake_new(&a, 16384, true);
    StMeta ma = meta_of(0xA1, 11, 22, 333);
    EXPECT(!state_store_save(st, &a.lay, ids, 4000, &ma, NULL, 0, NULL, 0, noop_release)); // under the minimum
    EXPECT(state_store_save(st, &a.lay, ids, n, &ma, NULL, 0, NULL, 0, noop_release));
    state_store_wait(st);
    EXPECT_EQ(count_files(D.chunks, ".kv"), 2);
    EXPECT_EQ(count_files(D.states, ".st"), 1);
    char path[600];
    EXPECT(first_file(D.states, ".st", path, sizeof path));
    EXPECT_EQ(mode_of(path), 0600);
    EXPECT(!state_store_save(st, &a.lay, ids, n, &ma, NULL, 0, NULL, 0, noop_release)); // sound and there already

    char name[33];
    EXPECT_EQ(state_store_find(st, ids, n + 50, 0, name, NULL), n);
    EXPECT_EQ(state_store_find(st, ids, n, 0, name, NULL), 0);
    EXPECT_EQ(state_store_find(st, ids, n + 50, n, name, NULL), 0);
    uint32_t *other = malloc(20000 * 4);
    memcpy(other, ids, 20000 * 4);
    other[9000] ^= 1;
    EXPECT_EQ(state_store_find(st, other, n + 50, 0, name, NULL), 0);

    // read into buffers of another capacity, with its metadata
    EXPECT_EQ(state_store_find(st, ids, n + 50, 0, name, NULL), n);
    Fake b;
    fake_new(&b, 32768, false);
    size_t got = 0, nk = 0;
    StMeta mb;
    uint64_t bytes = 0;
    uint8_t (*kk)[32] = NULL;
    bool tch = false;
    // the load checks the prompt itself: one that does not extend the state is refused
    EXPECT_ERR(state_store_load(st, name, other, n + 50, &b.lay, &got, &mb, &kk, &nk, &bytes, &tch), "StateMismatch");
    EXPECT(!tch); // refused before any read into the buffers
    EXPECT(state_store_save(st, &a.lay, ids, n, &ma, NULL, 0, NULL, 0, noop_release)); // it was dropped: again
    state_store_wait(st);
    EXPECT_EQ(state_store_find(st, ids, n + 50, 0, name, NULL), n);
    EXPECT_OK(state_store_load(st, name, ids, n + 50, &b.lay, &got, &mb, &kk, &nk, &bytes, &tch));
    EXPECT_EQ(got, n);
    EXPECT(tch);
    EXPECT_EQ(nk, 2);
    EXPECT(fake_same(&a, &b, n));
    EXPECT_EQ(mb.snap[0].prev[0], 11);
    EXPECT_EQ(mb.snap[0].prev[1], 22);
    EXPECT_EQ(mb.snap[0].since_eos, 333);
    EXPECT(!memcmp(mb.lineage, ma.lineage, 16));
    EXPECT_EQ(mb.extra_len, 5);
    EXPECT(!memcmp(mb.extra, "spec!", 5));
    EXPECT_EQ(bytes, 4 * n * 64 + 2 * (n / 4) * 32 + 100000 + 777);
    Fake tiny;
    fake_new(&tiny, 8192, false);
    EXPECT_ERR(state_store_load(st, name, ids, n + 50, &tiny.lay, &got, &mb, &kk, &nk, &bytes, &tch), "StateTooLarge");
    fake_free(&tiny);

    // a longer state of the same lineage: one new chunk, the first two shared
    size_t n2 = 13000;
    for (size_t i = n * 64; i < n2 * 64; i++) a.kv[i] ^= 0x5a;
    EXPECT(state_store_save(st, &a.lay, ids, n2, &ma, NULL, 0, NULL, 0, noop_release));
    state_store_wait(st);
    EXPECT_EQ(count_files(D.chunks, ".kv"), 3);
    EXPECT_EQ(count_files(D.states, ".st"), 2);
    EXPECT_EQ(state_store_find(st, ids, 20000, 0, name, NULL), n2);
    EXPECT_EQ(state_store_find(st, ids, 12000, 0, name, NULL), n);
    // the same ids in another lineage (another computation): its own chunks, none shared
    Fake o;
    fake_new(&o, 16384, true);
    StMeta mo = meta_of(0xB2, 1, 2, 3);
    EXPECT(state_store_save(st, &o.lay, ids, n2, &mo, NULL, 0, NULL, 0, noop_release));
    state_store_wait(st);
    EXPECT_EQ(count_files(D.chunks, ".kv"), 6);
    EXPECT_EQ(count_files(D.states, ".st"), 3);
    Fake c;
    fake_new(&c, 16384, false);
    EXPECT_EQ(state_store_find(st, ids, 20000, 0, name, NULL), n2);
    EXPECT_OK(state_store_load(st, name, ids, 20000, &c.lay, &got, &mb, &kk, &nk, &bytes, &tch));
    EXPECT_EQ(got, n2);
    // whichever lineage it was, its rows are its own
    EXPECT(fake_same(mb.lineage[0] == 0xA1 ? &a : &o, &c, n2));
    fake_free(&c);

    // a conversation restored from A's 10000-token state computes its own rows past 10000 (other
    // numbers) and saves 13000 tokens under its own lineage C, listing A's chunks 0 and 1 by their
    // keys: A's chunk 2 (same ids, written by A's computation) is not reused, C writes its own
    {
        EXPECT_EQ(state_store_find(st, ids, 12000, 0, name, NULL), n);
        Fake rc;
        fake_new(&rc, 16384, false);
        EXPECT_OK(state_store_load(st, name, ids, 12000, &rc.lay, &got, &mb, &kk, &nk, &bytes, &tch));
        EXPECT_EQ(nk, 2);
        for (size_t i = n * 64; i < n2 * 64; i++) rc.kv[i] = (uint8_t)(i * 7); // its own rows past n
        for (size_t p = 1; p < 4; p++)
            for (size_t i = n * 64; i < n2 * 64; i++) rc.kv[p * 16384 * 64 + i] = (uint8_t)(i * 3 + p);
        size_t before_ck = count_files(D.chunks, ".kv");
        StMeta mc = meta_of(0xC3, 5, 6, 7);
        EXPECT(state_store_save(st, &rc.lay, ids, n2, &mc, (const uint8_t (*)[32])kk, nk, NULL, 0, noop_release));
        state_store_wait(st);
        EXPECT_EQ(count_files(D.chunks, ".kv"), before_ck + 1); // chunk 2 under C; 0 and 1 are A's
        char a2[600], c2[600];
        chunk_path(D.chunks, 0xA1, ids, 2, a2, sizeof a2);
        chunk_path(D.chunks, 0xC3, ids, 2, c2, sizeof c2);
        EXPECT(access(a2, F_OK) == 0 && access(c2, F_OK) == 0);
        free(kk);
        // C's state reads back C's rows past n and A's below
        uint8_t cn[32];
        {
            uint8_t l[16];
            memset(l, 0xC3, 16);
            Sha256 sh;
            sha256_init(&sh);
            sha256_update(&sh, l, 16);
            sha256_update(&sh, ids, n2 * 4);
            sha256_final(&sh, cn);
        }
        char cname[33];
        hash_hex(cn, 16, cname);
        Fake back;
        fake_new(&back, 16384, false);
        EXPECT_OK(state_store_load(st, cname, ids, 20000, &back.lay, &got, &mb, &kk, &nk, &bytes, &tch));
        EXPECT(fake_same(&rc, &back, n2));
        free(kk);
        char sp[700];
        snprintf(sp, sizeof sp, "%s/%s.st", D.states, cname);
        EXPECT_EQ(unlink(sp), 0);
        state_store_gc(st, (uint64_t)1 << 40, ST_MAX_AGE_S, (int64_t)time(NULL));
        EXPECT_EQ(count_files(D.chunks, ".kv"), before_ck);
        fake_free(&back);
        fake_free(&rc);
    }

    // a damaged chunk (lineage A's first, which both of A's states use): the load fails, the
    // state that read it is dropped, the chunk too; the other A state then misses it and goes
    char key0[600];
    chunk_path(D.chunks, 0xA1, ids, 0, key0, sizeof key0);
    EXPECT(access(key0, F_OK) == 0);
    int fd = open(key0, O_RDWR);
    EXPECT(fd >= 0);
    uint8_t x;
    EXPECT_EQ(pread(fd, &x, 1, ST_PAGE + 100), 1);
    x ^= 1;
    EXPECT_EQ(pwrite(fd, &x, 1, ST_PAGE + 100), 1);
    close(fd);
    // drop lineage B's state (and so its chunks) so the finds below see A's
    {
        uint8_t l[16], h[32];
        memset(l, 0xB2, 16);
        Sha256 sh;
        sha256_init(&sh);
        sha256_update(&sh, l, 16);
        sha256_update(&sh, ids, n2 * 4);
        sha256_final(&sh, h);
        char hx[33], sp[700];
        hash_hex(h, 16, hx);
        snprintf(sp, sizeof sp, "%s/%s.st", D.states, hx);
        EXPECT_EQ(unlink(sp), 0);
        state_store_gc(st, (uint64_t)1 << 40, ST_MAX_AGE_S, (int64_t)time(NULL));
        EXPECT_EQ(count_files(D.chunks, ".kv"), 3);
    }
    // a load of the A state reads chunk 0 now: it fails and both go
    Fake d;
    fake_new(&d, 16384, false);
    size_t before = count_files(D.states, ".st");
    EXPECT_EQ(state_store_find(st, ids, 20000, 0, name, NULL), n2);
    EXPECT_ERR(state_store_load(st, name, ids, 20000, &d.lay, &got, &mb, &kk, &nk, &bytes, &tch), "StateCorrupt");
    EXPECT_EQ(count_files(D.states, ".st"), before - 1);
    EXPECT(access(key0, F_OK) != 0);
    EXPECT_EQ(state_store_find(st, ids, 20000, 0, name, NULL), n);
    EXPECT_ERR(state_store_load(st, name, ids, 20000, &d.lay, &got, &mb, &kk, &nk, &bytes, &tch), "StateMissing");
    EXPECT_EQ(state_store_find(st, ids, 20000, 0, name, NULL), 0);
    fake_free(&d);
    state_store_close(st);

    // crash leftovers: a dead process's temporary files go, a live one's stay; a state cut to
    // its header page goes
    char tmp[700];
    snprintf(tmp, sizeof tmp, "%s/00112233445566778899aabbccddeeff.st.tmp.999999.0000beef", D.states);
    EXPECT_OK(write_file(tmp, "x", 1));
    snprintf(tmp, sizeof tmp, "%s/00112233445566778899aabbccddeeff.kv.tmp.999999.0000beef", D.chunks);
    EXPECT_OK(write_file(tmp, "x", 1));
    char live[700];
    snprintf(live, sizeof live, "%s/00112233445566778899aabbccddeeff.kv.tmp.%d.0000beef", D.chunks, (int)getpid());
    EXPECT_OK(write_file(live, "x", 1));
    snprintf(tmp, sizeof tmp, "%s/00112233445566778899aabbccddeeff.st", D.states);
    EXPECT_OK(write_file(tmp, "SAETTAST", 8));
    EXPECT_OK(state_store_open(sroot, ident, shape, &st));
    EXPECT_EQ(count_files(D.states, ".0000beef"), 0);
    EXPECT_EQ(count_files(D.chunks, ".0000beef"), 1); // this process's
    EXPECT(access(tmp, F_OK) != 0);
    unlink(live);
    EXPECT(state_store_save(st, &a.lay, ids, n, &ma, NULL, 0, NULL, 0, noop_release));
    state_store_wait(st);
    EXPECT(first_file(D.states, ".st", path, sizeof path));
    EXPECT_EQ(truncate(path, ST_PAGE + 100), 0);
    EXPECT_EQ(state_store_find(st, ids, 20000, 0, name, NULL), 0); // its sections are not all there
    // a header whose lengths are forged (n of 2^40 tokens, crc made valid): refused without
    // allocating, and removed at the next scan. StateHead: n at byte 160, nchunks at 168.
    EXPECT(state_store_save(st, &a.lay, ids, n + 1, &ma, NULL, 0, NULL, 0, noop_release));
    state_store_wait(st);
    {
        DIR *dd = opendir(D.states);
        struct dirent *de;
        path[0] = 0;
        while (dd && (de = readdir(dd)))
            if (strstr(de->d_name, ".st")) {
                char f[700];
                snprintf(f, sizeof f, "%s/%s", D.states, de->d_name);
                struct stat sb;
                if (!stat(f, &sb) && (uint64_t)sb.st_size > ST_PAGE + 100) snprintf(path, sizeof path, "%s", f);
            }
        if (dd) closedir(dd);
    }
    EXPECT(path[0]);
    {
        uint8_t *page = malloc(ST_PAGE);
        fd = open(path, O_RDWR);
        EXPECT(fd >= 0);
        EXPECT_EQ(pread(fd, page, ST_PAGE, 0), (ssize_t)ST_PAGE);
        uint64_t big = (uint64_t)1 << 40, bigc = big / ST_CHUNK;
        memcpy(page + 160, &big, 8);
        memcpy(page + 168, &bigc, 8);
        memset(page + 12, 0, 4);
        uint32_t crc = crc32c(0, page, ST_PAGE);
        memcpy(page + 12, &crc, 4);
        EXPECT_EQ(pwrite(fd, page, ST_PAGE, 0), (ssize_t)ST_PAGE);
        close(fd);
        free(page);
    }
    EXPECT_EQ(state_store_find(st, ids, 20000, 0, name, NULL), 0);
    state_store_gc(st, (uint64_t)1 << 40, ST_MAX_AGE_S, (int64_t)time(NULL));
    EXPECT(access(path, F_OK) != 0);

    // eviction: over the cap the least recently used state goes, then the chunks no state uses;
    // past the age limit everything goes
    while (first_file(D.states, ".st", path, sizeof path)) unlink(path);
    state_store_gc(st, (uint64_t)1 << 40, ST_MAX_AGE_S, (int64_t)time(NULL));
    EXPECT_EQ(count_files(D.chunks, ".kv"), 0);
    EXPECT(state_store_save(st, &a.lay, ids, n, &ma, NULL, 0, NULL, 0, noop_release));
    state_store_wait(st);
    char older[700];
    EXPECT(first_file(D.states, ".st", older, sizeof older));
    struct timeval tv[2] = {{time(NULL) - 3600, 0}, {time(NULL) - 3600, 0}}; // an hour ago: under the age limit
    EXPECT_EQ(utimes(older, tv), 0);
    for (size_t i = 0; i < 20000; i++) other[i] = ids[i] ^ 7; // another conversation
    EXPECT(state_store_save(st, &a.lay, other, 9000, &ma, NULL, 0, NULL, 0, noop_release));
    state_store_wait(st);
    EXPECT_EQ(count_files(D.states, ".st"), 2);
    EXPECT_EQ(count_files(D.chunks, ".kv"), 4);
    uint64_t used;
    size_t nst;
    state_store_usage(st, &used, &nst);
    EXPECT_EQ(nst, 2);
    state_store_gc(st, used - 1, ST_MAX_AGE_S, (int64_t)time(NULL));
    EXPECT(access(older, F_OK) != 0);
    EXPECT_EQ(count_files(D.states, ".st"), 1);
    EXPECT_EQ(count_files(D.chunks, ".kv"), 2);
    state_store_gc(st, 0, ST_MAX_AGE_S, (int64_t)time(NULL)); // a cap of 0: everything, and the loop ends
    EXPECT_EQ(count_files(D.states, ".st") + count_files(D.chunks, ".kv"), 0);
    EXPECT(state_store_save(st, &a.lay, ids, n, &ma, NULL, 0, NULL, 0, noop_release));
    state_store_wait(st);
    state_store_gc(st, (uint64_t)1 << 40, 3600, (int64_t)time(NULL) + 7200);
    EXPECT_EQ(count_files(D.states, ".st") + count_files(D.chunks, ".kv"), 0);

    // another build or layout (shape B): its own folder, so it never reads or reuses A's files;
    // it saves fresh chunks and its resume restores
    EXPECT(state_store_save(st, &a.lay, ids, n, &ma, NULL, 0, NULL, 0, noop_release));
    state_store_close(st);
    uint8_t shapeB[32] = {8};
    Dirs DB;
    dirs_for(&DB, sroot, ident, shapeB);
    EXPECT_OK(state_store_open(sroot, ident, shapeB, &st));
    EXPECT_EQ(state_store_find(st, ids, 20000, 0, name, NULL), 0);
    // even a chunk file of shape A copied under B's name for the same key is not reused: its
    // header names shape A, so B writes its own
    char ka[600], kb[600];
    chunk_path(D.chunks, 0xA1, ids, 0, ka, sizeof ka);
    chunk_path(DB.chunks, 0xA1, ids, 0, kb, sizeof kb);
    EXPECT(access(ka, F_OK) == 0);
    EXPECT_OK(copy_file(ka, kb));
    Fake bb;
    fake_new(&bb, 16384, true);
    EXPECT(state_store_save(st, &bb.lay, ids, n, &ma, NULL, 0, NULL, 0, noop_release));
    state_store_wait(st);
    EXPECT_EQ(count_files(DB.chunks, ".kv"), 2);
    Fake r;
    fake_new(&r, 16384, false);
    EXPECT_EQ(state_store_find(st, ids, 20000, 0, name, NULL), n);
    EXPECT_OK(state_store_load(st, name, ids, 20000, &r.lay, &got, &mb, &kk, &nk, &bytes, &tch));
    EXPECT(fake_same(&bb, &r, n));
    // a chunk that fails its header at a load (A's file put back under B's name) is deleted
    EXPECT_OK(copy_file(ka, kb));
    EXPECT_ERR(state_store_load(st, name, ids, 20000, &r.lay, &got, &mb, &kk, &nk, &bytes, &tch), "StateCorrupt");
    EXPECT(access(kb, F_OK) != 0);
    state_store_close(st);
    fake_free(&r);
    fake_free(&bb);

    fake_free(&a);
    fake_free(&b);
    fake_free(&o);
    free(ids);
    free(other);
    char cmd[400];
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", root);
    (void)system(cmd);
}

static uint64_t du(const char *dir) {
    uint64_t t = 0;
    const char *subs[] = {"states", "chunks"};
    for (int i = 0; i < 2; i++) {
        char d[600];
        snprintf(d, sizeof d, "%s/%s", dir, subs[i]);
        DIR *dd = opendir(d);
        struct dirent *de;
        while (dd && (de = readdir(dd))) {
            if (de->d_name[0] == '.') continue;
            char f[900];
            snprintf(f, sizeof f, "%s/%s", d, de->d_name);
            struct stat sb;
            if (!stat(f, &sb)) t += (uint64_t)sb.st_size;
        }
        if (dd) closedir(dd);
    }
    return t;
}

// Eviction across the whole tree, the load's file lock, and a state that cannot be read.
TEST(state_store_tree_eviction_and_locks) {
    char root[] = "/tmp/saetta-state-tree-XXXXXX";
    EXPECT(mkdtemp(root) != NULL);
    char sroot[128];
    snprintf(sroot, sizeof sroot, "%s/state", root);
    uint8_t ident[32] = {1}, ident2[32] = {2}, shapeA[32] = {3}, shapeB[32] = {4};
    Dirs DA, DB, DP;
    dirs_for(&DA, sroot, ident, shapeA);  // an older engine build of pack 1
    dirs_for(&DB, sroot, ident, shapeB);  // this build of pack 1
    dirs_for(&DP, sroot, ident2, shapeB); // another pack
    uint32_t *ids = malloc(20000 * 4), *ids2 = malloc(20000 * 4);
    for (size_t i = 0; i < 20000; i++) {
        ids[i] = (uint32_t)(rnd() % 248000);
        ids2[i] = ids[i] ^ 0x55;
    }
    Fake a;
    fake_new(&a, 16384, true);
    StMeta m = meta_of(0xA1, 1, 2, 3);
    StateStore *st;
    // the older build's store and the other pack's, written first (older files)
    EXPECT_OK(state_store_open(sroot, ident, shapeA, &st));
    EXPECT(state_store_save(st, &a.lay, ids, 10000, &m, NULL, 0, NULL, 0, noop_release));
    state_store_close(st);
    EXPECT_OK(state_store_open(sroot, ident2, shapeB, &st));
    EXPECT(state_store_save(st, &a.lay, ids2, 10000, &m, NULL, 0, NULL, 0, noop_release));
    state_store_close(st);
    // age them by an hour (under the age limit)
    struct timeval old[2] = {{time(NULL) - 3600, 0}, {time(NULL) - 3600, 0}};
    char path[700];
    const Dirs *olds[] = {&DA, &DP};
    for (int k = 0; k < 2; k++)
        for (int sub = 0; sub < 2; sub++) {
            const char *d = sub ? olds[k]->chunks : olds[k]->states;
            DIR *dd = opendir(d);
            struct dirent *de;
            while (dd && (de = readdir(dd)))
                if (de->d_name[0] != '.') {
                    snprintf(path, sizeof path, "%s/%s", d, de->d_name);
                    utimes(path, old);
                }
            if (dd) closedir(dd);
        }
    // this build's store
    EXPECT_OK(state_store_open(sroot, ident, shapeB, &st));
    EXPECT(state_store_save(st, &a.lay, ids, 10000, &m, NULL, 0, NULL, 0, noop_release));
    state_store_wait(st);
    uint64_t own = du(DB.dir), total;
    size_t nst;
    state_store_usage(st, &total, &nst);
    EXPECT_EQ(total, own + du(DA.dir) + du(DP.dir));
    // a cap of just this store's bytes: every other store's files go, oldest first, this one's stay
    state_store_gc(st, own, ST_MAX_AGE_S, (int64_t)time(NULL));
    EXPECT_EQ(du(DA.dir) + du(DP.dir), 0);
    EXPECT_EQ(du(DB.dir), own);
    EXPECT(access(DA.dir, F_OK) != 0); // emptied stores' folders go too
    // the age limit applies to the other stores as well, under the cap
    state_store_close(st);
    EXPECT_OK(state_store_open(sroot, ident2, shapeB, &st));
    EXPECT(state_store_save(st, &a.lay, ids2, 10000, &m, NULL, 0, NULL, 0, noop_release));
    state_store_close(st);
    EXPECT_OK(state_store_open(sroot, ident, shapeB, &st));
    EXPECT(du(DP.dir) > 0);
    state_store_gc(st, (uint64_t)1 << 40, 3600, (int64_t)time(NULL) + 7200);
    EXPECT_EQ(du(DP.dir) + du(DB.dir), 0);

    // a state a load holds (a shared flock on its file, from any process) is not evicted, nor its
    // chunks; once released it goes
    EXPECT(state_store_save(st, &a.lay, ids, 10000, &m, NULL, 0, NULL, 0, noop_release));
    state_store_wait(st);
    EXPECT(first_file(DB.states, ".st", path, sizeof path));
    int lfd = open(path, O_RDONLY);
    EXPECT(lfd >= 0);
    EXPECT_EQ(flock(lfd, LOCK_SH), 0);
    state_store_gc(st, 0, ST_MAX_AGE_S, (int64_t)time(NULL));
    EXPECT(access(path, F_OK) == 0);
    EXPECT_EQ(count_files(DB.chunks, ".kv"), 2);
    close(lfd);
    state_store_gc(st, 0, ST_MAX_AGE_S, (int64_t)time(NULL));
    EXPECT(access(path, F_OK) != 0);
    EXPECT_EQ(count_files(DB.chunks, ".kv"), 0);

    // a state that cannot be read in a scan keeps every chunk that round
    EXPECT(state_store_save(st, &a.lay, ids, 10000, &m, NULL, 0, NULL, 0, noop_release));
    state_store_wait(st);
    char s1[700];
    EXPECT(first_file(DB.states, ".st", s1, sizeof s1));
    EXPECT(state_store_save(st, &a.lay, ids2, 9000, &m, NULL, 0, NULL, 0, noop_release));
    state_store_wait(st);
    EXPECT_EQ(count_files(DB.chunks, ".kv"), 4);
    char s2[700];
    {
        DIR *dd = opendir(DB.states);
        struct dirent *de;
        s2[0] = 0;
        while (dd && (de = readdir(dd))) {
            if (!strstr(de->d_name, ".st")) continue;
            snprintf(path, sizeof path, "%s/%s", DB.states, de->d_name);
            if (strcmp(path, s1)) snprintf(s2, sizeof s2, "%s", path);
        }
        if (dd) closedir(dd);
    }
    EXPECT(s2[0]);
    EXPECT_EQ(unlink(s2), 0);      // its two chunks now unused
    EXPECT_EQ(chmod(s1, 0000), 0); // and the other state unreadable
    state_store_gc(st, (uint64_t)1 << 40, ST_MAX_AGE_S, (int64_t)time(NULL));
    EXPECT_EQ(count_files(DB.chunks, ".kv"), 4);
    EXPECT_EQ(chmod(s1, 0600), 0);
    state_store_gc(st, (uint64_t)1 << 40, ST_MAX_AGE_S, (int64_t)time(NULL));
    EXPECT_EQ(count_files(DB.chunks, ".kv"), 2);
    state_store_close(st);
    fake_free(&a);
    free(ids);
    free(ids2);
    char cmd[400];
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", root);
    (void)system(cmd);
}

static int64_t mtime_of(const char *p) {
    struct stat sb;
    return stat(p, &sb) ? -1 : (int64_t)sb.st_mtime;
}

// A load refreshes the times of the chunks it read (they are in use); another store's chunks
// go only when no state of that store uses them, however old the chunk files are, and none
// while one of its states cannot be read.
TEST(state_store_chunk_times_and_foreign_sweep) {
    char root[] = "/tmp/saetta-state-times-XXXXXX";
    EXPECT(mkdtemp(root) != NULL);
    char sroot[128];
    snprintf(sroot, sizeof sroot, "%s/state", root);
    uint8_t ident[32] = {1}, shapeA[32] = {3}, shapeB[32] = {4};
    Dirs DA, DB;
    dirs_for(&DA, sroot, ident, shapeA);
    dirs_for(&DB, sroot, ident, shapeB);
    uint32_t *ids = malloc(20000 * 4);
    for (size_t i = 0; i < 20000; i++) ids[i] = (uint32_t)(rnd() % 248000);
    Fake a;
    fake_new(&a, 16384, true);
    StMeta m = meta_of(0xA1, 1, 2, 3);
    StateStore *st;
    EXPECT_OK(state_store_open(sroot, ident, shapeA, &st));
    EXPECT(state_store_save(st, &a.lay, ids, 10000, &m, NULL, 0, NULL, 0, noop_release));
    state_store_wait(st);
    // chunks made old, then read: their times are now
    char c0[600], c1[600];
    chunk_path(DA.chunks, 0xA1, ids, 0, c0, sizeof c0);
    chunk_path(DA.chunks, 0xA1, ids, 1, c1, sizeof c1);
    struct timeval old[2] = {{time(NULL) - 5 * 86400, 0}, {time(NULL) - 5 * 86400, 0}};
    EXPECT_EQ(utimes(c0, old), 0);
    EXPECT_EQ(utimes(c1, old), 0);
    char name[33];
    EXPECT_EQ(state_store_find(st, ids, 12000, 0, name, NULL), 10000);
    Fake b;
    fake_new(&b, 16384, false);
    size_t got, nk;
    StMeta mb;
    uint8_t (*kk)[32];
    uint64_t bytes;
    bool tch;
    EXPECT_OK(state_store_load(st, name, ids, 12000, &b.lay, &got, &mb, &kk, &nk, &bytes, &tch));
    free(kk);
    EXPECT(mtime_of(c0) >= time(NULL) - 60 && mtime_of(c1) >= time(NULL) - 60);
    state_store_close(st);
    // from shape B's store, A's is another's: its chunk files older than the age limit stay while
    // A's state (recent) uses them
    EXPECT_EQ(utimes(c0, old), 0);
    EXPECT_EQ(utimes(c1, old), 0);
    EXPECT_OK(state_store_open(sroot, ident, shapeB, &st));
    state_store_gc(st, (uint64_t)1 << 40, 3 * 86400, (int64_t)time(NULL));
    EXPECT(access(c0, F_OK) == 0 && access(c1, F_OK) == 0);
    // an unused chunk in A's store goes; none while a state of A's cannot be read
    char orphan[600];
    snprintf(orphan, sizeof orphan, "%s/00112233445566778899aabbccddeeff.kv", DA.chunks);
    EXPECT_OK(write_file(orphan, "x", 1));
    char bad[600];
    snprintf(bad, sizeof bad, "%s/ffeeddccbbaa99887766554433221100.st", DA.states);
    EXPECT_OK(write_file(bad, "not a state of this format", 26));
    state_store_gc(st, (uint64_t)1 << 40, ST_MAX_AGE_S, (int64_t)time(NULL));
    EXPECT(access(orphan, F_OK) == 0); // kept: the unreadable state's chunks are unknown
    EXPECT(access(bad, F_OK) == 0);    // another store's state is not this build's to judge
    EXPECT_EQ(unlink(bad), 0);
    state_store_gc(st, (uint64_t)1 << 40, ST_MAX_AGE_S, (int64_t)time(NULL));
    EXPECT(access(orphan, F_OK) != 0);
    EXPECT(access(c0, F_OK) == 0 && access(c1, F_OK) == 0);
    // past the age limit A's state goes, and then its chunks
    char sa[600];
    EXPECT(first_file(DA.states, ".st", sa, sizeof sa));
    EXPECT_EQ(utimes(sa, old), 0);
    state_store_gc(st, (uint64_t)1 << 40, 3 * 86400, (int64_t)time(NULL));
    EXPECT(access(sa, F_OK) != 0);
    EXPECT(access(c0, F_OK) != 0 && access(c1, F_OK) != 0);
    state_store_close(st);
    fake_free(&a);
    fake_free(&b);
    free(ids);
    char cmd[400];
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", root);
    (void)system(cmd);
}

// A state with two snapshots (the restore point at n and the checkpoint before it): a prompt that
// extends the whole state reads both, one that leaves it between the two reads the second only
// (rows of the whole state, its snapshot into xrec), one that leaves it before the second none.
TEST(state_store_two_snapshots) {
    char root[] = "/tmp/saetta-state-snap-XXXXXX";
    EXPECT(mkdtemp(root) != NULL);
    uint8_t ident[32] = {8}, shape[32] = {3};
    StateStore *st;
    EXPECT_OK(state_store_open(root, ident, shape, &st));
    size_t n = 9000, p1 = 6500;
    uint32_t *ids = malloc(20000 * 4);
    for (size_t i = 0; i < 20000; i++) ids[i] = (uint32_t)(rnd() % 248000);
    Fake a;
    fake_new(&a, 16384, true);
    uint8_t *x0 = malloc(100000), *x1 = malloc(777);
    fill(x0, 100000);
    fill(x1, 777);
    a.lay.xrec[0] = x0;
    a.lay.xrec[1] = x1;
    StMeta ma = meta_of(0xB2, 1, 2, 3);
    ma.nsnap = 2;
    ma.snap[1] = (StSnap){p1, {4, 5}, 6, false};
    // a second snapshot not before n is refused
    StMeta bad = ma;
    bad.snap[1].pos = n;
    EXPECT(!state_store_save(st, &a.lay, ids, n, &bad, NULL, 0, NULL, 0, noop_release));
    EXPECT(state_store_save(st, &a.lay, ids, n, &ma, NULL, 0, NULL, 0, noop_release));
    state_store_wait(st);
    char name[33];
    size_t at[ST_MAX_SNAP];
    // the whole state: both snapshots
    EXPECT_EQ(state_store_find(st, ids, n + 5, 0, name, at), n);
    EXPECT_EQ(at[0], n);
    EXPECT_EQ(at[1], p1);
    Fake b;
    fake_new(&b, 16384, false);
    uint8_t *y0 = calloc(1, 100000), *y1 = calloc(1, 777);
    b.lay.xrec[0] = y0;
    b.lay.xrec[1] = y1;
    size_t got, nk;
    StMeta mb;
    uint8_t (*kk)[32];
    uint64_t bytes;
    bool tch;
    EXPECT_OK(state_store_load(st, name, ids, n + 5, &b.lay, &got, &mb, &kk, &nk, &bytes, &tch));
    free(kk);
    EXPECT_EQ(got, n);
    EXPECT(fake_same(&a, &b, n));
    EXPECT(!memcmp(y0, x0, 100000) && !memcmp(y1, x1, 777));
    EXPECT_EQ(mb.nsnap, 2);
    EXPECT(mb.snap[0].loaded && mb.snap[1].loaded);
    EXPECT_EQ(mb.snap[1].pos, p1);
    EXPECT_EQ(mb.snap[1].prev[1], 5);
    EXPECT_EQ(mb.snap[1].since_eos, 6);
    // a prompt that leaves the state at 8000: the second snapshot only
    uint32_t *other = malloc(20000 * 4);
    memcpy(other, ids, 20000 * 4);
    other[8000] ^= 1;
    EXPECT_EQ(state_store_find(st, other, 12000, 0, name, at), p1);
    EXPECT_EQ(at[0], 0);
    EXPECT_EQ(at[1], p1);
    EXPECT_EQ(state_store_find(st, other, 12000, p1, name, at), 0); // no more than min_n
    Fake c;
    fake_new(&c, 16384, false);
    uint8_t *z0 = calloc(1, 100000), *z1 = calloc(1, 777);
    c.lay.xrec[0] = z0;
    c.lay.xrec[1] = z1;
    c.lay.rec[0] = c.lay.rec[1] = NULL; // snapshot 0 is not read
    EXPECT_OK(state_store_load(st, name, other, 12000, &c.lay, &got, &mb, &kk, &nk, &bytes, &tch));
    free(kk);
    EXPECT_EQ(got, p1);
    EXPECT(!mb.snap[0].loaded && mb.snap[1].loaded);
    EXPECT(!memcmp(z0, x0, 100000) && !memcmp(z1, x1, 777));
    // only the chunks up to the second snapshot (two of 4096 rows; not the tail past 8192)
    EXPECT_EQ(bytes, 4 * 8192 * 64 + 2 * (8192 / 4) * 32 + 100000 + 777);
    for (size_t pl = 0; pl < 4; pl++)
        EXPECT(!memcmp(a.kv + pl * a.cap * 64, c.kv + pl * c.cap * 64, 8192 * 64));
    // a capacity that holds those rows but not the state's n is enough for it
    Fake small;
    fake_new(&small, 8192, false);
    small.lay.xrec[0] = z0;
    small.lay.xrec[1] = z1;
    small.lay.rec[0] = small.lay.rec[1] = NULL;
    EXPECT_OK(state_store_load(st, name, other, 12000, &small.lay, &got, &mb, &kk, &nk, &bytes, &tch));
    free(kk);
    EXPECT_EQ(got, p1);
    fake_free(&small);
    // a prompt that is the state's history to the second snapshot exactly: it must leave a token
    EXPECT_EQ(state_store_find(st, ids, p1, 0, name, at), 0);
    EXPECT_EQ(state_store_find(st, ids, p1 + 1, 0, name, at), p1);
    // leaving before the second snapshot: nothing
    other[8000] ^= 1;
    other[100] ^= 1;
    EXPECT_EQ(state_store_find(st, other, 12000, 0, name, at), 0);
    state_store_close(st);
    char cmd[200];
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", root);
    (void)system(cmd);
    fake_free(&a);
    fake_free(&b);
    fake_free(&c);
    free(x0), free(x1), free(y0), free(y1), free(z0), free(z1), free(ids), free(other);
}
