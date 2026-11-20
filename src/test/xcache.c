#include "../xcache.h"
#include "../test.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// A synthetic pack: one file holding 3 layers x 9 stacked tensors of NE experts each, every
// byte a function of its file offset, so a slot's contents can be checked against the pack.
enum { NL = 3, NT = 9, NE = 8 };
static const uint64_t PER[NT] = {20000, 1000, 1000, 20000, 1000, 1000, 20000, 1000, 1000};

static uint8_t byte_at(uint64_t off) { return (uint8_t)(off * 2654435761u >> 13); }

typedef struct {
    char dir[256];
    MappedFile file;
    Pack pack;
    TensorInfo t[NL][NT];
    XcLayer layers[NL];
} Fake;

static bool fake_make(Fake *f) {
    snprintf(f->dir, sizeof f->dir, "%s/saetta-xcache-XXXXXX", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    if (!mkdtemp(f->dir)) return false;
    char path[300];
    snprintf(path, sizeof path, "%s/x.safetensors", f->dir);
    FILE *fp = fopen(path, "wb");
    if (!fp) return false;
    uint64_t off = 64; // a header-like gap, so offsets are not zero
    for (uint64_t i = 0; i < off; i++) fputc(byte_at(i), fp);
    for (int l = 0; l < NL; l++)
        for (int r = 0; r < NT; r++) {
            f->t[l][r] = (TensorInfo){.dtype = DTYPE_U8, .rank = 1, .offset = off, .nbytes = PER[r] * NE, .file = 0};
            f->t[l][r].shape[0] = PER[r] * NE;
            for (uint64_t i = 0; i < PER[r] * NE; i++) fputc(byte_at(off + i), fp);
            off += PER[r] * NE;
            f->layers[l].t[r] = &f->t[l][r];
        }
    fclose(fp);
    for (int l = 0; l < NL; l++) f->layers[l].ne = NE;
    f->file = (MappedFile){.name = "x.safetensors"};
    f->pack = (Pack){.dir = f->dir, .files = &f->file, .nfiles = 1};
    return true;
}

static void fake_free(Fake *f) {
    char path[300];
    snprintf(path, sizeof path, "%s/x.safetensors", f->dir);
    unlink(path);
    rmdir(f->dir);
}

static bool slot_holds(XCache *c, const Fake *f, uint32_t slot, int layer, uint32_t e) {
    for (uint32_t r = 0; r < NT; r++) {
        const uint8_t *p = (const uint8_t *)sx_buffer_contents(xc_region(c, r)) + (size_t)slot * xc_stride(c, r);
        uint64_t off = f->t[layer][r].offset + (uint64_t)e * PER[r];
        for (uint64_t i = 0; i < PER[r]; i++)
            if (p[i] != byte_at(off + i)) return false;
    }
    return true;
}

static uint64_t slot_bytes(void) {
    uint64_t b = 0;
    for (int r = 0; r < NT; r++) b += (PER[r] + 16383) / 16384 * 16384;
    return b;
}

TEST(xcache_reads_experts_and_evicts_lru) {
    Fake *f = calloc(1, sizeof *f);
    EXPECT(f && fake_make(f));
    sx_device *d = sx_device_create();
    EXPECT(d != NULL);
    Arena a = {0};
    XCache *c = NULL;
    // smaller than one layer's set: refused
    EXPECT_ERR(xc_open(&a, &f->pack, d, f->layers, NL, NT, slot_bytes() * (NE - 1), 3, &c), "CacheTooSmall");
    // exactly one layer's set
    EXPECT_OK(xc_open(&a, &f->pack, d, f->layers, NL, NT, slot_bytes() * NE, 3, &c));
    EXPECT_EQ(xc_slots(c), NE);
    uint32_t slots[NE * 2];

    uint32_t ids0[] = {3, 5, 3, 7};
    EXPECT_OK(xc_map(c, 0, ids0, 4, slots));
    EXPECT_EQ(slots[0], slots[2]);
    EXPECT(slot_holds(c, f, slots[0], 0, 3) && slot_holds(c, f, slots[1], 0, 5) && slot_holds(c, f, slots[3], 0, 7));
    XcStats s = xc_stats(c);
    EXPECT_EQ(s.requests, 3); // the repeated 3 counts once
    EXPECT_EQ(s.misses, 3);
    EXPECT_EQ(s.hits, 0);

    EXPECT_OK(xc_map(c, 0, ids0, 4, slots)); // all hits now
    s = xc_stats(c);
    EXPECT_EQ(s.hits, 3);
    EXPECT_EQ(s.misses, 3);

    // a whole other layer: evicts the least recently used, never a slot of this call
    uint32_t all[NE];
    for (uint32_t i = 0; i < NE; i++) all[i] = NE - 1 - i;
    EXPECT_OK(xc_map(c, 2, all, NE, slots));
    for (uint32_t i = 0; i < NE; i++) EXPECT(slot_holds(c, f, slots[i], 2, all[i]));
    for (uint32_t i = 0; i < NE; i++)
        for (uint32_t j = i + 1; j < NE; j++) EXPECT(slots[i] != slots[j]);

    // layer 0's experts were evicted: reading them again misses and the data is right again
    xc_reset_stats(c);
    EXPECT_OK(xc_map(c, 0, ids0, 4, slots));
    EXPECT_EQ(xc_stats(c).misses, 3);
    EXPECT(slot_holds(c, f, slots[0], 0, 3) && slot_holds(c, f, slots[1], 0, 5) && slot_holds(c, f, slots[3], 0, 7));

    uint32_t bad[] = {NE};
    EXPECT_ERR(xc_map(c, 1, bad, 1, slots), "BadExpertId");
    EXPECT_ERR(xc_map(c, NL, ids0, 1, slots), "BadExpertId");

    xc_close(c);
    arena_reset(&a);
    sx_device_destroy(d);
    fake_free(f);
    free(f);
}

TEST(xcache_random_maps_match_the_pack) {
    Fake *f = calloc(1, sizeof *f);
    EXPECT(f && fake_make(f));
    sx_device *d = sx_device_create();
    Arena a = {0};
    XCache *c = NULL;
    EXPECT_OK(xc_open(&a, &f->pack, d, f->layers, NL, NT, slot_bytes() * (NE + 5), 4, &c));
    uint64_t seed = 12345;
    uint32_t ids[6], slots[6];
    for (int it = 0; it < 300; it++) {
        int layer = (int)(seed % NL);
        for (int i = 0; i < 6; i++) {
            seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            ids[i] = (uint32_t)(seed >> 33) % NE;
        }
        EXPECT_OK(xc_map(c, (uint32_t)layer, ids, 6, slots));
        for (int i = 0; i < 6; i++) EXPECT(slot_holds(c, f, slots[i], layer, ids[i]));
    }
    XcStats s = xc_stats(c);
    EXPECT(s.hits > 0 && s.misses > 0);
    EXPECT_EQ(s.hits + s.misses, s.requests);
    xc_close(c);
    arena_reset(&a);
    sx_device_destroy(d);
    fake_free(f);
    free(f);
}
