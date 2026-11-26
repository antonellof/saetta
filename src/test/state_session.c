// Tests at the session API level with real session buffers (a Metal device, no model weights, no
// kernels run): restore points dropped when another conversation's rows replace theirs, and the
// MTP slot's rows at n going through the state store with a state. They allocate about 2 GB of
// shared buffers, so they run with SAETTA_PACK set (as the other GPU tests), not in the default suite.
#include "../gpu.h"
#include "../model.h"
#include "../model_arch.h"
#include "../state_store.h"
#include "../test.h"

#include <stdlib.h>
#include <unistd.h>

static uint32_t *ids_from(uint32_t base, size_t n) {
    uint32_t *p = malloc(n * sizeof *p);
    for (size_t i = 0; i < n; i++) p[i] = base + (uint32_t)i;
    return p;
}

// The reviewer's X/Y case: conversation X holds a restore point at 3000; a request for Y starts in
// X's slot from another conversation's prefix point at 4096 (START_SHARED: Y's rows copied in).
// X's restore point would then pair X's recurrent state with Y's rows: it must be gone, both in
// process (not offered to X's next prompt) and for a save after the request aborts (nothing left
// to persist).
TEST(state_session_restore_point_after_shared_start) {
    if (!getenv("SAETTA_PACK")) SKIP("SAETTA_PACK not set (allocates session buffers)");
    Arena arena = {0};
    Pack empty = {0};
    Gpu g;
    EXPECT_OK(gpu_init(&arena, &empty, &g));
    Session *s;
    EXPECT_OK(dense_arch.s_create_cap(&g, true, 8192, 8192, &s));
    uint32_t *X = ids_from(1000, 8000), *Y = ids_from(500000, 8000);
    // Y's prefix point at 4096, in a held conversation
    Conv held;
    EXPECT_OK(session_new_conv(s, &g, 8192, &held));
    s->pos = 4096;
    EXPECT_OK(session_save_point(s, &g, Y, 4096, true));
    session_swap_conv(s, &held); // held: Y's; s: an empty conversation
    // X's restore point at 3000
    s->pos = 3000;
    EXPECT_OK(session_save_checkpoint_at(s, 1, X, 3000));
    EXPECT_EQ(session_reusable_in(s->ck, X, 5000).n, 3000);
    // Y's request starts here from held's point (what apply does for START_SHARED), then aborts
    EXPECT_EQ(session_shared_point(held.ptoks.ptr, held.ptoks.len, held.pts, Y, 6000), 4096);
    EXPECT_OK(session_start_from(s, &g, &held, 4096, PT_MAX));
    session_invalidate_from(s, 4096);
    EXPECT_EQ(session_reusable_in(s->ck, X, 5000).n, 0);
    // the end-of-turn save takes the longer restore point: none is left
    EXPECT_EQ(s->ck[0].tokens.len, 0);
    EXPECT_EQ(s->ck[1].tokens.len, 0);
    // the shared start's own point is adopted, valid (offered again once a save past it says which
    // token follows it: the request's start keeps it so, server.c)
    EXPECT(session_point_at(s->pts, 4096) >= 0);
    s->pos = 4096;
    EXPECT_OK(session_save_point(s, &g, Y, 4096, true));
    EXPECT_EQ(session_shared_point(s->ptoks.ptr, s->ptoks.len, s->pts, Y, 6000), 4096);
    free(X);
    free(Y);
    gpu_deinit(&g);
    arena_reset(&arena);
}

// The MTP slot's rows at n (Flash-Next: K, V, indexer key, the pooled block holding n) travel with
// a state: a restored conversation's MTP head reads what the resident one would.
TEST(state_session_mtp_rows_at_n) {
    if (!getenv("SAETTA_PACK")) SKIP("SAETTA_PACK not set (allocates session buffers)");
    Arena arena = {0};
    Pack empty = {0};
    Gpu g;
    EXPECT_OK(gpu_init(&arena, &empty, &g));
    Session *a, *b;
    EXPECT_OK(flash_arch.s_create_cap(&g, true, 8192, 8192, &a));
    EXPECT_OK(flash_arch.s_create_cap(&g, true, 8192, 8192, &b));
    size_t n = 6001;
    uint32_t *ids = ids_from(7, 8000);
    a->pos = n;
    EXPECT_OK(session_save_checkpoint_at(a, 1, ids, n));
    StLayout la, lb;
    EXPECT(session_state_layout(a, 1, n, &la));
    EXPECT_EQ(la.nrec, 7);
    // the rows at n: distinct bytes in a, zeros in b
    for (size_t r = 3; r < la.nrec; r++)
        for (size_t i = 0; i < la.rec_bytes[r]; i++) la.rec[r][i] = (uint8_t)(r * 31 + i);
    // they lie at row n of the MTP slot: past every row a state of n tokens holds
    EXPECT(la.rec[3] == la.reg[0].base + (la.reg[0].planes - 2) * la.reg[0].plane_stride + n * la.reg[0].row_bytes);
    EXPECT(la.rec[5] == la.reg[1].base + (la.reg[1].planes - 1) * la.reg[1].plane_stride + n * la.reg[1].row_bytes);
    EXPECT(la.rec[6] == la.reg[2].base + (la.reg[2].planes - 1) * la.reg[2].plane_stride + (n / 4) * la.reg[2].row_bytes);
    char root[] = "/tmp/saetta-state-mtp-XXXXXX";
    EXPECT(mkdtemp(root) != NULL);
    uint8_t ident[32] = {4}, shape[32] = {5};
    StateStore *st;
    EXPECT_OK(state_store_open(root, ident, shape, &st));
    StMeta meta = {.nsnap = 1, .snap = {{.pos = n}}};
    meta.lineage[0] = 1;
    EXPECT(state_store_save(st, &la, ids, n, &meta, NULL, 0, NULL, 0, NULL));
    state_store_wait(st);
    char name[33];
    EXPECT_EQ(state_store_find(st, ids, n + 10, 0, name, NULL), n);
    EXPECT(session_state_layout(b, 1, n, &lb));
    size_t got, nk;
    StMeta mb;
    uint8_t (*keys)[32];
    uint64_t bytes;
    bool touched;
    EXPECT_OK(state_store_load(st, name, ids, n + 10, &lb, &got, &mb, &keys, &nk, &bytes, &touched));
    free(keys);
    for (size_t r = 3; r < lb.nrec; r++) EXPECT(!memcmp(la.rec[r], lb.rec[r], la.rec_bytes[r]));
    state_store_close(st);
    char cmd[200];
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", root);
    (void)system(cmd);
    free(ids);
    gpu_deinit(&g);
    arena_reset(&arena);
}

// The 27B with an MTP head (its KV holds the head's slot, n_attn): the head's K and V rows at n
// travel with a state (the first chunk after a restore rewrites row n and keeps n - 1, which the
// state's rows hold; the saved row n is only overwritten, and restored all the same). Needs
// SAETTA_STATE_TEST_MTP27 (a 27B pack with an MTP head: loads its weights).
TEST(state_session_27b_mtp_rows_at_n) {
    const char *dir = getenv("SAETTA_STATE_TEST_MTP27");
    if (!dir) SKIP("SAETTA_STATE_TEST_MTP27 not set (loads a 27B pack with an MTP head)");
    Arena arena = {0};
    Pack pack;
    EXPECT_OK(pack_open(&arena, dir, &pack));
    Gpu g;
    EXPECT_OK(gpu_init(&arena, &pack, &g));
    Model *m;
    EXPECT_OK(model_load(&arena, &pack, &g, &m));
    EXPECT(model_has_mtp(m));
    Session *a, *b;
    EXPECT_OK(session_create_cap(m, &g, true, 8192, 8192, &a));
    EXPECT_OK(session_create_cap(m, &g, true, 8192, 8192, &b));
    size_t n = 5003;
    uint32_t *ids = ids_from(11, 8000);
    a->pos = n;
    EXPECT_OK(session_save_checkpoint_at(a, 1, ids, n));
    StLayout la, lb;
    EXPECT(session_state_layout(a, 1, n, &la));
    EXPECT_EQ(la.nrec, 4);
    // the head's slot is the last pair of KV planes; its rows at n
    EXPECT(la.rec[2] == la.reg[0].base + (la.reg[0].planes - 2) * la.reg[0].plane_stride + n * la.reg[0].row_bytes);
    EXPECT(la.rec[3] == la.reg[0].base + (la.reg[0].planes - 1) * la.reg[0].plane_stride + n * la.reg[0].row_bytes);
    for (size_t r = 2; r < 4; r++)
        for (size_t i = 0; i < la.rec_bytes[r]; i++) la.rec[r][i] = (uint8_t)(r * 17 + i);
    // and rows below n of the head's planes, as any rows
    uint8_t *h0 = la.reg[0].base + (la.reg[0].planes - 2) * la.reg[0].plane_stride;
    for (size_t i = 0; i < 64; i++) h0[(n - 1) * la.reg[0].row_bytes + i] = (uint8_t)(i ^ 0x5a);
    char root[] = "/tmp/saetta-state-mtp27-XXXXXX";
    EXPECT(mkdtemp(root) != NULL);
    uint8_t ident[32] = {6}, shape[32] = {7};
    StateStore *st;
    EXPECT_OK(state_store_open(root, ident, shape, &st));
    StMeta meta = {.nsnap = 1, .snap = {{.pos = n}}};
    meta.lineage[0] = 2;
    EXPECT(state_store_save(st, &la, ids, n, &meta, NULL, 0, NULL, 0, NULL));
    state_store_wait(st);
    char name[33];
    EXPECT_EQ(state_store_find(st, ids, n + 10, 0, name, NULL), n);
    EXPECT(session_state_layout(b, 1, n, &lb));
    size_t got, nk;
    StMeta mb;
    uint8_t (*keys)[32];
    uint64_t bytes;
    bool touched;
    EXPECT_OK(state_store_load(st, name, ids, n + 10, &lb, &got, &mb, &keys, &nk, &bytes, &touched));
    free(keys);
    for (size_t r = 2; r < 4; r++) EXPECT(!memcmp(la.rec[r], lb.rec[r], la.rec_bytes[r]));
    uint8_t *b0 = lb.reg[0].base + (lb.reg[0].planes - 2) * lb.reg[0].plane_stride;
    EXPECT(!memcmp(h0 + (n - 1) * la.reg[0].row_bytes, b0 + (n - 1) * lb.reg[0].row_bytes, 64));
    state_store_close(st);
    char cmd[200];
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", root);
    (void)system(cmd);
    free(ids);
}

// A conversation starting from another's checkpoint at a position that is no multiple of PT_EVERY
// or of the pooled block (a message start): the rows below it and the MTP slot's rows at it (K, V,
// indexer key, the pooled block holding it) are copied, as a state holds them, with the source's
// valid checkpoints up to it; a checkpoint's slot gives a state layout of its own buffers.
TEST(state_session_start_from_unaligned_point) {
    if (!getenv("SAETTA_PACK")) SKIP("SAETTA_PACK not set (allocates session buffers)");
    Arena arena = {0};
    Pack empty = {0};
    Gpu g;
    EXPECT_OK(gpu_init(&arena, &empty, &g));
    Session *s;
    EXPECT_OK(flash_arch.s_create_cap(&g, true, 8192, 8192, &s));
    uint32_t *X = ids_from(1000, 8000);
    size_t p = 6001;
    // X's checkpoints at 2500 and at p in a held conversation, its rows filled
    Conv held;
    EXPECT_OK(session_new_conv(s, &g, 8192, &held));
    s->pos = 2500;
    EXPECT_OK(session_save_point(s, &g, X, 2500, true));
    s->pos = p;
    EXPECT_OK(session_save_point(s, &g, X, p, true));
    StLayout lx;
    int k = session_point_at(s->pts, p);
    EXPECT(k >= 0);
    EXPECT(session_state_layout(s, 2 + (size_t)k, p, &lx));
    EXPECT(lx.rec[0] == sx_buffer_contents(s->pts[k].gdn));
    for (size_t r = 0; r < lx.nreg; r++)
        for (size_t pl = 0; pl < lx.reg[r].planes; pl++) {
            uint8_t *b = lx.reg[r].base + pl * lx.reg[r].plane_stride;
            size_t rows = p / lx.reg[r].per_row + 1;
            for (size_t i = 0; i < rows * lx.reg[r].row_bytes; i++) b[i] = (uint8_t)(i * 7 + pl + r);
        }
    session_swap_conv(s, &held); // held: X's; s: an empty conversation
    EXPECT_EQ(session_shared_point(held.ptoks.ptr, held.ptoks.len, held.pts, X, 7000), p);
    EXPECT_OK(session_start_from(s, &g, &held, p, PT_MAX));
    EXPECT_EQ(s->pos, p);
    StLayout ls;
    EXPECT(session_state_layout(s, 1, p, &ls));
    // every row below p and the MTP rows at p (the rec entries past the recurrent buffers)
    for (size_t r = 0; r < ls.nreg; r++)
        for (size_t pl = 0; pl < ls.reg[r].planes; pl++) {
            size_t rows = p / ls.reg[r].per_row;
            EXPECT(!memcmp(ls.reg[r].base + pl * ls.reg[r].plane_stride, lx.reg[r].base + pl * lx.reg[r].plane_stride,
                           rows * ls.reg[r].row_bytes));
        }
    StLayout lh;
    session_swap_conv(s, &held);
    EXPECT(session_state_layout(s, 1, p, &lh)); // the source's MTP rows at p
    session_swap_conv(s, &held);
    for (size_t r = 3; r < ls.nrec; r++) EXPECT(!memcmp(ls.rec[r], lh.rec[r], ls.rec_bytes[r]));
    // both checkpoints adopted, at their positions
    EXPECT(session_point_at(s->pts, 2500) >= 0);
    EXPECT(session_point_at(s->pts, p) >= 0);
    EXPECT_EQ(session_shared_point(s->ptoks.ptr, s->ptoks.len, s->pts, X, 5000), 2500);
    // an emptied conversation gives its checkpoints' buffers back
    EXPECT_OK(session_conv_shrink(s, &g, &held, 4096));
    for (size_t i = 0; i < PT_MAX; i++) EXPECT(held.pts[i].gdn == NULL && !held.pts[i].valid);
    free(X);
    gpu_deinit(&g);
    arena_reset(&arena);
}
