// Tests for the checkpoints at message boundaries (prefix points) at the session API level
// without a GPU (src/model_api.c): which positions a prompt's prefill cuts at
// (session_point_cuts) and how its chunks follow them (prefill_chunk_len), the slot a new
// checkpoint takes and the thinning past the budget (session_point_slot), the matching of a later
// prompt (session_shared_point), and the validity rules (session_keep_points,
// session_invalidate_from): a slot's buffers outlive its conversation, so a checkpoint left by an
// earlier occupant must never be offered as a start for the next one's prompt.
#include "../model.h"
#include "../test.h"

#include <stdlib.h>

// What session_save_point records for a checkpoint at pos (its buffers are not read here: a
// non-NULL marker stands for them), in the slot session_point_slot picks.
static int save_point(Session *s, const uint32_t *ids, size_t pos) {
    session_keep_points(s, pos);
    int k = session_point_slot(s->pts, pos, true);
    if (k < 0) return k;
    s->pts[k].gdn = (sx_buffer *)(uintptr_t)0x1000;
    s->pts[k].pos = pos;
    s->pts[k].valid = true;
    // as session_save_point: the prefix and the token after it
    s->ptoks.len = 0;
    vec_reserve(&s->ptoks, pos + 1);
    memcpy(s->ptoks.ptr, ids, (pos + 1) * sizeof(uint32_t));
    s->ptoks.len = pos + 1;
    return k;
}

// A prompt that shares the first `common` tokens of base, then differs.
static uint32_t *diverging(const uint32_t *base, size_t common, size_t n) {
    uint32_t *p = malloc(n * sizeof *p);
    for (size_t i = 0; i < n; i++) p[i] = i < common ? base[i] : 900000 + (uint32_t)i;
    return p;
}

static size_t shared(const Session *s, const uint32_t *prompt, size_t n) {
    return session_shared_point(s->ptoks.ptr, s->ptoks.len, s->pts, prompt, n);
}

static void session_free_test(Session *s) {
    vec_free(&s->ptoks);
    vec_free(&s->ck[0].tokens);
    vec_free(&s->ck[1].tokens);
    free(s);
}

static size_t valid_count(const Session *s) {
    size_t k = 0;
    for (size_t i = 0; i < PT_MAX; i++) k += s->pts[i].gdn && s->pts[i].valid;
    return k;
}

TEST(points_restore_then_fork) {
    Session *s = calloc(1, sizeof *s);
    size_t n = 40000;
    uint32_t *A = malloc(n * 4), *B = malloc(n * 4);
    for (size_t i = 0; i < n; i++) {
        A[i] = (uint32_t)i;
        B[i] = (uint32_t)(i < 3000 ? i : 500000 + i); // B shares A's first 3000 tokens only
    }
    // A fills the slot's points (prefill from 0 to 32768)
    for (size_t pos = PT_EVERY; pos <= 8 * PT_EVERY; pos += PT_EVERY) save_point(s, A, pos);
    uint32_t *fa = diverging(A, 9000, 20000);
    EXPECT_EQ(shared(s, fa, 20000), 8192);
    // B is restored into the slot at 10000 (the state store: invalidated from 0), then reads its
    // prompt on to 13000: the prefill saves its point at 12288 only
    session_invalidate_from(s, 0);
    save_point(s, B, 12288);
    // a fork of B that leaves it at 9000: A's points at 4096 and 8192 still have their buffers, but
    // they are not B's: no start point
    uint32_t *fb = diverging(B, 9000, 20000);
    EXPECT_EQ(shared(s, fb, 20000), 0);
    // past B's own point, it is offered
    uint32_t *fb2 = diverging(B, 12500, 20000);
    EXPECT_EQ(shared(s, fb2, 20000), 12288);
    // a request that starts B again at 10000 (below its point) keeps nothing past 10000
    session_invalidate_from(s, 10000);
    EXPECT_EQ(shared(s, fb2, 20000), 0);
    // a slot whose point was invalidated is reused before a new one is allocated
    int k = session_point_slot(s->pts, 11000, true);
    EXPECT(k >= 0 && s->pts[k].gdn != NULL);
    free(fa);
    free(fb);
    free(fb2);
    session_free_test(s);
    free(A);
    free(B);
}

TEST(points_decode_crossing) {
    Session *s = calloc(1, sizeof *s);
    size_t n = 40000;
    uint32_t *A = malloc(n * 4), *C = malloc(n * 4);
    for (size_t i = 0; i < n; i++) {
        A[i] = (uint32_t)i;
        C[i] = 700000 + (uint32_t)i;
    }
    for (size_t pos = PT_EVERY; pos <= 8 * PT_EVERY; pos += PT_EVERY) save_point(s, A, pos);
    // the slot is reused for C from 0 (LRU): the prefill to 5000 saves 4096; its reply decodes
    // past 8192 (decode saves no point); the next request restores at 9000 and reads to 13000,
    // saving 12288: the point at 8192 is still A's
    session_invalidate_from(s, 0);
    save_point(s, C, 4096);
    session_invalidate_from(s, 9000);
    save_point(s, C, 12288);
    uint32_t *fc = diverging(C, 10000, 20000);
    EXPECT_EQ(shared(s, fc, 20000), 4096);
    uint32_t *fc2 = diverging(C, 12400, 20000);
    EXPECT_EQ(shared(s, fc2, 20000), 12288);
    free(fc);
    free(fc2);
    session_free_test(s);
    free(A);
    free(C);
}

// Points at any position (message starts), past the old 32K limit; the match is the highest valid
// point within the common prefix, never the prompt's full length.
TEST(points_any_position_and_matching) {
    Session *s = calloc(1, sizeof *s);
    size_t n = 120000;
    uint32_t *A = malloc(n * 4);
    for (size_t i = 0; i < n; i++) A[i] = 3 + (uint32_t)i;
    size_t at[] = {1500, 17333, 40001, 77777, 100003};
    for (size_t i = 0; i < 5; i++) EXPECT(save_point(s, A, at[i]) >= 0);
    EXPECT_EQ(valid_count(s), 5);
    uint32_t *f = diverging(A, 77778, 90000);
    EXPECT_EQ(shared(s, f, 90000), 77777); // the point just before the divergence
    free(f);
    f = diverging(A, 77777, 90000);
    EXPECT_EQ(shared(s, f, 90000), 40001); // at the divergence itself: an earlier one
    free(f);
    // the prompt is exactly the history up to a point: leave one token to run
    EXPECT_EQ(shared(s, A, 100003), 77777);
    EXPECT_EQ(shared(s, A, 100004), 100003);
    EXPECT_EQ(session_point_at(s->pts, 40001) >= 0, true);
    EXPECT_EQ(session_point_at(s->pts, 40002), -1);
    // invalidation keeps a point at the cut (valid, but unused until a save says which token follows
    // it), drops the ones past it
    session_invalidate_from(s, 40001);
    EXPECT_EQ(valid_count(s), 3);
    EXPECT_EQ(shared(s, A, 100004), 17333);
    session_free_test(s);
    free(A);
}

// The budget: PT_MAX points; past it, the oldest and the PT_RECENT most recent stay and the others
// thin to a spacing that grows with the distance from the end.
TEST(points_thinning_geometric) {
    Session *s = calloc(1, sizeof *s);
    size_t n = 400000;
    uint32_t *A = malloc(n * 4);
    for (size_t i = 0; i < n; i++) A[i] = (uint32_t)i;
    for (size_t pos = 1024; pos <= 300 * 1024; pos += 1024) EXPECT(save_point(s, A, pos) >= 0);
    EXPECT_EQ(valid_count(s), PT_MAX);
    size_t ps[PT_MAX], m = 0;
    for (size_t i = 0; i < PT_MAX; i++)
        if (s->pts[i].valid) ps[m++] = s->pts[i].pos;
    for (size_t i = 1; i < m; i++)
        for (size_t j = i; j > 0 && ps[j - 1] > ps[j]; j--) {
            size_t t = ps[j];
            ps[j] = ps[j - 1];
            ps[j - 1] = t;
        }
    EXPECT_EQ(ps[0], 1024);                         // the oldest
    for (size_t i = 0; i < PT_RECENT; i++) EXPECT_EQ(ps[m - 1 - i], (300 - i) * 1024); // the most recent
    // gaps grow towards the start: each gap at least as large as the next one (a geometric spacing)
    size_t end = 300 * 1024;
    for (size_t i = 1; i + 1 < m; i++) {
        double g0 = (double)(ps[i] - ps[i - 1]), g1 = (double)(ps[i + 1] - ps[i]);
        EXPECT(g0 + 1024 >= g1);
    }
    // and the distance from the end spans the whole history in about PT_MAX steps
    EXPECT(end - ps[1] > end / 4);
    // without memory for new buffers, a full list still takes a new point (a thinned one's slot)
    EXPECT(session_point_slot(s->pts, end + 1024, false) >= 0);
    // with nothing valid and nothing allocated, no slot unless allocation is allowed
    Session *t = calloc(1, sizeof *t);
    EXPECT_EQ(session_point_slot(t->pts, 5000, false), -1);
    EXPECT_EQ(session_point_slot(t->pts, 5000, true), 0);
    session_free_test(t);
    session_free_test(s);
    free(A);
}

// The cuts: message starts and multiples of PT_EVERY, each PT_GAP after the previous one taken (the
// request's start at first) and PT_GAP before the end; nothing else is forced (a short first prompt
// is read in one chunk).
TEST(points_cuts) {
    enum { IM = 7 };
    size_t n = 20000;
    uint32_t *ids = calloc(n, 4);
    for (size_t i = 0; i < n; i++) ids[i] = 100;
    // message starts at 0, 5000, 5600, 5900, 12000, 12100 and the generation prompt at 19995
    size_t starts[] = {0, 5000, 5600, 5900, 12000, 12100, 19995};
    for (size_t i = 0; i < 7; i++) ids[starts[i]] = IM;
    size_t out[64];
    size_t k = session_point_cuts(ids, n, 0, IM, out, 64);
    // 4096, 5000 (904 after: no), 5600 (1504: yes), 5900 (300: no), 8192, 12000 (3808), 12100 (100:
    // no), 12288 (288: no), 16384, not 19995 (under PT_GAP before the end)
    size_t want[] = {4096, 5600, 8192, 12000, 16384};
    EXPECT_EQ(k, 5);
    for (size_t i = 0; i < k && i < 5; i++) EXPECT_EQ(out[i], want[i]);
    // from a later start only what lies after it, the spacing counted from that start
    k = session_point_cuts(ids, n, 11500, IM, out, 64);
    EXPECT_EQ(k, 1);
    EXPECT_EQ(out[0], 16384);
    EXPECT_EQ(session_point_cuts(ids, n, n - 2, IM, out, 64), 0);
    // no message marks (another chat format): the multiples of PT_EVERY alone
    k = session_point_cuts(ids, n, 0, UINT32_MAX, out, 64);
    EXPECT_EQ(k, 4);
    EXPECT_EQ(out[3], 16384);
    // capacity respected
    EXPECT_EQ(session_point_cuts(ids, n, 0, IM, out, 2), 2);
    // a short first prompt (the bench's first request, 1268 tokens, its user message near 1200): no cut
    uint32_t *sh = calloc(1268, 4);
    sh[0] = IM, sh[1150] = IM, sh[1262] = IM;
    EXPECT_EQ(session_point_cuts(sh, 1268, 0, IM, out, 64), 0);
    free(sh);
    free(ids);
}

// A checkpoint is used only when the prompt's token at its position matches too (its MTP row at pos
// pairs that token): a prompt that leaves the history exactly there starts at an earlier one.
TEST(points_match_needs_the_token_at_pos) {
    Session *s = calloc(1, sizeof *s);
    size_t n = 30000;
    uint32_t *A = malloc(n * 4);
    for (size_t i = 0; i < n; i++) A[i] = 5 + (uint32_t)i;
    EXPECT(save_point(s, A, 6000) >= 0);
    EXPECT(save_point(s, A, 9000) >= 0);
    uint32_t *f = diverging(A, 9000, 20000); // differs at 9000 itself
    EXPECT_EQ(shared(s, f, 20000), 6000);
    free(f);
    f = diverging(A, 9001, 20000);
    EXPECT_EQ(shared(s, f, 20000), 9000);
    // after a start at 9000 (keep_points(9000): the token at 9000 is not known any more), the
    // checkpoint at 9000 waits for the next save to say which token follows it
    session_keep_points(s, 9000);
    EXPECT_EQ(shared(s, f, 20000), 6000);
    EXPECT(save_point(s, A, 12000) >= 0);
    EXPECT_EQ(shared(s, f, 20000), 9000);
    free(f);
    session_free_test(s);
    free(A);
}

// The chunks of a prefill end at each cut, and are split evenly between cuts (at most CH tokens).
TEST(points_chunks_follow_cuts) {
    size_t cuts[] = {4096, 5900, 8192, 12000, 12100, 16384};
    PrefillHook h = {.cuts = cuts, .ncuts = 6};
    size_t pos0 = 0, n = 19999, i = 0, ends[32], ne = 0;
    while (i < n) {
        size_t T = prefill_chunk_len(&h, pos0, n, i);
        EXPECT(T > 0 && T <= CH);
        i += T;
        ends[ne++] = pos0 + i;
        if (ne == 32) break;
    }
    // 12100 to 16384 is 4284 tokens: two chunks of 2142
    size_t want[] = {4096, 5900, 8192, 12000, 12100, 14242, 16384, 19999};
    EXPECT_EQ(ne, 8);
    for (size_t k = 0; k < ne && k < 8; k++) EXPECT_EQ(ends[k], want[k]);
    // no cuts: the fewest equal chunks, as before
    EXPECT_EQ(prefill_chunk_len(NULL, 0, 10000, 0), 3334);
    EXPECT_EQ(prefill_chunk_len(NULL, 0, 10000, 6668), 3332);
    // a long segment between two cuts: equal chunks of at most CH
    size_t c2[] = {10000};
    PrefillHook h2 = {.cuts = c2, .ncuts = 1};
    EXPECT_EQ(prefill_chunk_len(&h2, 1000, 20000, 0), 3000); // 9000 tokens to the cut: 3 x 3000
    EXPECT_EQ(prefill_chunk_len(&h2, 1000, 20000, 6000), 3000);
    EXPECT_EQ(prefill_chunk_len(&h2, 1000, 20000, 9000), 3667); // 11000 after it: 3 x 3667
}
