// Tests for the prefix points' validity (src/model_api.c session_keep_points,
// session_invalidate_from, session_shared_point), at the session API level without a GPU: a
// slot's point buffers outlive its conversation, so a point left by an earlier occupant must
// never be offered as a start for the next one's prompt.
#include "../model.h"
#include "../test.h"

#include <stdlib.h>

// What session_save_point records for point k at pos == (k + 1) * PT_EVERY (its buffers are not
// read here: a non-NULL marker stands for them).
static void save_point(Session *s, const uint32_t *ids, size_t pos) {
    size_t k = pos / PT_EVERY - 1;
    s->pts[k].gdn = (sx_buffer *)(uintptr_t)0x1000;
    s->pts[k].valid = true;
    s->ptoks.len = 0;
    vec_reserve(&s->ptoks, pos);
    memcpy(s->ptoks.ptr, ids, pos * sizeof(uint32_t));
    s->ptoks.len = pos;
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

TEST(points_restore_then_fork) {
    Session *s = calloc(1, sizeof *s);
    size_t n = 40000;
    uint32_t *A = malloc(n * 4), *B = malloc(n * 4);
    for (size_t i = 0; i < n; i++) {
        A[i] = (uint32_t)i;
        B[i] = (uint32_t)(i < 3000 ? i : 500000 + i); // B shares A's first 3000 tokens only
    }
    // A fills the slot's points (prefill from 0 to 32768)
    for (size_t pos = PT_EVERY; pos <= PT_MAX * PT_EVERY; pos += PT_EVERY) save_point(s, A, pos);
    uint32_t *fa = diverging(A, 9000, 20000);
    EXPECT_EQ(shared(s, fa, 20000), 8192);
    // B is restored into the slot at 10000 (the state store: invalidated from 0), then reads its
    // prompt on to 13000: the prefill saves its point at 12288 only
    session_invalidate_from(s, 0);
    save_point(s, B, 12288);
    // a fork of B that leaves it at 9000: A's points at 4096 and 8192 are still allocated, but not
    // B's: no start point (before the fix: 8192, A's state)
    uint32_t *fb = diverging(B, 9000, 20000);
    EXPECT_EQ(shared(s, fb, 20000), 0);
    // past B's own point, it is offered
    uint32_t *fb2 = diverging(B, 12500, 20000);
    EXPECT_EQ(shared(s, fb2, 20000), 12288);
    // a request that starts B again at 10000 (below its point) keeps nothing past 10000
    session_invalidate_from(s, 10000);
    EXPECT_EQ(shared(s, fb2, 20000), 0);
    free(fa);
    free(fb);
    free(fb2);
    vec_free(&s->ptoks);
    vec_free(&s->ck[0].tokens);
    vec_free(&s->ck[1].tokens);
    free(s);
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
    for (size_t pos = PT_EVERY; pos <= PT_MAX * PT_EVERY; pos += PT_EVERY) save_point(s, A, pos);
    // the slot is reused for C from 0 (LRU): the prefill to 5000 saves 4096; its reply decodes
    // past 8192 (decode saves no point); the next request restores at 9000 and reads to 13000,
    // saving 12288: the point at 8192 is still A's
    session_invalidate_from(s, 0);
    save_point(s, C, 4096);
    session_invalidate_from(s, 9000);
    save_point(s, C, 12288);
    uint32_t *fc = diverging(C, 10000, 20000);
    EXPECT_EQ(shared(s, fc, 20000), 4096); // before the fix: 8192, A's state
    uint32_t *fc2 = diverging(C, 12400, 20000);
    EXPECT_EQ(shared(s, fc2, 20000), 12288);
    free(fc);
    free(fc2);
    vec_free(&s->ptoks);
    vec_free(&s->ck[0].tokens);
    vec_free(&s->ck[1].tokens);
    free(s);
    free(A);
    free(C);
}
