#include "../spec.h"
#include "../test.h"

TEST(lookup_drafts_the_continuation_of_a_repeated_trigram) {
    Lookup l = {0};
    const uint32_t toks[] = {1, 2, 3, 4, 5, 6, 9, 1, 2, 3};
    for (size_t i = 0; i < COUNT(toks); i++) lookup_push(&l, toks[i]);
    uint32_t out[4];
    size_t n = lookup_draft(&l, out, 4);
    lookup_free(&l);
    EXPECT_EQ(n, 4);
    EXPECT_EQ(out[0], 4);
    EXPECT_EQ(out[1], 5);
    EXPECT_EQ(out[2], 6);
    EXPECT_EQ(out[3], 9);
}

TEST(depth_controller) {
    Depth d = DEPTH_INIT;
    // always accepted: goes deep
    for (int i = 0; i < 50; i++) depth_record(&d, 4, 4, 40);
    EXPECT(depth_pick(&d, 4) >= 3);
    // never accepted: plain steps (with an occasional probe)
    for (int i = 0; i < 50; i++) depth_record(&d, 1, 0, 20);
    size_t zeros = 0;
    for (int i = 0; i < 16; i++) zeros += depth_pick(&d, 4) == 0;
    EXPECT(zeros >= 14);
}

TEST(lookup_draft_in_another_source) {
    // the history: ... 7 8 9 | the source: 1 7 8 9 4 5 SEP 6 7 8 9 2 3
    Lookup h = {0}, f = {0};
    const uint32_t ht[] = {20, 21, 6, 7, 8, 9};
    for (size_t i = 0; i < COUNT(ht); i++) lookup_push(&h, ht[i]);
    const uint32_t ft[] = {1, 7, 8, 9, 4, 5, SPEC_SEP, 6, 7, 8, 9, 2, 3, SPEC_SEP, 30, 31, 32, 33, 34, SPEC_SEP, 40, 31, 32, 33, SPEC_SEP};
    for (size_t i = 0; i < COUNT(ft); i++) {
        if (ft[i] == SPEC_SEP) lookup_push_sep(&f);
        else lookup_push(&f, ft[i]);
    }
    uint32_t out[8];
    size_t m;
    // the most recent occurrence of 7 8 9 in the source, with 4 tokens of context (6 7 8 9)
    size_t n = lookup_draft_in(&f, h.hist.ptr, h.hist.len, out, 8, &m);
    EXPECT_EQ(n, 2);
    EXPECT_EQ(out[0], 2);
    EXPECT_EQ(out[1], 3); // stopped before the separator
    EXPECT_EQ(m, 4);
    // the history drafting from itself: what lookup_draft drafts
    Lookup s = {0};
    const uint32_t st[] = {1, 2, 3, 4, 5, 6, 9, 1, 2, 3};
    for (size_t i = 0; i < COUNT(st); i++) lookup_push(&s, st[i]);
    uint32_t a[4], b[4];
    size_t na = lookup_draft(&s, a, 4), nb = lookup_draft_in(&s, s.hist.ptr, s.hist.len, b, 4, &m);
    EXPECT_EQ(na, nb);
    EXPECT(memcmp(a, b, na * sizeof *a) == 0);
    EXPECT_EQ(m, 3); // the first 1 2 3 starts the history: no context before it
    // a context that matches past the trigram and then differs: 5 1 2 3 against 0 1 2 3
    const uint32_t c1[] = {7, 5, 1, 2, 3};
    Lookup s2 = {0};
    const uint32_t s2t[] = {8, 0, 1, 2, 3, 4, 9};
    for (size_t i = 0; i < COUNT(s2t); i++) lookup_push(&s2, s2t[i]);
    EXPECT_EQ(lookup_draft_in(&s2, c1, 5, out, 8, &m), 2);
    EXPECT_EQ(m, 3);
    const uint32_t c4[] = {8, 0, 1, 2, 3};
    EXPECT_EQ(lookup_draft_in(&s2, c4, 5, out, 8, &m), 2);
    EXPECT_EQ(m, 5);
    lookup_free(&s2);
    // a trigram that ends a text (31 32 33 before the last separator) still drafts from its earlier
    // occurrence; an unknown one drafts nothing
    const uint32_t c2[] = {50, 31, 32, 33};
    EXPECT_EQ(lookup_draft_in(&f, c2, 4, out, 8, &m), 1);
    EXPECT_EQ(out[0], 34);
    const uint32_t c3[] = {50, 60, 61, 62};
    EXPECT_EQ(lookup_draft_in(&f, c3, 4, out, 8, &m), 0);
    EXPECT_EQ(m, 0);
    lookup_free(&h);
    lookup_free(&f);
    lookup_free(&s);
}
