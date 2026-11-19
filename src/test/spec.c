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
