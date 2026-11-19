#include "../anthropic.h"
#include "../test.h"

TEST(billing_stamp_pinned_in_place) {
    Arena a = {0};
    Str t = anthropic_pin_billing_stamp(&a, S("x-anthropic-billing-header: cc_version=2.1.170.bf4; cc_entrypoint=cli; cch=b145e;\nYou are"));
    EXPECT_STR(t, S("x-anthropic-billing-header: cc_version=2.1.170.fff; cc_entrypoint=cli; cch=fffff;\nYou are"));
    arena_reset(&a);
}
