#include "../test.h"
#include "../verify.h"

// Zig's {e} / {e:.3}, as std.fmt prints them.
TEST(verify_fmt_e_matches_zig_scientific) {
    char b[64];
    verify_fmt_e(b, sizeof b, 5e-5f, true, -1);
    EXPECT_STR(str(b), S("5e-5"));
    verify_fmt_e(b, sizeof b, 1.5e-6f, true, -1);
    EXPECT_STR(str(b), S("1.5e-6"));
    verify_fmt_e(b, sizeof b, 4.5887e-6f, true, 3);
    EXPECT_STR(str(b), S("4.589e-6"));
    verify_fmt_e(b, sizeof b, 1e-6f, true, 3);
    EXPECT_STR(str(b), S("1.000e-6"));
    verify_fmt_e(b, sizeof b, 9.9996e-3, false, 3);
    EXPECT_STR(str(b), S("1.000e-2"));
    verify_fmt_e(b, sizeof b, 123.4, false, 3);
    EXPECT_STR(str(b), S("1.234e2"));
    verify_fmt_e(b, sizeof b, 0, false, 3);
    EXPECT_STR(str(b), S("0.000e0"));
}
