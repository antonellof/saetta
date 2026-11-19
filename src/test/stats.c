#include "../stats.h"
#include "../test.h"

// Expected strings from the Zig 0.17.1 std ("{d:.0}|{d:.1}|{d:.2}").
TEST(fmt_fixed_matches_zig) {
    static const struct { double x; const char *want; } cases[] = {
        {0, "0|0.0|0.00"},       {-0.0, "-0|-0.0|-0.00"}, {0.5, "1|0.5|0.50"},     {2.5, "3|2.5|2.50"},
        {0.05, "0|0.1|0.05"},    {0.06, "0|0.1|0.06"},    {0.96, "1|1.0|0.96"},    {0.125, "0|0.1|0.13"},
        {2.25, "2|2.3|2.25"},    {9.95, "10|10.0|9.95"},  {99.95, "100|100.0|99.95"}, {0.005, "0|0.0|0.01"},
        {123456.789, "123457|123456.8|123456.79"}, {19.6, "20|19.6|19.60"},
    };
    for (size_t i = 0; i < COUNT(cases); i++) {
        char a[64], b[64], c[64], got[200];
        fmt_fixed(a, sizeof a, cases[i].x, 0);
        fmt_fixed(b, sizeof b, cases[i].x, 1);
        fmt_fixed(c, sizeof c, cases[i].x, 2);
        snprintf(got, sizeof got, "%s|%s|%s", a, b, c);
        EXPECT_STR(str(got), str(cases[i].want));
    }
}
