#include "../control.h"
#include "../test.h"

static int ord(const char *a, const char *b) {
    SemanticVersion x, y;
    if (semver_parse(str(a), &x) || semver_parse(str(b), &y)) return 99;
    return semver_order(&x, &y);
}

TEST(semantic_version_parse_and_order) {
    SemanticVersion v;
    EXPECT_OK(semver_parse(S("1.1.2-prerelease+meta"), &v));
    EXPECT_EQ(v.major, 1);
    EXPECT_EQ(v.patch, 2);
    EXPECT_STR(v.pre, S("prerelease"));
    EXPECT_STR(v.build, S("meta"));
    EXPECT_ERR(semver_parse(S("1.2"), &v), "InvalidVersion");
    EXPECT_ERR(semver_parse(S("01.2.3"), &v), "InvalidVersion");
    EXPECT_ERR(semver_parse(S("1.2.3.4"), &v), "InvalidVersion");
    EXPECT_ERR(semver_parse(S("+1.2.3"), &v), "InvalidVersion");
    EXPECT_EQ(ord("0.6.3", "0.6.2"), 1);
    EXPECT_EQ(ord("0.6.2", "0.6.2"), 0);
    EXPECT_EQ(ord("0.10.0", "0.9.9"), 1);
    EXPECT_EQ(ord("1.0.0-alpha", "1.0.0"), -1);
    EXPECT_EQ(ord("1.0.0-alpha", "1.0.0-alpha.1"), -1);
    EXPECT_EQ(ord("1.0.0-alpha.beta", "1.0.0-beta"), -1);
    EXPECT_EQ(ord("1.0.0-beta.2", "1.0.0-beta.11"), -1);
    EXPECT_EQ(ord("1.0.0-rc.1", "1.0.0-beta.11"), 1);
}
