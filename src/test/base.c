// Tests for src/base.c. Float expectations are Zig 0.17.1 "{d}" output.
#include "../test.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>

static Str f64s(char *b, double x) { fmt_f64(b, 400, x); return str(b); }
static Str f32s(char *b, float x) { fmt_f32(b, 400, x); return str(b); }

TEST(base_fmt_f64) {
    char b[400];
    EXPECT_STR(f64s(b, 0.1), S("0.1"));
    EXPECT_STR(f64s(b, 1.0), S("1"));
    EXPECT_STR(f64s(b, 1e-7), S("0.0000001"));
    EXPECT_STR(f64s(b, 123456789.0), S("123456789"));
    EXPECT_STR(f64s(b, 3.4028234663852886e38), S("340282346638528860000000000000000000000"));
    EXPECT_STR(f64s(b, -0.0), S("-0"));
    EXPECT_STR(f64s(b, 0.0), S("0"));
    EXPECT_STR(f64s(b, 1e21), S("1000000000000000000000"));
    EXPECT_STR(f64s(b, 2.5), S("2.5"));
    EXPECT_STR(f64s(b, -1.5), S("-1.5"));
    EXPECT_STR(f64s(b, 100.0), S("100"));
    EXPECT_STR(f64s(b, 0.1 + 0.2), S("0.30000000000000004"));
    EXPECT_STR(f64s(b, 9007199254740993.0), S("9007199254740992"));
    EXPECT_STR(f64s(b, NAN), S("nan"));
    EXPECT_STR(f64s(b, -NAN), S("-nan"));
    EXPECT_STR(f64s(b, INFINITY), S("inf"));
    EXPECT_STR(f64s(b, -INFINITY), S("-inf"));
    // Smallest subnormal: "0." then 323 zeros then "5".
    Str s = f64s(b, 5e-324);
    EXPECT_EQ(s.len, 326);
    EXPECT(str_starts(s, S("0.000000000")) && s.ptr[325] == '5');
    s = f64s(b, 1.7976931348623157e308);
    EXPECT_EQ(s.len, 309);
    EXPECT(str_starts(s, S("17976931348623157000")));
}

TEST(base_fmt_f32) {
    char b[400];
    EXPECT_STR(f32s(b, 0.1f), S("0.1"));
    EXPECT_STR(f32s(b, 1.0f), S("1"));
    EXPECT_STR(f32s(b, 1e-7f), S("0.0000001"));
    EXPECT_STR(f32s(b, 123456789.0f), S("123456790"));
    EXPECT_STR(f32s(b, 3.4028235e38f), S("340282350000000000000000000000000000000"));
    EXPECT_STR(f32s(b, 1e-45f), S("0.000000000000000000000000000000000000000000001"));
    EXPECT_STR(f32s(b, 0.3f), S("0.3"));
    EXPECT_STR(f32s(b, 16777217.0f), S("16777216"));
    EXPECT_STR(f32s(b, -2.75f), S("-2.75"));
    EXPECT_STR(f32s(b, 1.17549435e-38f), S("0.000000000000000000000000000000000000011754944"));
    EXPECT_STR(f32s(b, -INFINITY), S("-inf"));
    char small[4];
    EXPECT_EQ(fmt_f32(small, sizeof small, 123.5f), 5);
    EXPECT_STR(str(small), S("123"));
}

TEST(base_err_errno) {
    EXPECT_STR(str(err_errno(ENOENT)), S("FileNotFound"));
    EXPECT_STR(str(err_errno(EACCES)), S("AccessDenied"));
    EXPECT_STR(str(err_errno(EISDIR)), S("IsDir"));
    EXPECT_STR(str(err_errno(EEXIST)), S("PathAlreadyExists"));
    EXPECT_STR(str(err_errno(99999)), S("Unexpected"));
    EXPECT(err_errno(0) == OK);
}

TEST(base_parse_int) {
    uint64_t u;
    int64_t i;
    EXPECT_OK(parse_u64(S("255"), &u));
    EXPECT_EQ(u, 255);
    EXPECT_OK(parse_u64(S("+10"), &u));
    EXPECT_EQ(u, 10);
    EXPECT_OK(parse_u64(S("1_000"), &u));
    EXPECT_EQ(u, 1000);
    EXPECT_OK(parse_u64(S("18446744073709551615"), &u));
    EXPECT(u == UINT64_MAX);
    EXPECT_ERR(parse_u64(S("18446744073709551616"), &u), "Overflow");
    EXPECT_ERR(parse_u64(S("-10"), &u), "Overflow");
    EXPECT_ERR(parse_u64(S(" 10"), &u), "InvalidCharacter");
    EXPECT_ERR(parse_u64(S("_10"), &u), "InvalidCharacter");
    EXPECT_ERR(parse_u64(S(""), &u), "InvalidCharacter");
    EXPECT_OK(parse_i64(S("-9223372036854775808"), &i));
    EXPECT(i == INT64_MIN);
    EXPECT_ERR(parse_i64(S("9223372036854775808"), &i), "Overflow");
    EXPECT_OK(parse_i64(S("-10"), &i));
    EXPECT_EQ(i, -10);
}

TEST(base_str) {
    Str s = S("  hello world \n");
    EXPECT_STR(str_trim(s), S("hello world"));
    EXPECT_EQ(str_find(s, S("world"), 0), 8);
    EXPECT_EQ(str_find(s, S("world"), 9), -1);
    EXPECT_EQ(str_find_char(s, 'o', 7), 9);
    EXPECT(str_starts(s, S("  he")) && str_ends(s, S(" \n")));
    EXPECT_STR(str_sub(s, 2, 7), S("hello"));
    Arena a = {0};
    EXPECT_STR(str(path_join(&a, "a/", "/b")), S("a/b"));
    EXPECT_STR(str(path_join(&a, "a", "b")), S("a/b"));
    EXPECT_STR(str(path_join(&a, "", "b")), S("b"));
    EXPECT_STR(str(arena_printf(&a, "%d-%s", 7, "x")), S("7-x"));
    for (int k = 0; k < 10000; k++) EXPECT(((uintptr_t)arena_alloc(&a, (size_t)k % 700) & 15) == 0);
    char *big = arena_alloc(&a, 1 << 20);
    EXPECT(big[12345] == 0);
    arena_reset(&a);
}

TEST(base_buf_vec_map) {
    Buf b = {0};
    for (int k = 0; k < 1000; k++) buf_printf(&b, "%d,", k);
    EXPECT(str_starts(buf_str(&b), S("0,1,2,")) && str_ends(buf_str(&b), S("999,")));
    EXPECT(b.ptr[b.len] == 0);
    buf_free(&b);
    VecU32 v = {0};
    for (uint32_t k = 0; k < 100; k++) vec_push(&v, k * 3);
    EXPECT_EQ(v.len, 100);
    EXPECT_EQ(v.ptr[99], 297);
    vec_free(&v);
    Arena a = {0};
    StrMap m = {0};
    for (int k = 0; k < 5000; k++) strmap_put(&m, str(arena_printf(&a, "key%d", k)), (void *)(intptr_t)k);
    strmap_put(&m, S("key7"), (void *)(intptr_t)-7);
    EXPECT_EQ(m.len, 5000);
    EXPECT_EQ((intptr_t)*strmap_get(&m, S("key4999")), 4999);
    EXPECT_EQ((intptr_t)*strmap_get(&m, S("key7")), -7);
    EXPECT(!strmap_has(&m, S("key5000")));
    size_t it = 0, n = 0;
    Str k;
    void *val;
    while (strmap_next(&m, &it, &k, &val)) n++;
    EXPECT_EQ(n, 5000);
    strmap_free(&m);
    arena_reset(&a);
    EXPECT(hash_bytes("", 0) != 0);
    EXPECT(hash_bytes("abcdefghi", 9) != hash_bytes("abcdefghj", 9));
}

TEST(base_files) {
    char dir[] = "/tmp/saetta-test-XXXXXX";
    EXPECT(mkdtemp(dir) != NULL);
    Arena a = {0};
    char *sub = arena_printf(&a, "%s/x/y/z", dir);
    EXPECT_OK(mkdir_p(sub));
    EXPECT_OK(mkdir_p(sub));
    char *f = path_join(&a, sub, "f.txt");
    EXPECT_OK(write_file(f, "hello", 5));
    EXPECT(file_exists(f));
    char *p;
    size_t n;
    EXPECT_OK(read_file(f, 100, &p, &n));
    EXPECT_STR(str_n(p, n), S("hello"));
    free(p);
    EXPECT_OK(read_file(f, 6, &p, &n));
    free(p);
    EXPECT_ERR(read_file(f, 5, &p, &n), "StreamTooLong");
    EXPECT_ERR(read_file(f, 4, &p, &n), "StreamTooLong");
    EXPECT_ERR(read_file(path_join(&a, dir, "missing"), 100, &p, &n), "FileNotFound");
    EXPECT_ERR(read_file(dir, 100, &p, &n), "IsDir");
    unlink(f);
    char *cmd = arena_printf(&a, "rm -rf %s", dir);
    EXPECT(system(cmd) == 0);
    arena_reset(&a);
    uint64_t t0 = now_ns();
    EXPECT(now_ns() >= t0 && wall_ns() > 1700000000ull * 1000000000ull);
    EXPECT_STR(str(env_or("SAETTA_SURELY_UNSET_VAR", "dflt")), S("dflt"));
}

static Str fx(char *b, double x, int p) { fmt_fixed(b, 400, x, p); return str(b); }

// Expectations are Zig 0.17.1 "{d:.N}" output.
TEST(base_fmt_fixed) {
    char b[400];
    EXPECT_STR(fx(b, 0.5, 0), S("1"));
    EXPECT_STR(fx(b, 2.5, 0), S("3"));
    EXPECT_STR(fx(b, 2.25, 1), S("2.3"));
    EXPECT_STR(fx(b, 0.125, 2), S("0.13"));
    EXPECT_STR(fx(b, 9.95, 1), S("10.0"));
    EXPECT_STR(fx(b, 0.05, 0), S("0"));
    EXPECT_STR(fx(b, 0.005, 2), S("0.01"));
    EXPECT_STR(fx(b, -0.0, 2), S("-0.00"));
    EXPECT_STR(fx(b, 0, 0), S("0"));
    EXPECT_STR(fx(b, 1e-5, 3), S("0.000"));
    EXPECT_STR(fx(b, 0.49999999999999994, 0), S("0"));
    EXPECT_STR(fx(b, 999.9996, 3), S("1000.000"));
    EXPECT_STR(fx(b, 123456.789, 1), S("123456.8"));
    EXPECT_STR(fx(b, -INFINITY, 2), S("-inf"));
    EXPECT_STR(fx(b, 1e-5, 400), S("(float)")); // wider than Zig's buffer
    EXPECT_EQ(fx(b, 1e300, 0).len, 301);
    EXPECT_EQ(fx(b, 1e300, 2).len, 304);
    // An f32 rounds its own shortest digits.
    fmt_fixed_f32(b, sizeof b, 0.35f, 1);
    EXPECT_STR(str(b), S("0.4"));
    EXPECT_STR(fx(b, (double)0.35f, 1), S("0.3"));
}
