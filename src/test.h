// Minimal unit test harness: TEST(name) { ... } registers a test; `make test` runs them all.
// Assertions report file:line and fail the current test (it returns at the first failure).
#ifndef SAETTA_TEST_H
#define SAETTA_TEST_H

#include "base.h"

typedef void (*test_fn)(int *failed);
void test_register(const char *name, test_fn fn);
// Skips the current test with a reason (e.g. no $SAETTA_PACK).
void test_skip(const char *why);
#define SKIP(why) do { test_skip(why); return; } while (0)

#define TEST(name)                                                                  \
    static void test_##name(int *failed_);                                          \
    __attribute__((constructor)) static void test_reg_##name(void) {                \
        test_register(#name, test_##name);                                          \
    }                                                                               \
    static void test_##name(int *failed_)

#define EXPECT(cond)                                                                \
    do { if (!(cond)) { dbg("%s:%d: expected %s\n", __FILE__, __LINE__, #cond);      \
        *failed_ = 1; return; } } while (0)
#define EXPECT_EQ(a, b)                                                             \
    do { long long a_ = (long long)(a), b_ = (long long)(b); if (a_ != b_) {        \
        dbg("%s:%d: %s == %lld, expected %lld\n", __FILE__, __LINE__, #a, a_, b_);     \
        *failed_ = 1; return; } } while (0)
#define EXPECT_STR(a, b)                                                            \
    do { Str a_ = (a), b_ = (b); if (!str_eq(a_, b_)) {                             \
        dbg("%s:%d: %s == \"%.*s\", expected \"%.*s\"\n", __FILE__, __LINE__, #a,       \
            (int)a_.len, a_.ptr, (int)b_.len, b_.ptr); *failed_ = 1; return; } } while (0)
#define EXPECT_OK(e)                                                                \
    do { Err e_ = (e); if (e_) { dbg("%s:%d: %s failed: %s\n", __FILE__, __LINE__,    \
        #e, e_); *failed_ = 1; return; } } while (0)
#define EXPECT_ERR(e, name)                                                         \
    do { Err e_ = (e); if (!e_ || strcmp(e_, name) != 0) {                          \
        dbg("%s:%d: %s gave %s, expected %s\n", __FILE__, __LINE__, #e,               \
            e_ ? e_ : "success", name); *failed_ = 1; return; } } while (0)

#endif
