// Test runner for src/test.h: runs every registered test, or those whose name contains argv[1].
#include "../test.h"

#include <stdlib.h>

#define MAX_TESTS 4096

static struct {
    const char *name;
    test_fn fn;
} tests[MAX_TESTS];
static size_t ntests;
static const char *skip_reason;

void test_register(const char *name, test_fn fn) {
    if (ntests == MAX_TESTS) die("too many tests\n");
    tests[ntests].name = name;
    tests[ntests].fn = fn;
    ntests++;
}

void test_skip(const char *why) { skip_reason = why ? why : ""; }

int main(int argc, char **argv) {
    const char *filter = argc > 1 ? argv[1] : NULL;
    int passed = 0, skipped = 0, failed = 0;
    for (size_t i = 0; i < ntests; i++) {
        if (filter && !strstr(tests[i].name, filter)) continue;
        int f = 0;
        skip_reason = NULL;
        double t0 = now_s();
        tests[i].fn(&f);
        double dt = now_s() - t0;
        if (f) {
            failed++;
            dbg("FAIL %s\n", tests[i].name);
        } else if (skip_reason) {
            skipped++;
            dbg("skip %s: %s\n", tests[i].name, skip_reason);
        } else {
            passed++;
            if (dt > 1.0) dbg("ok   %s (%.1f s)\n", tests[i].name, dt);
        }
    }
    dbg("%d passed, %d skipped, %d failed\n", passed, skipped, failed);
    return failed ? 1 : 0;
}
