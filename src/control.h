// `saetta status`, `saetta stop` and `saetta update`: the server on this machine
// (src/instance.h), whether it was started by `saetta start` or by hand, and the binary.
// Each writes its report to `out` (stdout; the caller flushes) and sets *code to the exit
// code. An Err is printed by the caller as "saetta <cmd>: <err>" with exit code 1.
#ifndef SAETTA_CONTROL_H
#define SAETTA_CONTROL_H

#include <stdio.h>

#include "base.h"

// Prints the running server and its sessions; *code 0 when one runs, 1 when none.
Err control_status(FILE *out, int *code);
// Stops the running server (SIGTERM, then waits up to 30 s for it to release its lock).
Err control_stop(FILE *out, int *code);
// Replaces this binary with the latest GitHub release when it is newer than `current`
// (checksum verified, then an atomic rename over the running binary's path).
Err control_update(FILE *out, const char *current, int *code);

// std.SemanticVersion: major.minor.patch[-pre][+build].
typedef struct {
    size_t major, minor, patch;
    Str pre;   // ptr NULL when absent
    Str build; // ptr NULL when absent
} SemanticVersion;
// Errors: "InvalidVersion", "Overflow". pre and build point into text.
Err semver_parse(Str text, SemanticVersion *out);
// -1, 0, 1 as lhs is lower, equal or greater than rhs (precedence ignores build).
int semver_order(const SemanticVersion *lhs, const SemanticVersion *rhs);

#endif
