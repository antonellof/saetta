// `saetta verify <golden-dir>`: runs the golden token sequence through the engine and
// compares, against a numpy reference of the model, the residual streams after every
// layer and the logits at the dumped positions, and the greedy argmax at every
// position. Tolerances: src/tolerances.h. Exits non-zero on any failure.
#ifndef SAETTA_VERIFY_H
#define SAETTA_VERIFY_H

#include "base.h"

// Runs every check with the pack in pack_dir against the golden data in golden, printing
// the report to out; *ok gets the verdict. Errors: "GoldenShapeMismatch",
// "GoldenSizeMismatch", "GoldenLongerThanChunk", file, JSON, pack and model errors.
Err verify_run(Arena *a, FILE *out, const char *pack_dir, const char *golden, bool *ok);

// Zig's "{e}" (prec < 0) and "{e:.N}" for an f32 (is_f32) or f64 value: the shortest digits,
// rounded half up on those digits to N places, as d.ddde<exp> (1.5e-6, 2.000e0).
int verify_fmt_e(char *out, size_t cap, double x, bool is_f32, int prec);

#endif
