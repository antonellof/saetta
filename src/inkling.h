// Inkling 2 Mini (src/inkling.c): the verification hooks beyond model.h. With
// SAETTA_INK_ROUTES=<golden dir> at load, the model reads the reference's expert choices
// (routes.u8 [visits][n][8]) and their 8th - 9th score margins (margins.f32 [visits][n]); for
// positions below n every router then compares its own top-8 with the reference's, and in
// INK_ROUTES_REPLAY mode uses the reference's experts (the weights stay its own).
#ifndef SAETTA_INKLING_H
#define SAETTA_INKLING_H

#include "model.h"

typedef enum { INK_ROUTES_OWN, INK_ROUTES_REPLAY } InkRouteMode;

typedef struct {
    uint32_t flips_sure; // own top-8 differs where the reference's margin exceeds TOL_INK_ROUTE_MARGIN
    uint32_t flips_near; // differs at a near tie (margin at most that)
    uint32_t first_flip; // the first position with a difference (UINT32_MAX: none)
    float sure_margin;   // the largest reference margin among the flips_sure ones
    uint32_t first_sure; // the first position of those (UINT32_MAX: none)
} InkRouteStats;

// Whether m is an Inkling model with reference routes loaded.
bool inkling_has_routes(const Model *m);
void inkling_set_routes(Model *m, InkRouteMode mode);
// The counts since the last call (then reset).
InkRouteStats inkling_route_stats(Model *m);
// Prompt chunks on the tensor ops with half activations (M5 and later; not with SAETTA_NO_ACCEL=1):
// whether their kernels loaded, and switching them on (the default) or off (the exact f32 chunks).
bool inkling_half_available(const Model *m);
void inkling_set_half(Model *m, bool on);
bool inkling_half(const Model *m);

#endif
