// Prompt lookup, the speculation gate and the MTP depth controller (see spec.h).
#include "spec.h"

#include <math.h>
#include <stdlib.h>

static uint64_t key(const uint32_t *t) {
    return ((uint64_t)t[0] << 42) ^ ((uint64_t)t[1] << 21) ^ (uint64_t)t[2];
}

static size_t slot_of(uint64_t k, size_t cap) {
    uint64_t h = k * 0x9E3779B97F4A7C15ull;
    h ^= h >> 29;
    return (size_t)h & (cap - 1);
}

static void map_grow(Lookup *l) {
    size_t ncap = l->cap ? l->cap * 2 : 64;
    uint64_t *keys = xcalloc(ncap, sizeof *keys);
    uint32_t *vals = xcalloc(ncap, sizeof *vals);
    uint8_t *used = xcalloc(ncap, 1);
    for (size_t i = 0; i < l->cap; i++) {
        if (!l->used[i]) continue;
        size_t s = slot_of(l->keys[i], ncap);
        while (used[s]) s = (s + 1) & (ncap - 1);
        used[s] = 1;
        keys[s] = l->keys[i];
        vals[s] = l->vals[i];
    }
    free(l->keys);
    free(l->vals);
    free(l->used);
    l->keys = keys;
    l->vals = vals;
    l->used = used;
    l->cap = ncap;
}

static void map_put(Lookup *l, uint64_t k, uint32_t v) {
    if ((l->len + 1) * 4 > l->cap * 3) map_grow(l);
    size_t s = slot_of(k, l->cap);
    while (l->used[s]) {
        if (l->keys[s] == k) {
            l->vals[s] = v;
            return;
        }
        s = (s + 1) & (l->cap - 1);
    }
    l->used[s] = 1;
    l->keys[s] = k;
    l->vals[s] = v;
    l->len++;
}

static bool map_get(const Lookup *l, uint64_t k, uint32_t *v) {
    if (l->cap == 0) return false;
    size_t s = slot_of(k, l->cap);
    while (l->used[s]) {
        if (l->keys[s] == k) {
            *v = l->vals[s];
            return true;
        }
        s = (s + 1) & (l->cap - 1);
    }
    return false;
}

void lookup_reset(Lookup *l) {
    l->hist.len = 0;
    if (l->cap) memset(l->used, 0, l->cap);
    l->len = 0;
}

void lookup_push(Lookup *l, uint32_t tok) {
    size_t n = l->hist.len;
    if (n >= SPEC_N) map_put(l, key(l->hist.ptr + n - SPEC_N), (uint32_t)n);
    vec_push(&l->hist, tok);
}

size_t lookup_draft(const Lookup *l, uint32_t *out, size_t n) {
    size_t hn = l->hist.len;
    if (hn < SPEC_N + 1) return 0;
    uint32_t at;
    if (!map_get(l, key(l->hist.ptr + hn - SPEC_N), &at)) return 0;
    if (at >= hn) return 0;
    size_t m = MIN(n, hn - at);
    memcpy(out, l->hist.ptr + at, m * sizeof *out);
    return m;
}

size_t lookup_draft_in(const Lookup *src, const uint32_t *ctx, size_t nctx, uint32_t *out, size_t n, size_t *match) {
    *match = 0;
    size_t hn = src->hist.len;
    if (nctx < SPEC_N + 1) return 0;
    uint32_t at;
    if (!map_get(src, key(ctx + nctx - SPEC_N), &at)) return 0;
    if (at >= hn || at < SPEC_N) return 0;
    const uint32_t *h = src->hist.ptr;
    size_t m = 0;
    while (m < n && at + m < hn && h[at + m] != SPEC_SEP) m++;
    if (m == 0) return 0;
    memcpy(out, h + at, m * sizeof *out);
    size_t c = SPEC_N;
    while (c < SPEC_CTX && c < at && c < nctx && h[at - c - 1] == ctx[nctx - c - 1]) c++;
    *match = c;
    return m;
}

void lookup_push_sep(Lookup *l) { vec_push(&l->hist, SPEC_SEP); }

void lookup_free(Lookup *l) {
    vec_free(&l->hist);
    free(l->keys);
    free(l->vals);
    free(l->used);
    l->keys = NULL;
    l->vals = NULL;
    l->used = NULL;
    l->cap = l->len = 0;
}

bool gate_allow(Gate *g) {
    if (g->pause == 0) return true;
    g->pause -= 1;
    if (g->pause == 0) g->avg = GATE_BREAK_EVEN + 0.5f; // retry with a neutral history
    return false;
}

size_t gate_depth(const Gate *g, size_t max) {
    float r = roundf(g->avg * 1.5f);
    size_t d = (size_t)(r > 2.0f ? r : 2.0f);
    return MIN(d, max);
}

void gate_record(Gate *g, size_t tokens) {
    g->avg = 0.7f * g->avg + 0.3f * (float)tokens;
    if (g->avg < GATE_BREAK_EVEN) g->pause = GATE_COOLDOWN;
}

size_t depth_pick(Depth *d, size_t kmax) {
    d->rounds += 1;
    size_t best = 0;
    float best_rate = 1.0f / d->ms[0];
    float e = 1.0f;
    float run = 1.0f;
    size_t top = MIN(kmax, COUNT(d->p));
    for (size_t k = 1; k <= top; k++) {
        run *= d->p[k - 1];
        e += run;
        float rate = e / d->ms[k];
        if (rate > best_rate * 1.01f) { // a deeper round must win (1%: 3% kept depth 1 off on prose, where it wins 3.6%)
            best_rate = rate;
            best = k;
        }
    }
    // a plain step learns nothing about acceptance: probe depth 1 now and then
    if (best == 0 && d->rounds % 16 == 0) best = 1;
    return best;
}

void depth_record(Depth *d, size_t k, size_t a, float ms) {
    if (k < COUNT(d->ms)) d->ms[k] = 0.9f * d->ms[k] + 0.1f * ms;
    size_t top = MIN(k, COUNT(d->p));
    for (size_t i = 0; i < top; i++) {
        if (i < a) {
            d->p[i] = 0.9f * d->p[i] + 0.1f;
        } else {
            d->p[i] = 0.9f * d->p[i];
            break;
        }
    }
}
