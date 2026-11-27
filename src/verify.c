// Golden-data check of the engine (see verify.h).
#include "verify.h"

#include <math.h>
#include <stdlib.h>
#include <unistd.h>

#include "gpu.h"
#include "inkling.h"
#include "json.h"
#include "model.h"
#include "tolerances.h"
#include "weights.h"

int verify_fmt_e(char *out, size_t cap, double x, bool is_f32, int prec) {
    char b[400];
    size_t i = 0;
    bool neg = signbit(x) != 0;
    if (isnan(x) || isinf(x)) {
        i = (size_t)snprintf(b, sizeof b, "%s", isnan(x) ? "nan" : neg ? "-inf" : "inf");
        goto done;
    }
    // shortest digits from the decimal formatter, then the exponent of the first one
    char d[400];
    int n = is_f32 ? fmt_f32(d, sizeof d, (float)fabs(x)) : fmt_f64(d, sizeof d, fabs(x));
    char dig[400];
    int nd = 0, dot = -1;
    for (int k = 0; k < n; k++) {
        if (d[k] == '.') dot = nd;
        else dig[nd++] = d[k];
    }
    if (dot < 0) dot = nd;
    int first = 0;
    while (first < nd && dig[first] == '0') first++;
    uint64_t mant = 0;
    int ol = 0, e10 = 0;
    if (first < nd) {
        int last = nd;
        while (last > first + 1 && dig[last - 1] == '0') last--;
        for (int k = first; k < last; k++) mant = mant * 10 + (uint64_t)(dig[k] - '0');
        ol = last - first;
        e10 = dot - 1 - first;
    } else {
        ol = 1; // zero
    }
    // round() in scientific mode: keep 1 + prec + 1 digits, then half up on the last
    if (prec >= 0) {
        int rd = 1 + prec;
        if (rd < ol) {
            for (int k = rd + 1; k < ol; k++) mant /= 10;
            ol = rd + 1;
            if (mant % 10 >= 5) {
                mant = mant / 10 + 1;
                ol = rd;
                uint64_t p = 1;
                for (int k = 0; k < rd; k++) p *= 10;
                if (mant == p) {
                    mant /= 10;
                    e10 += 1;
                }
            }
        }
    }
    char md[24];
    snprintf(md, sizeof md, "%llu", (unsigned long long)mant);
    int ml = (int)strlen(md);
    if (neg) b[i++] = '-';
    b[i++] = md[0];
    b[i++] = '.';
    size_t dp = i;
    for (int k = 1; k < ml; k++) b[i++] = md[k];
    if (ml == 1) i--; // no fraction digits: drop the point
    if (prec >= 0) {
        if (ml == 1) i++;
        if (prec > ml - 1) {
            for (int k = 0; k < prec - (ml - 1); k++) b[i++] = '0';
        } else
            i = dp + (size_t)prec - (prec == 0);
    }
    i += (size_t)snprintf(b + i, sizeof b - i, "e%d", e10);
done:
    b[i] = 0;
    if (cap) {
        size_t m = i < cap - 1 ? i : cap - 1;
        memcpy(out, b, m);
        out[m] = 0;
    }
    return (int)i;
}

// "{e:.3}" of an f32 / f64 into a caller buffer.
static const char *e3(char *buf, float x) {
    verify_fmt_e(buf, 64, x, true, 3);
    return buf;
}
static const char *e3d(char *buf, double x) {
    verify_fmt_e(buf, 64, x, false, 3);
    return buf;
}

static Err read_f32(Arena *a, const char *dir, const char *name, size_t n, const float **out) {
    char *bytes;
    size_t len;
    TRY(read_file(path_join(a, dir, name), (size_t)1 << 31, &bytes, &len));
    if (len != n * 4) return "GoldenSizeMismatch";
    float *f = ARENA_ARRAY(a, float, n);
    memcpy(f, bytes, n * 4);
    free(bytes);
    *out = f;
    return OK;
}

// max |a - b| / max |b|
static float rel(const float *a, const float *b, size_t n) {
    float m = 0, s = 1e-30f;
    for (size_t i = 0; i < n; i++) {
        m = fmaxf(m, fabsf(a[i] - b[i]));
        s = fmaxf(s, fabsf(b[i]));
    }
    return m / s;
}

static double jf(const JValue *v) {
    if (v && v->kind == J_FLOAT) return v->f;
    if (v && v->kind == J_INTEGER) return (double)v->i;
    return 0;
}

// KL(p || q) of the softmax distributions of two logit vectors, in nats, f64.
static double kl(const float *ref, const float *eng, size_t n) {
    double mr = -INFINITY, me = -INFINITY;
    for (size_t i = 0; i < n; i++) {
        mr = fmax(mr, ref[i]);
        me = fmax(me, eng[i]);
    }
    double zr = 0, ze = 0;
    for (size_t i = 0; i < n; i++) {
        zr += exp((double)ref[i] - mr);
        ze += exp((double)eng[i] - me);
    }
    double d = 0;
    for (size_t i = 0; i < n; i++) {
        double lp = (double)ref[i] - mr - log(zr);
        double lq = (double)eng[i] - me - log(ze);
        d += exp(lp) * (lp - lq);
    }
    return d;
}

// Scalar argmax (first largest), as the reference scan.
static size_t argmax_scan(const float *v, size_t n) {
    size_t best = 0;
    for (size_t k = 0; k < n; k++)
        if (v[k] > v[best]) best = k;
    return best;
}

static Err read_json(Arena *a, const char *dir, const char *name, JValue **out) {
    char *bytes;
    size_t len;
    TRY(read_file(path_join(a, dir, name), (size_t)64 << 20, &bytes, &len)); // long goldens: a top2.json of MBs
    char *copy = arena_dup(a, bytes, len + 1);
    free(bytes);
    return json_parse(a, str_n(copy, len), true, out);
}

static int64_t top2_argmax(const JValue *top2, size_t t) {
    return json_get(top2->arr.items[t], "argmax")->i;
}

// ---- DeepSeek-V4-Flash ----------------------------------------------------------------------
//
// The model rounds to BF16 and FP8 at nearly every op as its reference does, so the engine's
// f32 sums against the oracle's f64 ones drift across layers (an input one BF16 step off flips
// e4m3 codes of the next linear's input). Two gates instead of per-layer bounds on a free run:
// - forced, on a golden that dumps every position: each layer starts from the oracle's streams
//   of the layer before (SAETTA_DSV4_FORCE), so its error is its own; tight bounds on every
//   layer and on the logits KL, equal greedy argmax at every position outside near ties, and
//   the prefill path's streams bit-identical to decode's;
// - functional, on any golden: greedy argmax agreement outside near ties; KL and the per-layer
//   drift are printed, not gated.

typedef struct {
    const char *dir;
    const JValue *toks, *poss, *top2;
    const float *xs, *ls;
    size_t ntok, npos, layers, width, vocab;
} DsGolden;

static Err verify_dsv4(Arena *a, FILE *out, const Pack *pack, const char *pack_dir, const DsGolden *G, bool *okp) {
    // SAETTA_VERIFY_FREE_RUN=1 (diagnostic): the functional check on an all-positions golden too
    const bool forced = G->npos == G->ntok && !getenv("SAETTA_VERIFY_FREE_RUN");
    if (forced) setenv("SAETTA_DSV4_FORCE", path_join(a, G->dir, "x_layers.f32"), 1);
    else unsetenv("SAETTA_DSV4_FORCE");
    Gpu g;
    TRY(gpu_init(a, pack, &g));
    Model *m;
    Session *s;
    Err err = model_load(a, pack, &g, &m);
    const ModelInfo *mi = err ? NULL : model_info(m);
    if (!err && (G->layers != mi->layers || G->width != mi->width || G->vocab != mi->vocab)) err = "GoldenShapeMismatch";
    if (!err) err = session_create_kv(m, &g, false, &s);
    if (err) {
        gpu_deinit(&g);
        return err;
    }
    const size_t L = G->layers, W = G->width, V = G->vocab;
    fprintf(out, "model: %s, pack %s\n", mi->name, pack_dir);
    fprintf(out, "mode: %s\n", forced ? "forced (every layer on the reference's input streams)" : "functional (free run)");
    float *trace = ARENA_ARRAY(a, float, L * W);
    float *dec = forced ? ARENA_ARRAY(a, float, G->npos * L * W) : NULL; // decode streams, for prefill
    float *worst = ARENA_ARRAY(a, float, L);
    for (size_t li = 0; li < L; li++) worst[li] = 0;
    double worst_kl = 0, sum_kl = 0;
    size_t nontie = 0, agree = 0, ties = 0, tie_agree = 0;
    char b1[64], b2[64], b3[64];
    const char *dump_path = getenv("SAETTA_VERIFY_DUMP"); // diagnostic: decode streams, x_layers layout
    FILE *df = dump_path ? fopen(dump_path, "wb") : NULL;
    for (size_t t = 0; t < G->ntok && !err; t++) {
        size_t pi = G->npos;
        for (size_t i = 0; i < G->npos; i++)
            if ((size_t)G->poss->arr.items[i]->i == t) pi = i;
        s->trace = pi < G->npos ? trace : NULL;
        s->trace_len = pi < G->npos ? L * W : 0;
        float *logits;
        err = model_step(m, s, (uint32_t)G->toks->arr.items[t]->i, &logits);
        if (err) break;
        if (pi < G->npos) {
            for (size_t li = 0; li < L; li++)
                worst[li] = fmaxf(worst[li], rel(trace + li * W, G->xs + (pi * L + li) * W, W));
            double k = kl(G->ls + pi * V, logits, V);
            worst_kl = fmax(worst_kl, k);
            sum_kl += k;
            if (dec) memcpy(dec + pi * L * W, trace, L * W * sizeof(float));
            if (df) fwrite(trace, sizeof(float), L * W, df);
        }
        const JValue *ref = G->top2->arr.items[t];
        size_t best = argmax_scan(logits, V), want = (size_t)json_get(ref, "argmax")->i;
        double margin = jf(json_get(ref, "top1")) - jf(json_get(ref, "top2"));
        if (margin < TOL_ARGMAX_MARGIN) {
            ties++;
            tie_agree += best == want;
        } else {
            nontie++;
            agree += best == want;
            if (best != want) {
                fmt_fixed(b1, sizeof b1, margin, 4);
                fprintf(out, "argmax differs at position %zu: engine %zu, reference %zu (margin %s)\n", t, best, want, b1);
            }
        }
    }
    if (df) fclose(df);
    bool ok = !err;
    // prefill (successive decode steps writing the trace): the same streams as decode, bit for bit
    if (ok && forced) {
        session_reset(s);
        uint32_t *tokens = ARENA_ARRAY(a, uint32_t, G->ntok);
        for (size_t t = 0; t < G->ntok; t++) tokens[t] = (uint32_t)G->toks->arr.items[t]->i;
        sx_buffer *tb;
        float *plog;
        err = gpu_new_buffer(&g, L * G->ntok * W * 4, &tb);
        if (!err) err = model_prefill(m, s, tokens, G->ntok, tb, &plog);
        if (!err) {
            const float *tr = gpu_floats(tb);
            size_t ndiff = 0;
            for (size_t i = 0; i < G->npos; i++) {
                size_t t = (size_t)G->poss->arr.items[i]->i;
                for (size_t li = 0; li < L; li++)
                    ndiff += memcmp(tr + (li * G->ntok + t) * W, dec + (i * L + li) * W, W * sizeof(float)) != 0;
            }
            ok = ndiff == 0;
            fprintf(out, "prefill: streams %s decode's (%zu of %zu layer rows differ)%s\n", ndiff ? "differ from" : "identical to",
                    ndiff, G->npos * L, ndiff ? "  FAIL" : "");
            sx_buffer_release(tb);
        }
        ok = ok && !err;
    }
    fprintf(out, "layer  max rel err (residual streams, %zu positions)%s\n", G->npos, forced ? "" : ", drift: not gated");
    for (size_t li = 0; li < L; li++) {
        bool bad = forced && worst[li] > TOL_DSV4_FORCE_LAYER_REL;
        ok = ok && !bad;
        fprintf(out, "%5zu  %s%s\n", li, e3(b1, worst[li]), bad ? "  FAIL" : "");
    }
    bool kbad = forced && worst_kl > TOL_DSV4_FORCE_KL;
    fprintf(out, "logits KL(ref||engine) at the dumped positions: worst %s, mean %s%s\n", e3d(b1, worst_kl),
            e3d(b2, G->npos ? sum_kl / (double)G->npos : 0), kbad ? "  FAIL" : "");
    double frac = nontie ? (double)agree / (double)nontie : 1.0;
    const bool gated = forced || nontie >= TOL_DSV4_AGREE_MIN_POSITIONS;
    bool abad = forced ? agree != nontie : gated && frac < TOL_DSV4_ARGMAX_AGREE;
    fmt_fixed(b1, sizeof b1, frac * 100, 2);
    fprintf(out, "greedy argmax: %zu of %zu equal outside near ties (%s%%), near ties (margin below %g) %zu of %zu equal%s\n",
            agree, nontie, b1, TOL_ARGMAX_MARGIN, tie_agree, ties, abad ? "  FAIL" : "");
    ok = ok && !kbad && !abad;
    if (forced) {
        verify_fmt_e(b1, sizeof b1, TOL_DSV4_FORCE_LAYER_REL, true, -1);
        verify_fmt_e(b2, sizeof b2, TOL_DSV4_FORCE_KL, false, -1);
        fprintf(out, "verify: %s (forced; tolerances: layer %s, KL %s, argmax all outside ties)\n", ok ? "PASS" : "FAIL", b1, b2);
    } else {
        fmt_fixed(b3, sizeof b3, TOL_DSV4_ARGMAX_AGREE * 100, 1);
        if (gated)
            fprintf(out, "verify: %s (functional; argmax agreement outside near ties at least %s%%)\n", ok ? "PASS" : "FAIL", b3);
        else
            fprintf(out, "verify: %s (functional; %zu positions outside near ties, under %d: agreement and KL not gated)\n",
                    ok ? "PASS" : "FAIL", nontie, TOL_DSV4_AGREE_MIN_POSITIONS);
    }
    gpu_deinit(&g);
    if (err) return err;
    *okp = ok;
    return OK;
}

// ---- Inkling 2 Mini ----------------------------------------------------------------------------
//
// Its 48 visits route every token to 8 of 128 experts by a hard top-8, and over a long prompt
// some tokens' 8th and 9th scores sit within f32 rounding of each other (measured: a gap of
// 3.2e-7 at one token of the 1815-token golden, gaps under 1e-6 in most visits at 8329 tokens):
// the oracle and the engine's paths can pick different experts there, and one such token's
// difference reaches every later token through the convs and attention. The golden holds the
// oracle's choices and their margins (routes.u8, margins.f32; src/inkling.h), so:
// - on the oracle's experts (INK_ROUTES_REPLAY; the weights stay the engine's): decode token by
//   token, the chunked prefill (PCH tokens) and the prefill in 200-token chunks (the ring wraps
//   inside a chunk), each with the generic bounds on every dumped visit and the logits, greedy
//   argmax outside near ties; every router's own top-8 is compared with the oracle's, and a
//   difference where the oracle's margin exceeds TOL_INK_ROUTE_MARGIN fails;
// - on its own experts, as served: decode's per-visit bound (TOL_INK_OWN_LAYER_REL) at the dumped
//   positions before its first difference from the oracle's routes, greedy argmax at every
//   position outside near ties (at most TOL_INK_ARGMAX_SKIP_FRAC of them skipped), KL at the
//   dumped positions; the chunked prefill's last-token KL and argmax (TOL_PREFILL_FAST_KL); a
//   verification chunk;
// - where the tensor ops run, the accelerated prompt chunks (half activations, the default there)
//   in whole prompt chunks: last-token KL (TOL_PREFILL_FAST_KL) and argmax on the f16 KV cache with
//   both routings and on an f32 cache with the oracle's; on the oracle's experts also the
//   per-visit bound TOL_PREFILL_FAST_LAYER_REL and the router differences TOL_INK_HALF_ROUTE_FRAC
//   (the exact checks above run with them off).
// Goldens longer than a prompt chunk are run in pieces (the prefill trace of a piece only).

#define INK_PIECE 1024
#define INK_DECODE_MAX 16384

typedef struct {
    float worst_logits;
    double worst_kl;
    size_t aok, askip, abad;
    size_t first_bad; // the first position whose argmax differs outside near ties (ntok: none)
} InkDec;

// Decode over the golden from position `from` (the tokens before it prefilled in chunks: the
// long goldens): err [npos][L] the per-visit error at each dumped position decoded (others 0).
static Err ink_decode(Arena *a, FILE *out, Model *m, Session *s, const DsGolden *G, const uint32_t *tokens, const size_t *pidx,
                      size_t from, float *err, InkDec *r) {
    const size_t L = G->layers, W = G->width, V = G->vocab;
    float *trace = ARENA_ARRAY(a, float, L * W);
    *r = (InkDec){.first_bad = G->ntok};
    for (size_t i = 0; i < G->npos * L; i++) err[i] = 0;
    session_reset(s);
    if (from) {
        float *lg;
        TRY(model_prefill(m, s, tokens, from, NULL, &lg));
    }
    char b1[64];
    for (size_t t = from; t < G->ntok; t++) {
        const size_t pi = pidx[t];
        s->trace = pi < G->npos ? trace : NULL;
        s->trace_len = pi < G->npos ? L * W : 0;
        float *logits;
        Err e = model_step(m, s, tokens[t], &logits);
        if (e) {
            s->trace = NULL;
            return e;
        }
        if (pi < G->npos) {
            for (size_t li = 0; li < L; li++) err[pi * L + li] = rel(trace + li * W, G->xs + (pi * L + li) * W, W);
            r->worst_logits = fmaxf(r->worst_logits, rel(logits, G->ls + pi * V, V));
            r->worst_kl = fmax(r->worst_kl, kl(G->ls + pi * V, logits, V));
        }
        const JValue *ref = G->top2->arr.items[t];
        const size_t best = argmax_scan(logits, V), want = (size_t)json_get(ref, "argmax")->i;
        const double margin = jf(json_get(ref, "top1")) - jf(json_get(ref, "top2"));
        if (best == want)
            r->aok++;
        else if (margin < TOL_ARGMAX_MARGIN)
            r->askip++;
        else {
            r->abad++;
            r->first_bad = MIN(r->first_bad, t);
            fmt_fixed(b1, sizeof b1, margin, 4);
            fprintf(out, "argmax differs at position %zu: engine %zu, reference %zu (margin %s)\n", t, best, want, b1);
        }
    }
    s->trace = NULL;
    s->trace_len = 0;
    return OK;
}

// The prefill over the golden in pieces of `piece` tokens: per-visit worst at the dumped
// positions; the last logits in *plog.
static Err ink_prefill_in(Model *m, Session *s, sx_buffer *tb, const DsGolden *G, const uint32_t *tokens, const size_t *pidx,
                          size_t piece, float *pw, float **plog) {
    const size_t L = G->layers, W = G->width, N = G->ntok;
    for (size_t li = 0; li < L; li++) pw[li] = 0;
    session_reset(s);
    for (size_t t0 = 0; t0 < N; t0 += piece) {
        const size_t n = MIN(piece, N - t0);
        TRY(model_prefill(m, s, tokens + t0, n, tb, plog));
        const float *tr = gpu_floats(tb);
        for (size_t t = t0; t < t0 + n; t++) {
            const size_t pi = pidx[t];
            if (pi == G->npos) continue;
            for (size_t li = 0; li < L; li++) pw[li] = fmaxf(pw[li], rel(tr + (li * n + t - t0) * W, G->xs + (pi * L + li) * W, W));
        }
    }
    return OK;
}
static Err ink_prefill(Model *m, Session *s, sx_buffer *tb, const DsGolden *G, const uint32_t *tokens, const size_t *pidx, float *pw,
                       float **plog) {
    return ink_prefill_in(m, s, tb, G, tokens, pidx, INK_PIECE, pw, plog);
}

// The routers' own top-8 against the oracle's over the last run: a difference at a clear margin fails.
static bool ink_routes_ok(FILE *out, Model *m, const char *what, InkRouteStats *st) {
    *st = inkling_route_stats(m);
    char b1[64];
    verify_fmt_e(b1, sizeof b1, TOL_INK_ROUTE_MARGIN, true, -1);
    bool ok = st->flips_sure <= TOL_INK_ROUTE_RARE && (st->flips_sure == 0 || st->sure_margin <= TOL_INK_ROUTE_MARGIN_MAX);
    fprintf(out, "  %s: own top-8 differs from the oracle's at %u (visit, token) pairs with margin over %s%s, %u at near ties\n", what,
            st->flips_sure, b1, ok ? "" : "  FAIL", st->flips_near);
    if (st->flips_sure)
        fprintf(out, "    first at position %u, largest margin %g (allowed: at most %u, none above %g)\n", st->first_sure,
                (double)st->sure_margin, TOL_INK_ROUTE_RARE, (double)TOL_INK_ROUTE_MARGIN_MAX);
    return ok;
}

static Err verify_inkling(Arena *a, FILE *out, const Pack *pack, const char *pack_dir, const DsGolden *G, bool *okp) {
    if (access(path_join(a, G->dir, "routes.u8"), R_OK) != 0 || access(path_join(a, G->dir, "margins.f32"), R_OK) != 0)
        return "GoldenWithoutRoutes";
    setenv("SAETTA_INK_ROUTES", G->dir, 1);
    Gpu g;
    Err err = gpu_init(a, pack, &g);
    if (err) {
        unsetenv("SAETTA_INK_ROUTES");
        return err;
    }
    Model *m;
    Session *s;
    err = model_load(a, pack, &g, &m);
    unsetenv("SAETTA_INK_ROUTES");
    const ModelInfo *mi = err ? NULL : model_info(m);
    if (!err && (G->layers != mi->layers || G->width != mi->width || G->vocab != mi->vocab)) err = "GoldenShapeMismatch";
    if (!err && !inkling_has_routes(m)) err = "GoldenWithoutRoutes";
    // the exact checks on the f32 prompt chunks; the accelerated ones (tensor ops, half
    // activations) are gated on their own below
    const bool half_av = !err && inkling_half_available(m);
    if (!err) inkling_set_half(m, false);
    // a context that holds the golden (sessions of SAETTA_CTX tokens otherwise)
    const size_t capv = MAX((size_t)SESSION_CTX_DEFAULT, ALIGN_UP(G->ntok + 64, (size_t)SESSION_CTX_ALIGN));
    if (!err) err = session_create_cap(m, &g, false, capv, capv, &s);
    if (err) {
        gpu_deinit(&g);
        return err;
    }
    const size_t L = G->layers, W = G->width, V = G->vocab, N = G->ntok, P = G->npos;
    fprintf(out, "model: %s, pack %s, %zu tokens, %zu dumped positions\n", mi->name, pack_dir, N, P);
    char b1[64], b2[64], b3[64], b4[64];
    uint32_t *tokens = ARENA_ARRAY(a, uint32_t, N);
    for (size_t t = 0; t < N; t++) tokens[t] = (uint32_t)G->toks->arr.items[t]->i;
    size_t *pidx = ARENA_ARRAY(a, size_t, N); // position -> dump index, or npos
    for (size_t t = 0; t < N; t++) pidx[t] = P;
    for (size_t i = 0; i < P; i++) pidx[(size_t)G->poss->arr.items[i]->i] = i;
    const float *last = G->ls + (P - 1) * V;
    const size_t want_last = (size_t)top2_argmax(G->top2, N - 1);
    bool ok = true;
    float *derr = ARENA_ARRAY(a, float, P * L), *pw = ARENA_ARRAY(a, float, L);
    sx_buffer *tb = NULL;
    InkDec dr;
    InkRouteStats rs;

    // goldens past INK_DECODE_MAX tokens decode only their last 32 (after a chunked prefill)
    const size_t from = N > INK_DECODE_MAX ? N - 32 : 0, ND = N - from;
    if (from) fprintf(out, "decode runs: positions %zu to %zu after a prefill of the rest\n", from, N - 1);

    // on the oracle's experts
    inkling_set_routes(m, INK_ROUTES_REPLAY);
    err = ink_decode(a, out, m, s, G, tokens, pidx, from, derr, &dr);
    if (!err) {
        fprintf(out, "decode, the oracle's experts: visit  max rel err (hidden states, %zu positions)\n", P);
        for (size_t li = 0; li < L; li++) {
            float w = 0;
            for (size_t i = 0; i < P; i++) w = fmaxf(w, derr[i * L + li]);
            const bool bad = w > TOL_LAYER_REL;
            ok = ok && !bad;
            fprintf(out, "%5zu  %s%s\n", li, e3(b1, w), bad ? "  FAIL" : "");
        }
        const bool lbad = dr.worst_logits > TOL_LOGITS_REL;
        ok = ok && !lbad && dr.abad == 0;
        fprintf(out, "decode, the oracle's experts: logits max rel err %s%s; greedy argmax %zu equal, %zu near ties skipped, %zu different%s\n",
                e3(b1, dr.worst_logits), lbad ? "  FAIL" : "", dr.aok, dr.askip, dr.abad, dr.abad ? "  FAIL" : "");
        ok = ink_routes_ok(out, m, "decode routers", &rs) && ok;
    }
    float *plog = NULL;
    // the trace of a piece: INK_PIECE tokens, or a whole prompt chunk (the accelerated checks)
    const size_t tb_rows = half_av ? MAX((size_t)INK_PIECE, inkling_prompt_chunk()) : (size_t)INK_PIECE;
    if (!err) err = gpu_new_buffer(&g, L * tb_rows * W * 4, &tb);
    static const char *const chunks[2] = {NULL, "200"};
    for (int k = 0; k < 2 && !err; k++) {
        if (chunks[k]) setenv("SAETTA_INK_PCH", chunks[k], 1);
        err = ink_prefill(m, s, tb, G, tokens, pidx, pw, &plog);
        unsetenv("SAETTA_INK_PCH");
        if (err) break;
        float pmax = 0;
        for (size_t li = 0; li < L; li++) pmax = fmaxf(pmax, pw[li]);
        const float lr = rel(plog, last, V);
        const bool eq = argmax_scan(plog, V) == want_last;
        bool pok = lr <= TOL_LOGITS_REL && eq && pmax <= TOL_LAYER_REL;
        ok = ok && pok;
        fprintf(out, "prefill (%s-token chunks), the oracle's experts: worst visit rel err %s, last-token logits %s, argmax %s%s\n",
                chunks[k] ? chunks[k] : "prompt", e3(b1, pmax), e3(b2, lr), eq ? "equal" : "DIFFERENT", pok ? "" : "  FAIL");
        for (size_t li = 0; li < L; li++)
            if (pw[li] > TOL_LAYER_REL) fprintf(out, "  visit %zu: %s  FAIL\n", li, e3(b1, pw[li]));
        ok = ink_routes_ok(out, m, "prefill routers", &rs) && ok;
    }

    // the default f16 KV cache, on the oracle's experts: greedy argmax and KL like the other
    // models' f16-KV checks (TOL_PREFILL_FAST_KL)
    Session *sh = NULL;
    if (!err) err = session_create_cap(m, &g, true, capv, capv, &sh);
    if (!err) err = ink_decode(a, out, m, sh, G, tokens, pidx, from, derr, &dr);
    if (!err) {
        (void)inkling_route_stats(m);
        float w = 0;
        for (size_t i = 0; i < P * L; i++) w = fmaxf(w, derr[i]);
        const bool hok = dr.abad == 0 && dr.worst_kl <= TOL_PREFILL_FAST_KL && (double)dr.askip <= TOL_INK_ARGMAX_SKIP_FRAC * (double)ND;
        ok = ok && hok;
        fprintf(out, "decode, f16 KV, the oracle's experts: worst visit rel err %s (not gated), max KL(ref||engine) %s; greedy argmax %zu equal, %zu near ties skipped, %zu different%s\n",
                e3(b1, w), e3d(b2, dr.worst_kl), dr.aok, dr.askip, dr.abad, hok ? "" : "  FAIL");
    }
    if (!err) err = ink_prefill(m, sh, tb, G, tokens, pidx, pw, &plog);
    if (!err) {
        (void)inkling_route_stats(m);
        const double k = kl(last, plog, V);
        const bool eq = argmax_scan(plog, V) == want_last, pok = k <= TOL_PREFILL_FAST_KL && eq;
        ok = ok && pok;
        fprintf(out, "prefill (chunks), f16 KV, the oracle's experts: last-token KL(ref||engine) %s, argmax %s%s\n", e3d(b1, k),
                eq ? "equal" : "DIFFERENT", pok ? "" : "  FAIL");
    }

    // on its own experts
    inkling_set_routes(m, INK_ROUTES_OWN);
    if (!err) err = ink_decode(a, out, m, s, G, tokens, pidx, from, derr, &dr);
    if (!err) {
        rs = inkling_route_stats(m);
        float before = 0, after = 0;
        size_t nbefore = 0;
        for (size_t i = 0; i < P; i++) {
            const size_t pos = (size_t)G->poss->arr.items[i]->i;
            if (pos < from) continue;
            for (size_t li = 0; li < L; li++) {
                if (pos < rs.first_flip) before = fmaxf(before, derr[i * L + li]);
                else after = fmaxf(after, derr[i * L + li]);
            }
            nbefore += pos < rs.first_flip;
        }
        // after its first route that differs from the oracle's, the engine's own continuation may
        // differ (one flipped expert moves every later token): its argmax there is gated by
        // agreement, before it exactly
        const bool skip_bad = (double)dr.askip > TOL_INK_ARGMAX_SKIP_FRAC * (double)ND;
        const bool early = dr.first_bad < MIN((size_t)rs.first_flip, N);
        const size_t nt = dr.aok + dr.abad;
        const double agree = nt ? (double)dr.aok / (double)nt : 1.0;
        // a long golden decodes only its tail (32 tokens after a prefill on the engine's own
        // routing, which left the oracle's path long before): reported, not gated; its own-routing
        // checks are the prefill's last-token KL below and the self-consistency test
        // (src/test/inkling_long.c)
        const bool gated = from == 0;
        const bool dok = !gated || (before <= TOL_INK_OWN_LAYER_REL && !early && agree >= TOL_INK_OWN_ARGMAX_AGREE && !skip_bad &&
                                    dr.worst_kl <= TOL_PREFILL_FAST_KL);
        ok = ok && dok;
        if (rs.first_flip == UINT32_MAX) snprintf(b3, sizeof b3, "none");
        else snprintf(b3, sizeof b3, "position %u", rs.first_flip);
        fprintf(out,
                "decode, its own experts: first route different from the oracle's: %s; worst visit rel err %s at the %zu dumped "
                "positions before it, %s after it (not gated); max KL(ref||engine) %s; greedy argmax %zu equal, %zu near ties "
                "skipped, %zu different (%s before the first different route)%s%s\n",
                b3, e3(b1, before), nbefore, e3(b2, after), e3d(b4, dr.worst_kl), dr.aok, dr.askip, dr.abad, early ? "some" : "none",
                gated ? "" : " (the tail of a long golden: not gated)", dok ? "" : "  FAIL");
    }
    if (!err) err = ink_prefill(m, s, tb, G, tokens, pidx, pw, &plog);
    if (!err) {
        (void)inkling_route_stats(m);
        float pmax = 0;
        for (size_t li = 0; li < L; li++) pmax = fmaxf(pmax, pw[li]);
        const double k = kl(last, plog, V);
        const bool eq = argmax_scan(plog, V) == want_last, pok = k <= TOL_PREFILL_FAST_KL && eq;
        ok = ok && pok;
        fprintf(out, "prefill (chunks), its own experts: worst visit rel err %s (not gated), last-token KL(ref||engine) %s, argmax %s%s\n",
                e3(b1, pmax), e3d(b2, k), eq ? "equal" : "DIFFERENT", pok ? "" : "  FAIL");
    }
    // the accelerated prompt chunks (the default where the tensor ops run), in whole prompt chunks
    // as served (a chunk wider than the window writes the ring), gated as the other models'
    // accelerated prefill: last-token KL and argmax; on the oracle's experts also the per-visit
    // bound for gross errors (TOL_PREFILL_FAST_LAYER_REL) and the share of (visit, token) pairs
    // whose router differs from the oracle's (half activations move the scores by more than the
    // exact paths' route margins: bounded by TOL_INK_HALF_ROUTE_FRAC, not by them). On the f16 KV
    // cache (the default) with both routings, and once on an f32 cache (SAETTA_KV_F32: the
    // exact attention tiles after half projections).
    if (!err && half_av) {
        inkling_set_half(m, true);
        const size_t piece = inkling_prompt_chunk();
        static const struct { bool own, f32kv; } runs[3] = {{false, false}, {true, false}, {false, true}};
        for (int r = 0; r < 3 && !err; r++) {
            const bool own = runs[r].own;
            inkling_set_routes(m, own ? INK_ROUTES_OWN : INK_ROUTES_REPLAY);
            err = ink_prefill_in(m, runs[r].f32kv ? s : sh, tb, G, tokens, pidx, piece, pw, &plog);
            if (err) break;
            const InkRouteStats hs = inkling_route_stats(m);
            float pmax = 0;
            for (size_t li = 0; li < L; li++) pmax = fmaxf(pmax, pw[li]);
            const double k = kl(last, plog, V);
            const bool eq = argmax_scan(plog, V) == want_last;
            const uint32_t flips = hs.flips_sure + hs.flips_near;
            const double frac = (double)flips / ((double)L * (double)N);
            const bool pok = k <= TOL_PREFILL_FAST_KL && eq && (own || (pmax <= TOL_PREFILL_FAST_LAYER_REL && frac <= TOL_INK_HALF_ROUTE_FRAC));
            ok = ok && pok;
            fprintf(out,
                    "prefill (%zu-token chunks), accelerated (tensor ops, half activations, %s KV), %s: worst visit rel err %s%s, "
                    "last-token KL(ref||engine) %s, argmax %s; routers differ from the oracle's at %u (visit, token) pairs (%.3f%%%s)%s\n",
                    piece, runs[r].f32kv ? "f32" : "f16", own ? "its own experts" : "the oracle's experts", e3(b1, pmax),
                    own ? " (not gated)" : "", e3d(b2, k), eq ? "equal" : "DIFFERENT", flips, 100.0 * frac, own ? ", not gated" : "",
                    pok ? "" : "  FAIL");
            if (!own && getenv("SAETTA_VERIFY_LAYERS"))
                for (size_t li = 0; li < L; li++) fprintf(out, "  accel visit %2zu: %s\n", li, e3(b1, pw[li]));
        }
        inkling_set_half(m, false);
    } else if (!err)
        fprintf(out, "prefill, accelerated: not available on this GPU (tensor ops), not checked\n");
    if (tb) sx_buffer_release(tb);

    // a verification chunk: every position's logits from one chunk
    if (!err) {
        session_reset(s);
        const size_t nv = MIN(N, (size_t)KMAX);
        float *vl;
        err = model_verify_chunk(m, s, tokens, nv, &vl);
        if (!err) {
            size_t vbad = 0;
            double vkl = 0;
            for (size_t t = 0; t < nv; t++) {
                vbad += argmax_scan(vl + t * V, V) != (size_t)top2_argmax(G->top2, t);
                if (pidx[t] < P) vkl = fmax(vkl, kl(G->ls + pidx[t] * V, vl + t * V, V));
            }
            const bool vok = vbad == 0 && vkl <= TOL_PREFILL_FAST_KL;
            ok = ok && vok;
            fprintf(out, "verify chunk (%zu tokens, all logits): greedy %zu/%zu equal, max KL %s%s\n", nv, nv - vbad, nv, e3d(b1, vkl),
                    vok ? "" : "  FAIL");
        }
    }
    verify_fmt_e(b1, sizeof b1, TOL_LAYER_REL, true, -1);
    verify_fmt_e(b2, sizeof b2, TOL_LOGITS_REL, true, -1);
    verify_fmt_e(b3, sizeof b3, TOL_PREFILL_FAST_KL, false, -1);
    if (!err)
        fprintf(out, "verify: %s (tolerances: visit %s, logits %s on the oracle's experts; KL %s on the engine's own)\n",
                ok ? "PASS" : "FAIL", b1, b2, b3);
    gpu_deinit(&g);
    if (err) return err;
    *okp = ok;
    return OK;
}

Err verify_run(Arena *a, FILE *out, const char *pack_dir, const char *golden, bool *okp) {
    JValue *man;
    TRY(read_json(a, golden, "manifest.json", &man));
    const JValue *toks = json_get(man, "tokens");
    const JValue *poss = json_get(man, "positions");
    size_t ntok = toks->arr.len, npos = poss->arr.len;
    size_t layers = (size_t)json_get(man, "layers")->i;
    size_t width = (size_t)json_get(man, "width")->i;
    // goldens of hyper-connection models give the width of one stream and their count
    const JValue *hcm = json_get(man, "hc_mult");
    if (hcm && hcm->kind == J_INTEGER) width *= (size_t)hcm->i;
    size_t vocab = (size_t)json_get(man, "vocab")->i;
    // checked against the model once it is loaded
    const float *xs, *ls;
    TRY(read_f32(a, golden, "x_layers.f32", npos * layers * width, &xs));
    TRY(read_f32(a, golden, "logits.f32", npos * vocab, &ls));
    JValue *top2;
    TRY(read_json(a, golden, "top2.json", &top2));

    Pack pack;
    TRY(pack_open(a, pack_dir, &pack));
    const ModelInfo *pre = model_info_for(a, &pack);
    if (pre && strcmp(pre->model_type, "deepseek_v4") == 0) {
        DsGolden dg = {golden, toks, poss, top2, xs, ls, ntok, npos, layers, width, vocab};
        Err e = verify_dsv4(a, out, &pack, pack_dir, &dg, okp);
        pack_close(&pack);
        return e;
    }
    if (pre && strcmp(pre->model_type, "inkling2_loop_text") == 0) {
        DsGolden dg = {golden, toks, poss, top2, xs, ls, ntok, npos, layers, width, vocab};
        Err e = verify_inkling(a, out, &pack, pack_dir, &dg, okp);
        pack_close(&pack);
        return e;
    }
    Gpu g;
    Err err = gpu_init(a, &pack, &g);
    if (err) {
        pack_close(&pack);
        return err;
    }
    char b1[64], b2[64], b3[64];
#define FAIL_CLOSE(expr)                 \
    do {                                 \
        Err e_ = (expr);                 \
        if (e_) {                        \
            gpu_deinit(&g);              \
            pack_close(&pack);           \
            return e_;                   \
        }                                \
    } while (0)
    Model *m;
    FAIL_CLOSE(model_load(a, &pack, &g, &m));
    const ModelInfo *mi = model_info(m);
    if (layers != mi->layers || width != mi->width || vocab != mi->vocab) FAIL_CLOSE("GoldenShapeMismatch");
    fprintf(out, "model: %s, pack %s\n", mi->name, pack_dir);
    // a golden made with a LoRA adapter (manifest "lora"): the same adapter, from the
    // manifest's path
    const JValue *lora = json_get(man, "lora");
    if (lora && lora->kind == J_OBJECT) {
        const char *dir = json_str(json_get(lora, "path"));
        if (!dir) FAIL_CLOSE("GoldenLoraPath");
        if (dir[0] != '/') dir = path_join(a, golden, dir); // relative to the golden
        FAIL_CLOSE(model_lora_load(a, m, "golden", dir));
        if (model_lora_count(m) != 1) FAIL_CLOSE("GoldenLoraNotLoaded");
        model_lora_select(m, 0);
        fprintf(out, "lora: %s\n", dir);
    }
    Session *s;
    FAIL_CLOSE(session_create_kv(m, &g, false, &s)); // exact checks: f32 KV cache
    float *trace = ARENA_ARRAY(a, float, layers * width);

    float *worst_layer = ARENA_ARRAY(a, float, layers);
    float worst_logits = 0;
    size_t argmax_ok = 0, argmax_skip = 0, argmax_bad = 0;
    for (size_t t = 0; t < ntok; t++) {
        bool has_pi = false;
        size_t pi = 0;
        for (size_t i = 0; i < npos; i++)
            if ((size_t)poss->arr.items[i]->i == t) {
                pi = i;
                has_pi = true;
            }
        s->trace = has_pi ? trace : NULL;
        s->trace_len = has_pi ? layers * width : 0;
        float *logits;
        FAIL_CLOSE(model_step(m, s, (uint32_t)toks->arr.items[t]->i, &logits));
        if (has_pi) {
            for (size_t li = 0; li < layers; li++)
                worst_layer[li] = fmaxf(worst_layer[li], rel(trace + li * width, xs + (pi * layers + li) * width, width));
            worst_logits = fmaxf(worst_logits, rel(logits, ls + pi * vocab, vocab));
        }
        size_t best = argmax_scan(logits, vocab);
        const JValue *ref = top2->arr.items[t];
        size_t want = (size_t)json_get(ref, "argmax")->i;
        double margin = jf(json_get(ref, "top1")) - jf(json_get(ref, "top2"));
        if (best == want)
            argmax_ok++;
        else if (margin < TOL_ARGMAX_MARGIN)
            argmax_skip++;
        else {
            argmax_bad++;
            fmt_fixed(b1, sizeof b1, margin, 4);
            fprintf(out, "argmax differs at position %zu: engine %zu, reference %zu (margin %s)\n", t, best, want, b1);
        }
    }
    // prefill path: the whole prompt as one chunk, exact f32 first, then the accelerator path
    if (ntok > CH) FAIL_CLOSE("GoldenLongerThanChunk");
    bool fast_available = model_fast_prefill(m);
    model_set_fast_prefill(m, false);
    session_reset(s);
    uint32_t *tokens = ARENA_ARRAY(a, uint32_t, ntok);
    for (size_t t = 0; t < ntok; t++) tokens[t] = (uint32_t)toks->arr.items[t]->i;
    sx_buffer *tb;
    FAIL_CLOSE(gpu_new_buffer(&g, layers * ntok * width * 4, &tb));
    float *plog;
    FAIL_CLOSE(model_prefill(m, s, tokens, ntok, tb, &plog));
    const float *tr = gpu_floats(tb);
    float *pworst = ARENA_ARRAY(a, float, layers);
    for (size_t i = 0; i < npos; i++) {
        size_t t = (size_t)poss->arr.items[i]->i;
        for (size_t li = 0; li < layers; li++)
            pworst[li] = fmaxf(pworst[li], rel(tr + (li * ntok + t) * width, xs + (i * layers + li) * width, width));
    }
    float plogit = rel(plog, ls + (npos - 1) * vocab, vocab);
    size_t pbest = argmax_scan(plog, vocab);
    size_t plast = (size_t)top2_argmax(top2, ntok - 1);

    bool ok = true;
    float pmax = pworst[0];
    for (size_t li = 1; li < layers; li++) pmax = fmaxf(pmax, pworst[li]);
    fprintf(out, "prefill (one chunk of %zu): worst layer rel err %s, last-token logits %s, argmax %s\n", ntok, e3(b1, pmax),
            e3(b2, plogit), pbest == plast ? "equal" : "DIFFERENT");
    for (size_t li = 0; li < layers; li++)
        if (pworst[li] > TOL_LAYER_REL) {
            ok = false;
            fprintf(out, "  prefill layer %zu: %s  FAIL\n", li, e3(b1, pworst[li]));
        }
    if (plogit > TOL_LOGITS_REL || pbest != plast) ok = false;

    // default prefill on the f16 KV cache (the accelerator half-input matmuls when they load, else
    // the plain path and its prompt attention): quality gate on KL of the last-token distribution
    {
        model_set_fast_prefill(m, true); // stays off without the accelerator kernels
        const char *fpath = fast_available ? "accelerator half-input matmuls" : "plain path, f32 matmuls";
        Session *sf;
        FAIL_CLOSE(session_create_kv(m, &g, true, &sf)); // the default: f16 KV cache
        float *flog;
        FAIL_CLOSE(model_prefill(m, sf, tokens, ntok, tb, &flog));
        float fworst = 0;
        float *flayer = ARENA_ARRAY(a, float, layers);
        for (size_t i = 0; i < npos; i++) {
            size_t t = (size_t)poss->arr.items[i]->i;
            for (size_t li = 0; li < layers; li++)
                flayer[li] = fmaxf(flayer[li], rel(tr + (li * ntok + t) * width, xs + (i * layers + li) * width, width));
        }
        for (size_t li = 0; li < layers; li++) fworst = fmaxf(fworst, flayer[li]);
        if (getenv("SAETTA_VERIFY_LAYERS") != NULL)
            for (size_t li = 0; li < layers; li++) fprintf(out, "  accel layer %2zu: %s\n", li, e3(b1, flayer[li]));
        double fkl = kl(ls + (npos - 1) * vocab, flog, vocab);
        size_t fbest = argmax_scan(flog, vocab);
        bool fok = fkl <= TOL_PREFILL_FAST_KL && fbest == plast && fworst <= TOL_PREFILL_FAST_LAYER_REL;
        ok = ok && fok;
        fprintf(out,
                "prefill, default (%s, f16 KV): worst layer rel err %s, last-token KL(ref||engine) %s, argmax %s%s\n",
                fpath, e3(b1, fworst), e3d(b2, fkl), fbest == plast ? "equal" : "DIFFERENT", fok ? "" : "  FAIL");
        // default decode: step by step on an f16-KV session, every greedy token and the last KL
        model_set_fast_prefill(m, fast_available);
        Session *sd;
        FAIL_CLOSE(session_create_kv(m, &g, true, &sd));
        size_t dbad = 0;
        float *dlog = NULL;
        for (size_t t = 0; t < ntok; t++) {
            FAIL_CLOSE(model_step(m, sd, tokens[t], &dlog));
            size_t bi = argmax_scan(dlog, vocab);
            if (bi != (size_t)top2_argmax(top2, t)) dbad++;
        }
        double dkl = kl(ls + (npos - 1) * vocab, dlog, vocab);
        bool dok = dbad == 0 && dkl <= TOL_PREFILL_FAST_KL;
        ok = ok && dok;
        fprintf(out, "decode, default (f16 KV): greedy %zu/%zu equal, last-token KL %s%s\n", ntok - dbad, ntok,
                e3d(b1, dkl), dok ? "" : "  FAIL");
        // speculative verification chunk: logits of every position from one chunk
        Session *sv;
        FAIL_CLOSE(session_create_kv(m, &g, true, &sv));
        size_t nv = MIN(ntok, (size_t)KMAX);
        float *vl;
        FAIL_CLOSE(model_verify_chunk(m, sv, tokens, nv, &vl));
        size_t vbad = 0;
        for (size_t t = 0; t < nv; t++) {
            size_t bi = argmax_scan(vl + t * vocab, vocab);
            if (bi != (size_t)top2_argmax(top2, t)) vbad++;
        }
        double vkl = 0;
        for (size_t i = 0; i < npos; i++) {
            size_t t = (size_t)poss->arr.items[i]->i;
            if (t < nv) vkl = fmax(vkl, kl(ls + i * vocab, vl + t * vocab, vocab));
        }
        bool vok = vbad == 0 && vkl <= TOL_PREFILL_FAST_KL;
        ok = ok && vok;
        fprintf(out, "verify chunk (%zu tokens, all logits): greedy %zu/%zu equal, max KL %s%s\n", nv, nv - vbad, nv,
                e3d(b1, vkl), vok ? "" : "  FAIL");
    }
    fprintf(out, "decode:\nlayer  max rel err (residual streams, %zu positions)\n", npos);
    for (size_t li = 0; li < layers; li++) {
        bool bad = worst_layer[li] > TOL_LAYER_REL;
        ok = ok && !bad;
        fprintf(out, "%5zu  %s%s\n", li, e3(b1, worst_layer[li]), bad ? "  FAIL" : "");
    }
    bool lbad = worst_logits > TOL_LOGITS_REL;
    ok = ok && !lbad && argmax_bad == 0;
    fprintf(out, "logits max rel err %s%s\n", e3(b1, worst_logits), lbad ? "  FAIL" : "");
    fprintf(out, "greedy argmax: %zu equal, %zu near ties skipped, %zu different\n", argmax_ok, argmax_skip, argmax_bad);
    verify_fmt_e(b1, sizeof b1, TOL_LAYER_REL, true, -1);
    verify_fmt_e(b2, sizeof b2, TOL_LOGITS_REL, true, -1);
    fmt_f64(b3, sizeof b3, TOL_ARGMAX_MARGIN);
    fprintf(out, "verify: %s (tolerances: layer %s, logits %s, tie margin %s)\n", ok ? "PASS" : "FAIL", b1, b2, b3);
#undef FAIL_CLOSE
    gpu_deinit(&g);
    pack_close(&pack);
    *okp = ok;
    return OK;
}
