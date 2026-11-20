// Golden-data check of the engine (see verify.h).
#include "verify.h"

#include <math.h>
#include <stdlib.h>

#include "gpu.h"
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
    TRY(read_file(path_join(a, dir, name), (size_t)1 << 20, &bytes, &len));
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

    // accelerator prefill (half inputs): quality gate on KL of the last-token distribution
    if (fast_available) {
        model_set_fast_prefill(m, true);
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
                "prefill, default (accelerator half-input matmuls, f16 KV): worst layer rel err %s, last-token "
                "KL(ref||engine) %s, argmax %s%s\n",
                e3(b1, fworst), e3d(b2, fkl), fbest == plast ? "equal" : "DIFFERENT", fok ? "" : "  FAIL");
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
    } else
        fprintf(out, "prefill, accelerators: not available, skipped\n");
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
