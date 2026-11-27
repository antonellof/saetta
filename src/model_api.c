// The model API (model.h) over the models in this binary: model_load picks Flash-Next,
// Qwen3.8-27B, DeepSeek-V4-Flash or Inkling 2 (registered, not run yet) by the pack's
// config.json model_type, and every call goes to that model's ModelArch (src/model_arch.h).
// Also the functions the models share (they depend on no model constant): model_argmax,
// restore-point and prefix-point matching.
#include "model.h"

#include <stdlib.h>
#include <string.h>

#include "fit.h"
#include "json.h"
#include "model_arch.h"
#include "xcache.h"

static const ModelArch *const archs[] = {&flash_arch, &dense_arch, &dsv4_arch, &inkling_arch};

// Model starts with its ModelArch pointer in both implementations.
static const ModelArch *arch_of(const Model *m) { return *(const ModelArch *const *)m; }

// The text model_type of the pack (config.json text_config.model_type, else model_type).
static const char *pack_model_type(Arena *a, const Pack *pack) {
    char *text;
    size_t len;
    if (read_file(path_join(a, pack->dir, "config.json"), 4u << 20, &text, &len)) return NULL;
    JValue *root;
    Err e = json_parse(a, str_n(text, len), false, &root);
    free(text);
    if (e) return NULL;
    const JValue *tc = json_get(root, "text_config");
    const JValue *mt = tc ? json_get(tc, "model_type") : NULL;
    if (!mt) mt = json_get(root, "model_type");
    return mt && mt->kind == J_STRING ? arena_strndup(a, mt->s.ptr, mt->s.len) : NULL;
}

static const ModelArch *arch_for(Arena *a, const Pack *pack) {
    const char *mt = pack_model_type(a, pack);
    if (!mt) return NULL;
    for (size_t i = 0; i < COUNT(archs); i++)
        if (!strcmp(mt, archs[i]->info.model_type)) return archs[i];
    return NULL;
}

Err model_load(Arena *a, const Pack *pack, Gpu *g, Model **out) {
    const ModelArch *A = arch_for(a, pack);
    if (!A) {
        dbg("model: %s is not a Qwen3.8-Flash-Next, Qwen3.8-27B, DeepSeek-V4-Flash or Inkling 2 pack\n", pack->dir);
        return "UnknownModelType";
    }
    return A->load(a, pack, g, out);
}

Err model_fit_check(Arena *a, const Pack *pack) {
    const ModelArch *A = pack->nfiles ? arch_for(a, pack) : NULL;
    if (!A) return OK;
    // packs that stream their experts from the SSD are larger than memory by design: DeepSeek-V4-Flash,
    // and Flash-Next with an expert cache (SAETTA_EXPERT_CACHE_GB above 0, as xcache.c reads it)
    if (strstr(A->info.model_type, "deepseek") != NULL || xc_requested_bytes() > 0) return OK;
    return fit_check_pack(pack);
}

Err session_ctx_parse(const char *text, size_t max, size_t *out) {
    if (!text) {
        *out = SESSION_CTX_DEFAULT;
        return OK;
    }
    uint64_t x;
    TRY(parse_u64(str(text), &x));
    if (x > max) return "ContextTooLarge";
    x -= x % SESSION_CTX_ALIGN;
    if (x < SESSION_CTX_MIN) return "ContextTooSmall";
    *out = (size_t)x;
    return OK;
}

const ModelInfo *model_info(const Model *m) { return &arch_of(m)->info; }
const ModelInfo *model_info_for(Arena *a, const Pack *pack) {
    const ModelArch *A = arch_for(a, pack);
    return A ? &A->info : NULL;
}
const ModelInfo *model_info_dir(Arena *a, const char *dir) {
    Pack p = {.dir = dir};
    return model_info_for(a, &p);
}
size_t model_check_pack(Arena *a, const Pack *pack, FILE *out) {
    const ModelArch *A = arch_for(a, pack);
    return A ? A->check_pack(a, pack, out) : 1;
}
bool model_has_mtp(const Model *m) { return arch_of(m)->has_mtp(m); }
bool model_fast_prefill(const Model *m) { return arch_of(m)->fast_prefill(m); }
bool model_fast_prefill_available(const Model *m) { return arch_of(m)->fast_prefill_available(m); }
void model_set_fast_prefill(Model *m, bool on) { arch_of(m)->set_fast_prefill(m, on); }

Err model_step(Model *m, Session *s, uint32_t token, float **logits) { return arch_of(m)->step(m, s, token, logits); }
Err model_prefill(Model *m, Session *s, const uint32_t *tokens, size_t n, sx_buffer *trace, float **logits) {
    return arch_of(m)->prefill(m, s, tokens, n, trace, logits);
}
Err model_prefill_with(Model *m, Session *s, const uint32_t *tokens, size_t n, sx_buffer *trace, const PrefillHook *hook,
                       float **logits) {
    return arch_of(m)->prefill_with(m, s, tokens, n, trace, hook, logits);
}
Err model_verify_chunk(Model *m, Session *s, const uint32_t *toks, size_t n, float **logits) {
    return arch_of(m)->verify_chunk(m, s, toks, n, logits);
}
Err model_spec_greedy(Model *m, Session *s, uint32_t x, const uint32_t *drafts, size_t nd, SpecResult *out) {
    return arch_of(m)->spec_greedy(m, s, x, drafts, nd, out);
}
Err model_spec_verify(Model *m, Session *s, uint32_t x, const uint32_t *drafts, size_t nd, SpecPick pick, void *ctx,
                      SpecResult *out) {
    return arch_of(m)->spec_verify(m, s, x, drafts, nd, pick, ctx, out);
}
Err model_draft_mtp(Model *m, Session *s, uint32_t x, size_t k, uint32_t *out, size_t *nd) {
    return arch_of(m)->draft_mtp(m, s, x, k, out, nd);
}
Err model_lora_load(Arena *a, Model *m, const char *name, const char *dir) { return arch_of(m)->lora_load(a, m, name, dir); }
int model_lora_find(const Model *m, const char *name) { return arch_of(m)->lora_find(m, name); }
void model_lora_select(Model *m, int idx) { arch_of(m)->lora_select(m, idx); }
int model_lora_selected(const Model *m) { return arch_of(m)->lora_selected(m); }
size_t model_lora_count(const Model *m) { return arch_of(m)->lora_count(m); }
const char *model_lora_name(const Model *m, size_t i) { return arch_of(m)->lora_name(m, i); }

// The profiler is process-wide: both models keep their own tables (only the loaded one counts).
void model_prof_init(void) {
    for (size_t i = 0; i < COUNT(archs); i++) archs[i]->prof_init();
}
void model_prof_reset(void) {
    for (size_t i = 0; i < COUNT(archs); i++) archs[i]->prof_reset();
}
void model_prof_report(size_t tokens) {
    for (size_t i = 0; i < COUNT(archs); i++) archs[i]->prof_report(tokens);
}

Err session_create(const Model *m, Gpu *g, Session **out) { return arch_of(m)->s_create(g, out); }
Err session_create_kv(const Model *m, Gpu *g, bool kv_half, Session **out) { return arch_of(m)->s_create_kv(g, kv_half, out); }
Err session_create_cap(const Model *m, Gpu *g, bool kv_half, size_t cap, size_t max_cap, Session **out) {
    return arch_of(m)->s_create_cap(g, kv_half, cap, max_cap, out);
}
size_t session_kv_elem(const Session *s) { return s->arch->s_kv_elem(s); }
size_t session_kv_bytes(const Session *s, size_t cap) { return s->arch->s_kv_bytes(s, cap); }
Err session_grow(Session *s, Gpu *g, size_t cap) { return s->arch->s_grow(s, g, cap); }
Err session_save_checkpoint(Session *s, const uint32_t *toks, size_t n) { return s->arch->s_save_checkpoint(s, toks, n); }
Err session_save_checkpoint_at(Session *s, size_t slot, const uint32_t *toks, size_t n) {
    return s->arch->s_save_checkpoint_at(s, slot, toks, n);
}
size_t session_reusable(Session *s, const uint32_t *prompt, size_t n) { return s->arch->s_reusable(s, prompt, n); }
size_t session_conv_bytes(const Session *s, size_t cap) { return s->arch->s_conv_bytes(s, cap); }
Err session_new_conv(const Session *s, Gpu *g, size_t cap, Conv *out) { return s->arch->s_new_conv(s, g, cap, out); }
Err session_conv_shrink(const Session *s, Gpu *g, Conv *cv, size_t cap) { return s->arch->s_conv_shrink(s, g, cv, cap); }
void session_swap_conv(Session *s, Conv *cv) { s->arch->s_swap_conv(s, cv); }
bool session_roll_to(Session *s, size_t p, const uint32_t *toks) { return s->arch->s_roll_to(s, p, toks); }
Err session_save_point(Session *s, Gpu *g, const uint32_t *toks, size_t n, bool may_alloc) {
    return s->arch->s_save_point(s, g, toks, n, may_alloc);
}
void session_start_at_own_point(Session *s, size_t n) { s->arch->s_start_at_own_point(s, n); }
Err session_start_from(Session *s, Gpu *g, const Conv *src, size_t n, size_t alloc_max) {
    return s->arch->s_start_from(s, g, src, n, alloc_max);
}
void session_restore(Session *s) { s->arch->s_restore(s); }
void session_push_history(Session *s, uint32_t token) { s->arch->s_push_history(s, token); }
void session_reset(Session *s) { s->arch->s_reset(s); }
bool session_state_layout(const Session *s, size_t slot, size_t n, StLayout *out) { return s->arch->s_layout(s, slot, n, out); }

// ---- shared by both models ---------------------------------------------------------------------

// Index of the first largest value (as a scalar `x > best` scan): the maximum in one vector
// pass, then its first position (the scalar loop took 0.38 ms per 248K-logit row, three rows
// per speculative pass).
uint32_t model_argmax(const float *v, size_t n) {
    enum { L = 16 };
    typedef float V __attribute__((ext_vector_type(L)));
    if (n < L) {
        size_t best = 0;
        for (size_t i = 0; i < n; i++)
            if (v[i] > v[best]) best = i;
        return (uint32_t)best;
    }
    V mx;
    memcpy(&mx, v, sizeof mx);
    size_t i = L;
    for (; i + L <= n; i += L) {
        V x;
        memcpy(&x, v + i, sizeof x);
        mx = __builtin_elementwise_max(mx, x);
    }
    float m = __builtin_reduce_max(mx);
    for (; i < n; i++) m = __builtin_fmaxf(m, v[i]);
    V mv = m;
    i = 0;
    for (; i + L <= n; i += L) {
        V x;
        memcpy(&x, v + i, sizeof x);
        if (__builtin_reduce_or(x == mv)) {
            for (size_t j = 0; j < L; j++)
                if (v[i + j] == m) return (uint32_t)(i + j);
        }
    }
    for (; i < n; i++)
        if (v[i] == m) return (uint32_t)i;
    return 0;
}

Reusable session_reusable_in(const Ckpt ck[2], const uint32_t *prompt, size_t np) {
    size_t best = 0, pick = 0;
    for (size_t i = 0; i < 2; i++) {
        size_t n = ck[i].tokens.len;
        if (n == 0 || n >= np || n <= best) continue;
        if (ck[i].has_next && prompt[n] != ck[i].next) continue;
        if (memcmp(ck[i].tokens.ptr, prompt, n * sizeof(uint32_t)) == 0) {
            best = n;
            pick = i;
        }
    }
    return (Reusable){best, pick};
}

void session_keep_points(Session *s, size_t n) {
    s->ptoks.len = MIN(s->ptoks.len, n);
    for (size_t k = 0; k < PT_MAX; k++)
        if (s->pts[k].pos > s->ptoks.len) s->pts[k].valid = false;
}

static bool point_ok(const Point *p) { return p->gdn != NULL && p->valid && p->pos > 0; }

size_t session_shared_point(const uint32_t *ptoks, size_t nptoks, const Point pts[PT_MAX], const uint32_t *prompt,
                            size_t n) {
    if (n == 0) return 0;
    size_t lcp = 0;
    size_t lim = MIN(nptoks, n);
    while (lcp < lim && ptoks[lcp] == prompt[lcp]) lcp += 1;
    // pos < lcp: the prompt's token at pos is the one the checkpoint's MTP row at pos was written
    // with (and pos < n: a token is left to run)
    size_t best = 0;
    for (size_t k = 0; k < PT_MAX; k++)
        if (point_ok(&pts[k]) && pts[k].pos < lcp && pts[k].pos > best) best = pts[k].pos;
    return best;
}

int session_point_at(const Point pts[PT_MAX], size_t pos) {
    for (int k = 0; k < PT_MAX; k++)
        if (point_ok(&pts[k]) && pts[k].pos == pos) return k;
    return -1;
}

int session_point_slot(const Point pts[PT_MAX], size_t pos, bool may_alloc) {
    int at = session_point_at(pts, pos);
    if (at >= 0) return at;
    for (int k = 0; k < PT_MAX; k++)
        if (pts[k].gdn != NULL && !point_ok(&pts[k])) return k;
    if (may_alloc)
        for (int k = 0; k < PT_MAX; k++)
            if (pts[k].gdn == NULL) return k;
    // thinning: the valid points by position
    int idx[PT_MAX];
    size_t m = 0;
    for (int k = 0; k < PT_MAX; k++)
        if (point_ok(&pts[k])) idx[m++] = k;
    if (m == 0) return -1;
    for (size_t i = 1; i < m; i++)
        for (size_t j = i; j > 0 && pts[idx[j - 1]].pos > pts[idx[j]].pos; j--) {
            int t = idx[j];
            idx[j] = idx[j - 1];
            idx[j - 1] = t;
        }
    // the oldest and the PT_RECENT most recent stay (fewer kept when there are fewer points)
    size_t lo = m > 1 ? 1 : 0, hi = m > PT_RECENT + 1 ? m - PT_RECENT : m;
    if (hi <= lo) hi = m;
    size_t pick = lo;
    double best = 0;
    for (size_t i = lo; i < hi; i++) {
        size_t p = pts[idx[i]].pos, prev = i > 0 ? pts[idx[i - 1]].pos : 0, next = i + 1 < m ? pts[idx[i + 1]].pos : pos;
        double score = (double)(next - prev) / (double)(pos > p ? pos - p : 1);
        if (i == lo || score < best) {
            best = score;
            pick = i;
        }
    }
    return idx[pick];
}

uint64_t session_points_reserve(const Session *s, size_t convs) {
    return (uint64_t)(PT_MAX + (convs > 1 ? convs - 1 : 0) * PT_HELD) * session_point_bytes(s);
}

size_t session_conv_trim_points(Conv *cv, size_t keep) {
    bool stay[PT_MAX] = {false};
    size_t kept = 0;
    // pass 0: the most recent; 1: the most recent request start; 2: the oldest; 3+: the most recent
    for (size_t pass = 0; kept < keep; pass++) {
        int best = -1;
        for (int i = 0; i < PT_MAX; i++) {
            const Point *p = &cv->pts[i];
            if (stay[i] || !point_ok(p) || (pass == 1 && !p->start)) continue;
            bool better = best < 0 || (pass == 2 ? p->pos < cv->pts[best].pos : p->pos > cv->pts[best].pos);
            if (better) best = i;
        }
        if (best >= 0) {
            stay[best] = true;
            kept++;
        } else if (pass >= 3) break;
    }
    size_t released = 0;
    for (int i = 0; i < PT_MAX; i++) {
        Point *p = &cv->pts[i];
        if (stay[i] || p->gdn == NULL) continue;
        sx_buffer_release(p->gdn);
        if (p->conv) sx_buffer_release(p->conv);
        if (p->ple) sx_buffer_release(p->ple);
        *p = (Point){0};
        released++;
    }
    return released;
}

size_t session_point_bytes(const Session *s) { return (session_conv_bytes(s, s->cap) - session_kv_bytes(s, s->cap)) / 2; }

size_t session_point_cuts(const uint32_t *ids, size_t n, size_t pos0, uint32_t im_start, size_t *out, size_t cap) {
    if (n < 2 || pos0 + 1 >= n - 1) return 0;
    size_t end = n - 1, k = 0, last = pos0;
    // each one PT_GAP after the previous one taken and PT_GAP before the end: no chunk shorter than
    // PT_GAP for a checkpoint (the request's own start keeps one without a cut, server.c)
    for (size_t p = pos0 + 1; p < end && k < cap; p++) {
        if (!(p % PT_EVERY == 0 || (im_start != UINT32_MAX && ids[p] == im_start))) continue;
        if (p - last >= PT_GAP && end - p >= PT_GAP) {
            out[k++] = p;
            last = p;
        }
    }
    return k;
}

size_t prefill_chunk_len(const PrefillHook *hook, size_t pos0, size_t n, size_t i) {
    // the segment between the cuts around offset i: [a, b)
    size_t a = 0, b = n;
    for (size_t k = 0; hook && k < hook->ncuts; k++) {
        size_t c = hook->cuts[k];
        if (c <= pos0 || c >= pos0 + n) continue;
        c -= pos0;
        if (c <= i) a = c;
        else {
            b = c;
            break;
        }
    }
    size_t len = b - a, chunks = (len + CH - 1) / CH;
    size_t per = (len + chunks - 1) / MAX(chunks, (size_t)1);
    return MIN(per, b - i);
}

void session_invalidate_from(Session *s, size_t n) {
    for (int i = 0; i < 2; i++)
        if (s->ck[i].tokens.len > n) s->ck[i].tokens.len = 0;
    session_keep_points(s, n);
}
