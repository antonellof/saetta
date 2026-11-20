// DeepSeek-V4-Flash: the pack check and the loader's expert layout. The forward pass is not
// written yet; every entry that would run the model returns "NotImplemented".
//
// The official checkpoint is read in place: dense weights FP8 e4m3 with e8m0 128 x 128 block
// scales (plus BF16 and F32 tensors), routed experts FP4 e2m1 (two per I8 byte) with an e8m0
// scale per 32 values. One tensor per expert matrix; an expert's three weights lie back to
// back (w1, w2, w3) and so do its three scales, so the expert cache reads an expert as two
// ranges.
#include "model.h"

#include "config_dsv4.h"
#define ARCH_PFX dsv4_
#include "model_rename.h"
#include "weights.h"
#include "xcache.h"

#define H CONFIG_HIDDEN_SIZE
#define NLAYERS CONFIG_NUM_HIDDEN_LAYERS
#define VOCAB CONFIG_VOCAB_SIZE
#define NE CONFIG_N_ROUTED_EXPERTS
#define FF CONFIG_MOE_INTERMEDIATE_SIZE
#define HC CONFIG_HC_MULT

static const uint32_t dsv4_eos[1] = {(uint32_t)CONFIG_EOS_TOKEN_ID};

// ---- pack check -------------------------------------------------------------------------

typedef struct {
    const char *name; // after "layers.N." (or the full name at top level)
    DType dtype;
    uint64_t d0, d1; // shape; d1 0 for rank 1
} Want;

static size_t check_one(const Pack *pack, const char *name, DType dt, uint64_t d0, uint64_t d1, FILE *out) {
    const TensorInfo *t = pack_get(pack, str(name));
    if (!t) {
        fprintf(out, "  MISSING %s\n", name);
        return 1;
    }
    bool rank_ok = d1 ? t->rank == 2 && t->shape[1] == d1 : t->rank == 1;
    if (t->dtype != dt || !rank_ok || t->shape[0] != d0) {
        fprintf(out, "  MISMATCH %s: %s rank %u [%llu, %llu], want %s [%llu, %llu]\n", name, dtype_name(t->dtype),
                t->rank, (unsigned long long)t->shape[0], (unsigned long long)(t->rank > 1 ? t->shape[1] : 0),
                dtype_name(dt), (unsigned long long)d0, (unsigned long long)d1);
        return 1;
    }
    return 0;
}

// Expert e of layer l: its two ranges (weights w1 w2 w3, scales w1 w2 w3), or false with a
// message when the six tensors are missing or not back to back in one file.
static bool expert_ranges(Arena *a, const Pack *pack, size_t l, size_t e, XcRange r[2], FILE *out) {
    static const char *const mat[3] = {"w1", "w2", "w3"};
    static const uint64_t wlen[3] = {FF * H / 2, H * FF / 2, FF * H / 2};
    static const uint64_t slen[3] = {FF * H / 32, H * FF / 32, FF * H / 32};
    for (int k = 0; k < 2; k++) {
        uint64_t next = 0;
        uint16_t file = 0;
        for (int i = 0; i < 3; i++) {
            const char *nm = arena_printf(a, "layers.%zu.ffn.experts.%zu.%s.%s", l, e, mat[i], k ? "scale" : "weight");
            const TensorInfo *t = pack_get(pack, str(nm));
            uint64_t want = k ? slen[i] : wlen[i];
            if (!t || t->nbytes != want || t->dtype != (k ? DTYPE_F8_E8M0 : DTYPE_I8)) {
                if (out) fprintf(out, "  EXPERT %s: missing or wrong size\n", nm);
                return false;
            }
            if (i == 0) {
                r[k].off = t->offset;
                file = t->file;
            } else if (t->offset != next || t->file != file) {
                if (out) fprintf(out, "  EXPERT layers.%zu.ffn.experts.%zu: %s not contiguous\n", l, e, k ? "scales" : "weights");
                return false;
            }
            next = t->offset + t->nbytes;
        }
        r[k].len = next - r[k].off;
        r[k].file = file;
    }
    return true;
}

size_t model_check_pack(Arena *a, const Pack *pack, FILE *out) {
    const DType B = DTYPE_BF16, F = DTYPE_F32, Q = DTYPE_F8_E4M3, S = DTYPE_F8_E8M0;
    size_t bad = 0, n = 0;
    const Want top[] = {
        {"embed.weight", B, VOCAB, H}, {"head.weight", B, VOCAB, H}, {"norm.weight", B, H, 0},
        {"hc_head_fn", F, HC, HC * H}, {"hc_head_base", F, HC, 0},  {"hc_head_scale", F, 1, 0},
    };
    for (size_t i = 0; i < COUNT(top); i++, n++) bad += check_one(pack, top[i].name, top[i].dtype, top[i].d0, top[i].d1, out);
    const size_t QL = CONFIG_Q_LORA_RANK, NH = CONFIG_NUM_ATTENTION_HEADS, HD = CONFIG_HEAD_DIM, OL = CONFIG_O_LORA_RANK;
    const size_t OG = CONFIG_O_GROUPS, IH = CONFIG_INDEX_N_HEADS, ID = CONFIG_INDEX_HEAD_DIM, MIX = (2 + HC) * HC;
    const Want every[] = {
        {"attn.wq_a.weight", Q, QL, H},          {"attn.wq_a.scale", S, QL / 128, H / 128},
        {"attn.wq_b.weight", Q, NH * HD, QL},    {"attn.wq_b.scale", S, NH * HD / 128, QL / 128},
        {"attn.wkv.weight", Q, HD, H},           {"attn.wkv.scale", S, HD / 128, H / 128},
        {"attn.wo_a.weight", Q, OG * OL, NH * HD / OG}, {"attn.wo_a.scale", S, OG * OL / 128, NH * HD / OG / 128},
        {"attn.wo_b.weight", Q, H, OG * OL},     {"attn.wo_b.scale", S, H / 128, OG * OL / 128},
        {"attn.q_norm.weight", B, QL, 0},        {"attn.kv_norm.weight", B, HD, 0},
        {"attn.attn_sink", F, NH, 0},            {"attn_norm.weight", B, H, 0},
        {"ffn_norm.weight", B, H, 0},            {"ffn.gate.weight", B, NE, H},
        {"ffn.shared_experts.w1.weight", Q, FF, H}, {"ffn.shared_experts.w1.scale", S, FF / 128, H / 128},
        {"ffn.shared_experts.w3.weight", Q, FF, H}, {"ffn.shared_experts.w3.scale", S, FF / 128, H / 128},
        {"ffn.shared_experts.w2.weight", Q, H, FF}, {"ffn.shared_experts.w2.scale", S, H / 128, FF / 128},
        {"hc_attn_fn", F, MIX, HC * H},          {"hc_attn_base", F, MIX, 0}, {"hc_attn_scale", F, 3, 0},
        {"hc_ffn_fn", F, MIX, HC * H},           {"hc_ffn_base", F, MIX, 0},  {"hc_ffn_scale", F, 3, 0},
    };
    const Want csa[] = {
        {"attn.compressor.wkv.weight", B, 2 * HD, H}, {"attn.compressor.wgate.weight", B, 2 * HD, H},
        {"attn.compressor.ape", F, 4, 2 * HD},        {"attn.compressor.norm.weight", B, HD, 0},
        {"attn.indexer.compressor.wkv.weight", B, 2 * ID, H}, {"attn.indexer.compressor.wgate.weight", B, 2 * ID, H},
        {"attn.indexer.compressor.ape", F, 4, 2 * ID},        {"attn.indexer.compressor.norm.weight", B, ID, 0},
        {"attn.indexer.wq_b.weight", Q, IH * ID, QL},         {"attn.indexer.wq_b.scale", S, IH * ID / 128, QL / 128},
        {"attn.indexer.weights_proj.weight", B, IH, H},
    };
    const Want hca[] = {
        {"attn.compressor.wkv.weight", B, HD, H}, {"attn.compressor.wgate.weight", B, HD, H},
        {"attn.compressor.ape", F, 128, HD},      {"attn.compressor.norm.weight", B, HD, 0},
    };
    for (size_t l = 0; l < NLAYERS; l++) {
        const uint32_t ratio = config_compress_ratios[l];
        const Want *sets[2] = {every, ratio == 4 ? csa : ratio == 128 ? hca : NULL};
        size_t counts[2] = {COUNT(every), ratio == 4 ? COUNT(csa) : ratio == 128 ? COUNT(hca) : 0};
        for (int s = 0; s < 2; s++)
            for (size_t i = 0; i < counts[s]; i++, n++)
                bad += check_one(pack, arena_printf(a, "layers.%zu.%s", l, sets[s][i].name), sets[s][i].dtype, sets[s][i].d0,
                                 sets[s][i].d1, out);
        n++;
        if (l < CONFIG_NUM_HASH_LAYERS)
            bad += check_one(pack, arena_printf(a, "layers.%zu.ffn.gate.tid2eid", l), DTYPE_I64, VOCAB, CONFIG_NUM_EXPERTS_PER_TOK, out);
        else
            bad += check_one(pack, arena_printf(a, "layers.%zu.ffn.gate.bias", l), F, NE, 0, out);
        for (size_t e = 0; e < NE; e++, n++) {
            XcRange r[2];
            if (!expert_ranges(a, pack, l, e, r, out)) bad++;
        }
    }
    fprintf(out, "tensors and experts: %zu checked, %zu problems\n", n, bad);
    return bad;
}

// ---- load --------------------------------------------------------------------------------

Err model_load(Arena *a, const Pack *pack, Gpu *g, Model **out) {
    (void)g;
    (void)out;
    FILE *devnull = fopen("/dev/null", "w");
    size_t bad = model_check_pack(a, pack, devnull ? devnull : stderr);
    if (devnull) fclose(devnull);
    if (bad) {
        (void)model_check_pack(a, pack, stderr);
        return "BadPack";
    }
    // the expert cache layout: two ranges per expert (weights, scales), validated above
    XcLayer *xl = ARENA_ARRAY(a, XcLayer, NLAYERS);
    if (!xl) return "OutOfMemory";
    for (size_t l = 0; l < NLAYERS; l++) {
        XcRange *r = ARENA_ARRAY(a, XcRange, NE * 2);
        if (!r) return "OutOfMemory";
        for (size_t e = 0; e < NE; e++)
            if (!expert_ranges(a, pack, l, e, &r[e * 2], NULL)) return "BadPack";
        xl[l] = (XcLayer){.ne = (uint32_t)NE, .ranges = r};
    }
    dbg("model: DeepSeek-V4-Flash pack checked (%zu layers, %zu experts each); the forward pass is not implemented yet\n",
        (size_t)NLAYERS, (size_t)NE);
    return "NotImplemented";
}

// ---- not implemented yet -------------------------------------------------------------------

#define NI "NotImplemented"
bool model_has_mtp(const Model *m) { (void)m; return false; }
bool model_fast_prefill(const Model *m) { (void)m; return false; }
bool model_fast_prefill_available(const Model *m) { (void)m; return false; }
void model_set_fast_prefill(Model *m, bool on) { (void)m, (void)on; }
Err model_step(Model *m, Session *s, uint32_t token, float **logits) { (void)m, (void)s, (void)token, (void)logits; return NI; }
Err model_prefill(Model *m, Session *s, const uint32_t *tokens, size_t n, sx_buffer *trace, float **logits) {
    (void)m, (void)s, (void)tokens, (void)n, (void)trace, (void)logits;
    return NI;
}
Err model_prefill_with(Model *m, Session *s, const uint32_t *tokens, size_t n, sx_buffer *trace, const PrefillHook *hook,
                       float **logits) {
    (void)m, (void)s, (void)tokens, (void)n, (void)trace, (void)hook, (void)logits;
    return NI;
}
Err model_verify_chunk(Model *m, Session *s, const uint32_t *toks, size_t n, float **logits) {
    (void)m, (void)s, (void)toks, (void)n, (void)logits;
    return NI;
}
Err model_spec_greedy(Model *m, Session *s, uint32_t x, const uint32_t *drafts, size_t nd, SpecResult *out) {
    (void)m, (void)s, (void)x, (void)drafts, (void)nd, (void)out;
    return NI;
}
Err model_spec_verify(Model *m, Session *s, uint32_t x, const uint32_t *drafts, size_t nd, SpecPick pick, void *ctx,
                      SpecResult *out) {
    (void)m, (void)s, (void)x, (void)drafts, (void)nd, (void)pick, (void)ctx, (void)out;
    return NI;
}
Err model_draft_mtp(Model *m, Session *s, uint32_t x, size_t k, uint32_t *out, size_t *nd) {
    (void)m, (void)s, (void)x, (void)k, (void)out, (void)nd;
    return NI;
}
Err model_lora_load(Arena *a, Model *m, const char *name, const char *dir) { (void)a, (void)m, (void)name, (void)dir; return NI; }
int model_lora_find(const Model *m, const char *name) { (void)m, (void)name; return -1; }
void model_lora_select(Model *m, int idx) { (void)m, (void)idx; }
int model_lora_selected(const Model *m) { (void)m; return -1; }
size_t model_lora_count(const Model *m) { (void)m; return 0; }
const char *model_lora_name(const Model *m, size_t i) { (void)m, (void)i; return NULL; }
void model_prof_init(void) {}
void model_prof_reset(void) {}
void model_prof_report(size_t tokens) { (void)tokens; }
Err session_create(Gpu *g, Session **out) { (void)g, (void)out; return NI; }
Err session_create_kv(Gpu *g, bool kv_half, Session **out) { (void)g, (void)kv_half, (void)out; return NI; }
Err session_create_cap(Gpu *g, bool kv_half, size_t cap, size_t max_cap, Session **out) {
    (void)g, (void)kv_half, (void)cap, (void)max_cap, (void)out;
    return NI;
}
size_t session_kv_elem(const Session *s) { (void)s; return 0; }
size_t session_kv_bytes(const Session *s, size_t cap) { (void)s, (void)cap; return 0; }
Err session_grow(Session *s, Gpu *g, size_t cap) { (void)s, (void)g, (void)cap; return NI; }
Err session_save_checkpoint(Session *s, const uint32_t *toks, size_t n) { (void)s, (void)toks, (void)n; return NI; }
Err session_save_checkpoint_at(Session *s, size_t slot, const uint32_t *toks, size_t n) {
    (void)s, (void)slot, (void)toks, (void)n;
    return NI;
}
size_t session_reusable(Session *s, const uint32_t *prompt, size_t n) { (void)s, (void)prompt, (void)n; return 0; }
size_t session_conv_bytes(const Session *s, size_t cap) { (void)s, (void)cap; return 0; }
Err session_new_conv(const Session *s, Gpu *g, size_t cap, Conv *out) { (void)s, (void)g, (void)cap, (void)out; return NI; }
Err session_conv_shrink(const Session *s, Gpu *g, Conv *cv, size_t cap) { (void)s, (void)g, (void)cv, (void)cap; return NI; }
void session_swap_conv(Session *s, Conv *cv) { (void)s, (void)cv; }
bool session_roll_to(Session *s, size_t p, const uint32_t *toks) { (void)s, (void)p, (void)toks; return false; }
Err session_save_point(Session *s, Gpu *g, const uint32_t *toks, size_t n) { (void)s, (void)g, (void)toks, (void)n; return NI; }
void session_start_at_own_point(Session *s, size_t n) { (void)s, (void)n; }
Err session_start_from(Session *s, Gpu *g, const Conv *src, size_t n) { (void)s, (void)g, (void)src, (void)n; return NI; }
void session_restore(Session *s) { (void)s; }
void session_push_history(Session *s, uint32_t token) { (void)s, (void)token; }
void session_reset(Session *s) { (void)s; }

const ModelArch dsv4_arch = MODEL_ARCH_TABLE({
    .model_type = CONFIG_MODEL_TYPE,
    .id = "deepseek-v4-flash",
    .name = "DeepSeek-V4-Flash",
    .layers = NLAYERS,
    .width = HC * H,
    .vocab = VOCAB,
    .hidden = H,
    .eos = dsv4_eos,
    .n_eos = 1,
});
