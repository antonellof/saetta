// Inkling 2 (HF thinkingmachines/Inkling-2-Mini b85c3cf, config.json model_type inkling2_loop):
// registration only, so its host parts (tokenizer, chat template, tool calls) and then the
// model can land step by step. The pack is recognized by its text model_type; loading it
// reports that the model is not supported yet, and every other entry point refuses.
// Constants: src/config_inkling.h (tools/gen_config.py).
#include "model.h"

#include "config_inkling.h"
#define ARCH_PFX inkling_
#include "model_rename.h"

#define NS "NotSupportedYet"

Err model_load(Arena *a, const Pack *pack, Gpu *g, Model **out) {
    (void)a, (void)g, (void)out;
    dbg("model: %s is an Inkling 2 pack, which this saetta does not run yet\n", pack->dir);
    return NS;
}

size_t model_check_pack(Arena *a, const Pack *pack, FILE *out) {
    (void)a, (void)pack;
    fprintf(out, "  Inkling 2: not supported yet\n");
    return 1;
}

bool model_has_mtp(const Model *m) { (void)m; return false; }
bool model_fast_prefill(const Model *m) { (void)m; return false; }
bool model_fast_prefill_available(const Model *m) { (void)m; return false; }
void model_set_fast_prefill(Model *m, bool on) { (void)m, (void)on; }
Err model_step(Model *m, Session *s, uint32_t token, float **logits) { (void)m, (void)s, (void)token, (void)logits; return NS; }
Err model_prefill(Model *m, Session *s, const uint32_t *tokens, size_t n, sx_buffer *trace, float **logits) {
    (void)m, (void)s, (void)tokens, (void)n, (void)trace, (void)logits;
    return NS;
}
Err model_prefill_with(Model *m, Session *s, const uint32_t *tokens, size_t n, sx_buffer *trace, const PrefillHook *hook,
                       float **logits) {
    (void)m, (void)s, (void)tokens, (void)n, (void)trace, (void)hook, (void)logits;
    return NS;
}
Err model_verify_chunk(Model *m, Session *s, const uint32_t *toks, size_t n, float **logits) {
    (void)m, (void)s, (void)toks, (void)n, (void)logits;
    return NS;
}
Err model_spec_greedy(Model *m, Session *s, uint32_t x, const uint32_t *drafts, size_t nd, SpecResult *out) {
    (void)m, (void)s, (void)x, (void)drafts, (void)nd, (void)out;
    return NS;
}
Err model_spec_verify(Model *m, Session *s, uint32_t x, const uint32_t *drafts, size_t nd, SpecPick pick, void *ctx,
                      SpecResult *out) {
    (void)m, (void)s, (void)x, (void)drafts, (void)nd, (void)pick, (void)ctx, (void)out;
    return NS;
}
Err model_draft_mtp(Model *m, Session *s, uint32_t x, size_t k, uint32_t *out, size_t *nd) {
    (void)m, (void)s, (void)x, (void)k, (void)out;
    *nd = 0;
    return NS;
}
Err model_lora_load(Arena *a, Model *m, const char *name, const char *dir) { (void)a, (void)m, (void)name, (void)dir; return NS; }
int model_lora_find(const Model *m, const char *name) { (void)m, (void)name; return -1; }
void model_lora_select(Model *m, int idx) { (void)m, (void)idx; }
int model_lora_selected(const Model *m) { (void)m; return -1; }
size_t model_lora_count(const Model *m) { (void)m; return 0; }
const char *model_lora_name(const Model *m, size_t i) { (void)m, (void)i; return NULL; }
void model_prof_init(void) {}
void model_prof_reset(void) {}
void model_prof_report(size_t tokens) { (void)tokens; }
Err session_create(Gpu *g, Session **out) { (void)g, (void)out; return NS; }
Err session_create_kv(Gpu *g, bool kv_half, Session **out) { (void)g, (void)kv_half, (void)out; return NS; }
Err session_create_cap(Gpu *g, bool kv_half, size_t cap, size_t max_cap, Session **out) {
    (void)g, (void)kv_half, (void)cap, (void)max_cap, (void)out;
    return NS;
}
size_t session_kv_elem(const Session *s) { (void)s; return 2; }
size_t session_kv_bytes(const Session *s, size_t cap) { (void)s, (void)cap; return 0; }
Err session_grow(Session *s, Gpu *g, size_t cap) { (void)s, (void)g, (void)cap; return NS; }
Err session_save_checkpoint(Session *s, const uint32_t *toks, size_t n) { (void)s, (void)toks, (void)n; return NS; }
Err session_save_checkpoint_at(Session *s, size_t slot, const uint32_t *toks, size_t n) {
    (void)s, (void)slot, (void)toks, (void)n;
    return NS;
}
size_t session_reusable(Session *s, const uint32_t *prompt, size_t n) { (void)s, (void)prompt, (void)n; return 0; }
size_t session_conv_bytes(const Session *s, size_t cap) { (void)s, (void)cap; return 0; }
Err session_new_conv(const Session *s, Gpu *g, size_t cap, Conv *out) { (void)s, (void)g, (void)cap, (void)out; return NS; }
Err session_conv_shrink(const Session *s, Gpu *g, Conv *cv, size_t cap) { (void)s, (void)g, (void)cv, (void)cap; return NS; }
void session_swap_conv(Session *s, Conv *cv) { (void)s, (void)cv; }
bool session_roll_to(Session *s, size_t p, const uint32_t *toks) { (void)s, (void)p, (void)toks; return false; }
Err session_save_point(Session *s, Gpu *g, const uint32_t *toks, size_t n) { (void)s, (void)g, (void)toks, (void)n; return NS; }
void session_start_at_own_point(Session *s, size_t n) { (void)s, (void)n; }
Err session_start_from(Session *s, Gpu *g, const Conv *src, size_t n) { (void)s, (void)g, (void)src, (void)n; return NS; }
void session_restore(Session *s) { (void)s; }
void session_push_history(Session *s, uint32_t token) { (void)s, (void)token; }
void session_reset(Session *s) { (void)s; }

const ModelArch inkling_arch = MODEL_ARCH_TABLE({
    .model_type = CONFIG_MODEL_TYPE,
    .id = "inkling-2-mini",
    .name = "Inkling 2 Mini",
    .layers = CONFIG_NUM_HIDDEN_LAYERS,
    .width = CONFIG_HIDDEN_SIZE,
    .vocab = CONFIG_VOCAB_SIZE,
    .hidden = CONFIG_HIDDEN_SIZE,
    .eos = config_eos_token_ids,
    .n_eos = CONFIG_N_EOS_TOKEN_IDS,
    .max_positions = CONFIG_MODEL_MAX_LENGTH,
    .no_spec = true,
    .no_serve = true,
    .chat = CHAT_INKLING,
});
