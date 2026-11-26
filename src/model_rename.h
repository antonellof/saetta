// Included by a model's implementation (src/model.c, src/dense.c) after model.h, with
// ARCH_PFX defined (flash_, dense_): the API names it defines become prefixed (flash_model_step,
// ...), declared here with the implementation's signatures, and MODEL_ARCH_TABLE({info initializer}) builds
// its ModelArch from them. The shared functions (model_argmax, session_reusable_in, ...) are
// in src/model_api.c.
#ifndef SAETTA_MODEL_RENAME_H
#define SAETTA_MODEL_RENAME_H

#include "model_arch.h"

#define MR_CAT2(a, b) a##b
#define MR_CAT(a, b) MR_CAT2(a, b)
#define model_load MR_CAT(ARCH_PFX, model_load)
#define model_check_pack MR_CAT(ARCH_PFX, model_check_pack)
#define model_has_mtp MR_CAT(ARCH_PFX, model_has_mtp)
#define model_fast_prefill MR_CAT(ARCH_PFX, model_fast_prefill)
#define model_fast_prefill_available MR_CAT(ARCH_PFX, model_fast_prefill_available)
#define model_set_fast_prefill MR_CAT(ARCH_PFX, model_set_fast_prefill)
#define model_step MR_CAT(ARCH_PFX, model_step)
#define model_prefill MR_CAT(ARCH_PFX, model_prefill)
#define model_prefill_with MR_CAT(ARCH_PFX, model_prefill_with)
#define model_verify_chunk MR_CAT(ARCH_PFX, model_verify_chunk)
#define model_spec_greedy MR_CAT(ARCH_PFX, model_spec_greedy)
#define model_spec_verify MR_CAT(ARCH_PFX, model_spec_verify)
#define model_draft_mtp MR_CAT(ARCH_PFX, model_draft_mtp)
#define model_lora_load MR_CAT(ARCH_PFX, model_lora_load)
#define model_lora_find MR_CAT(ARCH_PFX, model_lora_find)
#define model_lora_select MR_CAT(ARCH_PFX, model_lora_select)
#define model_lora_selected MR_CAT(ARCH_PFX, model_lora_selected)
#define model_lora_count MR_CAT(ARCH_PFX, model_lora_count)
#define model_lora_name MR_CAT(ARCH_PFX, model_lora_name)
#define model_prof_init MR_CAT(ARCH_PFX, model_prof_init)
#define model_prof_reset MR_CAT(ARCH_PFX, model_prof_reset)
#define model_prof_report MR_CAT(ARCH_PFX, model_prof_report)
#define session_create MR_CAT(ARCH_PFX, session_create)
#define session_create_kv MR_CAT(ARCH_PFX, session_create_kv)
#define session_create_cap MR_CAT(ARCH_PFX, session_create_cap)
#define session_kv_elem MR_CAT(ARCH_PFX, session_kv_elem)
#define session_kv_bytes MR_CAT(ARCH_PFX, session_kv_bytes)
#define session_grow MR_CAT(ARCH_PFX, session_grow)
#define session_save_checkpoint MR_CAT(ARCH_PFX, session_save_checkpoint)
#define session_save_checkpoint_at MR_CAT(ARCH_PFX, session_save_checkpoint_at)
#define session_reusable MR_CAT(ARCH_PFX, session_reusable)
#define session_conv_bytes MR_CAT(ARCH_PFX, session_conv_bytes)
#define session_new_conv MR_CAT(ARCH_PFX, session_new_conv)
#define session_conv_shrink MR_CAT(ARCH_PFX, session_conv_shrink)
#define session_swap_conv MR_CAT(ARCH_PFX, session_swap_conv)
#define session_roll_to MR_CAT(ARCH_PFX, session_roll_to)
#define session_save_point MR_CAT(ARCH_PFX, session_save_point)
#define session_start_at_own_point MR_CAT(ARCH_PFX, session_start_at_own_point)
#define session_start_from MR_CAT(ARCH_PFX, session_start_from)
#define session_restore MR_CAT(ARCH_PFX, session_restore)
#define session_push_history MR_CAT(ARCH_PFX, session_push_history)
#define session_reset MR_CAT(ARCH_PFX, session_reset)
#define session_state_layout MR_CAT(ARCH_PFX, session_state_layout)

Err model_load(Arena *a, const Pack *pack, Gpu *g, Model **out);
size_t model_check_pack(Arena *a, const Pack *pack, FILE *out);
bool model_has_mtp(const Model *m);
bool model_fast_prefill(const Model *m);
bool model_fast_prefill_available(const Model *m);
void model_set_fast_prefill(Model *m, bool on);
Err model_step(Model *m, Session *s, uint32_t token, float **logits);
Err model_prefill(Model *m, Session *s, const uint32_t *tokens, size_t n, sx_buffer *trace, float **logits);
Err model_prefill_with(Model *m, Session *s, const uint32_t *tokens, size_t n, sx_buffer *trace, const PrefillHook *hook,
                       float **logits);
Err model_verify_chunk(Model *m, Session *s, const uint32_t *toks, size_t n, float **logits);
Err model_spec_greedy(Model *m, Session *s, uint32_t x, const uint32_t *drafts, size_t nd, SpecResult *out);
Err model_spec_verify(Model *m, Session *s, uint32_t x, const uint32_t *drafts, size_t nd, SpecPick pick, void *ctx,
                      SpecResult *out);
Err model_draft_mtp(Model *m, Session *s, uint32_t x, size_t k, uint32_t *out, size_t *nd);
Err model_lora_load(Arena *a, Model *m, const char *name, const char *dir);
int model_lora_find(const Model *m, const char *name);
void model_lora_select(Model *m, int idx);
int model_lora_selected(const Model *m);
size_t model_lora_count(const Model *m);
const char *model_lora_name(const Model *m, size_t i);
void model_prof_init(void);
void model_prof_reset(void);
void model_prof_report(size_t tokens);
Err session_create(Gpu *g, Session **out);
Err session_create_kv(Gpu *g, bool kv_half, Session **out);
Err session_create_cap(Gpu *g, bool kv_half, size_t cap, size_t max_cap, Session **out);
size_t session_kv_elem(const Session *s);
size_t session_kv_bytes(const Session *s, size_t cap);
Err session_grow(Session *s, Gpu *g, size_t cap);
Err session_save_checkpoint(Session *s, const uint32_t *toks, size_t n);
Err session_save_checkpoint_at(Session *s, size_t slot, const uint32_t *toks, size_t n);
size_t session_reusable(Session *s, const uint32_t *prompt, size_t n);
size_t session_conv_bytes(const Session *s, size_t cap);
Err session_new_conv(const Session *s, Gpu *g, size_t cap, Conv *out);
Err session_conv_shrink(const Session *s, Gpu *g, Conv *cv, size_t cap);
void session_swap_conv(Session *s, Conv *cv);
bool session_roll_to(Session *s, size_t p, const uint32_t *toks);
Err session_save_point(Session *s, Gpu *g, const uint32_t *toks, size_t n);
void session_start_at_own_point(Session *s, size_t n);
Err session_start_from(Session *s, Gpu *g, const Conv *src, size_t n);
void session_restore(Session *s);
void session_push_history(Session *s, uint32_t token);
void session_reset(Session *s);
bool session_state_layout(const Session *s, size_t slot, size_t n, StLayout *out);

#define MODEL_ARCH_TABLE(...) \
    { .info = __VA_ARGS__, \
      .load = model_load, \
      .check_pack = model_check_pack, \
      .has_mtp = model_has_mtp, \
      .fast_prefill = model_fast_prefill, \
      .fast_prefill_available = model_fast_prefill_available, \
      .set_fast_prefill = model_set_fast_prefill, \
      .step = model_step, \
      .prefill = model_prefill, \
      .prefill_with = model_prefill_with, \
      .verify_chunk = model_verify_chunk, \
      .spec_greedy = model_spec_greedy, \
      .spec_verify = model_spec_verify, \
      .draft_mtp = model_draft_mtp, \
      .lora_load = model_lora_load, \
      .lora_find = model_lora_find, \
      .lora_select = model_lora_select, \
      .lora_selected = model_lora_selected, \
      .lora_count = model_lora_count, \
      .lora_name = model_lora_name, \
      .prof_init = model_prof_init, \
      .prof_reset = model_prof_reset, \
      .prof_report = model_prof_report, \
      .s_create = session_create, \
      .s_create_kv = session_create_kv, \
      .s_create_cap = session_create_cap, \
      .s_kv_elem = session_kv_elem, \
      .s_kv_bytes = session_kv_bytes, \
      .s_grow = session_grow, \
      .s_save_checkpoint = session_save_checkpoint, \
      .s_save_checkpoint_at = session_save_checkpoint_at, \
      .s_reusable = session_reusable, \
      .s_conv_bytes = session_conv_bytes, \
      .s_new_conv = session_new_conv, \
      .s_conv_shrink = session_conv_shrink, \
      .s_swap_conv = session_swap_conv, \
      .s_roll_to = session_roll_to, \
      .s_save_point = session_save_point, \
      .s_start_at_own_point = session_start_at_own_point, \
      .s_start_from = session_start_from, \
      .s_restore = session_restore, \
      .s_push_history = session_push_history, \
      .s_reset = session_reset, \
      .s_layout = session_state_layout, \
    }

#endif
