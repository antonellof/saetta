// Saetta Code's configuration files: pi's settings.json and models.json in the agent folder,
// changed in place (the other keys and providers kept), and what the footer calls a model pack.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at
// v1.5.2, b4afddd: packages/coding-agent/src/core/settings-manager.ts, model-config.ts,
// resolve-config-value.ts).
#ifndef SAETTA_CODE_CFG_H
#define SAETTA_CODE_CFG_H

#include "base.h"
#include "json.h"

// v as JSON.stringify(v, null, 2) writes it (two-space indent, "{}" and "[]" when empty).
void cfg_json_pretty(Buf *b, const JValue *v, int depth);

// The edit applied to a file's root object; false leaves the file as it is.
typedef bool (*CfgEditFn)(Arena *a, JValue *root, void *ctx);

// pi's locked read-modify-write (settings-manager.ts withLock): under a "<path>.lock" folder
// (proper-lockfile's, stale after 10 s) the file is read again, edited, and written whole through
// a temporary file and a rename. A missing or empty file is an empty object. A file that cannot
// be read or does not parse as an object is never written: false, and why says so. Numbers
// keep their literal text.
bool cfg_json_update(const char *path, CfgEditFn edit, void *ctx, Buf *why);

// settings.json: defaultProvider and defaultModel set, every other key kept.
bool cfg_set_default_model(const char *agent_dir, const char *provider, const char *model, Buf *why);

// models.json: providers[name] = entry (where old_name was when renaming, else in place or at the
// end); entry NULL removes name. The other providers and the other top-level keys are kept.
bool cfg_put_provider(const char *agent_dir, const char *name, const char *old_name, const JValue *entry, Buf *why);

// models.json as it is now (NULL when missing or not an object).
JValue *cfg_read_models(Arena *a, const char *agent_dir);

// An OpenAI-compatible provider entry in pi's format: baseUrl, api "openai-completions", apiKey,
// models [{id, name}]. old (may be NULL): the entry being edited, whose other fields and whose
// models' other fields (for the ids kept) stay. api_key empty: "none" (pi lists a provider only
// with an API key configured; a local server ignores it). ids: comma or space separated.
JValue *cfg_openai_provider(Arena *a, const JValue *old, const char *base_url, const char *api_key, const char *ids);

// pi's resolveConfigValue: "!command" runs the command (its trimmed stdout, cached), otherwise
// "$NAME" / "${NAME}" take the environment variable ("$$" and "$!" escape), the rest literal.
// NULL when a variable is unset or the command fails.
const char *cfg_resolve_value(Arena *a, const char *config);

// The footer's name for a pack: title (the catalog's, a trailing " 4-bit" / " 8-bit" dropped)
// and the quantization, from the GGUF file name it was imported from (Q4_K_M, Q8_0) or else the
// pack's config.json "quantization" ("4-bit", "4-bit (mixed 4/5/6/8)"). *full: everything,
// *brief: without the "(mixed ...)" part; NULL when there is nothing to add to the title.
void cfg_pack_label(Arena *a, const char *title, const char *gguf_file, const char *pack_dir, const char **full, const char **brief);

#endif
