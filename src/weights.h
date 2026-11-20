// The model pack: an MLX safetensors directory (model.safetensors.index.json plus
// shards), mapped read-only and never copied. Quantized linear layers are
// affine: w = q * scale + bias per group (bits and group size from the pack's config.json).
#ifndef SAETTA_WEIGHTS_H
#define SAETTA_WEIGHTS_H

#include "base.h"
#include "safetensors.h"

#define WEIGHTS_PREFIX "language_model.model."

typedef struct {
    const char *name;     // shard file name as in the index (NUL terminated)
    const uint8_t *bytes; // the whole mapped file
    size_t len;
} MappedFile;

typedef struct {
    TensorInfo weight, scales, biases;
    uint32_t bits, group_size;
    uint64_t out_features, in_features;
    // Streamed routed experts (xcache.h): 1 + the cache region of weight (scales and biases
    // follow it); 0 when the tensors are read from the pack.
    uint8_t xr;
} Linear;

typedef struct {
    uint32_t bits, group_size;
} Quant;

typedef struct {
    const char *dir;
    MappedFile *files; // index i is TensorInfo.file == i
    size_t nfiles;
    StrMap tensors;    // name -> TensorInfo * (count: tensors.len)
    // Per-module quantization from the pack's config.json ("quantization": default bits and
    // group_size plus one entry per module that differs): module name -> Quant *. Empty when
    // the pack has none, and then the compiled-in table (Flash-Next's, config_flash.h) applies. Packs mix 4/5/6/8
    // bits and groups of 32/64/128 (the Coder conversion).
    StrMap quant;
    bool has_quant_default;
    Quant quant_default;
} Pack;
// A zero Pack ((Pack){0}) is an empty pack with no files (gpu_init on it gives a bare device).

// Maps every shard listed in the index and parses its header. Errors: "MapFailed",
// "BadIndex", JSON parse errors, safetensors_parse errors.
Err pack_open(Arena *a, const char *dir, Pack *out);
void pack_close(Pack *p);

// Quantization of module: its config.json entry, else the pack default, else the
// compiled-in table (weights_quant_for).
Quant pack_quant_of(const Pack *p, Str module);

// The tensor called name, or NULL.
const TensorInfo *pack_get(const Pack *p, Str name);

// Bytes of a tensor inside its mapped file (t->nbytes long). t->file must be a pack file.
static inline const uint8_t *pack_data(const Pack *p, const TensorInfo *t) {
    return p->files[t->file].bytes + t->offset;
}

// Resolves <module>.weight/.scales/.biases and checks their shapes against the quantization
// the config declares for that module. Errors: "MissingTensor", "BadQuantShape",
// "NoSpaceLeft" (name over 1 KB).
Err pack_linear(const Pack *p, Str module, Linear *out);

// Compiled-in quantization: the first config_quant_overrides suffix module ends with, else
// the default bits and group size.
Quant weights_quant_for(Str module);

// MLX affine quantization (bits 4 with group 32, or 8 with group 64) of a BF16 matrix
// [out][in] (row-major), written into dst at byte at: weight u32 [out][in*bits/32], then
// scales and biases bf16 [out][in/group], each 16-byte aligned. Returns the Linear (tensors in
// GPU "file" file) and the bytes used. Per group: scale = (max - min) / (2^bits - 1) and
// bias = min, both rounded to bf16 first, q = round((w - bias) / scale) clamped. Used for the
// MTP head, whose dense matrices the pack keeps in BF16: drafts are verified, so this changes
// acceptance, never output. group must be at most 64.
typedef struct {
    Linear lin;
    uint64_t used;
} Quantized;
Quantized weights_quantize(const uint8_t *src, uint64_t out, uint64_t in, uint32_t bits, uint32_t group,
                           uint8_t *dst, uint64_t at, uint16_t file);

// f32 to bf16, round to nearest even (as weights_quantize rounds scales and biases).
static inline uint16_t weights_to_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

#endif
