// GGUF import for Qwen3.8-27B (llama.cpp architecture "qwen35", dense): turns a GGUF file into
// a saetta pack (MLX-layout safetensors), so fine-tunes published only as GGUF run on the 27B
// path unchanged.
//
// What the import undoes (llama.cpp's HF -> GGUF conversion for Qwen3.5): the reorder of the
// linear-attention value heads from grouped to tiled order (in_proj_qkv value rows, in_proj_z,
// in_proj_a/b, A_log, dt_bias, conv1d value channels, out_proj columns), A_log stored as
// -exp(A_log), the squeezed conv1d, the merged attn_qkv_gate / ssm_beta_alpha tensors, the
// reversed dimension order and the names. Norm weights keep llama.cpp's +1, which is the
// MLX packs' convention too.
//
// Quantized blocks map onto MLX affine groups of 32 (w = q * scale + bias):
//   Q8_0 -> 8-bit (q + 128, bias = -128 d), Q4_K -> 4-bit, Q5_K -> 5-bit (scale = d * sc,
//   bias = -dmin * m), all on the same integer grid; only scale and bias round to bf16.
//   Q6_K (16-value groups, which the kernels do not take) is requantized to 8-bit, group 32.
//   F32 / F16 / BF16 matrices (llama.cpp keeps lm_head in BF16 in some files) go to 8-bit,
//   group 64. Vectors (norms, A_log, dt_bias, conv1d) are written as F32.
// Other types (IQ*, Q2_K, Q3_K, Q4_0, ...) are refused: "GgufUnsupportedType".
#ifndef SAETTA_GGUF_H
#define SAETTA_GGUF_H

#include "base.h"

// Writes <out_dir>/model.safetensors, model.safetensors.index.json and config.json. The
// config is <like_dir>/config.json (the model's Hugging Face config) with the per-module
// quantization added; tokenizer and chat template files are not touched (the caller puts
// them in out_dir). Prints progress on stderr. Errors: "MapFailed", "GgufBadFile",
// "GgufNotQwen35" (another architecture or shape), "GgufUnsupportedType", "GgufMissingTensor",
// write errors.
Err gguf_import(const char *gguf_path, const char *like_dir, const char *out_dir);

// `saetta import <file.gguf> <out dir> [--like <dir with config.json>]` (default: out dir).
Err gguf_import_run(const char *const *args, size_t n);

#endif
