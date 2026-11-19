// Safetensors header parsing over a memory-mapped file.
// Layout: u64 little-endian header length, JSON header, then raw tensor data.
// The JSON maps tensor name -> {dtype, shape, data_offsets: [begin, end]},
// offsets relative to the end of the header.
#ifndef SAETTA_SAFETENSORS_H
#define SAETTA_SAFETENSORS_H

#include "base.h"

// Order and names as the Zig enum (dtype_name gives the tag name, e.g. "bf16").
typedef enum { DTYPE_BF16, DTYPE_F16, DTYPE_F32, DTYPE_U32, DTYPE_I64, DTYPE_U8, DTYPE_I32 } DType;
#define DTYPE_COUNT 7

// "BF16", "F16", "F32", "U32", "I64", "U8", "I32"; false for any other string.
bool dtype_from_string(Str s, DType *out);
size_t dtype_size(DType d);   // bytes per element
const char *dtype_name(DType d); // lowercase tag name: "bf16", "f16", ...

#define SAFETENSORS_MAX_RANK 5 // the vision patch embedding is rank 5

typedef struct {
    DType dtype;
    uint8_t rank;
    uint64_t shape[SAFETENSORS_MAX_RANK]; // unused dims are 1 (0 in quantized tensors built at load)
    uint64_t offset; // absolute byte offset of the tensor data within the file
    uint64_t nbytes;
    uint16_t file;   // index of the file in the pack
} TensorInfo;

// Product of the first rank dims.
static inline uint64_t tensor_numel(const TensorInfo *t) {
    uint64_t n = 1;
    for (uint8_t i = 0; i < t->rank; i++) n *= t->shape[i];
    return n;
}

// Parses the header of a safetensors file mapped at bytes[0..len), adding every tensor to
// out (name -> TensorInfo *, both allocated in the arena; an existing name is replaced).
// Errors: "TooShort", "BadHeader", "BadTensor", "UnknownDType", "SizeMismatch".
Err safetensors_parse(Arena *a, const uint8_t *bytes, size_t len, uint16_t file_index, StrMap *out);

#endif
