// Safetensors header parsing (see safetensors.h).
#include "safetensors.h"

#include "json.h"

static const char *const dtype_strs[DTYPE_COUNT] = {"BF16", "F16", "F32", "U32", "I64", "U8", "I32", "F8_E4M3", "F8_E8M0", "I8"};
static const char *const dtype_names[DTYPE_COUNT] = {"bf16", "f16", "f32", "u32", "i64", "u8", "i32", "f8_e4m3", "f8_e8m0", "i8"};

bool dtype_from_string(Str s, DType *out) {
    for (int i = 0; i < DTYPE_COUNT; i++)
        if (str_eqz(s, dtype_strs[i])) {
            *out = (DType)i;
            return true;
        }
    return false;
}

size_t dtype_size(DType d) {
    switch (d) {
    case DTYPE_BF16: case DTYPE_F16: return 2;
    case DTYPE_F32: case DTYPE_U32: case DTYPE_I32: return 4;
    case DTYPE_I64: return 8;
    case DTYPE_U8: case DTYPE_F8_E4M3: case DTYPE_F8_E8M0: case DTYPE_I8: return 1;
    }
    return 0;
}

const char *dtype_name(DType d) { return (unsigned)d < DTYPE_COUNT ? dtype_names[d] : "?"; }

static bool as_u64(const JValue *v, uint64_t *out) {
    if (!v || v->kind != J_INTEGER || v->i < 0) return false;
    *out = (uint64_t)v->i;
    return true;
}

Err safetensors_parse(Arena *a, const uint8_t *bytes, size_t len, uint16_t file_index, StrMap *out) {
    if (len < 8) return "TooShort";
    uint64_t hlen;
    memcpy(&hlen, bytes, 8); // little-endian host
    if (hlen > len - 8) return "TooShort";
    uint64_t data_start = 8 + hlen;
    JValue *root;
    if (json_parse(a, str_n((const char *)bytes + 8, hlen), true, &root)) return "BadHeader";
    if (root->kind != J_OBJECT) return "BadHeader";
    for (size_t k = 0; k < root->obj.len; k++) {
        JMember *e = &root->obj.items[k];
        if (str_eqz(e->key, "__metadata__")) continue;
        const JValue *obj = e->val;
        if (obj->kind != J_OBJECT) return "BadTensor";
        const JValue *dt = json_get(obj, "dtype");
        if (!dt || dt->kind != J_STRING) return "BadTensor";
        DType dtype;
        if (!dtype_from_string(dt->s, &dtype)) return "UnknownDType";
        const JValue *shape_v = json_get(obj, "shape");
        if (!shape_v || shape_v->kind != J_ARRAY) return "BadTensor";
        const JValue *offs = json_get(obj, "data_offsets");
        if (!offs || offs->kind != J_ARRAY) return "BadTensor";
        if (shape_v->arr.len > SAFETENSORS_MAX_RANK || offs->arr.len != 2) return "BadTensor";
        TensorInfo info = {.dtype = dtype, .rank = (uint8_t)shape_v->arr.len, .shape = {1, 1, 1, 1, 1}, .file = file_index};
        for (size_t i = 0; i < shape_v->arr.len; i++)
            if (!as_u64(shape_v->arr.items[i], &info.shape[i])) return "BadTensor";
        uint64_t begin, end;
        if (!as_u64(offs->arr.items[0], &begin)) return "BadTensor";
        if (!as_u64(offs->arr.items[1], &end)) return "BadTensor";
        if (end < begin) return "BadTensor";
        // every product and sum checked: a crafted header must not wrap into a small size
        uint64_t numel = 1, bytes, stop;
        for (uint8_t i = 0; i < info.rank; i++)
            if (__builtin_mul_overflow(numel, info.shape[i], &numel)) return "SizeMismatch";
        if (__builtin_mul_overflow(numel, (uint64_t)dtype_size(dtype), &bytes)) return "SizeMismatch";
        if (__builtin_add_overflow(data_start, begin, &info.offset)) return "TooShort";
        info.nbytes = end - begin;
        if (bytes != info.nbytes) return "SizeMismatch";
        if (__builtin_add_overflow(info.offset, info.nbytes, &stop) || stop > len) return "TooShort";
        TensorInfo *t = arena_dup(a, &info, sizeof info);
        char *name = arena_strndup(a, e->key.ptr, e->key.len);
        strmap_put(out, str_n(name, e->key.len), t);
    }
    return OK;
}
