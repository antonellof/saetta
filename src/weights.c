// The model pack (see weights.h).
#include "weights.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>

#include "config.h"
#include "json.h"
#include "metal/shim.h"

static Err map_whole(const char *path, const uint8_t **ptr, size_t *len) {
    const void *p = sx_map_file(path, len);
    if (!p) return "MapFailed";
    *ptr = p;
    return OK;
}

static bool num_get(const JValue *o, const char *k, uint32_t *out) {
    const JValue *v = json_get(o, k);
    if (!v || v->kind != J_INTEGER) return false;
    *out = (uint32_t)v->i;
    return true;
}

static Err load_quant(Pack *p, Arena *a) {
    const char *path = path_join(a, p->dir, "config.json");
    const uint8_t *bytes;
    size_t len;
    if (map_whole(path, &bytes, &len)) return OK;
    JValue *root;
    Err e = json_parse(a, str_n((const char *)bytes, len), true, &root);
    if (e) {
        sx_unmap_file(bytes, len);
        return e;
    }
    const JValue *q = json_get(root, "quantization");
    if (q && q->kind == J_OBJECT) {
        uint32_t b, g;
        if (num_get(q, "bits", &b) && num_get(q, "group_size", &g)) {
            p->has_quant_default = true;
            p->quant_default = (Quant){b, g};
        }
        for (size_t i = 0; i < q->obj.len; i++) {
            const JMember *m = &q->obj.items[i];
            if (m->val->kind != J_OBJECT) continue;
            if (num_get(m->val, "bits", &b) && num_get(m->val, "group_size", &g)) {
                Quant *qq = ARENA_NEW(a, Quant);
                *qq = (Quant){b, g};
                char *k = arena_strndup(a, m->key.ptr, m->key.len);
                strmap_put(&p->quant, str_n(k, m->key.len), qq);
            }
        }
    }
    sx_unmap_file(bytes, len);
    return OK;
}

Err pack_open(Arena *a, const char *dir, Pack *out) {
    const char *index_path = path_join(a, dir, "model.safetensors.index.json");
    const uint8_t *index;
    size_t index_len;
    TRY(map_whole(index_path, &index, &index_len));
    JValue *root;
    Err e = json_parse(a, str_n((const char *)index, index_len), true, &root);
    if (e) {
        sx_unmap_file(index, index_len);
        return e;
    }
    const JValue *wmap = json_get(root, "weight_map");
    if (!wmap || wmap->kind != J_OBJECT) {
        sx_unmap_file(index, index_len);
        return "BadIndex";
    }
    // shard names in order of first appearance
    StrMap seen = {0};
    VecStr names = {0};
    for (size_t i = 0; i < wmap->obj.len; i++) {
        const JValue *v = wmap->obj.items[i].val;
        if (v->kind != J_STRING) {
            strmap_free(&seen);
            vec_free(&names);
            sx_unmap_file(index, index_len);
            return "BadIndex";
        }
        if (strmap_has(&seen, v->s)) continue;
        Str s = str_n(arena_strndup(a, v->s.ptr, v->s.len), v->s.len);
        strmap_put(&seen, s, NULL);
        vec_push(&names, s);
    }
    strmap_free(&seen);
    sx_unmap_file(index, index_len);

    Pack pack = {.dir = dir, .nfiles = names.len};
    pack.files = ARENA_ARRAY(a, MappedFile, names.len ? names.len : 1);
    for (size_t i = 0; i < names.len; i++) {
        const char *path = path_join(a, dir, names.ptr[i].ptr);
        const uint8_t *bytes;
        size_t len;
        e = map_whole(path, &bytes, &len);
        if (!e) {
            pack.files[i] = (MappedFile){names.ptr[i].ptr, bytes, len};
            e = safetensors_parse(a, bytes, len, (uint16_t)i, &pack.tensors);
        }
        if (e) {
            vec_free(&names);
            return e;
        }
    }
    vec_free(&names);
    TRY(load_quant(&pack, a));
    *out = pack;
    return OK;
}

void pack_close(Pack *p) {
    for (size_t i = 0; i < p->nfiles; i++) sx_unmap_file(p->files[i].bytes, p->files[i].len);
}

Quant pack_quant_of(const Pack *p, Str module) {
    void **q = strmap_get(&p->quant, module);
    if (q) return *(const Quant *)*q;
    if (p->has_quant_default) return p->quant_default;
    return weights_quant_for(module);
}

const TensorInfo *pack_get(const Pack *p, Str name) {
    void **v = strmap_get(&p->tensors, name);
    return v ? *v : NULL;
}

static const TensorInfo *get_suffixed(const Pack *p, char *buf, size_t cap, Str module, const char *suffix, Err *err) {
    int n = snprintf(buf, cap, "%.*s.%s", (int)module.len, module.ptr, suffix);
    if (n < 0 || (size_t)n >= cap) {
        *err = "NoSpaceLeft";
        return NULL;
    }
    const TensorInfo *t = pack_get(p, str_n(buf, (size_t)n));
    if (!t) *err = "MissingTensor";
    return t;
}

Err pack_linear(const Pack *p, Str module, Linear *out) {
    Quant q = pack_quant_of(p, module);
    char buf[1024];
    Err e = OK;
    const TensorInfo *w = get_suffixed(p, buf, sizeof buf, module, "weight", &e);
    if (!w) return e;
    const TensorInfo *s = get_suffixed(p, buf, sizeof buf, module, "scales", &e);
    if (!s) return e;
    const TensorInfo *b = get_suffixed(p, buf, sizeof buf, module, "biases", &e);
    if (!b) return e;
    if (w->dtype != DTYPE_U32 || w->rank < 2) return "BadQuantShape";
    uint64_t o = w->shape[w->rank - 2];
    uint64_t in = w->shape[w->rank - 1] * 32 / q.bits;
    if (s->rank < 2) return "BadQuantShape";
    if (s->shape[s->rank - 1] * q.group_size != in || s->shape[s->rank - 2] != o) return "BadQuantShape";
    if (b->nbytes != s->nbytes) return "BadQuantShape";
    *out = (Linear){.weight = *w, .scales = *s, .biases = *b, .bits = q.bits, .group_size = q.group_size,
                    .out_features = o, .in_features = in};
    return OK;
}

Quant weights_quant_for(Str module) {
    for (size_t i = 0; i < CONFIG_N_QUANT_OVERRIDES; i++) {
        const QuantOverride *o = &config_quant_overrides[i];
        if (str_ends(module, str(o->suffix))) return (Quant){o->bits, o->group_size};
    }
    return (Quant){CONFIG_QUANT_BITS, CONFIG_QUANT_GROUP_SIZE};
}

static inline float bf16_bits(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static inline uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static inline void wr16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static inline void wr32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); } // little-endian host

static TensorInfo tensor2(DType dt, uint64_t a, uint64_t b, uint64_t off, uint64_t nbytes, uint16_t file) {
    return (TensorInfo){.dtype = dt, .rank = 2, .shape = {a, b, 0, 0, 0}, .offset = off, .nbytes = nbytes, .file = file};
}

Quantized weights_quantize(const uint8_t *src, uint64_t out, uint64_t in, uint32_t bits, uint32_t group,
                           uint8_t *dst, uint64_t at, uint16_t file) {
    uint64_t groups = in / group;
    uint64_t wbytes = out * in * bits / 8;
    uint64_t sbytes = out * groups * 2;
    uint64_t w_off = at;
    uint64_t s_off = ALIGN_UP(w_off + wbytes, 16);
    uint64_t b_off = ALIGN_UP(s_off + sbytes, 16);
    uint32_t per = 32 / bits;
    float qmax = (float)((1u << bits) - 1);
    float v[64];
    for (uint64_t r = 0; r < out; r++)
        for (uint64_t g = 0; g < groups; g++) {
            float lo = INFINITY, hi = -INFINITY;
            for (uint32_t j = 0; j < group; j++) {
                uint64_t i = r * in + g * group + j;
                v[j] = bf16_bits(rd16(src + i * 2));
                lo = fminf(lo, v[j]);
                hi = fmaxf(hi, v[j]);
            }
            uint16_t sb = weights_to_bf16((hi - lo) / qmax);
            uint16_t bb = weights_to_bf16(lo);
            float sc = bf16_bits(sb), bi = bf16_bits(bb);
            wr16(dst + s_off + (r * groups + g) * 2, sb);
            wr16(dst + b_off + (r * groups + g) * 2, bb);
            for (uint32_t wi = 0; wi < group / per; wi++) {
                uint32_t word = 0;
                for (uint32_t j = 0; j < per; j++) {
                    float q = sc > 0 ? roundf((v[wi * per + j] - bi) / sc) : 0;
                    uint32_t qi = (uint32_t)fmaxf(0, fminf(q, qmax)); // std.math.clamp
                    word |= qi << (bits * j);
                }
                wr32(dst + w_off + (r * in + g * group) * bits / 8 + wi * 4, word);
            }
        }
    Quantized res = {
        .lin = {.weight = tensor2(DTYPE_U32, out, in * bits / 32, w_off, wbytes, file),
                .scales = tensor2(DTYPE_BF16, out, groups, s_off, sbytes, file),
                .biases = tensor2(DTYPE_BF16, out, groups, b_off, sbytes, file),
                .bits = bits, .group_size = group, .out_features = out, .in_features = in},
        .used = b_off + sbytes - at,
    };
    return res;
}
