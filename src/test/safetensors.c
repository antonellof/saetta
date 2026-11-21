#include "../safetensors.h"
#include "../test.h"

TEST(safetensors_parse_a_small_header) {
    Arena arena = {0};
    static const char header[] = "{\"__metadata__\":{\"format\":\"mlx\"},\"a\":{\"dtype\":\"BF16\",\"shape\":[2,3],\"data_offsets\":[0,12]}}";
    enum { HL = sizeof header - 1 };
    uint8_t buf[8 + HL + 12] = {0};
    uint64_t hl = HL;
    memcpy(buf, &hl, 8);
    memcpy(buf + 8, header, HL);
    StrMap map = {0};
    Err e = safetensors_parse(&arena, buf, sizeof buf, 0, &map);
    if (e) arena_reset(&arena);
    EXPECT_OK(e);
    void **v = strmap_get(&map, S("a"));
    EXPECT(v != NULL);
    const TensorInfo *t = *v;
    EXPECT_EQ(t->dtype, DTYPE_BF16);
    EXPECT_EQ(tensor_numel(t), 6);
    EXPECT_EQ(t->offset, 8 + HL);
    strmap_free(&map);
    arena_reset(&arena);
}

// A header in memory: 8 length bytes, the JSON, then data bytes of zeros.
static Err parse_header(Arena *a, const char *header, size_t data, StrMap *map) {
    size_t hl = strlen(header), n = 8 + hl + data;
    uint8_t *buf = arena_alloc(a, n);
    memset(buf, 0, n);
    uint64_t h = hl;
    memcpy(buf, &h, 8);
    memcpy(buf + 8, header, hl);
    return safetensors_parse(a, buf, n, 0, map);
}

// Offsets out of order, a shape whose element count wraps, offsets past the file and
// offsets whose sum wraps are refused instead of giving a small or wrapped size.
TEST(safetensors_rejects_overflowing_headers) {
    Arena arena = {0};
    StrMap map = {0};
    EXPECT_ERR(parse_header(&arena, "{\"a\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[8,4]}}", 16, &map),
               "BadTensor");
    // 2^32 * 2^32 wraps to 0, which would match empty offsets
    EXPECT_ERR(parse_header(&arena,
                            "{\"a\":{\"dtype\":\"U8\",\"shape\":[4294967296,4294967296],\"data_offsets\":[0,0]}}", 16,
                            &map),
               "SizeMismatch");
    // 2^62 elements of 4 bytes wrap to 0
    EXPECT_ERR(parse_header(&arena, "{\"a\":{\"dtype\":\"F32\",\"shape\":[4611686018427387904],\"data_offsets\":[0,0]}}",
                            16, &map),
               "SizeMismatch");
    EXPECT_ERR(parse_header(&arena, "{\"a\":{\"dtype\":\"U8\",\"shape\":[32],\"data_offsets\":[0,32]}}", 16, &map),
               "TooShort");
    // offsets past i64 are not integers to the parser; offsets near 2^63 are past the file
    EXPECT_ERR(parse_header(&arena,
                            "{\"a\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[9223372036854775800,9223372036854775808]}}",
                            16, &map),
               "BadTensor");
    EXPECT_ERR(parse_header(&arena,
                            "{\"a\":{\"dtype\":\"U8\",\"shape\":[7],\"data_offsets\":[9223372036854775800,9223372036854775807]}}",
                            16, &map),
               "TooShort");
    EXPECT_OK(parse_header(&arena, "{\"a\":{\"dtype\":\"U8\",\"shape\":[2,8],\"data_offsets\":[0,16]}}", 16, &map));
    strmap_free(&map);
    arena_reset(&arena);
}
