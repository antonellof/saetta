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
