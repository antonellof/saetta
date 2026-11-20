#include <math.h>

#include "../test.h"
#include "../weights.h"

#include "../config.h"

#if CONFIG_ARCH_27B
TEST(weights_quant_overrides) {
    // 27B packs: one quantization for every linear (4-bit group 64 compiled in)
    EXPECT_EQ(weights_quant_for(S("language_model.model.layers.3.mlp.gate_proj")).bits, 4);
    EXPECT_EQ(weights_quant_for(S("language_model.model.layers.3.mlp.gate_proj")).group_size, 64);
}
#else
TEST(weights_quant_overrides) {
    EXPECT_EQ(weights_quant_for(S("language_model.model.layers.3.mlp.gate")).bits, 8);
    EXPECT_EQ(weights_quant_for(S("language_model.model.layers.3.mlp.shared_expert_gate")).bits, 8);
    EXPECT_EQ(weights_quant_for(S("language_model.model.layers.3.mlp.switch_mlp.up_proj")).bits, 4);
}
#endif

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static float bf(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

TEST(weights_quantize_round_trip) {
    uint8_t src[2 * 128 * 2];
    for (size_t i = 0; i < 256; i++) {
        float f = ((float)(i % 128) - 40.0f) / 32.0f;
        uint16_t b = weights_to_bf16(f);
        src[i * 2] = (uint8_t)b;
        src[i * 2 + 1] = (uint8_t)(b >> 8);
    }
    static _Alignas(16) uint8_t dst[8192];
    static const uint32_t bgs[2][2] = {{4, 32}, {8, 64}};
    for (int c = 0; c < 2; c++) {
        uint32_t bits = bgs[c][0], group = bgs[c][1];
        Quantized r = weights_quantize(src, 2, 128, bits, group, dst, 0, 0);
        // every element of row 1 dequantizes to within half a step of the source
        for (uint32_t k = 0; k < 128; k++) {
            uint32_t g = k / group;
            float sc = bf(rd16(dst + r.lin.scales.offset + (1 * (128 / group) + g) * 2));
            float bi = bf(rd16(dst + r.lin.biases.offset + (1 * (128 / group) + g) * 2));
            uint32_t bit = (128 + k) * bits;
            uint32_t word;
            memcpy(&word, dst + r.lin.weight.offset + bit / 32 * 4, 4);
            uint32_t q = (word >> (bit % 32)) & ((1u << bits) - 1);
            float want = bf(rd16(src + (128 + k) * 2));
            EXPECT(fabsf((float)q * sc + bi - want) <= sc * 0.51f + 1e-6f);
        }
    }
}
