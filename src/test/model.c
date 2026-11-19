#include "../model.h"
#include "../test.h"

TEST(model_argmax_first_largest_as_the_scalar_scan) {
    float v[1000];
    for (size_t i = 0; i < 1000; i++) v[i] = (float)((i * 7919) % 613);
    size_t best = 0;
    for (size_t i = 0; i < 1000; i++)
        if (v[i] > v[best]) best = i;
    EXPECT_EQ(model_argmax(v, 1000), best);
    v[999] = 1e9f;
    EXPECT_EQ(model_argmax(v, 1000), 999);
    v[3] = 1e9f;
    EXPECT_EQ(model_argmax(v, 1000), 3);
    const float w[4] = {1, 5, 5, 2};
    EXPECT_EQ(model_argmax(w, 4), 1);
}
