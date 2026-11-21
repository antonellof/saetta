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

// SAETTA_CTX / --ctx: rounded down to a multiple of SESSION_CTX_ALIGN, small or huge values refused.
TEST(model_session_ctx_parse) {
    size_t c = 0;
    EXPECT_OK(session_ctx_parse(NULL, 262144, &c));
    EXPECT_EQ(c, SESSION_CTX_DEFAULT);
    EXPECT_OK(session_ctx_parse("204800", 262144, &c));
    EXPECT_EQ(c, 204800);
    EXPECT_OK(session_ctx_parse("100001", 262144, &c));
    EXPECT_EQ(c, 99968);
    EXPECT_EQ(c % 4, 0);
    EXPECT_OK(session_ctx_parse("1024", 262144, &c));
    EXPECT_EQ(c, 1024);
    EXPECT_OK(session_ctx_parse("1151", 262144, &c));
    EXPECT_EQ(c, 1024);
    EXPECT_ERR(session_ctx_parse("1023", 262144, &c), "ContextTooSmall");
    EXPECT_ERR(session_ctx_parse("0", 262144, &c), "ContextTooSmall");
    EXPECT_ERR(session_ctx_parse("4096000000000", 262144, &c), "ContextTooLarge");
    EXPECT_ERR(session_ctx_parse("262145", 262144, &c), "ContextTooLarge");
    EXPECT_OK(session_ctx_parse("262144", 262144, &c));
    EXPECT_EQ(c, 262144);
    EXPECT(session_ctx_parse("32k", 262144, &c) != OK);
    EXPECT(session_ctx_parse("-5", 262144, &c) != OK);
}
