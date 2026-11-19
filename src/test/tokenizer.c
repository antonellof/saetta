#include "../test.h"
#include "../tokenizer.h"

TEST(pre_tokenizer_splits_like_the_qwen_regex) {
    static const struct {
        const char *in;
        const char *out[10];
    } cases[] = {
        {"Hello world", {"Hello", " world"}},
        {"x  = 100;\n\n  y", {"x", " ", " =", " ", "1", "0", "0", ";\n\n", " ", " y"}},
        {"don't", {"don", "'t"}},
        {"a\n  \n b", {"a", "\n  \n", " b"}},
        {"end   ", {"end", "   "}},
    };
    for (size_t c = 0; c < COUNT(cases); c++) {
        Str in = str(cases[c].in);
        size_t want = 0;
        while (want < 10 && cases[c].out[want]) want++;
        size_t got = 0, p = 0;
        while (p < in.len) {
            size_t e = tokenizer_next_piece(in, p);
            EXPECT(got < want);
            EXPECT_STR(str_sub(in, p, e), str(cases[c].out[got]));
            got++;
            p = e;
        }
        EXPECT_EQ(got, want);
    }
}
