// Tests for hmac_sha256 (src/hash.c): RFC 4231 test cases 1, 2 and 6.
#include "../test.h"
#include "../hash.h"

static Str hmac_hex(const void *k, size_t nk, const char *m, char out[65]) {
    uint8_t d[SHA256_LEN];
    hmac_sha256(k, nk, m, strlen(m), d);
    hash_hex(d, SHA256_LEN, out);
    return str(out);
}

TEST(hmac_sha256_rfc4231) {
    char h[65];
    uint8_t k1[20], k6[131];
    memset(k1, 0x0b, sizeof k1);
    memset(k6, 0xaa, sizeof k6);
    EXPECT_STR(hmac_hex(k1, sizeof k1, "Hi There", h), S("b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"));
    EXPECT_STR(hmac_hex("Jefe", 4, "what do ya want for nothing?", h), S("5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"));
    EXPECT_STR(hmac_hex(k6, sizeof k6, "Test Using Larger Than Block-Size Key - Hash Key First", h),
               S("60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"));
}
