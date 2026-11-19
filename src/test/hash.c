// Tests for src/hash.c: FIPS 180 example vectors.
#include "../hash.h"
#include "../test.h"

#include <stdlib.h>

static Str hex256(char *out, const void *p, size_t n) {
    uint8_t d[SHA256_LEN];
    sha256(p, n, d);
    hash_hex(d, SHA256_LEN, out);
    return str(out);
}

static Str hex1(char *out, const void *p, size_t n) {
    uint8_t d[SHA1_LEN];
    sha1(p, n, d);
    hash_hex(d, SHA1_LEN, out);
    return str(out);
}

TEST(hash_sha256) {
    char h[65];
    EXPECT_STR(hex256(h, "", 0), S("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
    EXPECT_STR(hex256(h, "abc", 3), S("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
    const char *m = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    EXPECT_STR(hex256(h, m, strlen(m)), S("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"));
    // One million 'a', fed in uneven pieces.
    char *buf = malloc(1000000);
    memset(buf, 'a', 1000000);
    Sha256 s;
    sha256_init(&s);
    size_t off = 0, step = 1;
    while (off < 1000000) {
        size_t n = MIN(step, 1000000 - off);
        sha256_update(&s, buf + off, n);
        off += n;
        step = step * 3 % 997 + 1;
    }
    uint8_t d[SHA256_LEN];
    sha256_final(&s, d);
    hash_hex(d, SHA256_LEN, h);
    EXPECT_STR(str(h), S("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"));
    EXPECT_STR(hex256(h, buf, 1000000), S("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"));
    free(buf);
    // Lengths around the padding boundary (55, 56, 64 bytes).
    char z[64];
    memset(z, 'a', sizeof z);
    EXPECT_STR(hex256(h, z, 55), S("9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"));
    EXPECT_STR(hex256(h, z, 56), S("b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"));
    EXPECT_STR(hex256(h, z, 64), S("ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"));
}

TEST(hash_sha1) {
    char h[41];
    EXPECT_STR(hex1(h, "", 0), S("da39a3ee5e6b4b0d3255bfef95601890afd80709"));
    EXPECT_STR(hex1(h, "abc", 3), S("a9993e364706816aba3e25717850c26c9cd0d89d"));
    const char *m = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    EXPECT_STR(hex1(h, m, strlen(m)), S("84983e441c3bd26ebaae4aa1f95129e5e54670f1"));
    // Git blob id of "hello\n".
    EXPECT_STR(hex1(h, "blob 6\0hello\n", 13), S("ce013625030ba8dba906f756967f9e9ca394464a"));
    Sha1 s;
    sha1_init(&s);
    for (int k = 0; k < 1000; k++) sha1_update(&s, "aaaaaaaaaa", 10);
    uint8_t d[SHA1_LEN];
    sha1_final(&s, d);
    hash_hex(d, SHA1_LEN, h);
    EXPECT_STR(str(h), S("a080cbda64850abb7b7f67ee875ba068074ff6fe"));
}
