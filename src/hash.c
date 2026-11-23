// SHA-256 and SHA-1, straight from FIPS 180-4.
#include "hash.h"

#include <string.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

[[maybe_unused]] static inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static inline uint32_t rotl(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
static inline uint32_t be32(const uint8_t *p) {
    return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
}
static inline void put_be32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16); p[2] = (uint8_t)(v >> 8); p[3] = (uint8_t)v;
}

// --- SHA-256 ---------------------------------------------------------------

static const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

#if defined(__ARM_FEATURE_SHA2)
#include <arm_neon.h>

// ARMv8 SHA-256 instructions: four rounds per step.
static void sha256_block(uint32_t h[8], const uint8_t *p) {
    uint32x4_t abcd = vld1q_u32(h), efgh = vld1q_u32(h + 4);
    uint32x4_t abcd0 = abcd, efgh0 = efgh;
    uint32x4_t m[4];
    for (int i = 0; i < 4; i++) m[i] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p + 16 * i)));
    for (int i = 0; i < 16; i++) {
        uint32x4_t w = m[i & 3];
        if (i >= 4) {
            w = vsha256su1q_u32(vsha256su0q_u32(m[i & 3], m[(i + 1) & 3]), m[(i + 2) & 3], m[(i + 3) & 3]);
            m[i & 3] = w;
        }
        uint32x4_t wk = vaddq_u32(w, vld1q_u32(K256 + 4 * i));
        uint32x4_t t = abcd;
        abcd = vsha256hq_u32(abcd, efgh, wk);
        efgh = vsha256h2q_u32(efgh, t, wk);
    }
    vst1q_u32(h, vaddq_u32(abcd, abcd0));
    vst1q_u32(h + 4, vaddq_u32(efgh, efgh0));
}
#else
static void sha256_block(uint32_t h[8], const uint8_t *p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = be32(p + 4 * i);
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = k + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K256[i] + w[i];
        uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        k = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += k;
}
#endif

void sha256_init(Sha256 *s) {
    static const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    memcpy(s->h, iv, sizeof iv);
    s->total = 0;
    s->n = 0;
}

typedef void (*block_fn)(uint32_t *h, const uint8_t *p);

// Shared buffering for both hashes (64-byte blocks).
static inline void update(uint32_t *h, uint64_t *total, uint8_t *buf, size_t *bn, block_fn block,
                          const uint8_t *p, size_t n) {
    *total += n;
    if (*bn) {
        size_t t = MIN(64 - *bn, n);
        memcpy(buf + *bn, p, t);
        *bn += t;
        p += t;
        n -= t;
        if (*bn < 64) return;
        block(h, buf);
        *bn = 0;
    }
    for (; n >= 64; p += 64, n -= 64) block(h, p);
    if (n) {
        memcpy(buf, p, n);
        *bn = n;
    }
}

// Pads with 0x80, zeros and the big-endian bit length.
static inline void final(uint32_t *h, uint64_t total, uint8_t *buf, size_t bn, block_fn block) {
    uint64_t bits = total * 8;
    buf[bn++] = 0x80;
    if (bn > 56) {
        memset(buf + bn, 0, 64 - bn);
        block(h, buf);
        bn = 0;
    }
    memset(buf + bn, 0, 56 - bn);
    put_be32(buf + 56, (uint32_t)(bits >> 32));
    put_be32(buf + 60, (uint32_t)bits);
    block(h, buf);
}

void sha256_update(Sha256 *s, const void *p, size_t n) {
    update(s->h, &s->total, s->buf, &s->n, sha256_block, p, n);
}

void sha256_final(Sha256 *s, uint8_t out[SHA256_LEN]) {
    final(s->h, s->total, s->buf, s->n, sha256_block);
    for (int i = 0; i < 8; i++) put_be32(out + 4 * i, s->h[i]);
}

void sha256(const void *p, size_t n, uint8_t out[SHA256_LEN]) {
    Sha256 s;
    sha256_init(&s);
    sha256_update(&s, p, n);
    sha256_final(&s, out);
}

// --- SHA-1 -----------------------------------------------------------------

static void sha1_block(uint32_t h[5], const uint8_t *p) {
    uint32_t w[80];
    for (int i = 0; i < 16; i++) w[i] = be32(p + 4 * i);
    for (int i = 16; i < 80; i++) w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20) { f = (b & c) | (~b & d); k = 0x5a827999; }
        else if (i < 40) { f = b ^ c ^ d; k = 0x6ed9eba1; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8f1bbcdc; }
        else { f = b ^ c ^ d; k = 0xca62c1d6; }
        uint32_t t = rotl(a, 5) + f + e + k + w[i];
        e = d; d = c; c = rotl(b, 30); b = a; a = t;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
}

void sha1_init(Sha1 *s) {
    static const uint32_t iv[5] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0};
    memcpy(s->h, iv, sizeof iv);
    s->total = 0;
    s->n = 0;
}

void sha1_update(Sha1 *s, const void *p, size_t n) {
    update(s->h, &s->total, s->buf, &s->n, sha1_block, p, n);
}

void sha1_final(Sha1 *s, uint8_t out[SHA1_LEN]) {
    final(s->h, s->total, s->buf, s->n, sha1_block);
    for (int i = 0; i < 5; i++) put_be32(out + 4 * i, s->h[i]);
}

void sha1(const void *p, size_t n, uint8_t out[SHA1_LEN]) {
    Sha1 s;
    sha1_init(&s);
    sha1_update(&s, p, n);
    sha1_final(&s, out);
}

void hash_hex(const uint8_t *d, size_t n, char *out) {
    static const char hx[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[2 * i] = hx[d[i] >> 4];
        out[2 * i + 1] = hx[d[i] & 15];
    }
    out[2 * n] = 0;
}

void hmac_sha256(const void *key, size_t nkey, const void *msg, size_t nmsg, uint8_t out[SHA256_LEN]) {
    uint8_t k[64] = {0}, ipad[64], opad[64], inner[SHA256_LEN];
    if (nkey > 64) sha256(key, nkey, k);
    else memcpy(k, key, nkey);
    for (int i = 0; i < 64; i++) {
        ipad[i] = k[i] ^ 0x36;
        opad[i] = k[i] ^ 0x5c;
    }
    Sha256 s;
    sha256_init(&s);
    sha256_update(&s, ipad, 64);
    sha256_update(&s, msg, nmsg);
    sha256_final(&s, inner);
    sha256_init(&s);
    sha256_update(&s, opad, 64);
    sha256_update(&s, inner, SHA256_LEN);
    sha256_final(&s, out);
}
