// SHA-256 and SHA-1 (FIPS 180-4): streaming and one-shot, with lowercase hex output.
#ifndef SAETTA_HASH_H
#define SAETTA_HASH_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t h[8];
    uint64_t total; // bytes hashed so far
    uint8_t buf[64];
    size_t n; // bytes pending in buf
} Sha256;

typedef struct {
    uint32_t h[5];
    uint64_t total;
    uint8_t buf[64];
    size_t n;
} Sha1;

#define SHA256_LEN 32
#define SHA1_LEN 20

void sha256_init(Sha256 *s);
void sha256_update(Sha256 *s, const void *p, size_t n);
void sha256_final(Sha256 *s, uint8_t out[SHA256_LEN]);
void sha256(const void *p, size_t n, uint8_t out[SHA256_LEN]);

void sha1_init(Sha1 *s);
void sha1_update(Sha1 *s, const void *p, size_t n);
void sha1_final(Sha1 *s, uint8_t out[SHA1_LEN]);
void sha1(const void *p, size_t n, uint8_t out[SHA1_LEN]);

// HMAC-SHA256 (RFC 2104), for the challenge a shared engine answers (code_share.c): the holder
// proves it knows the token without sending it.
void hmac_sha256(const void *key, size_t nkey, const void *msg, size_t nmsg, uint8_t out[SHA256_LEN]);

// Writes 2*n lowercase hex digits and a NUL into out (2*n + 1 bytes).
void hash_hex(const uint8_t *d, size_t n, char *out);

#endif
