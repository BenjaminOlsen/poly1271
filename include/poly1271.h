/* SPDX-License-Identifier: MIT */

/*
 * poly1271.h - Polynomial MAC using Mersenne prime 2^127 - 1
 *
 * Author: Benjamin Olsen (2025)
 *
 * Block size: 15 bytes
 * Security: ~100+ bits (same ballpark as Poly1305)
 * Key: 32 bytes (r: 16 bytes, s: 16 bytes)
 * Tag: 16 bytes
 */

#ifndef POLY1271_H
#define POLY1271_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define POLY1271_BLOCK_SIZE 15
#define POLY1271_KEY_SIZE   32
#define POLY1271_TAG_SIZE   16

typedef struct {
    uint64_t acc[3];   /* accumulator */
    uint64_t r[2];     /* r key */
    uint64_t r2[2];    /* r^2 mod p */
    uint64_t r3[2];    /* r^3 mod p */
    uint64_t r4[2];    /* r^4 mod p */
    uint64_t s[2];     /* s key */
    uint8_t  buf[15];  /* partial block buffer */
    uint8_t  buflen;
    uint8_t  blkcnt;   /* blocks since last reduction */
} poly1271_ctx_t;

/* streaming API */
void poly1271_init(poly1271_ctx_t* ctx, const uint8_t key[32]);
void poly1271_update(poly1271_ctx_t* ctx, const uint8_t* msg, size_t len);
void poly1271_finish(poly1271_ctx_t* ctx, uint8_t tag[16]);

/* one-shot */
void poly1271(uint8_t tag[16], const uint8_t* msg, size_t len, const uint8_t key[32]);

/* constant-time verify, returns 0 on match */
int poly1271_verify(const uint8_t tag[16], const uint8_t* msg, size_t len, const uint8_t key[32]);

/* AVX2 implementation - radix-2^26, 4-way parallel */
#if defined(__AVX2__) || defined(__AVX2)
#include <immintrin.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4324) /* structure padded due to alignment */
#endif

typedef struct {
    uint64_t acc[5];     /* accumulator, radix-26 */
    uint64_t r[5];       /* r, radix-26 */
    uint64_t r2[5];      /* r^2 */
    uint64_t r3[5];      /* r^3 */
    uint64_t r4[5];      /* r^4 */
    __m256i rv[5];       /* rv[i] = {r4[i], r3[i], r2[i], r[i]} */
    uint64_t s[2];       /* s key */
    uint8_t  buf[60];    /* partial block buffer (4 blocks) */
    uint8_t  buflen;
} poly1271_avx2_ctx_t;

void poly1271_avx2_init(poly1271_avx2_ctx_t* ctx, const uint8_t key[32]);
void poly1271_avx2_update(poly1271_avx2_ctx_t* ctx, const uint8_t* msg, size_t len);
void poly1271_avx2_finish(poly1271_avx2_ctx_t* ctx, uint8_t tag[16]);
void poly1271_avx2(uint8_t tag[16], const uint8_t* msg, size_t len, const uint8_t key[32]);
int poly1271_avx2_verify(const uint8_t tag[16], const uint8_t* msg, size_t len, const uint8_t key[32]);

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif /* AVX2 */

#ifdef __cplusplus
}
#endif

#endif /* POLY1271_H */
