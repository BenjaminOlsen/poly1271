/* SPDX-License-Identifier: MIT */
/* Poly1305 implementation for benchmarking comparison */

#include "poly1305.h"
#include <string.h>

#ifdef _MSC_VER
#define INLINE __forceinline
#else
#define INLINE static inline __attribute__((always_inline))
#endif

INLINE uint32_t load32_le(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

INLINE void store32_le(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static void secure_zero(void* ptr, size_t len) {
    volatile uint8_t* p = (volatile uint8_t*)ptr;
    while (len--) *p++ = 0;
}

#define SUM(a,b,c,d,e) ((a)+(b)+(c)+(d)+(e))
#define MUL(a,b) ((uint64_t)(a)*(uint64_t)(b))

/* Multiply two radix-2^26 values mod 2^130-5, result in radix-2^26 */
static void mul_r_mod(uint32_t out[5],
                      const uint32_t a[5], const uint64_t sa[4],
                      const uint32_t b[5], const uint64_t sb[4]) {
    /* Full product: a * b, using precomputed sa[i] = a[i+1] * 5, sb[i] = b[i+1] * 5 */
    uint64_t d0 = SUM(MUL(a[0],b[0]), MUL(sa[3],b[1]), MUL(sa[2],b[2]), MUL(sa[1],b[3]), MUL(sa[0],b[4]));
    uint64_t d1 = SUM(MUL(a[0],b[1]), MUL(a[1],b[0]), MUL(sa[3],b[2]), MUL(sa[2],b[3]), MUL(sa[1],b[4]));
    uint64_t d2 = SUM(MUL(a[0],b[2]), MUL(a[1],b[1]), MUL(a[2],b[0]), MUL(sa[3],b[3]), MUL(sa[2],b[4]));
    uint64_t d3 = SUM(MUL(a[0],b[3]), MUL(a[1],b[2]), MUL(a[2],b[1]), MUL(a[3],b[0]), MUL(sa[3],b[4]));
    uint64_t d4 = SUM(MUL(a[0],b[4]), MUL(a[1],b[3]), MUL(a[2],b[2]), MUL(a[3],b[1]), MUL(a[4],b[0]));

    /* Carry propagation and reduction */
    uint32_t c;
    c = (uint32_t)(d0 >> 26); out[0] = (uint32_t)d0 & 0x3ffffffu; d1 += c;
    c = (uint32_t)(d1 >> 26); out[1] = (uint32_t)d1 & 0x3ffffffu; d2 += c;
    c = (uint32_t)(d2 >> 26); out[2] = (uint32_t)d2 & 0x3ffffffu; d3 += c;
    c = (uint32_t)(d3 >> 26); out[3] = (uint32_t)d3 & 0x3ffffffu; d4 += c;
    c = (uint32_t)(d4 >> 26); out[4] = (uint32_t)d4 & 0x3ffffffu;

    /* Reduce: multiply overflow by 5 and add back */
    out[0] += c * 5u;
    c = out[0] >> 26;
    out[0] &= 0x3ffffffu;
    out[1] += c;
}

/* Square a radix-2^26 value mod 2^130-5 */
static void square_r_mod(uint32_t out[5], const uint32_t a[5], const uint64_t sa[4]) {
    /* Squaring: cross terms (i≠j) get 2x, diagonal terms (i=j) get 1x
     * sa[k] = a[k+1] * 5 for reduction of terms with i+j >= 5 */
    uint64_t d0 = MUL(a[0],a[0]) + 2u*(MUL(a[1],sa[3]) + MUL(a[2],sa[2]));
    uint64_t d1 = 2u*(MUL(a[0],a[1]) + MUL(a[2],sa[3])) + MUL(a[3],sa[2]);
    uint64_t d2 = 2u*MUL(a[0],a[2]) + MUL(a[1],a[1]) + 2u*MUL(a[3],sa[3]);
    uint64_t d3 = 2u*(MUL(a[0],a[3]) + MUL(a[1],a[2])) + MUL(a[4],sa[3]);
    uint64_t d4 = 2u*(MUL(a[0],a[4]) + MUL(a[1],a[3])) + MUL(a[2],a[2]);

    uint32_t c;
    c = (uint32_t)(d0 >> 26); out[0] = (uint32_t)d0 & 0x3ffffffu; d1 += c;
    c = (uint32_t)(d1 >> 26); out[1] = (uint32_t)d1 & 0x3ffffffu; d2 += c;
    c = (uint32_t)(d2 >> 26); out[2] = (uint32_t)d2 & 0x3ffffffu; d3 += c;
    c = (uint32_t)(d3 >> 26); out[3] = (uint32_t)d3 & 0x3ffffffu; d4 += c;
    c = (uint32_t)(d4 >> 26); out[4] = (uint32_t)d4 & 0x3ffffffu;

    out[0] += c * 5u;
    c = out[0] >> 26;
    out[0] &= 0x3ffffffu;
    out[1] += c;
}

typedef struct {
    /* r^1 */
    uint32_t r0, r1, r2, r3, r4;
    uint64_t s1, s2, s3, s4;
    /* r^2 */
    uint32_t r2_0, r2_1, r2_2, r2_3, r2_4;
    uint64_t s2_1, s2_2, s2_3, s2_4;
    /* r^3 */
    uint32_t r3_0, r3_1, r3_2, r3_3, r3_4;
    uint64_t s3_1, s3_2, s3_3, s3_4;
    /* r^4 */
    uint32_t r4_0, r4_1, r4_2, r4_3, r4_4;
    uint64_t s4_1, s4_2, s4_3, s4_4;
    /* pad and accumulator */
    uint32_t pad0, pad1, pad2, pad3;
    uint32_t h0, h1, h2, h3, h4;
    uint8_t  buffer[16];
    uint8_t  buflen;
    uint8_t  done;
} poly1305_state_t;

void poly1305_init(poly1305_ctx_t* ctx, const uint8_t key[32]) {
    poly1305_state_t* s = (poly1305_state_t*)ctx->opaque;

    /* Load and clamp r */
    s->r0 =  load32_le(&key[0])  & 0x3ffffffu;
    s->r1 = (load32_le(&key[3])  >> 2) & 0x3ffff03u;
    s->r2 = (load32_le(&key[6])  >> 4) & 0x3ffc0ffu;
    s->r3 = (load32_le(&key[9])  >> 6) & 0x3f03fffu;
    s->r4 = (load32_le(&key[12]) >> 8) & 0x00fffffu;
    s->s1 = (uint64_t)s->r1 * 5u;
    s->s2 = (uint64_t)s->r2 * 5u;
    s->s3 = (uint64_t)s->r3 * 5u;
    s->s4 = (uint64_t)s->r4 * 5u;

    /* Precompute r^2 */
    uint32_t r1[5] = {s->r0, s->r1, s->r2, s->r3, s->r4};
    uint64_t sr1[4] = {s->s1, s->s2, s->s3, s->s4};
    uint32_t r2[5];
    square_r_mod(r2, r1, sr1);
    s->r2_0 = r2[0]; s->r2_1 = r2[1]; s->r2_2 = r2[2]; s->r2_3 = r2[3]; s->r2_4 = r2[4];
    s->s2_1 = (uint64_t)r2[1] * 5u;
    s->s2_2 = (uint64_t)r2[2] * 5u;
    s->s2_3 = (uint64_t)r2[3] * 5u;
    s->s2_4 = (uint64_t)r2[4] * 5u;

    /* Precompute r^3 = r^2 * r */
    uint64_t sr2[4] = {s->s2_1, s->s2_2, s->s2_3, s->s2_4};
    uint32_t r3[5];
    mul_r_mod(r3, r2, sr2, r1, sr1);
    s->r3_0 = r3[0]; s->r3_1 = r3[1]; s->r3_2 = r3[2]; s->r3_3 = r3[3]; s->r3_4 = r3[4];
    s->s3_1 = (uint64_t)r3[1] * 5u;
    s->s3_2 = (uint64_t)r3[2] * 5u;
    s->s3_3 = (uint64_t)r3[3] * 5u;
    s->s3_4 = (uint64_t)r3[4] * 5u;

    /* Precompute r^4 = r^2 * r^2 */
    uint32_t r4[5];
    square_r_mod(r4, r2, sr2);
    s->r4_0 = r4[0]; s->r4_1 = r4[1]; s->r4_2 = r4[2]; s->r4_3 = r4[3]; s->r4_4 = r4[4];
    s->s4_1 = (uint64_t)r4[1] * 5u;
    s->s4_2 = (uint64_t)r4[2] * 5u;
    s->s4_3 = (uint64_t)r4[3] * 5u;
    s->s4_4 = (uint64_t)r4[4] * 5u;

    /* Load pad */
    s->pad0 = load32_le(&key[16]);
    s->pad1 = load32_le(&key[20]);
    s->pad2 = load32_le(&key[24]);
    s->pad3 = load32_le(&key[28]);

    /* Initialize accumulator */
    s->h0 = 0;
    s->h1 = 0;
    s->h2 = 0;
    s->h3 = 0;
    s->h4 = 0;
    memset(s->buffer, 0, sizeof s->buffer);
    s->buflen = 0;
    s->done = 0;
}

INLINE void poly1305_block(
    uint32_t *h0, uint32_t *h1, uint32_t *h2, uint32_t *h3, uint32_t *h4,
    uint32_t  r0, uint32_t  r1, uint32_t  r2, uint32_t  r3, uint32_t  r4,
    uint64_t  s1, uint64_t  s2, uint64_t  s3, uint64_t  s4,
    const uint8_t *p)
{
    uint32_t t0, t1, t2, t3, t4;
    uint64_t d0, d1, d2, d3, d4;
    uint32_t c;
    const uint32_t hibit = 1u << 24;

    uint32_t H0 = *h0, H1 = *h1, H2 = *h2, H3 = *h3, H4 = *h4;

    t0 =  load32_le(p + 0)  & 0x3ffffffu;
    t1 = (load32_le(p + 3)  >> 2) & 0x3ffffffu;
    t2 = (load32_le(p + 6)  >> 4) & 0x3ffffffu;
    t3 = (load32_le(p + 9)  >> 6) & 0x3ffffffu;
    t4 = (load32_le(p + 12) >> 8) | hibit;

    H0 += t0; H1 += t1; H2 += t2; H3 += t3; H4 += t4;

    d0 = SUM(MUL(H0,r0), MUL(H1,s4), MUL(H2,s3), MUL(H3,s2), MUL(H4,s1));
    d1 = SUM(MUL(H0,r1), MUL(H1,r0), MUL(H2,s4), MUL(H3,s3), MUL(H4,s2));
    d2 = SUM(MUL(H0,r2), MUL(H1,r1), MUL(H2,r0), MUL(H3,s4), MUL(H4,s3));
    d3 = SUM(MUL(H0,r3), MUL(H1,r2), MUL(H2,r1), MUL(H3,r0), MUL(H4,s4));
    d4 = SUM(MUL(H0,r4), MUL(H1,r3), MUL(H2,r2), MUL(H3,r1), MUL(H4,r0));

    c  = (uint32_t)(d0 >> 26); H0 = (uint32_t)d0 & 0x3ffffffu; d1 += c;
    c  = (uint32_t)(d1 >> 26); H1 = (uint32_t)d1 & 0x3ffffffu; d2 += c;
    c  = (uint32_t)(d2 >> 26); H2 = (uint32_t)d2 & 0x3ffffffu; d3 += c;
    c  = (uint32_t)(d3 >> 26); H3 = (uint32_t)d3 & 0x3ffffffu; d4 += c;
    c  = (uint32_t)(d4 >> 26); H4 = (uint32_t)d4 & 0x3ffffffu;

    H0 += c * 5u;
    c   = H0 >> 26;
    H0 &= 0x3ffffffu;
    H1 += c;

    *h0 = H0; *h1 = H1; *h2 = H2; *h3 = H3; *h4 = H4;
}

/* Load a 16-byte block into radix-2^26 with high bit set */
INLINE void load_block_26(uint32_t t[5], const uint8_t* p) {
    t[0] =  load32_le(p + 0)  & 0x3ffffffu;
    t[1] = (load32_le(p + 3)  >> 2) & 0x3ffffffu;
    t[2] = (load32_le(p + 6)  >> 4) & 0x3ffffffu;
    t[3] = (load32_le(p + 9)  >> 6) & 0x3ffffffu;
    t[4] = (load32_le(p + 12) >> 8) | (1u << 24);
}

/* Process 4 blocks in parallel:
 * h = (h + b1) * r^4 + b2 * r^3 + b3 * r^2 + b4 * r
 */
INLINE void poly1305_4blocks(
    uint32_t *h0, uint32_t *h1, uint32_t *h2, uint32_t *h3, uint32_t *h4,
    /* r^1 */ uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3, uint32_t r4,
              uint64_t s1, uint64_t s2, uint64_t s3, uint64_t s4,
    /* r^2 */ uint32_t r2_0, uint32_t r2_1, uint32_t r2_2, uint32_t r2_3, uint32_t r2_4,
              uint64_t s2_1, uint64_t s2_2, uint64_t s2_3, uint64_t s2_4,
    /* r^3 */ uint32_t r3_0, uint32_t r3_1, uint32_t r3_2, uint32_t r3_3, uint32_t r3_4,
              uint64_t s3_1, uint64_t s3_2, uint64_t s3_3, uint64_t s3_4,
    /* r^4 */ uint32_t r4_0, uint32_t r4_1, uint32_t r4_2, uint32_t r4_3, uint32_t r4_4,
              uint64_t s4_1, uint64_t s4_2, uint64_t s4_3, uint64_t s4_4,
    const uint8_t* p)
{
    uint32_t b1[5], b2[5], b3[5], b4[5];
    load_block_26(b1, p);
    load_block_26(b2, p + 16);
    load_block_26(b3, p + 32);
    load_block_26(b4, p + 48);

    /* Add b1 to accumulator */
    uint32_t H0 = *h0 + b1[0];
    uint32_t H1 = *h1 + b1[1];
    uint32_t H2 = *h2 + b1[2];
    uint32_t H3 = *h3 + b1[3];
    uint32_t H4 = *h4 + b1[4];

    /* Compute (h + b1) * r^4 */
    uint64_t d0 = SUM(MUL(H0,r4_0), MUL(H1,s4_4), MUL(H2,s4_3), MUL(H3,s4_2), MUL(H4,s4_1));
    uint64_t d1 = SUM(MUL(H0,r4_1), MUL(H1,r4_0), MUL(H2,s4_4), MUL(H3,s4_3), MUL(H4,s4_2));
    uint64_t d2 = SUM(MUL(H0,r4_2), MUL(H1,r4_1), MUL(H2,r4_0), MUL(H3,s4_4), MUL(H4,s4_3));
    uint64_t d3 = SUM(MUL(H0,r4_3), MUL(H1,r4_2), MUL(H2,r4_1), MUL(H3,r4_0), MUL(H4,s4_4));
    uint64_t d4 = SUM(MUL(H0,r4_4), MUL(H1,r4_3), MUL(H2,r4_2), MUL(H3,r4_1), MUL(H4,r4_0));

    /* Add b2 * r^3 */
    d0 += SUM(MUL(b2[0],r3_0), MUL(b2[1],s3_4), MUL(b2[2],s3_3), MUL(b2[3],s3_2), MUL(b2[4],s3_1));
    d1 += SUM(MUL(b2[0],r3_1), MUL(b2[1],r3_0), MUL(b2[2],s3_4), MUL(b2[3],s3_3), MUL(b2[4],s3_2));
    d2 += SUM(MUL(b2[0],r3_2), MUL(b2[1],r3_1), MUL(b2[2],r3_0), MUL(b2[3],s3_4), MUL(b2[4],s3_3));
    d3 += SUM(MUL(b2[0],r3_3), MUL(b2[1],r3_2), MUL(b2[2],r3_1), MUL(b2[3],r3_0), MUL(b2[4],s3_4));
    d4 += SUM(MUL(b2[0],r3_4), MUL(b2[1],r3_3), MUL(b2[2],r3_2), MUL(b2[3],r3_1), MUL(b2[4],r3_0));

    /* Add b3 * r^2 */
    d0 += SUM(MUL(b3[0],r2_0), MUL(b3[1],s2_4), MUL(b3[2],s2_3), MUL(b3[3],s2_2), MUL(b3[4],s2_1));
    d1 += SUM(MUL(b3[0],r2_1), MUL(b3[1],r2_0), MUL(b3[2],s2_4), MUL(b3[3],s2_3), MUL(b3[4],s2_2));
    d2 += SUM(MUL(b3[0],r2_2), MUL(b3[1],r2_1), MUL(b3[2],r2_0), MUL(b3[3],s2_4), MUL(b3[4],s2_3));
    d3 += SUM(MUL(b3[0],r2_3), MUL(b3[1],r2_2), MUL(b3[2],r2_1), MUL(b3[3],r2_0), MUL(b3[4],s2_4));
    d4 += SUM(MUL(b3[0],r2_4), MUL(b3[1],r2_3), MUL(b3[2],r2_2), MUL(b3[3],r2_1), MUL(b3[4],r2_0));

    /* Add b4 * r^1 */
    d0 += SUM(MUL(b4[0],r0), MUL(b4[1],s4), MUL(b4[2],s3), MUL(b4[3],s2), MUL(b4[4],s1));
    d1 += SUM(MUL(b4[0],r1), MUL(b4[1],r0), MUL(b4[2],s4), MUL(b4[3],s3), MUL(b4[4],s2));
    d2 += SUM(MUL(b4[0],r2), MUL(b4[1],r1), MUL(b4[2],r0), MUL(b4[3],s4), MUL(b4[4],s3));
    d3 += SUM(MUL(b4[0],r3), MUL(b4[1],r2), MUL(b4[2],r1), MUL(b4[3],r0), MUL(b4[4],s4));
    d4 += SUM(MUL(b4[0],r4), MUL(b4[1],r3), MUL(b4[2],r2), MUL(b4[3],r1), MUL(b4[4],r0));

    /* Carry propagation and reduction */
    uint32_t c;
    c  = (uint32_t)(d0 >> 26); H0 = (uint32_t)d0 & 0x3ffffffu; d1 += c;
    c  = (uint32_t)(d1 >> 26); H1 = (uint32_t)d1 & 0x3ffffffu; d2 += c;
    c  = (uint32_t)(d2 >> 26); H2 = (uint32_t)d2 & 0x3ffffffu; d3 += c;
    c  = (uint32_t)(d3 >> 26); H3 = (uint32_t)d3 & 0x3ffffffu; d4 += c;
    c  = (uint32_t)(d4 >> 26); H4 = (uint32_t)d4 & 0x3ffffffu;

    H0 += c * 5u;
    c   = H0 >> 26;
    H0 &= 0x3ffffffu;
    H1 += c;

    *h0 = H0; *h1 = H1; *h2 = H2; *h3 = H3; *h4 = H4;
}

void poly1305_update(poly1305_ctx_t* ctx, const uint8_t* msg, size_t mlen) {
    poly1305_state_t* s = (poly1305_state_t*)ctx->opaque;

    if (s->done || mlen == 0) return;
    uint8_t buflen = s->buflen;

    uint32_t h0 = s->h0, h1 = s->h1, h2 = s->h2, h3 = s->h3, h4 = s->h4;
    /* r^1 */
    const uint32_t r0 = s->r0, r1 = s->r1, r2 = s->r2, r3 = s->r3, r4 = s->r4;
    const uint64_t s1 = s->s1, s2 = s->s2, s3 = s->s3, s4 = s->s4;
    /* r^2 */
    const uint32_t r2_0 = s->r2_0, r2_1 = s->r2_1, r2_2 = s->r2_2, r2_3 = s->r2_3, r2_4 = s->r2_4;
    const uint64_t s2_1 = s->s2_1, s2_2 = s->s2_2, s2_3 = s->s2_3, s2_4 = s->s2_4;
    /* r^3 */
    const uint32_t r3_0 = s->r3_0, r3_1 = s->r3_1, r3_2 = s->r3_2, r3_3 = s->r3_3, r3_4 = s->r3_4;
    const uint64_t s3_1 = s->s3_1, s3_2 = s->s3_2, s3_3 = s->s3_3, s3_4 = s->s3_4;
    /* r^4 */
    const uint32_t r4_0 = s->r4_0, r4_1 = s->r4_1, r4_2 = s->r4_2, r4_3 = s->r4_3, r4_4 = s->r4_4;
    const uint64_t s4_1 = s->s4_1, s4_2 = s->s4_2, s4_3 = s->s4_3, s4_4 = s->s4_4;

    if (buflen) {
        size_t remain = 16u - buflen;
        if (mlen < remain) {
            memcpy(s->buffer + buflen, msg, mlen);
            s->buflen += (uint8_t)mlen;
            return;
        }
        memcpy(s->buffer + buflen, msg, remain);
        poly1305_block(&h0, &h1, &h2, &h3, &h4, r0, r1, r2, r3, r4, s1, s2, s3, s4, s->buffer);
        msg += remain;
        mlen -= remain;
        s->buflen = 0;
    }

    /* Process 4 blocks at a time for better ILP */
    while (mlen >= 64) {
        poly1305_4blocks(&h0, &h1, &h2, &h3, &h4,
                         r0, r1, r2, r3, r4, s1, s2, s3, s4,
                         r2_0, r2_1, r2_2, r2_3, r2_4, s2_1, s2_2, s2_3, s2_4,
                         r3_0, r3_1, r3_2, r3_3, r3_4, s3_1, s3_2, s3_3, s3_4,
                         r4_0, r4_1, r4_2, r4_3, r4_4, s4_1, s4_2, s4_3, s4_4,
                         msg);
        msg  += 64;
        mlen -= 64;
    }

    /* Handle remaining blocks one at a time */
    while (mlen >= 16) {
        poly1305_block(&h0, &h1, &h2, &h3, &h4, r0, r1, r2, r3, r4, s1, s2, s3, s4, msg);
        msg  += 16;
        mlen -= 16;
    }

    if (mlen > 0) {
        memcpy(s->buffer, msg, mlen);
        s->buflen = (uint8_t)mlen;
    }

    s->h0 = h0; s->h1 = h1; s->h2 = h2; s->h3 = h3; s->h4 = h4;
}

void poly1305_finish(poly1305_ctx_t* ctx, uint8_t tag[16]) {
    poly1305_state_t* s = (poly1305_state_t*)ctx->opaque;

    if (s->done) return;

    uint32_t r0 = s->r0, r1 = s->r1, r2 = s->r2, r3 = s->r3, r4 = s->r4;
    uint64_t s1 = s->s1, s2 = s->s2, s3 = s->s3, s4 = s->s4;
    uint32_t pad0 = s->pad0, pad1 = s->pad1, pad2 = s->pad2, pad3 = s->pad3;
    uint32_t h0 = s->h0, h1 = s->h1, h2 = s->h2, h3 = s->h3, h4 = s->h4;

    if (s->buflen) {
        uint8_t buf[16] = {0};
        uint32_t t0, t1, t2, t3, t4;
        uint64_t d0, d1, d2, d3, d4;
        uint32_t c;
        const uint32_t hibit = 0u;

        for (size_t i = 0; i < s->buflen; i++) buf[i] = s->buffer[i];
        buf[s->buflen] = 1;

        t0 =  load32_le(buf + 0)  & 0x3ffffffu;
        t1 = (load32_le(buf + 3)  >> 2) & 0x3ffffffu;
        t2 = (load32_le(buf + 6)  >> 4) & 0x3ffffffu;
        t3 = (load32_le(buf + 9)  >> 6) & 0x3ffffffu;
        t4 = (load32_le(buf + 12) >> 8) | hibit;

        h0 += t0; h1 += t1; h2 += t2; h3 += t3; h4 += t4;

        d0 = SUM(MUL(h0,r0), MUL(h1,s4), MUL(h2,s3), MUL(h3,s2), MUL(h4,s1));
        d1 = SUM(MUL(h0,r1), MUL(h1,r0), MUL(h2,s4), MUL(h3,s3), MUL(h4,s2));
        d2 = SUM(MUL(h0,r2), MUL(h1,r1), MUL(h2,r0), MUL(h3,s4), MUL(h4,s3));
        d3 = SUM(MUL(h0,r3), MUL(h1,r2), MUL(h2,r1), MUL(h3,r0), MUL(h4,s4));
        d4 = SUM(MUL(h0,r4), MUL(h1,r3), MUL(h2,r2), MUL(h3,r1), MUL(h4,r0));

        c  = (uint32_t)(d0 >> 26); h0 = (uint32_t)d0 & 0x3ffffffu; d1 += c;
        c  = (uint32_t)(d1 >> 26); h1 = (uint32_t)d1 & 0x3ffffffu; d2 += c;
        c  = (uint32_t)(d2 >> 26); h2 = (uint32_t)d2 & 0x3ffffffu; d3 += c;
        c  = (uint32_t)(d3 >> 26); h3 = (uint32_t)d3 & 0x3ffffffu; d4 += c;
        c  = (uint32_t)(d4 >> 26); h4 = (uint32_t)d4 & 0x3ffffffu;

        h0 += c * 5u;
        c   = h0 >> 26;
        h0 &= 0x3ffffffu;
        h1 += c;

        secure_zero(buf, sizeof(buf));
    }

    uint32_t c;
    c  = h1 >> 26; h1 &= 0x3ffffffu; h2 += c;
    c  = h2 >> 26; h2 &= 0x3ffffffu; h3 += c;
    c  = h3 >> 26; h3 &= 0x3ffffffu; h4 += c;
    c  = h4 >> 26; h4 &= 0x3ffffffu; h0 += c * 5u;
    c  = h0 >> 26; h0 &= 0x3ffffffu; h1 += c;

    uint32_t g0 = h0 + 5u;
    c  = g0 >> 26; g0 &= 0x3ffffffu;
    uint32_t g1 = h1 + c;
    c  = g1 >> 26; g1 &= 0x3ffffffu;
    uint32_t g2 = h2 + c;
    c  = g2 >> 26; g2 &= 0x3ffffffu;
    uint32_t g3 = h3 + c;
    c  = g3 >> 26; g3 &= 0x3ffffffu;
    uint32_t g4 = h4 + c - (1u << 26);

    uint32_t mask  = (g4 >> 31) - 1u;
    uint32_t nmask = ~mask;

    h0 = (h0 & nmask) | (g0 & mask);
    h1 = (h1 & nmask) | (g1 & mask);
    h2 = (h2 & nmask) | (g2 & mask);
    h3 = (h3 & nmask) | (g3 & mask);
    h4 = (h4 & nmask) | (g4 & mask);

    uint32_t t0 = ( h0        | (h1 << 26))        & 0xffffffffu;
    uint32_t t1 = ((h1 >> 6)  | (h2 << 20))        & 0xffffffffu;
    uint32_t t2 = ((h2 >> 12) | (h3 << 14))        & 0xffffffffu;
    uint32_t t3 = ((h3 >> 18) | (h4 << 8))         & 0xffffffffu;

    uint64_t f;
    f  = (uint64_t)t0 + pad0;
    t0 = (uint32_t)f;
    f  = (uint64_t)t1 + pad1 + (f >> 32);
    t1 = (uint32_t)f;
    f  = (uint64_t)t2 + pad2 + (f >> 32);
    t2 = (uint32_t)f;
    f  = (uint64_t)t3 + pad3 + (f >> 32);
    t3 = (uint32_t)f;

    store32_le(tag + 0, t0);
    store32_le(tag + 4, t1);
    store32_le(tag + 8, t2);
    store32_le(tag + 12, t3);

    secure_zero(s, sizeof(*s));
    s->done = 1;
}

void poly1305(uint8_t tag[16], const uint8_t* msg, size_t mlen, const uint8_t key[32]) {
    poly1305_ctx_t ctx;
    poly1305_init(&ctx, key);
    poly1305_update(&ctx, msg, mlen);
    poly1305_finish(&ctx, tag);
}
