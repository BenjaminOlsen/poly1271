/* SPDX-License-Identifier: MIT */
/* poly1271 AVX2 - radix-2^26, 4-way parallel */

#ifdef __AVX2__

#include "poly1271.h"
#include <immintrin.h>
#include <string.h>

#define MASK26 ((1ULL << 26) - 1)
#define MASK23 ((1ULL << 23) - 1)

/* -------------------------------------------------------------------------- */
/* radix-26 conversions                                                       */
/* -------------------------------------------------------------------------- */

static inline void decode_block_to_limbs(uint64_t limbs[5], const uint8_t* block) {
    uint64_t lo = (uint64_t)block[0] | ((uint64_t)block[1] << 8) |
                  ((uint64_t)block[2] << 16) | ((uint64_t)block[3] << 24) |
                  ((uint64_t)block[4] << 32) | ((uint64_t)block[5] << 40) |
                  ((uint64_t)block[6] << 48) | ((uint64_t)block[7] << 56);
    uint64_t hi = (uint64_t)block[8] | ((uint64_t)block[9] << 8) |
                  ((uint64_t)block[10] << 16) | ((uint64_t)block[11] << 24) |
                  ((uint64_t)block[12] << 32) | ((uint64_t)block[13] << 40) |
                  ((uint64_t)block[14] << 48) | (1ULL << 56); /* delimiter */

    limbs[0] = lo & MASK26;
    limbs[1] = (lo >> 26) & MASK26;
    limbs[2] = ((lo >> 52) | (hi << 12)) & MASK26;
    limbs[3] = (hi >> 14) & MASK26;
    limbs[4] = (hi >> 40) & MASK26;
}

static inline void decode_partial_to_limbs(uint64_t limbs[5], const uint8_t* block, size_t len) {
    uint8_t pad[16] = {0};
    memcpy(pad, block, len);
    pad[len] = 0x01;

    uint64_t lo = (uint64_t)pad[0] | ((uint64_t)pad[1] << 8) |
                  ((uint64_t)pad[2] << 16) | ((uint64_t)pad[3] << 24) |
                  ((uint64_t)pad[4] << 32) | ((uint64_t)pad[5] << 40) |
                  ((uint64_t)pad[6] << 48) | ((uint64_t)pad[7] << 56);
    uint64_t hi = (uint64_t)pad[8] | ((uint64_t)pad[9] << 8) |
                  ((uint64_t)pad[10] << 16) | ((uint64_t)pad[11] << 24) |
                  ((uint64_t)pad[12] << 32) | ((uint64_t)pad[13] << 40) |
                  ((uint64_t)pad[14] << 48) | ((uint64_t)pad[15] << 56);

    limbs[0] = lo & MASK26;
    limbs[1] = (lo >> 26) & MASK26;
    limbs[2] = ((lo >> 52) | (hi << 12)) & MASK26;
    limbs[3] = (hi >> 14) & MASK26;
    limbs[4] = (hi >> 40) & MASK26;
}

static inline void convert_to_limbs(uint64_t limbs[5], const uint64_t v[2]) {
    uint64_t lo = v[0], hi = v[1];
    limbs[0] = lo & MASK26;
    limbs[1] = (lo >> 26) & MASK26;
    limbs[2] = ((lo >> 52) | (hi << 12)) & MASK26;
    limbs[3] = (hi >> 14) & MASK26;
    limbs[4] = (hi >> 40) & MASK26;
}

static inline void convert_from_limbs(uint64_t v[2], const uint64_t limbs[5]) {
    v[0] = limbs[0] | (limbs[1] << 26) | (limbs[2] << 52);
    v[1] = (limbs[2] >> 12) | (limbs[3] << 14) | (limbs[4] << 40);
}

/* -------------------------------------------------------------------------- */
/* scalar radix-26 arithmetic                                                 */
/* -------------------------------------------------------------------------- */

static inline void carry26_scalar(uint64_t limbs[5]) {
    uint64_t c;
    c = limbs[0] >> 26; limbs[0] &= MASK26; limbs[1] += c;
    c = limbs[1] >> 26; limbs[1] &= MASK26; limbs[2] += c;
    c = limbs[2] >> 26; limbs[2] &= MASK26; limbs[3] += c;
    c = limbs[3] >> 26; limbs[3] &= MASK26; limbs[4] += c;
}

/* reduce mod 2^127-1; bit 127 = limb4 bit 23 */
static inline void reduce_p_scalar(uint64_t limbs[5]) {
    uint64_t c = limbs[4] >> 23;
    limbs[4] &= MASK23;
    limbs[0] += c;
    carry26_scalar(limbs);
    c = limbs[4] >> 23;
    limbs[4] &= MASK23;
    limbs[0] += c;
    carry26_scalar(limbs);
}

/* schoolbook multiply mod p */
static void mul_limbs_scalar(uint64_t out[5], const uint64_t a[5], const uint64_t b[5]) {
    uint64_t d0 = a[0] * b[0];
    uint64_t d1 = a[0] * b[1] + a[1] * b[0];
    uint64_t d2 = a[0] * b[2] + a[1] * b[1] + a[2] * b[0];
    uint64_t d3 = a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0];
    uint64_t d4 = a[0] * b[4] + a[1] * b[3] + a[2] * b[2] + a[3] * b[1] + a[4] * b[0];
    uint64_t d5 = a[1] * b[4] + a[2] * b[3] + a[3] * b[2] + a[4] * b[1];
    uint64_t d6 = a[2] * b[4] + a[3] * b[3] + a[4] * b[2];
    uint64_t d7 = a[3] * b[4] + a[4] * b[3];
    uint64_t d8 = a[4] * b[4];

    /* fold: 2^130 = 8 (mod p) */
    d0 += d5 << 3;
    d1 += d6 << 3;
    d2 += d7 << 3;
    d3 += d8 << 3;

    uint64_t c;
    c = d0 >> 26; d0 &= MASK26; d1 += c;
    c = d1 >> 26; d1 &= MASK26; d2 += c;
    c = d2 >> 26; d2 &= MASK26; d3 += c;
    c = d3 >> 26; d3 &= MASK26; d4 += c;

    out[0] = d0; out[1] = d1; out[2] = d2; out[3] = d3; out[4] = d4;
    reduce_p_scalar(out);
}

static void square_limbs_scalar(uint64_t out[5], const uint64_t a[5]) {
    uint64_t d0 = a[0] * a[0];
    uint64_t d1 = 2 * a[0] * a[1];
    uint64_t d2 = 2 * a[0] * a[2] + a[1] * a[1];
    uint64_t d3 = 2 * a[0] * a[3] + 2 * a[1] * a[2];
    uint64_t d4 = 2 * a[0] * a[4] + 2 * a[1] * a[3] + a[2] * a[2];
    uint64_t d5 = 2 * a[1] * a[4] + 2 * a[2] * a[3];
    uint64_t d6 = 2 * a[2] * a[4] + a[3] * a[3];
    uint64_t d7 = 2 * a[3] * a[4];
    uint64_t d8 = a[4] * a[4];

    d0 += d5 << 3;
    d1 += d6 << 3;
    d2 += d7 << 3;
    d3 += d8 << 3;

    uint64_t c;
    c = d0 >> 26; d0 &= MASK26; d1 += c;
    c = d1 >> 26; d1 &= MASK26; d2 += c;
    c = d2 >> 26; d2 &= MASK26; d3 += c;
    c = d3 >> 26; d3 &= MASK26; d4 += c;

    out[0] = d0; out[1] = d1; out[2] = d2; out[3] = d3; out[4] = d4;
    reduce_p_scalar(out);
}

/* -------------------------------------------------------------------------- */
/* AVX2 helpers                                                               
*/

static inline uint64_t load64_le(const uint8_t* p) {
    return (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) |
           ((uint64_t)p[3] << 24) | ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

static inline void store64_le(uint8_t* p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (i * 8));
}

static void clamp_r(uint8_t r[16]) {
    r[3] &= 0x0F; r[7] &= 0x0F; r[11] &= 0x0F; r[15] &= 0x07;
    r[4] &= 0xFC; r[8] &= 0xFC; r[12] &= 0xFC;
}

static void secure_zero(void* ptr, size_t len) {
    volatile uint8_t* p = (volatile uint8_t*)ptr;
    while (len--) *p++ = 0;
}

/* multiply 4 parallel accumulators by r^4 (broadcast) */
static inline void mul_r4_avx2(__m256i acc[5], const uint64_t r4[5]) {
    __m256i r0 = _mm256_set1_epi64x((int64_t)r4[0]);
    __m256i r1 = _mm256_set1_epi64x((int64_t)r4[1]);
    __m256i r2 = _mm256_set1_epi64x((int64_t)r4[2]);
    __m256i r3 = _mm256_set1_epi64x((int64_t)r4[3]);
    __m256i r4v = _mm256_set1_epi64x((int64_t)r4[4]);

    __m256i a0 = acc[0], a1 = acc[1], a2 = acc[2], a3 = acc[3], a4 = acc[4];

    __m256i d0 = _mm256_mul_epu32(a0, r0);
    __m256i d1 = _mm256_add_epi64(_mm256_mul_epu32(a0, r1), _mm256_mul_epu32(a1, r0));
    __m256i d2 = _mm256_add_epi64(_mm256_add_epi64(_mm256_mul_epu32(a0, r2), _mm256_mul_epu32(a1, r1)),
                                  _mm256_mul_epu32(a2, r0));
    __m256i d3 = _mm256_add_epi64(_mm256_add_epi64(_mm256_mul_epu32(a0, r3), _mm256_mul_epu32(a1, r2)),
                                  _mm256_add_epi64(_mm256_mul_epu32(a2, r1), _mm256_mul_epu32(a3, r0)));
    __m256i d4 = _mm256_add_epi64(_mm256_add_epi64(_mm256_mul_epu32(a0, r4v), _mm256_mul_epu32(a1, r3)),
                                  _mm256_add_epi64(_mm256_add_epi64(_mm256_mul_epu32(a2, r2), _mm256_mul_epu32(a3, r1)),
                                                   _mm256_mul_epu32(a4, r0)));
    __m256i d5 = _mm256_add_epi64(_mm256_add_epi64(_mm256_mul_epu32(a1, r4v), _mm256_mul_epu32(a2, r3)),
                                  _mm256_add_epi64(_mm256_mul_epu32(a3, r2), _mm256_mul_epu32(a4, r1)));
    __m256i d6 = _mm256_add_epi64(_mm256_add_epi64(_mm256_mul_epu32(a2, r4v), _mm256_mul_epu32(a3, r3)),
                                  _mm256_mul_epu32(a4, r2));
    __m256i d7 = _mm256_add_epi64(_mm256_mul_epu32(a3, r4v), _mm256_mul_epu32(a4, r3));
    __m256i d8 = _mm256_mul_epu32(a4, r4v);

    d0 = _mm256_add_epi64(d0, _mm256_slli_epi64(d5, 3));
    d1 = _mm256_add_epi64(d1, _mm256_slli_epi64(d6, 3));
    d2 = _mm256_add_epi64(d2, _mm256_slli_epi64(d7, 3));
    d3 = _mm256_add_epi64(d3, _mm256_slli_epi64(d8, 3));

    __m256i mask26 = _mm256_set1_epi64x(MASK26);
    __m256i mask23 = _mm256_set1_epi64x(MASK23);
    __m256i c;

    c = _mm256_srli_epi64(d0, 26); d0 = _mm256_and_si256(d0, mask26); d1 = _mm256_add_epi64(d1, c);
    c = _mm256_srli_epi64(d1, 26); d1 = _mm256_and_si256(d1, mask26); d2 = _mm256_add_epi64(d2, c);
    c = _mm256_srli_epi64(d2, 26); d2 = _mm256_and_si256(d2, mask26); d3 = _mm256_add_epi64(d3, c);
    c = _mm256_srli_epi64(d3, 26); d3 = _mm256_and_si256(d3, mask26); d4 = _mm256_add_epi64(d4, c);
    c = _mm256_srli_epi64(d4, 23); d4 = _mm256_and_si256(d4, mask23); d0 = _mm256_add_epi64(d0, c);
    c = _mm256_srli_epi64(d0, 26); d0 = _mm256_and_si256(d0, mask26); d1 = _mm256_add_epi64(d1, c);

    acc[0] = d0; acc[1] = d1; acc[2] = d2; acc[3] = d3; acc[4] = d4;
}

static inline void add_4blocks_avx2(__m256i acc[5],
                                     const uint64_t m0[5], const uint64_t m1[5],
                                     const uint64_t m2[5], const uint64_t m3[5]) {
    for (int i = 0; i < 5; i++) {
        __m256i mv = _mm256_set_epi64x((int64_t)m3[i], (int64_t)m2[i],
                                        (int64_t)m1[i], (int64_t)m0[i]);
        acc[i] = _mm256_add_epi64(acc[i], mv);
    }
}

/* combine: lane0*r^4 + lane1*r^3 + lane2*r^2 + lane3*r */
static void combine_4lanes_avx2(uint64_t out[5], const __m256i acc[5], const __m256i rv[5]) {
    __m256i d[9];
    for (int k = 0; k < 9; k++) d[k] = _mm256_setzero_si256();

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            __m256i prod = _mm256_mul_epu32(acc[i], rv[j]);
            d[i + j] = _mm256_add_epi64(d[i + j], prod);
        }
    }

    d[0] = _mm256_add_epi64(d[0], _mm256_slli_epi64(d[5], 3));
    d[1] = _mm256_add_epi64(d[1], _mm256_slli_epi64(d[6], 3));
    d[2] = _mm256_add_epi64(d[2], _mm256_slli_epi64(d[7], 3));
    d[3] = _mm256_add_epi64(d[3], _mm256_slli_epi64(d[8], 3));

    __m256i mask26 = _mm256_set1_epi64x(MASK26);
    __m256i mask23 = _mm256_set1_epi64x(MASK23);
    __m256i c;

    c = _mm256_srli_epi64(d[0], 26); d[0] = _mm256_and_si256(d[0], mask26); d[1] = _mm256_add_epi64(d[1], c);
    c = _mm256_srli_epi64(d[1], 26); d[1] = _mm256_and_si256(d[1], mask26); d[2] = _mm256_add_epi64(d[2], c);
    c = _mm256_srli_epi64(d[2], 26); d[2] = _mm256_and_si256(d[2], mask26); d[3] = _mm256_add_epi64(d[3], c);
    c = _mm256_srli_epi64(d[3], 26); d[3] = _mm256_and_si256(d[3], mask26); d[4] = _mm256_add_epi64(d[4], c);
    c = _mm256_srli_epi64(d[4], 23); d[4] = _mm256_and_si256(d[4], mask23); d[0] = _mm256_add_epi64(d[0], c);
    c = _mm256_srli_epi64(d[0], 26); d[0] = _mm256_and_si256(d[0], mask26); d[1] = _mm256_add_epi64(d[1], c);

    for (int i = 0; i < 5; i++) {
        __m128i lo = _mm256_castsi256_si128(d[i]);
        __m128i hi = _mm256_extracti128_si256(d[i], 1);
        __m128i sum = _mm_add_epi64(lo, hi);
        out[i] = (uint64_t)_mm_extract_epi64(sum, 0) + (uint64_t)_mm_extract_epi64(sum, 1);
    }

    carry26_scalar(out);
    reduce_p_scalar(out);
}

/* process n rounds of 4 blocks, combine once at end */
static void process_nblocks_avx2(uint64_t acc[5], const uint8_t* msg, int nrounds,
                                  const uint64_t r4[5], const __m256i rv[5]) {
    __m256i pacc[5];
    for (int i = 0; i < 5; i++)
        pacc[i] = _mm256_set_epi64x(0, 0, 0, (int64_t)acc[i]);

    for (int round = 0; round < nrounds; round++) {
        uint64_t b0[5], b1[5], b2[5], b3[5];
        decode_block_to_limbs(b0, msg + (round * 4 + 0) * 15);
        decode_block_to_limbs(b1, msg + (round * 4 + 1) * 15);
        decode_block_to_limbs(b2, msg + (round * 4 + 2) * 15);
        decode_block_to_limbs(b3, msg + (round * 4 + 3) * 15);

        add_4blocks_avx2(pacc, b0, b1, b2, b3);

        if (round < nrounds - 1)
            mul_r4_avx2(pacc, r4);
    }

    combine_4lanes_avx2(acc, pacc, rv);
}

static inline void process_4blocks_avx2(uint64_t acc[5], const uint8_t* msg,
                                         const uint64_t r4[5], const __m256i rv[5]) {
    process_nblocks_avx2(acc, msg, 1, r4, rv);
}

static inline void process_16blocks_avx2(uint64_t acc[5], const uint8_t* msg,
                                          const uint64_t r4[5], const __m256i rv[5]) {
    process_nblocks_avx2(acc, msg, 4, r4, rv);
}

static inline void process_32blocks_avx2(uint64_t acc[5], const uint8_t* msg,
                                          const uint64_t r4[5], const __m256i rv[5]) {
    process_nblocks_avx2(acc, msg, 8, r4, rv);
}

static void process_1block_scalar(uint64_t acc[5], const uint8_t* m, const uint64_t r[5]) {
    uint64_t b[5];
    decode_block_to_limbs(b, m);
    for (int i = 0; i < 5; i++) acc[i] += b[i];
    carry26_scalar(acc);
    mul_limbs_scalar(acc, acc, r);
}

static void finalize_avx2(uint64_t acc[5], const uint64_t s[2], uint8_t tag[16]) {
    reduce_p_scalar(acc);

    uint64_t v[2];
    convert_from_limbs(v, acc);

    /* ct check if acc == p */
    uint64_t eq_lo = ~(v[0] ^ 0xFFFFFFFFFFFFFFFFULL);
    eq_lo &= eq_lo >> 32; eq_lo &= eq_lo >> 16; eq_lo &= eq_lo >> 8;
    eq_lo &= eq_lo >> 4;  eq_lo &= eq_lo >> 2;  eq_lo &= eq_lo >> 1;

    uint64_t eq_hi = ~(v[1] ^ 0x7FFFFFFFFFFFFFFFULL);
    eq_hi &= eq_hi >> 32; eq_hi &= eq_hi >> 16; eq_hi &= eq_hi >> 8;
    eq_hi &= eq_hi >> 4;  eq_hi &= eq_hi >> 2;  eq_hi &= eq_hi >> 1;

    uint64_t wrap = (uint64_t)0 - (eq_lo & eq_hi & 1);
    v[0] += wrap & 1;
    uint64_t c = (v[0] == 0) & (wrap & 1);
    v[1] = (v[1] + c) & 0x7FFFFFFFFFFFFFFFULL;

    uint64_t t0 = v[0] + s[0];
    c = (t0 < v[0]);
    uint64_t t1 = v[1] + s[1] + c;

    store64_le(tag, t0);
    store64_le(tag + 8, t1);
}

/* -------------------------------------------------------------------------- */
/* public API                                                                 */
/* -------------------------------------------------------------------------- */

void poly1271_avx2_init(poly1271_avx2_ctx_t* ctx, const uint8_t key[32]) {
    uint8_t rc[16];
    memcpy(rc, key, 16);
    clamp_r(rc);

    uint64_t r64[2] = {load64_le(rc), load64_le(rc + 8)};
    convert_to_limbs(ctx->r, r64);

    square_limbs_scalar(ctx->r2, ctx->r);
    mul_limbs_scalar(ctx->r3, ctx->r2, ctx->r);
    mul_limbs_scalar(ctx->r4, ctx->r2, ctx->r2);

    /* rv[i] = {r^4, r^3, r^2, r^1} per limb for combine */
    for (int i = 0; i < 5; i++) {
        ctx->rv[i] = _mm256_set_epi64x((int64_t)ctx->r[i], (int64_t)ctx->r2[i],
                                        (int64_t)ctx->r3[i], (int64_t)ctx->r4[i]);
    }

    ctx->s[0] = load64_le(key + 16);
    ctx->s[1] = load64_le(key + 24);

    for (int i = 0; i < 5; i++) ctx->acc[i] = 0;
    ctx->buflen = 0;

    secure_zero(rc, 16);
}

void poly1271_avx2_update(poly1271_avx2_ctx_t* ctx, const uint8_t* msg, size_t len) {
    if (ctx->buflen) {
        size_t need = 15 - ctx->buflen;
        if (len < need) {
            memcpy(ctx->buf + ctx->buflen, msg, len);
            ctx->buflen += (uint8_t)len;
            return;
        }
        memcpy(ctx->buf + ctx->buflen, msg, need);
        process_1block_scalar(ctx->acc, ctx->buf, ctx->r);
        msg += need;
        len -= need;
        ctx->buflen = 0;
    }

    while (len >= 480) {
        process_32blocks_avx2(ctx->acc, msg, ctx->r4, ctx->rv);
        msg += 480;
        len -= 480;
    }

    while (len >= 240) {
        process_16blocks_avx2(ctx->acc, msg, ctx->r4, ctx->rv);
        msg += 240;
        len -= 240;
    }

    while (len >= 60) {
        process_4blocks_avx2(ctx->acc, msg, ctx->r4, ctx->rv);
        msg += 60;
        len -= 60;
    }

    while (len >= 15) {
        process_1block_scalar(ctx->acc, msg, ctx->r);
        msg += 15;
        len -= 15;
    }

    if (len) {
        memcpy(ctx->buf, msg, len);
        ctx->buflen = (uint8_t)len;
    }
}

void poly1271_avx2_finish(poly1271_avx2_ctx_t* ctx, uint8_t tag[16]) {
    if (ctx->buflen) {
        uint64_t b[5];
        decode_partial_to_limbs(b, ctx->buf, ctx->buflen);
        for (int i = 0; i < 5; i++) ctx->acc[i] += b[i];
        carry26_scalar(ctx->acc);
        mul_limbs_scalar(ctx->acc, ctx->acc, ctx->r);
    }
    finalize_avx2(ctx->acc, ctx->s, tag);
    secure_zero(ctx, sizeof(*ctx));
}

void poly1271_avx2(uint8_t tag[16], const uint8_t* msg, size_t len, const uint8_t key[32]) {
    poly1271_avx2_ctx_t ctx;
    poly1271_avx2_init(&ctx, key);
    poly1271_avx2_update(&ctx, msg, len);
    poly1271_avx2_finish(&ctx, tag);
}

int poly1271_avx2_verify(const uint8_t tag[16], const uint8_t* msg, size_t len, const uint8_t key[32]) {
    uint8_t computed[16];
    poly1271_avx2(computed, msg, len, key);
    uint8_t diff = 0;
    for (int i = 0; i < 16; i++) diff |= computed[i] ^ tag[i];
    secure_zero(computed, 16);
    return diff ? -1 : 0;
}

#endif /* __AVX2__ */
