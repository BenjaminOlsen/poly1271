/* SPDX-License-Identifier: MIT */

#include "poly1271.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

static void print_hex(const char* label, const uint8_t* buf, size_t len) {
    printf("%s: ", label);
    for (size_t i = 0; i < len; i++) printf("%02x", buf[i]);
    printf("\n");
}

static int hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

static void hex_to_bytes(const char* hex, uint8_t* out, size_t len) {
    for (size_t i = 0; i < len; i++)
        out[i] = (uint8_t)((hex_nibble(hex[2*i]) << 4) | hex_nibble(hex[2*i+1]));
}

static int check_vector(const char* name, const char* key_hex,
                        const uint8_t* msg, size_t msg_len,
                        const char* expected_hex) {
    uint8_t key[32], expected[16], tag[16];

    hex_to_bytes(key_hex, key, 32);
    hex_to_bytes(expected_hex, expected, 16);

    poly1271(tag, msg, msg_len, key);

    if (memcmp(tag, expected, 16) != 0) {
        printf("FAIL %s\n", name);
        print_hex("  expected", expected, 16);
        print_hex("  got     ", tag, 16);
        return 0;
    }

    /* also test verify */
    if (poly1271_verify(tag, msg, msg_len, key) != 0) {
        printf("FAIL %s (verify returned failure for valid tag)\n", name);
        return 0;
    }

    /* modify tag, verify should fail */
    tag[0] ^= 1;
    if (poly1271_verify(tag, msg, msg_len, key) == 0) {
        printf("FAIL %s (verify returned success for invalid tag)\n", name);
        return 0;
    }

    printf("  %s: ok\n", name);
    return 1;
}

static void test_empty_message(void) {
    printf("test_empty_message\n");
    uint8_t key[32] = {
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
        0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
        0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20
    };
    uint8_t tag1[16], tag2[16];

    poly1271(tag1, NULL, 0, key);
    poly1271(tag2, NULL, 0, key);

    print_hex("  empty tag", tag1, 16);
    ASSERT(memcmp(tag1, tag2, 16) == 0);
}

static void test_single_block(void) {
    printf("test_single_block\n");
    uint8_t key[32] = {0};
    for (int i = 0; i < 32; i++) key[i] = (uint8_t)i;

    uint8_t msg[15] = "Hello, Poly127";
    uint8_t tag1[16], tag2[16];

    poly1271(tag1, msg, 15, key);
    poly1271(tag2, msg, 15, key);

    print_hex("  single block tag", tag1, 16);
    ASSERT(memcmp(tag1, tag2, 16) == 0);
}

static void test_streaming(void) {
    printf("test_streaming\n");
    uint8_t key[32];
    for (int i = 0; i < 32; i++) key[i] = (uint8_t)(i ^ 0xAA);

    uint8_t msg[100];
    for (int i = 0; i < 100; i++) msg[i] = (uint8_t)(i * 7);

    uint8_t tag_oneshot[16];
    poly1271(tag_oneshot, msg, 100, key);

    size_t chunk_sizes[] = {1, 3, 7, 13, 15, 16, 17, 31, 50};
    for (size_t c = 0; c < sizeof(chunk_sizes)/sizeof(chunk_sizes[0]); c++) {
        size_t chunk = chunk_sizes[c];
        poly1271_ctx_t ctx;
        poly1271_init(&ctx, key);

        size_t pos = 0;
        while (pos < 100) {
            size_t n = (pos + chunk > 100) ? (100 - pos) : chunk;
            poly1271_update(&ctx, msg + pos, n);
            pos += n;
        }

        uint8_t tag_stream[16];
        poly1271_finish(&ctx, tag_stream);

        ASSERT(memcmp(tag_oneshot, tag_stream, 16) == 0);
    }
    printf("  streaming: ok (all chunk sizes)\n");
}

static void test_different_messages(void) {
    printf("test_different_messages\n");
    uint8_t key[32] = {0};
    for (int i = 0; i < 32; i++) key[i] = (uint8_t)i;

    uint8_t msg1[20] = "message one here!!!";
    uint8_t msg2[20] = "message two here!!!";

    uint8_t tag1[16], tag2[16];
    poly1271(tag1, msg1, 20, key);
    poly1271(tag2, msg2, 20, key);

    ASSERT(memcmp(tag1, tag2, 16) != 0);
    printf("  different messages produce different tags: ok\n");
}

static void test_different_keys(void) {
    printf("test_different_keys\n");
    uint8_t key1[32] = {0};
    uint8_t key2[32] = {0};
    for (int i = 0; i < 32; i++) {
        key1[i] = (uint8_t)i;
        key2[i] = (uint8_t)(i + 1);
    }

    uint8_t msg[30] = "test message for key diff!!!";
    uint8_t tag1[16], tag2[16];

    poly1271(tag1, msg, 30, key1);
    poly1271(tag2, msg, 30, key2);

    ASSERT(memcmp(tag1, tag2, 16) != 0);
    printf("  different keys produce different tags: ok\n");
}

static void test_reference_vectors(void) {
    printf("test_reference_vectors\n");
    int passed = 1;

    /* length 0 */
    passed &= check_vector("len_0",
        "b5739948a249856c49e54909ebb2f31d497377aea932d57ae80e81139e4bb6dd",
        NULL, 0,
        "497377aea932d57ae80e81139e4bb6dd");

    /* length 1: msg = 0x0d */
    {
        uint8_t msg[] = {0x0d};
        passed &= check_vector("len_1",
            "f298f36369b075bf9168ceda5e9500704ac3c1475720556168946cb49dfcf6d3",
            msg, 1,
            "9479b96ea37dff9fc873500f55ee93d4");
    }

    /* length 15 (exactly one block) */
    {
        uint8_t msg[15];
        for (int i = 0; i < 15; i++) msg[i] = (uint8_t)((i * 7 + 13) & 0xff);
        passed &= check_vector("len_15",
            "f6118ce0bcdae1277b9b82025e5aeb46c9c4015d0f8b0ea4cdb292950dbca174",
            msg, 15,
            "00bd5684bbb9ec0fdb6812fec2b683da");
    }

    /* length 16 (one block + 1 byte) */
    {
        uint8_t msg[16];
        for (int i = 0; i < 16; i++) msg[i] = (uint8_t)((i * 7 + 13) & 0xff);
        passed &= check_vector("len_16",
            "bdce2ecf9a9f4388b2e3c055eb3940f7f6ecfcd6405806aa01b11984937b4a5f",
            msg, 16,
            "1a7e08badf6a311a1326e180535fe5b0");
    }

    /* length 30 (exactly two blocks) */
    {
        uint8_t msg[30];
        for (int i = 0; i < 30; i++) msg[i] = (uint8_t)((i * 7 + 13) & 0xff);
        passed &= check_vector("len_30",
            "cef0ca2975a63992cd201d7a474f4f37d284bc4c4c402ccecafe52f8eac671d3",
            msg, 30,
            "d64d14c694fc3ade68fa7e6ee9862108");
    }

    /* length 100 */
    {
        uint8_t msg[100];
        for (int i = 0; i < 100; i++) msg[i] = (uint8_t)((i * 7 + 13) & 0xff);
        passed &= check_vector("len_100",
            "b0fdf754bb59965e413060dedf8ed9b0d96e8c23544792dca17e89998da75a3a",
            msg, 100,
            "68977a391e9bdf4d904fdb851cde1599");
    }

    /* length 256 */
    {
        uint8_t msg[256];
        for (int i = 0; i < 256; i++) msg[i] = (uint8_t)((i * 7 + 13) & 0xff);
        passed &= check_vector("len_256",
            "595f063a7941f6cb8b74934e31a7c0f477b7240be9f726a321ee09b53eadf174",
            msg, 256,
            "1fff8a650754ea5eb55963a614335ab0");
    }

    /* length 1000 */
    {
        uint8_t* msg = (uint8_t*)malloc(1000);
        ASSERT(msg != NULL);
        for (int i = 0; i < 1000; i++) msg[i] = (uint8_t)((i * 7 + 13) & 0xff);
        passed &= check_vector("len_1000",
            "5476a825f45840087522474e83e1ec8d70a34954735d043659b18aa9244668cf",
            msg, 1000,
            "53d0bba62df4206e277f411f099fa60c");
        free(msg);
    }

    /* Walt Whitman quote */
    {
        const char* uncle_walt =
            "Answer.\n"
            "That you are here-that life exists and identity,\n"
            "That the powerful play goes on, and you may contribute a verse.";
        passed &= check_vector("uncle_walt",
            "c20ae296d65cb1ef04ef2cd32acdeb835f9bfacd77a2ff311b1e573f9f949677",
            (const uint8_t*)uncle_walt, 120,
            "e0840ca53a6744daa78e6d9c65c3bbe9");
    }

    ASSERT(passed);
}

static void test_verify(void) {
    printf("test_verify\n");
    uint8_t key[32];
    for (int i = 0; i < 32; i++) key[i] = (uint8_t)(i * 3);

    uint8_t msg[] = "Test message for verification";
    uint8_t tag[16];

    poly1271(tag, msg, sizeof(msg) - 1, key);

    /* valid tag should verify */
    ASSERT(poly1271_verify(tag, msg, sizeof(msg) - 1, key) == 0);

    /* modified tag should fail */
    tag[5] ^= 0x01;
    ASSERT(poly1271_verify(tag, msg, sizeof(msg) - 1, key) != 0);

    /* restore tag, modify message */
    tag[5] ^= 0x01;
    msg[0] ^= 0x01;
    ASSERT(poly1271_verify(tag, msg, sizeof(msg) - 1, key) != 0);

    printf("  verify: ok\n");
}

/* test block boundary lengths with sequential bytes (00 01 02 ...) */
static void test_block_boundaries(void) {
    printf("test_block_boundaries\n");
    uint8_t key[32];
    for (int i = 0; i < 32; i++) key[i] = (uint8_t)i;

    /* lengths around 15-byte block boundary */
    size_t lens[] = {1, 14, 15, 16, 29, 30, 31};

    for (size_t i = 0; i < sizeof(lens)/sizeof(lens[0]); i++) {
        size_t len = lens[i];
        uint8_t msg[32];
        for (size_t j = 0; j < len; j++) msg[j] = (uint8_t)j;

        uint8_t tag1[16], tag2[16];
        poly1271(tag1, msg, len, key);
        poly1271(tag2, msg, len, key);
        ASSERT(memcmp(tag1, tag2, 16) == 0);

        /* verify streaming produces same result */
        poly1271_ctx_t ctx;
        poly1271_init(&ctx, key);
        poly1271_update(&ctx, msg, len);
        poly1271_finish(&ctx, tag2);
        ASSERT(memcmp(tag1, tag2, 16) == 0);

        printf("  len_%zu: ok\n", len);
    }
}

/* test streaming equivalence: same message, different chunking */
static void test_streaming_equivalence(void) {
    printf("test_streaming_equivalence\n");
    uint8_t key[32];
    for (int i = 0; i < 32; i++) key[i] = (uint8_t)i;

    /* 47 bytes = 3 full blocks + 2 partial */
    uint8_t msg[47];
    for (int i = 0; i < 47; i++) msg[i] = (uint8_t)i;

    uint8_t tag_oneshot[16];
    poly1271(tag_oneshot, msg, 47, key);

    /* chunking: 1 + 46 */
    {
        poly1271_ctx_t ctx;
        poly1271_init(&ctx, key);
        poly1271_update(&ctx, msg, 1);
        poly1271_update(&ctx, msg + 1, 46);
        uint8_t tag[16];
        poly1271_finish(&ctx, tag);
        ASSERT(memcmp(tag_oneshot, tag, 16) == 0);
    }

    /* chunking: 14 + 1 + 15 + 17 (nasty) */
    {
        poly1271_ctx_t ctx;
        poly1271_init(&ctx, key);
        poly1271_update(&ctx, msg, 14);
        poly1271_update(&ctx, msg + 14, 1);
        poly1271_update(&ctx, msg + 15, 15);
        poly1271_update(&ctx, msg + 30, 17);
        uint8_t tag[16];
        poly1271_finish(&ctx, tag);
        ASSERT(memcmp(tag_oneshot, tag, 16) == 0);
    }

    /* byte-at-a-time */
    {
        poly1271_ctx_t ctx;
        poly1271_init(&ctx, key);
        for (int i = 0; i < 47; i++)
            poly1271_update(&ctx, msg + i, 1);
        uint8_t tag[16];
        poly1271_finish(&ctx, tag);
        ASSERT(memcmp(tag_oneshot, tag, 16) == 0);
    }

    printf("  47 bytes, all chunking patterns: ok\n");
}

/* test multi-block loop boundaries (2-block and 4-block paths) */
static void test_multiblock_boundaries(void) {
    printf("test_multiblock_boundaries\n");
    uint8_t key[32];
    for (int i = 0; i < 32; i++) key[i] = (uint8_t)i;

    /* len=30: exactly 2 blocks */
    /* len=45: 3 blocks (2-block path + 1-block path) */
    /* len=60: exactly 4 blocks */
    /* len=61: 4 blocks + 1 byte partial */
    size_t lens[] = {30, 45, 60, 61};

    for (size_t i = 0; i < sizeof(lens)/sizeof(lens[0]); i++) {
        size_t len = lens[i];
        uint8_t* msg = (uint8_t*)malloc(len);
        ASSERT(msg != NULL);
        for (size_t j = 0; j < len; j++) msg[j] = (uint8_t)(j & 0xff);

        uint8_t tag1[16], tag2[16];
        poly1271(tag1, msg, len, key);

        /* verify via streaming byte-at-a-time */
        poly1271_ctx_t ctx;
        poly1271_init(&ctx, key);
        for (size_t j = 0; j < len; j++)
            poly1271_update(&ctx, msg + j, 1);
        poly1271_finish(&ctx, tag2);
        ASSERT(memcmp(tag1, tag2, 16) == 0);

        printf("  len_%zu: ok\n", len);
        free(msg);
    }
}

/* carry stress: all-0x00 and all-0xFF messages */
static void test_carry_stress(void) {
    printf("test_carry_stress\n");
    uint8_t key[32];
    for (int i = 0; i < 32; i++) key[i] = (uint8_t)i;

    /* 60 bytes of 0x00 */
    {
        uint8_t msg[60] = {0};
        uint8_t tag1[16], tag2[16];
        poly1271(tag1, msg, 60, key);
        /* streaming verification */
        poly1271_ctx_t ctx;
        poly1271_init(&ctx, key);
        for (int i = 0; i < 60; i++)
            poly1271_update(&ctx, msg + i, 1);
        poly1271_finish(&ctx, tag2);
        ASSERT(memcmp(tag1, tag2, 16) == 0);
        printf("  60x 0x00: ok\n");
    }

    /* 60 bytes of 0xFF */
    {
        uint8_t msg[60];
        memset(msg, 0xFF, 60);
        uint8_t tag1[16], tag2[16];
        poly1271(tag1, msg, 60, key);
        poly1271_ctx_t ctx;
        poly1271_init(&ctx, key);
        for (int i = 0; i < 60; i++)
            poly1271_update(&ctx, msg + i, 1);
        poly1271_finish(&ctx, tag2);
        ASSERT(memcmp(tag1, tag2, 16) == 0);
        printf("  60x 0xFF: ok\n");
    }

    /* 61 bytes of 0xFF (partial tail) */
    {
        uint8_t msg[61];
        memset(msg, 0xFF, 61);
        uint8_t tag1[16], tag2[16];
        poly1271(tag1, msg, 61, key);
        poly1271_ctx_t ctx;
        poly1271_init(&ctx, key);
        for (int i = 0; i < 61; i++)
            poly1271_update(&ctx, msg + i, 1);
        poly1271_finish(&ctx, tag2);
        ASSERT(memcmp(tag1, tag2, 16) == 0);
        printf("  61x 0xFF: ok\n");
    }
}

/* ========================================================================
 * AVX2 tests
 * ======================================================================== */

#ifdef __AVX2__

static void test_avx2_vs_scalar(void) {
    printf("test_avx2_vs_scalar\n");
    uint8_t key[32];
    for (int i = 0; i < 32; i++) key[i] = (uint8_t)(i * 3 + 7);

    /* test various lengths */
    size_t lens[] = {0, 1, 14, 15, 16, 29, 30, 31, 45, 59, 60, 61, 100, 256, 1000};

    for (size_t i = 0; i < sizeof(lens)/sizeof(lens[0]); i++) {
        size_t len = lens[i];
        uint8_t* msg = len > 0 ? (uint8_t*)malloc(len) : NULL;
        if (len > 0) {
            ASSERT(msg != NULL);
            for (size_t j = 0; j < len; j++) msg[j] = (uint8_t)((j * 7 + 13) & 0xff);
        }

        uint8_t tag_scalar[16], tag_avx2[16];
        poly1271(tag_scalar, msg, len, key);
        poly1271_avx2(tag_avx2, msg, len, key);

        if (memcmp(tag_scalar, tag_avx2, 16) != 0) {
            printf("FAIL len_%zu\n", len);
            print_hex("  scalar", tag_scalar, 16);
            print_hex("  avx2  ", tag_avx2, 16);
            ASSERT(0);
        }

        if (len > 0) free(msg);
    }
    printf("  all lengths match: ok\n");
}

static void test_avx2_streaming(void) {
    printf("test_avx2_streaming\n");
    uint8_t key[32];
    for (int i = 0; i < 32; i++) key[i] = (uint8_t)i;

    uint8_t msg[200];
    for (int i = 0; i < 200; i++) msg[i] = (uint8_t)i;

    uint8_t tag_oneshot[16];
    poly1271_avx2(tag_oneshot, msg, 200, key);

    /* test various chunk sizes */
    size_t chunks[] = {1, 7, 15, 16, 30, 60, 61};
    for (size_t c = 0; c < sizeof(chunks)/sizeof(chunks[0]); c++) {
        size_t chunk = chunks[c];
        poly1271_avx2_ctx_t ctx;
        poly1271_avx2_init(&ctx, key);

        size_t pos = 0;
        while (pos < 200) {
            size_t n = (pos + chunk > 200) ? (200 - pos) : chunk;
            poly1271_avx2_update(&ctx, msg + pos, n);
            pos += n;
        }

        uint8_t tag[16];
        poly1271_avx2_finish(&ctx, tag);

        ASSERT(memcmp(tag_oneshot, tag, 16) == 0);
    }
    printf("  streaming: ok\n");
}

static void test_avx2_reference_vectors(void) {
    printf("test_avx2_reference_vectors\n");
    int passed = 1;

    /* reuse same vectors as scalar */
    {
        uint8_t key[32], expected[16], tag[16];
        hex_to_bytes("b5739948a249856c49e54909ebb2f31d497377aea932d57ae80e81139e4bb6dd", key, 32);
        hex_to_bytes("497377aea932d57ae80e81139e4bb6dd", expected, 16);
        poly1271_avx2(tag, NULL, 0, key);
        if (memcmp(tag, expected, 16) != 0) {
            printf("FAIL len_0\n");
            passed = 0;
        }
    }

    {
        uint8_t key[32], expected[16], tag[16], msg[] = {0x0d};
        hex_to_bytes("f298f36369b075bf9168ceda5e9500704ac3c1475720556168946cb49dfcf6d3", key, 32);
        hex_to_bytes("9479b96ea37dff9fc873500f55ee93d4", expected, 16);
        poly1271_avx2(tag, msg, 1, key);
        if (memcmp(tag, expected, 16) != 0) {
            printf("FAIL len_1\n");
            passed = 0;
        }
    }

    {
        uint8_t key[32], expected[16], tag[16], msg[256];
        for (int i = 0; i < 256; i++) msg[i] = (uint8_t)((i * 7 + 13) & 0xff);
        hex_to_bytes("595f063a7941f6cb8b74934e31a7c0f477b7240be9f726a321ee09b53eadf174", key, 32);
        hex_to_bytes("1fff8a650754ea5eb55963a614335ab0", expected, 16);
        poly1271_avx2(tag, msg, 256, key);
        if (memcmp(tag, expected, 16) != 0) {
            printf("FAIL len_256\n");
            passed = 0;
        }
    }

    ASSERT(passed);
    printf("  reference vectors: ok\n");
}

static void test_avx2_verify(void) {
    printf("test_avx2_verify\n");
    uint8_t key[32];
    for (int i = 0; i < 32; i++) key[i] = (uint8_t)(i * 3);

    uint8_t msg[] = "Test message for AVX2 verification";
    uint8_t tag[16];

    poly1271_avx2(tag, msg, sizeof(msg) - 1, key);

    ASSERT(poly1271_avx2_verify(tag, msg, sizeof(msg) - 1, key) == 0);

    tag[5] ^= 0x01;
    ASSERT(poly1271_avx2_verify(tag, msg, sizeof(msg) - 1, key) != 0);

    printf("  verify: ok\n");
}

#endif /* __AVX2__ */

/* count set bits */
static int popcount128(const uint8_t* a) {
    int count = 0;
    for (int i = 0; i < 16; i++) {
        uint8_t x = a[i];
        while (x) { count += x & 1; x >>= 1; }
    }
    return count;
}

/* hamming distance between two 128-bit values */
static int hamming128(const uint8_t* a, const uint8_t* b) {
    uint8_t diff[16];
    for (int i = 0; i < 16; i++) diff[i] = a[i] ^ b[i];
    return popcount128(diff);
}

/* simple xorshift PRNG for reproducibility */
static uint64_t xorshift_state = 0x123456789ABCDEF0ULL;
static uint64_t xorshift64(void) {
    uint64_t x = xorshift_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    xorshift_state = x;
    return x;
}

static void rand_bytes(uint8_t* buf, size_t len) {
    for (size_t i = 0; i < len; i += 8) {
        uint64_t r = xorshift64();
        for (size_t j = 0; j < 8 && i + j < len; j++)
            buf[i + j] = (uint8_t)(r >> (j * 8));
    }
}

/* avalanche test: flip one bit, expect ~50% of output bits to flip */
#define AVALANCHE_MSG_LEN 32
#define AVALANCHE_TRIALS 1000

static void test_avalanche(void) {
    printf("test_avalanche\n");

    long total_distance = 0;
    int min_dist = 128, max_dist = 0;
    int hist[129] = {0};  /* histogram of hamming distances */

    for (int t = 0; t < AVALANCHE_TRIALS; t++) {
        uint8_t key[32], msg[AVALANCHE_MSG_LEN];
        rand_bytes(key, 32);
        rand_bytes(msg, AVALANCHE_MSG_LEN);

        uint8_t tag_orig[16];
        poly1271(tag_orig, msg, AVALANCHE_MSG_LEN, key);

        /* flip each bit of message, measure hamming distance */
        for (size_t byte = 0; byte < AVALANCHE_MSG_LEN; byte++) {
            for (int bit = 0; bit < 8; bit++) {
                msg[byte] ^= (1 << bit);

                uint8_t tag_flip[16];
                poly1271(tag_flip, msg, AVALANCHE_MSG_LEN, key);

                int dist = hamming128(tag_orig, tag_flip);
                total_distance += dist;
                if (dist < min_dist) min_dist = dist;
                if (dist > max_dist) max_dist = dist;
                hist[dist]++;

                msg[byte] ^= (1 << bit);  /* restore */
            }
        }
    }

    long total_flips = (long)AVALANCHE_TRIALS * AVALANCHE_MSG_LEN * 8;
    double mean = (double)total_distance / total_flips;

    /* compute std dev */
    double variance = 0;
    for (int i = 0; i <= 128; i++) {
        if (hist[i] > 0) {
            double diff = i - mean;
            variance += hist[i] * diff * diff;
        }
    }
    variance /= total_flips;
    double stddev = 0;
    for (int i = 1; i * i <= (int)(variance * 4); i++) stddev = i;  /* rough sqrt */
    while ((stddev + 1) * (stddev + 1) <= variance) stddev++;
    while (stddev * stddev > variance && stddev > 0) stddev--;

    printf("  %ld bit flips tested\n", total_flips);
    printf("  mean hamming distance: %.2f (ideal: 64.0)\n", mean);
    printf("  range: [%d, %d]\n", min_dist, max_dist);

    /* sanity checks */
    ASSERT(mean > 60.0 && mean < 68.0);  /* within ~6% of ideal */
    ASSERT(min_dist >= 20);  /* no catastrophically low distances */
    ASSERT(max_dist <= 110); /* no catastrophically high distances */

    printf("  avalanche: ok\n");
}

/* test acc==p wrap edge case with r=1, s=0 */
static void test_wrap_edge(void) {
    printf("test_wrap_edge\n");

    /* key with r=1 (after clamping) and s=0 */
    uint8_t key[32] = {0};
    key[0] = 0x01;  /* r = 1 in little-endian, rest zeros */
    /* s = key[16..31] = all zeros */

    /* With r=1, polynomial eval is just sum of block values mod p */
    /* Test various messages - mainly checking finalize doesn't crash */
    uint8_t msg[64];
    for (int i = 0; i < 64; i++) msg[i] = (uint8_t)i;

    uint8_t tag1[16], tag2[16];
    poly1271(tag1, msg, 64, key);

    /* streaming should match */
    poly1271_ctx_t ctx;
    poly1271_init(&ctx, key);
    poly1271_update(&ctx, msg, 64);
    poly1271_finish(&ctx, tag2);
    ASSERT(memcmp(tag1, tag2, 16) == 0);

    /* test with all-FF to maximize accumulator */
    memset(msg, 0xFF, 64);
    poly1271(tag1, msg, 64, key);
    poly1271_init(&ctx, key);
    poly1271_update(&ctx, msg, 64);
    poly1271_finish(&ctx, tag2);
    ASSERT(memcmp(tag1, tag2, 16) == 0);

    printf("  r=1 s=0 key: ok\n");
}

int main(void) {
    printf("Poly1271 Test Suite\n");
    printf("===================\n\n");

    test_empty_message();
    test_single_block();
    test_streaming();
    test_different_messages();
    test_different_keys();
    test_reference_vectors();
    test_verify();
    test_block_boundaries();
    test_streaming_equivalence();
    test_multiblock_boundaries();
    test_carry_stress();
    test_avalanche();
    test_wrap_edge();

#ifdef __AVX2__
    printf("\nAVX2 Tests\n");
    printf("----------\n");
    test_avx2_vs_scalar();
    test_avx2_streaming();
    test_avx2_reference_vectors();
    test_avx2_verify();
#else
    printf("\nAVX2 not available, skipping AVX2 tests\n");
#endif

    printf("\nAll tests passed!\n");
    return 0;
}
