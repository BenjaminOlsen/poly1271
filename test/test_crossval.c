/* SPDX-License-Identifier: MIT */

/*
 * cross-validation test: scalar vs AVX2
 *
 * runs millions of random inputs through both implementations
 * and verifies they produce identical results
 */

#include "poly1271.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* xorshift64 prng - fast and good enough for testing */
static uint64_t rng_state;

static uint64_t xorshift64(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

static void rand_bytes(uint8_t* buf, size_t len) {
    size_t i = 0;
    while (i + 8 <= len) {
        uint64_t r = xorshift64();
        memcpy(buf + i, &r, 8);
        i += 8;
    }
    if (i < len) {
        uint64_t r = xorshift64();
        memcpy(buf + i, &r, len - i);
    }
}

static void print_hex(const char* label, const uint8_t* buf, size_t len) {
    printf("%s: ", label);
    for (size_t i = 0; i < len; i++) printf("%02x", buf[i]);
    printf("\n");
}

#ifdef __AVX2__

/* test one-shot api: poly1271() vs poly1271_avx2() */
static int test_oneshot(const uint8_t* key, const uint8_t* msg, size_t len) {
    uint8_t tag_scalar[16], tag_avx2[16];

    poly1271(tag_scalar, msg, len, key);
    poly1271_avx2(tag_avx2, msg, len, key);

    if (memcmp(tag_scalar, tag_avx2, 16) != 0) {
        printf("MISMATCH oneshot len=%zu\n", len);
        print_hex("  key   ", key, 32);
        print_hex("  msg   ", msg, len > 64 ? 64 : len);
        if (len > 64) printf("  ... (%zu more bytes)\n", len - 64);
        print_hex("  scalar", tag_scalar, 16);
        print_hex("  avx2  ", tag_avx2, 16);
        return 0;
    }
    return 1;
}

/* test streaming api with random chunk sizes */
static int test_streaming(const uint8_t* key, const uint8_t* msg, size_t len) {
    uint8_t tag_scalar[16], tag_avx2[16];

    /* scalar streaming */
    poly1271_ctx_t ctx_s;
    poly1271_init(&ctx_s, key);

    size_t pos = 0;
    while (pos < len) {
        size_t chunk = (xorshift64() % 64) + 1;  /* 1-64 bytes */
        if (pos + chunk > len) chunk = len - pos;
        poly1271_update(&ctx_s, msg + pos, chunk);
        pos += chunk;
    }
    poly1271_finish(&ctx_s, tag_scalar);

    /* avx2 streaming */
    poly1271_avx2_ctx_t ctx_a;
    poly1271_avx2_init(&ctx_a, key);

    pos = 0;
    while (pos < len) {
        size_t chunk = (xorshift64() % 64) + 1;
        if (pos + chunk > len) chunk = len - pos;
        poly1271_avx2_update(&ctx_a, msg + pos, chunk);
        pos += chunk;
    }
    poly1271_avx2_finish(&ctx_a, tag_avx2);

    if (memcmp(tag_scalar, tag_avx2, 16) != 0) {
        printf("MISMATCH streaming len=%zu\n", len);
        print_hex("  scalar", tag_scalar, 16);
        print_hex("  avx2  ", tag_avx2, 16);
        return 0;
    }

    /* also verify streaming matches oneshot */
    uint8_t tag_oneshot[16];
    poly1271(tag_oneshot, msg, len, key);
    if (memcmp(tag_scalar, tag_oneshot, 16) != 0) {
        printf("MISMATCH streaming vs oneshot len=%zu\n", len);
        return 0;
    }

    return 1;
}

/* test verify api */
static int test_verify(const uint8_t* key, const uint8_t* msg, size_t len) {
    uint8_t tag[16];
    poly1271(tag, msg, len, key);

    /* scalar verify should pass */
    if (poly1271_verify(tag, msg, len, key) != 0) {
        printf("FAIL scalar verify len=%zu\n", len);
        return 0;
    }

    /* avx2 verify should pass */
    if (poly1271_avx2_verify(tag, msg, len, key) != 0) {
        printf("FAIL avx2 verify len=%zu\n", len);
        return 0;
    }

    /* corrupted tag should fail both */
    tag[0] ^= 1;
    if (poly1271_verify(tag, msg, len, key) == 0) {
        printf("FAIL scalar verify accepted bad tag len=%zu\n", len);
        return 0;
    }
    if (poly1271_avx2_verify(tag, msg, len, key) == 0) {
        printf("FAIL avx2 verify accepted bad tag len=%zu\n", len);
        return 0;
    }

    return 1;
}

int main(int argc, char** argv) {
    int num_tests = 1000000;  /* default: 1M tests */
    size_t max_len = 16384;   /* default: up to 16KB messages */

    if (argc > 1) num_tests = atoi(argv[1]);
    if (argc > 2) max_len = (size_t)atoi(argv[2]);

    printf("poly1271 cross-validation: scalar vs avx2\n");
    printf("==========================================\n");
    printf("tests: %d, max message size: %zu bytes\n\n", num_tests, max_len);

    /* seed rng */
    rng_state = (uint64_t)time(NULL) ^ 0xDEADBEEFCAFEBABEULL;
    printf("rng seed: 0x%016llx\n", (unsigned long long)rng_state);

    uint8_t* msg = (uint8_t*)malloc(max_len);
    if (!msg) {
        printf("failed to allocate %zu bytes\n", max_len);
        return 1;
    }

    uint8_t key[32];
    int passed = 0, failed = 0;

    /* progress tracking */
    int milestone = num_tests / 10;
    clock_t start = clock();

    for (int i = 0; i < num_tests; i++) {
        /* random key */
        rand_bytes(key, 32);

        /* random length with bias toward interesting sizes */
        size_t len;
        int r = xorshift64() % 100;
        if (r < 10) {
            /* 10%: tiny (0-16 bytes) */
            len = xorshift64() % 17;
        } else if (r < 30) {
            /* 20%: around block boundaries */
            len = 13 + (xorshift64() % 5);  /* 13-17 */
        } else if (r < 50) {
            /* 20%: around 4-block boundaries */
            len = 58 + (xorshift64() % 5);  /* 58-62 */
        } else if (r < 70) {
            /* 20%: medium (64-1024) */
            len = 64 + (xorshift64() % 960);
        } else {
            /* 30%: large (up to max_len) */
            len = xorshift64() % (max_len + 1);
        }

        /* random message */
        rand_bytes(msg, len);

        /* run tests */
        int ok = 1;
        ok &= test_oneshot(key, msg, len);
        ok &= test_streaming(key, msg, len);
        ok &= test_verify(key, msg, len);

        if (ok) {
            passed++;
        } else {
            failed++;
            if (failed >= 10) {
                printf("\ntoo many failures, stopping\n");
                break;
            }
        }

        /* progress */
        if (milestone > 0 && (i + 1) % milestone == 0) {
            int pct = ((i + 1) * 100) / num_tests;
            printf("  %d%% (%d tests)...\n", pct, i + 1);
        }
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    free(msg);

    printf("\n");
    printf("results: %d passed, %d failed\n", passed, failed);
    printf("time: %.2f seconds (%.0f tests/sec)\n", elapsed, passed / elapsed);

    if (failed == 0) {
        printf("\nall tests passed!\n");
        return 0;
    } else {
        printf("\nFAILED\n");
        return 1;
    }
}

#else /* no AVX2 */

int main(void) {
    printf("AVX2 not available, skipping cross-validation\n");
    return 0;
}

#endif
