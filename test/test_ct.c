/* SPDX-License-Identifier: MIT */
/* Welch's t-test on timing distributions */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "poly1271.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#define ENOUGH_MEASUREMENTS 10000
#define NUMBER_MEASUREMENTS 100000
#define MSG_LEN 256
#define THRESHOLD 4.5

static uint64_t get_time(void) {
#ifdef _WIN32
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (uint64_t)t.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
#endif
}

/* xorshift64 */
static uint64_t rng_state = 0xdeadbeefcafebabeULL;

static uint64_t rand64(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

static void rand_bytes(uint8_t *buf, size_t len) {
    for (size_t i = 0; i < len; i += 8) {
        uint64_t r = rand64();
        size_t chunk = (len - i < 8) ? (len - i) : 8;
        memcpy(buf + i, &r, chunk);
    }
}

/* Welford's online algorithm for mean/variance */
typedef struct {
    double n, mean, m2;
} ttest_ctx;

static void ttest_init(ttest_ctx *ctx) {
    ctx->n = ctx->mean = ctx->m2 = 0;
}

static void ttest_push(ttest_ctx *ctx, double x) {
    ctx->n++;
    double delta = x - ctx->mean;
    ctx->mean += delta / ctx->n;
    ctx->m2 += delta * (x - ctx->mean);
}

static double ttest_compute(ttest_ctx *a, ttest_ctx *b) {
    if (a->n < 2 || b->n < 2) return 0;
    double var_a = a->m2 / (a->n - 1);
    double var_b = b->m2 / (b->n - 1);
    double se = sqrt(var_a / a->n + var_b / b->n);
    if (se < 1e-10) return 0;
    return (a->mean - b->mean) / se;
}

static void test_oneshot(void) {
    printf("Testing poly1271()...\n");

    ttest_ctx class0, class1;
    ttest_init(&class0);
    ttest_init(&class1);

    uint8_t key_fixed[32], msg_fixed[MSG_LEN];
    memset(key_fixed, 0x42, 32);
    memset(msg_fixed, 0x00, MSG_LEN);

    uint8_t key[32], msg[MSG_LEN], tag[16];

    for (int i = 0; i < NUMBER_MEASUREMENTS; i++) {
        int class_id = rand64() & 1;

        if (class_id) {
            rand_bytes(key, 32);
            rand_bytes(msg, MSG_LEN);
        } else {
            memcpy(key, key_fixed, 32);
            memcpy(msg, msg_fixed, MSG_LEN);
        }

        uint64_t t0 = get_time();
        poly1271(tag, msg, MSG_LEN, key);
        uint64_t t1 = get_time();

        volatile uint8_t sink = tag[0]; (void)sink;

        ttest_push(class_id ? &class1 : &class0, (double)(t1 - t0));

        if (i > ENOUGH_MEASUREMENTS && (i % 10000) == 0) {
            double t = ttest_compute(&class0, &class1);
            printf("  %d: t = %.3f\n", i, t);
            if (fabs(t) > THRESHOLD) { printf("  FAIL\n"); return; }
        }
    }

    double t = ttest_compute(&class0, &class1);
    printf("  final: t = %.3f %s\n", t, fabs(t) > THRESHOLD ? "FAIL" : "PASS");
}

static void test_streaming(void) {
    printf("Testing poly1271 streaming...\n");

    ttest_ctx class0, class1;
    ttest_init(&class0);
    ttest_init(&class1);

    uint8_t key_fixed[32], msg_fixed[MSG_LEN];
    memset(key_fixed, 0x42, 32);
    memset(msg_fixed, 0x00, MSG_LEN);

    uint8_t key[32], msg[MSG_LEN], tag[16];
    poly1271_ctx_t ctx;

    for (int i = 0; i < NUMBER_MEASUREMENTS; i++) {
        int class_id = rand64() & 1;

        if (class_id) {
            rand_bytes(key, 32);
            rand_bytes(msg, MSG_LEN);
        } else {
            memcpy(key, key_fixed, 32);
            memcpy(msg, msg_fixed, MSG_LEN);
        }

        uint64_t t0 = get_time();
        poly1271_init(&ctx, key);
        poly1271_update(&ctx, msg, MSG_LEN);
        poly1271_finish(&ctx, tag);
        uint64_t t1 = get_time();

        volatile uint8_t sink = tag[0]; (void)sink;

        ttest_push(class_id ? &class1 : &class0, (double)(t1 - t0));

        if (i > ENOUGH_MEASUREMENTS && (i % 10000) == 0) {
            double t = ttest_compute(&class0, &class1);
            printf("  %d: t = %.3f\n", i, t);
            if (fabs(t) > THRESHOLD) { printf("  FAIL\n"); return; }
        }
    }

    double t = ttest_compute(&class0, &class1);
    printf("  final: t = %.3f %s\n", t, fabs(t) > THRESHOLD ? "FAIL" : "PASS");
}

/* verify: class0 = correct tag, class1 = wrong tag */
static void test_verify(void) {
    printf("Testing poly1271_verify()...\n");

    ttest_ctx class0, class1;
    ttest_init(&class0);
    ttest_init(&class1);

    uint8_t key[32], msg[MSG_LEN], tag[16], bad_tag[16];
    rand_bytes(key, 32);
    rand_bytes(msg, MSG_LEN);
    poly1271(tag, msg, MSG_LEN, key);
    memcpy(bad_tag, tag, 16);
    bad_tag[0] ^= 0x01;

    uint8_t test_tag[16];

    for (int i = 0; i < NUMBER_MEASUREMENTS; i++) {
        int class_id = rand64() & 1;
        memcpy(test_tag, class_id ? bad_tag : tag, 16);

        uint64_t t0 = get_time();
        volatile int r = poly1271_verify(test_tag, msg, MSG_LEN, key);
        uint64_t t1 = get_time();
        (void)r;

        ttest_push(class_id ? &class1 : &class0, (double)(t1 - t0));

        if (i > ENOUGH_MEASUREMENTS && (i % 10000) == 0) {
            double t = ttest_compute(&class0, &class1);
            printf("  %d: t = %.3f\n", i, t);
            if (fabs(t) > THRESHOLD) { printf("  FAIL\n"); return; }
        }
    }

    double t = ttest_compute(&class0, &class1);
    printf("  final: t = %.3f %s\n", t, fabs(t) > THRESHOLD ? "FAIL" : "PASS");
}

#if defined(__AVX2__) || defined(__AVX2)
static void test_avx2(void) {
    printf("Testing poly1271_avx2()...\n");

    ttest_ctx class0, class1;
    ttest_init(&class0);
    ttest_init(&class1);

    uint8_t key_fixed[32], msg_fixed[MSG_LEN];
    memset(key_fixed, 0x42, 32);
    memset(msg_fixed, 0x00, MSG_LEN);

    uint8_t key[32], msg[MSG_LEN], tag[16];

    for (int i = 0; i < NUMBER_MEASUREMENTS; i++) {
        int class_id = rand64() & 1;

        if (class_id) {
            rand_bytes(key, 32);
            rand_bytes(msg, MSG_LEN);
        } else {
            memcpy(key, key_fixed, 32);
            memcpy(msg, msg_fixed, MSG_LEN);
        }

        uint64_t t0 = get_time();
        poly1271_avx2(tag, msg, MSG_LEN, key);
        uint64_t t1 = get_time();

        volatile uint8_t sink = tag[0]; (void)sink;

        ttest_push(class_id ? &class1 : &class0, (double)(t1 - t0));

        if (i > ENOUGH_MEASUREMENTS && (i % 10000) == 0) {
            double t = ttest_compute(&class0, &class1);
            printf("  %d: t = %.3f\n", i, t);
            if (fabs(t) > THRESHOLD) { printf("  FAIL\n"); return; }
        }
    }

    double t = ttest_compute(&class0, &class1);
    printf("  final: t = %.3f %s\n", t, fabs(t) > THRESHOLD ? "FAIL" : "PASS");
}
#endif

int main(void) {
    printf("ct test\n");
    printf("measurements: %d, threshold: |t| > %.1f\n\n", NUMBER_MEASUREMENTS, THRESHOLD);

    rng_state = get_time();

    test_oneshot();
    printf("\n");

    test_streaming();
    printf("\n");

    test_verify();
    printf("\n");

#if defined(__AVX2__) || defined(__AVX2)
    test_avx2();
    printf("\n");
#endif

    return 0;
}
