/* Direct RDTSC cycle counting benchmark for Poly1271 vs Poly1305 */

#include "poly1271.h"
#include "../ref/poly1305.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

/*
 * RDTSC = Read Time Stamp Counter
 *
 * The CPU maintains a 64-bit counter that increments at a fixed frequency.
 * We use it to count cycles with near-zero overhead (~20-40 cycles).
 *
 * Serialization is critical:
 * - Modern CPUs execute instructions out-of-order
 * - Without barriers, RDTSC can execute before/after we intend
 * - lfence prevents reordering; rdtscp is the serializing variant
 */

static inline uint64_t rdtsc_start(void) {
    _mm_lfence();  /* Serialize: complete all prior instructions */
    return __rdtsc();
}

static inline uint64_t rdtsc_end(void) {
    unsigned int aux;
    uint64_t tsc = __rdtscp(&aux);  /* Serializing read + processor ID */
    _mm_lfence();
    return tsc;
}

/* comparator for qsort */
static int cmp_u64(const void* a, const void* b) {
    uint64_t va = *(const uint64_t*)a;
    uint64_t vb = *(const uint64_t*)b;
    return (va > vb) - (va < vb);
}

/* measure RDTSC overhead (empty measurement) */
static uint64_t measure_overhead(void) {
    uint64_t samples[1000];

    for (int i = 0; i < 1000; i++) {
        uint64_t start = rdtsc_start();
        /* nothing */
        uint64_t end = rdtsc_end();
        samples[i] = end - start;
    }

    qsort(samples, 1000, sizeof(uint64_t), cmp_u64);
    return samples[500];  /* Median */
}

static const int WARMUP_ITERS = 200;

/* benchmark a function, return median cycles */
static uint64_t benchmark_func(
    void (*func)(uint8_t*, const uint8_t*, size_t, const uint8_t*),
    const uint8_t* msg, size_t len, const uint8_t* key,
    uint64_t overhead, int iterations)
{
    uint8_t tag[16];
    uint64_t* samples = (uint64_t*)malloc(iterations * sizeof(uint64_t));

    /* warmup is critical for cache/branch predictor */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        func(tag, msg, len, key);
    }

    /* measure */
    for (int i = 0; i < iterations; i++) {
        uint64_t start = rdtsc_start();
        func(tag, msg, len, key);
        uint64_t end = rdtsc_end();
        samples[i] = (end - start > overhead) ? (end - start - overhead) : 0;
    }

    qsort(samples, iterations, sizeof(uint64_t), cmp_u64);
    uint64_t median = samples[iterations / 2];
    free(samples);
    return median;
}

static void poly1271_wrapper(uint8_t* tag, const uint8_t* msg, size_t len, const uint8_t* key) {
    poly1271(tag, msg, len, key);
}

static void poly1305_wrapper(uint8_t* tag, const uint8_t* msg, size_t len, const uint8_t* key) {
    poly1305(tag, msg, len, key);
}

#ifdef __AVX2__
static void poly1271_avx2_wrapper(uint8_t* tag, const uint8_t* msg, size_t len, const uint8_t* key) {
    poly1271_avx2(tag, msg, len, key);
}
#endif

int main(void) {
    printf("RDTSC Cycle Counting Benchmark\n");
    printf("==============================\n\n");

    uint64_t overhead = measure_overhead();
    printf("RDTSC overhead: %llu cycles\n\n", (unsigned long long)overhead);

    uint8_t key[32] = {
        0x85, 0xd6, 0xbe, 0x78, 0x57, 0x55, 0x6d, 0x33,
        0x7f, 0x44, 0x52, 0xfe, 0x42, 0xd5, 0x06, 0xa8,
        0x01, 0x03, 0x80, 0x8a, 0xfb, 0x0d, 0xb2, 0xfd,
        0x4a, 0xbf, 0xf6, 0xaf, 0x41, 0x49, 0xf5, 0x1b
    };

    size_t sizes[] = { 64, 256, 1024, 4096, 16384, 65536 };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    /* allocate largest message */
    uint8_t* msg = (uint8_t*)malloc(65536);
    for (size_t i = 0; i < 65536; i++) {
        msg[i] = (uint8_t)(i & 0xFF);
    }

    printf("%-10s %12s %8s %12s %8s",
           "Size", "Poly1305", "cpb", "Poly1271", "cpb");
#ifdef __AVX2__
    printf(" %12s %8s", "Poly1271-AVX2", "cpb");
#endif
    printf("\n");

    printf("%-10s %12s %8s %12s %8s",
           "----", "--------", "---", "--------", "---");
#ifdef __AVX2__
    printf(" %12s %8s", "-------------", "---");
#endif
    printf("\n");

    for (int s = 0; s < num_sizes; s++) {
        size_t len = sizes[s];
        int iterations = (len < 1024) ? 10000 : (len < 16384) ? 5000 : 1000;

        uint64_t cycles_1305 = benchmark_func(poly1305_wrapper, msg, len, key, overhead, iterations);
        uint64_t cycles_1271 = benchmark_func(poly1271_wrapper, msg, len, key, overhead, iterations);

        double cpb_1305 = (double)cycles_1305 / len;
        double cpb_1271 = (double)cycles_1271 / len;

        printf("%-10zu %12llu %8.2f %12llu %8.2f",
               len,
               (unsigned long long)cycles_1305, cpb_1305,
               (unsigned long long)cycles_1271, cpb_1271);

#ifdef __AVX2__
        uint64_t cycles_avx2 = benchmark_func(poly1271_avx2_wrapper, msg, len, key, overhead, iterations);
        double cpb_avx2 = (double)cycles_avx2 / len;
        printf(" %12llu %8.2f", (unsigned long long)cycles_avx2, cpb_avx2);
#endif

        printf("\n");
    }

    printf("\n");
    printf("Measurement: median of N iterations, %d warmup rounds\n", WARMUP_ITERS);
    printf("Overhead subtracted from each measurement\n");

    free(msg);
    return 0;
}
