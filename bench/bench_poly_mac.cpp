/* SPDX-License-Identifier: MIT */

#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include "poly1271.h"
#include "poly1305.h"

static void fill_data(std::vector<uint8_t>& data) {
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = static_cast<uint8_t>(i & 0xff);
    }
}

static void BM_Poly1305(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    std::vector<uint8_t> data(size);
    fill_data(data);

    uint8_t key[32];
    for (int i = 0; i < 32; i++) key[i] = static_cast<uint8_t>(i);

    uint8_t tag[16];

    for (auto _ : state) {
        poly1305_ctx_t ctx;
        poly1305_init(&ctx, key);
        poly1305_update(&ctx, data.data(), data.size());
        poly1305_finish(&ctx, tag);

        benchmark::DoNotOptimize(tag);
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * size);
}

static void BM_Poly1271(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    std::vector<uint8_t> data(size);
    fill_data(data);

    uint8_t key[32];
    for (int i = 0; i < 32; i++) key[i] = static_cast<uint8_t>(i);

    uint8_t tag[16];

    for (auto _ : state) {
        poly1271_ctx_t ctx;
        poly1271_init(&ctx, key);
        poly1271_update(&ctx, data.data(), data.size());
        poly1271_finish(&ctx, tag);

        benchmark::DoNotOptimize(tag);
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * size);
}

static void BM_Poly1305_OneShot(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    std::vector<uint8_t> data(size);
    fill_data(data);

    uint8_t key[32];
    for (int i = 0; i < 32; i++) key[i] = static_cast<uint8_t>(i);

    uint8_t tag[16];

    for (auto _ : state) {
        poly1305(tag, data.data(), data.size(), key);

        benchmark::DoNotOptimize(tag);
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * size);
}

static void BM_Poly1271_OneShot(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    std::vector<uint8_t> data(size);
    fill_data(data);

    uint8_t key[32];
    for (int i = 0; i < 32; i++) key[i] = static_cast<uint8_t>(i);

    uint8_t tag[16];

    for (auto _ : state) {
        poly1271(tag, data.data(), data.size(), key);

        benchmark::DoNotOptimize(tag);
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * size);
}

#ifdef __AVX2__
static void BM_Poly1271_AVX2(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    std::vector<uint8_t> data(size);
    fill_data(data);

    uint8_t key[32];
    for (int i = 0; i < 32; i++) key[i] = static_cast<uint8_t>(i);

    uint8_t tag[16];

    for (auto _ : state) {
        poly1271_avx2_ctx_t ctx;
        poly1271_avx2_init(&ctx, key);
        poly1271_avx2_update(&ctx, data.data(), data.size());
        poly1271_avx2_finish(&ctx, tag);

        benchmark::DoNotOptimize(tag);
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * size);
}

static void BM_Poly1271_AVX2_OneShot(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    std::vector<uint8_t> data(size);
    fill_data(data);

    uint8_t key[32];
    for (int i = 0; i < 32; i++) key[i] = static_cast<uint8_t>(i);

    uint8_t tag[16];

    for (auto _ : state) {
        poly1271_avx2(tag, data.data(), data.size(), key);

        benchmark::DoNotOptimize(tag);
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * size);
}
#endif

#define MAC_BENCH_SIZES \
    ->Arg(64) \
    ->Arg(256) \
    ->Arg(1 << 10) \
    ->Arg(4 << 10) \
    ->Arg(16 << 10) \
    ->Arg(64 << 10)

BENCHMARK(BM_Poly1305) MAC_BENCH_SIZES;
BENCHMARK(BM_Poly1271) MAC_BENCH_SIZES;
BENCHMARK(BM_Poly1305_OneShot) MAC_BENCH_SIZES;
BENCHMARK(BM_Poly1271_OneShot) MAC_BENCH_SIZES;
#ifdef __AVX2__
BENCHMARK(BM_Poly1271_AVX2) MAC_BENCH_SIZES;
BENCHMARK(BM_Poly1271_AVX2_OneShot) MAC_BENCH_SIZES;
#endif

BENCHMARK_MAIN();
