cmake_minimum_required(VERSION 3.14)
project(poly1271 VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(POLY1271_BUILD_TESTS "Build test suite" ON)
option(POLY1271_BUILD_BENCH "Build benchmarks" OFF)

add_library(poly1271
    src/poly1271.c
    src/poly1271_avx2.c
)

target_include_directories(poly1271 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(poly1271 PRIVATE /W4 /O2 /arch:AVX2)
else()
    target_compile_options(poly1271 PRIVATE -Wall -Wextra -Wpedantic -O3 -mavx2)
endif()

if(POLY1271_BUILD_TESTS)
    enable_testing()

    add_executable(test_poly1271 test/test_poly1271.c)
    target_link_libraries(test_poly1271 PRIVATE poly1271)

    add_executable(test_ct test/test_ct.c)
    target_link_libraries(test_ct PRIVATE poly1271)

    add_executable(test_poly1305
        test/test_poly1305.c
        ref/poly1305.c
    )
    target_include_directories(test_poly1305 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ref
    )

    add_executable(test_crossval test/test_crossval.c)
    target_link_libraries(test_crossval PRIVATE poly1271)

    if(MSVC)
        target_compile_options(test_poly1271 PRIVATE /arch:AVX2)
        target_compile_options(test_ct PRIVATE /arch:AVX2)
        target_compile_options(test_poly1305 PRIVATE /arch:AVX2)
        target_compile_options(test_crossval PRIVATE /arch:AVX2)
    else()
        target_compile_options(test_poly1271 PRIVATE -mavx2)
        target_compile_options(test_ct PRIVATE -mavx2)
        target_compile_options(test_poly1305 PRIVATE -mavx2)
        target_compile_options(test_crossval PRIVATE -mavx2)
    endif()

    add_test(NAME poly1271_tests COMMAND test_poly1271)
    add_test(NAME poly1271_ct COMMAND test_ct)
    add_test(NAME poly1305_kat COMMAND test_poly1305)
    add_test(NAME poly1271_crossval COMMAND test_crossval 10000)
endif()

if(POLY1271_BUILD_BENCH)
    add_executable(bench_rdtsc
        bench/bench_rdtsc.c
        ref/poly1305.c
    )
    target_include_directories(bench_rdtsc PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/ref
    )
    target_link_libraries(bench_rdtsc PRIVATE poly1271)
    if(MSVC)
        target_compile_options(bench_rdtsc PRIVATE /O2 /arch:AVX2)
    else()
        target_compile_options(bench_rdtsc PRIVATE -O3 -mavx2)
    endif()

    # Google Benchmark
    find_package(benchmark QUIET)
    if(benchmark_FOUND)
        add_executable(bench_poly_mac
            bench/bench_poly_mac.cpp
            ref/poly1305.c
        )
        target_include_directories(bench_poly_mac PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/ref
        )
        target_link_libraries(bench_poly_mac PRIVATE poly1271 benchmark::benchmark)
        if(MSVC)
            target_compile_options(bench_poly_mac PRIVATE /O2 /arch:AVX2)
        else()
            target_compile_options(bench_poly_mac PRIVATE -O3 -mavx2)
        endif()
    else()
        message(STATUS "Google Benchmark not found, skipping bench_poly_mac")
    endif()
endif()
