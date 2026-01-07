# Poly1271

Poly1271 is a polynomial MAC using the Mersenne prime 2^127 - 1, similar to
Poly1305. Security is in the same ballpark as Poly1305 (~100+ bits, pending
more rigorous analysis). May run faster than a scalar Poly1305
implementation on messages over 256 bytes depending on platform, thanks to
the simpler modular reduction and instruction level parallelism.

## Usage

```c
#include "poly1271.h"

uint8_t key[32], tag[16];
// ... fill key ...

// compute tag
poly1271(tag, message, len, key);

// verify (constant-time, returns 0 on success)
if (poly1271_verify(tag, message, len, key) != 0) {
    // authentication failed
}
```

For large messages processed in chunks, use the streaming version:

```c
poly1271_ctx_t ctx;
poly1271_init(&ctx, key);
poly1271_update(&ctx, chunk1, len1);
poly1271_update(&ctx, chunk2, len2);
poly1271_finish(&ctx, tag);
```

## Building

```
cmake -B build
cmake --build build
ctest --test-dir build
```

### Benchmarks

Requires [Google Benchmark](https://github.com/google/benchmark):

```
cmake -B build -DPOLY1271_BUILD_BENCH=ON
cmake --build build --config Release
./build/Release/bench_poly_mac.exe   # Windows
./build/bench_poly_mac               # Unix
```

can do  `--benchmark_filter="65536"` to run only that benchmark size, also.


## Performance

i7-1265U, MSVC, 64KB messages:

| Implementation | Throughput | vs Poly1305 |
|----------------|------------|-------------|
| Poly1305       | 2.17 GiB/s | -           |
| Poly1271       | 2.90 GiB/s | +34%        |

Slower for <256 byte messages due to key setup (precomputing r^2, r^3, r^4).

### AVX2 (prototype)

A SIMD implementation exists using radix-2^26 representation and 4-way
parallel processing. On Alder Lake it hits 3.75 GiB/s (+73% vs Poly1305).
Performance varies by platform - some older chips throttle AVX2 frequency
enough that scalar wins. Not part of stable API.

```c
#ifdef __AVX2__
poly1271_avx2(tag, msg, len, key);
#endif
```

## How it works

Like Poly1305, we evaluate a polynomial over message blocks:

    tag = (m1*r^n + m2*r^(n-1) + ... + mn*r + s) mod 2^128

The difference is the prime: Poly1305 uses 2^130 - 5, we use 2^127 - 1.
This Mersenne prime has the nice property that 2^127 ≡ 1, so reduction
is just addition (no multiply by 5).

The main optimization is 4 block parallel processing. Instead of:

    acc = ((((acc + m1)*r + m2)*r + m3)*r + m4)*r

we compute:

    acc = (acc + m1)*r^4 + m2*r^3 + m3*r^2 + m4*r

The four multiplies are independent, so the CPU can overlap them. We
precompute r^2, r^3, r^4 during key setup.

## Testing

A Python reference implementation verifies test vectors using arbitrary-precision
integers (no overflow bugs possible):

```
python test/poly1271_ref.py
```

If the output matches the C implementation, both are almost certainly correct.

## Security

- Like poly1305, *each key must only be used once!* Pair with a cipher that derives fresh
  keys per message (e.g., ChaCha20-Poly1271).
- Constant-time properties tested via Welch's t-test on timing distributions. Run with
  `./build/Release/test_ct`.

## License

MIT
