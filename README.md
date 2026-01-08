# Poly1271

Poly1271 is a polynomial MAC using the Mersenne prime 2^127 - 1, similar to
Poly1305 in design - create an unforgeable tag of a message from the evaluation 
of a polynomial at a secret point in a prime field.

Security is in the same ballpark as Poly1305 (~100+ bits, pending
more rigorous analysis). May run faster than a scalar Poly1305
implementation on messages over 256 bytes depending on platform, thanks to
the simpler modular reduction.

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


### AVX2 (prototype)

A SIMD implementation exists using radix-2^26 representation and 4-way
parallel processing. On Alder Lake it hits > 3.75 GiB/s, but it's a prototype.

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

### Precomputation: Then and Now

Bernstein's original hash127 (1999) also used 2^127 - 1, but with 4-byte blocks.
A 1KB message meant 256 blocks, requiring precomputed powers r¹ through r²⁵⁶.
This needed ~10KB of tables per key—a "disaster for applications handling many
keys simultaneously" (from the Poly1305 paper).

Poly1305 solved this by switching to 16-byte blocks and Horner's method:

    acc = ((((acc + m1)*r + m2)*r + m3)*r + m4)*r

No precomputation needed—just multiply by r each iteration.

### Multi-Block Optimization

Modern implementations (including ours) precompute just r², r³, r⁴ (~48 bytes)
to enable 4-way parallel processing:

    acc = (acc + m1)*r^4 + m2*r^3 + m3*r^2 + m4*r

The four multiplies are independent, so the CPU can pipeline them. This is NOT
the same as hash127's precomputation problem—we store 3 extra values, not
hundreds.

This optimization benefits both Poly1305 and Poly1271 equally. Poly1271's
advantage is simpler reduction: no multiply-by-5 after each of those four
multiplies.


## Testing

A Python reference implementation verifies test vectors using arbitrary-precision
integers (no overflow bugs possible):

```
python ref/poly1271_ref.py
```

If the output matches the C implementation, both are almost certainly correct.

## Security

- Like poly1305, *each key must only be used once!* Pair with a cipher that derives fresh
  keys per message (e.g., ChaCha20-Poly1271).
- Constant-time properties tested via Welch's t-test on timing distributions. Run with
  `./build/Release/test_ct`.

## License

MIT
