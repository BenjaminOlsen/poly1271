/* SPDX-License-Identifier: MIT */
/* Poly1305 implementation for benchmarking comparison */

#ifndef POLY1305_H
#define POLY1305_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define POLY1305_CTX_MAX_SIZE 224u

typedef union {
    uint8_t opaque[POLY1305_CTX_MAX_SIZE];
    uint64_t aligner;
} poly1305_ctx_t;

void poly1305_init(poly1305_ctx_t* ctx, const uint8_t key[32]);
void poly1305_update(poly1305_ctx_t* ctx, const uint8_t* msg, size_t mlen);
void poly1305_finish(poly1305_ctx_t* ctx, uint8_t tag[16]);
void poly1305(uint8_t tag[16], const uint8_t* msg, size_t mlen, const uint8_t key[32]);

#ifdef __cplusplus
}
#endif

#endif /* POLY1305_H */
