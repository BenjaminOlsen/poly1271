#!/usr/bin/env python3
"""
Poly1271 reference implementation for test vector verification.
Uses arbitrary-precision integers - if this matches the C output,
the C is almost certainly correct.
"""

P = (1 << 127) - 1
BLOCK_SIZE = 15


def clamp_r(r_bytes: bytes) -> int:
    r = list(r_bytes)
    r[3] &= 0x0F
    r[7] &= 0x0F
    r[11] &= 0x0F
    r[15] &= 0x07
    r[4] &= 0xFC
    r[8] &= 0xFC
    r[12] &= 0xFC
    return int.from_bytes(bytes(r), 'little')


def load_block(data: bytes) -> int:
    assert len(data) == BLOCK_SIZE
    return int.from_bytes(data, 'little') | (1 << 120)


def load_partial(data: bytes) -> int:
    assert len(data) < BLOCK_SIZE
    padded = data + b'\x01' + b'\x00' * (15 - len(data))
    return int.from_bytes(padded, 'little')


def poly1271_ref(msg: bytes, key: bytes) -> bytes:
    assert len(key) == 32

    r = clamp_r(key[:16])
    s = int.from_bytes(key[16:32], 'little')

    acc = 0
    pos = 0

    while pos + BLOCK_SIZE <= len(msg):
        block = load_block(msg[pos:pos + BLOCK_SIZE])
        acc = (acc + block) * r % P
        pos += BLOCK_SIZE

    if pos < len(msg):
        block = load_partial(msg[pos:])
        acc = (acc + block) * r % P

    if acc == P:
        acc = 0

    result = (acc + s) & ((1 << 128) - 1)
    return result.to_bytes(16, 'little')


def test_vectors():
    print("Poly1271 Reference Implementation")
    print("==================================\n")

    vectors = [
        ("len_0",
         "b5739948a249856c49e54909ebb2f31d497377aea932d57ae80e81139e4bb6dd",
         b"",
         "497377aea932d57ae80e81139e4bb6dd"),

        ("len_1",
         "f298f36369b075bf9168ceda5e9500704ac3c1475720556168946cb49dfcf6d3",
         bytes([0x0d]),
         "9479b96ea37dff9fc873500f55ee93d4"),

        ("len_15",
         "f6118ce0bcdae1277b9b82025e5aeb46c9c4015d0f8b0ea4cdb292950dbca174",
         bytes([(i * 7 + 13) & 0xff for i in range(15)]),
         "00bd5684bbb9ec0fdb6812fec2b683da"),

        ("len_16",
         "bdce2ecf9a9f4388b2e3c055eb3940f7f6ecfcd6405806aa01b11984937b4a5f",
         bytes([(i * 7 + 13) & 0xff for i in range(16)]),
         "1a7e08badf6a311a1326e180535fe5b0"),

        ("len_30",
         "cef0ca2975a63992cd201d7a474f4f37d284bc4c4c402ccecafe52f8eac671d3",
         bytes([(i * 7 + 13) & 0xff for i in range(30)]),
         "d64d14c694fc3ade68fa7e6ee9862108"),

        ("len_100",
         "b0fdf754bb59965e413060dedf8ed9b0d96e8c23544792dca17e89998da75a3a",
         bytes([(i * 7 + 13) & 0xff for i in range(100)]),
         "68977a391e9bdf4d904fdb851cde1599"),

        ("len_256",
         "595f063a7941f6cb8b74934e31a7c0f477b7240be9f726a321ee09b53eadf174",
         bytes([(i * 7 + 13) & 0xff for i in range(256)]),
         "1fff8a650754ea5eb55963a614335ab0"),

        ("len_1000",
         "5476a825f45840087522474e83e1ec8d70a34954735d043659b18aa9244668cf",
         bytes([(i * 7 + 13) & 0xff for i in range(1000)]),
         "53d0bba62df4206e277f411f099fa60c"),

        ("uncle_walt",
         "c20ae296d65cb1ef04ef2cd32acdeb835f9bfacd77a2ff311b1e573f9f949677",
         b"Answer.\n"
         b"That you are here-that life exists and identity,\n"
         b"That the powerful play goes on, and you may contribute a verse.",
         "e0840ca53a6744daa78e6d9c65c3bbe9"),
    ]

    all_passed = True
    for name, key_hex, msg, expected_hex in vectors:
        key = bytes.fromhex(key_hex)
        expected = bytes.fromhex(expected_hex)
        computed = poly1271_ref(msg, key)

        if computed == expected:
            print(f"  {name}: ok")
        else:
            print(f"  {name}: FAIL")
            print(f"    expected: {expected.hex()}")
            print(f"    got:      {computed.hex()}")
            all_passed = False

    print()
    if all_passed:
        print("All test vectors verified!")
    else:
        print("SOME TESTS FAILED")
        return 1

    return 0


def test_properties():
    print("\nProperty Tests")
    print("--------------")

    key = bytes([0] * 16) + bytes([0x42] * 16)
    tag = poly1271_ref(b"", key)
    s = int.from_bytes(key[16:], 'little')
    expected = s.to_bytes(16, 'little')
    if tag == expected:
        print("  empty message returns s: ok")
    else:
        print(f"  empty message: FAIL (got {tag.hex()}, expected {expected.hex()})")

    import random
    random.seed(42)
    key = bytes(random.getrandbits(8) for _ in range(32))
    msg = bytes(random.getrandbits(8) for _ in range(100))
    t1 = poly1271_ref(msg, key)
    t2 = poly1271_ref(msg, key)
    if t1 == t2:
        print("  deterministic output: ok")
    else:
        print("  deterministic output: FAIL")


if __name__ == "__main__":
    import sys
    rc = test_vectors()
    test_properties()
    sys.exit(rc)
