"""MoonEP-specific PPU bulk-memory primitives.

These are warp-cooperative register loads/stores, not single-lane AIU copies.
All 32 lanes pass the same word-aligned base and offset; each lane supplies
or receives one word per 128-byte stripe. The caller must provide the entire
128/256/512-byte region. Partial stripes require bounds-checked scalar access.
"""

import tilelang.ppu.language as T


@T.macro
def fast_bulk_load_128_ref(src: T.Ref, remote):
    """Load one 128-byte stripe from an indexed reference."""
    value = T.alloc_var(T.uint32)
    with T.sblock("moonep_fast_bulk_load_128"):
        T.import_source(
            r"""
#ifndef MOONEP_FAST_BULK_LOAD_128_
#define MOONEP_FAST_BULK_LOAD_128_
__attribute__((device)) __attribute__((always_inline)) inline void
moonep_fast_bulk_load_128(
    uint* output, const void* source, bool remote) {
  *output = remote
      ? __ppu_remote_load_bulk_ncp_b32(source, 0)
      : __ppu_global_ldcv_bulk_b32(source);
}
#endif
"""
        )
        T.call_extern(
            "void",
            "moonep_fast_bulk_load_128",
            T.address_of(value),
            T.address_of(src),
            remote,
        )
    return value


@T.macro
def fast_bulk_store_128_ref(dst: T.Ref, value, remote):
    """Store one 128-byte stripe to an indexed reference."""
    with T.sblock("moonep_fast_bulk_store_128"):
        T.import_source(
            r"""
#ifndef MOONEP_FAST_BULK_STORE_128_
#define MOONEP_FAST_BULK_STORE_128_
__attribute__((device)) __attribute__((always_inline)) inline void
moonep_fast_bulk_store_128(
    void* destination, uint value, bool remote) {
  if (remote) {
    __ppu_remote_store_bulk_volatile_b32(destination, value, 0);
  } else {
    __ppu_global_stwt_bulk_b32(destination, value);
  }
}
#endif
"""
        )
        T.call_extern(
            "void",
            "moonep_fast_bulk_store_128",
            T.address_of(dst),
            value,
            remote,
        )


@T.macro
def fast_bulk_load_128(base: T.Ref, byte_offset, remote):
    """Load one 128-byte stripe from ``base + byte_offset``."""
    value = T.alloc_var(T.uint32)
    with T.sblock("moonep_fast_bulk_load_128"):
        T.import_source(
            r"""
#ifndef MOONEP_FAST_BULK_LOAD_128_
#define MOONEP_FAST_BULK_LOAD_128_
__attribute__((device)) __attribute__((always_inline)) inline void
moonep_fast_bulk_load_128(
    uint* output, const void* base, long long byte_offset, bool remote) {
  const void* source = static_cast<const char*>(base) + byte_offset;
  *output = remote
      ? __ppu_remote_load_bulk_ncp_b32(source, 0)
      : __ppu_global_ldcv_bulk_b32(source);
}
#endif
"""
        )
        T.call_extern(
            "void",
            "moonep_fast_bulk_load_128",
            T.address_of(value),
            T.address_of(base),
            byte_offset,
            remote,
        )
    return value


@T.macro
def fast_bulk_load_256(base: T.Ref, byte_offset, remote):
    """Load two 128-byte stripes from ``base + byte_offset``."""
    value = T.alloc_var(T.uint32x2)
    with T.sblock("moonep_fast_bulk_load_256"):
        T.import_source(
            r"""
#ifndef MOONEP_FAST_BULK_LOAD_256_
#define MOONEP_FAST_BULK_LOAD_256_
__attribute__((device)) __attribute__((always_inline)) inline void
moonep_fast_bulk_load_256(
    uint2* output, const void* base, long long byte_offset, bool remote) {
  const void* source = static_cast<const char*>(base) + byte_offset;
  *output = remote
      ? __ppu_remote_load_bulk_ncp_b32x2(source, 0)
      : __ppu_global_ldcv_bulk_b32x2(source);
}
#endif
"""
        )
        T.call_extern(
            "void",
            "moonep_fast_bulk_load_256",
            T.address_of(value),
            T.address_of(base),
            byte_offset,
            remote,
        )
    return value


@T.macro
def fast_bulk_load_512(base: T.Ref, byte_offset, remote):
    """Load four 128-byte stripes from ``base + byte_offset``."""
    value = T.alloc_var(T.uint32x4)
    with T.sblock("moonep_fast_bulk_load_512"):
        T.import_source(
            r"""
#ifndef MOONEP_FAST_BULK_LOAD_512_
#define MOONEP_FAST_BULK_LOAD_512_
__attribute__((device)) __attribute__((always_inline)) inline void
moonep_fast_bulk_load_512(
    uint4* output, const void* base, long long byte_offset, bool remote) {
  const void* source = static_cast<const char*>(base) + byte_offset;
  *output = remote
      ? __ppu_remote_load_bulk_ncp_b32x4(source, 0)
      : __ppu_global_ldcv_bulk_b32x4(source);
}
#endif
"""
        )
        T.call_extern(
            "void",
            "moonep_fast_bulk_load_512",
            T.address_of(value),
            T.address_of(base),
            byte_offset,
            remote,
        )
    return value


@T.macro
def fast_bulk_store_128(base: T.Ref, byte_offset, value, remote):
    """Store one 128-byte stripe to ``base + byte_offset``."""
    with T.sblock("moonep_fast_bulk_store_128"):
        T.import_source(
            r"""
#ifndef MOONEP_FAST_BULK_STORE_128_
#define MOONEP_FAST_BULK_STORE_128_
__attribute__((device)) __attribute__((always_inline)) inline void
moonep_fast_bulk_store_128(
    void* base, long long byte_offset, uint value, bool remote) {
  void* destination = static_cast<char*>(base) + byte_offset;
  if (remote) {
    __ppu_remote_store_bulk_volatile_b32(destination, value, 0);
  } else {
    __ppu_global_stwt_bulk_b32(destination, value);
  }
}
#endif
"""
        )
        T.call_extern(
            "void",
            "moonep_fast_bulk_store_128",
            T.address_of(base),
            byte_offset,
            value,
            remote,
        )


@T.macro
def fast_bulk_store_256(base: T.Ref, byte_offset, value, remote):
    """Store two 128-byte stripes to ``base + byte_offset``."""
    with T.sblock("moonep_fast_bulk_store_256"):
        T.import_source(
            r"""
#ifndef MOONEP_FAST_BULK_STORE_256_
#define MOONEP_FAST_BULK_STORE_256_
__attribute__((device)) __attribute__((always_inline)) inline void
moonep_fast_bulk_store_256(
    void* base, long long byte_offset, uint2 value, bool remote) {
  void* destination = static_cast<char*>(base) + byte_offset;
  if (remote) {
    __ppu_remote_store_bulk_volatile_b32x2(destination, value, 0);
  } else {
    __ppu_global_stwt_bulk_b32x2(destination, value);
  }
}
#endif
"""
        )
        T.call_extern(
            "void",
            "moonep_fast_bulk_store_256",
            T.address_of(base),
            byte_offset,
            value,
            remote,
        )


@T.macro
def fast_bulk_store_512(base: T.Ref, byte_offset, value, remote):
    """Store four 128-byte stripes to ``base + byte_offset``."""
    with T.sblock("moonep_fast_bulk_store_512"):
        T.import_source(
            r"""
#ifndef MOONEP_FAST_BULK_STORE_512_
#define MOONEP_FAST_BULK_STORE_512_
__attribute__((device)) __attribute__((always_inline)) inline void
moonep_fast_bulk_store_512(
    void* base, long long byte_offset, uint4 value, bool remote) {
  void* destination = static_cast<char*>(base) + byte_offset;
  if (remote) {
    __ppu_remote_store_bulk_volatile_b32x4(destination, value, 0);
  } else {
    __ppu_global_stwt_bulk_b32x4(destination, value);
  }
}
#endif
"""
        )
        T.call_extern(
            "void",
            "moonep_fast_bulk_store_512",
            T.address_of(base),
            byte_offset,
            value,
            remote,
        )


__all__ = [
    "fast_bulk_load_128_ref",
    "fast_bulk_store_128_ref",
    "fast_bulk_load_128",
    "fast_bulk_load_256",
    "fast_bulk_load_512",
    "fast_bulk_store_128",
    "fast_bulk_store_256",
    "fast_bulk_store_512",
]
