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
    with T.sblock("moonep_fast_bulk_load_128_ref"):
        T.import_source(
            r"""
#ifndef MOONEP_FAST_BULK_LOAD_128_REF_
#define MOONEP_FAST_BULK_LOAD_128_REF_
__attribute__((device)) __attribute__((always_inline)) inline void
moonep_fast_bulk_load_128_ref(
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
            "moonep_fast_bulk_load_128_ref",
            T.address_of(value),
            T.address_of(src),
            remote,
        )
    return value


@T.macro
def fast_bulk_store_128_ref(dst: T.Ref, value, remote):
    """Store one 128-byte stripe to an indexed reference."""
    with T.sblock("moonep_fast_bulk_store_128_ref"):
        T.import_source(
            r"""
#ifndef MOONEP_FAST_BULK_STORE_128_REF_
#define MOONEP_FAST_BULK_STORE_128_REF_
__attribute__((device)) __attribute__((always_inline)) inline void
moonep_fast_bulk_store_128_ref(
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
            "moonep_fast_bulk_store_128_ref",
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


# OKI macros adapt SDK-defined pointers and references. The calling kernel
# imports the SDK declarations and links its device library.
@T.macro
def oki_reserve(queue, count):
    """Reserve WQEs, updating the one-element uint32 count buffer."""
    base = T.alloc_var(T.uint32)
    with T.sblock("moonep_oki_reserve"):
        T.import_source(r"""
#ifndef MOONEP_OKI_RESERVE_
#define MOONEP_OKI_RESERVE_
__device__ __forceinline__ uint32_t moonep_oki_reserve(
    uint64_t queue, uint32_t* count) {
  uint32_t skip = 0;
  return ltOkiReserveWqeSlots(reinterpret_cast<ltOkiQP_t>(queue), *count, skip);
}
#endif
""")
        base = T.call_extern("uint32", "moonep_oki_reserve", queue,
                             T.access_ptr(count, "rw", extent=1))
    return base


@T.macro
def oki_copy_local(queue, src, dst, src_offset, dst_offset, size, upload):
    """Enqueue a local DMA copy between byte buffers."""
    with T.sblock("moonep_oki_copy_local"):
        T.import_source(r"""
#ifndef MOONEP_OKI_COPY_LOCAL_
#define MOONEP_OKI_COPY_LOCAL_
__device__ __forceinline__ void moonep_oki_copy_local(
    uint64_t queue, const void* src, void* dst, uint32_t bytes, bool upload) {
  ltOkiQPWriteDmaCopyWqe(reinterpret_cast<ltOkiQP_t>(queue),
      reinterpret_cast<uint64_t>(src), reinterpret_cast<uint64_t>(dst),
      bytes, upload, false);
}
#endif
""")
        T.call_extern("void", "moonep_oki_copy_local", queue,
                      T.access_ptr(src, "r", offset=src_offset, extent=size),
                      T.access_ptr(dst, "w", offset=dst_offset, extent=size), size, upload)


@T.macro
def oki_copy_remote(queue, src, dst, src_offset, dst_offset, size):
    """Enqueue a local-to-remote DMA copy between byte buffers."""
    with T.sblock("moonep_oki_copy_remote"):
        T.import_source(r"""
#ifndef MOONEP_OKI_COPY_REMOTE_
#define MOONEP_OKI_COPY_REMOTE_
__device__ __forceinline__ void moonep_oki_copy_remote(
    uint64_t queue, const void* src, void* dst, uint32_t bytes) {
  ltOkiQPWriteIcnDmaCopyWqe(reinterpret_cast<ltOkiQP_t>(queue),
      reinterpret_cast<uint64_t>(src), reinterpret_cast<uint64_t>(dst), bytes, true);
}
#endif
""")
        T.call_extern("void", "moonep_oki_copy_remote", queue,
                      T.access_ptr(src, "r", offset=src_offset, extent=size),
                      T.access_ptr(dst, "w", offset=dst_offset, extent=size), size)


@T.macro
def oki_fence_remote(queue, fences, index, step):
    """Enqueue a remote completion fence in a uint64 buffer."""
    with T.sblock("moonep_oki_fence_remote"):
        T.import_source(r"""
#ifndef MOONEP_OKI_FENCE_REMOTE_
#define MOONEP_OKI_FENCE_REMOTE_
__device__ __forceinline__ void moonep_oki_fence_remote(
    uint64_t queue, uint64_t* address, uint64_t step) {
  ltOkiQPWriteIcnFenceWqe(reinterpret_cast<ltOkiQP_t>(queue),
      reinterpret_cast<uint64_t>(address), step, false);
}
#endif
""")
        T.call_extern("void", "moonep_oki_fence_remote", queue,
                      T.access_ptr(fences, "w", offset=index, extent=1), step)


@T.macro
def oki_fence_local(queue, step, upload):
    """Enqueue a local completion fence on the queue's completion counter."""
    with T.sblock("moonep_oki_fence_local"):
        T.import_source(r"""
#ifndef MOONEP_OKI_FENCE_LOCAL_
#define MOONEP_OKI_FENCE_LOCAL_
__device__ __forceinline__ void moonep_oki_fence_local(
    uint64_t queue, uint64_t step, bool upload) {
  auto qp = reinterpret_cast<ltOkiQP_t>(queue);
  ltOkiQPWriteFenceWqe(qp, reinterpret_cast<uint64_t>(ltOkiQPGetCQ(qp)),
      step, upload, true);
}
#endif
""")
        T.call_extern("void", "moonep_oki_fence_local", queue, step, upload)


@T.macro
def oki_post(queue, base, count):
    """Post reserved WQEs without ringing the doorbell."""
    with T.sblock("moonep_oki_post"):
        T.import_source(r"""
#ifndef MOONEP_OKI_POST_
#define MOONEP_OKI_POST_
__device__ __forceinline__ void moonep_oki_post(
    uint64_t queue, uint32_t base, uint32_t count) {
  ltOkiPostSend(reinterpret_cast<ltOkiQP_t>(queue), base, count);
}
#endif
""")
        T.call_extern("void", "moonep_oki_post", queue, base, count)


@T.macro
def oki_ring(queue):
    """Ring the queue's doorbell after posting WQEs."""
    with T.sblock("moonep_oki_ring"):
        T.import_source(r"""
#ifndef MOONEP_OKI_RING_
#define MOONEP_OKI_RING_
__device__ __forceinline__ void moonep_oki_ring(uint64_t queue) {
  ltOkiRingDB(reinterpret_cast<ltOkiQP_t>(queue));
}
#endif
""")
        T.call_extern("void", "moonep_oki_ring", queue)


@T.macro
def oki_poll(queue, step):
    """Wait for the queue's completion counter to reach the requested step."""
    with T.sblock("moonep_oki_poll"):
        T.import_source(r"""
#ifndef MOONEP_OKI_POLL_
#define MOONEP_OKI_POLL_
__device__ __forceinline__ void moonep_oki_poll(uint64_t queue, uint64_t step) {
  ltOkiPollCQ(ltOkiQPGetCQ(reinterpret_cast<ltOkiQP_t>(queue)), step);
}
#endif
""")
        T.call_extern("void", "moonep_oki_poll", queue, step)


__all__ = [
    "fast_bulk_load_128_ref",
    "fast_bulk_store_128_ref",
    "fast_bulk_load_128",
    "fast_bulk_load_256",
    "fast_bulk_load_512",
    "fast_bulk_store_128",
    "fast_bulk_store_256",
    "fast_bulk_store_512",
    "oki_reserve",
    "oki_copy_local",
    "oki_copy_remote",
    "oki_fence_remote",
    "oki_fence_local",
    "oki_post",
    "oki_ring",
    "oki_poll",
]
