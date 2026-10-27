"""Stream-ordered rank synchronization over MoonEP VMM mappings."""

import threading

import torch


# Device clock cycles, not nanoseconds. Keep the timeout independent of
# host scheduling while allowing peers to finish a cold JIT compilation.
BARRIER_TIMEOUT_CYCLES = 600_000_000_000


_SYNC_COMPILE_LOCK = threading.Lock()
_SYNC_KERNELS: dict[tuple, object] = {}


def _num_threads_for_ranks(R: int) -> int:
    assert R > 0, f"R must be positive, got {R}"
    assert R <= 1024, f"inter_rank_sync supports at most 1024 ranks, got {R}"
    return max(32, ((R + 31) // 32) * 32)


def _compile_inter_rank_sync(world_size: int, meta_stride: int, barrier_off: int):
    num_threads = _num_threads_for_ranks(world_size)
    device = torch.cuda.current_device()
    key = (
        device,
        torch.cuda.get_device_capability(device),
        world_size,
        meta_stride,
        barrier_off,
    )
    compiled = _SYNC_KERNELS.get(key)
    if compiled is not None:
        return compiled
    with _SYNC_COMPILE_LOCK:
        compiled = _SYNC_KERNELS.get(key)
        if compiled is not None:
            return compiled

        import tilelang
        import tilelang.ppu.language as T

        @T.prim_func
        def kernel(
            meta_ptr: T.handle,
            rank: T.int32,
        ):
            meta = T.match_buffer(
                meta_ptr, (world_size * meta_stride,), dtype=T.int32
            )
            with T.Kernel(1, threads=num_threads):
                tx = T.get_thread_binding()
                state = meta[rank * meta_stride + barrier_off + 2]
                phase = state & 1
                sign = T.if_then_else((state & 2) != 0, -1, 1)
                target = T.if_then_else(sign > 0, world_size, 0)
                if tx < world_size:
                    T.atomic_add(
                        meta[tx * meta_stride + barrier_off + phase],
                        sign,
                        memory_order="release",
                        scope="system",
                    )
                T.sync_threads()
                if tx == 0:
                    loaded = T.alloc_local((1,), T.int32)
                    local_index = rank * meta_stride + barrier_off + phase
                    # Publish the next phase before waiting so faster ranks
                    # may safely pre-arrive.
                    state_index = rank * meta_stride + barrier_off + 2
                    meta[state_index] = (meta[state_index] + 1) & 3
                    start = T.alloc_local((1,), T.uint64)
                    start[0] = T.call_extern("uint64", "clock64")
                    loaded[0] = T.atomic_load(
                        meta[local_index],
                        memory_order="acquire",
                        scope="system",
                    )
                    while loaded[0] != target:
                        if (
                            T.call_extern("uint64", "clock64") - start[0]
                            > T.uint64(BARRIER_TIMEOUT_CYCLES)
                        ):
                            T.call_extern("void", "__builtin_trap")
                        loaded[0] = T.atomic_load(
                            meta[local_index],
                            memory_order="acquire",
                            scope="system",
                        )

        compiled = tilelang.compile(
            kernel,
            target="ppu",
            compile_flags=["--dopt=3"],
            execution_backend="tvm_ffi",
        )
        _SYNC_KERNELS[key] = compiled
        return compiled


def launch_inter_rank_sync(ctx: dict) -> None:
    meta = ctx["meta_buf"]
    compiled = _compile_inter_rank_sync(
        int(ctx["R"]), int(ctx["meta_chunk_padded"]), int(ctx["BARRIER_OFF"])
    )
    import tvm_ffi

    with torch.cuda.device(meta.device), tvm_ffi.use_torch_stream():
        compiled(meta, int(ctx["rank"]))
