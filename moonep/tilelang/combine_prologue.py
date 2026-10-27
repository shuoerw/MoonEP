"""TileLang PPU combine prologue."""

from __future__ import annotations

import threading

import torch

from .planning import MoonEPCommPlan, _validate_dedup_plan


_THREADS = 256
_WARPS = _THREADS // 32
_BULK_HIDDEN = 512 // 2
_DEDUP_GRID_MULTIPLIER = 128
_COMPILE_LOCK = threading.Lock()
_COMPILED: dict[tuple, object] = {}


def _compile_cached(key: tuple, builder):
    device = torch.cuda.current_device()
    full_key = (
        device,
        torch.cuda.get_device_capability(device),
        *key,
    )
    compiled = _COMPILED.get(full_key)
    if compiled is not None:
        return compiled
    with _COMPILE_LOCK:
        compiled = _COMPILED.get(full_key)
        if compiled is None:
            with torch.cuda.device(device):
                compiled = builder()
            _COMPILED[full_key] = compiled
    return compiled


def _compile_reduce_duplicates(*, NvS: int, H: int, num_sms: int):
    grid_size = min(NvS, num_sms * _DEDUP_GRID_MULTIPLIER)
    full_chunks = H // _BULK_HIDDEN
    bulk_hidden = full_chunks * _BULK_HIDDEN
    tail_pairs = (H - bulk_hidden) // 2
    paired_chunk_rounds = (
        full_chunks + 2 * _WARPS - 1
    ) // (2 * _WARPS)

    def build():
        import tilelang
        import tilelang.ppu.language as T
        from .primitives import fast_bulk_load_512, fast_bulk_store_512

        @T.prim_func
        def kernel(
            hidden_ptr: T.handle,
            groups_ptr: T.handle,
            loffs_ptr: T.handle,
            counts_ptr: T.handle,
        ):
            hidden = T.match_buffer(
                hidden_ptr, (NvS * H,), dtype=T.bfloat16
            )
            groups = T.match_buffer(
                groups_ptr, (NvS * 3,), dtype=T.int32
            )
            loffs = T.match_buffer(loffs_ptr, (NvS,), dtype=T.int32)
            counts = T.match_buffer(counts_ptr, (2,), dtype=T.int32)
            with T.Kernel(grid_size, threads=_THREADS) as bx:
                tx = T.get_thread_binding()
                warp = tx // 32
                for group_index in T.serial(
                    bx, counts[0], grid_size
                ):
                    primary = T.cast(groups[group_index * 3], T.int64)
                    begin = groups[group_index * 3 + 1]
                    count = groups[group_index * 3 + 2]
                    for round_idx in T.serial(paired_chunk_rounds):
                        chunk0 = warp + round_idx * 2 * _WARPS
                        chunk0_valid = chunk0 < full_chunks
                        chunk0_safe = T.if_then_else(
                            chunk0_valid, chunk0, 0
                        )
                        chunk1 = chunk0 + _WARPS
                        sums0 = T.alloc_local((8,), T.float32)
                        sums1 = T.alloc_local((8,), T.float32)
                        values0 = T.alloc_local((8,), T.bfloat16)
                        values1 = T.alloc_local((8,), T.bfloat16)
                        primary0 = (
                            primary * H + chunk0_safe * _BULK_HIDDEN
                        )
                        primary1 = primary * H + chunk1 * _BULK_HIDDEN
                        values0[0:8] = T.reinterpret(
                            fast_bulk_load_512(
                                hidden[0], primary0 * 2, T.bool(False)
                            ),
                            "bfloat16x8",
                        )
                        for value in T.unroll(8):
                            sums0[value] = T.cast(values0[value], T.float32)
                        if chunk1 < full_chunks:
                            values1[0:8] = T.reinterpret(
                                fast_bulk_load_512(
                                    hidden[0], primary1 * 2, T.bool(False)
                                ),
                                "bfloat16x8",
                            )
                            for value in T.unroll(8):
                                sums1[value] = T.cast(
                                    values1[value], T.float32
                                )

                        for duplicate in T.serial(count):
                            row = T.cast(loffs[begin + duplicate], T.int64)
                            source0 = (
                                row * H + chunk0_safe * _BULK_HIDDEN
                            )
                            values0[0:8] = T.reinterpret(
                                fast_bulk_load_512(
                                    hidden[0], source0 * 2, T.bool(False)
                                ),
                                "bfloat16x8",
                            )
                            for value in T.unroll(8):
                                sums0[value] += T.cast(
                                    values0[value], T.float32
                                )
                            if chunk1 < full_chunks:
                                source1 = (
                                    row * H + chunk1 * _BULK_HIDDEN
                                )
                                values1[0:8] = T.reinterpret(
                                    fast_bulk_load_512(
                                        hidden[0], source1 * 2, T.bool(False)
                                    ),
                                    "bfloat16x8",
                                )
                                for value in T.unroll(8):
                                    sums1[value] += T.cast(
                                        values1[value], T.float32
                                    )

                        packed0 = T.alloc_local((4,), T.uint32)
                        packed1 = T.alloc_local((4,), T.uint32)
                        for word in T.unroll(4):
                            low0 = T.cast(sums0[word * 2], T.bfloat16)
                            high0 = T.cast(
                                sums0[word * 2 + 1], T.bfloat16
                            )
                            packed0[word] = T.cast(
                                T.reinterpret(low0, T.uint16), T.uint32
                            ) | (
                                T.cast(
                                    T.reinterpret(high0, T.uint16),
                                    T.uint32,
                                )
                                << 16
                            )
                            if chunk1 < full_chunks:
                                low1 = T.cast(
                                    sums1[word * 2], T.bfloat16
                                )
                                high1 = T.cast(
                                    sums1[word * 2 + 1], T.bfloat16
                                )
                                packed1[word] = T.cast(
                                    T.reinterpret(low1, T.uint16), T.uint32
                                ) | (
                                    T.cast(
                                        T.reinterpret(high1, T.uint16),
                                        T.uint32,
                                    )
                                    << 16
                                )
                        if chunk0_valid:
                            fast_bulk_store_512(
                                hidden[0],
                                primary0 * 2,
                                packed0[T.Ramp(0, 1, 4)],
                                T.bool(False),
                            )
                        if chunk1 < full_chunks:
                            fast_bulk_store_512(
                                hidden[0],
                                primary1 * 2,
                                packed1[T.Ramp(0, 1, 4)],
                                T.bool(False),
                            )

                    if tx < tail_pairs:
                        pair = tx
                        value0 = T.alloc_local((1,), T.float32)
                        value1 = T.alloc_local((1,), T.float32)
                        column = bulk_hidden + pair * 2
                        value0[0] = T.cast(
                            hidden[primary * H + column], T.float32
                        )
                        value1[0] = T.cast(
                            hidden[primary * H + column + 1], T.float32
                        )
                        for duplicate in T.serial(count):
                            row = T.cast(loffs[begin + duplicate], T.int64)
                            value0[0] += T.cast(
                                hidden[row * H + column], T.float32
                            )
                            value1[0] += T.cast(
                                hidden[row * H + column + 1], T.float32
                            )
                        low = T.cast(value0[0], T.bfloat16)
                        high = T.cast(value1[0], T.bfloat16)
                        hidden[primary * H + column] = low
                        hidden[primary * H + column + 1] = high
        return tilelang.compile(
            kernel,
            execution_backend="tvm_ffi",
            target="ppu",
            compile_flags=["--dopt=3"],
        )

    return _compile_cached(
        ("combine-reduce-duplicates", NvS, H, grid_size), build
    )


def launch_combine_prologue(
    ctx: dict,
    plan: MoonEPCommPlan,
    *,
    pdl_trigger: bool = False,
) -> None:
    """Reduce duplicate rows into their primary rows before combine."""
    del pdl_trigger
    hidden = ctx["hidden_buf_local"]
    assert hidden.dtype == torch.bfloat16 and hidden.is_contiguous()
    assert tuple(hidden.shape) == (int(ctx["NvS"]), int(ctx["H"]))
    _validate_dedup_plan(ctx, plan, hidden.device)
    with torch.cuda.device(hidden.device):
        kernel = _compile_reduce_duplicates(
            NvS=int(ctx["NvS"]),
            H=int(ctx["H"]),
            num_sms=int(ctx["num_sms_dedup"]),
        )
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            kernel(
                hidden.reshape(-1),
                plan.dup_groups.reshape(-1),
                plan.dup_loffs,
                plan.dup_counts,
            )
