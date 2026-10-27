"""TileLang PPU dispatch epilogue."""

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


def _compile_expand_duplicates(*, NvS: int, H: int, num_sms: int):
    grid_size = min(NvS, num_sms * _DEDUP_GRID_MULTIPLIER)
    full_chunks = H // _BULK_HIDDEN
    tail_hidden = H - full_chunks * _BULK_HIDDEN
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
                    # Load each primary chunk once and reuse it for every
                    # duplicate row.
                    for round_idx in T.serial(paired_chunk_rounds):
                        chunk0 = warp + round_idx * 2 * _WARPS
                        if chunk0 < full_chunks:
                            chunk1 = chunk0 + _WARPS
                            packed0 = T.alloc_local((4,), T.uint32)
                            packed1 = T.alloc_local((4,), T.uint32)
                            source0 = (
                                primary * H + chunk0 * _BULK_HIDDEN
                            )
                            packed0[0:4] = fast_bulk_load_512(
                                hidden[0], source0 * 2, T.bool(False)
                            )
                            if chunk1 < full_chunks:
                                source1 = (
                                    primary * H
                                    + chunk1 * _BULK_HIDDEN
                                )
                                packed1[0:4] = fast_bulk_load_512(
                                    hidden[0], source1 * 2, T.bool(False)
                                )
                            for duplicate in T.serial(count):
                                row = T.cast(loffs[begin + duplicate], T.int64)
                                destination0 = (
                                    row * H
                                    + chunk0 * _BULK_HIDDEN
                                )
                                fast_bulk_store_512(
                                    hidden[0],
                                    destination0 * 2,
                                    packed0[T.Ramp(0, 1, 4)],
                                    T.bool(False),
                                )
                                if chunk1 < full_chunks:
                                    destination1 = (
                                        row * H
                                        + chunk1 * _BULK_HIDDEN
                                    )
                                    fast_bulk_store_512(
                                        hidden[0],
                                        destination1 * 2,
                                        packed1[T.Ramp(0, 1, 4)],
                                        T.bool(False),
                                    )
                    if tx < tail_hidden:
                        column = full_chunks * _BULK_HIDDEN + tx
                        value = hidden[primary * H + column]
                        for duplicate in T.serial(count):
                            row = T.cast(loffs[begin + duplicate], T.int64)
                            hidden[row * H + column] = value

        return tilelang.compile(
            kernel,
            execution_backend="tvm_ffi",
            target="ppu",
            compile_flags=["--dopt=3"],
        )

    return _compile_cached(
        ("dispatch-expand-duplicates", NvS, H, grid_size), build
    )


def launch_dispatch_epilogue(
    ctx: dict,
    plan: MoonEPCommPlan,
    *,
    pdl_launch: bool = False,
) -> None:
    """Expand primary rows into duplicate slots after dispatch."""
    del pdl_launch
    NvS = int(ctx["NvS"])
    H = int(ctx["H"])
    hidden = ctx["hidden_buf_local"]
    assert hidden.dtype == torch.bfloat16 and hidden.is_contiguous(), (
        "hidden_buf_local must be contiguous bf16"
    )
    assert tuple(hidden.shape) == (NvS, H), (
        f"hidden_buf_local must be shape ({NvS}, {H}), "
        f"got {tuple(hidden.shape)}"
    )
    _validate_dedup_plan(ctx, plan, hidden.device)
    with torch.cuda.device(hidden.device):
        kernel = _compile_expand_duplicates(
            NvS=NvS,
            H=H,
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
