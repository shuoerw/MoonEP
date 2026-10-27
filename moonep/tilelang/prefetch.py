"""TileLang implementation of owner-push expert prefetch on PPU."""

from __future__ import annotations

import math
import threading

import torch


_BULK_BYTES = 512
_THREADS = 1024
_WARPS = _THREADS // 32
_COMPILE_LOCK = threading.Lock()
_COMPILED_PREFETCH: dict[tuple, object] = {}


def _compile_push_prefetch(
    *,
    local_bytes: int,
    pool_span_bytes: int,
    rank_stride_bytes: int,
    row_bytes: int,
    world_size: int,
    local_rank: int,
    slots: int,
    num_sms: int,
    weight_count: int = 1,
):
    device = torch.cuda.current_device()
    key = (
        device,
        torch.cuda.get_device_capability(device),
        local_bytes,
        pool_span_bytes,
        rank_stride_bytes,
        row_bytes,
        world_size,
        local_rank,
        slots,
        num_sms,
        weight_count,
    )
    compiled = _COMPILED_PREFETCH.get(key)
    if compiled is not None:
        return compiled

    with _COMPILE_LOCK:
        compiled = _COMPILED_PREFETCH.get(key)
        if compiled is not None:
            return compiled

        import tilelang
        import tilelang.ppu.language as T

        from .primitives import (
            fast_bulk_load_128,
            fast_bulk_load_256,
            fast_bulk_load_512,
            fast_bulk_store_128,
            fast_bulk_store_256,
            fast_bulk_store_512,
        )

        epn = local_bytes // row_bytes
        full_chunks = row_bytes // _BULK_BYTES
        remainder = row_bytes - full_chunks * _BULK_BYTES
        has_tail_256 = remainder >= 256
        remainder -= 256 if has_tail_256 else 0
        has_tail_128 = remainder >= 128
        remainder -= 128 if has_tail_128 else 0
        tail_words = remainder // 4
        plan_entries = world_size * slots
        plan_tiles = (plan_entries + _THREADS - 1) // _THREADS
        entries_per_tile = max(1, min(plan_entries, _THREADS))

        @T.macro
        def copy_rows(local, pool, experts, copy_block, copy_sms):
            max_chunks = (full_chunks + copy_sms - 1) // copy_sms
            rounds = (max_chunks + _WARPS - 1) // _WARPS
            tx = T.get_thread_binding()
            warp = tx // 32
            lane = tx % 32
            begin_chunk = full_chunks * copy_block // copy_sms
            end_chunk = full_chunks * (copy_block + 1) // copy_sms

            copy_entries = T.alloc_shared(
                (entries_per_tile,), T.int32, scope="shared"
            )
            warp_counts = T.alloc_shared(
                (_WARPS,), T.int32, scope="shared"
            )
            warp_offsets = T.alloc_shared(
                (_WARPS,), T.int32, scope="shared"
            )
            tile_count = T.alloc_shared((1,), T.int32, scope="shared")
            copy_count = T.alloc_local((1,), T.int32)
            lanes_before = (
                T.uint64(1) << T.cast(lane, T.uint64)
            ) - T.uint64(1)

            # Compact owned entries in stable rank/slot order. Tiling
            # bounds shared memory independently of the plan size.
            for plan_tile in T.serial(plan_tiles):
                entry = plan_tile * _THREADS + tx
                owned_expert = T.alloc_local((1,), T.int32)
                owned_expert[0] = -1
                if entry < plan_entries:
                    owned_expert[0] = experts[entry]
                owned = (
                    owned_expert[0] >= local_rank * epn
                    and owned_expert[0] < (local_rank + 1) * epn
                )
                owned_mask = T.ballot_sync(owned)
                if lane == 0:
                    warp_counts[warp] = T.popcount(owned_mask)
                # Also wait for the previous tile's copies before
                # reusing copy_entries.
                T.sync_threads()
                if warp == 0:
                    inclusive = T.alloc_local((1,), T.int32)
                    original_count = T.alloc_local((1,), T.int32)
                    shuffled = T.alloc_local((1,), T.int32)
                    original_count[0] = warp_counts[lane]
                    inclusive[0] = original_count[0]
                    shuffled[0] = T.shfl_up(inclusive[0], 1)
                    if lane >= 1:
                        inclusive[0] = inclusive[0] + shuffled[0]
                    shuffled[0] = T.shfl_up(inclusive[0], 2)
                    if lane >= 2:
                        inclusive[0] = inclusive[0] + shuffled[0]
                    shuffled[0] = T.shfl_up(inclusive[0], 4)
                    if lane >= 4:
                        inclusive[0] = inclusive[0] + shuffled[0]
                    shuffled[0] = T.shfl_up(inclusive[0], 8)
                    if lane >= 8:
                        inclusive[0] = inclusive[0] + shuffled[0]
                    shuffled[0] = T.shfl_up(inclusive[0], 16)
                    if lane >= 16:
                        inclusive[0] = inclusive[0] + shuffled[0]
                    warp_offsets[lane] = inclusive[0] - original_count[0]
                    if lane == 31:
                        tile_count[0] = inclusive[0]
                T.sync_threads()
                if owned:
                    compact_index = warp_offsets[warp] + T.popcount(
                        owned_mask & lanes_before
                    )
                    copy_entries[compact_index] = entry
                T.sync_threads()
                copy_count[0] = tile_count[0]
                for copy_index in T.serial(copy_count[0]):
                    copy_entry = copy_entries[copy_index]
                    destination = copy_entry // slots
                    slot = copy_entry % slots
                    expert = experts[destination * slots + slot]
                    if (
                        expert >= local_rank * epn
                        and expert < (local_rank + 1) * epn
                    ):
                        local_expert = expert - local_rank * epn
                        row_source = (
                            T.cast(local_expert, T.int64) * row_bytes
                        )
                        row_output = (
                            T.cast(destination, T.int64)
                            * rank_stride_bytes
                            + T.cast(slot, T.int64) * row_bytes
                        )
                        remote = destination != local_rank

                        if full_chunks > 0:
                            for copy_round in T.serial(rounds):
                                chunk = (
                                    begin_chunk
                                    + warp
                                    + copy_round * _WARPS
                                )
                                if chunk < end_chunk:
                                    offset = (
                                        T.cast(chunk, T.int64) * _BULK_BYTES
                                    )
                                    value = fast_bulk_load_512(
                                        local[0], row_source + offset,
                                        T.bool(False),
                                    )
                                    fast_bulk_store_512(
                                        pool[0], row_output + offset,
                                        value, remote,
                                    )

                        if copy_block == 0 and warp == 0:
                            tail_offset = full_chunks * _BULK_BYTES
                            if has_tail_256:
                                tail256 = fast_bulk_load_256(
                                    local[0], row_source + tail_offset,
                                    T.bool(False),
                                )
                                fast_bulk_store_256(
                                    pool[0], row_output + tail_offset,
                                    tail256, remote,
                                )
                            tail_offset += 256 if has_tail_256 else 0
                            if has_tail_128:
                                tail128 = fast_bulk_load_128(
                                    local[0], row_source + tail_offset,
                                    T.bool(False),
                                )
                                fast_bulk_store_128(
                                    pool[0], row_output + tail_offset,
                                    tail128, remote,
                                )
                            tail_offset += 128 if has_tail_128 else 0
                            if lane < tail_words:
                                word_offset = tail_offset + lane * 4
                                tail_word = T.ldg32(
                                    local[
                                        row_source + word_offset :
                                        row_source + word_offset + 4
                                    ]
                                )
                                if remote:
                                    destination_word = T.decl_buffer(
                                        (pool_span_bytes // 4,),
                                        dtype=T.uint32, data=pool.data,
                                    )
                                    T.atomic_store(
                                        destination_word[
                                            (row_output + word_offset) // 4
                                        ],
                                        tail_word,
                                        memory_order="relaxed",
                                        scope="system",
                                    )
                                else:
                                    T.stg32(
                                        pool[
                                            row_output + word_offset :
                                            row_output + word_offset + 4
                                        ],
                                        tail_word,
                                    )

        @T.prim_func
        def kernel(
            local0_ptr: T.handle, pool0_ptr: T.handle,
            local1_ptr: T.handle, pool1_ptr: T.handle,
            local2_ptr: T.handle, pool2_ptr: T.handle,
            experts_ptr: T.handle,
        ):
            local0 = T.match_buffer(local0_ptr, (local_bytes,), dtype=T.uint8)
            local1 = T.match_buffer(local1_ptr, (local_bytes,), dtype=T.uint8)
            local2 = T.match_buffer(local2_ptr, (local_bytes,), dtype=T.uint8)
            pool0 = T.match_buffer(pool0_ptr, (pool_span_bytes,), dtype=T.uint8)
            pool1 = T.match_buffer(pool1_ptr, (pool_span_bytes,), dtype=T.uint8)
            pool2 = T.match_buffer(pool2_ptr, (pool_span_bytes,), dtype=T.uint8)
            experts = T.match_buffer(
                experts_ptr, (world_size * slots,), dtype=T.int32
            )
            with T.Kernel(num_sms, threads=_THREADS) as bx:
                copy_rows(local0, pool0, experts, bx, num_sms)
                if weight_count == 3:
                    copy_rows(local1, pool1, experts, bx, num_sms)
                    copy_rows(local2, pool2, experts, bx, num_sms)

        compiled = tilelang.compile(
            kernel,
            execution_backend="tvm_ffi",
            target="ppu",
            compile_flags=["--dopt=3"],
        )
        _COMPILED_PREFETCH[key] = compiled
        return compiled


def _pool_byte_view(pool: torch.Tensor) -> tuple[torch.Tensor, int]:
    rank_elements = pool[0].numel()
    rank_stride = int(pool.stride(0))
    span_elements = (int(pool.shape[0]) - 1) * rank_stride + rank_elements
    flat = pool.as_strided(
        (span_elements,),
        (1,),
        storage_offset=pool.storage_offset(),
    )
    return flat.view(torch.uint8), rank_stride * pool.element_size()


def _copy_tensors(
    pairs: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    experts_to_copy: torch.Tensor,
    *,
    rank: int,
    num_sms: int,
) -> None:
    local_views = tuple(local.view(torch.uint8).reshape(-1) for local, _ in pairs)
    pool_views = tuple(_pool_byte_view(pool) for _, pool in pairs)
    local_bytes = int(local_views[0].numel())
    row_bytes = math.prod(int(value) for value in pairs[0][0].shape[1:])
    row_bytes *= pairs[0][0].element_size()
    kernel = _compile_push_prefetch(
        local_bytes=local_bytes,
        pool_span_bytes=int(pool_views[0][0].numel()),
        rank_stride_bytes=pool_views[0][1],
        row_bytes=row_bytes,
        world_size=int(experts_to_copy.shape[0]),
        local_rank=rank,
        slots=int(experts_to_copy.shape[1]),
        num_sms=num_sms,
        weight_count=len(pairs),
    )

    import tvm_ffi

    # Unused handles retain one launch signature for single and triple copies.
    indices = (0, 1, 2) if len(pairs) == 3 else (0, 0, 0)
    arguments = tuple(
        tensor
        for index in indices
        for tensor in (local_views[index], pool_views[index][0])
    )
    with tvm_ffi.use_torch_stream():
        kernel(*arguments, experts_to_copy.reshape(-1))


def launch_prefetch(
    local_weights: tuple[torch.Tensor, ...],
    prefetch_buffers: tuple[torch.Tensor, ...],
    experts_to_copy: torch.Tensor,
    *,
    rank: int,
    num_sms: int,
    meta_buf: torch.Tensor,
    meta_stride: int,
    barrier_off: int,
    grid_sync_bar: torch.Tensor,
    scales: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
) -> None:
    """Push local expert rows into all destination ranks' selected slots."""
    del grid_sync_bar
    if len(local_weights) != 3 or len(prefetch_buffers) != 3:
        raise ValueError("prefetch requires three weights and three pools")
    if (
        experts_to_copy.dtype != torch.int32
        or not experts_to_copy.is_contiguous()
    ):
        raise ValueError("experts_to_copy must be contiguous int32 [R, epn]")
    if experts_to_copy.ndim != 2:
        raise ValueError("experts_to_copy must have rank 2")
    world_size, slots = (int(value) for value in experts_to_copy.shape)
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is outside world_size={world_size}")
    if not isinstance(num_sms, int) or num_sms <= 0:
        raise ValueError(f"num_sms must be a positive int, got {num_sms}")

    pairs = list(zip(local_weights, prefetch_buffers, strict=True))
    pairs.extend(scales or ())
    for local, pool in pairs:
        if not local.is_contiguous() or local.ndim < 2:
            raise ValueError("local experts must be contiguous with rank >= 2")
        if int(local.shape[0]) != slots:
            raise ValueError("local expert count must equal the slot count")
        if pool.dtype != local.dtype or pool.ndim != local.ndim + 1:
            raise ValueError(
                "prefetch pool rank or dtype does not match local experts"
            )
        if tuple(pool.shape[:2]) != (world_size, slots):
            raise ValueError("prefetch pool must have leading shape [R, epn]")
        if tuple(pool.shape[2:]) != tuple(local.shape[1:]):
            raise ValueError("prefetch pool row shape does not match local experts")
        if not pool[0].is_contiguous():
            raise ValueError("each rank's prefetch pool must be contiguous")
        if int(pool.stride(0)) < pool[0].numel():
            raise ValueError("prefetch pool rank payloads must not overlap")
        row_bytes = math.prod(int(value) for value in local.shape[1:])
        row_bytes *= local.element_size()
        if row_bytes <= 0 or row_bytes % 4:
            raise ValueError(
                "expert rows must have a positive 4-byte-aligned size, "
                f"got {row_bytes}"
            )
        if (
            local.data_ptr() % 4 or pool.data_ptr() % 4
            or (int(pool.stride(0)) * pool.element_size()) % 4
        ):
            raise ValueError(
                "prefetch base addresses and rank strides must be 4-byte aligned"
            )

    # The three weight matrices may share a kernel when their byte layouts
    # match. Each CTA copies its slice of every matrix in one launch.
    layouts = [
        (local.numel() * local.element_size(),
         pool[0].numel() * pool.element_size(),
         int(pool.stride(0)) * pool.element_size())
        for local, pool in pairs[:3]
    ]
    source_ranges = [
        (local.data_ptr(), local.data_ptr() + local.numel() * local.element_size())
        for local, _ in pairs[:3]
    ]
    output_ranges = [
        (pool.data_ptr(), pool.data_ptr() + (
            (world_size - 1) * int(pool.stride(0)) + pool[0].numel()
        ) * pool.element_size())
        for _, pool in pairs[:3]
    ]
    disjoint = all(
        end_a <= begin_b or end_b <= begin_a
        for i, (begin_a, end_a) in enumerate(output_ranges)
        for begin_b, end_b in output_ranges[i + 1:] + source_ranges
    )
    fuse_weights = (
        disjoint and all(layout == layouts[0] for layout in layouts)
    )
    first_separate = 3 if fuse_weights else 0
    if fuse_weights:
        _copy_tensors(
            tuple(pairs[:3]), experts_to_copy, rank=rank, num_sms=num_sms
        )
    for pair in pairs[first_separate:]:
        _copy_tensors((pair,), experts_to_copy, rank=rank, num_sms=num_sms)

    from .sync import launch_inter_rank_sync

    launch_inter_rank_sync(
        {
            "meta_buf": meta_buf,
            "R": world_size,
            "meta_chunk_padded": meta_stride,
            "BARRIER_OFF": barrier_off,
            "rank": rank,
        }
    )
