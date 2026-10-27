"""TileLang PPU expert-gradient reduction and slot clearing."""

from __future__ import annotations

import threading

import torch

_BULK_FLOATS = 512 // 4
_DIM_ALIGN = 128
_THREADS = 1024
_WARPS = _THREADS // 32
_REDUCE_CHUNK_BATCH = 4
_CLEAR_CHUNK_BATCH = 8
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


def _compile_grad_reduce(
    *,
    world_size: int,
    local_experts: int,
    slots: int,
    H: int,
    Hp: int,
    rank_stride: int,
    num_sms: int,
    meta_stride: int,
    barrier_off: int,
):
    """Compile the fused reduction, rank barrier, and slot-clear kernel."""
    epn = local_experts
    matrix = H * Hp
    reduce_elements = (world_size - 1) * rank_stride + slots * matrix
    chunks_per_matrix = matrix // _BULK_FLOATS
    reduce_batches_per_matrix = chunks_per_matrix // _REDUCE_CHUNK_BATCH
    clear_batches_per_matrix = chunks_per_matrix // _CLEAR_CHUNK_BATCH

    def build():
        import tilelang
        import tilelang.ppu.language as T

        from .primitives import fast_bulk_load_512, fast_bulk_store_512
        from .sync import BARRIER_TIMEOUT_CYCLES

        @T.macro
        def cross_rank_barrier(
            meta,
            barrier_state,
            barrier_phase,
            barrier_sign,
            barrier_target,
            rank,
            bx,
            tx,
        ):
            # Complete all remote reads before publishing this rank's arrival.
            T.sync_grid()
            local_meta = rank * meta_stride
            if tx == 0:
                barrier_state[0] = meta[local_meta + barrier_off + 2]
                barrier_phase[0] = barrier_state[0] & 1
                barrier_sign[0] = T.if_then_else(
                    (barrier_state[0] & 2) != 0, -1, 1
                )
                barrier_target[0] = T.if_then_else(
                    barrier_sign[0] > 0, world_size, 0
                )
            T.sync_threads()
            if bx == 0 and tx < world_size:
                T.atomic_add(
                    meta[
                        tx * meta_stride
                        + barrier_off
                        + barrier_phase[0]
                    ],
                    barrier_sign[0],
                    memory_order="release",
                    scope="system",
                )
            T.sync_threads()
            if bx == 0 and tx == 0:
                own_index = local_meta + barrier_off + barrier_phase[0]
                meta[local_meta + barrier_off + 2] = (
                    meta[local_meta + barrier_off + 2] + 1
                ) & 3
                start = T.alloc_local((1,), T.uint64)
                loaded_value = T.alloc_local((1,), T.int32)
                start[0] = T.call_extern("uint64", "clock64")
                loaded_value[0] = T.atomic_load(
                    meta[own_index],
                    memory_order="acquire",
                    scope="system",
                )
                while loaded_value[0] != barrier_target[0]:
                    if (
                        T.call_extern("uint64", "clock64") - start[0]
                        > T.uint64(BARRIER_TIMEOUT_CYCLES)
                    ):
                        T.call_extern("void", "__builtin_trap")
                    loaded_value[0] = T.atomic_load(
                        meta[own_index],
                        memory_order="acquire",
                        scope="system",
                    )
            T.sync_grid()

        @T.prim_func
        def kernel(
            grads_ptr: T.handle,
            reduce_ptr: T.handle,
            experts_ptr: T.handle,
            meta_ptr: T.handle,
            rank: T.int32,
        ):
            grads = T.match_buffer(
                grads_ptr, (epn * matrix,), dtype=T.float32
            )
            reduce = T.match_buffer(
                reduce_ptr,
                (reduce_elements,),
                dtype=T.float32,
            )
            experts = T.match_buffer(
                experts_ptr, (world_size * slots,), dtype=T.int32
            )
            meta = T.match_buffer(
                meta_ptr, (world_size * meta_stride,), dtype=T.int32
            )
            with T.Kernel(num_sms, threads=_THREADS) as bx:
                tx = T.get_thread_binding()
                warp = tx // 32
                lane = tx % 32
                local_expert_by_entry = T.alloc_shared(
                    (world_size * slots,), T.int32, scope="shared"
                )
                active_experts = T.alloc_shared(
                    (epn,), T.int32, scope="shared"
                )
                active_count = T.alloc_shared(
                    (1,), T.int32, scope="shared"
                )
                matched_offsets = T.alloc_shared(
                    (epn + 1,), T.int32, scope="shared"
                )
                # Dense [rank, slot] row index, not a byte/element address.
                matched_entries = T.alloc_shared(
                    (world_size * slots,), T.int32, scope="shared"
                )
                clear_slots = T.alloc_shared(
                    (slots,), T.int32, scope="shared"
                )
                clear_count = T.alloc_shared(
                    (1,), T.int32, scope="shared"
                )
                barrier_state = T.alloc_shared((1,), T.int32)
                barrier_phase = T.alloc_shared((1,), T.int32)
                barrier_sign = T.alloc_shared((1,), T.int32)
                barrier_target = T.alloc_shared((1,), T.int32)

                for entry_index in T.serial(
                    tx, world_size * slots, _THREADS
                ):
                    local_expert_by_entry[entry_index] = (
                        experts[entry_index] - rank * epn
                    )
                for local_expert in T.serial(
                    tx, epn + 1, _THREADS
                ):
                    matched_offsets[local_expert] = 0
                if tx == 0:
                    active_count[0] = 0
                    clear_count[0] = 0
                T.sync_threads()
                # Clear work is partitioned by compacted index, so every CTA
                # must derive the same slot order. Warp 0 builds a deterministic
                # slot-ordered list with one ballot per 32 slots.
                if warp == 0:
                    lanes_lt = (
                        T.uint64(1) << T.cast(lane, T.uint64)
                    ) - T.uint64(1)
                    for chunk in T.serial((slots + 31) // 32):
                        slot = chunk * 32 + lane
                        live = slot < slots and experts[
                            rank * slots + slot
                        ] >= 0
                        live_mask = T.ballot_sync(live)
                        base = clear_count[0]
                        if live:
                            clear_slots[
                                base + T.popcount(live_mask & lanes_lt)
                            ] = slot
                        T.sync_warp()
                        if lane == 0:
                            clear_count[0] = base + T.popcount(live_mask)
                        T.sync_warp()
                T.sync_threads()

                for entry_index in T.serial(
                    tx, world_size * slots, _THREADS
                ):
                    local_expert = local_expert_by_entry[entry_index]
                    if 0 <= local_expert and local_expert < epn:
                        T.atomic_add(
                            matched_offsets[local_expert + 1], 1
                        )
                T.sync_threads()

                # Build ascending expert offsets and active indices in warp 0.
                if warp == 0:
                    running = T.alloc_local((1,), T.int32)
                    active_running = T.alloc_local((1,), T.int32)
                    count = T.alloc_local((1,), T.int32)
                    inclusive = T.alloc_local((1,), T.int32)
                    shuffled = T.alloc_local((1,), T.int32)
                    running[0] = 0
                    active_running[0] = 0
                    lower_lanes = (
                        T.uint64(1) << T.cast(lane, T.uint64)
                    ) - T.uint64(1)
                    for tile in T.serial((epn + 31) // 32):
                        local_expert = tile * 32 + lane
                        count[0] = 0
                        if local_expert < epn:
                            count[0] = matched_offsets[local_expert + 1]
                        live = T.ballot_sync(count[0] > 0)
                        inclusive[0] = count[0]
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
                        if local_expert < epn:
                            matched_offsets[local_expert + 1] = (
                                running[0] + inclusive[0]
                            )
                            if count[0] > 0:
                                active_experts[
                                    active_running[0]
                                    + T.popcount(live & lower_lanes)
                                ] = local_expert
                        running[0] = (
                            running[0] + T.shfl_sync(inclusive[0], 31)
                        )
                        active_running[0] = (
                            active_running[0] + T.popcount(live)
                        )
                    if lane == 0:
                        active_count[0] = active_running[0]
                T.sync_threads()

                # Each warp builds one expert's stable rank/slot list.
                # Independent buckets preserve the accumulation order.
                lanes_lt = (
                    T.uint64(1) << T.cast(lane, T.uint64)
                ) - T.uint64(1)
                for bucket in T.serial(warp, epn, _WARPS):
                    if matched_offsets[bucket + 1] > matched_offsets[bucket]:
                        running_entry = T.alloc_local((1,), T.int32)
                        running_entry[0] = matched_offsets[bucket]
                        for chunk in T.serial(
                            (world_size * slots + 31) // 32
                        ):
                            entry_index = chunk * 32 + lane
                            matching = T.alloc_local((1,), T.int32)
                            matching[0] = 0
                            if entry_index < world_size * slots:
                                matching[0] = T.cast(
                                    local_expert_by_entry[entry_index] == bucket,
                                    T.int32,
                                )
                            mask = T.ballot_sync(matching[0] != 0)
                            if matching[0] != 0:
                                matched_entries[
                                    running_entry[0]
                                    + T.popcount(mask & lanes_lt)
                                ] = entry_index
                            running_entry[0] = (
                                running_entry[0] + T.popcount(mask)
                            )
                T.sync_threads()

                total_batches = (
                    active_count[0] * reduce_batches_per_matrix
                )
                sums = T.alloc_local(
                    (_REDUCE_CHUNK_BATCH * 4,), T.float32
                )
                first_values = T.alloc_local(
                    (_REDUCE_CHUNK_BATCH * 4,), T.float32
                )
                second_values = T.alloc_local(
                    (_REDUCE_CHUNK_BATCH * 4,), T.float32
                )
                third_values = T.alloc_local(
                    (_REDUCE_CHUNK_BATCH * 4,), T.float32
                )
                grad_offset = T.alloc_local((1,), T.int64)
                first_remote_offset = T.alloc_local((1,), T.int64)
                second_remote_offset = T.alloc_local((1,), T.int64)
                third_remote_offset = T.alloc_local((1,), T.int64)
                for work in T.serial(
                    bx * _WARPS + warp, total_batches, num_sms * _WARPS
                ):
                    active_index = work // reduce_batches_per_matrix
                    batch = (
                        work - active_index * reduce_batches_per_matrix
                    )
                    local_expert = active_experts[active_index]
                    batch_floats = T.cast(
                        batch, T.int64
                    ) * (_REDUCE_CHUNK_BATCH * _BULK_FLOATS)
                    grad_offset[0] = (
                        T.cast(local_expert, T.int64) * matrix
                        + batch_floats
                    )
                    for c in T.unroll(_REDUCE_CHUNK_BATCH):
                        sums[c * 4 : c * 4 + 4] = T.reinterpret(
                            fast_bulk_load_512(
                                grads[0],
                                (grad_offset[0] + c * _BULK_FLOATS) * 4,
                                T.bool(False),
                            ),
                            "float32x4",
                        )
                    match_begin = matched_offsets[local_expert]
                    match_count = (
                        matched_offsets[local_expert + 1] - match_begin
                    )
                    group_count = match_count // 3
                    for group_index in T.serial(group_count):
                        first_entry = matched_entries[
                            match_begin + group_index * 3
                        ]
                        second_entry = matched_entries[
                            match_begin + group_index * 3 + 1
                        ]
                        third_entry = matched_entries[
                            match_begin + group_index * 3 + 2
                        ]
                        first_remote_offset[0] = (
                            T.cast(first_entry // slots, T.int64)
                            * rank_stride
                            + T.cast(first_entry % slots, T.int64) * matrix
                            + batch_floats
                        )
                        second_remote_offset[0] = (
                            T.cast(second_entry // slots, T.int64)
                            * rank_stride
                            + T.cast(second_entry % slots, T.int64) * matrix
                            + batch_floats
                        )
                        third_remote_offset[0] = (
                            T.cast(third_entry // slots, T.int64)
                            * rank_stride
                            + T.cast(third_entry % slots, T.int64) * matrix
                            + batch_floats
                        )
                        first_remote = first_entry // slots != rank
                        second_remote = second_entry // slots != rank
                        third_remote = third_entry // slots != rank
                        for c in T.unroll(_REDUCE_CHUNK_BATCH):
                            first_values[
                                c * 4 : c * 4 + 4
                            ] = T.reinterpret(
                                fast_bulk_load_512(
                                    reduce[0],
                                    (
                                        first_remote_offset[0]
                                        + c * _BULK_FLOATS
                                    )
                                    * 4,
                                    first_remote,
                                ),
                                "float32x4",
                            )
                        for c in T.unroll(_REDUCE_CHUNK_BATCH):
                            second_values[
                                c * 4 : c * 4 + 4
                            ] = T.reinterpret(
                                fast_bulk_load_512(
                                    reduce[0],
                                    (
                                        second_remote_offset[0]
                                        + c * _BULK_FLOATS
                                    )
                                    * 4,
                                    second_remote,
                                ),
                                "float32x4",
                            )
                        for c in T.unroll(_REDUCE_CHUNK_BATCH):
                            third_values[
                                c * 4 : c * 4 + 4
                            ] = T.reinterpret(
                                fast_bulk_load_512(
                                    reduce[0],
                                    (
                                        third_remote_offset[0]
                                        + c * _BULK_FLOATS
                                    )
                                    * 4,
                                    third_remote,
                                ),
                                "float32x4",
                            )
                        # Peer loads are prefetched without changing entry
                        # accumulation order.
                        for c in T.unroll(_REDUCE_CHUNK_BATCH):
                            for value in T.unroll(4):
                                sums[c * 4 + value] += first_values[
                                    c * 4 + value
                                ]
                        for c in T.unroll(_REDUCE_CHUNK_BATCH):
                            for value in T.unroll(4):
                                sums[c * 4 + value] += second_values[
                                    c * 4 + value
                                ]
                        for c in T.unroll(_REDUCE_CHUNK_BATCH):
                            for value in T.unroll(4):
                                sums[c * 4 + value] += third_values[
                                    c * 4 + value
                                ]
                    remainder = match_count - group_count * 3
                    if remainder > 0:
                        first_entry = matched_entries[
                            match_begin + group_count * 3
                        ]
                        first_remote_offset[0] = (
                            T.cast(first_entry // slots, T.int64)
                            * rank_stride
                            + T.cast(first_entry % slots, T.int64) * matrix
                            + batch_floats
                        )
                        first_remote = first_entry // slots != rank
                        for c in T.unroll(_REDUCE_CHUNK_BATCH):
                            first_values[
                                c * 4 : c * 4 + 4
                            ] = T.reinterpret(
                                fast_bulk_load_512(
                                    reduce[0],
                                    (
                                        first_remote_offset[0]
                                        + c * _BULK_FLOATS
                                    )
                                    * 4,
                                    first_remote,
                                ),
                                "float32x4",
                            )
                    if remainder > 1:
                        second_entry = matched_entries[
                            match_begin + group_count * 3 + 1
                        ]
                        second_remote_offset[0] = (
                            T.cast(second_entry // slots, T.int64)
                            * rank_stride
                            + T.cast(second_entry % slots, T.int64) * matrix
                            + batch_floats
                        )
                        second_remote = second_entry // slots != rank
                        for c in T.unroll(_REDUCE_CHUNK_BATCH):
                            second_values[
                                c * 4 : c * 4 + 4
                            ] = T.reinterpret(
                                fast_bulk_load_512(
                                    reduce[0],
                                    (
                                        second_remote_offset[0]
                                        + c * _BULK_FLOATS
                                    )
                                    * 4,
                                    second_remote,
                                ),
                                "float32x4",
                            )
                    if remainder > 0:
                        for c in T.unroll(_REDUCE_CHUNK_BATCH):
                            for value in T.unroll(4):
                                sums[c * 4 + value] += first_values[
                                    c * 4 + value
                                ]
                    if remainder > 1:
                        for c in T.unroll(_REDUCE_CHUNK_BATCH):
                            for value in T.unroll(4):
                                sums[c * 4 + value] += second_values[
                                    c * 4 + value
                                ]
                    for c in T.unroll(_REDUCE_CHUNK_BATCH):
                        fast_bulk_store_512(
                            grads[0],
                            (grad_offset[0] + c * _BULK_FLOATS) * 4,
                            T.reinterpret(
                                sums[T.Ramp(c * 4, 1, 4)], "uint32x4"
                            ),
                            T.bool(False),
                        )

                cross_rank_barrier(
                    meta,
                    barrier_state,
                    barrier_phase,
                    barrier_sign,
                    barrier_target,
                    rank,
                    bx,
                    tx,
                )

                zeros = T.alloc_local((4,), T.uint32)
                for value in T.unroll(4):
                    zeros[value] = 0
                total_clear_batches = (
                    clear_count[0] * clear_batches_per_matrix
                )
                clear_offset = T.alloc_local((1,), T.int64)
                for work in T.serial(
                    bx * _WARPS + warp,
                    total_clear_batches,
                    num_sms * _WARPS,
                ):
                    clear_index = work // clear_batches_per_matrix
                    batch = (
                        work - clear_index * clear_batches_per_matrix
                    )
                    slot = clear_slots[clear_index]
                    clear_offset[0] = (
                        T.cast(rank, T.int64) * rank_stride
                        + T.cast(slot, T.int64) * matrix
                        + T.cast(batch, T.int64)
                        * (_CLEAR_CHUNK_BATCH * _BULK_FLOATS)
                    )
                    for c in T.unroll(_CLEAR_CHUNK_BATCH):
                        fast_bulk_store_512(
                            reduce[0],
                            (clear_offset[0] + c * _BULK_FLOATS) * 4,
                            zeros[T.Ramp(0, 1, 4)],
                            T.bool(False),
                        )

        return tilelang.compile(
            kernel,
            execution_backend="tvm_ffi",
            target="ppu",
            compile_flags=["--dopt=3"],
            pass_configs={
                # Shared tables are published once behind T.sync_threads() and
                # then read-only, so per-access sync is unnecessary; the pass
                # also rejects shared reads inside the warp-strided work loop.
                "tl.disable_thread_storage_sync": True,
                "tl.disable_shuffle_elect": True,
            },
        )

    return _compile_cached(
        (
            "grad-reduce-stream",
            world_size,
            local_experts,
            slots,
            H,
            Hp,
            rank_stride,
            num_sms,
            meta_stride,
            barrier_off,
        ),
        build,
    )


def launch_grad_reduce(
    local_expert_grads: torch.Tensor,
    reduce_buffers: torch.Tensor,
    experts_to_copy: torch.Tensor,
    rank: int,
    num_sms: int,
    meta_buf: torch.Tensor,
    meta_stride: int,
    barrier_off: int,
    grid_sync_bar: torch.Tensor,
) -> None:
    """Launch fused gradient reduction and consumed-slot clearing."""
    # Retained for a backend-compatible call signature.
    del grid_sync_bar
    if reduce_buffers.numel() == 0 or experts_to_copy.numel() == 0:
        return

    if (
        local_expert_grads.dtype != torch.float32
        or not local_expert_grads.is_contiguous()
    ):
        raise ValueError(
            "local_expert_grads must be contiguous fp32 [epn,H,H']"
        )
    if reduce_buffers.dtype != torch.float32:
        raise ValueError("reduce_buffers must be fp32 [R,epn,H,H']")
    if (
        experts_to_copy.dtype != torch.int32
        or not experts_to_copy.is_contiguous()
    ):
        raise ValueError("experts_to_copy must be contiguous int32 [R,B]")
    if local_expert_grads.ndim != 3:
        raise ValueError("local_expert_grads must have rank 3")
    if reduce_buffers.ndim != 4:
        raise ValueError("reduce_buffers must have rank 4")
    if experts_to_copy.ndim != 2:
        raise ValueError("experts_to_copy must have rank 2")

    world_size, slots = (int(value) for value in experts_to_copy.shape)
    local_experts, H, Hp = (
        int(value) for value in local_expert_grads.shape
    )
    reduce_world, reduce_slots, reduce_H, reduce_Hp = (
        int(value) for value in reduce_buffers.shape
    )
    if (reduce_world, reduce_slots) != (world_size, slots):
        raise ValueError(
            "reduce_buffers and experts_to_copy disagree on [R,epn]"
        )
    if (reduce_H, reduce_Hp) != (H, Hp):
        raise ValueError(
            "reduce_buffers and local_expert_grads row shapes differ"
        )
    if local_experts != slots:
        raise ValueError(
            "local expert count must equal the plan slot count, got "
            f"{local_experts} and {slots}"
        )
    if not reduce_buffers[0].is_contiguous():
        raise ValueError("each rank's reduce buffer must be contiguous")
    rank_stride = int(reduce_buffers.stride(0))
    rank_elements = reduce_buffers[0].numel()
    if rank_stride < rank_elements:
        raise ValueError("reduce-buffer rank payloads must not overlap")
    rank = int(rank)
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is outside world_size={world_size}")
    if not isinstance(num_sms, int) or num_sms <= 0:
        raise ValueError(f"num_sms must be a positive int, got {num_sms}")
    if H % _DIM_ALIGN or Hp % _DIM_ALIGN:
        raise ValueError(
            "H and H' must be multiples of 128, "
            f"got H={H}, H'={Hp}"
        )
    span_elements = (world_size - 1) * rank_stride + rank_elements
    reduce_flat = reduce_buffers.as_strided(
        (span_elements,),
        (1,),
        storage_offset=reduce_buffers.storage_offset(),
    )
    with torch.cuda.device(local_expert_grads.device):
        kernel = _compile_grad_reduce(
            world_size=world_size,
            local_experts=local_experts,
            slots=slots,
            H=H,
            Hp=Hp,
            rank_stride=rank_stride,
            num_sms=num_sms,
            meta_stride=int(meta_stride),
            barrier_off=int(barrier_off),
        )
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            kernel(
                local_expert_grads.reshape(-1),
                reduce_flat,
                experts_to_copy.reshape(-1),
                meta_buf,
                rank,
            )
