"""All-SM TileLang PPU dispatch data plane."""

from __future__ import annotations

import threading

import torch

from .planning import MoonEPCommPlan
from .sync import launch_inter_rank_sync


_THREADS = 256
_WARPS = _THREADS // 32
_BULK_BYTES = 512
_BULK_HIDDEN = _BULK_BYTES // 2
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


def _compile_init_dedup(*, keys: int, K: int):
    entries = keys * K
    blocks = min(1024, (entries + _THREADS - 1) // _THREADS)
    grid_threads = blocks * _THREADS
    rounds = (entries + grid_threads - 1) // grid_threads

    def build():
        import tilelang
        import tilelang.ppu.language as T

        @T.prim_func
        def kernel(
            kidx_to_loff_ptr: T.handle,
            dup_counts_ptr: T.handle,
        ):
            kidx_to_loff = T.match_buffer(
                kidx_to_loff_ptr, (entries,), dtype=T.int32
            )
            dup_counts = T.match_buffer(
                dup_counts_ptr, (2,), dtype=T.int32
            )
            with T.Kernel(blocks, threads=_THREADS) as bx:
                tx = T.get_thread_binding()
                tid = bx * _THREADS + tx
                # Plain stores: the 512-B bulk store is write-through, which
                # would evict this table out from under scan and emit.
                #
                # Keep the grid-stride rounds compile-time and guard the
                # final round explicitly.  A dynamic-start T.serial loop is
                # unsafe here: the PPU safe-memory legalization can clamp
                # inactive thread indices and make several threads write the
                # same element for small tensors.
                for round_idx in T.serial(rounds):
                    entry = tid + round_idx * grid_threads
                    if entry < entries:
                        kidx_to_loff[entry] = -1
                if tid < 2:
                    dup_counts[tid] = 0

        return tilelang.compile(
            kernel,
            execution_backend="tvm_ffi",
            target="ppu",
            compile_flags=["--dopt=3"],
        )

    return _compile_cached(("dispatch-init-dedup-direct", keys, K), build)


def _compile_scan_dedup(
    *, R: int, S: int, K: int, NvS: int, meta_elements: int
):
    keys = R * S
    blocks = min(1024, (NvS + _THREADS - 1) // _THREADS)
    grid_threads = blocks * _THREADS
    rounds = (NvS + grid_threads - 1) // grid_threads

    def build():
        import tilelang
        import tilelang.ppu.language as T

        @T.prim_func
        def kernel(
            meta_ptr: T.handle,
            kidx_to_loff_ptr: T.handle,
            src_info_off: T.int32,
        ):
            meta = T.match_buffer(
                meta_ptr, (meta_elements,), dtype=T.int32
            )
            kidx_to_loff = T.match_buffer(
                kidx_to_loff_ptr, (keys * K,), dtype=T.int32
            )
            with T.Kernel(blocks, threads=_THREADS) as bx:
                tx = T.get_thread_binding()
                tid = bx * _THREADS + tx
                for round_idx in T.serial(rounds):
                    loff = tid + round_idx * grid_threads
                    if loff < NvS:
                        # Planning produced this local metadata earlier on
                        # the same stream. It is neither a peer read nor a
                        # polling load, so a normal buffer load is sufficient.
                        info = meta[src_info_off + loff]
                        if info >= 0:
                            src_rank = info // NvS
                            offv = info - src_rank * NvS
                            token = offv // K
                            kidx = offv - token * K
                            key = src_rank * S + token
                            # Each source route has exactly one receiver-local
                            # offset, so this address has exactly one writer.
                            kidx_to_loff[key * K + kidx] = loff

        return tilelang.compile(
            kernel,
            execution_backend="tvm_ffi",
            target="ppu",
            compile_flags=["--dopt=3"],
        )

    return _compile_cached(
        ("dispatch-scan-dedup-direct", R, S, K, NvS, meta_elements), build
    )


def _compile_weight_slot_map(*, R: int, S: int, K: int, NvS: int):
    """Invert the weight destination map: slot -> route, plus a per-window count.

    The caller pre-clears the slot region to -1 and the count region to 0, so
    this only scatters.  Exactly one route writes each slot, hence a plain store.
    """
    routes = S * K
    slots = R * NvS
    windows = (slots + 31) // 32
    blocks = min(1024, (routes + _THREADS - 1) // _THREADS)
    grid_threads = blocks * _THREADS
    rounds = (routes + grid_threads - 1) // grid_threads

    def build():
        import tilelang
        import tilelang.ppu.language as T

        @T.prim_func
        def kernel(
            dst_ptr: T.handle,
            slot_route_ptr: T.handle,
        ):
            dst = T.match_buffer(dst_ptr, (routes,), dtype=T.int32)
            # Slot map followed by the per-window owned-slot counts.
            slot_route = T.match_buffer(
                slot_route_ptr, (slots + windows,), dtype=T.int32
            )
            with T.Kernel(blocks, threads=_THREADS) as bx:
                tx = T.get_thread_binding()
                tid = bx * _THREADS + tx
                for round_idx in T.serial(rounds):
                    route = tid + round_idx * grid_threads
                    if route < routes:
                        dst_value = dst[route]
                        # Duplicate routes still own a weight slot.
                        raw = dst_value ^ (dst_value >> 31)
                        T.assume(0 <= raw < R * NvS)
                        slot_route[raw] = route
                        T.atomic_add(
                            slot_route[slots + (raw >> 5)],
                            1,
                            memory_order="relaxed",
                        )

        return tilelang.compile(
            kernel,
            execution_backend="tvm_ffi",
            target="ppu",
            compile_flags=["--dopt=3"],
        )

    return _compile_cached(
        ("dispatch-weight-slot-map", R, S, K, NvS), build
    )


def _compile_weight_scatter(
    *, R: int, S: int, K: int, NvS: int, meta_stride: int
):
    """Publish route weights, one 128-B warp store per fully-owned window."""
    routes = S * K
    slots = R * NvS
    windows = (slots + 31) // 32
    blocks = min(4096, max(1, (windows + _WARPS - 1) // _WARPS))
    total_warps = blocks * _WARPS
    rounds = (windows + total_warps - 1) // total_warps

    def build():
        import tilelang
        import tilelang.ppu.language as T
        from .primitives import fast_bulk_store_128

        @T.prim_func
        def kernel(
            slot_route_ptr: T.handle,
            weights_ptr: T.handle,
            meta_ptr: T.handle,
            rank: T.int32,
            weights_off: T.int32,
        ):
            slot_route = T.match_buffer(
                slot_route_ptr, (slots + windows,), dtype=T.int32
            )
            weights = T.match_buffer(weights_ptr, (routes,), dtype=T.int32)
            meta = T.match_buffer(
                meta_ptr, (R * meta_stride,), dtype=T.int32
            )
            with T.Kernel(blocks, threads=_THREADS) as bx:
                tx = T.get_thread_binding()
                warp = tx // 32
                lane = tx % 32
                for round_idx in T.serial(rounds):
                    window = bx * _WARPS + warp + round_idx * total_warps
                    if window < windows:
                        first = window * 32
                        slot = first + lane
                        owner = T.alloc_local((1,), T.int32)
                        owner[0] = -1
                        if slot < slots:
                            owner[0] = slot_route[slot]
                        value = T.alloc_local((1,), T.int32)
                        value[0] = 0
                        if owner[0] >= 0:
                            value[0] = weights[owner[0]]
                        # Every lane reads the same count, so this needs no
                        # cross-lane communication.
                        full = slot_route[slots + window] == 32
                        first_peer = first // NvS
                        # A 128-B store writes all 32 lanes, so the window must
                        # be owned outright and must not straddle two peers.
                        same_peer = (first + 31) // NvS == first_peer
                        if full and same_peer:
                            fast_bulk_store_128(
                                meta[0],
                                (
                                    weights_off
                                    + first_peer * meta_stride
                                    + (first - first_peer * NvS)
                                )
                                * 4,
                                value[0],
                                first_peer != rank,
                            )
                        else:
                            if owner[0] >= 0:
                                slot_peer = slot // NvS
                                T.atomic_store(
                                    meta[
                                        weights_off
                                        + slot_peer * meta_stride
                                        + (slot - slot_peer * NvS)
                                    ],
                                    value[0],
                                    memory_order="relaxed",
                                    scope="system",
                                )

        return tilelang.compile(
            kernel,
            execution_backend="tvm_ffi",
            target="ppu",
            compile_flags=["--dopt=3"],
        )

    return _compile_cached(
        ("dispatch-weight-scatter", R, S, K, NvS, meta_stride), build
    )


def _compile_dispatch_rows(
    *,
    R: int,
    S: int,
    H: int,
    K: int,
    NvS: int,
    NvS_padded: int,
    zero_groups: int,
    meta_stride: int,
    with_weights: bool,
    build_dedup_map: bool,
):
    """Scatter rows, emit duplicate lists, and clear padding in one grid."""
    routes = S * K
    payload_blocks = (routes + _WARPS - 1) // _WARPS
    keys = R * S
    emit_blocks = min(1024, (keys + _THREADS - 1) // _THREADS)
    emit_grid_threads = emit_blocks * _THREADS
    emit_rounds = (keys + emit_grid_threads - 1) // emit_grid_threads
    scan_steps = _THREADS.bit_length() - 1
    assert 1 << scan_steps == _THREADS, "block scan assumes a power-of-two CTA"
    # Init/scan finish before this grid. Emission and payload CTAs have disjoint
    # outputs, so interleave them to overlap local list work with peer transfers.
    payloads_per_emit = (payload_blocks + emit_blocks - 1) // emit_blocks
    work_blocks = (
        emit_blocks * (payloads_per_emit + 1)
        if build_dedup_map
        else payload_blocks
    )
    blocks = work_blocks + zero_groups
    full_chunks = H // _BULK_HIDDEN
    tail_hidden = H - full_chunks * _BULK_HIDDEN
    tail_256_chunks = (tail_hidden * 2) // 256
    tail_after_256 = tail_hidden - tail_256_chunks * (256 // 2)
    tail_128_chunks = (tail_after_256 * 2) // 128
    scalar_hidden = (
        full_chunks * _BULK_HIDDEN
        + tail_256_chunks * (256 // 2)
        + tail_128_chunks * (128 // 2)
    )
    scalar_pairs = (H - scalar_hidden) // 2

    def build():
        import tilelang
        import tilelang.ppu.language as T
        from .primitives import (
            fast_bulk_load_256,
            fast_bulk_load_512,
            fast_bulk_store_128,
            fast_bulk_store_256,
            fast_bulk_store_512,
        )

        @T.macro
        def emit_tables(
            kidx_to_loff, dup_groups, dup_loffs, dup_counts, emit_block, tx
        ):
            tid = emit_block * _THREADS + tx
            group_scan = T.alloc_shared(
                (_THREADS,), T.int32, scope="shared"
            )
            extra_scan = T.alloc_shared(
                (_THREADS,), T.int32, scope="shared"
            )
            group_base = T.alloc_shared((1,), T.int32, scope="shared")
            extra_base = T.alloc_shared((1,), T.int32, scope="shared")
            for round_idx in T.serial(emit_rounds):
                key = tid + round_idx * emit_grid_threads
                # Hold the row in registers; the second pass would
                # otherwise re-read addresses K apart per thread.
                loffs = T.alloc_local((K,), T.int32)
                count = T.alloc_local((1,), T.int32)
                primary_kidx = T.alloc_local((1,), T.int32)
                primary_loff = T.alloc_local((1,), T.int32)
                count[0] = 0
                primary_kidx[0] = -1
                primary_loff[0] = -1
                for kidx in T.unroll(K):
                    loffs[kidx] = -1
                if key < keys:
                    for kidx in T.unroll(K):
                        loffs[kidx] = kidx_to_loff[key * K + kidx]
                        if loffs[kidx] >= 0:
                            if primary_kidx[0] < 0:
                                primary_kidx[0] = kidx
                                primary_loff[0] = loffs[kidx]
                            count[0] += 1

                group = T.alloc_local((1,), T.int32)
                begin = T.alloc_local((1,), T.int32)
                group_scan[tx] = T.if_then_else(
                    count[0] > 1, 1, 0
                )
                extra_scan[tx] = T.if_then_else(
                    count[0] > 1, count[0] - 1, 0
                )
                T.sync_threads()
                # Inclusive Hillis-Steele scan; the offsets are
                # Python-level so each step is a compile-time shift.
                for shift in range(scan_steps):
                    offset = 1 << shift
                    carry_group = T.alloc_local((1,), T.int32)
                    carry_extra = T.alloc_local((1,), T.int32)
                    carry_group[0] = 0
                    carry_extra[0] = 0
                    if tx >= offset:
                        carry_group[0] = group_scan[tx - offset]
                        carry_extra[0] = extra_scan[tx - offset]
                    T.sync_threads()
                    if tx >= offset:
                        group_scan[tx] = (
                            group_scan[tx] + carry_group[0]
                        )
                        extra_scan[tx] = (
                            extra_scan[tx] + carry_extra[0]
                        )
                    T.sync_threads()
                if tx == _THREADS - 1:
                    group_base[0] = T.atomic_add(
                        dup_counts[0],
                        group_scan[_THREADS - 1],
                        return_prev=True,
                    )
                    extra_base[0] = T.atomic_add(
                        dup_counts[1],
                        extra_scan[_THREADS - 1],
                        return_prev=True,
                    )
                T.sync_threads()
                # Exclusive prefix: inclusive minus our own count.
                group[0] = group_base[0] + group_scan[tx] - 1
                begin[0] = (
                    extra_base[0] + extra_scan[tx] - (count[0] - 1)
                )

                if key < keys:
                    if count[0] > 1:
                        dup_groups[group[0] * 3] = primary_loff[0]
                        dup_groups[group[0] * 3 + 1] = begin[0]
                        dup_groups[group[0] * 3 + 2] = count[0] - 1
                        write = T.alloc_local((1,), T.int32)
                        write[0] = begin[0]
                        for kidx in T.unroll(K):
                            if kidx != primary_kidx[0]:
                                if loffs[kidx] >= 0:
                                    dup_loffs[write[0]] = loffs[kidx]
                                    write[0] += 1

        @T.prim_func
        def kernel(
            input_ptr: T.handle,
            hidden_ptr: T.handle,
            dst_ptr: T.handle,
            rank: T.int32,
            zero_ptr: T.handle,
            meta_ptr: T.handle,
            weights_off: T.int32,
            kidx_ptr: T.handle,
            groups_ptr: T.handle,
            loffs_ptr: T.handle,
            counts_ptr: T.handle,
        ):
            input = T.match_buffer(
                input_ptr, (S * H,), dtype=T.bfloat16
            )
            hidden = T.match_buffer(
                hidden_ptr, (R * NvS_padded * H,), dtype=T.bfloat16
            )
            dst = T.match_buffer(dst_ptr, (routes,), dtype=T.int32)
            zero_ranges = T.match_buffer(
                zero_ptr, (zero_groups * 2,), dtype=T.int32
            )
            meta = T.match_buffer(meta_ptr, (R * meta_stride,), dtype=T.int32)
            kidx_to_loff = T.match_buffer(kidx_ptr, (keys * K,), dtype=T.int32)
            dup_groups = T.match_buffer(groups_ptr, (NvS * 3,), dtype=T.int32)
            dup_loffs = T.match_buffer(loffs_ptr, (NvS,), dtype=T.int32)
            dup_counts = T.match_buffer(counts_ptr, (2,), dtype=T.int32)
            with T.Kernel(blocks, threads=_THREADS) as bx:
                tx = T.get_thread_binding()
                warp = tx // 32
                lane = tx % 32
                payload_block = T.alloc_local((1,), T.int32)
                payload_block[0] = bx
                is_payload = T.alloc_local((1,), T.int32)
                is_payload[0] = 1
                if build_dedup_map:
                    emit_block = bx // (payloads_per_emit + 1)
                    item = bx % (payloads_per_emit + 1)
                    payload_block[0] = emit_block * payloads_per_emit + item - 1
                    is_payload[0] = T.cast(item != 0, T.int32)
                    if bx < work_blocks and item == 0:
                        emit_tables(
                            kidx_to_loff,
                            dup_groups,
                            dup_loffs,
                            dup_counts,
                            emit_block,
                            tx,
                        )
                route = payload_block[0] * _WARPS + warp
                if bx < work_blocks and is_payload[0] != 0 and route < routes:
                    dst_value = dst[route]
                    # Duplicate routes encode raw destination r as -r - 1.
                    # This branchless decode avoids duplicating a Select in
                    # pointer expressions after TIR inlining.
                    raw = dst_value ^ (dst_value >> 31)
                    T.assume(0 <= raw < R * NvS)
                    peer = raw // NvS
                    T.assume(0 <= peer < R)
                    if dst_value >= 0:
                        source_base = T.cast(route // K, T.int64) * H
                        destination_base = (
                            T.cast(raw, T.int64)
                            + T.cast(peer, T.int64)
                            * (NvS_padded - NvS)
                        ) * H
                        for chunk in T.serial(full_chunks):
                            source = source_base + chunk * _BULK_HIDDEN
                            destination = (
                                destination_base + chunk * _BULK_HIDDEN
                            )
                            packed = fast_bulk_load_512(
                                input[0], source * 2, T.bool(False)
                            )
                            fast_bulk_store_512(
                                hidden[0],
                                destination * 2,
                                packed,
                                peer != rank,
                            )
                        for tail_chunk in T.serial(tail_256_chunks):
                            source = (
                                source_base
                                + full_chunks * _BULK_HIDDEN
                                + tail_chunk * (256 // 2)
                            )
                            destination = (
                                destination_base
                                + full_chunks * _BULK_HIDDEN
                                + tail_chunk * (256 // 2)
                            )
                            value = fast_bulk_load_256(
                                input[0], source * 2, T.bool(False)
                            )
                            fast_bulk_store_256(
                                hidden[0],
                                destination * 2,
                                value,
                                peer != rank,
                            )
                        for tail_chunk in T.serial(tail_128_chunks):
                            source = (
                                source_base
                                + full_chunks * _BULK_HIDDEN
                                + tail_256_chunks * (256 // 2)
                                + tail_chunk * (128 // 2)
                            )
                            destination = (
                                destination_base
                                + full_chunks * _BULK_HIDDEN
                                + tail_256_chunks * (256 // 2)
                                + tail_chunk * (128 // 2)
                            )
                            value = T.ldg32(
                                input[source + lane * 2 : source + lane * 2 + 2]
                            )
                            fast_bulk_store_128(
                                hidden[0],
                                destination * 2,
                                value,
                                peer != rank,
                            )
                        if scalar_pairs > 0:
                            if lane < scalar_pairs:
                                source = (
                                    source_base + scalar_hidden + lane * 2
                                )
                                destination = (
                                    destination_base
                                    + scalar_hidden
                                    + lane * 2
                                )
                                value = T.ldg32(
                                    input[source : source + 2]
                                )
                                destination_word = T.decl_buffer(
                                    (R * NvS_padded * H // 2,),
                                    dtype=T.uint32, data=hidden.data,
                                )
                                T.atomic_store(
                                    destination_word[destination // 2],
                                    value,
                                    memory_order="relaxed",
                                    scope="system",
                                )

                # Padding ranges are local and disjoint from routed rows.
                if bx >= work_blocks:
                    zero_group = bx - work_blocks
                    begin = zero_ranges[zero_group * 2]
                    count = zero_ranges[zero_group * 2 + 1]
                    hidden_base = (
                        T.cast(rank, T.int64) * NvS_padded
                        + T.cast(begin, T.int64)
                    ) * H
                    total_hidden = count * H
                    bulk_chunks = total_hidden // _BULK_HIDDEN
                    zeros = T.alloc_local((4,), T.uint32)
                    for word in T.unroll(4):
                        zeros[word] = T.uint32(0)
                    for chunk in T.serial(warp, bulk_chunks, _WARPS):
                        offset = hidden_base + chunk * _BULK_HIDDEN
                        fast_bulk_store_512(
                            hidden[0],
                            offset * 2,
                            zeros[T.Ramp(0, 1, 4)],
                            T.bool(False),
                        )
                    bulk_hidden = bulk_chunks * _BULK_HIDDEN
                    for index in T.serial(
                        tx, total_hidden - bulk_hidden, _THREADS
                    ):
                        hidden[hidden_base + bulk_hidden + index] = (
                            T.bfloat16(0)
                        )
                    if with_weights:
                        for row in T.serial(tx, count, _THREADS):
                            meta[
                                rank * meta_stride + weights_off + begin + row
                            ] = 0

        return tilelang.compile(
            kernel,
            execution_backend="tvm_ffi",
            target="ppu",
            compile_flags=["--dopt=3"],
        )

    return _compile_cached(
        (
            "dispatch-rows",
            R,
            S,
            H,
            K,
            NvS,
            NvS_padded,
            zero_groups,
            meta_stride,
            with_weights,
            build_dedup_map,
        ),
        build,
    )


def _launch_dispatch(
    ctx: dict,
    hidden_sh: torch.Tensor,
    route_weights_bits: torch.Tensor,
    plan: MoonEPCommPlan,
    *,
    with_weights: bool,
    build_dedup_map: bool,
) -> None:
    """Scatter a plan and publish its peer writes on the current stream."""

    R = int(ctx["R"])
    S = int(ctx["S"])
    H = int(ctx["H"])
    K = int(ctx["K"])
    NvS = int(ctx["NvS"])
    NvS_padded = int(ctx["NvS_padded"])
    hidden_full = ctx["hidden_buf"]
    meta_full = ctx["meta_buf"]
    meta_local = ctx["meta_buf_local"]

    with torch.cuda.device(hidden_sh.device):
        if build_dedup_map:
            init_dedup_kernel = _compile_init_dedup(keys=R * S, K=K)
            scan_dedup_kernel = _compile_scan_dedup(
                R=R,
                S=S,
                K=K,
                NvS=NvS,
                meta_elements=int(meta_local.numel()),
            )
        if with_weights:
            slot_map_kernel = _compile_weight_slot_map(
                R=R, S=S, K=K, NvS=NvS
            )
            weight_scatter_kernel = _compile_weight_scatter(
                R=R,
                S=S,
                K=K,
                NvS=NvS,
                meta_stride=int(ctx["meta_chunk_padded"]),
            )
        dispatch_kernel = _compile_dispatch_rows(
            R=R,
            S=S,
            H=H,
            K=K,
            NvS=NvS,
            NvS_padded=NvS_padded,
            zero_groups=2 * (int(ctx["E"]) // R),
            meta_stride=int(ctx["meta_chunk_padded"]),
            with_weights=with_weights,
            build_dedup_map=build_dedup_map,
        )
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            if build_dedup_map:
                init_dedup_kernel(
                    ctx["kidx_to_loff"],
                    plan.dup_counts,
                )
                scan_dedup_kernel(
                    meta_local,
                    ctx["kidx_to_loff"],
                    int(ctx["SRC_INFO_OFF"]),
                )
            dispatch_kernel(
                hidden_sh.reshape(-1),
                hidden_full.reshape(-1),
                plan.dst,
                int(ctx["rank"]),
                plan.zero_fill_ranges.reshape(-1),
                meta_full,
                int(ctx["WEIGHTS_OFF"]),
                ctx["kidx_to_loff"],
                plan.dup_groups.reshape(-1),
                plan.dup_loffs,
                plan.dup_counts,
            )
            if with_weights:
                slot_route = ctx["weight_slot_route"]
                slots = R * NvS
                # Stale slots must not be republished, and the window counts
                # must start at zero.
                slot_route[:slots].fill_(-1)
                slot_route[slots:].zero_()
                slot_map_kernel(plan.dst, slot_route)
                weight_scatter_kernel(
                    slot_route,
                    route_weights_bits.reshape(-1).view(torch.int32),
                    meta_full,
                    int(ctx["rank"]),
                    int(ctx["WEIGHTS_OFF"]),
                )
        launch_inter_rank_sync(ctx)


def _validate_dispatch_plan(
    ctx: dict, plan: MoonEPCommPlan, device: torch.device
) -> None:
    S = int(ctx["S"])
    K = int(ctx["K"])
    R = int(ctx["R"])
    E = int(ctx["E"])
    epn = E // R
    NvS = int(ctx["NvS"])
    assert isinstance(plan, MoonEPCommPlan)
    assert plan.N == S * K and plan.R == R and plan.K == K
    assert plan.NvS == NvS

    def check(tensor, name, shape):
        assert tensor.dtype == torch.int32 and tensor.is_contiguous(), (
            f"{name} must be contiguous int32"
        )
        assert tuple(tensor.shape) == shape, (
            f"{name} must be shape {shape}, got {tuple(tensor.shape)}"
        )
        assert tensor.device == device, f"{name} must be on {device}"

    check(plan.dst, "dst", (S * K,))
    check(plan.zero_fill_ranges, "zero_fill_ranges", (2 * epn, 2))
    check(plan.dup_groups, "dup_groups", (NvS, 3))
    check(plan.dup_loffs, "dup_loffs", (NvS,))
    check(plan.dup_counts, "dup_counts", (2,))


def launch_dispatch(
    ctx: dict,
    hidden_sh: torch.Tensor,
    route_weights_sk: torch.Tensor | None,
    plan: MoonEPCommPlan,
    *,
    build_dedup_map: bool = True,
    pdl_trigger: bool = False,
) -> None:
    """Launch the all-SM TileLang dispatch kernels."""
    del pdl_trigger
    S = int(ctx["S"])
    H = int(ctx["H"])
    K = int(ctx["K"])
    assert hidden_sh.dtype == torch.bfloat16 and hidden_sh.is_contiguous(), (
        "hidden_sh must be contiguous bf16"
    )
    assert hidden_sh.is_cuda, "hidden_sh must be an accelerator tensor"
    assert tuple(hidden_sh.shape) == (S, H), (
        f"hidden_sh must be shape [S={S}, H={H}], got {tuple(hidden_sh.shape)}"
    )
    _validate_dispatch_plan(ctx, plan, hidden_sh.device)
    with_weights = route_weights_sk is not None
    if with_weights:
        assert (
            route_weights_sk.dtype == torch.float32
            and route_weights_sk.is_contiguous()
        ), "route_weights_sk must be contiguous fp32"
        assert tuple(route_weights_sk.shape) == (S, K), (
            f"route_weights_sk must be shape [S={S}, K={K}], "
            f"got {tuple(route_weights_sk.shape)}"
        )
        assert route_weights_sk.device == hidden_sh.device
    if route_weights_sk is None:
        route_weights_sk = plan.dst
    _launch_dispatch(
        ctx,
        hidden_sh,
        route_weights_sk,
        plan,
        with_weights=with_weights,
        build_dedup_map=build_dedup_map,
    )
