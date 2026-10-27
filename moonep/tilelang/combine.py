"""All-SM TileLang PPU combine data plane."""

from __future__ import annotations

import threading

import torch

from .sync import launch_inter_rank_sync


_THREADS = 256
_WARPS = _THREADS // 32
_BULK_HIDDEN = 512 // 2
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


def _compile_combine(
    *,
    R: int,
    S: int,
    H: int,
    K: int,
    NvS: int,
    NvS_padded: int,
    meta_stride: int,
    with_weights: bool,
):
    # Match the native kernel's one-CTA-per-token launch.  Serializing multiple
    # tokens in each CTA removes the occupancy that hides remote-load latency.
    grid_blocks = S
    # Bulk primitives require word alignment, not transaction-width alignment.
    # Even H keeps every bf16 row word-aligned; only copy complete chunks.
    full_chunks = H // _BULK_HIDDEN
    bulk_hidden = full_chunks * _BULK_HIDDEN
    tail_hidden = H - bulk_hidden
    tail_256_chunks = (tail_hidden * 2) // 256
    tail_after_256 = tail_hidden - tail_256_chunks * (256 // 2)
    tail_128_chunks = (tail_after_256 * 2) // 128
    scalar_hidden = (
        bulk_hidden
        + tail_256_chunks * (256 // 2)
        + tail_128_chunks * (128 // 2)
    )
    scalar_pairs = (H - scalar_hidden) // 2
    # The route-weight gather is a remote scalar load that nothing downstream
    # consumes, so a reserved warp issues it after the descriptor barrier
    # instead of every thread waiting on it.  One warp fans it across its lanes,
    # hence the K bound.
    gather_warp = with_weights
    # Holding a warp out of the accumulation keeps the per-warp chunk count
    # uniform.
    acc_warps = _WARPS - 1
    paired_chunk_rounds = (
        full_chunks + 2 * acc_warps - 1
    ) // (2 * acc_warps)
    # Pair independent remote route loads to expose memory-level parallelism.
    all_second_chunks_valid = (
        paired_chunk_rounds == 1 and full_chunks == 2 * acc_warps
    )
    route_pair_batches = (K + 1) // 2
    tail_256_rounds = (
        tail_256_chunks + acc_warps - 1
    ) // acc_warps
    tail_128_rounds = (
        tail_128_chunks + acc_warps - 1
    ) // acc_warps

    def build():
        import tilelang
        import tilelang.ppu.language as T
        from .primitives import (
            fast_bulk_load_128,
            fast_bulk_load_256,
            fast_bulk_load_512,
            fast_bulk_store_512,
        )

        @T.macro
        def accumulate_tail(
            hidden_full,
            route_addresses,
            route_peers,
            route_primary,
            rank,
            hidden_offset,
            sums,
            values_per_lane,
        ):
            for value in T.serial(values_per_lane):
                sums[value] = T.float32(0)
            for k in T.serial(K):
                if route_primary[k] != 0:
                    values = T.alloc_local(
                        (values_per_lane,), T.bfloat16
                    )
                    address = route_addresses[k] + hidden_offset
                    if values_per_lane == 4:
                        values[0:4] = T.reinterpret(
                            fast_bulk_load_256(
                                hidden_full[0],
                                address * 2,
                                route_peers[k] != rank,
                            ),
                            "bfloat16x4",
                        )
                    else:
                        values[0:2] = T.reinterpret(
                            fast_bulk_load_128(
                                hidden_full[0],
                                address * 2,
                                route_peers[k] != rank,
                            ),
                            "bfloat16x2",
                        )
                    for value in T.serial(values_per_lane):
                        sums[value] += T.cast(
                            values[value], T.float32
                        )

        @T.macro
        def store_tail(
            output, token, hidden_offset, lane, sums, values_per_lane
        ):
            for value in T.unroll(values_per_lane):
                output[
                    token * H
                    + hidden_offset
                    + (value // 2) * (128 // 2)
                    + lane * 2
                    + value % 2
                ] = T.cast(sums[value], T.bfloat16)

        @T.prim_func
        def kernel(
            output_ptr: T.handle,
            output_weights_ptr: T.handle,
            hidden_full_ptr: T.handle,
            meta_ptr: T.handle,
            dst_ptr: T.handle,
            rank: T.int32,
            weights_off: T.int32,
        ):
            output = T.match_buffer(
                output_ptr, (S * H,), dtype=T.bfloat16
            )
            output_weights = T.match_buffer(
                output_weights_ptr, (S * K,), dtype=T.int32
            )
            hidden_full = T.match_buffer(
                hidden_full_ptr,
                (R * NvS_padded * H,),
                dtype=T.bfloat16,
            )
            meta = T.match_buffer(
                meta_ptr, (R * meta_stride,), dtype=T.int32
            )
            dst = T.match_buffer(dst_ptr, (S * K,), dtype=T.int32)
            with T.Kernel(
                grid_blocks,
                threads=_THREADS,
            ) as bx:
                tx = T.get_thread_binding()
                warp = tx // 32
                lane = tx % 32
                route_addresses = T.alloc_shared(
                    (K,), T.int64, scope="shared"
                )
                route_peers = T.alloc_shared(
                    (K,), T.int32, scope="shared"
                )
                route_primary = T.alloc_shared(
                    (K,), T.int32, scope="shared"
                )
                if gather_warp:
                    route_loffs = T.alloc_shared(
                        (K,), T.int32, scope="shared"
                    )
                # Each CTA owns exactly one token.
                token = bx
                if tx < K:
                    k = tx
                    destination = dst[token * K + k]
                    raw = T.if_then_else(
                        destination < 0,
                        -destination - 1,
                        destination,
                    )
                    peer = raw // NvS
                    loff = raw - peer * NvS
                    route_addresses[k] = (
                        T.cast(peer, T.int64) * NvS_padded * H
                        + T.cast(loff, T.int64) * H
                    )
                    route_peers[k] = peer
                    route_primary[k] = destination >= 0
                    if gather_warp:
                        # Keep remote traffic off the barrier; the gather
                        # warp reads meta[] once it is released.
                        route_loffs[k] = loff
                T.sync_threads()

                acc_active = warp < acc_warps
                if gather_warp:
                    if warp == acc_warps:
                        if lane < K:
                            output_weights[token * K + lane] = meta[
                                route_peers[lane] * meta_stride
                                + weights_off
                                + route_loffs[lane]
                            ]
                if acc_active:
                    for round_idx in T.serial(paired_chunk_rounds):
                        chunk0 = warp + round_idx * 2 * acc_warps
                        chunk0_valid = chunk0 < full_chunks
                        chunk0_safe = T.if_then_else(
                            chunk0_valid, chunk0, 0
                        )
                        chunk1 = chunk0 + acc_warps
                        sums0 = T.alloc_local((8,), T.float32)
                        sums1 = T.alloc_local((8,), T.float32)
                        for value in T.serial(8):
                            sums0[value] = T.float32(0)
                            sums1[value] = T.float32(0)
                        for pair in T.unroll(route_pair_batches):
                            first_k = pair * 2
                            second_k = first_k + 1
                            second_valid = second_k < K
                            second_safe = T.if_then_else(
                                second_valid, second_k, 0
                            )
                            first_values = T.alloc_local(
                                (8,), T.bfloat16
                            )
                            second_values = T.alloc_local(
                                (8,), T.bfloat16
                            )

                            if route_primary[first_k] != 0:
                                first_address = (
                                    route_addresses[first_k]
                                    + chunk0_safe * _BULK_HIDDEN
                                )
                                first_values[0:8] = T.reinterpret(
                                    fast_bulk_load_512(
                                        hidden_full[0],
                                        first_address * 2,
                                        route_peers[first_k] != rank,
                                    ),
                                    "bfloat16x8",
                                )
                            if (
                                second_valid
                                and route_primary[second_safe] != 0
                            ):
                                second_address = (
                                    route_addresses[second_safe]
                                    + chunk0_safe * _BULK_HIDDEN
                                )
                                second_values[0:8] = T.reinterpret(
                                    fast_bulk_load_512(
                                        hidden_full[0],
                                        second_address * 2,
                                        route_peers[second_safe] != rank,
                                    ),
                                    "bfloat16x8",
                                )
                            for value in T.unroll(8):
                                if route_primary[first_k] != 0:
                                    sums0[value] += T.cast(
                                        first_values[value], T.float32
                                    )
                                if (
                                    second_valid
                                    and route_primary[second_safe] != 0
                                ):
                                    sums0[value] += T.cast(
                                        second_values[value], T.float32
                                    )

                            if (
                                all_second_chunks_valid
                                or chunk1 < full_chunks
                            ):
                                if route_primary[first_k] != 0:
                                    first_address = (
                                        route_addresses[first_k]
                                        + chunk1 * _BULK_HIDDEN
                                    )
                                    first_values[0:8] = T.reinterpret(
                                        fast_bulk_load_512(
                                            hidden_full[0],
                                            first_address * 2,
                                            route_peers[first_k] != rank,
                                        ),
                                        "bfloat16x8",
                                    )
                                if (
                                    second_valid
                                    and route_primary[second_safe] != 0
                                ):
                                    second_address = (
                                        route_addresses[second_safe]
                                        + chunk1 * _BULK_HIDDEN
                                    )
                                    second_values[0:8] = T.reinterpret(
                                        fast_bulk_load_512(
                                            hidden_full[0],
                                            second_address * 2,
                                            route_peers[second_safe] != rank,
                                        ),
                                        "bfloat16x8",
                                    )
                                for value in T.unroll(8):
                                    if route_primary[first_k] != 0:
                                        sums1[value] += T.cast(
                                            first_values[value], T.float32
                                        )
                                    if (
                                        second_valid
                                        and route_primary[second_safe] != 0
                                    ):
                                        sums1[value] += T.cast(
                                            second_values[value], T.float32
                                        )
                        result0 = T.alloc_local((4,), T.uint32)
                        result1 = T.alloc_local((4,), T.uint32)
                        for word in T.unroll(4):
                            low0 = T.cast(sums0[word * 2], T.bfloat16)
                            high0 = T.cast(
                                sums0[word * 2 + 1], T.bfloat16
                            )
                            result0[word] = T.cast(
                                T.reinterpret(low0, T.uint16), T.uint32
                            ) | (
                                T.cast(
                                    T.reinterpret(high0, T.uint16), T.uint32
                                )
                                << 16
                            )
                            if (
                                all_second_chunks_valid
                                or chunk1 < full_chunks
                            ):
                                low1 = T.cast(
                                    sums1[word * 2], T.bfloat16
                                )
                                high1 = T.cast(
                                    sums1[word * 2 + 1], T.bfloat16
                                )
                                result1[word] = T.cast(
                                    T.reinterpret(low1, T.uint16), T.uint32
                                ) | (
                                    T.cast(
                                        T.reinterpret(high1, T.uint16),
                                        T.uint32,
                                    )
                                    << 16
                                )
                        output0 = (
                            token * H + chunk0_safe * _BULK_HIDDEN
                        )
                        if chunk0_valid:
                            fast_bulk_store_512(
                                output[0],
                                output0 * 2,
                                result0[T.Ramp(0, 1, 4)],
                                T.bool(False),
                            )
                        if (
                            all_second_chunks_valid
                            or chunk1 < full_chunks
                        ):
                            output1 = token * H + chunk1 * _BULK_HIDDEN
                            fast_bulk_store_512(
                                output[0],
                                output1 * 2,
                                result1[T.Ramp(0, 1, 4)],
                                T.bool(False),
                            )

                    for tail_round in T.serial(tail_256_rounds):
                        tail_chunk = warp + tail_round * acc_warps
                        tail_valid = tail_chunk < tail_256_chunks
                        tail_safe = T.if_then_else(
                            tail_valid, tail_chunk, 0
                        )
                        sums = T.alloc_local((4,), T.float32)
                        hidden_offset = (
                            bulk_hidden + tail_safe * (256 // 2)
                        )
                        accumulate_tail(
                            hidden_full,
                            route_addresses,
                            route_peers,
                            route_primary,
                            rank,
                            hidden_offset,
                            sums,
                            4,
                        )
                        if tail_valid:
                            store_tail(
                                output, token, hidden_offset, lane, sums, 4
                            )

                    for tail_round in T.serial(tail_128_rounds):
                        tail_chunk = warp + tail_round * acc_warps
                        tail_valid = tail_chunk < tail_128_chunks
                        tail_safe = T.if_then_else(
                            tail_valid, tail_chunk, 0
                        )
                        sums = T.alloc_local((2,), T.float32)
                        hidden_offset = (
                            bulk_hidden
                            + tail_256_chunks * (256 // 2)
                            + tail_safe * (128 // 2)
                        )
                        accumulate_tail(
                            hidden_full,
                            route_addresses,
                            route_peers,
                            route_primary,
                            rank,
                            hidden_offset,
                            sums,
                            2,
                        )
                        if tail_valid:
                            store_tail(
                                output, token, hidden_offset, lane, sums, 2
                            )

                    if scalar_pairs > 0 and warp == 0:
                        if lane < scalar_pairs:
                            # A partial stripe must not read the next row or
                            # the unmapped page after the final peer shard.
                            scalar_sums = T.alloc_local((2,), T.float32)
                            scalar_sums[0] = T.float32(0)
                            scalar_sums[1] = T.float32(0)
                            for k in T.serial(K):
                                if route_primary[k] != 0:
                                    address = (
                                        route_addresses[k]
                                        + scalar_hidden + lane * 2
                                    )
                                    word = T.decl_buffer(
                                        (R * NvS_padded * H // 2,),
                                        dtype=T.int32, data=hidden_full.data,
                                    )
                                    values = T.alloc_local((2,), T.bfloat16)
                                    values[0:2] = T.reinterpret(
                                        T.atomic_load(
                                            word[address // 2],
                                            memory_order="relaxed",
                                            scope="system",
                                        ),
                                        "bfloat16x2",
                                    )
                                    for value in T.unroll(2):
                                        scalar_sums[value] += T.cast(
                                            values[value], T.float32
                                        )
                            store_tail(
                                output,
                                token,
                                scalar_hidden,
                                lane,
                                scalar_sums,
                                2,
                            )

        return tilelang.compile(
            kernel,
            execution_backend="tvm_ffi",
            target="ppu",
            compile_flags=["--dopt=3"],
        )

    return _compile_cached(
        (
            "combine",
            R,
            S,
            H,
            K,
            NvS,
            NvS_padded,
            meta_stride,
            with_weights,
        ),
        build,
    )

def _launch_combine(
    ctx: dict,
    output_sh: torch.Tensor,
    dst: torch.Tensor,
    output_weight_bits: torch.Tensor,
    *,
    with_weights: bool,
) -> None:
    R = int(ctx["R"])
    S = int(ctx["S"])
    H = int(ctx["H"])
    K = int(ctx["K"])
    NvS = int(ctx["NvS"])
    if H % 2 != 0:
        raise ValueError("TileLang PPU combine requires even H")
    if not 0 < K <= 32:
        raise ValueError("TileLang PPU combine requires 1 <= K <= 32")
    hidden = ctx["hidden_buf"]
    meta = ctx["meta_buf"]
    with torch.cuda.device(output_sh.device):
        kernel = _compile_combine(
            R=R,
            S=S,
            H=H,
            K=K,
            NvS=NvS,
            NvS_padded=int(ctx["NvS_padded"]),
            meta_stride=int(ctx["meta_chunk_padded"]),
            with_weights=with_weights,
        )
        import tvm_ffi

        launch_inter_rank_sync(ctx)
        with tvm_ffi.use_torch_stream():
            kernel(
                output_sh.reshape(-1),
                output_weight_bits.reshape(-1),
                hidden.reshape(-1),
                meta.reshape(-1),
                dst.reshape(-1),
                int(ctx["rank"]),
                int(ctx["WEIGHTS_OFF"]),
            )


def launch_combine(
    ctx: dict,
    output_sh: torch.Tensor,
    dst: torch.Tensor,
    output_sk: torch.Tensor | None = None,
    *,
    pdl_launch: bool = False,
) -> None:
    """Launch the all-SM TileLang combine kernel."""
    del pdl_launch
    S = int(ctx["S"])
    H = int(ctx["H"])
    K = int(ctx["K"])
    assert output_sh.dtype == torch.bfloat16 and output_sh.is_contiguous(), (
        "hidden_sh must be contiguous bf16"
    )
    assert tuple(output_sh.shape) == (S, H), (
        f"hidden_sh must be shape [S={S}, H={H}], "
        f"got {tuple(output_sh.shape)}"
    )
    assert dst.dtype == torch.int32 and dst.is_contiguous(), (
        "dst must be contiguous int32"
    )
    assert tuple(dst.shape) == (S * K,), (
        f"dst must be shape [{S * K}], got {tuple(dst.shape)}"
    )
    assert dst.device == output_sh.device
    with_weights = output_sk is not None
    if with_weights:
        assert output_sk.dtype == torch.float32 and output_sk.is_contiguous(), (
            "output_sk must be contiguous fp32"
        )
        assert tuple(output_sk.shape) == (S, K), (
            f"output_sk must be shape [S={S}, K={K}], "
            f"got {tuple(output_sk.shape)}"
        )
        assert output_sk.device == output_sh.device
    if output_sk is None:
        output_sk = dst
    _launch_combine(
        ctx,
        output_sh,
        dst,
        output_sk.reshape(-1).view(torch.int32),
        with_weights=with_weights,
    )
