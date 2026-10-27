"""PPU planning with rank-local allocation and stable expert-major routing.

Each rank replicates token counts and computes the same deterministic plan.
This exchanges R*E counts per rank to avoid concentrated reads of rank 0's
plan on a fabric without hardware multicast; route sorting stays rank-local.
Disjoint CTA groups overlap allocation with local route sorting when scratch
space and the CTA budget permit. Both schedules share the same algorithms;
the final grid rendezvous precedes publishing routes to their destinations.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class MoonEPCommPlan:
    dst: torch.Tensor
    experts_to_copy: torch.Tensor
    zero_fill_ranges: torch.Tensor
    remote_stats: torch.Tensor
    N: int
    R: int
    E: int
    NvS: int
    K: int
    dup_groups: torch.Tensor
    dup_loffs: torch.Tensor
    dup_counts: torch.Tensor

    def __post_init__(self) -> None:
        assert self.dst.dtype == torch.int32 and self.dst.is_contiguous()
        assert tuple(self.dst.shape) == (self.N,)
        assert (
            self.experts_to_copy.dtype == torch.int32
            and self.experts_to_copy.is_contiguous()
        )
        epn = self.E // self.R
        assert tuple(self.experts_to_copy.shape) == (self.R, epn)
        assert (
            self.zero_fill_ranges.dtype == torch.int32
            and self.zero_fill_ranges.is_contiguous()
        )
        assert tuple(self.zero_fill_ranges.shape) == (2 * epn, 2)
        assert (
            self.remote_stats.dtype == torch.int32
            and self.remote_stats.is_contiguous()
        )
        assert tuple(self.remote_stats.shape) == (2,)
        assert (
            self.dup_groups.dtype == torch.int32
            and self.dup_groups.is_contiguous()
        )
        assert tuple(self.dup_groups.shape) == (self.NvS, 3)
        assert (
            self.dup_loffs.dtype == torch.int32
            and self.dup_loffs.is_contiguous()
        )
        assert tuple(self.dup_loffs.shape) == (self.NvS,)
        assert (
            self.dup_counts.dtype == torch.int32
            and self.dup_counts.is_contiguous()
        )
        assert tuple(self.dup_counts.shape) == (2,)

    def clone(self) -> "MoonEPCommPlan":
        return type(self)(
            dst=self.dst.clone(),
            experts_to_copy=self.experts_to_copy.clone(),
            zero_fill_ranges=self.zero_fill_ranges.clone(),
            remote_stats=self.remote_stats.clone(),
            dup_groups=self.dup_groups.clone(),
            dup_loffs=self.dup_loffs.clone(),
            dup_counts=self.dup_counts.clone(),
            N=self.N,
            R=self.R,
            E=self.E,
            NvS=self.NvS,
            K=self.K,
        )


def _validate_dedup_plan(
    ctx: dict, plan: MoonEPCommPlan, device: torch.device
) -> None:
    NvS = int(ctx["NvS"])
    assert isinstance(plan, MoonEPCommPlan)
    assert plan.NvS == NvS

    def check(tensor, name, shape):
        assert tensor.dtype == torch.int32 and tensor.is_contiguous(), (
            f"{name} must be contiguous int32"
        )
        assert tuple(tensor.shape) == shape, (
            f"{name} must be shape {shape}, got {tuple(tensor.shape)}"
        )
        assert tensor.device == device, f"{name} must be on {device}"

    check(plan.dup_groups, "dup_groups", (NvS, 3))
    check(plan.dup_loffs, "dup_loffs", (NvS,))
    check(plan.dup_counts, "dup_counts", (2,))


def allocate_planning_outputs(ctx: dict) -> tuple[MoonEPCommPlan, torch.Tensor]:
    E = int(ctx["E"])
    epn = E // int(ctx["R"])
    N = int(ctx["S"]) * int(ctx["K"])
    NvS = int(ctx["NvS"])
    device = ctx["meta_buf"].device
    plan = MoonEPCommPlan(
        dst=torch.empty(N, dtype=torch.int32, device=device),
        experts_to_copy=torch.empty(
            int(ctx["R"]), epn, dtype=torch.int32, device=device
        ),
        zero_fill_ranges=torch.empty(
            2 * epn, 2, dtype=torch.int32, device=device
        ),
        remote_stats=torch.empty(2, dtype=torch.int32, device=device),
        dup_groups=torch.empty(NvS, 3, dtype=torch.int32, device=device),
        dup_loffs=torch.empty(NvS, dtype=torch.int32, device=device),
        dup_counts=torch.empty(2, dtype=torch.int32, device=device),
        N=N,
        R=int(ctx["R"]),
        E=E,
        NvS=NvS,
        K=int(ctx["K"]),
    )
    return plan, torch.empty(2 * epn, dtype=torch.int32, device=device)


_PLANNING_THREAD_CHOICES = (512, 256, 128, 64)
_MAX_PLANNING_SHARED_BYTES = 192 * 1024
_PHASE_D_STAGE_INTS = 128
_PLANNING_AUX_SHARED_INTS = _PHASE_D_STAGE_INTS + 5
_COMPILE_LOCK = threading.Lock()
_COMPILED_KERNELS: dict[tuple, object] = {}


def _planning_threads(E: int) -> int:
    for threads in _PLANNING_THREAD_CHOICES:
        num_warps = threads // 32
        shared_ints = max(
            (num_warps + 1) * (E + 1),
            2 * threads,
        )
        if (
            shared_ints + _PLANNING_AUX_SHARED_INTS
        ) * 4 <= _MAX_PLANNING_SHARED_BYTES:
            return threads
    raise ValueError(
        f"planning configuration E={E} "
        f"exceeds {_MAX_PLANNING_SHARED_BYTES} bytes of shared memory"
    )


def _compile_planning(
    *,
    R: int,
    E: int,
    S: int,
    K: int,
    NvS_capacity: int,
    NvS: int,
    meta_stride: int,
    TPE_OFF: int,
    PLAN_OFF: int,
    ORDER_OFF: int,
    BARRIER_OFF: int,
    SRC_INFO_OFF: int,
    token_padding: int,
    num_sms: int,
):
    device = torch.cuda.current_device()
    shape_key = (
        device,
        torch.cuda.get_device_capability(device),
        R,
        E,
        S,
        K,
        NvS_capacity,
        NvS,
        meta_stride,
        TPE_OFF,
        PLAN_OFF,
        ORDER_OFF,
        BARRIER_OFF,
        SRC_INFO_OFF,
        token_padding,
        num_sms,
    )
    compiled = _COMPILED_KERNELS.get(shape_key)
    if compiled is not None:
        return compiled

    with _COMPILE_LOCK:
        compiled = _COMPILED_KERNELS.get(shape_key)
        if compiled is not None:
            return compiled

        compiled = _build_planning_kernel(
            R=R,
            E=E,
            S=S,
            K=K,
            NvS_capacity=NvS_capacity,
            NvS=NvS,
            meta_stride=meta_stride,
            TPE_OFF=TPE_OFF,
            PLAN_OFF=PLAN_OFF,
            ORDER_OFF=ORDER_OFF,
            BARRIER_OFF=BARRIER_OFF,
            SRC_INFO_OFF=SRC_INFO_OFF,
            token_padding=token_padding,
            num_sms=num_sms,
        )
        _COMPILED_KERNELS[shape_key] = compiled
        return compiled


def _build_planning_kernel(
    *,
    R: int,
    E: int,
    S: int,
    K: int,
    NvS_capacity: int,
    NvS: int,
    meta_stride: int,
    TPE_OFF: int,
    PLAN_OFF: int,
    ORDER_OFF: int,
    BARRIER_OFF: int,
    SRC_INFO_OFF: int,
    token_padding: int,
    num_sms: int,
):
    import tilelang
    import tilelang.ppu.language as T

    from .primitives import fast_bulk_store_128_ref

    N = S * K
    epn = E // R
    # local_hist is allocated in 2048-route tiles by the buffer context.
    num_vblocks = (N + 2047) // 2048
    threads = _planning_threads(E)
    num_warps = threads // 32
    warp_bins = E + 1
    desired_sort_ctas = max(1, (num_sms + 1) // 2)
    routes_per_sort_cta = (
        (N + desired_sort_ctas * threads - 1)
        // (desired_sort_ctas * threads)
    ) * threads
    route_block_size = (
        min(8192, max(4096, routes_per_sort_cta)) if num_sms > 1 else 4096
    )
    route_num_vblocks = (N + route_block_size - 1) // route_block_size
    routes_per_thread = route_block_size // threads

    # TOPK0 and ORDER0 are unused by this backend's stable sort. During
    # concurrent planning, TOPK0 holds E destination hints and ORDER0 holds
    # two pairs of subgroup counters. ORDER itself remains the route output.
    route_scratch_elements = ((N + 3) // 4) * 4
    hint_offset = ORDER_OFF - route_scratch_elements
    sort_barrier_offset = ORDER_OFF + route_scratch_elements
    core_barrier_offset = sort_barrier_offset + 2
    parallel_planning = (
        num_sms > 1
        and route_num_vblocks < num_sms
        and route_scratch_elements >= max(E, 4)
    )
    sort_ctas = route_num_vblocks if parallel_planning else num_sms
    core_ctas = num_sms - sort_ctas if parallel_planning else num_sms
    sort_expert_tile = ((E + sort_ctas - 1) // sort_ctas + 31) // 32 * 32
    core_expert_tile = ((E + core_ctas - 1) // core_ctas + 31) // 32 * 32
    core_expert_tile_rounds = max(1, core_expert_tile // 32)
    core_rank_tiles = max(1, (R + core_ctas - 1) // core_ctas)
    quota_items = max(1, (R + 31) // 32)
    owner_expert_items = max(1, (epn + 31) // 32)
    # The cumulative row has R entries and may begin with zero, so an
    # upper-bound search needs one extra probe for power-of-two world sizes.
    rank_search_steps = max(R.bit_length(), 1)
    expert_scan_items = (E + 31) // 32
    output_groups = 2 * epn
    eb_pad = 1 << max((output_groups - 1).bit_length(), 0)
    layout_items = max(1, (eb_pad + threads - 1) // threads)
    prefix_sources_per_warp = max(1, (R + num_warps - 1) // num_warps)
    shared_ints = max(
        (num_warps + 1) * warp_bins,
        2 * threads,
    )
    candidate_items = max(1, (E + 31) // 32)
    tpe_vector_words = (E // 32) * 32
    phase_d_group_tile = (
        ((output_groups + num_sms - 1) // num_sms + 31) // 32 * 32
    )
    phase_d_copy_tile = ((E + num_sms - 1) // num_sms + 31) // 32 * 32
    phase_d_use_aiu = (
        phase_d_group_tile <= 32 and phase_d_copy_tile <= 32
    )
    allocation_prefix_offset = 0
    source_token_prefix_offset = E * R
    expert_offsets_offset = 2 * E * R
    cu_seqlens_offset = 3 * E * R
    zero_fill_offset = cu_seqlens_offset + R * output_groups
    experts_to_copy_offset = zero_fill_offset + 2 * R * output_groups
    stats_offset = experts_to_copy_offset + R * epn

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
        # Complete all CTA writes before publishing this rank's arrival.
        T.sync_grid()

        local_meta = rank * meta_stride
        gtid = bx * threads + tx
        if tx == 0:
            barrier_state[0] = meta[local_meta + BARRIER_OFF + 2]
            barrier_phase[0] = barrier_state[0] & 1
            barrier_sign[0] = T.if_then_else(
                (barrier_state[0] & 2) != 0, -1, 1
            )
            barrier_target[0] = T.if_then_else(
                barrier_sign[0] > 0, R, 0
            )
        T.sync_threads()
        if bx == 0 and tx < R:
            T.atomic_add(
                meta[
                    tx * meta_stride
                    + BARRIER_OFF
                    + barrier_phase[0]
                ],
                barrier_sign[0],
                memory_order="release",
                scope="system",
            )
        T.sync_threads()
        if gtid == 0:
            own_index = local_meta + BARRIER_OFF + barrier_phase[0]
            meta[local_meta + BARRIER_OFF + 2] = (
                meta[local_meta + BARRIER_OFF + 2] + 1
            ) & 3
            start = T.alloc_local((1,), T.uint64)
            loaded = T.alloc_local((1,), T.int32)
            start[0] = T.call_extern("uint64", "clock64")
            loaded[0] = T.atomic_load(
                meta[own_index], memory_order="acquire", scope="system"
            )
            while loaded[0] != barrier_target[0]:
                if (
                    T.call_extern("uint64", "clock64") - start[0]
                    > T.uint64(BARRIER_TIMEOUT_CYCLES)
                ):
                    T.call_extern("void", "__builtin_trap")
                loaded[0] = T.atomic_load(
                    meta[own_index],
                    memory_order="acquire",
                    scope="system",
                )
        T.sync_grid()

    @T.macro
    def planning_group_barrier(meta, counter_base, phase, tx, participants):
        """Synchronize a resident CTA subgroup, or the whole serial grid."""
        if parallel_planning:
            # Participants must be resident, as required by sync_grid too.
            # Every participant uses the same phase. Alternating counters
            # prevent the next arrival from overtaking a previous waiter.
            # The surrounding full-grid barriers protect initialization and
            # reuse; no other group may access this pair of counters.
            T.sync_threads()
            if tx == 0:
                slot = phase[0] & 1
                sign = T.if_then_else((phase[0] & 2) != 0, -1, 1)
                target = T.if_then_else(sign > 0, participants, 0)
                T.atomic_add(
                    meta[counter_base + slot], sign,
                    memory_order="release", scope="device",
                )
                loaded = T.alloc_local((1,), T.int32)
                started = T.alloc_local((1,), T.uint64)
                started[0] = T.call_extern("uint64", "clock64")
                loaded[0] = T.atomic_load(
                    meta[counter_base + slot],
                    memory_order="acquire", scope="device",
                )
                while loaded[0] != target:
                    if (
                        T.call_extern("uint64", "clock64") - started[0]
                        > T.uint64(BARRIER_TIMEOUT_CYCLES)
                    ):
                        T.call_extern("void", "__builtin_trap")
                    loaded[0] = T.atomic_load(
                        meta[counter_base + slot],
                        memory_order="acquire", scope="device",
                    )
                phase[0] = (phase[0] + 1) & 3
            T.sync_threads()
        else:
            T.sync_grid()

    @T.macro
    def build_expert_prefix(tpe, route_shared, tx):
        """Exclusive expert prefix in shared memory, visible to the CTA."""
        for expert in T.serial(tx, E, threads):
            route_shared[expert] = tpe[expert]
        T.sync_threads()
        if tx < 32:
            segment_sum = T.alloc_local((1,), T.int32)
            segment_sum[0] = 0
            for item in T.serial(expert_scan_items):
                expert = tx * expert_scan_items + item
                value = T.alloc_local((1,), T.int32)
                value[0] = T.if_then_else(
                    expert < E, route_shared[expert], 0
                )
                if expert < E:
                    route_shared[expert] = segment_sum[0]
                segment_sum[0] += value[0]
            inclusive = T.alloc_local((1,), T.int32)
            shuffled = T.alloc_local((1,), T.int32)
            inclusive[0] = segment_sum[0]
            shuffled[0] = T.shfl_up(inclusive[0], 1)
            if tx >= 1:
                inclusive[0] += shuffled[0]
            shuffled[0] = T.shfl_up(inclusive[0], 2)
            if tx >= 2:
                inclusive[0] += shuffled[0]
            shuffled[0] = T.shfl_up(inclusive[0], 4)
            if tx >= 4:
                inclusive[0] += shuffled[0]
            shuffled[0] = T.shfl_up(inclusive[0], 8)
            if tx >= 8:
                inclusive[0] += shuffled[0]
            shuffled[0] = T.shfl_up(inclusive[0], 16)
            if tx >= 16:
                inclusive[0] += shuffled[0]
            segment_offset = inclusive[0] - segment_sum[0]
            for item in T.serial(expert_scan_items):
                expert = tx * expert_scan_items + item
                if expert < E:
                    route_shared[expert] += segment_offset
        T.sync_threads()

    @T.macro
    def build_local_route_order(
        topk,
        tpe,
        meta,
        local_hist,
        phase,
        route_shared,
        order_base,
        bx,
        tx,
    ):
        """Build the local rank's expert-major route order."""
        warp = tx // 32

        for vblock in T.serial(bx, route_num_vblocks, sort_ctas):
            for expert_base in T.serial(0, E, threads):
                expert = expert_base + tx
                if expert < E:
                    route_shared[expert] = 0
            T.sync_threads()
            vblock_begin = vblock * route_block_size
            vblock_end = T.min(vblock_begin + route_block_size, N)
            for route_base in T.serial(vblock_begin, vblock_end, threads):
                offv = route_base + tx
                if offv < vblock_end:
                    route_expert = topk[offv]
                    T.atomic_add(route_shared[route_expert], 1)
            T.sync_threads()
            for expert_base in T.serial(0, E, threads):
                expert = expert_base + tx
                if expert < E:
                    local_hist[vblock * E + expert] = route_shared[expert]
            T.sync_threads()
        planning_group_barrier(
            meta, order_base + route_scratch_elements, phase, tx, sort_ctas
        )

        # Each lane scans one contiguous expert segment; a warp scan supplies
        # the segment offsets.
        build_expert_prefix(tpe, route_shared, tx)
        expert_begin = bx * sort_expert_tile
        expert_end = T.min(expert_begin + sort_expert_tile, E)
        for expert in T.serial(
            expert_begin + tx, expert_end, threads
        ):
            prefix = T.alloc_local((1,), T.int32)
            prefix[0] = route_shared[expert]
            for vblock in T.serial(route_num_vblocks):
                count = T.alloc_local((1,), T.int32)
                count[0] = local_hist[vblock * E + expert]
                local_hist[vblock * E + expert] = prefix[0]
                prefix[0] += count[0]
        planning_group_barrier(
            meta, order_base + route_scratch_elements, phase, tx, sort_ctas
        )

        for vblock in T.serial(bx, route_num_vblocks, sort_ctas):
            vblock_begin = vblock * route_block_size
            vblock_end = T.min(vblock_begin + route_block_size, N)
            for i in T.serial(tx, num_warps * warp_bins, threads):
                route_shared[i] = 0
            T.sync_threads()

            route_experts = T.alloc_local((routes_per_thread,), T.int32)
            route_within_warp = T.alloc_local(
                (routes_per_thread,), T.int32
            )
            for item in T.serial(routes_per_thread):
                route_in_vblock = (
                    warp * (32 * routes_per_thread)
                    + item * 32
                    + tx % 32
                )
                offv = vblock_begin + route_in_vblock
                valid = offv < vblock_end
                route_experts[item] = T.if_then_else(valid, topk[offv], E)
                peers = T.match_any_sync(route_experts[item])
                lanes_before = (
                    T.uint32(1) << T.cast(tx % 32, T.uint32)
                ) - T.uint32(1)
                base = T.alloc_local((1,), T.int32)
                base[0] = T.atomic_load(
                    route_shared[
                        warp * warp_bins + route_experts[item]
                    ],
                    memory_order="relaxed",
                    scope="block",
                )
                route_within_warp[item] = base[0] + T.cast(
                    T.popcount(peers & lanes_before), T.int32
                )
                T.sync_warp()
                if (peers & lanes_before) == 0:
                    T.atomic_store(
                        route_shared[
                            warp * warp_bins + route_experts[item]
                        ],
                        base[0] + T.cast(T.popcount(peers), T.int32),
                        memory_order="relaxed",
                        scope="block",
                    )
                T.sync_warp()
            T.sync_threads()

            for expert_base in T.serial(0, E, threads):
                expert = expert_base + tx
                if expert < E:
                    warp_prefix = T.alloc_local((1,), T.int32)
                    warp_count = T.alloc_local((1,), T.int32)
                    warp_prefix[0] = 0
                    for warp_index in T.serial(num_warps):
                        warp_count[0] = T.atomic_load(
                            route_shared[
                                warp_index * warp_bins + expert
                            ],
                            memory_order="relaxed",
                            scope="block",
                        )
                        T.atomic_store(
                            route_shared[
                                warp_index * warp_bins + expert
                            ],
                            warp_prefix[0],
                            memory_order="relaxed",
                            scope="block",
                        )
                        warp_prefix[0] += warp_count[0]
            T.sync_threads()

            for item in T.serial(routes_per_thread):
                route_in_vblock = (
                    warp * (32 * routes_per_thread)
                    + item * 32
                    + tx % 32
                )
                offv = vblock_begin + route_in_vblock
                if offv < vblock_end:
                    expert = route_experts[item]
                    warp_route_prefix = T.atomic_load(
                        route_shared[warp * warp_bins + expert],
                        memory_order="relaxed",
                        scope="block",
                    )
                    order_position = (
                        local_hist[vblock * E + expert]
                        + warp_route_prefix
                        + route_within_warp[item]
                    )
                    meta[order_base + order_position] = offv
            T.sync_threads()
        planning_group_barrier(
            meta, order_base + route_scratch_elements, phase, tx, sort_ctas
        )

    @T.macro
    def build_core_plan(
        meta, alloc, group_tokens, z, experts_to_copy,
        local_hist, route_shared, phase, rank, bx, tx,
    ):
        """Replicate deterministic allocation and compact layout on each rank."""
        local_meta = rank * meta_stride
        gtid = bx * threads + tx
        warp = tx // 32
        lane = tx % 32
        global_warp = bx * num_warps + warp
        core_grid_threads = core_ctas * threads
        core_total_warps = core_ctas * num_warps
        # Assign a contiguous expert segment to every CTA. Warps split the
        # source-rank dimension into consecutive chunks, compute local
        # prefixes, and add the preceding chunks' totals after one CTA
        # rendezvous. This preserves source-major coalescing without a
        # single warp's serial work growing linearly with R.
        expert_begin = bx * core_expert_tile
        for expert_round in T.serial(core_expert_tile_rounds):
            expert_base = expert_begin + expert_round * 32
            expert = expert_base + lane
            run = T.alloc_local((1,), T.int32)
            local_prefixes = T.alloc_local(
                (prefix_sources_per_warp,), T.int32
            )
            run[0] = 0
            for item in T.unroll(prefix_sources_per_warp):
                local_prefixes[item] = 0
                source = warp * prefix_sources_per_warp + item
                if expert < E and source < R:
                    source_count = T.alloc_local((1,), T.int32)
                    source_count[0] = meta[
                        local_meta
                        + TPE_OFF
                        + source * E
                        + expert
                    ]
                    run[0] += source_count[0]
                    local_prefixes[item] = run[0]
            route_shared[warp * 32 + lane] = run[0]
            T.sync_threads()

            carry = T.alloc_local((1,), T.int32)
            carry[0] = 0
            for prior_warp in T.unroll(num_warps):
                if prior_warp < warp:
                    carry[0] += route_shared[
                        prior_warp * 32 + lane
                    ]
            for item in T.unroll(prefix_sources_per_warp):
                source = warp * prefix_sources_per_warp + item
                if expert < E and source < R:
                    meta[
                        local_meta
                        + PLAN_OFF
                        + source_token_prefix_offset
                        + source * E
                        + expert
                    ] = local_prefixes[item] + carry[0]

            if warp == 0 and expert < E:
                total = T.alloc_local((1,), T.int32)
                total[0] = 0
                for source_warp in T.unroll(num_warps):
                    total[0] += route_shared[
                        source_warp * 32 + lane
                    ]
                owner = expert // epn
                alloc[owner * E + expert] = total[0]
                T.atomic_add(group_tokens[owner], total[0])
            T.sync_threads()
        planning_group_barrier(
            meta, local_meta + core_barrier_offset, phase, tx, core_ctas
        )
        if bx == 0 and tx < 32:
            balances = T.alloc_local((quota_items,), T.int32)
            for item in T.unroll(quota_items):
                candidate_rank = tx + item * 32
                balances[item] = T.if_then_else(
                    candidate_rank < R,
                    group_tokens[candidate_rank] - NvS_capacity,
                    0,
                )
            keep_balancing = T.alloc_local((1,), T.int32)
            keep_balancing[0] = 1
            while keep_balancing[0] != 0:
                surplus_key = T.alloc_local((1,), T.int32)
                deficit_key = T.alloc_local((1,), T.int32)
                surplus_key[0] = 0
                deficit_key[0] = 0
                for item in T.unroll(quota_items):
                    candidate_rank = tx + item * 32
                    if candidate_rank < R:
                        if balances[item] > 0:
                            candidate = (
                                balances[item] * (R + 1)
                                + R
                                - candidate_rank
                            )
                            if candidate > surplus_key[0]:
                                surplus_key[0] = candidate
                        if balances[item] < 0:
                            candidate = (
                                -balances[item] * (R + 1)
                                + R
                                - candidate_rank
                            )
                            if candidate > deficit_key[0]:
                                deficit_key[0] = candidate
                surplus_winner = T.warp_reduce_max(surplus_key[0])
                deficit_winner = T.warp_reduce_max(deficit_key[0])
                surplus = surplus_winner // (R + 1)
                deficit = deficit_winner // (R + 1)
                surplus_rank = R - surplus_winner % (R + 1)
                deficit_rank = R - deficit_winner % (R + 1)
                if surplus <= 0 or deficit <= 0:
                    keep_balancing[0] = 0
                else:
                    for item in T.unroll(quota_items):
                        candidate_rank = tx + item * 32
                        if candidate_rank == surplus_rank:
                            balances[item] -= deficit
                        elif candidate_rank == deficit_rank:
                            balances[item] = 0
                    if tx == 0:
                        z[surplus_rank * R + deficit_rank] = deficit

        planning_group_barrier(
            meta, local_meta + core_barrier_offset, phase, tx, core_ctas
        )
        # One CTA owns each expert-owner row. Its first warp keeps the
        # destination quotas and owner-expert counts in registers.
        for owner in T.serial(bx, R, core_ctas):
            expert_base = owner * epn
            for index in T.serial(tx, R * epn, threads):
                destination = index // epn
                local_expert = index - destination * epn
                route_shared[index] = alloc[
                    destination * E + expert_base + local_expert
                ]
            T.sync_threads()

            if tx < 32:
                quotas = T.alloc_local((quota_items,), T.int32)
                owner_remaining = T.alloc_local(
                    (owner_expert_items,), T.int32
                )
                for item in T.unroll(quota_items):
                    destination = tx + item * 32
                    quotas[item] = T.if_then_else(
                        destination < R,
                        z[owner * R + destination],
                        0,
                    )
                for item in T.unroll(owner_expert_items):
                    local_expert = tx + item * 32
                    owner_remaining[item] = T.if_then_else(
                        local_expert < epn,
                        route_shared[owner * epn + local_expert],
                        0,
                    )

                keep_balancing = T.alloc_local((1,), T.int32)
                keep_balancing[0] = 1
                while keep_balancing[0] != 0:
                    quota_key = T.alloc_local((1,), T.int32)
                    quota_key[0] = 0
                    for item in T.unroll(quota_items):
                        destination = tx + item * 32
                        if destination < R and quotas[item] > 0:
                            candidate = (
                                quotas[item] * (R + 1)
                                + R
                                - destination
                            )
                            if candidate > quota_key[0]:
                                quota_key[0] = candidate
                    quota_winner = T.warp_reduce_max(quota_key[0])
                    max_quota = quota_winner // (R + 1)
                    target_rank = R - quota_winner % (R + 1)

                    remaining_key = T.alloc_local((1,), T.int32)
                    remaining_key[0] = 0
                    for item in T.unroll(owner_expert_items):
                        local_expert = tx + item * 32
                        if (
                            local_expert < epn
                            and owner_remaining[item] > 0
                        ):
                            candidate = (
                                owner_remaining[item] * (epn + 1)
                                + epn
                                - local_expert
                            )
                            if candidate > remaining_key[0]:
                                remaining_key[0] = candidate
                    remaining_winner = T.warp_reduce_max(
                        remaining_key[0]
                    )
                    max_remaining = remaining_winner // (epn + 1)
                    selected_expert = (
                        epn - remaining_winner % (epn + 1)
                    )

                    if max_quota <= 0 or max_remaining <= 0:
                        keep_balancing[0] = 0
                    else:
                        take = T.min(max_quota, max_remaining)
                        for item in T.unroll(quota_items):
                            if tx + item * 32 == target_rank:
                                quotas[item] = max_quota - take
                        for item in T.unroll(owner_expert_items):
                            if tx + item * 32 == selected_expert:
                                owner_remaining[item] = (
                                    max_remaining - take
                                )
                        if tx == 0:
                            target_index = (
                                target_rank * epn + selected_expert
                            )
                            target_value = T.atomic_load(
                                route_shared[target_index],
                                memory_order="relaxed",
                                scope="block",
                            )
                            T.atomic_store(
                                route_shared[target_index],
                                target_value + take,
                                memory_order="relaxed",
                                scope="block",
                            )
                            T.atomic_store(
                                route_shared[
                                    owner * epn + selected_expert
                                ],
                                max_remaining - take,
                                memory_order="relaxed",
                                scope="block",
                            )
            T.sync_threads()

            for index in T.serial(tx, R * epn, threads):
                destination = index // epn
                local_expert = index - destination * epn
                alloc[
                    destination * E + expert_base + local_expert
                ] = route_shared[index]
            T.sync_threads()
        planning_group_barrier(
            meta, local_meta + core_barrier_offset, phase, tx, core_ctas
        )
        # Store cumulative allocation counts as
        # [expert, destination_rank].
        for expert in T.serial(
            global_warp, E, core_total_warps
        ):
            carry = T.alloc_local((1,), T.int32)
            nonzero_destinations = T.alloc_local((1,), T.int32)
            destination_sum = T.alloc_local((1,), T.int32)
            carry[0] = 0
            nonzero_destinations[0] = 0
            destination_sum[0] = 0
            for item in T.unroll(quota_items):
                destination = lane + item * 32
                allocation = T.alloc_local((1,), T.int32)
                inclusive = T.alloc_local((1,), T.int32)
                shuffled = T.alloc_local((1,), T.int32)
                allocation[0] = T.if_then_else(
                    destination < R,
                    alloc[destination * E + expert],
                    0,
                )
                inclusive[0] = allocation[0]
                shuffled[0] = T.shfl_up(inclusive[0], 1)
                if lane >= 1:
                    inclusive[0] += shuffled[0]
                shuffled[0] = T.shfl_up(inclusive[0], 2)
                if lane >= 2:
                    inclusive[0] += shuffled[0]
                shuffled[0] = T.shfl_up(inclusive[0], 4)
                if lane >= 4:
                    inclusive[0] += shuffled[0]
                shuffled[0] = T.shfl_up(inclusive[0], 8)
                if lane >= 8:
                    inclusive[0] += shuffled[0]
                shuffled[0] = T.shfl_up(inclusive[0], 16)
                if lane >= 16:
                    inclusive[0] += shuffled[0]
                inclusive[0] += carry[0]
                if destination < R:
                    meta[
                        local_meta
                        + PLAN_OFF
                        + allocation_prefix_offset
                        + expert * R
                        + destination
                    ] = inclusive[0]
                carry[0] = T.shfl_sync(inclusive[0], 31)
                tile_nonzero = T.warp_reduce_sum(
                    T.if_then_else(allocation[0] > 0, 1, 0)
                )
                tile_destination_sum = T.warp_reduce_sum(
                    T.if_then_else(
                        allocation[0] > 0, destination + 1, 0
                    )
                )
                if lane == 0:
                    nonzero_destinations[0] += tile_nonzero
                    destination_sum[0] += tile_destination_sum
            # Keep the single-destination hint until expert offsets are ready.
            if lane == 0:
                hint = T.if_then_else(
                    nonzero_destinations[0] == 1, destination_sum[0] - 1, -1
                )
                if parallel_planning:
                    meta[local_meta + hint_offset + expert] = hint
                else:
                    local_hist[expert] = hint
        # Keep nonzero candidates in registers and select the
        # smallest expert ID for each slot, matching the public
        # ascending expert order.
        for destination_tile in T.serial(
            warp, core_rank_tiles, num_warps
        ):
            destination = bx + destination_tile * core_ctas
            if destination < R:
                counts = T.alloc_local(
                    (candidate_items,), T.int32
                )
                nonzero = T.alloc_local((1,), T.int32)
                nonzero[0] = 0
                for item in T.serial(candidate_items):
                    expert = lane + item * 32
                    count = T.alloc_local((1,), T.int32)
                    count[0] = 0
                    if (
                        expert < E
                        and expert // epn != destination
                    ):
                        count[0] = alloc[
                            destination * E + expert
                        ]
                    counts[item] = count[0]
                    if count[0] > 0:
                        nonzero[0] += 1

                total_nonzero = T.warp_reduce_sum(nonzero[0])
                if lane == 0:
                    meta[
                        local_meta
                        + PLAN_OFF
                        + stats_offset
                        + destination * 2
                    ] = total_nonzero
                if lane == 0 and total_nonzero > epn:
                    T.call_extern("void", "__builtin_trap")

                selected_slots = T.alloc_local((1,), T.int32)
                slot = T.alloc_local((1,), T.int32)
                selected_slots[0] = T.min(total_nonzero, epn)
                slot[0] = 0
                while slot[0] < selected_slots[0]:
                    best_key = T.alloc_local((1,), T.int32)
                    best_key[0] = 0
                    for item in T.serial(candidate_items):
                        expert = lane + item * 32
                        if expert < E and counts[item] > 0:
                            candidate = (
                                E - expert
                            )
                            if candidate > best_key[0]:
                                best_key[0] = candidate
                    winner = T.warp_reduce_max(best_key[0])
                    selected = E - winner
                    if lane == 0:
                        experts_to_copy[
                            destination * epn + slot[0]
                        ] = selected
                        T.atomic_add(
                            meta[
                                local_meta
                                + PLAN_OFF
                                + stats_offset
                                + (selected // epn) * 2
                                + 1
                            ],
                            1,
                        )
                    for item in T.serial(candidate_items):
                        if lane + item * 32 == selected:
                            counts[item] = 0
                    slot[0] += 1
        # The full CTA consumes experts_to_copy while constructing
        # the layout below; wait for participating warps to publish
        # every slot.
        T.sync_threads()

        for destination in T.serial(
            bx, R, core_ctas
        ):
            # Each thread scans a contiguous group segment in registers,
            # followed by one block-wide prefix scan.
            count_values = T.alloc_local((layout_items,), T.int32)
            expert_values = T.alloc_local((layout_items,), T.int32)
            padded_values = T.alloc_local((layout_items,), T.int32)
            total_padded = T.alloc_local((1,), T.int32)
            total_padded[0] = 0
            for item in T.serial(layout_items):
                group = tx * layout_items + item
                expert_values[item] = -1
                count_values[item] = 0
                if group < output_groups:
                    if group < epn:
                        expert_values[item] = destination * epn + group
                    else:
                        expert_values[item] = experts_to_copy[
                            destination * epn + group - epn
                        ]
                    if expert_values[item] >= 0:
                        count_values[item] = alloc[
                            destination * E + expert_values[item]
                        ]
                padded_values[item] = 0
                if count_values[item] > 0:
                    padded_values[item] = (
                        (count_values[item] + token_padding - 1)
                        // token_padding
                        * token_padding
                    )
                total_padded[0] += padded_values[item]

            inclusive = T.alloc_local((1,), T.int32)
            shuffled = T.alloc_local((1,), T.int32)
            inclusive[0] = total_padded[0]
            shuffled[0] = T.shfl_up(inclusive[0], 1)
            if tx % 32 >= 1:
                inclusive[0] += shuffled[0]
            shuffled[0] = T.shfl_up(inclusive[0], 2)
            if tx % 32 >= 2:
                inclusive[0] += shuffled[0]
            shuffled[0] = T.shfl_up(inclusive[0], 4)
            if tx % 32 >= 4:
                inclusive[0] += shuffled[0]
            shuffled[0] = T.shfl_up(inclusive[0], 8)
            if tx % 32 >= 8:
                inclusive[0] += shuffled[0]
            shuffled[0] = T.shfl_up(inclusive[0], 16)
            if tx % 32 >= 16:
                inclusive[0] += shuffled[0]
            if tx % 32 == 31:
                route_shared[warp] = inclusive[0]
            T.sync_threads()
            if tx == 0:
                block_total = T.alloc_local((1,), T.int32)
                block_total[0] = 0
                for scan_warp in T.serial(num_warps):
                    warp_total = route_shared[scan_warp]
                    route_shared[scan_warp] = block_total[0]
                    block_total[0] += warp_total
                route_shared[num_warps] = block_total[0]
            T.sync_threads()

            base = T.alloc_local((1,), T.int32)
            base[0] = (
                route_shared[warp]
                + inclusive[0]
                - total_padded[0]
            )
            for item in T.serial(layout_items):
                group = tx * layout_items + item
                if group < output_groups:
                    if count_values[item] > 0:
                        meta[
                            local_meta
                            + PLAN_OFF
                            + expert_offsets_offset
                            + destination * E
                            + expert_values[item]
                        ] = base[0]
                    meta[
                        local_meta
                        + PLAN_OFF
                        + cu_seqlens_offset
                        + destination * output_groups
                        + group
                    ] = base[0] + padded_values[item]
                    if padded_values[item] > count_values[item]:
                        meta[
                            local_meta
                            + PLAN_OFF
                            + zero_fill_offset
                            + (
                                destination * output_groups + group
                            ) * 2
                        ] = base[0] + count_values[item]
                    else:
                        meta[
                            local_meta
                            + PLAN_OFF
                            + zero_fill_offset
                            + (
                                destination * output_groups + group
                            ) * 2
                        ] = 0
                    meta[
                        local_meta
                        + PLAN_OFF
                        + zero_fill_offset
                        + (destination * output_groups + group) * 2
                        + 1
                    ] = (
                        padded_values[item] - count_values[item]
                    )
                    base[0] += padded_values[item]
            if tx == 0 and route_shared[num_warps] > NvS:
                T.call_extern("void", "__builtin_trap")
            T.sync_threads()

        planning_group_barrier(
            meta, local_meta + core_barrier_offset, phase, tx, core_ctas
        )
        for index in T.serial(
            gtid, E, core_grid_threads
        ):
            source = rank
            expert = index
            destination = T.alloc_local((1,), T.int32)
            direct_base = T.alloc_local((1,), T.int32)
            previous_source_tokens = T.alloc_local((1,), T.int32)
            source_tokens_end = T.alloc_local((1,), T.int32)
            previous_allocation = T.alloc_local((1,), T.int32)
            if parallel_planning:
                destination[0] = meta[local_meta + hint_offset + expert]
            else:
                destination[0] = local_hist[expert]
            direct_base[0] = -1
            previous_source_tokens[0] = 0
            if source > 0:
                previous_source_tokens[0] = meta[
                    local_meta
                    + PLAN_OFF
                    + source_token_prefix_offset
                    + (source - 1) * E
                    + expert
                ]
            source_tokens_end[0] = meta[
                local_meta
                + PLAN_OFF
                + source_token_prefix_offset
                + source * E
                + expert
            ]

            # The global single-destination hint above is the cheapest
            # case. Otherwise, find the destination containing this
            # source's first token once and use a direct base when the
            # complete (source, expert) interval fits there. This moves a
            # binary search out of the per-route writeback.
            if (
                destination[0] < 0
                and source_tokens_end[0] > previous_source_tokens[0]
            ):
                destination_lo = T.alloc_local((1,), T.int32)
                destination_hi = T.alloc_local((1,), T.int32)
                destination_lo[0] = 0
                destination_hi[0] = R
                for _step in T.unroll(rank_search_steps):
                    middle = (
                        destination_lo[0] + destination_hi[0]
                    ) // 2
                    cumulative = meta[
                        local_meta
                        + PLAN_OFF
                        + allocation_prefix_offset
                        + expert * R
                        + middle
                    ]
                    if cumulative > previous_source_tokens[0]:
                        destination_hi[0] = middle
                    else:
                        destination_lo[0] = middle + 1
                candidate = destination_lo[0]
                if (
                    candidate < R
                    and source_tokens_end[0]
                    <= meta[
                        local_meta
                        + PLAN_OFF
                        + allocation_prefix_offset
                        + expert * R
                        + candidate
                    ]
                ):
                    destination[0] = candidate

            if destination[0] >= 0:
                previous_allocation[0] = 0
                if destination[0] > 0:
                    previous_allocation[0] = meta[
                        local_meta
                        + PLAN_OFF
                        + allocation_prefix_offset
                        + expert * R
                        + destination[0]
                        - 1
                    ]
                direct_base[0] = (
                    destination[0] * NvS
                    + meta[
                        local_meta
                        + PLAN_OFF
                        + expert_offsets_offset
                        + destination[0] * E
                        + expert
                    ]
                    + previous_source_tokens[0]
                    - previous_allocation[0]
                )
            # TPE reads are rank-local and complete before the grid sync;
            # this rank's row can now hold its direct routing offsets.
            meta[
                local_meta + TPE_OFF + rank * E + expert
            ] = direct_base[0]
        for index in T.serial(
            gtid, E, core_grid_threads
        ):
            meta[
                local_meta + PLAN_OFF + experts_to_copy_offset + index
            ] = experts_to_copy[index]

    @T.macro
    def stage_plan_slice(
        meta, stage, stage_offset, source_offset, begin, count, tile_words
    ):
        """Stage full tiles; partial tiles are loaded directly during writeback."""
        for tile in T.unroll(tile_words // 32):
            source_begin = begin + tile * 32
            shared_begin = stage_offset + tile * 32
            if source_begin + 32 <= count:
                T.ppu_aiu_load(
                    T.access_ptr(stage[shared_begin], "w"),
                    T.access_ptr(meta[source_offset + source_begin], "r"),
                    128, 1, 128, 1, 128, 0, 0, 0, 0,
                )

    @T.prim_func
    def kernel(
        topk_ptr: T.handle,
        tpe_ptr: T.handle,
        meta_ptr: T.handle,
        dst_ptr: T.handle,
        cu_seqlens_ptr: T.handle,
        experts_to_copy_ptr: T.handle,
        zero_fill_ranges_ptr: T.handle,
        remote_stats_ptr: T.handle,
        alloc_ptr: T.handle,
        group_tokens_ptr: T.handle,
        z_ptr: T.handle,
        local_hist_ptr: T.handle,
        rank: T.int32,
    ):
        topk = T.match_buffer(topk_ptr, (N,), dtype=T.int32)
        tpe = T.match_buffer(tpe_ptr, (E,), dtype=T.int32)
        meta = T.match_buffer(
            meta_ptr, (R * meta_stride,), dtype=T.int32
        )
        dst = T.match_buffer(dst_ptr, (N,), dtype=T.int32)
        cu_seqlens = T.match_buffer(
            cu_seqlens_ptr, (output_groups,), dtype=T.int32
        )
        experts_to_copy = T.match_buffer(
            experts_to_copy_ptr, (R * epn,), dtype=T.int32
        )
        zero_fill_ranges = T.match_buffer(
            zero_fill_ranges_ptr, (output_groups * 2,), dtype=T.int32
        )
        remote_stats = T.match_buffer(
            remote_stats_ptr, (2,), dtype=T.int32
        )
        alloc = T.match_buffer(alloc_ptr, (R * E,), dtype=T.int32)
        group_tokens = T.match_buffer(
            group_tokens_ptr, (R,), dtype=T.int32
        )
        z = T.match_buffer(z_ptr, (R * R,), dtype=T.int32)
        local_hist = T.match_buffer(
            local_hist_ptr, (num_vblocks * E,), dtype=T.int32
        )

        with T.Kernel(num_sms, threads=threads) as bx:
            tx = T.get_thread_binding()
            local_meta = rank * meta_stride
            gtid = bx * threads + tx
            grid_threads = num_sms * threads
            warp = tx // 32
            route_shared = T.alloc_shared(
                (shared_ints,), T.int32, scope="shared"
            )
            phase_d_stage = T.alloc_shared(
                (_PHASE_D_STAGE_INTS,), T.int32, scope="shared"
            )
            # Replicate each rank's TPE row so every rank reads a local copy.
            global_warp = bx * num_warps + warp
            total_warps = num_sms * num_warps
            lane = tx % 32
            tpe_chunks = tpe_vector_words // 32
            if tpe_chunks > 0:
                for publish_chunk in T.serial(
                    global_warp, R * tpe_chunks, total_warps
                ):
                    destination = publish_chunk // tpe_chunks
                    expert_base = (
                        publish_chunk - destination * tpe_chunks
                    ) * 32
                    fast_bulk_store_128_ref(
                        meta[
                            destination * meta_stride
                            + TPE_OFF
                            + rank * E
                            + expert_base
                        ],
                        T.cast(tpe[expert_base + lane], T.uint32),
                        T.cast(destination != rank, T.bool),
                    )
            tpe_tail = E - tpe_vector_words
            if tpe_tail > 0:
                for publish_tail in T.serial(
                    gtid, R * tpe_tail, grid_threads
                ):
                    destination = publish_tail // tpe_tail
                    expert = (
                        tpe_vector_words
                        + publish_tail
                        - destination * tpe_tail
                    )
                    T.atomic_store(
                        meta[
                            T.cast(destination, T.int64) * meta_stride
                            + TPE_OFF
                            + rank * E
                            + expert
                        ],
                        tpe[expert],
                        memory_order="relaxed",
                        scope="system",
                    )
            for i in T.serial(gtid, NvS, grid_threads):
                meta[local_meta + SRC_INFO_OFF + i] = -1

            for i in T.serial(gtid, R * E, grid_threads):
                alloc[i] = 0
            for i in T.serial(gtid, R, grid_threads):
                group_tokens[i] = 0
            for i in T.serial(gtid, R * R, grid_threads):
                z[i] = 0
            for i in T.serial(gtid, E, grid_threads):
                experts_to_copy[i] = -1
            if gtid < 2:
                remote_stats[gtid] = 0
            for i in T.serial(gtid, R * 2, grid_threads):
                meta[local_meta + PLAN_OFF + stats_offset + i] = 0
            planning_phase = T.alloc_shared((1,), T.int32)
            if parallel_planning:
                if gtid < 4:
                    meta[local_meta + sort_barrier_offset + gtid] = 0
                if tx == 0:
                    planning_phase[0] = 0
            barrier_state = T.alloc_shared((1,), T.int32)
            barrier_phase = T.alloc_shared((1,), T.int32)
            barrier_sign = T.alloc_shared((1,), T.int32)
            barrier_target = T.alloc_shared((1,), T.int32)
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
            if parallel_planning:
                if bx < core_ctas:
                    build_core_plan(
                        meta, alloc, group_tokens, z, experts_to_copy,
                        local_hist, route_shared, planning_phase, rank, bx, tx,
                    )
                else:
                    build_local_route_order(
                        topk, tpe, meta, local_hist, planning_phase, route_shared,
                        local_meta + ORDER_OFF, bx - core_ctas, tx,
                    )
                T.sync_grid()
            else:
                build_core_plan(
                    meta, alloc, group_tokens, z, experts_to_copy,
                    local_hist, route_shared, planning_phase, rank, bx, tx,
                )
                # Finish consuming destination hints before sorting reuses
                # local_hist for per-tile expert counts.
                T.sync_grid()
                build_local_route_order(
                    topk, tpe, meta, local_hist, planning_phase, route_shared,
                    local_meta + ORDER_OFF, bx, tx,
                )
            group_begin = bx * phase_d_group_tile
            group_end = T.min(group_begin + phase_d_group_tile, output_groups)
            copy_begin = bx * phase_d_copy_tile
            copy_end = T.min(copy_begin + phase_d_copy_tile, R * epn)
            if phase_d_use_aiu:
                if warp == 0:
                    stage_plan_slice(
                        meta, phase_d_stage, 0,
                        local_meta + PLAN_OFF + cu_seqlens_offset
                        + rank * output_groups,
                        group_begin, output_groups, 32,
                    )
                    stage_plan_slice(
                        meta, phase_d_stage, 32,
                        local_meta + PLAN_OFF + zero_fill_offset
                        + rank * output_groups * 2,
                        group_begin * 2, output_groups * 2, 64,
                    )
                    stage_plan_slice(
                        meta, phase_d_stage, 96,
                        local_meta + PLAN_OFF + experts_to_copy_offset,
                        copy_begin, R * epn, 32,
                    )
                    T.ptx_commit_group()

            build_expert_prefix(tpe, route_shared, tx)

            for order_position in T.serial(
                gtid, N, grid_threads
            ):
                offv = meta[local_meta + ORDER_OFF + order_position]
                expert = topk[offv]
                destination = T.alloc_local((1,), T.int32)
                local_offset = T.alloc_local((1,), T.int32)
                direct_base = T.alloc_local((1,), T.int32)
                direct_base[0] = meta[
                    local_meta + TPE_OFF + rank * E + expert
                ]
                if direct_base[0] >= 0:
                    destination[0] = direct_base[0] // NvS
                    local_offset[0] = (
                        direct_base[0]
                        - destination[0] * NvS
                        + order_position
                        - route_shared[expert]
                    )
                else:
                    previous_source_tokens = T.alloc_local((1,), T.int32)
                    previous_source_tokens[0] = 0
                    if rank > 0:
                        previous_source_tokens[0] = meta[
                            local_meta
                            + PLAN_OFF
                            + source_token_prefix_offset
                            + (rank - 1) * E
                            + expert
                        ]
                    global_index = (
                        previous_source_tokens[0]
                        + order_position
                        - route_shared[expert]
                    )
                    destination_lo = T.alloc_local((1,), T.int32)
                    destination_hi = T.alloc_local((1,), T.int32)
                    previous_allocation = T.alloc_local((1,), T.int32)
                    destination_lo[0] = 0
                    destination_hi[0] = R
                    previous_allocation[0] = 0
                    for _step in T.unroll(rank_search_steps):
                        middle = (
                            destination_lo[0] + destination_hi[0]
                        ) // 2
                        cumulative = T.alloc_local((1,), T.int32)
                        cumulative[0] = meta[
                            local_meta
                            + PLAN_OFF
                            + allocation_prefix_offset
                            + expert * R
                            + middle
                        ]
                        if cumulative[0] > global_index:
                            destination_hi[0] = middle
                        else:
                            destination_lo[0] = middle + 1
                            previous_allocation[0] = cumulative[0]
                    destination[0] = destination_lo[0]
                    local_offset[0] = (
                        meta[
                            local_meta
                            + PLAN_OFF
                            + expert_offsets_offset
                            + destination[0] * E
                            + expert
                        ]
                        + global_index
                        - previous_allocation[0]
                    )
                dst[offv] = destination[0] * NvS + local_offset[0]
                T.atomic_store(
                    meta[
                        destination[0] * meta_stride
                        + SRC_INFO_OFF
                        + local_offset[0]
                    ],
                    T.cast(rank * NvS + offv, T.int32),
                    memory_order="relaxed",
                    scope="system",
                )

            if phase_d_use_aiu:
                if warp == 0:
                    T.ptx_wait_group(0)
                T.sync_threads()

                for group in T.serial(
                    group_begin + tx, group_end, threads
                ):
                    local_group = group - group_begin
                    if group_begin + 32 <= output_groups:
                        cu_seqlens[group] = phase_d_stage[local_group]
                        zero_fill_ranges[group * 2] = phase_d_stage[
                            32 + local_group * 2
                        ]
                        zero_fill_ranges[group * 2 + 1] = phase_d_stage[
                            32 + local_group * 2 + 1
                        ]
                    else:
                        cu_seqlens[group] = meta[
                            local_meta + PLAN_OFF + cu_seqlens_offset
                            + rank * output_groups + group
                        ]
                        zero_fill_ranges[group * 2] = meta[
                            local_meta + PLAN_OFF + zero_fill_offset
                            + (rank * output_groups + group) * 2
                        ]
                        zero_fill_ranges[group * 2 + 1] = meta[
                            local_meta + PLAN_OFF + zero_fill_offset
                            + (rank * output_groups + group) * 2 + 1
                        ]
                for index in T.serial(
                    copy_begin + tx, copy_end, threads
                ):
                    if copy_begin + 32 <= R * epn:
                        experts_to_copy[index] = phase_d_stage[
                            96 + index - copy_begin
                        ]
                    else:
                        experts_to_copy[index] = meta[
                            local_meta + PLAN_OFF + experts_to_copy_offset
                            + index
                        ]
            else:
                for group in T.serial(gtid, output_groups, grid_threads):
                    cu_seqlens[group] = meta[
                        local_meta
                        + PLAN_OFF
                        + cu_seqlens_offset
                        + rank * output_groups
                        + group
                    ]
                    zero_fill_ranges[group * 2] = meta[
                        local_meta
                        + PLAN_OFF
                        + zero_fill_offset
                        + (rank * output_groups + group) * 2
                    ]
                    zero_fill_ranges[group * 2 + 1] = meta[
                        local_meta
                        + PLAN_OFF
                        + zero_fill_offset
                        + (rank * output_groups + group) * 2
                        + 1
                    ]
                for index in T.serial(gtid, R * epn, grid_threads):
                    experts_to_copy[index] = meta[
                        local_meta
                        + PLAN_OFF
                        + experts_to_copy_offset
                        + index
                    ]
            if gtid < 2:
                remote_stats[gtid] = meta[
                    local_meta + PLAN_OFF + stats_offset + rank * 2 + gtid
                ]

            # Ensure every rank has published src_info before dispatch can
            # consume it.
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
            tokens_per_block = (S + num_sms - 1) // num_sms
            token_begin = bx * tokens_per_block
            token_end = T.min(token_begin + tokens_per_block, S)
            for token in T.serial(
                token_begin + tx, token_end, threads
            ):
                for k in T.serial(K):
                    index = token * K + k
                    raw = dst[index]
                    destination = raw // NvS
                    duplicate = T.alloc_local((1,), T.int32)
                    duplicate[0] = 0
                    for prior in T.serial(k):
                        prior_value = dst[token * K + prior]
                        prior_raw = T.if_then_else(
                            prior_value < 0,
                            -prior_value - 1,
                            prior_value,
                        )
                        if prior_raw // NvS == destination:
                            duplicate[0] = 1
                    if duplicate[0] != 0:
                        dst[index] = -raw - 1

    with torch.cuda.device(torch.cuda.current_device()):
        return tilelang.compile(
            kernel,
            target="ppu",
            compile_flags=["--dopt=3"],
            execution_backend="tvm_ffi",
        )


def launch_planning(
    ctx: dict,
    topk_experts_flat: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    cu_seqlens: torch.Tensor,
    plan,
) -> None:
    R = int(ctx["R"])
    E = int(ctx["E"])
    meta = ctx["meta_buf"]
    with torch.cuda.device(meta.device):
        compiled = _compile_planning(
            R=R,
            E=E,
            S=int(ctx["S"]),
            K=int(ctx["K"]),
            NvS_capacity=int(ctx["NvS_capacity"]),
            NvS=int(ctx["NvS"]),
            meta_stride=int(ctx["meta_chunk_padded"]),
            TPE_OFF=int(ctx["TPE_OFF"]),
            PLAN_OFF=int(ctx["PLAN_OFF"]),
            ORDER_OFF=int(ctx["ORDER_OFF"]),
            BARRIER_OFF=int(ctx["BARRIER_OFF"]),
            SRC_INFO_OFF=int(ctx["SRC_INFO_OFF"]),
            token_padding=int(ctx["token_padding"]),
            num_sms=int(ctx["num_sms"]),
        )
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            compiled(
                topk_experts_flat,
                tokens_per_expert,
                meta,
                plan.dst,
                cu_seqlens,
                plan.experts_to_copy.reshape(-1),
                plan.zero_fill_ranges.reshape(-1),
                plan.remote_stats,
                ctx["alloc"],
                ctx["group_tokens"],
                ctx["z"],
                ctx["local_hist"],
                int(ctx["rank"]),
            )
