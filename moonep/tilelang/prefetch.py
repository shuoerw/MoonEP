"""TileLang implementation of owner-push expert prefetch on PPU."""

from __future__ import annotations

import math

import torch

from .dma import PrefetchDMAContext


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
    dma_context: PrefetchDMAContext,
) -> None:
    local_views = tuple(local.view(torch.uint8).reshape(-1) for local, _ in pairs)
    pool_views = tuple(_pool_byte_view(pool) for _, pool in pairs)
    row_bytes = math.prod(int(value) for value in pairs[0][0].shape[1:])
    row_bytes *= pairs[0][0].element_size()
    dma_context.copy(local_views, pool_views, experts_to_copy, row_bytes, rank)


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
    dma_context: PrefetchDMAContext,
    scales: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
) -> None:
    """Push expert rows with DMA, then complete the inter-rank barrier."""
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
    # match. Each queue submits the three matrices for each selected row.
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
            tuple(pairs[:3]), experts_to_copy, rank=rank, dma_context=dma_context
        )
    for pair in pairs[first_separate:]:
        _copy_tensors((pair,), experts_to_copy, rank=rank, dma_context=dma_context)

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
