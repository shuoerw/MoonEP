"""PPU rank-mapped communication buffers used internally by MoonEP kernels.

Expert weights and gradients remain caller-owned local tensors, with
rank-strided distributed pools for prefetch and gradient reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import weakref
import torch
import torch.distributed as dist

from . import buffer as _vmm


@dataclass(slots=True)
class _RankMapping:
    key: int
    tensor: weakref.ReferenceType[torch.Tensor]
    base: int
    nbytes: int
    rank_stride: int
    local_rank: int
    world_size: int


_MAPPING_LOCK = threading.Lock()
_NEXT_MAPPING_KEY = 0
_MAPPINGS: dict[int, _RankMapping] = {}


pad_to_granularity = _vmm.pad_to_granularity


def pad_dim0_for_alignment(shape: list[int], dtype: torch.dtype) -> int:
    row_bytes = torch.empty((), dtype=dtype).element_size()
    for dimension in shape[1:]:
        row_bytes *= dimension
    padded_rows = pad_to_granularity(shape[0] * row_bytes) // row_bytes
    granularity = int(_vmm.get_vmm_granularity())
    while padded_rows * row_bytes % granularity:
        padded_rows += 1
    return padded_rows


def get_comm_alignment(_world_size: int) -> int:
    return int(_vmm.get_vmm_granularity())


def _register_mapping(
    tensor: torch.Tensor,
    *,
    rank_stride: int,
    local_rank: int,
    world_size: int,
) -> torch.Tensor:
    global _NEXT_MAPPING_KEY

    with _MAPPING_LOCK:
        key = _NEXT_MAPPING_KEY
        _NEXT_MAPPING_KEY += 1

        def unregister(_reference, *, mapping_key=key):
            with _MAPPING_LOCK:
                _MAPPINGS.pop(mapping_key, None)

        _MAPPINGS[key] = _RankMapping(
            key=key,
            tensor=weakref.ref(tensor, unregister),
            base=int(tensor.data_ptr()),
            nbytes=int(tensor.numel()) * int(tensor.element_size()),
            rank_stride=int(rank_stride),
            local_rank=int(local_rank),
            world_size=int(world_size),
        )
    return tensor


def mapping_info(tensor: torch.Tensor) -> dict:
    address = int(tensor.data_ptr())
    with _MAPPING_LOCK:
        mappings = tuple(_MAPPINGS.values())
    for mapping in mappings:
        if mapping.tensor() is None:
            continue
        if mapping.base <= address < mapping.base + mapping.nbytes:
            return {
                "rank_mapped": True,
                "mapping_key": mapping.key,
                "world_size": mapping.world_size,
                "local_rank": mapping.local_rank,
                "rank": mapping.local_rank,
                "rank_stride": mapping.rank_stride,
                "base": mapping.base,
                "offset_bytes": address - mapping.base,
                "bytes": mapping.nbytes,
            }
    return {"rank_mapped": False}


def create_rank_mapped_tensor(
    shape: list[int],
    dtype: torch.dtype,
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before VMM mapping")
    actual_rank = dist.get_rank(group=group)
    actual_world_size = dist.get_world_size(group=group)
    if (local_rank, world_size) != (actual_rank, actual_world_size):
        raise ValueError(
            "rank-mapped allocation does not match the process group: "
            f"got {local_rank}/{world_size}, expected "
            f"{actual_rank}/{actual_world_size}"
        )

    full = _vmm.create_icn_dist_tensor(
        shape, dtype, local_rank, world_size, group=group
    )
    rank_stride = int(full.numel() // world_size) * int(full.element_size())
    return _register_mapping(
        full,
        rank_stride=rank_stride,
        local_rank=local_rank,
        world_size=world_size,
    )


create_hidden_buffer = create_rank_mapped_tensor
create_icn_dist_tensor = create_rank_mapped_tensor


def create_meta_buffers(
    shape: list[int],
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = create_rank_mapped_tensor(
        shape, dtype, rank, world_size, group=group
    )
    return tensor, tensor


def local_rows(
    tensor: torch.Tensor,
    rank: int,
    padded_rows: int,
    logical_rows: int,
) -> torch.Tensor:
    begin = rank * padded_rows
    return tensor[begin : begin + logical_rows]


def local_flat_region(
    tensor: torch.Tensor,
    rank: int,
    padded_elements: int,
    offset: int,
    length: int,
) -> torch.Tensor:
    begin = rank * padded_elements + offset
    return tensor[begin : begin + length]


def physical_nbytes(tensor: torch.Tensor, _world_size: int) -> int:
    info = mapping_info(tensor)
    if info.get("rank_mapped", False) and int(info["rank_stride"]) > 0:
        return int(info["rank_stride"])
    return int(tensor.numel()) * int(tensor.element_size())


def record_stream(
    tensor: torch.Tensor,
    stream: torch.cuda.Stream,
) -> None:
    if not mapping_info(tensor).get("rank_mapped", False):
        tensor.record_stream(stream)


def release_mapped_tensor(tensor: torch.Tensor) -> None:
    info = mapping_info(tensor)
    if not info.get("rank_mapped", False):
        return
    key = int(info["mapping_key"])
    with _MAPPING_LOCK:
        _MAPPINGS.pop(key, None)


def release_comm_buffers(
    hidden: torch.Tensor,
    meta: torch.Tensor,
    group: dist.ProcessGroup | None = None,
) -> None:
    del group
    release_mapped_tensor(hidden)
    release_mapped_tensor(meta)
