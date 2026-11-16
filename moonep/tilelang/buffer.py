import math
import os
import shutil
import socket
import struct
import tempfile
from contextlib import contextmanager
from typing import NamedTuple

import torch
import torch.distributed as dist

from moonep._C import (
    FABRIC_HANDLE_BYTES as _FABRIC_HANDLE_BYTES,
    icn_dist_alloc,
    icn_dist_map,
    icn_fabric_supported,
    icn_release_mem_handle,
    get_vmm_granularity,
)

# How VMM allocations are shared between the ranks of an EP group. "auto" (the
# default) picks fabric handles when every rank supports creating them and
# falls back to same-node POSIX fds otherwise; "fabric" / "fd" force one.
_HANDLE_TYPE_ENV = "MOONEP_MEM_HANDLE_TYPE"


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _gather_objects(value, group: dist.ProcessGroup | None) -> list:
    values = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(values, value, group=group)
    return values


def _check_errors(error: Exception | None, phase: str, group) -> None:
    errors = _gather_objects(
        None if error is None else f"{type(error).__name__}: {error}", group
    )
    failures = [f"rank {rank}: {message}" for rank, message in enumerate(errors)
                if message is not None]
    if failures:
        raise RuntimeError(f"VMM {phase} failed: " + "; ".join(failures)) from error


@contextmanager
def _collective_step(phase: str, group):
    """Share local setup failures; all ranks must enter steps in the same order."""
    error = None
    try:
        yield
    except Exception as exc:
        error = exc
    _check_errors(error, phase, group)


class _AllocationRequest(NamedTuple):
    shape: tuple[int, ...]
    dtype: str
    mode: str
    fabric_supported: bool
    hostname: str
    error: str | None


def _validate_allocation_and_select_fabric(
    chunk_shape, dtype, local_rank, world_size, group,
) -> bool:
    mode = os.environ.get(_HANDLE_TYPE_ENV, "auto").lower()
    error = None
    supported = False
    try:
        if mode not in ("auto", "fabric", "fd"):
            raise ValueError(f"invalid {_HANDLE_TYPE_ENV}: {mode!r}")
        if (local_rank, world_size) != (dist.get_rank(group), dist.get_world_size(group)):
            raise ValueError("allocation rank/world_size does not match the process group")
        supported = mode != "fd" and bool(icn_fabric_supported())
    except Exception as exc:
        error = exc
    rows = _gather_objects(
        _AllocationRequest(
            tuple(chunk_shape), str(dtype), mode, supported, socket.gethostname(),
            None if error is None else f"{type(error).__name__}: {error}",
        ),
        group,
    )
    failures = [f"rank {rank}: {row.error}" for rank, row in enumerate(rows) if row.error]
    if failures:
        raise RuntimeError("VMM allocation validation failed: " + "; ".join(failures)) from error
    layouts = [(row.shape, row.dtype, row.mode) for row in rows]
    if any(layout != layouts[0] for layout in layouts):
        raise RuntimeError(
            "VMM chunk shape, dtype and handle mode must agree across ranks: "
            + repr(layouts)
        )
    unsupported = [rank for rank, row in enumerate(rows) if not row.fabric_supported]
    cross_node = len({row.hostname for row in rows}) > 1

    if mode == "fd":
        if cross_node:
            raise RuntimeError(
                f"{_HANDLE_TYPE_ENV}=fd cannot share VMM allocations across nodes"
            )
        return False

    if mode == "fabric":
        if unsupported:
            raise RuntimeError(
                f"{_HANDLE_TYPE_ENV}=fabric, but fabric memory handles are "
                f"unsupported on group ranks {unsupported}."
            )
        return True

    if unsupported and cross_node:
        raise RuntimeError(
            "cross-node MoonEP requires fabric memory handles, but they are "
            f"unsupported on group ranks {unsupported}"
        )
    return not unsupported


def _all_gather_shareables(
    local_handle: torch.Tensor,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Gather every rank's fabric handle into a uint8[world_size, 64] CPU tensor."""
    world_size = dist.get_world_size(group=group)
    with _collective_step("fabric exchange setup", group):
        local = local_handle.to(device=_device()).view(1, _FABRIC_HANDLE_BYTES)
        gathered = torch.empty(world_size, _FABRIC_HANDLE_BYTES,
                               dtype=torch.uint8, device=_device())
    dist.all_gather_into_tensor(
        gathered, local, group=group,
    )
    with _collective_step("fabric exchange download", group):
        result = gathered.cpu()
    return result


def pad_to_granularity(nbytes: int) -> int:
    """Round up nbytes to VMM granularity."""
    gran = get_vmm_granularity()
    return ((nbytes + gran - 1) // gran) * gran


def pad_dim0_for_alignment(chunk_shape: list[int], dtype: torch.dtype) -> int:
    """Compute the padded dim0 so that chunk bytes are aligned to VMM granularity.

    Returns the padded dim0 value (>= chunk_shape[0]).
    """
    if not chunk_shape or chunk_shape[0] < 0 or any(d <= 0 for d in chunk_shape[1:]):
        raise ValueError("shape must have a nonnegative dim0 and positive inner dimensions")
    row_bytes = dtype.itemsize * math.prod(chunk_shape[1:])
    granularity = get_vmm_granularity()
    row_alignment = granularity // math.gcd(row_bytes, granularity)
    return ((chunk_shape[0] + row_alignment - 1) // row_alignment) * row_alignment


def _exchange_ipc_fds(
    local_fd: int,
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup | None,
) -> dict[int, int]:
    """Pass POSIX fds between ranks via per-rank unix datagram sockets.

    Each rank passes its exported ``local_fd`` and receives one fd from
    every rank; the kernel dups the fd into the receiving process
    (SCM_RIGHTS), so the returned fds are owned by this process and must be
    closed by the caller after import. The sender's own fd may be closed as
    soon as this function returns. All ranks agree on success before returning.
    """
    # Group-rank 0 creates a shared dir; broadcast the path to the group.
    dir_path = None
    error = None
    if local_rank == 0:
        try:
            dir_path = tempfile.mkdtemp(prefix="moonep_ipc_")
        except Exception as exc:
            error = str(exc)
    obj = [dir_path, error]
    src = dist.get_global_rank(group, 0) if group is not None else 0
    dist.broadcast_object_list(obj, src=src, group=group)
    dir_path, error = obj
    if error is not None:
        raise RuntimeError(f"VMM IPC directory creation failed: {error}")
    sock = None
    fds = {}
    success = False
    try:
        with _collective_step("IPC socket setup", group):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            sock.settimeout(120)
            sock.bind(os.path.join(dir_path, f"rank_{local_rank}"))
            if local_fd is None:
                raise ValueError("IPC sender has no file descriptor")

        with _collective_step("IPC transfer", group):
            # Receive each sender before moving on to avoid filling socket queues.
            for sender in range(world_size):
                if local_rank == sender:
                    payload = struct.pack("<i", local_rank)
                    anc = [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                            struct.pack("<i", local_fd))]
                    for dst in range(world_size):
                        sock.sendmsg([payload], anc, 0,
                                     os.path.join(dir_path, f"rank_{dst}"))
                while sender not in fds:
                    msg, ancdata, flags, _addr = sock.recvmsg(16, socket.CMSG_SPACE(4))
                    received = []
                    for level, ctype, cdata in ancdata:
                        if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
                            received.extend(struct.unpack(f"{len(cdata) // 4}i", cdata))
                    src_rank = struct.unpack("<i", msg)[0] if len(msg) == 4 else -1
                    if (flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC)
                            or not 0 <= src_rank < world_size or src_rank in fds
                            or len(received) != 1):
                        for fd in received:
                            os.close(fd)
                        raise RuntimeError("invalid IPC file descriptor message")
                    # A faster peer may already have sent its next-round message.
                    fds[src_rank] = received[0]
        success = True
        return fds
    finally:
        if not success:
            for fd in fds.values():
                os.close(fd)
        if sock is not None:
            sock.close()
        if local_rank == 0:
            shutil.rmtree(dir_path, ignore_errors=True)


def _map_icn_dist_tensor(
    chunk_shape: list[int],
    dtype: torch.dtype,
    shareable: torch.Tensor,
    keepalive: torch.Tensor,
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup | None,
    use_fabric: bool,
) -> torch.Tensor:
    fds = {}
    if use_fabric:
        shareables = _all_gather_shareables(shareable, group)
    else:
        fds = _exchange_ipc_fds(int(shareable.item()), local_rank, world_size, group)
    try:
        # Keep exported physical handles alive until every peer finishes importing.
        with _collective_step("mapping", group):
            if not use_fabric:
                shareables = torch.tensor([fds[r] for r in range(world_size)],
                                          dtype=torch.int64)
            mapped = icn_dist_map(
                chunk_shape=chunk_shape,
                dtype=dtype,
                shareables=shareables,
                local_rank=local_rank,
                world_size=world_size,
                use_fabric=use_fabric,
            )
            mapped._keepalive = keepalive
    finally:
        for fd in fds.values():
            os.close(fd)
    return mapped


def create_icn_dist_tensor(
    chunk_shape: list[int],
    dtype: torch.dtype,
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Allocate an ICN distributed tensor with all-RW access.

    chunk_shape MUST already be padded to VMM granularity alignment.
    Use pad_dim0_for_alignment() to compute the padded dim0.

    `local_rank` and `world_size` must match the given `group` (or the default
    group when `group is None`). All ranks in the group exchange memory handles.
    """
    use_fabric = _validate_allocation_and_select_fabric(
        chunk_shape, dtype, local_rank, world_size, group,
    )
    allocation = None
    try:
        with _collective_step("allocation", group):
            allocation = icn_dist_alloc(shape=chunk_shape, dtype=dtype, use_fabric=use_fabric)
        keepalive, shareable, owned_handle = allocation
        return _map_icn_dist_tensor(
            chunk_shape, dtype, shareable, keepalive,
            local_rank, world_size, group, use_fabric,
        )
    finally:
        # Attempt both releases before sharing errors, even if one release fails.
        error = None
        if allocation is not None:
            try:
                if not use_fabric:
                    os.close(int(allocation[1].item()))
            except Exception as exc:
                error = exc
            try:
                icn_release_mem_handle(allocation[2])
            except Exception as exc:
                error = exc
        _check_errors(error, "handle release", group)
