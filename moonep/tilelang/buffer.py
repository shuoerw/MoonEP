import hashlib
import os
import shutil
import socket
import struct
import tempfile

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


def _use_fabric_for_group(group: dist.ProcessGroup | None) -> bool:
    mode = os.environ.get(_HANDLE_TYPE_ENV, "auto").lower()
    assert mode in ("auto", "fabric", "fd"), (
        f"{_HANDLE_TYPE_ENV} must be one of auto/fabric/fd, got {mode!r}"
    )
    supported = mode != "fd" and bool(icn_fabric_supported())
    world_size = (dist.get_world_size(group=group)
                  if dist.is_available() and dist.is_initialized() else 1)
    if world_size == 1:
        unsupported = [] if supported else [0]
        cross_node = False
    else:
        host_tag = int.from_bytes(
            hashlib.sha256(socket.gethostname().encode()).digest()[:8],
            "little",
        ) & ((1 << 63) - 1)
        local = torch.tensor(
            [int(supported), host_tag], dtype=torch.int64, device=_device()
        )
        gathered = torch.empty(
            world_size, 2, dtype=torch.int64, device=_device()
        )
        dist.all_gather_into_tensor(gathered, local, group=group)
        rows = gathered.cpu().tolist()
        unsupported = [rank for rank, row in enumerate(rows) if row[0] == 0]
        cross_node = len({row[1] for row in rows}) > 1

    if mode == "fd":
        if cross_node:
            raise RuntimeError(
                f"{_HANDLE_TYPE_ENV}=fd cannot share VMM allocations across nodes"
            )
        return False

    if mode == "fabric":
        assert not unsupported, (
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
    gathered = torch.empty(world_size, _FABRIC_HANDLE_BYTES,
                           dtype=torch.uint8, device=_device())
    dist.all_gather_into_tensor(
        gathered,
        local_handle.to(device=_device()).view(1, _FABRIC_HANDLE_BYTES),
        group=group,
    )
    return gathered.cpu()


def pad_to_granularity(nbytes: int) -> int:
    """Round up nbytes to VMM granularity."""
    gran = get_vmm_granularity()
    return ((nbytes + gran - 1) // gran) * gran


def pad_dim0_for_alignment(chunk_shape: list[int], dtype: torch.dtype) -> int:
    """Compute the padded dim0 so that chunk bytes are aligned to VMM granularity.

    Returns the padded dim0 value (>= chunk_shape[0]).
    """
    inner_size = dtype.itemsize
    for d in chunk_shape[1:]:
        inner_size *= d  # bytes per row

    nbytes = chunk_shape[0] * inner_size
    padded_bytes = pad_to_granularity(nbytes)
    padded_dim0 = padded_bytes // inner_size
    # Ensure exact alignment
    while padded_dim0 * inner_size % get_vmm_granularity() != 0:
        padded_dim0 += 1
    return padded_dim0


def _exchange_ipc_fds(
    local_fd: int | None,
    sender_ranks: list[int],
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup | None,
) -> dict[int, int]:
    """Pass POSIX fds between ranks via per-rank unix datagram sockets.

    Ranks listed in ``sender_ranks`` must pass their exported fd as
    ``local_fd`` (other ranks pass None). Every rank receives one fd from
    each sender; the kernel dups the fd into the receiving process
    (SCM_RIGHTS), so the returned fds are owned by this process and must be
    closed by the caller after import. The sender's own fd may be closed as
    soon as this function returns — a trailing barrier guarantees all peers
    have already received their copy.
    """
    # Group-rank 0 creates a shared dir; broadcast the path to the group.
    if local_rank == 0:
        dir_path = tempfile.mkdtemp(prefix="moonep_ipc_")
    else:
        dir_path = None
    obj = [dir_path]
    src = dist.get_global_rank(group, 0) if group is not None else 0
    dist.broadcast_object_list(obj, src=src, group=group)
    dir_path = obj[0]

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(os.path.join(dir_path, f"rank_{local_rank}"))
    sock.settimeout(120)
    # All sockets must be bound before any send.
    dist.barrier(group=group)

    try:
        if local_fd is not None:
            payload = struct.pack("<i", local_rank)
            anc = [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                    struct.pack("<i", local_fd))]
            for dst in range(world_size):
                sock.sendmsg([payload], anc, 0,
                             os.path.join(dir_path, f"rank_{dst}"))

        fds = {}
        try:
            while len(fds) < len(sender_ranks):
                msg, ancdata, _flags, _addr = sock.recvmsg(16, socket.CMSG_SPACE(4))
                src_rank = struct.unpack("<i", msg[:4])[0]
                for level, ctype, cdata in ancdata:
                    if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
                        fds[src_rank] = struct.unpack("<i", cdata[:4])[0]
                        break
                else:
                    raise RuntimeError("received IPC message without an fd")
        except Exception:
            # Closing the socket does not release descriptors received via
            # SCM_RIGHTS; this process owns those descriptors separately.
            for fd in fds.values():
                os.close(fd)
            raise
    finally:
        sock.close()
        # Everyone has received their fds; safe to tear down the sockets and
        # for senders to close their original fd.
        dist.barrier(group=group)
        if local_rank == 0:
            shutil.rmtree(dir_path, ignore_errors=True)
    return fds


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
    if use_fabric:
        shareables = _all_gather_shareables(shareable, group)
        mapped = icn_dist_map(
            chunk_shape=chunk_shape,
            dtype=dtype,
            shareables=shareables,
            local_rank=local_rank,
            world_size=world_size,
            use_fabric=True,
        )
    else:
        local_fd = int(shareable.item())
        try:
            fds = _exchange_ipc_fds(local_fd, list(range(world_size)),
                                    local_rank, world_size, group)
        finally:
            os.close(local_fd)
        all_fds = [fds[r] for r in range(world_size)]
        try:
            mapped = icn_dist_map(
                chunk_shape=chunk_shape,
                dtype=dtype,
                shareables=torch.tensor(all_fds, dtype=torch.int64),
                local_rank=local_rank,
                world_size=world_size,
                use_fabric=False,
            )
        finally:
            for fd in all_fds:
                os.close(fd)
    mapped._keepalive = keepalive
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
    use_fabric = _use_fabric_for_group(group)
    keepalive, shareable, owned_handle = icn_dist_alloc(
        shape=chunk_shape, dtype=dtype, use_fabric=use_fabric)
    try:
        return _map_icn_dist_tensor(
            chunk_shape, dtype, shareable, keepalive,
            local_rank, world_size, group, use_fabric,
        )
    finally:
        icn_release_mem_handle(owned_handle)
