"""Buffer-owned OKI queues and the device-issued prefetch copy primitive."""

import atexit
import ctypes
from functools import lru_cache
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
import warnings

import torch
import torch.distributed as dist

from ._compile_cache import CompileCache


_QUEUE_DEPTH = 32768
_QUEUE_PAIRS = 8
_DIES = 2
# OKI copy WQEs have a 30-bit byte count. Keep chunk boundaries 4-byte aligned.
_COPY_CHUNK_BYTES = (1 << 30) - 4
_HGGC_ERROR_INVALID_VALUE = 1
_DEVICES = {}
_DEVICE_LOCK = threading.Lock()
_compile_cached = CompileCache()


@lru_cache(maxsize=None)
def _load_oki(sdk: str):
    # OKI's host library references the native runtime even when PyTorch uses
    # its CUDA-compatible interface. Keep both libraries alive with the queues.
    runtime = ctypes.CDLL(f"{sdk}/lib/libhggcrt1.so", mode=ctypes.RTLD_GLOBAL)
    runtime.hggcGetLastError.argtypes = []
    runtime.hggcGetLastError.restype = ctypes.c_int
    library = ctypes.CDLL(f"{sdk}/lib/liblt_oki.so")
    pointer = ctypes.c_void_p
    signatures = {
        "ltOkiOpenDevice": [ctypes.POINTER(pointer), ctypes.c_int],
        "ltOkiCloseDevice": [pointer],
        "ltOkiCreateQP": [ctypes.POINTER(pointer), pointer, ctypes.c_bool,
                          ctypes.c_int, ctypes.c_int],
        "ltOkiDestroyQP": [pointer, pointer],
    }
    for name, arguments in signatures.items():
        function = getattr(library, name)
        function.argtypes = arguments
        function.restype = ctypes.c_int
    return runtime, library


def _close_device(runtime, library, handle) -> None:
    status = library.ltOkiCloseDevice(handle)
    runtime_status = runtime.hggcGetLastError()
    if status or runtime_status:
        warnings.warn(f"OKI device cleanup failed: SDK={status}, runtime={runtime_status}",
                      RuntimeWarning)


def _get_device(sdk: str, device: int):
    # Device sessions are process-scoped. Buffers release their own QPs, but
    # must not finalize and reopen the SDK session between communication calls.
    with _DEVICE_LOCK:
        key = (sdk, device)
        if key not in _DEVICES:
            runtime, library = _load_oki(sdk)
            handle = ctypes.c_void_p()
            status = library.ltOkiOpenDevice(ctypes.byref(handle), device)
            if status:
                raise RuntimeError(f"OKI open device failed with status {status}")
            _DEVICES[key] = (runtime, library, handle)
            atexit.register(_close_device, runtime, library, handle)
        return _DEVICES[key]


class PrefetchDMAContext:
    """Persistent queues for one Buffer's stream-ordered prefetch operations.

    As with the Buffer's rank barriers, calls and graph replays using the same
    context must not overlap. Queue initialization happens outside capture.
    """

    def __init__(self, ctx: dict):
        self.device = int(ctx["device"])
        self.rank = int(ctx["rank"])
        self.world_size = int(ctx["R"])
        self.sdk = str(Path(os.environ.get("PPU_SDK", "/usr/local/PPU_SDK")).resolve())
        self._device_handle = ctypes.c_void_p()
        self._queues = []
        self._closed = False
        self.handles = self.steps = self.fences = None
        self._library = None
        error = None
        try:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("DMA queues must be initialized before graph capture")
            for relative in ("include/lt_oki.h", "include/lt_oki_device.cuh",
                             "lib/liblt_oki_device_simt.a"):
                if not Path(self.sdk, relative).is_file():
                    raise RuntimeError(f"OKI SDK component is missing: {relative}")
            with torch.cuda.device(self.device):
                self._runtime, self._library, self._device_handle = _get_device(
                    self.sdk, self.device)
                # Check the SDK lifecycle before enabling DMA transfers.
                self._validate_queue_lifecycle()
                # Pair 0 copies to this rank. Pairs 1..7 service remote ranks,
                # reusing a pair only after its previous transfer completes.
                for pair in range(_QUEUE_PAIRS):
                    for die in range(_DIES):
                        queue = ctypes.c_void_p()
                        queue_id = die if pair == 0 else die * 8 + pair - 1
                        self._check(self._library.ltOkiCreateQP(
                            ctypes.byref(queue), self._device_handle, pair != 0,
                            queue_id, _QUEUE_DEPTH), "create queue")
                        self._queues.append(queue)
                self.handles = torch.tensor(
                    [queue.value for queue in self._queues],
                    dtype=torch.uint64, device=self.device,
                )
                self.steps = torch.zeros(len(self._queues), dtype=torch.uint64,
                                         device=self.device)
        except Exception as exc:
            error = str(exc)

        # A local SDK/resource failure must not leave peers entering the VMM
        # allocation or the first communication barrier on their own.
        failures = [None] * self.world_size
        try:
            dist.all_gather_object(failures, error, group=ctx.get("group"))
        except Exception:
            self._release_queues()
            self._closed = True
            raise
        if any(failure is not None for failure in failures):
            cleanup_errors = self._release_queues()
            self._closed = True
            raise RuntimeError(f"MoonEP DMA initialization failed by rank: {failures}; "
                               f"local cleanup errors: {cleanup_errors}")

        from .buffer import create_icn_dist_tensor, pad_to_granularity

        try:
            self.fence_stride = pad_to_granularity(self.world_size * _DIES * 8) // 8
            self.fences = create_icn_dist_tensor(
                [self.fence_stride], torch.uint64, self.rank, self.world_size,
                group=ctx.get("group"),
            )
            begin = self.rank * self.fence_stride
            self.fences[begin:begin + self.fence_stride].zero_()
            torch.cuda.synchronize(self.device)
            dist.barrier(group=ctx.get("group"))
        except Exception:
            self.close()
            raise

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"OKI {operation} failed with status {status}")

    def _validate_queue_lifecycle(self) -> None:
        self._check(self._runtime.hggcGetLastError(), "runtime before queue creation")
        queue = ctypes.c_void_p()
        self._check(self._library.ltOkiCreateQP(
            ctypes.byref(queue), self._device_handle, False, 0, _QUEUE_DEPTH),
            "create validation queue")
        errors = self._destroy_queue(queue)
        if errors:
            raise RuntimeError(f"OKI queue lifecycle validation failed: {errors}")

    def _destroy_queue(self, queue) -> list[str]:
        prior_error = self._runtime.hggcGetLastError()
        status = self._library.ltOkiDestroyQP(self._device_handle, queue)
        runtime_error = self._runtime.hggcGetLastError()
        # SDK compatibility: successful QP destruction may leave InvalidValue.
        # Consume that known error so it cannot poison subsequent runtime calls;
        # preserve prior errors, failed SDK calls, and all other runtime errors.
        if (prior_error == 0 and status == 0
                and runtime_error in (0, _HGGC_ERROR_INVALID_VALUE)):
            return []
        return [f"destroy queue: prior runtime={prior_error}, "
                f"SDK={status}, runtime={runtime_error}"]

    def _release_queues(self) -> list[str]:
        errors = []
        if self._library is None:
            return errors
        for queue in reversed(self._queues):
            errors.extend(self._destroy_queue(queue))
        self._queues.clear()
        self._device_handle = ctypes.c_void_p()
        return errors

    def close(self) -> None:
        if self._closed:
            return
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            errors = self._release_queues()
            self.handles = self.steps = self.fences = None
            self._closed = True
            if errors:
                raise RuntimeError(f"MoonEP DMA cleanup failed: {errors}")

    def copy(self, local_views, pool_views, experts, row_bytes: int, rank: int) -> None:
        if self._closed:
            raise RuntimeError("prefetch DMA context has been destroyed")
        if rank != self.rank or int(experts.shape[0]) != self.world_size:
            raise ValueError("prefetch DMA context does not match the plan's rank group")
        if torch.cuda.current_device() != self.device:
            raise ValueError("prefetch DMA context is on a different device")
        if row_bytes <= 0 or row_bytes % 4:
            raise ValueError("DMA expert rows must have a positive multiple of 4 bytes")
        kernel = _compile_copy(
            self.sdk, self.device, int(local_views[0].numel()),
            int(pool_views[0][0].numel()), pool_views[0][1], row_bytes,
            self.world_size, int(experts.shape[1]), len(local_views), self.fence_stride,
        )
        indices = (0, 1, 2) if len(local_views) == 3 else (0, 0, 0)
        arguments = tuple(tensor for i in indices
                          for tensor in (local_views[i], pool_views[i][0]))
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            kernel(*arguments, experts.reshape(-1), self.handles, self.steps,
                   self.fences, self.rank)


def create_prefetch_context(ctx: dict) -> PrefetchDMAContext:
    return PrefetchDMAContext(ctx)


def _compile_copy(sdk, device, local_bytes, pool_bytes, rank_stride, row_bytes,
                  world_size, slots, weight_count, fence_stride):
    key = (sdk, device, local_bytes, pool_bytes, rank_stride, row_bytes,
           world_size, slots, weight_count, fence_stride)
    return _compile_cached(key, lambda: _build_copy(*key))


def _build_copy(sdk, device, local_bytes, pool_bytes, rank_stride, row_bytes,
                world_size, slots, weight_count, fence_stride):
    import tilelang
    import tilelang.ppu.language as T
    from .primitives import (
        oki_copy_local, oki_copy_remote, oki_fence_local, oki_fence_remote,
        oki_poll, oki_post, oki_reserve, oki_ring,
    )

    sdk_headers = "".join(f"#include {json.dumps(str(header))}\n" for header in (
        Path(sdk, "include/lt_oki.h"), Path(sdk, "include/lt_oki_device.cuh"),
    ))
    peer_rounds = max(1, (world_size - 1 + _QUEUE_PAIRS - 2) // (_QUEUE_PAIRS - 1))
    half_bytes = row_bytes // _DIES
    chunks_per_row = (half_bytes + _COPY_CHUNK_BYTES - 1) // _COPY_CHUNK_BYTES
    # Preserve the copy-WQE budget when one expert row spans several commands.
    batch_slots = 1024 // chunks_per_row
    if batch_slots == 0:
        raise ValueError("DMA expert row exceeds the per-batch WQE budget")
    batch_count = (slots + batch_slots - 1) // batch_slots

    @T.macro
    def enqueue_row(local, pool, queue, src_offset, dst_offset, remote, die):
        for chunk in T.serial(chunks_per_row):
            offset = T.cast(chunk, T.int64) * _COPY_CHUNK_BYTES
            size = T.min(T.int64(half_bytes) - offset, _COPY_CHUNK_BYTES)
            if remote:
                oki_copy_remote(queue, local, pool, src_offset + offset,
                                dst_offset + offset, size)
            else:
                oki_copy_local(queue, local, pool, src_offset + offset,
                               dst_offset + offset, size, die != 0)

    def bind_dma(weights, plan, queues, steps, fences, rank, pair, die):
        """Bind device operands to compile-time macros; no runtime object.

        Every CTA thread calls fetch, commit and wait in order. Each queue pair
        permits one outstanding batch, which must be drained before the next
        fetch. Source data and plan entries remain unchanged until wait.
        """
        @T.macro
        def fetch(peer, first, stop):
            """Reserve and write a batch without posting or ringing queues."""
            copies = T.alloc_local((1,), T.int32)
            count = T.alloc_local((1,), T.uint32)
            base = T.alloc_local((1,), T.uint32)
            step = T.alloc_local((1,), T.uint64)
            copies[0] = 0
            if peer >= 0 and die < _DIES:
                for slot in T.serial(first, stop):
                    expert = plan[peer * slots + slot]
                    if expert >= rank * slots and expert < (rank + 1) * slots:
                        copies[0] += 1
                if copies[0] > 0:
                    qid = pair * _DIES + die
                    queue = queues[qid]
                    step[0] = steps[qid] + T.uint64(1)
                    steps[qid] = step[0]
                    count[0] = T.cast(copies[0] * weight_count * chunks_per_row
                                     + T.if_then_else(peer != rank, 2, 1), T.uint32)
                    base[0] = oki_reserve(queue, count)
                    for slot in T.serial(first, stop):
                        expert = plan[peer * slots + slot]
                        if expert >= rank * slots and expert < (rank + 1) * slots:
                            half_offset = T.cast(die, T.int64) * (row_bytes // _DIES)
                            src_offset = (T.cast(expert - rank * slots, T.int64)
                                          * row_bytes + half_offset)
                            dst_offset = (T.cast(peer, T.int64) * rank_stride
                                          + T.cast(slot, T.int64) * row_bytes + half_offset)
                            enqueue_row(weights[0], weights[1], queue, src_offset,
                                        dst_offset, peer != rank, die)
                            if weight_count == 3:
                                enqueue_row(weights[2], weights[3], queue, src_offset,
                                            dst_offset, peer != rank, die)
                                enqueue_row(weights[4], weights[5], queue, src_offset,
                                            dst_offset, peer != rank, die)
            return copies, base, count, step, peer

        @T.macro
        def commit(batch):
            """Publish both die queues before lane 0 rings their doorbells."""
            copies, base, count, step, peer = batch
            if die < _DIES and copies[0] > 0:
                queue = queues[pair * _DIES + die]
                if peer != rank:
                    fence_index = (T.cast(peer, T.int64) * fence_stride
                                   + rank * _DIES + die)
                    oki_fence_remote(queue, fences, fence_index, step[0])
                oki_fence_local(queue, step[0], die != 0)
                # Keep the SDK-adjusted reservation count, including padding.
                oki_post(queue, base[0], count[0])
            T.sync_threads()
            if die == 0 and copies[0] > 0:
                for d in T.serial(_DIES):
                    oki_ring(queues[pair * _DIES + d])
            T.sync_threads()
            return copies, step

        @T.macro
        def wait(ticket):
            """Drain this batch's captured step, not a later queue counter.

            Empty batches are no-ops except for CTA synchronization. This is
            queue completion, not an inter-rank barrier.
            """
            copies, step = ticket
            if die < _DIES and copies[0] > 0:
                oki_poll(queues[pair * _DIES + die], step[0])
            T.sync_threads()

        return SimpleNamespace(fetch=fetch, commit=commit, wait=wait)

    @T.prim_func
    def kernel(
        local0: T.Tensor((local_bytes,), T.uint8), pool0: T.Tensor((pool_bytes,), T.uint8),
        local1: T.Tensor((local_bytes,), T.uint8), pool1: T.Tensor((pool_bytes,), T.uint8),
        local2: T.Tensor((local_bytes,), T.uint8), pool2: T.Tensor((pool_bytes,), T.uint8),
        plan: T.Tensor((world_size * slots,), T.int32),
        queues: T.Tensor((_QUEUE_PAIRS * _DIES,), T.uint64),
        steps: T.Tensor((_QUEUE_PAIRS * _DIES,), T.uint64),
        fences: T.Tensor((world_size * fence_stride,), T.uint64), rank: T.int32,
    ):
        with T.Kernel(1, threads=_QUEUE_PAIRS * 32):
            T.import_source(sdk_headers)
            T.assume(rank >= 0)
            T.assume(rank < world_size)
            tx = T.get_thread_binding()
            pair = tx // 32
            die = tx % 32
            weights = (local0, pool0, local1, pool1, local2, pool2)
            dma = bind_dma(weights, plan, queues, steps, fences, rank, pair, die)

            # Each queue pair services one peer at a time.
            for peer_round in T.serial(peer_rounds):
                remote_index = pair - 1 + peer_round * (_QUEUE_PAIRS - 1)
                remote_peer = remote_index + T.if_then_else(remote_index >= rank, 1, 0)
                peer = T.if_then_else(
                    pair == 0, T.if_then_else(peer_round == 0, rank, -1),
                    T.if_then_else(remote_index < world_size - 1, remote_peer, -1),
                )

                for batch_index in T.serial(batch_count):
                    first = batch_index * batch_slots
                    stop = T.min(first + batch_slots, slots)
                    batch = dma.fetch(peer, first, stop)
                    ticket = dma.commit(batch)
                    dma.wait(ticket)

    return tilelang.compile(
        kernel, target="ppu", execution_backend="tvm_ffi",
        compile_flags=["--dopt=3", "-rdc=true", "-dlink", "-Xllvm", "-ppu-simt-branch=true",
                       f"-L{sdk}/lib", "-llt_oki_device_simt"],
    )
