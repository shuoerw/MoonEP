from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
import math
import threading
import weakref

import pytest
import torch

from moonep import api
from moonep.api import IS_PPU


@pytest.mark.skipif(not IS_PPU, reason="PPU padding")
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("shape", [[0, 3], [1, 3], [7, 5, 3], [28, 3584, 3072]])
def test_padding_matches_minimum_aligned_rows(monkeypatch, dtype, shape):
    from moonep.tilelang import buffer, memory

    calls = []
    monkeypatch.setattr(buffer, "get_vmm_granularity", lambda: calls.append(1) or 65536)
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("padding must not allocate"))
    rows = memory.pad_dim0_for_alignment(shape=shape, dtype=dtype)
    row_bytes = dtype.itemsize * math.prod(shape[1:])
    assert rows >= shape[0] and rows * row_bytes % 65536 == 0
    step = 65536 // math.gcd(row_bytes, 65536)
    assert rows - step < shape[0]
    assert len(calls) == 1


@pytest.mark.skipif(not IS_PPU, reason="PPU compile cache")
def test_compile_cache_isolation_and_failure_retry(monkeypatch):
    from moonep.tilelang._compile_cache import CompileCache

    device, capability = 0, (8, 0)
    guards = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: device)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: capability)

    @contextmanager
    def guard(index):
        guards.append(index)
        yield

    monkeypatch.setattr(torch.cuda, "device", guard)
    cache = CompileCache()
    first = cache(("kernel", 32), object)
    assert cache(("kernel", 32), lambda: pytest.fail("cache miss")) is first
    assert CompileCache()(("kernel", 32), object) is not first
    assert cache(("kernel", 39), object) is not first
    device = 1
    assert cache(("kernel", 32), object) is not first
    device, capability = 0, (9, 0)
    assert cache(("kernel", 32), object) is not first
    assert guards == [0, 0, 0, 1, 0]

    def fail():
        raise RuntimeError("compile failed")

    with pytest.raises(RuntimeError, match="compile failed"):
        cache(("retry",), fail)
    value = cache(("retry",), object)
    assert cache(("retry",), fail) is value


@pytest.mark.skipif(not IS_PPU, reason="PPU compile cache")
def test_compile_cache_builds_once_for_concurrent_callers(monkeypatch):
    from moonep.tilelang._compile_cache import CompileCache
    from contextlib import nullcontext

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (8, 0))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    cache = CompileCache()
    start = threading.Barrier(8)
    builds = []

    def build():
        value = object()
        builds.append(value)
        return value

    def call(_):
        start.wait(timeout=10)
        return cache(("kernel",), build)

    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(call, range(8)))
    assert len(builds) == 1
    assert all(value is builds[0] for value in values)


@pytest.mark.skipif(not IS_PPU, reason="PPU VMM storage lifetime")
@pytest.mark.parametrize("alias_kind", ["view", "detach", "dtype", "inference"])
def test_rank_mapping_tracks_storage_lifetime(alias_kind):
    from moonep.tilelang import memory

    with torch.inference_mode(alias_kind == "inference"):
        owner = torch.empty(128)
        memory._register_mapping(owner, rank_stride=256, local_rank=0, world_size=2)
        key = memory.mapping_info(owner)["mapping_key"]
        alias = (owner.detach() if alias_kind == "detach" else
                 owner.view(torch.uint8) if alias_kind == "dtype" else owner.view(-1))
    owner_ref = weakref.ref(owner)
    del owner
    gc.collect()
    if alias_kind != "view":
        assert owner_ref() is None
    assert memory.mapping_info(alias)["mapping_key"] == key
    storage_ref = memory._MAPPINGS[key].storage
    del alias
    gc.collect()
    assert storage_ref.expired()
    memory.mapping_info(torch.empty(1))
    assert key not in memory._MAPPINGS


@pytest.mark.skipif(not IS_PPU, reason="PPU collective allocation errors")
@pytest.mark.parametrize("failure", ["shape", "dtype", "mode", "rank", "alloc", "map"])
def test_vmm_failure_reaches_all_ranks(dist_env, monkeypatch, failure):
    from moonep.tilelang import buffer

    rank, world_size = dist_env
    if world_size < 2:
        pytest.skip("requires at least two ranks")
    shape = [buffer.get_vmm_granularity()]
    dtype = torch.uint8
    supplied_rank = rank
    with monkeypatch.context() as patch:
        if rank == 0:
            if failure == "shape":
                shape[0] *= 2
            elif failure == "dtype":
                dtype = torch.int8
            elif failure == "mode":
                patch.setenv("MOONEP_MEM_HANDLE_TYPE", "invalid")
            elif failure == "rank":
                supplied_rank = world_size
            else:
                def fail(**kwargs):
                    raise RuntimeError("injected VMM failure")
                patch.setattr(buffer, f"icn_dist_{failure}", fail)
        with pytest.raises(RuntimeError, match="VMM"):
            buffer.create_icn_dist_tensor(shape, dtype, supplied_rank, world_size)
    # A failed initialization must not strand peers or poison the next allocation.
    tensor = buffer.create_icn_dist_tensor(
        [buffer.get_vmm_granularity()], torch.uint8, rank, world_size
    )
    tensor[rank * buffer.get_vmm_granularity()] = rank
    torch.cuda.synchronize()
    api.dist.barrier()
    assert tensor[::buffer.get_vmm_granularity()].cpu().tolist() == list(range(world_size))


@pytest.mark.skipif(not IS_PPU, reason="PPU bf16-pair copy validation")
@pytest.mark.parametrize("hidden", [0, -2, 1, 3, True, 2.0])
def test_create_context_rejects_invalid_hidden(monkeypatch, hidden):
    monkeypatch.setattr(api.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        api.torch.cuda, "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=39),
    )
    monkeypatch.setattr(api.dist, "get_rank", lambda **_: 0)
    monkeypatch.setattr(api.dist, "get_world_size", lambda **_: 1)
    error = TypeError if isinstance(hidden, (bool, float)) else ValueError
    with pytest.raises(error, match="H must be a positive even int"):
        api._create_context(1, hidden, 1, 1, 1)


@pytest.mark.parametrize(
    ("num_sms", "error", "match"),
    [
        (0, ValueError, "positive int"),
        (-1, ValueError, "positive int"),
        (1.5, TypeError, "float"),
        (True, TypeError, "bool"),
    ],
)
def test_create_context_rejects_invalid_num_sms(
    monkeypatch, num_sms, error, match
):
    monkeypatch.setattr(
        api.torch.cuda,
        "current_device",
        lambda: pytest.fail("invalid num_sms should fail before CUDA access"),
    )

    with pytest.raises(error, match=match):
        api._create_context(1, 1, 1, 1, 1, num_sms=num_sms)


def test_create_context_rejects_num_sms_above_device_count(monkeypatch):
    monkeypatch.setattr(api.torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(
        api.torch.cuda,
        "get_device_properties",
        lambda device: (
            SimpleNamespace(multi_processor_count=8)
            if device == 3
            else pytest.fail(f"unexpected device: {device}")
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"num_sms \(9\) must not exceed the device SM count \(8\)",
    ):
        api._create_context(1, 1, 1, 1, 1, num_sms=9)


def test_create_context_allows_num_sms_equal_to_device_count(monkeypatch):
    class ValidationPassed(Exception):
        pass

    monkeypatch.setattr(api.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        api.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=8),
    )
    monkeypatch.setattr(
        api.dist,
        "get_rank",
        lambda **_kwargs: (_ for _ in ()).throw(ValidationPassed),
    )

    with pytest.raises(ValidationPassed):
        api._create_context(1, 1, 1, 1, 1, num_sms=8)


def test_create_context_allows_default_on_low_sm_device(monkeypatch):
    class ValidationPassed(Exception):
        pass

    monkeypatch.setattr(api.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        api.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=8),
    )
    monkeypatch.setattr(
        api.dist,
        "get_rank",
        lambda **_kwargs: (_ for _ in ()).throw(ValidationPassed),
    )

    with pytest.raises(ValidationPassed):
        api._create_context(1, 1, 1, 1, 1)
