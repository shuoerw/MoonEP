from types import SimpleNamespace

import pytest

from moonep._backend import IS_PPU, api


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
