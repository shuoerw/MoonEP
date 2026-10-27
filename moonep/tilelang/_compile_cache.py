"""Device-local, thread-safe caches for compiled PPU kernels."""

import threading
from collections.abc import Callable

import torch


class CompileCache:
    """Keep a separate key namespace and compile lock for each kernel module."""

    def __init__(self):
        self._lock = threading.Lock()
        self._compiled: dict[tuple, object] = {}

    def __call__(self, key: tuple, builder: Callable[[], object]):
        device = torch.cuda.current_device()
        full_key = (device, torch.cuda.get_device_capability(device), *key)
        compiled = self._compiled.get(full_key)
        if compiled is not None:
            return compiled
        with self._lock:
            compiled = self._compiled.get(full_key)
            if compiled is None:
                with torch.cuda.device(device):
                    compiled = builder()
                self._compiled[full_key] = compiled
        return compiled
