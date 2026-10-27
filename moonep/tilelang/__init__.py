"""TileLang kernels for MoonEP's PPU backend."""


def _check_compiler():
    try:
        import tilelang.ppu.language as T
        import tvm_ffi
        from tvm.target import Target

        Target("ppu")
        for name in ("ppu_aiu_load", "import_source", "sync_grid"):
            if not hasattr(T, name):
                raise AttributeError(f"TileLang language is missing {name}")
        if not hasattr(tvm_ffi, "use_torch_stream"):
            raise AttributeError("tvm_ffi is missing use_torch_stream")
    except (ImportError, AttributeError, ValueError, RuntimeError) as exc:
        raise RuntimeError(
            "MoonEP PPU requires a PPU-enabled T-HEAD TileLang build and "
            "tvm_ffi.use_torch_stream. See the validated compiler revision "
            "in README.md (PPU installation)."
        ) from exc


_check_compiler()
