"""
moonep build script. Compiles the extension `moonep._C` into the `moonep`
package so it can be imported as `from moonep import _C` after either:

  pip install -e .                              # editable install (preferred)

or, for a quick in-place build without installing:

  python setup.py build_ext --inplace

The .so lands at `moonep/_C.<abi-tag>.so`.
The default build uses CUDA; set MOONEP_BUILD_TARGET=ppu for the PPU SDK.
"""

import os

build_ppu = os.environ.get("MOONEP_BUILD_TARGET", "cuda").strip().lower() == "ppu"
if build_ppu:
    PPU_SDK = os.environ.get("PPU_SDK", "/usr/local/PPU_SDK")
    HGCC = os.path.join(PPU_SDK, "bin", "hgcc")
    os.environ.setdefault("CC", HGCC)
    os.environ.setdefault("CXX", HGCC)

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension


REPO_ROOT = os.path.abspath(os.path.dirname(__file__))


nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-std=c++20",
    "--expt-extended-lambda",
    "--expt-relaxed-constexpr",
    "-forward-unknown-to-host-compiler",
    "-Xcompiler=-Wno-psabi",
    "-Xcompiler=-fno-strict-aliasing",
    "-DNDEBUG",
    "-lineinfo",
    "-diag-suppress=3189",
    "-ftemplate-backtrace-limit=0",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
]

cxx_flags = [
    "-O3",
    "-fno-strict-aliasing",
    "-Wno-psabi",
]


if build_ppu:
    from torch.utils.cpp_extension import CppExtension

    ext_modules = [
        CppExtension(
            name="moonep._C",
            sources=[os.path.join("csrc", "ptg", "bindings.cpp")],
            include_dirs=[
                os.path.join(REPO_ROOT, "csrc", "ptg"),
                os.path.join(PPU_SDK, "include"),
                os.path.join(PPU_SDK, "targets", "x86_64-linux", "include"),
            ],
            library_dirs=[
                os.path.join(PPU_SDK, "lib"),
                os.path.join(PPU_SDK, "lib64"),
                os.path.join(PPU_SDK, "targets", "x86_64-linux", "lib"),
            ],
            libraries=["hggc"],
            extra_compile_args={
                "cxx": cxx_flags + [
                    "--forward-unknown-to-host-compiler",
                    "-std=c++17",
                    "-DMOONEP_TILELANG_PPU=1",
                ],
            },
            extra_link_args=[
                "--forward-unknown-to-host-compiler",
                "--forward-unknown-to-host-linker",
                "-Wl,-rpath,$ORIGIN",
            ],
        ),
    ]
    install_requires = []
else:
    from torch.utils.cpp_extension import CUDAExtension

    ext_modules = [
        CUDAExtension(
            name="moonep._C",
            sources=[
                os.path.join("csrc", "bindings.cu"),
            ],
            include_dirs=[
                os.path.join(REPO_ROOT, "csrc"),
            ],
            libraries=["cuda"],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        ),
    ]
    install_requires = ["nvidia-cutlass-dsl==4.6.2"]


setup(
    name="moonep",
    version="0.0.1",
    packages=find_packages(include=["moonep", "moonep.*"]),
    install_requires=install_requires,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
