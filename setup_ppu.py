"""PPU extension configuration, selected by MOONEP_BUILD_TARGET=ppu."""

import os


REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
PPU_SDK = os.environ.get("PPU_SDK", "/usr/local/PPU_SDK")
HGCC = os.path.join(PPU_SDK, "bin", "hgcc")
os.environ.setdefault("CC", HGCC)
os.environ.setdefault("CXX", HGCC)

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="moonep",
    version="0.0.1",
    packages=find_packages(include=["moonep", "moonep.*"]),
    ext_modules=[
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
                "cxx": [
                    "-O3",
                    "-fno-strict-aliasing",
                    "-Wno-psabi",
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
    ],
    cmdclass={"build_ext": BuildExtension},
)
