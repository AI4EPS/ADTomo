import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = os.path.dirname(os.path.abspath(__file__))

torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")

def cpp_ext(name, source):
    return CppExtension(
        name=name,
        sources=[source],
        extra_link_args=[f"-Wl,-rpath,{torch_lib_dir}"],
        language="c++",
    )

setup(
    name="adtomo",
    version="0.1.0",
    packages=["adtomo"],
    ext_modules=[
        cpp_ext("eikonal2d_op", os.path.join(ROOT, "adtomo/eikonal/Eikonal2D.cpp")),
        cpp_ext("eikonal3d_op", os.path.join(ROOT, "adtomo/eikonal/Eikonal3D.cpp")),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_cuda=True)},
    install_requires=[
        "torch>=2.10",
        "numpy>=1.24",
        "pandas>=2.0",
    ],
    python_requires=">=3.9",
)
