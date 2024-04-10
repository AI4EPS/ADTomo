from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="adtomo",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name="eik2d_cpp",
            sources=["Eikonal2D.cpp"],
            extra_compile_args=[],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_cuda=True)},  # Disables CUDA, compile only for CPU
    install_requires=[
        "torch",
    ],
    python_requires=">=3.6",
)
