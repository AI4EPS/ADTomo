import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


# Download Eigen library
def download_eigen(eigen_dir="./adtomo/eigen"):
    eigen_zip_url = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
    if not os.path.exists(eigen_dir):
        os.makedirs(eigen_dir, exist_ok=True)
        subprocess.check_call(["wget", "-O", "eigen.zip", eigen_zip_url])
        subprocess.check_call(["unzip", "eigen.zip", "-d", eigen_dir])
        eigen_unzipped_dir_name = os.listdir(eigen_dir)[0]
        for filename in os.listdir(os.path.join(eigen_dir, eigen_unzipped_dir_name)):
            os.rename(os.path.join(eigen_dir, eigen_unzipped_dir_name, filename), os.path.join(eigen_dir, filename))
        os.rmdir(os.path.join(eigen_dir, eigen_unzipped_dir_name))  # Clean up the unzipped directory


download_eigen()

setup(
    name="adtomo",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name="eik2d_cpp",
            sources=["adtomo/eikonal/Eikonal2D.cpp"],
            include_dirs=[
                "./adtomo/eigen",
            ],
            extra_compile_args=[],
            language="c++",
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_cuda=True)},  # Disables CUDA, compile only for CPU
    install_requires=[
        "torch",
    ],
    python_requires=">=3.6",
)
