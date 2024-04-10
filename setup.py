from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='eik2d_cpp',
      ext_modules=[cpp_extension.CppExtension('eik2d_cpp', ['Eikonal2Dt.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
