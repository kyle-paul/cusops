from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_layers',
    ext_modules=[CUDAExtension('cuda_layers', ['cuda_layers.cu'])],
    cmdclass={'build_ext': BuildExtension.with_options(n_jobs=16)}
)