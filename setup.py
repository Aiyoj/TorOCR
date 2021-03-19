from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cuda_ext_modules = [
    CUDAExtension(
        "torocr.layers.dcn.deform_conv_cuda",
        [
            "torocr/layers/dcn/src/deform_conv_cuda.cpp",
            "torocr/layers/dcn/src/deform_conv_cuda_kernel.cu",
        ]
    ),
    CUDAExtension(
        "torocr.layers.dcn.deform_pool_cuda",
        [
            "torocr/layers/dcn/src/deform_pool_cuda.cpp",
            "torocr/layers/dcn/src/deform_pool_cuda_kernel.cu"
        ]
    )
]

setup(
    name="torocr",
    version="0.0.1",
    author="allen",
    ext_modules=cuda_ext_modules,
    cmdclass={"build_ext": BuildExtension}
)
