from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="tetrahedron_renderer",
    packages=['tetrahedron_renderer'],
    ext_modules=[
        CUDAExtension(
            name="tetrahedron_renderer._C",
            sources=[
                "cuda_renderer/renderer_impl.cu",
                "cuda_renderer/forward.cu",
                "cuda_renderer/backward.cu",
                "render.cu",
                "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)