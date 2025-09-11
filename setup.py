from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True

extensions = [
    Extension(
        "crnindsp.cycrnindsp",           # module inside the package
        ["src/crnindsp/cycrnindsp.pyx"], # path to your Cython source
        include_dirs=[np.get_include()],
        extra_compile_args=["-w"]
    )
]

setup(
    name="crnindsp",
    version="0.0.3",
    author="Yong-Jin Huang",
    packages=find_packages(where="src"),  # discover packages under src/
    package_dir={"": "src"},
    ext_modules=cythonize(extensions, annotate=True),
    include_dirs=[np.get_include()],
    python_requires=">=3.8",
)
