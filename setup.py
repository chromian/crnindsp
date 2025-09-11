from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True

def get_extensions():
    import numpy as np
    return [
        Extension(
            "crnindsp.cycrnindsp",
            ["src/crnindsp/cycrnindsp.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-w"],
        )
    ]

setup(
    name="crnindsp",
    version="0.0.3",
    author="Yong-Jin Huang",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(get_extensions(), annotate=True),
    python_requires=">=3.8",
)
