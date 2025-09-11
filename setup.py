from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True

extensions = [
    Extension(
        "crnindsp.crnindsp",
        ["src/crnindsp/cycrnindsp.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-w"]
    )
]

setup(
    name='crnindsp',
    version='0.0.4',
    author='Yong-Jin Huang',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(extensions, annotate=True, language_level=3),
    include_dirs=[np.get_include()],
    zip_safe=False,
)



