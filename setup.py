from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True

extensions = [
    Extension(
        "crnindsp",
        ["src/crnindsp/cycrnindsp.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-w"]
    )
]

INSTALL_REQUIRES = [
    'numpy>=1.24.4',
    'scipy>=1.10.1',
]

setup(
    name='crnindsp',
    version='0.0.5',
    author='Yong-Jin Huang',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules = cythonize(extensions, annotate = True),
    include_dirs=[np.get_include()],
    zip_safe=False,
    install_requires=INSTALL_REQUIRES,
    python_requires='>=3',
)
