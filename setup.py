
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

setup(
    name = 'crnindsp',
    version = '0.0.1',
    author = 'Yong-Jin Huang',
    ext_modules = cythonize("src/crnindsp.pyx"),
    include_dirs=[np.get_include()],
)
