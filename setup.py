from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

extensions = [
    Extension("crnindsp.crnindsp",
              ["src/cycrnindsp/crnindsp.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args = ["-w"]
    )
]

setup(
    name = 'crnindsp',
    version = '0.0.3',
    author = 'Yong-Jin Huang',
    ext_modules = cythonize(extensions, annotate = True),
    include_dirs=[np.get_include()],
)

