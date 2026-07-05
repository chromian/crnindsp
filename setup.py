from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True

extensions = [
    Extension(
        "crnindsp.cycrnindsp",
        ["src/crnindsp/cycrnindsp.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-w"]
    )
]

INSTALL_REQUIRES = [
    'numpy>=1.24.4',
    'scipy>=1.10.1',
]

long_description = open('README.md').read()

setup(
    name='crnindsp',
    version='0.1.0',
    author='Yong-Jin Huang',
    description='A toolbox for the identification of indicator species in a chemical reaction network.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/chromian/crnindsp',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules = cythonize(extensions, annotate = True),
    include_dirs=[np.get_include()],
    zip_safe=False,
    install_requires=INSTALL_REQUIRES,
    python_requires='>=3',
)

