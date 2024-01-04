from Cython.Build import cythonize
from distutils.core import setup, Extension
import numpy

setup(
  name = 'nvjpeg2000-python',         # How you named your package folder (MyLib)
  packages = ['nvjpeg2000-python'],   # Chose the same as "name"
  version = '0.0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  ext_modules=cythonize("nvjpeg2000.pyx"),
  include_dirs=[numpy.get_include()]
)