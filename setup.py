from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension('lib.hitC', ['lib/hitC.pyx'])

setup(
  name = 'vmimodules',
  version='0.3',
  ext_modules = cythonize(ext),
# package_dir = {'': 'lib'},
  include_dirs=[numpy.get_include()]
)
