from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

hitC = Extension('vmimodules.hitdet.hitC', ['vmimodules/hitdet/hitC.c'])
abel = Extension('vmimodules.inv.abel', ['vmimodules/inv/legendre/legendre_polynomial.c',
                                         'vmimodules/inv/legendre/abel.c']
                )
setup(
  name = 'vmimodules',
  version='0.3',
  ext_modules = [abel, hitC],
# ext_modules = [abel],
  packages = ['vmimodules'],
# package_dir = {'': 'lib'},
  include_dirs=[numpy.get_include()]
)
