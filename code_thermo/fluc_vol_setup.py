import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

setup(
	name = "fluc_vol",
	include_dirs = [np.get_include()],
	ext_modules=cythonize(['fluc_vol.pyx','func.pyx'])
	)
