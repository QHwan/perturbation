import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

setup(
	name = "fluc_area",
	include_dirs = [np.get_include()],
	ext_modules=cythonize(['fluc_area.pyx','func.pyx'])
	)
