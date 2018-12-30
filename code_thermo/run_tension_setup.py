import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

setup(
	name = "run_tension",
	include_dirs = [np.get_include()],
	ext_modules=cythonize(['run_tension.pyx','func.pyx'])
	)
