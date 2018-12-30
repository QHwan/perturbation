import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

setup(
	name = "principal_radii",
	include_dirs = [np.get_include()],
	ext_modules=cythonize(['principal_radii.pyx','func.pyx'])
	)
