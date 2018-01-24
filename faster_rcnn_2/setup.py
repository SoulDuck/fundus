import numpy as np
from distutils.core import setup
from Cython.Build import cythonize
setup(ext_modules = cythonize("./faster_rcnn_2/bbox_overlaps.pyx"),include_dirs=[np.get_include()])
setup(ext_modules = cythonize("./faster_rcnn_2/bbox_transform.pyx"),include_dirs=[np.get_include()])
setup(ext_modules = cythonize("./faster_rcnn_2/cpu_nms.pyx"),include_dirs=[np.get_include()])


