import numpy as np
cimport numpy as cnp

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API.
cnp.import_array()

cimport cython
from cython.parallel cimport prange


