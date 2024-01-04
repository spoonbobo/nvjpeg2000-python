import numpy as np
cimport numpy as cnp

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API.
cnp.import_array()

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.float32

ctypedef cnp.float32_t DTYPE_t
ctypedef cnp.ndarray Array

def add(Array a, Array b):
    cdef Array h = np.zeros([3,3], dtype=DTYPE)
    return h
