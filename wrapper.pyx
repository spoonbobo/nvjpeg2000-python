# cimport numpy as cnp
from icecream import ic

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API.

cdef extern from "encode/kernel.h":
    float gpu_encode(float* a, float* b, float* c)

# MemoryView as lower-c performances 
def cuda_encode(float[:, :] a, float[:, :] b, float[:, :] c):
    return gpu_encode(&a[0, 0], &b[0, 0], &c[0, 0])

def cuda_decode():
    raise NotImplementedError