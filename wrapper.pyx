# cimport numpy as cnp
from icecream import ic

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API.

cdef extern from "encode/kernel.h":
    float gpu_encode(float* image, int batch_size,
                    int height, int width)

# MemoryView as lower-c performances 
def cuda_encode(float[:,:,:] image, int batch_size, int height, int width):
    return gpu_encode(&image[0, 0, 0], batch_size, height, width)

def cuda_decode():
    raise NotImplementedError