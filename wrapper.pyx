# cimport numpy as cnp
from icecream import ic

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API.

cdef extern from "encode/kernel.h":
    float gpu_encode(unsigned char* image, int batch_size,
                    int* height, int* width, int dev)

# MemoryView as lower-c performances 
def cuda_encode(unsigned char[:] image, int batch_size, int[:] height, int[:] width, int dev):
    """
    encode image arrays into jpeg2000 bitstream

    :param image: image(s) given batch_size
    :param batch_size: encode #batch_size image(s)
    :param height: height(s) of image(s), used for stride calculation
    :param width: width(s) of image(s), used for stride calculation
    :param dev: device used for encoding images
    :return: jpeg2000 bitstream@ host memory
    """
    return gpu_encode(&image[0], batch_size, &height[0], &width[0], dev)

def cuda_decode():
    raise NotImplementedError