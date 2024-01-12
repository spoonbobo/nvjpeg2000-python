# cimport numpy as cnp
from cython cimport wraparound, boundscheck, cdivision

cdef extern from "encode/nvjpeg2k_encoder.h":
    float encodeJpeg2k_(unsigned char* image, int batch_size,
                    int *height, int *width, int dev)

# MemoryView as lower-c performances 
cdef class NvJpegEncoder:

    def __init__(self):
        pass

    @boundscheck(False)
    @wraparound(False)
    cpdef encodeJpeg2k(self, unsigned char[:] image, int batch_size, int[:] height, int[:] width, int dev):
        """
        encode image arrays into jpeg2000 bitstream# 
        :param image: image(s) given batch_size
        :param batch_size: encode #batch_size image(s)
        :param height: height(s) of image(s), used for stride calculation
        :param width: width(s) of image(s), used for stride calculation
        :param dev: device used for encoding images
        :return: jpeg2000 bitstream@ host memory
        """
        cdef:
            float result
            unsigned char* image_ptr = &image[0]
            int* height_ptr = &height[0]
            int* width_ptr = &width[0]

        result = encodeJpeg2k_(image_ptr, batch_size, height_ptr, width_ptr, dev)
        return result

class NvJpegDecoder:

    def __init__(self):
        pass