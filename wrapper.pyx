# cimport numpy as cnp
from cython cimport wraparound, boundscheck, cdivision
import numpy as np
cimport numpy as cnp

cdef extern from "encode/nvjpeg2k/nvjpeg2k_encoder.h":

    float encodeJpeg2k_(unsigned char* image, int batch_size,
                    int *height, int *width, int dev)

    float encodeJpeg2kImageViewSingleBatch_(
        unsigned char* r, unsigned char* g, unsigned char* b,
        int height, int width, int dim, int dev)


# todo
cdef extern from "encode/nvjpeg/nvjpeg_encoder.h":
    float encodeJpeg_(unsigned char* r, int dev)
    

cdef class NvJpegEncoder:

    def __init__(self):
        pass

    # v1: non-optimized jpeg2k compression
    # it involves loading in numpy array in cpp file
    # it assumes dim3/rgb image
    @boundscheck(False)
    @wraparound(False)
    cpdef encodeJpeg2k(self, unsigned char[:] image, int batch_size, int[:] height, int[:] width, int dev):
        """
        encode image arrays into jpeg2000 bitstream# 
        :param image: image(s) given batch_size in MemoryView
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
        # TODO return bytes (low-priority)

    # v2: more optimized jpeg2k optimization
    @boundscheck(False)
    @wraparound(False)
    cpdef encodeJpeg2kImageViewSingleBatch(self, unsigned char[:,:,:] image, int dev):

        cdef:
            int height = image.shape[0]
            int width = image.shape[1]
            int dim = image.shape[2]
            unsigned char[:] r, g, b

        # 1d continguous array
        r = np.ravel(image[:,:,0])
        g = np.ravel(image[:,:,1])
        b = np.ravel(image[:,:,2])
        
        return encodeJpeg2kImageViewSingleBatch_(&r[0], &g[0], &b[0], height, width, dim, dev)


    # TODO
    @boundscheck(False)
    @wraparound(False)
    cpdef encodeJpegImageViewSingleBatch(self, unsigned char[:,:,:] image, int dev):
        pass
        

cdef class NvJpegDecoder:

    def __init__(self):
        pass

    # TODO
    @boundscheck(False)
    @wraparound(False)
    cpdef decodeJpegImageViewSingleBatch(self):
        pass
        