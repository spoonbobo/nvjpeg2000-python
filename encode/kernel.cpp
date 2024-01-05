#include <iostream>
#include <stdlib.h>
#include "kernel.h"

#include <cuda_runtime_api.h>
#include <nvjpeg2k.h>

float gpu_encode(unsigned char* image, int batch_size,
                    int height, int width, int dev) {
    encode_params_t params;
    params.batch_size = batch_size;
    params.height = height;
    params.width = width;
    params.cblk_w = 64;
    params.cblk_h = 64;
    params.dev = dev;
    params.verbose = 1;

    // query device
    cudaDeviceProp props;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&props, dev);
    
    if (params.verbose) {
        printf("Using GPU - %s with CC %d.%d\n", props.name, props.major, props.minor);
    }
    
    // initialize encoder
    CHECK_NVJPEG2K(nvjpeg2kEncoderCreateSimple(&params.enc_handle));
    CHECK_NVJPEG2K(nvjpeg2kEncodeStateCreate(params.enc_handle, &params.enc_state));
    CHECK_NVJPEG2K(nvjpeg2kEncodeParamsCreate(&params.enc_params));

    // read array
    

    // free encoder resources
    CHECK_NVJPEG2K(nvjpeg2kEncodeParamsDestroy(params.enc_params));
    CHECK_NVJPEG2K(nvjpeg2kEncodeStateDestroy(params.enc_state));
    CHECK_NVJPEG2K(nvjpeg2kEncoderDestroy(params.enc_handle));
    
    return image[0];
}
