#include <iostream>
#include <stdlib.h>
#include "kernel.h"

#include <cuda_runtime_api.h>
#include <nvjpeg2k.h>

float gpu_encode(unsigned char* image, int batch_size,
                    int height, int width) {
    encode_params_t params;
    params.batch_size = batch_size;
    params.height = height;
    params.width = width;
    params.cblk_w = 64;
    params.cblk_h = 64;

    CHECK_NVJPEG2K(nvjpeg2kEncoderCreateSimple(&params.enc_handle));
    CHECK_NVJPEG2K(nvjpeg2kEncodeStateCreate(params.enc_handle, &params.enc_state));
    CHECK_NVJPEG2K(nvjpeg2kEncodeParamsCreate(&params.enc_params));

    return image[0];
}
