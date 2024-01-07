#include <iostream>
#include <stdlib.h>
#include "kernel.h"
#include <icecream.hpp>
#include <cuda_runtime_api.h>
#include <nvjpeg2k.h>

/**
 * Load image arrays into nvJPEG2000 image inputs
 * 
 * @param images MemoryView, image(s) given batch_size
 * @param height height(s) of image(s)
 * @param width width(s) of image(s)
 * @param images_input nvJPEG2000 image inputs
 * @param params nvJPEG2000 encoding parameters 
 * @return EXIT status
 */
int read_batch_images(unsigned char *images, int* height, int* width,
                      std::vector<Image> &images_input, encode_params_t params)
{
    int counter, offset;
    counter = offset = 0;
    
    while (counter < params.batch_size)
    {
        unsigned char* image = &images[offset];    

        nvjpeg2kImageInfo_t nvjpeg2k_info;
        std::vector<nvjpeg2kImageComponentInfo_t> nvjpeg2k_comp_info;
        nvjpeg2kColorSpace_t color_space = NVJPEG2K_COLORSPACE_SRGB;
        
        nvjpeg2k_info.num_components = 3;
        nvjpeg2k_info.image_width = width[counter];
        nvjpeg2k_info.image_height = height[counter];
        nvjpeg2k_comp_info.resize(nvjpeg2k_info.num_components);

        for(auto& comp: nvjpeg2k_comp_info) {
            comp.component_width  = nvjpeg2k_info.image_width;
            comp.component_height = nvjpeg2k_info.image_height;
            comp.precision        = 8;
            comp.sgn              = 0;
        }

        images_input[counter].initialize(nvjpeg2k_info, nvjpeg2k_comp_info.data(), color_space);
        auto& img = images_input[counter].getImageHost();
        unsigned char* r = reinterpret_cast<unsigned char*>(img.pixel_data[0]);
        unsigned char* g = reinterpret_cast<unsigned char*>(img.pixel_data[1]);
        unsigned char* b = reinterpret_cast<unsigned char*>(img.pixel_data[2]);

        // Load data in image channels
        
        // next image
        offset += width[counter]*height[counter]*3;
        counter++;
    }

    return EXIT_SUCCESS;
}

float gpu_encode(unsigned char *images, int batch_size,
                 int *height, int *width, int dev)
{
    encode_params_t params;
    params.batch_size = batch_size;
    params.cblk_w = 64;
    params.cblk_h = 64;
    params.dev = dev;
    params.verbose = 1;

    // query device
    cudaDeviceProp props;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&props, dev);
    cudaSetDevice(dev);

    if (params.verbose)
    {
        printf("Using GPU - %s with CC %d.%d\n", props.name, props.major, props.minor);
    }

    // initialize encoder
    CHECK_NVJPEG2K(nvjpeg2kEncoderCreateSimple(&params.enc_handle));
    CHECK_NVJPEG2K(nvjpeg2kEncodeStateCreate(params.enc_handle, &params.enc_state));
    CHECK_NVJPEG2K(nvjpeg2kEncodeParamsCreate(&params.enc_params));

    // batch read array
    printf("batch size: %d\n", params.batch_size);
    std::vector<Image> input_images(params.batch_size);

    read_batch_images(images, height, width, input_images, params);

    // free encoder resources
    CHECK_NVJPEG2K(nvjpeg2kEncodeParamsDestroy(params.enc_params));
    CHECK_NVJPEG2K(nvjpeg2kEncodeStateDestroy(params.enc_state));
    CHECK_NVJPEG2K(nvjpeg2kEncoderDestroy(params.enc_handle));

    return images[0];
}
