#include <iostream>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <fstream>

#include <cuda_runtime_api.h>
#include <nvjpeg2k.h>
#include "kernel.h"

#define NUM_COMPONENTS 3

void check_cuda(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        std::cout << "CUDA Runtime failure: '#" << status << "' at " << __FILE__ << ":" << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }
}

int check_nvjpeg2k(nvjpeg2kStatus_t call)
{
    if (call != NVJPEG2K_STATUS_SUCCESS)
    {
        std::cout << "NVJPEG failure: '#" << call << "' at " << __FILE__ << ":" << __LINE__ << std::endl;
        return EXIT_FAILURE;
    }
    return call;
}

/**
 * encode MemoryView arrays into nvJPEG2000 bitstream
 *
 * @param images MemoryView, image(s) given batch_size
 * @param batch_size batch_size
 * @param height height(s) of image(s)
 * @param width width(s) of image(s)
 * @param dev device used for GPU encoding
 * @return nvJPEG2000 bitstream
 */
float gpu_encode(unsigned char *images, int batch_size,
                 int *height, int *width, int dev)

{
    nvjpeg2kEncoder_t enc_handle;      // nvJPEG2000 encoder handle
    nvjpeg2kEncodeState_t enc_state;   // store the encoder work buffers and intermediate results
    nvjpeg2kEncodeParams_t enc_params; // stores various parameters that control the compressed output

    // initialize the library handles
    check_nvjpeg2k(nvjpeg2kEncoderCreateSimple(&enc_handle));
    check_nvjpeg2k(nvjpeg2kEncodeStateCreate(enc_handle, &enc_state));
    check_nvjpeg2k(nvjpeg2kEncodeParamsCreate(&enc_params));

    // initialise image
    std::vector<void *> pixel_data_d_;
    std::vector<void *> pixel_data_h_;
    std::vector<size_t> pitch_in_bytes_d_;
    std::vector<size_t> pitch_in_bytes_h_;
    std::vector<size_t> pixel_data_size_;
    std::vector<nvjpeg2kImageComponentInfo_t> comp_info_;

    pixel_data_h_.resize(NUM_COMPONENTS, nullptr);
    pixel_data_d_.resize(NUM_COMPONENTS, nullptr);
    pitch_in_bytes_h_.resize(NUM_COMPONENTS, 0);
    pitch_in_bytes_d_.resize(NUM_COMPONENTS, 0);
    pixel_data_size_.resize(NUM_COMPONENTS, 0);

    /*
    typedef struct
    {
        void **pixel_data;
        size_t *pitch_in_bytes;
        nvjpeg2kImageType_t pixel_type;
        uint32_t num_components;
    } nvjpeg2kImage_t;
    */
    nvjpeg2kImage_t image_h_;
    nvjpeg2kImage_t image_d_;

    image_d_.pixel_data = pixel_data_d_.data();
    image_d_.pitch_in_bytes = pitch_in_bytes_d_.data();
    image_h_.pixel_data = pixel_data_h_.data();
    image_h_.pitch_in_bytes = pitch_in_bytes_h_.data();
    image_d_.pixel_type = NVJPEG2K_UINT8;
    image_h_.pixel_type = NVJPEG2K_UINT8;
    image_d_.num_components = NUM_COMPONENTS;
    image_h_.num_components = image_d_.num_components;
    int bytes_per_element = 1; // unsigned char

    nvjpeg2kImageComponentInfo_t image_comp_info[NUM_COMPONENTS];

    uint32_t image_width = width[0];
    uint32_t image_height = height[0];

    for (int c = 0; c < NUM_COMPONENTS; c++)
    {
        image_comp_info[c].component_width = image_width;
        image_comp_info[c].component_height = image_height;
        image_comp_info[c].precision = 8;
        image_comp_info[c].sgn = 0;
    }

    // malloc/ cudaMalloc
    for (uint32_t c = 0; c < NUM_COMPONENTS; c++)
    {
        image_d_.pitch_in_bytes[c] =
            image_h_.pitch_in_bytes[c] = image_comp_info[c].component_width * bytes_per_element * 3;
        size_t comp_size = image_comp_info[c].component_height * image_d_.pitch_in_bytes[c];
        if (comp_size > pixel_data_size_[c])
        {
            if (image_d_.pixel_data[c])
            {
                check_cuda(cudaFree(image_d_.pixel_data[c]));
            }
            if (image_h_.pixel_data[c])
            {
                free(image_h_.pixel_data[c]);
            }
            pixel_data_size_[c] = comp_size;
            check_cuda(cudaMalloc(&image_d_.pixel_data[c], comp_size));
            image_h_.pixel_data[c] = malloc(comp_size);
        }
    }

    auto &img_h = image_h_;
    unsigned char *r = reinterpret_cast<unsigned char *>(img_h.pixel_data[0]);
    unsigned char *g = reinterpret_cast<unsigned char *>(img_h.pixel_data[1]);
    unsigned char *b = reinterpret_cast<unsigned char *>(img_h.pixel_data[2]);

    // host image data initialise
    for (unsigned int y = 0; y < image_height; y++)
    {
        for (unsigned int x = 0; x < image_width; x++)
        {
            r[y * img_h.pitch_in_bytes[0] + x] = images[y * img_h.pitch_in_bytes[0] + (3 * x + 0)];
            g[y * img_h.pitch_in_bytes[1] + x] = images[y * img_h.pitch_in_bytes[1] + (3 * x + 1)];
            b[y * img_h.pitch_in_bytes[2] + x] = images[y * img_h.pitch_in_bytes[2] + (3 * x + 2)];
        }
    }

    // copy to device
    auto &img_d = image_d_;
    for (int c = 0; c < NUM_COMPONENTS; c++)
    {
        // cudaMallocPitch(&img_d.pixel_data[c], &img_d.pitch_in_bytes[c], image_comp_info[c].component_width, image_comp_info[c].component_height);
        check_cuda(cudaMemcpy2D(img_d.pixel_data[c], img_d.pitch_in_bytes[c], img_h.pixel_data[c], img_h.pitch_in_bytes[c],
                                image_comp_info[c].component_width * bytes_per_element,
                                image_comp_info[c].component_height, cudaMemcpyHostToDevice));
    }

    // populate config
    nvjpeg2kEncodeConfig_t enc_config;
    memset(&enc_config, 0, sizeof(enc_config));
    enc_config.stream_type = NVJPEG2K_STREAM_JP2;      // the bitstream will be in JP2 container format
    enc_config.color_space = NVJPEG2K_COLORSPACE_SRGB; // input image is in RGB format
    enc_config.image_width = image_width;
    enc_config.image_height = image_height;
    enc_config.num_components = NUM_COMPONENTS;
    enc_config.image_comp_info = image_comp_info;
    enc_config.code_block_w = 64;
    enc_config.code_block_h = 64;
    enc_config.irreversible = 0;
    enc_config.mct_mode = 1;
    enc_config.prog_order = NVJPEG2K_RPCL;
    enc_config.num_resolutions = 1;

    check_nvjpeg2k(nvjpeg2kEncodeParamsSetEncodeConfig(enc_params, &enc_config));
    check_nvjpeg2k(nvjpeg2kEncodeParamsSetQuality(enc_params, 1000));
    check_nvjpeg2k(nvjpeg2kEncode(enc_handle, enc_state, enc_params, &img_d, NULL));

    size_t compressed_size;
    check_nvjpeg2k(nvjpeg2kEncodeRetrieveBitstream(enc_handle, enc_state, NULL, &compressed_size, NULL));

    // unsigned char *compressed_data = new unsigned char [compressed_size];
    BitStreamData bitstreams(batch_size);
    bitstreams[0].resize(compressed_size);

    check_nvjpeg2k(nvjpeg2kEncodeRetrieveBitstream(enc_handle, enc_state, bitstreams[0].data(), &compressed_size, NULL));
    cudaDeviceSynchronize();

    std::ofstream bitstream_file("image.jp2",
                                 std::ios::out | std::ios::binary);
    bitstream_file.write((char *)bitstreams[0].data(), compressed_size);
    bitstream_file.close();

    // free encoder resources
    check_nvjpeg2k(nvjpeg2kEncodeParamsDestroy(enc_params));
    check_nvjpeg2k(nvjpeg2kEncodeStateDestroy(enc_state));
    check_nvjpeg2k(nvjpeg2kEncoderDestroy(enc_handle));

    return 1;
}