/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <stdlib.h>
#include "kernel.h"

#include <icecream.hpp>
#include <cuda_runtime_api.h>
#include <nvjpeg2k.h>

/** populate encoder config, encode input image
 * more references on https://docs.nvidia.com/cuda/nvjpeg2000/userguide.html#nvjpeg2kencodeconfig-label
 *
 * @param
 * @return
 */
void populate_encoderconfig(nvjpeg2kEncodeConfig_t &enc_config, Image &input_image, encode_params_t &params)
{
    memset(&enc_config, 0, sizeof(enc_config));
    enc_config.stream_type = NVJPEG2K_STREAM_JP2; // NVJPEG2K_STREAM_JP2 corresponds to the .jp2 container.
    enc_config.color_space = input_image.getColorSpace();
    enc_config.image_width = input_image.getnvjpeg2kImageInfo().image_width;
    enc_config.image_height = input_image.getnvjpeg2kImageInfo().image_height;
    enc_config.num_components = input_image.getnvjpeg2kImageInfo().num_components;
    enc_config.image_comp_info = input_image.getnvjpeg2kCompInfo();
    enc_config.code_block_w = (uint32_t)params.cblk_w;
    enc_config.code_block_h = (uint32_t)params.cblk_h;
    enc_config.irreversible = (uint32_t)params.irreversible;
    enc_config.mct_mode = 1; // 0 (YCC and Grayscale) , 1 (RGB)
    enc_config.prog_order = NVJPEG2K_LRCP; // defined in the JPEG2000 standard.
    enc_config.num_resolutions = 6;
}

/**
 * Load image arrays into nvJPEG2000 image inputs
 *
 * @param images MemoryView, image(s) given batch_size
 * @param height height(s) of image(s)
 * @param width width(s) of image(s)
 * @param images_input nvJPEG2000 image inputs
 * @param params nvJPEG2000 encoding parameters
 * @return exit code
 */
int read_batch_images(unsigned char *images, int *height, int *width,
                      std::vector<Image> &images_input, encode_params_t& params)
{
    int counter, offset, stride;
    counter = offset = 0;

    while (counter < params.batch_size)
    {
        unsigned char *raw_image = &images[offset];

        nvjpeg2kImageInfo_t nvjpeg2k_info;
        std::vector<nvjpeg2kImageComponentInfo_t> nvjpeg2k_comp_info;
        nvjpeg2kColorSpace_t color_space = NVJPEG2K_COLORSPACE_SRGB;

        nvjpeg2k_info.num_components = 3;
        nvjpeg2k_info.image_width = width[counter];
        nvjpeg2k_info.image_height = height[counter];
        stride = nvjpeg2k_info.image_width;
        nvjpeg2k_comp_info.resize(nvjpeg2k_info.num_components);

        for (auto &comp : nvjpeg2k_comp_info)
        {
            comp.component_width = nvjpeg2k_info.image_width;
            comp.component_height = nvjpeg2k_info.image_height;
            comp.precision = 8;
            comp.sgn = 0;
        }

        images_input[counter].initialize(nvjpeg2k_info, nvjpeg2k_comp_info.data(), color_space);
        auto &img = images_input[counter].getImageHost();
        unsigned char *r = reinterpret_cast<unsigned char *>(img.pixel_data[0]);
        unsigned char *g = reinterpret_cast<unsigned char *>(img.pixel_data[1]);
        unsigned char *b = reinterpret_cast<unsigned char *>(img.pixel_data[2]);

        // Load data in image channels
        for (unsigned int y = 0; y < nvjpeg2k_info.image_height; y++)
        {
            for (unsigned int x = 0; x < nvjpeg2k_info.image_width; x++)
            {
                // IC(raw_image[y * stride + (3 * x + 0)]);
                r[y * img.pitch_in_bytes[0] + x] = raw_image[y * stride + (3 * x + 0)];
                g[y * img.pitch_in_bytes[1] + x] = raw_image[y * stride + (3 * x + 1)];
                b[y * img.pitch_in_bytes[2] + x] = raw_image[y * stride + (3 * x + 2)];
            }
        }

        // copy to device
        images_input[counter].copyToDevice();

        // next image
        offset += width[counter] * height[counter] * 3;
        counter++;
    }
    return EXIT_SUCCESS;
}

/**
 * encode nvJPEG2000 image inputs into nvJPEG2000 images
 * @param images_input
 * @param time perf
 * @return exit code
 */
int encode_images(Image *input_images, BitStreamData &bitstreams, double &time, encode_params_t &params)
{
    cudaEvent_t startEvent = NULL, stopEvent = NULL;
    float loopTime = 0;

    CHECK_CUDA(cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync));
    CHECK_CUDA(cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync));
    nvjpeg2kEncodeConfig_t enc_config;
    size_t bs_sz;

    CHECK_CUDA(cudaEventRecord(startEvent, params.stream));
    for (int batch_id = 0; batch_id < params.batch_size; batch_id++)
    {
        populate_encoderconfig(enc_config, input_images[batch_id], params);
        CHECK_NVJPEG2K(nvjpeg2kEncodeParamsSetEncodeConfig(params.enc_params, &enc_config));
        CHECK_NVJPEG2K(nvjpeg2kEncodeParamsSetQuality(params.enc_params, params.target_psnr));
        CHECK_NVJPEG2K(nvjpeg2kEncode(params.enc_handle, params.enc_state, params.enc_params,
                                      &input_images[batch_id].getImageDevice(), params.stream));
        CHECK_NVJPEG2K(nvjpeg2kEncodeRetrieveBitstream(params.enc_handle, params.enc_state, NULL, &bs_sz,
                                                       params.stream));
        bitstreams[batch_id].resize(bs_sz);
        std::cout << bs_sz << std::endl;

        CHECK_NVJPEG2K(nvjpeg2kEncodeRetrieveBitstream(params.enc_handle, params.enc_state, bitstreams[batch_id].data(), &bs_sz,
                                                       params.stream));
        CHECK_CUDA(cudaStreamSynchronize(params.stream));
    }

    CHECK_CUDA(cudaEventRecord(stopEvent, params.stream));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&loopTime, startEvent, stopEvent));
    time += static_cast<double>(loopTime / 1000.0); // loopTime is in milliseconds

    CHECK_CUDA(cudaEventDestroy(stopEvent));
    CHECK_CUDA(cudaEventDestroy(startEvent));
    return EXIT_SUCCESS;
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
    encode_params_t params;
    params.batch_size = batch_size;
    params.cblk_w = 64;
    params.cblk_h = 64;
    params.dev = dev;
    params.verbose = 1;
    params.target_psnr = 6;
    params.irreversible = 1; // reversible wavelet transform

    // device setting
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
    if (params.verbose)
    {
        printf("batch size: %d\n", params.batch_size);
    }

    std::vector<Image> input_images(params.batch_size);
    CHECK_CUDA(cudaStreamCreateWithFlags(&params.stream, cudaStreamNonBlocking));

    if (read_batch_images(images, height, width, input_images, params))
    {
        return EXIT_FAILURE;
    }

    // encode images
    double time = 0; // perf
    // typedef std::vector<std::vector<unsigned char>> BitStreamData
    BitStreamData bitsteam_output(params.batch_size);

    if (encode_images(input_images.data(), bitsteam_output, time, params))
    {
        return EXIT_FAILURE;
    }

    CHECK_CUDA(cudaStreamDestroy(params.stream));

    // free encoder resources
    CHECK_NVJPEG2K(nvjpeg2kEncodeParamsDestroy(params.enc_params));
    CHECK_NVJPEG2K(nvjpeg2kEncodeStateDestroy(params.enc_state));
    CHECK_NVJPEG2K(nvjpeg2kEncoderDestroy(params.enc_handle));

    return images[0];
}
