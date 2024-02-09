#include <iostream>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <fstream>

#include <cuda_runtime_api.h>
#include <nvjpeg2k.h>

#include "nvjpeg2k_encoder.h"
#include <chrono>

/**
 * encode MemoryView arrays into nvJPEG2000 bitstream
 *
 * @param images MemoryView, image(s) given batch_size
 * @param batch_size batch_size
 * @param height height(s) of image(s)
 * @param width width(s) of image(s)
 * @param dev device used for GPU encoding
 * @return nvJPEG2000 bitstream
 * @todo pass in image shapes
 */
float encodeJpeg2k_(unsigned char *images, int batch_size,
                    int *height, int *width, int dev)

{
    nvjpeg2kEncoder_t enc_handle;      // nvJPEG2000 encoder handle
    nvjpeg2kEncodeState_t enc_state;   // store the encoder work buffers and intermediate results
    nvjpeg2kEncodeParams_t enc_params; // stores various parameters that control the compressed output

    // set device
    check_cuda(cudaSetDevice(dev));

    // initialize the library handles
    check_nvjpeg2k(nvjpeg2kEncoderCreateSimple(&enc_handle));
    check_nvjpeg2k(nvjpeg2kEncodeStateCreate(enc_handle, &enc_state));
    check_nvjpeg2k(nvjpeg2kEncodeParamsCreate(&enc_params));

    // non-blocking stream
    cudaStream_t stream;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    int offset = 0;
    std::vector<Image *> input_images;
    input_images.resize(batch_size, nullptr);
    BitStreamData bitstreams(batch_size);

    bool write_output = true;

    for (int batch_id = 0; batch_id < batch_size; batch_id++)
    {
        input_images[batch_id] = new Image(&images[offset], width[batch_id], height[batch_id], 3, NVJPEG2K_UINT8);

        check_nvjpeg2k(nvjpeg2kEncodeParamsSetEncodeConfig(enc_params, &input_images[batch_id]->enc_config));
        check_nvjpeg2k(nvjpeg2kEncodeParamsSetQuality(enc_params, 25));

        auto start = std::chrono::steady_clock::now();
        check_nvjpeg2k(nvjpeg2kEncode(enc_handle, enc_state, enc_params, &input_images[batch_id]->image_d_, stream));
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "V1 Encode Elapsed time: " << elapsed_seconds.count() << "s\n";

        size_t compressed_size;
        check_nvjpeg2k(nvjpeg2kEncodeRetrieveBitstream(enc_handle, enc_state, NULL, &compressed_size, stream));
        bitstreams[batch_id].resize(compressed_size);
        check_nvjpeg2k(nvjpeg2kEncodeRetrieveBitstream(enc_handle, enc_state, bitstreams[batch_id].data(), &compressed_size, stream));

        check_cuda(cudaStreamSynchronize(stream));

        // if (write_output)
        // {
        //     std::string fname("image" + std::to_string(batch_id)+".jp2");
        //     std::ofstream bitstream_file(fname,

        //                                  std::ios::out | std::ios::binary);
        //     bitstream_file.write((char *)bitstreams[batch_id].data(), compressed_size);
        //     bitstream_file.close();
        // }

        offset += width[batch_id] * height[batch_id] * 3;
    }

    check_cuda(cudaStreamDestroy(stream));

    // free image&encoder resources
    for (auto &img : input_images)
    {
        delete img;
    }

    check_nvjpeg2k(nvjpeg2kEncodeParamsDestroy(enc_params));
    check_nvjpeg2k(nvjpeg2kEncodeStateDestroy(enc_state));
    check_nvjpeg2k(nvjpeg2kEncoderDestroy(enc_handle));

    return EXIT_SUCCESS;
}

float encodeJpeg2kImageViewSingleBatch_(
    unsigned char *r, unsigned char *g, unsigned char *b,
    int height, int width, int dim, int dev)
{
    nvjpeg2kEncoder_t enc_handle;      // nvJPEG2000 encoder handle
    nvjpeg2kEncodeState_t enc_state;   // store the encoder work buffers and intermediate results
    nvjpeg2kEncodeParams_t enc_params; // stores various parameters that control the compressed output

    // set device
    check_cuda(cudaSetDevice(dev));

    // initialize the library handles
    check_nvjpeg2k(nvjpeg2kEncoderCreateSimple(&enc_handle));
    check_nvjpeg2k(nvjpeg2kEncodeStateCreate(enc_handle, &enc_state));
    check_nvjpeg2k(nvjpeg2kEncodeParamsCreate(&enc_params));

    ImageV2 *input_image = new ImageV2(r, g, b, width, height, dim, NVJPEG2K_UINT8);
    BitStreamData bitstreams(1);

    check_nvjpeg2k(nvjpeg2kEncodeParamsSetEncodeConfig(enc_params, &input_image->enc_config));
    // check_nvjpeg2k(nvjpeg2kEncodeParamsSetQuality(enc_params, 25));
    check_nvjpeg2k(nvjpeg2kEncode(enc_handle, enc_state, enc_params, &input_image->image_d_, nullptr));

    size_t compressed_size;
    check_nvjpeg2k(nvjpeg2kEncodeRetrieveBitstream(enc_handle, enc_state, NULL, &compressed_size, nullptr));
    bitstreams[0].resize(compressed_size);
    check_nvjpeg2k(nvjpeg2kEncodeRetrieveBitstream(enc_handle, enc_state, bitstreams[0].data(), &compressed_size, nullptr));
    // release resources
    delete input_image;

    check_nvjpeg2k(nvjpeg2kEncodeParamsDestroy(enc_params));
    check_nvjpeg2k(nvjpeg2kEncodeStateDestroy(enc_state));
    check_nvjpeg2k(nvjpeg2kEncoderDestroy(enc_handle));

    return EXIT_SUCCESS;
}