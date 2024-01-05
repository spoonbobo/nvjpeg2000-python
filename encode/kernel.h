#include <cuda_runtime_api.h>
#include <nvjpeg2k.h>
#include <icecream.hpp>

float gpu_encode(unsigned char* image, int batch_size,
                    int height, int width, int dev);

#define CHECK_CUDA(call)                                                                                          \
    {                                                                                                             \
        cudaError_t _e = (call);                                                                                  \
        if (_e != cudaSuccess)                                                                                    \
        {                                                                                                         \
            std::cout << "CUDA Runtime failure: '#" << _e << "' at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return EXIT_FAILURE;                                                                                  \
        }                                                                                                         \
    }

#define CHECK_NVJPEG2K(call)                                                                                \
    {                                                                                                       \
        nvjpeg2kStatus_t _e = (call);                                                                       \
        if (_e != NVJPEG2K_STATUS_SUCCESS)                                                                  \
        {                                                                                                   \
            std::cout << "NVJPEG failure: '#" << _e << "' at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return EXIT_FAILURE;                                                                            \
        }                                                                                                   \
    }

struct encode_params_t {
    int batch_size;
    int height;
    int width;
    int cblk_w;
    int cblk_h;
    int dev;
    int verbose;
    nvjpeg2kEncoder_t enc_handle;
    nvjpeg2kEncodeState_t enc_state;
    nvjpeg2kEncodeParams_t enc_params;
};