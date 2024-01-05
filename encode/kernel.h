#include <cuda_runtime_api.h>
#include <nvjpeg2k.h>

float gpu_encode(unsigned char* image, int batch_size,
                    int height, int width);

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
    nvjpeg2kEncoder_t enc_handle;
    nvjpeg2kEncodeState_t enc_state;
    nvjpeg2kEncodeParams_t enc_params;
};