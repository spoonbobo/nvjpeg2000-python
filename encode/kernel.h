#include <vector>

#include <cuda_runtime_api.h>
#include <nvjpeg2k.h>

float gpu_encode(unsigned char* image, int batch_size,
                    int *height, int *width, int dev);
                    
typedef std::vector<std::vector<unsigned char>> BitStreamData;

struct Image
{
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
    /*
    typedef struct
    {
        uint32_t component_width;
        uint32_t component_height;
        uint8_t  precision;
        uint8_t  sgn;
    } nvjpeg2kImageComponentInfo_t;
    */
    std::vector<nvjpeg2kImageComponentInfo_t> comp_info_;

    // initialize
    Image()
    {
    }

    // clean memory
    ~Image()
    {
        // for (auto &ptr : pixel_data_d_)
        // {
        //     if (ptr)
        //     {
        //         cudaFree(ptr);
        //         ptr = nullptr;
        //     }
        // }

        // for (auto &ptr : pixel_data_h_)
        // {
        //     if (ptr)
        //     {
        //         free(ptr);
        //         ptr = nullptr;
        //     }
        // }
    }
};