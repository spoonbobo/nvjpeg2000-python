# use PIL to decode image into RGB format and use cuda_encode to encode it into JPEG2000
from cudaext import NvJpegEncoder
import numpy as np
from PIL import Image
from icecream import ic
import time
from nvjpeg import NvJpeg
nj = NvJpeg()

ic.configureOutput(
    includeContext=False, contextAbsPath=False, prefix=""
)

nj_encoder = NvJpegEncoder()

height = []
width = []
images = []

image_paths = ["images/cat.jpg"]
for path in image_paths:
    image = np.array(Image.open(path)) # unsigned char: [0,255]

    image_height = image.shape[0]
    image_width = image.shape[1]
    ic(image.shape)

    height.append(image_height)
    width.append(image_width)
    images.append(image)

height = np.array(height).astype(np.int32)
width = np.array(width).astype(np.int32)

# in practise there might be varying batch-sizes
# to enable batching in cuda_encode, we need to flatten all images,
# concat them into single array with their heights info
# flatten an array takes very little of compute time
images_flatten = np.array([]).astype(np.uint8)
for img in images:
    images_flatten = np.append(images_flatten, img.flatten()).astype(np.uint8)

# ic(len(images_flatten), height, width)
dog = images[0]

for i in range(1):
    t_nj = time.perf_counter()
    nj.encode(images[0])
    t_nj_elapsed = time.perf_counter() - t_nj

    ic(f"nvjpegPk: {t_nj_elapsed:.10f}")

    t_nj2k = time.perf_counter()
    nj_encoder.encodeJpeg2k(images_flatten, len(images), height, width, 0)
    t_nj2k_elapsed = time.perf_counter() - t_nj2k

    ic(f"nvjpeg2k(v1): {t_nj2k_elapsed:.10f} ({t_nj_elapsed / t_nj2k_elapsed:.5f}x faster/ { t_nj2k_elapsed / t_nj_elapsed:.5f}x slower)")

    # nvjpeg2k v2
    t_nj2kv2 = time.perf_counter()
    r = nj_encoder.encodeJpeg2kImageViewSingleBatch(dog, 0)
    t_nj2k_elapsed = time.perf_counter() - t_nj2kv2

    ic(f"nvjpeg2k(v1): {t_nj2k_elapsed:.10f} ({t_nj_elapsed / t_nj2k_elapsed:.5f}x faster/ { t_nj2k_elapsed / t_nj_elapsed:.5f}x slower)")