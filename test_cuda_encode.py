# use PIL to decode image into RGB format and use cuda_encode to encode it into JPEG2000
from cudaext import cuda_encode
import numpy as np
from PIL import Image
from icecream import ic

height = []
width = []
images = []

image_paths = ["images/dog.jpeg", "images/cat.jpg"]
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
print(images[0][0][0])
# in practise there might be varying batch-sizes
# to enable batching in cuda_encode, we need to flatten all images,
# concat them into single array with their heights info
# flatten an array takes very little of compute time
images_flatten = np.array([]).astype(np.uint8)
for img in images:
    images_flatten = np.append(images_flatten, img.flatten())

ic(len(images_flatten), height, width)
ic(cuda_encode(images_flatten, len(images), height, width, 0))
