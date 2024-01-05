# use PIL to decode image into RGB format and use cuda_encode to encode it into JPEG2000

from cudaext import cuda_encode
import numpy as np
from PIL import Image
from icecream import ic

image_path = "images/dog.jpeg"
image = np.array(Image.open(image_path)) # int8
ic(image.shape)
ic(image[0][0][0])
ic(cuda_encode(image, 1, image.shape[1], image.shape[0]))
