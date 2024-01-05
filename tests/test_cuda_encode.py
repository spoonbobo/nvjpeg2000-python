from cudaext import cuda_encode
import numpy as np
from icecream import ic

a = np.random.randn(3,4).astype(np.float32)
ic(a)
ic(cuda_encode(a,a,a))