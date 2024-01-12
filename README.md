# nvJPEG2000-python
Python interface for Encoding/Decoding using GPU with nvJPEG.

# Requirements
* [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit)
* [nvJPEG2000](https://developer.nvidia.com/nvjpeg)
* python3

# Setup
`pip install nyjpeg-python`

# Build
`export CUDAHOME=<your-cuda-path>`

`python3 setup.py build_ext --inplace -f`
