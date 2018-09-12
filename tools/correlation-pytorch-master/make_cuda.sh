#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda-8.0

cd correlation-pytorch/correlation_package/src
echo "Compiling correlation layer kernels by nvcc..."

# TODO (JEB): Check which arches we need
nvcc -c -o corr_cuda_kernel.cu.o corr_cuda_kernel.cu -x cu -Xcompiler -fPIC \
                           -gencode arch=compute_35,code=sm_35 \
                           -gencode arch=compute_37,code=sm_37 \
                           -gencode arch=compute_52,code=sm_52 \
                           -gencode arch=compute_61,code=sm_61 

nvcc -c -o corr1d_cuda_kernel.cu.o corr1d_cuda_kernel.cu -x cu -Xcompiler -fPIC \
                           -gencode arch=compute_35,code=sm_35 \
                           -gencode arch=compute_37,code=sm_37 \
                           -gencode arch=compute_52,code=sm_52 \
                           -gencode arch=compute_61,code=sm_61 

cd ../../
python setup.py build install
