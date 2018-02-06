#!/usr/bin/env bash
TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")

cd src
echo "Compiling resample2d kernels by nvcc..."
rm Resample2d_kernel.o
rm -r ../_ext

nvcc -c -o Resample2d_kernel.o Resample2d_kernel.cu -x cu -Xcompiler -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC \
			   -gencode arch=compute_35,code=sm_35 \
			   -gencode arch=compute_37,code=sm_37 \
			   -gencode arch=compute_52,code=sm_52 \
			   -gencode arch=compute_60,code=sm_60 

cd ../
python build.py
