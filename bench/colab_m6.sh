#!/bin/bash
# M6 second-GPU data point on a free cloud GPU (Google Colab / Kaggle).
# Paste into a notebook cell prefixed with ! , or run: bash colab_m6.sh
# Needs: nvcc (preinstalled on Colab/Kaggle GPU runtimes) and a GPU runtime.
set -e

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

# detect compute capability -> sm_XX
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
ARCH="sm_${CC}"
echo "building for arch=${ARCH}"

if [ ! -d qubit ]; then git clone --depth 1 https://github.com/ArubikU/qubit.git; fi
cd qubit

NVCC_FLAGS="-O2 -std=c++17 -arch=${ARCH} -DQUBIT_CUDA -Xcompiler -fopenmp -I include"

echo "=== compiling ==="
nvcc $NVCC_FLAGS bench/benchsuite.cpp src/qubit_gpu.cu -o benchsuite_gpu
nvcc $NVCC_FLAGS tests/difftest_gpu.cpp src/qubit_gpu.cu -o difftest_gpu
nvcc $NVCC_FLAGS tests/blocks_gpu_test.cpp src/qubit_gpu.cu -o blocks_gpu_test
nvcc $NVCC_FLAGS bench/curve.cpp src/qubit_gpu.cu -o curve

echo "=== correctness (GPU vs CPU, seeded) ==="
./difftest_gpu 50
./blocks_gpu_test 25

echo "=== dense-gpu benchmark (median of 5) ==="
./benchsuite_gpu 28 gpu 5

echo "=== capacity: echo + GHZ past the dense ceiling ==="
./curve 28
echo "done"
