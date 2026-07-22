/*
 * The entire GPU side of qubit in one translation unit. Building this
 * file with nvcc and -DQUBIT_CUDA is the ONLY extra step a GPU user
 * takes; CPU-only users just include the header and compile nothing.
 *
 *   nvcc -O2 -std=c++17 -arch=sm_86 -DQUBIT_CUDA \
 *        -I include your_app.cpp src/qubit_gpu.cu -o your_app
 */
#include "backend_gpu.cu"
#include "blocks_gpu.cu"
