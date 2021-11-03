#include <cuda_runtime.h>

// NOTE: if you include stdio.h, you can use printf inside your kernel

#include "common.h"
#include "matrix.h"
#include "mul_gpu.h"
#include "stdio.h"

/**
 * Returns the global x-thread-id.
 */
__device__ unsigned int tidx() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

/**
 * Returns the global y-thread-id.
 */
__device__ unsigned int tidy() {
    return blockIdx.y * blockDim.y + threadIdx.y;
}

/**
 * See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c
 * for information on what this actually does.
 *
 * Also this helpful stackoverflow answer: https://stackoverflow.com/a/16119944/13728996 :)
 *
 * @param elements
 * @param tid
 * @param pos
 * @return
 */
__device__ float get_pitched_memory_float(const float* elements, unsigned int pitch, unsigned int row, unsigned int column) {
    return ((float*) ((char*) elements + row * pitch))[column];
}

__device__ void put_pitched_memory_float(const float* elements, unsigned int pitch, unsigned int row, unsigned int column, float data) {
    ((float*) ((char*) elements + row * pitch))[column] = data;
}

// NOTE: This is stupidly inefficient. Multiplying two 10000x10000 matrices takes 32s on my rtx2070!!!!!!!!!!!
// Thirty. Two. Seconds.
// I mean sure because there's absolutely no shared memory usage but still...
__global__ void matrix_mul_gpu_kernel(GPUMatrix m, GPUMatrix n, GPUMatrix p) {
    float sum = 0;
    if (tidx() < p.height && tidy() < p.width) { // only do a calculation if our thread index is inside our p-matrix
        for (int i = 0; i < m.width; i++) {
            float data_m = get_pitched_memory_float(m.elements, m.pitch, tidx(), i);
            float data_n = get_pitched_memory_float(n.elements, n.pitch, i, tidy());
            sum += data_m * data_n;
        }
        put_pitched_memory_float(p.elements, p.pitch, tidx(), tidy(), sum);
    }

#ifdef CRAPPY_DEBUG
    // debug check when computing an identity matrix, as matrix field [tidx][tidy] must always be 1 when multiplying
    // two identity matrices (obviously this only applies when the thread handling this is inside the matrix
    // dimensions, otherwise the result is garbage or (more often than not) 0)
    // TLDR: if this outputs all one's when multiplying Ident*Ident we're good
    if (tidx() == tidy()) {
        printf("[%d, %d] => %f\n", tidx(), tidy(), sum);
    }
#endif
}

void matrix_mul_gpu(const GPUMatrix &m, const GPUMatrix &n, GPUMatrix &p) {
	// Note: in the absolute worst case, this leads to 31x31 useless threads.
    // This could be prevented if I'd handle the lower right corner of the solution matrix with a special kernel
    // call only firing one block with remaining-x*remaining-y threads.
    // In this case I didn't implement that though ¯\_(ツ)_/¯
    dim3 dim_block(32, 32);
    dim3 dim_grid(div_up(p.height, 32), div_up(p.width, 32));
    std::cout << "\tBlock size: [x=" << 32 << " y=" << 32 << "] threads" << std::endl;
    std::cout << "\tGrid size:  [x=" << dim_grid.x << " y=" << dim_grid.y << "] blocks" << std::endl;
    std::cout << "\tTotal threads in x-direction: " << dim_grid.x * 32 << std::endl;
    std::cout << "\tTotal threads in y-direction: " << dim_grid.y * 32 << std::endl;

    // Note: for small 1k*1k matrices, one grid like this is enough. For larger matrices, we may need multiple grids, ideally
    // running on multiple devices or something
    matrix_mul_gpu_kernel<<<dim_grid, dim_block>>>(m, n, p); CUDA_CHECK_ERROR;
}
