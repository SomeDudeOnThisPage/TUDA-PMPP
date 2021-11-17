#include <cstdint>
#include "conv_gpu.h"
#include <cuda_runtime.h>

// Note: i was waaay to lazy to actually adhere to the DRY principle so there's a lot of repetition in here...

namespace util {
    __device__ unsigned int tidx() {
        return blockIdx.x * blockDim.x + threadIdx.x;
    }

    __device__ unsigned int tidy() {
        return blockIdx.y * blockDim.y + threadIdx.y;
    }

    template <typename T>
    __device__ T get_pitched_memory(T* elements, unsigned int pitch, unsigned int row, unsigned int column) {
        return ((T*) ((uint8_t*) elements + row * pitch))[column];
    }

    template <typename T>
    __device__ void put_pitched_memory(T* elements, unsigned int pitch, unsigned int row, unsigned int column, T data) {
        ((T*) ((uint8_t*) elements + row * pitch))[column] = data;
    }

    __device__ void convolute(unsigned int data, float kernel_data, float* r, float* g, float* b) {
            *r += (float) (data & 0xff) * kernel_data;
            *g += (float) ((data >> 8) & 0xff) * kernel_data;
            *b += (float) ((data >> 16) & 0xff) * kernel_data;
    }

    __device__ unsigned int transform_ppm(const float* r, const float* g, const float* b) {
        return (int) (*r + 0.5f) | (int) (*g + 0.5f) << 8 | (int) (*b + 0.5f) << 16;
    }

    __host__ void print_grid_info(dim3 &dim_grid, dim3 &dim_block) {
        std::cout << "\tBlock size: [x=" << dim_block.x << " y=" << dim_block.y << "] threads" << std::endl;
        std::cout << "\tGrid size:  [x=" << dim_grid.x << " y=" << dim_grid.y << "] blocks" << std::endl;
        std::cout << "\tTotal threads in x-direction: " << dim_grid.x * dim_block.x << std::endl;
        std::cout << "\tTotal threads in y-direction: " << dim_grid.y * dim_block.y << std::endl;
    }
}

__constant__ float c_kernel_data[128];

__host__ void upload_filterkernel(filterkernel_cpu &kernel) {
    cudaMemcpyToSymbol(c_kernel_data, kernel.data, kernel.ks * sizeof(float));
    CUDA_CHECK_ERROR;
}

__global__ void conv_h_gpu_smem_kernel(const image_gpu dst, const image_gpu src, const filterkernel_gpu kernel, bool const_kernel) {
    __shared__ unsigned int s_data[8][256];

    int tidx = util::tidx();
    int tidy = util::tidy();

    // not using float here causes a load of fp-accuracy and int-division accuracy related problems
    float ks_half = kernel.ks / 2;
    int base = (int) (ks_half + threadIdx.x); // "base" address of the actual pixel moved by ks_half places

    if (threadIdx.x < ks_half) { // Load left apron pixel (or clamp to edge).
        s_data[threadIdx.y][threadIdx.x] = (tidx - ks_half >= 0)
                                           ? util::get_pitched_memory<unsigned int>(src.data, src.pitch, tidy, tidx - ks_half)
                                           : util::get_pitched_memory<unsigned int>(src.data, src.pitch, tidy, 0);
    }

    if (base >= blockDim.x) { // Load right apron pixel (or clamp to edge).
        s_data[threadIdx.y][(unsigned int) (base + ks_half)] = (tidx + ks_half < src.width)
                                                               ? util::get_pitched_memory<unsigned int>(src.data, src.pitch, tidy, tidx + ks_half)
                                                               : util::get_pitched_memory<unsigned int>(src.data, src.pitch, tidy, src.width - 1);
    }

    // Load actual pixel.
    s_data[threadIdx.y][base] = util::get_pitched_memory<unsigned int>(src.data, src.pitch, tidy, tidx);

    __syncthreads();

    float r = 0.0f, g = 0.0f, b = 0.0f;
    if (tidx < src.width && tidy < src.height) {
        // add up from -ks/2 to +ks/2 with origin as current pixel defined by tid
        for (int i = 0; i < kernel.ks; i++) {
            int x_offset = (threadIdx.x + ks_half + (i - kernel.ks / 2));
            x_offset = max(x_offset, 0);

            // bank conflicts
            unsigned int data = s_data[threadIdx.y][x_offset];
            // __syncwarp() // I was hoping this would fix the bank conflicts but nooooo
            util::convolute(data, (const_kernel) ? c_kernel_data[i] : kernel.data[i], &r, &g, &b);
        }
    }

    util::put_pitched_memory<unsigned int>(dst.data, dst.pitch, tidy, tidx, util::transform_ppm(&r, &g, &b));
}

__global__ void conv_v_gpu_smem_kernel(const image_gpu dst, const image_gpu src, const filterkernel_gpu kernel, bool const_kernel) {
    __shared__ unsigned int s_data[8][256];

    int tidx = util::tidx();
    int tidy = util::tidy();

    // not using float here causes a load of fp-accuracy and int-division accuracy related problems
    float ks_half = kernel.ks / 2;
    int base = (int) (ks_half + threadIdx.y); // "base" address of the actual pixel moved by ks_half places

    if (threadIdx.y < ks_half) { // Load upper apron pixel (or clamp to edge).
        s_data[threadIdx.x][threadIdx.y] = (tidy - ks_half >= 0)
            ? util::get_pitched_memory<unsigned int>(src.data, src.pitch, tidy - ks_half, tidx)
            : util::get_pitched_memory<unsigned int>(src.data, src.pitch, 0, tidx);
    }

    if (base >= blockDim.y) { // Load lower apron pixel (or clamp to edge).
        s_data[threadIdx.x][(unsigned int) (base + ks_half)] = (tidy + ks_half < src.height)
           ? util::get_pitched_memory<unsigned int>(src.data, src.pitch, tidy + ks_half, tidx)
           : util::get_pitched_memory<unsigned int>(src.data, src.pitch, src.height - 1, tidx);
    }

    // Load actual pixel.
    s_data[threadIdx.x][base] = util::get_pitched_memory<unsigned int>(src.data, src.pitch, tidy, tidx);

    __syncthreads();

    float r = 0.0f, g = 0.0f, b = 0.0f;
    if (tidx < src.width && util::tidy() < src.height) {
        for (int i = 0; i < kernel.ks; i++) {
            int y_offset = (threadIdx.y + ks_half + (i - kernel.ks / 2));
            y_offset = max(y_offset, 0);

            // bank conflicts
            unsigned int data = s_data[threadIdx.x][y_offset];
            // __syncwarp() // I was hoping this would fix the bank conflicts but nooooo
            util::convolute(data, (const_kernel) ? c_kernel_data[i] : kernel.data[i], &r, &g, &b);
        }
    }

    util::put_pitched_memory<unsigned int>(dst.data, dst.pitch, tidy, tidx, util::transform_ppm(&r, &g, &b));
}

__global__ void conv_h_gpu_gmem_kernel(const image_gpu dst, const image_gpu src, const filterkernel_gpu kernel, bool const_kernel) {
    unsigned int tidx = util::tidx();
    unsigned int tidy = util::tidy();

    float r = 0.0f, g = 0.0f, b = 0.0f;
    if (tidx < src.width && tidy < src.height) {
        // add up from -ks/2 to +ks/2 with origin as current pixel defined by tid
        for (int i = 0; i < kernel.ks; i++) {
            int x_offset = tidx + (i - kernel.ks / 2);
            x_offset = max(min(x_offset, src.width - 1), 0);

            unsigned int data = util::get_pitched_memory<unsigned int>(src.data, src.pitch, tidy, x_offset);
            util::convolute(data, (const_kernel) ? c_kernel_data[i] : kernel.data[i], &r, &g, &b);
        }

    }
    util::put_pitched_memory<unsigned int>(dst.data, dst.pitch, tidy, tidx, util::transform_ppm(&r, &g, &b));
}

__global__ void conv_v_gpu_gmem_kernel(const image_gpu dst, const image_gpu src, const filterkernel_gpu kernel, bool const_kernel) {
    unsigned int tidx = util::tidx();
    unsigned int tidy = util::tidy();

    float r = 0.0f, g = 0.0f, b = 0.0f;
    if (tidx < src.width && util::tidy() < src.height) {
        // add up from -ks/2 to +ks/2 with origin as current pixel defined by tid
        for (int i = 0; i < kernel.ks; i++) {
            int y_offset = tidy + (i - kernel.ks / 2);
            y_offset = max(min(y_offset, src.height - 1), 0);

            unsigned int data = util::get_pitched_memory<unsigned int>(src.data, src.pitch, y_offset, tidx);
            util::convolute(data, (const_kernel) ? c_kernel_data[i] : kernel.data[i], &r, &g, &b);
        }
    }
    util::put_pitched_memory<unsigned int>(dst.data, dst.pitch, tidy, tidx, util::transform_ppm(&r, &g, &b));
}

__host__ float conv_h_gpu_all(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel) {
    std::cout << "[SHARED + CONSTANT MEMORY] PERFORMING HORIZONTAL CONVOLUTION" << std::endl;

    dim3 dim_block(128, 8);
    dim3 dim_grid(div_up(src.width, 128), div_up(src.height, 8));
    util::print_grid_info(dim_grid, dim_block);

    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);
    cudaEventRecord(evStart, nullptr);

    conv_h_gpu_smem_kernel<<<dim_grid, dim_block>>>(dst, src, kernel, true);

    cudaEventRecord(evStop, nullptr);
    cudaEventSynchronize(evStop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, evStart, evStop);
    return gpu_time;
}

__host__ float conv_v_gpu_all(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel) {
    std::cout << "[SHARED + CONSTANT MEMORY] PERFORMING VERTICAL CONVOLUTION" << std::endl;

    dim3 dim_block(8, 128);
    dim3 dim_grid(div_up(src.width, 8), div_up(src.height, 128));
    util::print_grid_info(dim_grid, dim_block);

    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);
    cudaEventRecord(evStart, nullptr);

    conv_v_gpu_smem_kernel<<<dim_grid, dim_block>>>(dst, src, kernel, true);

    cudaEventRecord(evStop, nullptr);
    cudaEventSynchronize(evStop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, evStart, evStop);
    return gpu_time;
}

__host__ float conv_h_gpu_cmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel) {
    std::cout << "[CONSTANT MEMORY] PERFORMING HORIZONTAL CONVOLUTION" << std::endl;

    dim3 dim_block(32, 32);
    dim3 dim_grid(div_up(src.width, 32), div_up(src.height, 32));
    util::print_grid_info(dim_grid, dim_block);

    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);
    cudaEventRecord(evStart, nullptr);

    conv_h_gpu_gmem_kernel<<<dim_grid, dim_block>>>(dst, src, kernel, true);

    cudaEventRecord(evStop, nullptr);
    cudaEventSynchronize(evStop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, evStart, evStop);
    return gpu_time;
}

__host__ float conv_v_gpu_cmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel) {
    std::cout << "[CONSTANT MEMORY] PERFORMING VERTICAL CONVOLUTION" << std::endl;

    dim3 dim_block(32, 32);
    dim3 dim_grid(div_up(src.width, 32), div_up(src.height, 32));
    util::print_grid_info(dim_grid, dim_block);

    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);
    cudaEventRecord(evStart, nullptr);

    conv_v_gpu_gmem_kernel<<<dim_grid, dim_block>>>(dst, src, kernel, true);

    cudaEventRecord(evStop, nullptr);
    cudaEventSynchronize(evStop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, evStart, evStop);
    return gpu_time;
}

__host__ float conv_h_gpu_smem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel) {
    std::cout << "[SHARED MEMORY] PERFORMING HORIZONTAL CONVOLUTION" << std::endl;

    dim3 dim_block(128, 8);
    dim3 dim_grid(div_up(src.width, 128), div_up(src.height, 8));
    util::print_grid_info(dim_grid, dim_block);

    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);
    cudaEventRecord(evStart, nullptr);

    conv_h_gpu_smem_kernel<<<dim_grid, dim_block>>>(dst, src, kernel, false);

    cudaEventRecord(evStop, nullptr);
    cudaEventSynchronize(evStop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, evStart, evStop);
    return gpu_time;
}

__host__ float conv_v_gpu_smem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel) {
    std::cout << "[SHARED MEMORY] PERFORMING VERTICAL CONVOLUTION" << std::endl;

    dim3 dim_block(8, 128);
    dim3 dim_grid(div_up(src.width, 8), div_up(src.height, 128));
    util::print_grid_info(dim_grid, dim_block);

    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);
    cudaEventRecord(evStart, nullptr);

    conv_v_gpu_smem_kernel<<<dim_grid, dim_block>>>(dst, src, kernel, false);

    cudaEventRecord(evStop, nullptr);
    cudaEventSynchronize(evStop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, evStart, evStop);
    return gpu_time;
}

__host__ float conv_h_gpu_gmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel) {
    std::cout << "[GLOBAL MEMORY] PERFORMING HORIZONTAL CONVOLUTION" << std::endl;

    dim3 dim_block(32, 32);
    dim3 dim_grid(div_up(src.width, 32), div_up(src.height, 32));
    util::print_grid_info(dim_grid, dim_block);

    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);
    cudaEventRecord(evStart, nullptr);

    conv_h_gpu_gmem_kernel<<<dim_grid, dim_block>>>(dst, src, kernel, false);

    cudaEventRecord(evStop, nullptr);
    cudaEventSynchronize(evStop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, evStart, evStop);
    return gpu_time;
}

__host__ float conv_v_gpu_gmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel) {
    std::cout << "[GLOBAL MEMORY] PERFORMING VERTICAL CONVOLUTION" << std::endl;

    dim3 dim_block(32, 32);
    dim3 dim_grid(div_up(src.width, 32), div_up(src.height, 32));
    util::print_grid_info(dim_grid, dim_block);

    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);
    cudaEventRecord(evStart, nullptr);

    conv_v_gpu_gmem_kernel<<<dim_grid, dim_block>>>(dst, src, kernel, false);

    cudaEventRecord(evStop, nullptr);
    cudaEventSynchronize(evStop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, evStart, evStop);
    return gpu_time;
}

