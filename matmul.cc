#include <iostream>

// #ifdef __CUDACC__
    #include <cuda_runtime.h>
// #endif

#include "matrix.h"
#include "matmul.h"
#include "test.h"
#include "common.h"
#include "mul_cpu.h"
#include "mul_gpu.h"
#include "timer.h"

#define KHZ_TO_GHZ_DIVISOR 1000000.0f
#define B_TO_MIB_DIVISOR 1048576.0f
#define B_TO_KIB_DIVISOR 1024.0f

void print_cuda_devices() {
//#ifdef __CUDACC__
    int32_t dev_count;
    cudaGetDeviceCount(&dev_count); CUDA_CHECK_ERROR;

    cudaDeviceProp p{};
    for (int32_t i = 0; i < dev_count; i++) {
        cudaGetDeviceProperties(&p, i); CUDA_CHECK_ERROR;

        std::cout << std::endl;
        std::cout << "----------CUDA DEVICE ATTRIBUTES #" << i << "----------" << std::endl;
        std::cout << "\tDevice-Designation:  " << p.name << std::endl;
        std::cout << "\tCC-Version:          " << p.major << "." << p.minor << std::endl;
        std::cout << "\tMultiprocessors:     " << p.multiProcessorCount << std::endl;
        std::cout << "\tClock Rate:          " << (float) p.clockRate / KHZ_TO_GHZ_DIVISOR << "GHz" << std::endl;
        std::cout << "\tGlobal Memory:       " << p.totalGlobalMem / B_TO_MIB_DIVISOR << "MiB" << std::endl;
        std::cout << "\tL2 Cache Size:       " << p.l2CacheSize / B_TO_KIB_DIVISOR << "KiB" << std::endl;
        std::cout << std::endl;
    }
//#else
//    std::cout << "You must compile the program using nvcc to list devices" << std::endl;
//#endif
}

void matmul(bool compare) {
    std::cout << "----------BEGIN CPU MULTIPLICATION----------" << std::endl;
    std::cout << "\t[M] Dimensions: w=" << pmpp::M_WIDTH << ", h=" << pmpp::M_HEIGHT << std::endl;
    std::cout << "\t[N] Dimensions: w=" << pmpp::N_WIDTH << ", h=" << pmpp::N_HEIGHT << std::endl;
    std::cout << "\t[P] Dimensions: w=" << pmpp::P_WIDTH << ", h=" << pmpp::P_HEIGHT << std::endl;

    CPUMatrix m = matrix_alloc_cpu(pmpp::M_WIDTH, pmpp::M_HEIGHT);
    CPUMatrix n = matrix_alloc_cpu(pmpp::N_WIDTH, pmpp::N_HEIGHT);
    CPUMatrix solution_CPU = matrix_alloc_cpu(pmpp::P_WIDTH, pmpp::P_HEIGHT);

    pmpp::fill(m, n);
    timer_tp cpu_t0 = timer_now();
    matrix_mul_cpu(m, n, solution_CPU);
    timer_tp cpu_t1 = timer_now();
    float cpu_time = timer_elapsed(cpu_t0, cpu_t1);

    std::cout << "\tCPU: " << cpu_time << "ms" << std::endl;
    pmpp::test_cpu(solution_CPU);

//#ifdef __CUDACC__
    std::cout << "----------BEGIN GPU MULTIPLICATION----------" << std::endl;
    cudaSetDevice(0);
    GPUMatrix gm0 = matrix_alloc_gpu(pmpp::M_WIDTH, pmpp::M_HEIGHT);
    GPUMatrix gm1 = matrix_alloc_gpu(pmpp::N_WIDTH, pmpp::N_HEIGHT);
    GPUMatrix solution = matrix_alloc_gpu(pmpp::P_WIDTH, pmpp::P_HEIGHT);
    CPUMatrix solution_GPU = matrix_alloc_cpu(pmpp::P_WIDTH, pmpp::P_HEIGHT);

    matrix_upload(m, gm0);
    matrix_upload(n, gm1);
    matrix_upload(solution_GPU, solution);

    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);
    cudaEventRecord(evStart, nullptr);

    matrix_mul_gpu(gm0, gm1, solution);

    cudaEventRecord(evStop, nullptr);
    cudaEventSynchronize(evStop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, evStart, evStop);
    std::cout << "\tGPU: " << gpu_time << "ms" << std::endl;

    matrix_download(solution, solution_GPU);

    std::cout << "\tWow, it worked!" << std::endl;

    pmpp::test_gpu(solution_GPU);

    std::cout << "----------BEGIN GPU/CPU COMPARISON----------" << std::endl;
    std::cout << "\tComputation times: \n\t\tCPU: " << cpu_time << "ms\n\t\tGPU: " << gpu_time << "ms" << std::endl;
    std::cout << "\tDifference in computation time: " << abs(cpu_time - gpu_time) << "ms" << std::endl;

    // the precision of the cpu floats and gpu floats don't match, so every field that doesn't contain a "round" number
    // is outputted...
    // for big matrices this takes a long time, so I have omitted this, so I can run the program without
    // waiting half an hour for it to finish printing. THANKS A LOT rounding error of fma great job smh...
    if (compare) {
        matrix_compare_cpu(solution_CPU, solution_GPU);
    } else {
        std::cout << "\tElement comparison is disabled. Please run with parameter 'compare' to enable element comparison" << std::endl;
    }

    std::cout << "----------CLEANUP----------" << std::endl;
    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    std::cout << "\tDestroyed timer events..." << std::endl;

    matrix_free_gpu(gm0);
    matrix_free_gpu(gm1);
    matrix_free_gpu(solution);
    std::cout << "\tFreed GPU memory..." << std::endl;
//#endif

    matrix_free_cpu(m);
    matrix_free_cpu(n);
    matrix_free_cpu(solution_CPU);
    matrix_free_cpu(solution_GPU);
    std::cout << "\tFreed CPU memory..." << std::endl;
}


/************************************************************
 *
 * 
 * (Task 4) 6. Where do the differences come from?
 * 
 * Answer: The problem lies in the way floating point numbers
 *         are handled on the GPU and the CPU. Mainly, this
 *         boils down to the rounding error defined in the IEEE-754 standard, as well as the order the operations are
 *         performed in (in this case the order is irrelevant as I only perform one multiply operation though...).
 *         (see https://docs.nvidia.com/cuda/floating-point/index.html, esp. section 2 and section 3.2 Figure 5)
 *         (also https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions-appendix (appendix H))
 *         (also https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html#799)
 *
 * 
 ************************************************************/
