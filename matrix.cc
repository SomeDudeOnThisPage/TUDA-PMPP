#include <iostream>
#include <iomanip>
#include <cstdlib>

#include <cuda_runtime.h>

#include "common.h"
#include "matrix.h"
#include "test.h"


CPUMatrix matrix_alloc_cpu(int width, int height) {
	CPUMatrix m{};
	m.width = width;
	m.height = height;
	m.elements = new float[m.width * m.height];
	return m;
}
void matrix_free_cpu(CPUMatrix &m) {
	delete[] m.elements;
}

GPUMatrix matrix_alloc_gpu(int width, int height) {
	GPUMatrix m{};
    m.width = width;
    m.height = height;

    size_t pitch = 0;
    cudaMallocPitch((void**)&m.elements, &pitch, width * sizeof(float), height); CUDA_CHECK_ERROR;
#ifdef CRAPPY_DEBUG
    std::cout << "PITCH=" << m.pitch << std::endl;
#endif
    m.pitch = pitch;

    return m;
}

void matrix_free_gpu(GPUMatrix &m) {
    cudaFree(&m.elements); // CUDA_CHECK_ERROR;
}

void matrix_upload(const CPUMatrix &src, GPUMatrix &dst) {
    cudaMemcpy2D(dst.elements, dst.pitch, src.elements, src.width * sizeof(float), src.width * sizeof(float), src.height, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR;
}

void matrix_download(const GPUMatrix &src, CPUMatrix &dst) {
    cudaMemcpy2D(dst.elements, dst.width * sizeof(float), src.elements, src.pitch, src.width * sizeof(float), src.height, cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR;
}

void matrix_compare_cpu(const CPUMatrix &a, const CPUMatrix &b) {
	// Note: should we do this with cuda as well? I've just done this on the CPU now...
    if (a.width != b.width || a.height != b.height) {
        std::cerr << "[COMPARE] matrix dimension mismatch" << std::endl;
        return;
    }

    bool mismatch = false;
    for (int i = 0; i < pmpp::P_WIDTH * pmpp::P_HEIGHT; i++) {
        if (a.elements[i] != b.elements[i]) {
            std::cerr << std::setprecision(10) << "[COMPARE] [" << i << "] mismatch: CPU="
            << a.elements[i] << ", GPU="
            << b.elements[i] << std::endl;
            mismatch = true;
        }
    }

    if (!mismatch) {
        std::cout << "[COMPARE] CPU and GPU solution matrices are identical :D" << std::endl;
    }
}
