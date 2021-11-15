#include <stdexcept>
#include <iostream>
#include <cmath>

#include <cuda_runtime.h>

#include "common.h"

# define M_PI 3.14159265358979323846

filterkernel_gpu::filterkernel_gpu(const filterkernel_gpu& old) : ks(old.ks), data(old.data) {
    // see class definition on why I did this...
    this->weird_cpp_destructor_behavior_bullshit = true;
}

filterkernel_gpu::filterkernel_gpu(int size) : ks(size) {
    cudaMalloc<float>(&this->data, size * sizeof(float));
    CUDA_CHECK_ERROR;
}

filterkernel_gpu::~filterkernel_gpu() {
    // see class definition on why I did this...
    if (!this->weird_cpp_destructor_behavior_bullshit) {
        cudaFree(this->data); CUDA_CHECK_ERROR;
    }
}

void filterkernel_cpu::upload(filterkernel_gpu &dst) const {
    cudaMemcpy(dst.data, this->data, dst.ks * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR;
}

void filterkernel_cpu::upload_cmem() const {
    std::cout << "no. use upload_filterkernel in conv_gpu instead" << std::endl;
    /*if (this->ks > 128) {
        std::cerr << "Cannot upload filterkernel to constant GPU memory - kernel size is too large." << std::endl;
    } else {
        cudaMemcpyToSymbol(&c_kernel_data, this->data, this->ks * sizeof(float));
        CUDA_CHECK_ERROR;
    }*/
}

// Initializes the given 1D kernel with a univariate Gaussian
filterkernel_cpu::filterkernel_cpu(int size) : ks(size), data(new float[size]) {
	if ((ks & 1) != 1 || ks < 1)
		throw std::runtime_error("Invalid kernel size");

	// Set the mean to lie in the middle of the kernel
	const float mu = (ks - 1) / 2.0f;

	// sigma is chosen so that the smallest kernel (k=3) will span +-0.8 sigma
	// of the normal distribution.
	// For larger kernels (k -> inf) sigma is chosen so that they span +-3
	// sigma.
	// Credit goes to OpenCV.
	const float sigma = (ks - 1) / (2.0f * 3.0f) + 0.8f;

	// PDF of the normal distribution
	const auto gaussian = [mu, sigma](const float x) -> float {
		const float v = x - mu;
		return std::exp(-v * v / (2.0 * sigma * sigma)) /
				(sigma * std::sqrt(2.0 * M_PI));
	};

	// Generate the kernel by point sampling the PDF at the pixel centers.
	// There are much better methods to do this but we don't need them here.
	float sum = 0.0f;
	for (int i = 0; i < ks; i++) {
		float temp = gaussian(i);
		data[i] = temp;
		sum += temp;
	}

	// Normalize kernel values
	const float rcpSum = 1.0f / sum;
	for (int i = 0; i < ks; i++) {
		data[i] *= rcpSum;
	}
}
