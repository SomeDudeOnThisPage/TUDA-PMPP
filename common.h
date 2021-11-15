#pragma once

#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>

#define CUDA_CHECK_ERROR                                                       \
    do {                                                                       \
        const cudaError_t err = cudaGetLastError();                            \
        if (err != cudaSuccess) {                                              \
            const char *const err_str = cudaGetErrorString(err);               \
            std::cerr << "Cuda error in " << __FILE__ << ":" << __LINE__ - 1   \
                      << ": " << err_str << " (" << err << ")" << std::endl;   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while(0)


inline unsigned int div_up(unsigned int numerator, unsigned int denominator)
{
	unsigned int result = numerator / denominator;
	if (numerator % denominator) ++result;
	return result;
}

struct filterkernel_gpu {
    int ks;
    float *data;

    // Alright look. You cannot pass by reference to a cuda kernel, thus, pass by value is used.
    // This uses function scope, and the destructor is called.
    // This leads to a segfault because the destructor tries to free the cuda memory.
    // So I set this to true when the object in question was copied, and check it when trying to free cuda memory...
    // This is literally the worst way I could have done this, but at this point I don't care anymore.
    bool weird_cpp_destructor_behavior_bullshit = false;

// #ifndef __CUDACC__
    // If you want, you can implement this stuff
    filterkernel_gpu(const filterkernel_gpu&);
    filterkernel_gpu &operator=(const filterkernel_gpu&) = delete;
// #endif

    filterkernel_gpu(int ks);
    ~filterkernel_gpu();
};

struct filterkernel_cpu {
    int ks;
    float *data;

#ifndef __CUDACC__
	// If you want, you can implement this stuff
	filterkernel_cpu(const filterkernel_cpu&) = delete;
	filterkernel_cpu &operator=(const filterkernel_cpu&) = delete;
#endif

	filterkernel_cpu(int size);
	~filterkernel_cpu()
	{
		delete[] data;
	}

    void upload(filterkernel_gpu &dst) const;
    void upload_cmem() const;
};

