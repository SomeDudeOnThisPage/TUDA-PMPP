#pragma once

#include <cstdlib>

struct CPUMatrix {
	int width;
	int height;
	float *elements;
};
struct GPUMatrix {
	int width;
	int height;
	size_t pitch; // row size in bytes
	float *elements;
};


CPUMatrix matrix_alloc_cpu(int width, int height);
void matrix_free_cpu(CPUMatrix &m);

void matrix_compare_cpu(const CPUMatrix &a, const CPUMatrix &b);

GPUMatrix matrix_alloc_gpu(int width, int height);
void matrix_free_gpu(GPUMatrix &m);

void matrix_upload(const CPUMatrix &src, GPUMatrix &dst);
void matrix_download(const GPUMatrix &src, CPUMatrix &dst);