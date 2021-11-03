#pragma once

#include "matrix.h"

void matrix_mul_gpu(const GPUMatrix &m, const GPUMatrix &n, GPUMatrix &p);
