#include "matrix.h"
#include "mul_cpu.h"
#include "common.h"

void matrix_mul_cpu(const CPUMatrix &m, const CPUMatrix &n, CPUMatrix &p) {
    for (unsigned int i = 0; i < p.height; i++) {        // rows
        for (unsigned int j = 0; j < p.width; j++) {     // columns
            // calculate line*row for line i and row j => position ij in p matrix
            float total_ij = 0;
            for (unsigned int l = 0; l < m.width; l++) { // iterate current row
                total_ij += (m.elements[i * m.width + l] * n.elements[l * n.width + j]);
            }
            p.elements[i * p.width + j] = total_ij;
#ifdef CRAPPY_DEBUG
            std::cout << "[" << i << "," << j << "]: " << total_ij << std::endl;
#endif
        }
    }
}
