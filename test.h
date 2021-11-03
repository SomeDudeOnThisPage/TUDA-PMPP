#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <stdexcept>

#include "matrix.h"

namespace pmpp {
	extern int M_WIDTH, M_HEIGHT, N_WIDTH, N_HEIGHT, P_WIDTH, P_HEIGHT;
	extern std::vector<float> m_data, n_data, p_data;
	extern std::random_device rd;
	extern std::default_random_engine re;
	inline void alloc()
	{
		m_data.resize(M_WIDTH * M_HEIGHT);
		n_data.resize(N_WIDTH * N_HEIGHT);
		p_data.resize(P_WIDTH * P_HEIGHT, 0.f);
	}
	inline float frand(float min, float max)
	{
		return std::uniform_real_distribution<float>(min, max)(re);
	}
	inline int rand(int min, int max)
	{
		return std::uniform_int_distribution<int>(min, max)(re);
	}
	inline void load(bool random = false)
	{
		float m_3x3[] = { 1, 2, 3, 4, 5.1245, 6, 7, 8, 9 };
		float n_3x3[] = { 10, 2.124563, 4, 6, 3.12467547848, 4, 7, 1, 10 };
		if (random) {
			M_WIDTH = rand(100, 1000);
			M_HEIGHT = rand(100, 1000);
			N_WIDTH = rand(100, 1000);
		} else {
			M_WIDTH = M_HEIGHT = N_WIDTH = 3;
		}
		N_HEIGHT = M_WIDTH; P_WIDTH = N_WIDTH; P_HEIGHT = M_HEIGHT;
		alloc();
		std::vector<float> tmp(n_data.size());
		for (int i = 0; i < M_WIDTH * M_HEIGHT; ++i) m_data[i] = random ? frand(0, 1) : m_3x3[i];
		for (int i = 0; i < N_WIDTH * N_HEIGHT; ++i) n_data[i] = tmp[(i % N_WIDTH) * N_HEIGHT + (i / N_WIDTH)] = random ? frand(0, 1) : n_3x3[i];
		for (int i = 0; i < P_WIDTH * P_HEIGHT; ++i) for (int e = 0; e < N_HEIGHT; ++e) p_data[i] += m_data[e + (i / P_WIDTH) * M_WIDTH] * tmp[e + (i % P_WIDTH) * N_HEIGHT];
	}
	inline void fill(CPUMatrix &m, CPUMatrix &n)
	{
		if (m.width != M_WIDTH || m.height != M_HEIGHT || n.width != N_WIDTH || n.height != N_HEIGHT) throw std::runtime_error("Cannot fill input matrices. Wrong dimensions");
		std::copy(m_data.begin(), m_data.end(), m.elements);
		std::copy(n_data.begin(), n_data.end(), n.elements);
	}
	inline void test(const CPUMatrix &p)
	{
		if (p.width != P_WIDTH || p.height != P_HEIGHT) throw std::runtime_error("Cannot check output matrix. Wrong dimensions");
		int e = 0;
		for (int i = 0; i < p.height * p.width; ++i) {
			if (std::fabs(p.elements[i] - p_data[i]) < 10e-3) continue;
			std::cerr << "Wrong value detected: Should be " << p_data[i] << " but is " << p.elements[i] << ". column " << (i % p.width) << ", row " << (i / p.width) << std::endl;
			++e;
			if (e == 20) {
				std::cerr << "Trucated error list after 20 lines." << std::endl;
			}
		}
		if (e != 0) throw std::runtime_error("Matrix value error");
	}
	inline void test_cpu(const CPUMatrix &p)
	{
		std::cout << "Starting CPU result check" << std::endl;
		test(p);
		std::cout << "CPU Result check successful" << std::endl;
	}
	inline void test_gpu(const CPUMatrix &p)
	{
		std::cout << "Starting GPU result check" << std::endl;
		test(p);
		std::cout << "GPU Result check successful" << std::endl;
	}

};
