#pragma once

#include <stdlib.h>

struct image_gpu {
	int width;
	int height;
	size_t pitch;
	unsigned int *data;

    // Alright look. You cannot pass by reference to a cuda kernel, thus, pass by value is used.
    // This uses function scope, and the destructor is called.
    // This leads to a segfault because the destructor tries to free the cuda memory.
    // So I set this to true when the object in question was copied, and check it when trying to free cuda memory...
    // This is literally the worst way I could have done this, but at this point I don't care anymore.
    bool weird_cpp_destructor_behavior_bullshit = false;

// #ifndef __CUDACC__
	// If you want, you can implement this stuff
	image_gpu(const image_gpu&);
	// image_gpu &operator=(const image_gpu&) = delete;
// #endif

	// Allocate and free gpu memory
	image_gpu(int w, int h);
	~image_gpu();
};

struct image_cpu {
	int width, height;
	unsigned int *data;

#ifndef __CUDACC__
	// If you want, you can implement this stuff
	image_cpu(const image_cpu&) = delete;
	image_cpu &operator=(const image_cpu&) = delete;
#endif

	image_cpu(int w, int h) : width(w), height(h), data(new unsigned int[w * h]) {}

	// Load and save methods
	image_cpu(const char *fn);
	void save(const char *fn) const;

	// Upload and download to GPU
	void upload(image_gpu &dst) const;
	void download(const image_gpu &src);

	~image_cpu() {
		delete[] data;
	}
};
