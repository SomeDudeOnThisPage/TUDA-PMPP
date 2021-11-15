#include <limits>
#include <fstream>
#include <stdexcept>

#include <cuda_runtime.h>

#include "image.h"
#include "common.h"

image_gpu::image_gpu(const image_gpu& old) : height(old.height), width(old.width), pitch(old.pitch), data(old.data) {
    // see class definition on why I did this...
    weird_cpp_destructor_behavior_bullshit = true;
}

image_gpu::image_gpu(int w, int h) : width(w), height(h) {
    cudaMallocPitch((void**) &this->data, &this->pitch, w * sizeof(float), h);
    CUDA_CHECK_ERROR;

    std::cout << "ALLOCATION " << w << " " << h << " PITCH " << this->pitch << std::endl;
}

image_gpu::~image_gpu() {
    // see class definition on why I did this...
    if (!this->weird_cpp_destructor_behavior_bullshit) {
        cudaFree(this->data); CUDA_CHECK_ERROR;
    }
}

void image_cpu::upload(image_gpu &dst) const {
    cudaMemcpy2D(dst.data, dst.pitch,
                 this->data, this->width * sizeof(unsigned int),
                 this->width * sizeof(unsigned int),
                 this->height,
                 cudaMemcpyHostToDevice
     );
    CUDA_CHECK_ERROR;
}

void image_cpu::download(const image_gpu &src) {
    cudaMemcpy2D(this->data, this->width * sizeof(unsigned int),
                 src.data, src.pitch,
                 this->width * sizeof(unsigned int),
                 this->height,
                 cudaMemcpyDeviceToHost
    );
    CUDA_CHECK_ERROR;
}

image_cpu::image_cpu(const char *fn)
{
	if (!fn)
		throw std::runtime_error("Invalid filename");
	std::ifstream is(fn, std::ofstream::binary);
	if (is.fail())
		throw std::runtime_error("Could not open file");

	char line[256];
	is.getline(line, sizeof(line));

	if (line[0] != 'P' || line[1] != '6')
		throw std::runtime_error("Invalid identification string");

	// Skip comment
	while (is.peek() == '#')
		is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	is >> width; is >> height;

	data = new unsigned int[width * height];

	unsigned short max;
	is >> max;
	if (max > 255)
		throw std::runtime_error("max > 255 is unsupported");

	is.ignore(1, '\n');

	// Read pixels into temporary buffer
	unsigned char *tmp = new unsigned char[width * height * 3];
	is.read((char*)tmp, width * height * 3);

	// Convert RGB to RGBA
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int idx = y * width * 3 + x * 3;
			unsigned char r = tmp[idx], g = tmp[idx + 1], b = tmp[idx + 2];
			data[y * width + x] = r | (g << 8) | (b << 16);
		}
	}

	delete[] tmp;
}

void image_cpu::save(const char *fn) const
{
	if (!fn)
		throw std::runtime_error("Invalid filename");
	std::ofstream os(fn, std::ofstream::binary);
	if (os.fail())
		throw std::runtime_error("Could not open file");

	os << "P6" << std::endl;
	os << width << std::endl;
	os << height << std::endl;
	os << "255" << std::endl;

	// Convert RGBA to RGB
	unsigned char *tmp = new unsigned char[width * height * 3];
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int idx = y * width * 3 + x * 3;
			unsigned int rgba = data[y * width + x];
			unsigned char r = rgba & 0xff, g = (rgba >> 8) & 0xff, b = (rgba >> 16) & 0xff;
			tmp[idx] = r; tmp[idx + 1] = g; tmp[idx + 2] = b;
		}
	}
	os.write((const char*)tmp, width * height * 3);
	delete[] tmp;
}
