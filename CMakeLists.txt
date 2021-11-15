project(filtering CXX CUDA)
cmake_minimum_required(VERSION 3.8)

# include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/include") # my local machine
include_directories("usr/local/cuda/include")

# Set the default build type
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

add_compile_definitions("CRAPPY_DEBUG")
# set(CMAKE_CUDA_FLAGS "-lineinfo")

add_executable(gauss_filter
	common.cc
	main.cc
	filtering.cc

	image.cc
	conv_cpu.cc
	conv_gpu.cu
)

set_target_properties(gauss_filter PROPERTIES CUDA_ARCHITECTURES "75")
