project(MatMul CXX CUDA)
cmake_minimum_required(VERSION 3.8)

# Temporary debug stuff...
#   add_compile_definitions(CRAPPY_DEBUG)
#
# Path to CUDA includes.
# include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/include") # my local machine
include_directories("/usr/local/cuda/include") # lab pcs

# Set the default build type
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# Testing stuff...
# set(CMAKE_CUDA_FLAGS "--relocatable-device-code=true")

# Specify the files that need to be compiled and linked.
# This will create an executable named 'matmul'.
add_executable(matmul
	matrix.h
	matrix.cc
	main.cc
	matmul.cc
	mul_cpu.cc
	test.cc
	mul_gpu.cu
)

set_target_properties(matmul PROPERTIES CUDA_ARCHITECTURES "75")
