#include "filtering.h"
#include "image.h"
#include "common.h"
#include "conv_cpu.h"
#include "conv_gpu.h"

#include <chrono>

typedef std::chrono::high_resolution_clock timer_clock;
typedef std::chrono::high_resolution_clock::time_point timer_tp;

timer_tp timer_now() {
    return timer_clock::now();
}

float timer_elapsed(const timer_tp &start, const timer_tp &end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

float filter_cpu(const image_cpu &base_image, const filterkernel_cpu &base_kernel) {
    image_cpu intermediate(base_image.width, base_image.height);
    image_cpu dst(base_image.width, base_image.height);

    timer_tp cpu_t0 = timer_now();
    conv_h_cpu(intermediate, base_image, base_kernel);
    conv_v_cpu(dst, intermediate, base_kernel);
    timer_tp cpu_t1 = timer_now();
    float cpu_time = timer_elapsed(cpu_t0, cpu_t1);

    dst.save(R"(F:\dev\TU Darmstadt\PMPP\ex2\image_output\out_cpu.ppm)");

    return cpu_time;
}

float filter_gpu_global_memory(const image_cpu &base_image, const filterkernel_cpu &base_kernel, bool const_kernel, std::string file) {
    auto *kernel_GPU = new filterkernel_gpu(base_kernel.ks);
    base_kernel.upload(*kernel_GPU);

    auto *image_source_GPU = new image_gpu(base_image.width, base_image.height);
    auto *image_intermediate_GPU = new image_gpu(base_image.width, base_image.height);
    base_image.upload(*image_source_GPU);

    auto *solution_GPU = new image_gpu(base_image.width, base_image.height);

    float elapsed = 0.0f;
    if (const_kernel) {
        elapsed += conv_h_gpu_cmem(*image_intermediate_GPU, *image_source_GPU, *kernel_GPU);
        elapsed += conv_v_gpu_cmem(*solution_GPU, *image_intermediate_GPU, *kernel_GPU);
    } else {
        elapsed += conv_h_gpu_gmem(*image_intermediate_GPU, *image_source_GPU, *kernel_GPU);
        elapsed += conv_v_gpu_gmem(*solution_GPU, *image_intermediate_GPU, *kernel_GPU);
    }

    image_cpu solution_CPU(base_image.width, base_image.height);
    solution_CPU.download(*solution_GPU);

    solution_CPU.save(("F:\\dev\\TU Darmstadt\\PMPP\\ex2\\image_output\\" + file + ".ppm").c_str());

    return elapsed;
}

float filter_gpu_shared_memory(const image_cpu &base_image, const filterkernel_cpu &base_kernel, bool const_kernel, std::string file) {
    auto *kernel_GPU = new filterkernel_gpu(base_kernel.ks);
    base_kernel.upload(*kernel_GPU);

    auto *image_source_GPU = new image_gpu(base_image.width, base_image.height);
    auto *image_intermediate_GPU = new image_gpu(base_image.height, base_image.height);
    base_image.upload(*image_source_GPU);

    auto *solution_GPU = new image_gpu(base_image.width, base_image.height);

    float elapsed = 0.0f;
    if (const_kernel) {
        elapsed += conv_h_gpu_all(*image_intermediate_GPU, *image_source_GPU, *kernel_GPU);
        elapsed += conv_v_gpu_all(*solution_GPU, *image_intermediate_GPU, *kernel_GPU);
    } else {
        elapsed += conv_h_gpu_smem(*image_intermediate_GPU, *image_source_GPU, *kernel_GPU);
        elapsed += conv_v_gpu_smem(*solution_GPU, *image_intermediate_GPU, *kernel_GPU);
    }

    image_cpu solution_CPU(base_image.width, base_image.height);
    solution_CPU.download(*solution_GPU);

    solution_CPU.save(("F:\\dev\\TU Darmstadt\\PMPP\\ex2\\image_output\\" + file + ".ppm").c_str());

    return elapsed;
}


void filtering(const char *imgfile, int ks) {
    image_cpu base_image(imgfile);
    filterkernel_cpu base_filterkernel(ks);

    upload_filterkernel(base_filterkernel); // Upload filterkernel to const memory once.

    // === Task 1 ===
    float cpu_time = 0.0f;//filter_cpu(base_image, base_filterkernel);

	// === Task 2 ===
    float gpu_gmem_time = filter_gpu_global_memory(base_image, base_filterkernel, false, "out_gpu_gmem");

	// === Task 3 ===
    float gpu_smem_time = filter_gpu_shared_memory(base_image, base_filterkernel, false, "out_gpu_smem");

	// === Task 4 ===
    float gpu_cmem_time = filter_gpu_global_memory(base_image, base_filterkernel, true, "out_gpu_cmem");

	// === Task 5 ===
	// Not part of the PDF tasks.

	// === Task 6 ===
    float gpu_all_time = filter_gpu_shared_memory(base_image, base_filterkernel, true, "out_gpu_all");

    std::cout << "----------------------TIMINGS----------------------" << std::endl;
    // latex formatting
    std::cout << cpu_time << " & " << gpu_gmem_time << " & " << gpu_smem_time << " & " << gpu_cmem_time << " & " << gpu_all_time << " \\\\" << std::endl;
}


/************************************************************
 * 
 * TODO: Write your text answers here!
 * 
 * (Task 7) nvprof output
 * 
 * Answer: TODO
 * 
 ************************************************************/
