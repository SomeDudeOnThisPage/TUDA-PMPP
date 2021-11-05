Note: You NEED the CUDA Toolkit installed under /usr/local/cuda, otherwise this won't compile without changing the
      include_directories path inside CMakeLists.

Compile and Run:
    cmake -B build
    cd build
    make
    ./matmul random compare

Note that if you want to see the element comparison, it must be specified by passing the command line parameter 'compare'.
I disabled it by default because it takes a long time to print for larger matrices.