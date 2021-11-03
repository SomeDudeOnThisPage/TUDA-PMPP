#include <iostream>
#include <iomanip>

#include "matmul.h"
#include "test.h"

int main(int argc, char **argv)
{
	std::cout << "PMPP Hello World!" << std::endl;

	pmpp::load(argc >= 2 && std::string(argv[1]) == std::string("random"));

	print_cuda_devices();
	std::cout << std::setprecision(10);
	matmul();
}
