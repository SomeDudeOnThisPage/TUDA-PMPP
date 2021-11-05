#include <iostream>
#include <iomanip>

#include "matmul.h"
#include "test.h"

int main(int argc, char **argv) {
    bool random = false;
    bool compare = false;

    for (int i = 1; i < argc; i++) {
        if (argv[i] == std::string("random")) {
            std::cout << "using random" << std::endl;
            random = true;
        } else if (argv[i] == std::string("compare")) {
            std::cout << "using compare" << std::endl;
            compare = true;
        }
    }

    pmpp::load(random);

	print_cuda_devices();
	std::cout << std::setprecision(10);
	matmul(compare);
}
