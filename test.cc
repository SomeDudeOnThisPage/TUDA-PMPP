#include "test.h"

int pmpp::M_WIDTH, pmpp::M_HEIGHT;
int pmpp::N_WIDTH, pmpp::N_HEIGHT;
int pmpp::P_WIDTH, pmpp::P_HEIGHT;
std::vector<float> pmpp::m_data, pmpp::n_data, pmpp::p_data;
std::random_device pmpp::rd;
std::default_random_engine pmpp::re(rd());
