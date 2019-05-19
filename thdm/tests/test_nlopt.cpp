//
// Created by Logan Morrison on 2019-05-03.
//

#include "thdm/potentials.hpp"
#include "thdm/parameters.hpp"
#include "thdm/fields.hpp"
#include "thdm/vacuua.hpp"
#include "thdm/model.hpp"
#include "thdm/minimize_global.hpp"
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <gtest/gtest.h>

using namespace thdm;

TEST(NLOPTTest, Test1) {
    double mu = 246.0;
    std::cout << std::setprecision(15);
    std::cout << std::boolalpha;

    Model model(mu);
    std::cout << "Starting global minimization." << std::endl;
    one_loop_global_minimization(model.params);
}


int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}