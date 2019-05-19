//
// Created by Logan Morrison on 2019-05-10.
//

#include "thdm/parameters.hpp"
#include "thdm/beta_functions.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

using namespace thdm;

TEST(RGETest, RGESystemClassTest) {
    Parameters<double> params{246.};
    params.yt = 1.0;
    std::cout << std::endl;

    std::cout << params << std::endl;
    RGESystem rge_system(params);

    std::cout << rge_system.get_params() << std::endl;
    std::vector<double> x = {1.0, 2.0, 3., 4., 5., 6., 7., 8., 9., 10., 11.0};
    std::vector<double> dxdt(x.size());

    rge_system(x, dxdt, 246.0);

    for (auto deriv : dxdt) {
        std::cout << deriv << std::endl;
    }

}

TEST(RGETest, RGERunnerTest) {
    Parameters<double> params{246.};
    std::cout << std::endl;

    std::cout << params << std::endl;
    auto new_params = run_parameters(params, 246.0, 50.0);
    std::cout << new_params << std::endl;

}


int main(int argc, char *argv[]) {

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}