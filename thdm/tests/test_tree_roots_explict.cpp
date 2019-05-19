//
// Created by Logan Morrison on 2019-05-04.
//
#include "thdm/tree_roots_explicit.hpp"
#include "thdm/model.hpp"
#include "thdm/parameters.hpp"
#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include <random>
#include <functional>
#include <iomanip>
#include <vector>
#include <Eigen/Dense>
#include <tuple>

using namespace thdm;

TEST(TestTreeRootsExplict, TestUnivariate) {
    std::vector<double> coeffs(5);
    coeffs[0] = 1.0;
    coeffs[1] = 3.0;
    coeffs[2] = 2.0;
    coeffs[3] = 4.0;
    coeffs[4] = -1.0;

    auto roots = univariate_polynomial_root_finder(coeffs);

    std::cout << roots << std::endl;


}

TEST(TestTreeRootsExplict, TestTreeRoots) {
    Model model(246.);
    std::cout << "Solved model." << std::endl;
    auto tree_roots = get_real_tree_roots(model.params);

    std::cout << model.params << std::endl;

    for (auto root:tree_roots) {
        double r1 = std::get<0>(root);
        double r2 = std::get<1>(root);
        double c1 = std::get<2>(root);

        std::cout << "("
                  << r1 << ", "
                  << r2 << ", "
                  << c1 << ")" << std::endl;
    }
}

int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}