//
// Created by Logan Morrison on 2019-04-26.
//

#include "thdm/potentials.hpp"
#include "thdm/parameters.hpp"
#include "thdm/fields.hpp"
#include "thdm/root_finding_eff.hpp"
#include "thdm/minimize.hpp"
#include "thdm/tree_roots.hpp"
#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include <random>
#include <functional>
#include <iomanip>

using namespace thdm;

TEST(MinimizeTest, Effective) {
    double mu = 246.0;
    std::cout << std::setprecision(15) << std::endl;

    auto res = solve_root_equations_eff(mu);
    auto params = std::get<2>(res);
    auto tree_roots = get_tree_roots(params);

    std::cout << "Parameters:" << std::endl;
    std::cout << params << std::endl;
    std::cout << std::endl;

    std::cout << "Current Extrema" << std::endl;
    std::cout << "---------------" << std::endl;
    std::cout << std::get<0>(res) << std::endl;
    std::cout << std::get<1>(res) << std::endl;

    std::cout << "Tree Roots and new mins:" << std::endl;
    std::cout << "------------------------" << std::endl;
    for (auto root: tree_roots) {
        std::cout << "Old Extrema: ";
        std::cout << root << std::endl;
        minimize_potential_eff(params, root);
        std::cout << "New Extrema: ";
        std::cout << root << std::endl;
    }


}

int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}