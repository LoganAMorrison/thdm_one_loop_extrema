//
// Created by Logan Morrison on 2019-04-26.
//

#include "thdm/potentials.hpp"
#include "thdm/parameters.hpp"
#include "thdm/fields.hpp"
#include "thdm/model.hpp"
#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include <random>
#include <functional>
#include <iomanip>
#include <boost/tuple/tuple.hpp>

using namespace thdm;

TEST(TreeRootsTest, Effective) {
    double mu = 246.0;
    std::cout << std::setprecision(15) << std::endl;

    Model model(mu);
    //auto res = solve_root_equations_eff(mu);
    //auto params = std::get<2>(res);
    auto tree_roots = get_tree_roots(model.params);

    std::cout << "Parameters:" << std::endl;
    std::cout << model.params << std::endl;
    std::cout << std::endl;

    std::cout << "Tree Roots:" << std::endl;
    for (const auto &root: model.tree_vacuua)
        std::cout << root << std::endl;

    std::cout << "Tree Roots:" << std::endl;
    for (const auto &root: tree_roots)
        std::cout << root << std::endl;


}

int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
