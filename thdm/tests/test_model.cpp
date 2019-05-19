//
// Created by Logan Morrison on 2019-05-03.
//

#include "thdm/potentials.hpp"
#include "thdm/parameters.hpp"
#include "thdm/fields.hpp"
#include "thdm/vacuua.hpp"
#include "thdm/model.hpp"
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <gtest/gtest.h>

using namespace thdm;

void display_model(Model &model) {
    std::cout << "Deepest Normal" << std::endl;
    std::cout << model.one_loop_deepest_normal << std::endl;
    std::cout << "Deepest CB" << std::endl;
    std::cout << model.one_loop_deepest_cb << std::endl;
    std::cout << "Deepest:" << std::endl;
    std::cout << model.one_loop_deepest << std::endl;
    std::cout << "Has CB and normal mininima:" << std::endl;
    std::cout << (model.has_cb_min && model.has_normal_min) << std::endl;
    std::cout << "Is CB the deepest:" << std::endl;
    std::cout << (model.is_cb_deepest) << std::endl;

    std::cout << "All One Loop Extrema" << std::endl;
    std::cout << "--------------------" << std::endl;
    for (const auto &vac: model.one_loop_vacuua)
        std::cout << vac << std::endl;

    std::cout << "All Tree Extrema" << std::endl;
    std::cout << "----------------" << std::endl;
    for (const auto &vac: model.tree_vacuua)
        std::cout << vac << std::endl;
}

TEST(ModelTest, Test1) {
    double mu = 246.0;
    std::cout << std::setprecision(15);
    std::cout << std::boolalpha;

    Model model;
    bool done = false;
    while (!done) {
        try {
            model = Model(mu);
            done = true;
        } catch (...) {
            done = false;
        }
    }

    display_model(model);

}


int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}