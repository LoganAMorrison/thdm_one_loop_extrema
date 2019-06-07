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


class ModelTest : public ::testing::Test {
protected:
    void SetUp() override {

    }

    double mu = 246.0;
    Model model{mu};
    Fields<double> fields{};
};

TEST_F(ModelTest, Derivatives) {
    fields.set_fields(model.one_loop_vacuua[0]);
    std::cout << "Normal Vacuum: " << model.one_loop_vacuua[0] << "\n";
    std::cout << "Normal vacuum derivs: \n";
    for (int i = 1; i <= 8; i++) {
        std::cout << potential_eff_deriv(fields, model.params, i) << "\n";
    }

    fields.set_fields(model.one_loop_vacuua[1]);
    std::cout << "CB Vacuum: " << model.one_loop_vacuua[1] << "\n";
    std::cout << "CB vacuum derivs: \n";
    for (int i = 1; i <= 8; i++) {
        std::cout << potential_eff_deriv(fields, model.params, i) << "\n";
    }
}


int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}