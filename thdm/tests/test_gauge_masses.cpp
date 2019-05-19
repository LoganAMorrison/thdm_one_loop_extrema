//
// Created by Logan Morrison on 2019-05-09.
//

#include "thdm/parameters.hpp"
#include "thdm/fields.hpp"
#include "thdm/gauge_masses.hpp"
#include "thdm/potentials.hpp"
#include <iostream>
#include <gtest/gtest.h>
#include <vector>
#include <tuple>

using namespace thdm;


TEST(GaugeMassesTest, TestGaugeMassMatrix) {
    Parameters<double> params{246.};
    Fields<double> fields{1., 2., 0.0};
    std::cout << params << std::endl;

    auto mass_matrix = gauge_sqaured_mass_matrix(fields, params);
    auto masses = gauge_squared_masses(fields, params);

    std::cout << mass_matrix << std::endl;
    for (size_t i = 0; i < 4; i++) {
        std::cout << masses[i] << ", ";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < 8; i++) {
        std::cout << "derivative " << i + 1 << " = ";
        auto derivs = gauge_squared_masses_deriv(fields, params, i + 1);
        for (int j = 0; j < 4; j++) {
            std::cout << std::get<1>(derivs[j]) << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 8; j++) {
            std::cout << "derivative " << i + 1 << ", " << j + 1 << " = ";
            auto derivs = gauge_squared_masses_deriv(fields, params, i + 1, j + 1);
            for (int k = 0; k < 4; k++) {
                std::cout << std::get<1>(derivs[k]) << ", ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    std::cout << potential_eff(fields, params);

}

int main(int argc, char *argv[]) {

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}