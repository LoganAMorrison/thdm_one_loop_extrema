//
// Created by Logan Morrison on 2019-04-25.
//

#include "thdm/potentials.hpp"
#include "thdm/parameters.hpp"
#include "thdm/fields.hpp"
#include "thdm/root_finding_eff.hpp"
#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include <random>
#include <functional>
#include <iomanip>
#include <tuple>

using namespace thdm;

TEST(RootFindingTest, Effective) {
    double mu = 246.0;
    std::cout << std::setprecision(15) << std::endl;
    Fields<double> fields{};
    auto res = solve_root_equations_eff(mu);
    auto nvac = std::get<0>(res);
    auto cbvac = std::get<1>(res);
    auto params = std::get<2>(res);

    std::cout << "nvac:" << std::endl;
    std::cout << nvac << std::endl;
    std::cout << std::endl;

    std::cout << "cbvac:" << std::endl;
    std::cout << cbvac << std::endl;
    std::cout << std::endl;

    std::cout << "Parameters:" << std::endl;
    std::cout << params << std::endl;
    std::cout << std::endl;

    std::cout << "Derivs nvac:" << std::endl;
    fields.set_fields(nvac);
    for (int i = 0; i < 8; i++) {
        std::cout << potential_eff_deriv(fields, params, i + 1) << std::endl;
    }

    std::cout << "Derivs cbvac:" << std::endl;
    fields.set_fields(cbvac);
    for (int i = 0; i < 8; i++) {
        std::cout << potential_eff_deriv(fields, params, i + 1) << std::endl;
    }
    std::cout << "Hessian Normal" << std::endl;
    fields.set_fields(nvac);
    auto hess = potential_eff_hessian(fields, params);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            std::cout << hess[i][j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "Hessian CB" << std::endl;
    fields.set_fields(cbvac);
    hess = potential_eff_hessian(fields, params);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            std::cout << hess[i][j] << ", ";
        }
        std::cout << std::endl;
    }

    std::cout << "Masses nvac:" << std::endl;
    fields.set_fields(nvac);
    auto nmasses = potential_eff_hessian_evals(fields, params);
    for (int i = 0; i < 8; i++) {
        std::cout << nmasses[i] << std::endl;
    }
    std::cout << "Masses cbvac:" << std::endl;
    fields.set_fields(cbvac);
    auto cbmasses = potential_eff_hessian_evals(fields, params);
    for (int i = 0; i < 8; i++) {
        std::cout << cbmasses[i] << std::endl;
    }

}


int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
