#include "thdm/potentials.hpp"
#include "thdm/parameters.hpp"
#include "thdm/fields.hpp"
#include "thdm/dual.hpp"
#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include <random>
#include <functional>
#include <iomanip>

using namespace thdm;

TEST(PotentialTest, FirstDerivativeTest) {
    double mu = 246.0;
    std::cout << std::setprecision(15) << std::endl;

    for (size_t count = 0; count < 100; count++) {
        std::random_device rd{};
        std::mt19937 engine{rd()};
        std::uniform_real_distribution<double> dist{-mu, mu};

        Fields<double> fields{dist(engine), dist(engine), dist(engine)};
        Parameters<double> params{246.0};

        Fields<Dual<double>> dual_fields{};
        Parameters<Dual<double>> dual_params{Dual<double>{mu, 0.0}};
        for (size_t i = 0; i < 8; i++) {
            dual_fields[i] = static_cast<Dual<double>>(fields[i]);
        }
        for (size_t i = 0; i < 8; i++) {
            dual_params[i] = static_cast<Dual<double>>(params[i]);
        }

        for (int i = 1; i <= 8; i++) {
            dual_fields[i - 1].eps = 1;
            double deriv1 = potential_tree_deriv(fields, params, i);
            double deriv2 = potential_tree(dual_fields, dual_params).eps;
            dual_fields[i - 1].eps = 0;
            std::cout << "deriv1 = " << deriv1 << std::endl;
            std::cout << "deriv2 = " << deriv2 << std::endl;
            std::cout << std::endl;
            ASSERT_NEAR(deriv1, deriv2, 1e-7);
        }
    }
}

TEST(PotentialTest, EffectivePotential) {
    double mu = 246.0;
    std::cout << std::setprecision(15) << std::endl;

    std::random_device rd{};
    std::mt19937 engine{rd()};
    std::uniform_real_distribution<double> dist{-mu, mu};

    Fields<double> fields{dist(engine), dist(engine), dist(engine)};
    Parameters<double> params{mu};

    std::cout << "Fields:" << std::endl;
    std::cout << fields << std::endl;
    std::cout << std::endl;

    std::cout << "Parameters:" << std::endl;
    std::cout << params << std::endl;
    std::cout << std::endl;

    std::cout << potential_eff(fields, params) << std::endl;

    std::cout << "{";
    for (int i = 1; i <= 8; i++) {
        if (i == 8)
            std::cout << potential_eff_deriv(fields, params, i) << "}";
        else
            std::cout << potential_eff_deriv(fields, params, i) << ",";
    }
    std::cout << std::endl;
    std::cout << "{";
    for (int i = 1; i <= 8; i++) {
        std::cout << "{";
        for (int j = 1; j <= 8; j++) {
            if (j == 8)
                std::cout << potential_eff_deriv(fields, params, i, j) << "}";
            else
                std::cout << potential_eff_deriv(fields, params, i, j) << ",";
        }
        std::cout << std::endl;
    }
    std::cout << "}";
}


int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}