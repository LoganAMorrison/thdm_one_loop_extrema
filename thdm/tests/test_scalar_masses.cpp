#include "thdm/jacobi.hpp"
#include "thdm/dual.hpp"
#include "thdm/scalar_masses.hpp"
#include "thdm/fields.hpp"
#include "thdm/parameters.hpp"
#include "thdm/root_finding_eff.hpp"
#include "thdm/vacuua.hpp"
#include "thdm/errors.hpp"
#include "thdm/validation.hpp"
#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include <random>
#include <functional>
#include <tuple>
#include <Eigen/Dense>

using namespace Eigen;
using namespace thdm;

TEST(ScalarMassesTest, Eigenvalues) {
    Fields<double> fields{1.0, 2.0, 3.0};
    Parameters<double> params{246.0};
    std::cout << std::setprecision(15) << std::endl;

    std::cout << "Fields:" << std::endl;
    std::cout << fields << std::endl;

    std::cout << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << params << std::endl;

    std::cout << std::setprecision(15) << std::endl;

    auto m = scalar_squared_mass_matrix(fields, params);

    std::cout << "{";
    for (int i = 0; i < 8; i++) {
        std::cout << "{";
        for (int j = 0; j < 8; j++) {
            if (j == 7)
                std::cout << m[i][j] << "},";
            else
                std::cout << m[i][j] << ",";
        }
        std::cout << std::endl;
    }

    auto evals = jacobi(m);

    for (double &eval : evals) {
        std::cout << eval << std::endl;
    }
}

TEST(ScalarMassesTest, EigenvalueDerivatives) {
    double mu = 246.0;
    std::cout << std::setprecision(15) << std::endl;

    Fields<double> fields{};
    Vacuum<double> nvac{};
    Vacuum<double> cbvac{};
    Parameters<double> params{mu};
    bool done = false;
    while (!done) {
        int status;
        // Initialize the parameters. Random values will be chosen
        // such that the potential is bounded from below
        params = Parameters<double>{mu};
        // Create random normal and cb vacuua
        nvac = generate_normal_vac(mu);
        cbvac = generate_cb_vac(mu);
        set_top_yukawa(params, nvac);
        try {
            status = try_solve_root_equations_eff(nvac, cbvac, params);
            // Check that nvac is valid
            done = is_vacuum_valid(params, nvac);
            // Check that cbvac is valid
            done = done && is_vacuum_valid(params, nvac);
            // Check that root-finder succeeded.
            done = done && (status == 0);
        } catch (THDMException &e) {
            std::cout << e.what() << std::endl;
            done = false;
        }
    }

    fields.set_fields(cbvac);
    std::cout << "Fields:" << std::endl;
    std::cout << fields << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << params << std::endl;
    std::cout << std::endl;

    auto evals3 = scalar_squared_masses_deriv_fld(fields, params, 8, 7);

    for (auto &it : evals3) {
        std::cout << "(" << std::get<0>(it) << ", "
                  << std::get<1>(it) << ", "
                  << std::get<2>(it) << ", "
                  << std::get<3>(it) << ")" << std::endl;
    }
}

TEST(ScalarMassesTest, DerivativesMassMatrix) {
    double mu = 246.0;
    Fields<Dual<Dual<double>>> fields{};
    Parameters<Dual<Dual<double>>> params{Dual<Dual<double>>{246.0}};

    std::random_device rd{};
    std::mt19937 engine{rd()};
    std::uniform_real_distribution<double> dist{-mu, mu};

    for (int i = 0; i < 8; i++) {
        fields[i].val.val = dist(engine);
    }


    std::cout << std::setprecision(15) << std::endl;
    std::cout << "FIELDS = {";
    for (int i = 0; i < 8; i++) {
        if (i == 7)
            std::cout << fields[i].val.val << "};\n";
        else
            std::cout << fields[i].val.val << ",";
    }
    std::cout << "PARAMS = {";
    for (int i = 0; i < 9; i++) {
        if (i == 8)
            std::cout << params[i].val.val << "};\n";
        else
            std::cout << params[i].val.val << ",";
    }

    std::cout << std::setprecision(15) << std::endl;


    for (int fld1 = 1; fld1 <= 8; fld1++) {

        for (int fld2 = 1; fld2 <= 8; fld2++) {
            std::cout << "d^2V/dphi_" << std::to_string(fld1)
                      << "dphi_" << std::to_string(fld2) << std::endl;

            for (int i = 0; i < 8; i++) {
                fields[i].val.eps = 0.0;
                fields[i].eps.val = 0.0;
                fields[i].eps.eps = 0.0;
            }

            fields[fld1 - 1].val.eps = 1.0;
            fields[fld2 - 1].eps.val = 1.0;

            auto m = scalar_squared_mass_matrix(fields, params);

            std::cout << "{";
            for (int i = 0; i < 8; i++) {
                std::cout << "{";
                for (int j = 0; j < 8; j++) {
                    if (j == 7)
                        std::cout << m[i][j].eps.eps;
                    else
                        std::cout << m[i][j].eps.eps << ",";
                }
                if (i == 7)
                    std::cout << "}}\n\n";
                else
                    std::cout << "},\n";
            }
        }
    }
}


int main(int argc, char *argv[]) {

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}