//
// Created by Logan Morrison on 2019-04-26.
//

#include "thdm/potentials.hpp"
#include "thdm/parameters.hpp"
#include "thdm/fields.hpp"
#include "thdm/root_finding_eff.hpp"
#include "thdm/minimize.hpp"
#include "thdm/tree_roots.hpp"
#include "thdm/vacuua.hpp"
#include "thdm/validation.hpp"
#include "thdm/root_refine.hpp"
#include "thdm/extrema_type.hpp"
#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include <random>
#include <functional>
#include <iomanip>

using namespace thdm;

static std::random_device rd{};
static std::mt19937 engine{rd()};
static std::uniform_real_distribution<double> dist{0.0, 1.0};

struct Point {
    Parameters<double> params;
    Vacuum<double> nvac;
    Vacuum<double> cbvac;
};

Point solve_root_equations_eff(double mu) {
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
            done = false;
        }
    }
    return Point{params, nvac, cbvac};
}

TEST(MinimizeTest, Effective) {
    double mu = 246.0;
    std::cout << std::setprecision(15) << std::endl;

    Fields<double> fields{};
    auto point = solve_root_equations_eff(mu);
    auto params = point.params;
    auto nvac = point.nvac;
    auto cbvac = point.cbvac;
    auto tree_roots = get_tree_roots(params);

    std::cout << "Parameters:" << std::endl;
    std::cout << params << std::endl;
    std::cout << std::endl;

    std::cout << "Current Extrema" << std::endl;
    std::cout << "---------------" << std::endl;
    std::cout << nvac << std::endl;
    std::cout << cbvac << std::endl;

    std::cout << "Tree Roots and new mins:" << std::endl;
    std::cout << "------------------------" << std::endl;
    for (auto root: tree_roots) {
        std::cout << "Old Extrema: ";
        std::cout << root << std::endl;
        minimize_potential_eff(params, root);
        std::cout << "New Extrema: ";
        std::cout << root << std::endl;
    }

    std::vector<Vacuum<double>> random_vacs(150, Vacuum<double>{});
    for (auto &vac: random_vacs) {
        vac.vevs[0] = 2.0 * mu * (dist(engine) - 0.5);
        vac.vevs[1] = 2.0 * mu * (dist(engine) - 0.5);
        vac.vevs[2] = 2.0 * mu * (dist(engine) - 0.5);
    }

    for (auto &vac: random_vacs) {
        minimize_potential_eff(params, vac);
        refine_root(params, vac);
    }

    std::vector<Vacuum<double>> new_vacs;
    for (auto &vac: random_vacs) {
        bool is_new = true;
        for (auto &current_vac: new_vacs) {
            try {
                if (are_vacuua_approx_equal(current_vac, vac)) {
                    is_new = false;
                    break;
                }
            } catch (THDMException &e) {
                is_new = false;
                break;
            }

        }
        if (is_new && is_vacuum_valid(params, vac)) {
            fields.set_fields(vac);
            vac.potential = potential_eff(fields, params);
            vac.extrema_type = determine_single_extrema_type_eff(params, vac);
            new_vacs.push_back(vac);
        }
    }

    std::cout << "new mins:" << std::endl;
    std::cout << "------------------------" << std::endl;
    for (auto vac: new_vacs) {
        std::cout << vac << std::endl;
    }


}

int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}