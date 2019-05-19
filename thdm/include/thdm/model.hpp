//
// Created by Logan Morrison on 2019-05-02.
//

#ifndef THDM_MODEL_HPP
#define THDM_MODEL_HPP

#include "thdm/fields.hpp"
#include "thdm/parameters.hpp"
#include "thdm/vacuua.hpp"
#include "thdm/tree_roots.hpp"
#include "thdm/root_finding_eff.hpp"
#include "thdm/root_refine.hpp"
#include "thdm/minimize.hpp"
#include "thdm/scalar_masses.hpp"
#include "thdm/potentials.hpp"
#include "thdm/extrema_type.hpp"
#include <vector>
#include <tuple>
#include <iostream>

namespace thdm {

/**
 * Check that all the derivatives at the normal and charge
 * breaking vacuums are zero.
 * @param point Point to check.
 * @return bool
 */
bool verify_derivatives_zero(Parameters<double> &params, Vacuum<double> &vac) {
    bool isvalid = true;

    Fields<double> fields{};
    try {
        fields.set_fields(vac);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 1)) < 1e-8);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 2)) < 1e-8);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 3)) < 1e-8);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 4)) < 1e-8);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 5)) < 1e-8);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 6)) < 1e-8);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 7)) < 1e-8);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 8)) < 1e-8);
    } catch (...) {
        isvalid = false;
    }
    if (isvalid) {
        // std::cout << "point is good." << std::endl;
    }
    return isvalid;
}

/**
 * Check that all the derivatives at the normal and charge
 * breaking vacuums are zero.
 * @param point Point to check.
 * @return bool
 */
bool verify_derivatives_zero_tree(Parameters<double> &params, Vacuum<double> &vac) {
    bool isvalid = true;

    Fields<double> fields{};
    try {
        fields.set_fields(vac);
        isvalid = isvalid && (std::abs(potential_tree_deriv(fields, params, 1)) < 1e-3);
        isvalid = isvalid && (std::abs(potential_tree_deriv(fields, params, 2)) < 1e-3);
        isvalid = isvalid && (std::abs(potential_tree_deriv(fields, params, 3)) < 1e-3);
        isvalid = isvalid && (std::abs(potential_tree_deriv(fields, params, 4)) < 1e-3);
        isvalid = isvalid && (std::abs(potential_tree_deriv(fields, params, 5)) < 1e-3);
        isvalid = isvalid && (std::abs(potential_tree_deriv(fields, params, 6)) < 1e-3);
        isvalid = isvalid && (std::abs(potential_tree_deriv(fields, params, 7)) < 1e-3);
        isvalid = isvalid && (std::abs(potential_tree_deriv(fields, params, 8)) < 1e-3);
    } catch (...) {
        isvalid = false;
    }
    if (isvalid) {
        // std::cout << "point is good." << std::endl;
    }
    return isvalid;
}


/**
 * Check that there are the correct number of goldstone
 * bosons. For normal extrema, there should be three,
 * for CB, there should be four.
 * @param params
 * @param vac
 * @param is_cb
 * @return
 */
bool verify_goldstones(Parameters<double> &params, Vacuum<double> &vac, bool is_cb = false) {
    bool hasgoldstones = true;

    Fields<double> fields{};
    try {
        fields.set_fields(vac);
        auto masses = potential_eff_hessian_evals(fields, params);
        // Make all masses positive
        for (double &mass : masses)
            mass = abs(mass);
        // Sort masses
        std::sort(masses.begin(), masses.end(), std::less<double>());
        // first three should have masses that are less
        // near zero.
        hasgoldstones = hasgoldstones && (masses[0] < 1e-7);
        hasgoldstones = hasgoldstones && (masses[1] < 1e-7);
        hasgoldstones = hasgoldstones && (masses[2] < 1e-7);

        if (is_cb) {
            hasgoldstones = hasgoldstones && (masses[3] < 1e-7);
        }


    } catch (...) {
        //std::cout << "Failed at goldstones" << std::endl;
        hasgoldstones = false;
    }

    return hasgoldstones;
}


class Model {

public:
    Fields<double> fields{};
    Parameters<double> params;

    std::vector<Vacuum<double>> one_loop_vacuua{};
    Vacuum<double> one_loop_deepest{};
    Vacuum<double> one_loop_deepest_cb{};
    Vacuum<double> one_loop_deepest_normal{};

    std::vector<Vacuum<double>> tree_vacuua{};

    bool has_normal_min = false;
    bool has_cb_min = false;
    bool is_cb_deepest = false;

    Model() = default;

    Model(Parameters<double> &params, Vacuum<double> &nvac, Vacuum<double> &cbvac) {
        tree_vacuua = get_tree_roots(params);

        for (auto &vac : tree_vacuua) {
            complete_vacuua(vac, true);
        }
        // Use tree-level roots to find new minima.
        minimize_from_tree_roots_and_refine();
        // Find deepest normal and cb extrema
        find_deepest_eff();

        determine_if_has_normal_cb_min();

        is_cb_deepest = (one_loop_deepest_cb.potential <
                one_loop_deepest_normal.potential);
    }

    explicit Model(double renorm_scale) {
        solve_model(renorm_scale);
        // Find all tree roots
        tree_vacuua = get_tree_roots(params);
        // complete roots
        for (auto &vac : tree_vacuua) {
            complete_vacuua(vac, true);
        }
        // Use tree-level roots to find new minima.
        minimize_from_tree_roots_and_refine();
        // Find deepest normal and cb extrema
        find_deepest_eff();

        determine_if_has_normal_cb_min();

        is_cb_deepest = (one_loop_deepest_cb.potential <
                one_loop_deepest_normal.potential);
    }

private:
    void complete_vacuua(Vacuum<double> &vac, bool tree = false) {
        if (tree) {
            fields.set_fields(vac);
            vac.potential = potential_tree(fields, params);
            vac.extrema_type = determine_single_extrema_type_tree(params, vac);
        } else {
            fields.set_fields(vac);
            vac.potential = potential_eff(fields, params);
            vac.extrema_type = determine_single_extrema_type_eff(params, vac);
        }
    }

    void solve_model(double renorm_scale) {
        bool done = false;
        Vacuum<double> nvac{};
        Vacuum<double> cbvac{};
        while (!done) {
            try {
                auto sol = solve_root_equations_eff(renorm_scale);
                params = std::get<2>(sol);
                nvac = std::get<0>(sol);
                cbvac = std::get<1>(sol);

                done = true;
                done = (done && verify_derivatives_zero(params, nvac));
                done = (done && verify_derivatives_zero(params, cbvac));
                done = (done && verify_goldstones(params, nvac));
                done = (done && verify_goldstones(params, cbvac, true));
            } catch (...) {
                done = false;
            }
        }
        one_loop_deepest_normal = nvac;
        one_loop_deepest_cb = cbvac;
        one_loop_deepest = (nvac.potential < cbvac.potential) ? nvac : cbvac;
        complete_vacuua(nvac);
        complete_vacuua(cbvac);
        one_loop_vacuua.push_back(nvac);
        one_loop_vacuua.push_back(cbvac);
    }

    void minimize_from_tree_roots_and_refine() {
        // Starting from tree-roots, try to find a new one-loop min
        for (const auto &vac : tree_vacuua) {
            try {
                auto new_vac = Vacuum<double>(vac);
                minimize_potential_eff(params, new_vac);
                refine_root(params, new_vac);
                if (verify_derivatives_zero(params, new_vac)) {
                    // Make sure root isn't duplicate
                    bool should_add = true;
                    for (const auto &one_loop_vac : one_loop_vacuua) {
                        if ((std::abs(new_vac.vevs[0] - one_loop_vac.vevs[0]) < 1e-5) &&
                                (std::abs(new_vac.vevs[1] - one_loop_vac.vevs[1]) < 1e-5) &&
                                (std::abs(new_vac.vevs[2] - one_loop_vac.vevs[2]) < 1e-5)) {
                            should_add = false;
                            break;
                        }
                    }
                    if (should_add) {
                        complete_vacuua(new_vac);
                        one_loop_vacuua.push_back(new_vac);
                    }
                }
            } catch (...) {
                // that one didn't work! don't add it.
            }
        }
    }

    void find_deepest_eff() {
        using std::abs;
        for (const auto &vac : one_loop_vacuua) {
            if (vac.potential < one_loop_deepest.potential)
                one_loop_deepest = vac;
            if (abs(vac.vevs[2]) < 1e-5) {
                if (vac.potential < one_loop_deepest_normal.potential) {
                    one_loop_deepest_normal = vac;
                }
            }
            if (abs(vac.vevs[2]) > 1e-5) {
                if (vac.potential < one_loop_deepest_cb.potential) {
                    one_loop_deepest_cb = vac;
                }
            }
        }
    }

    void determine_if_has_normal_cb_min() {
        for (auto vac : one_loop_vacuua) {
            if (abs(vac.vevs[2]) < 1e-5) {
                if (vac.extrema_type == SingleExtremaType::Minimum)
                    has_normal_min = true;
            }
            if (abs(vac.vevs[2]) > 1e-5) {
                if (vac.extrema_type == SingleExtremaType::Minimum)
                    has_cb_min = true;
            }
        }
    }
};

}


#endif //THDM_MODEL_HPP
