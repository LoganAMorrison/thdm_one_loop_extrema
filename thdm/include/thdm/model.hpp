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
        double TOL = 1e-8;
        for (int i = 1; i <= 8; i++) {
            double derivative = potential_eff_deriv(fields, params, i);
            isvalid = isvalid && (abs(derivative) < TOL);
        }

    } catch (...) {
        isvalid = false;
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
    bool has_goldstones = true;
    double TOL = 1e-7;

    Fields<double> fields{};
    try {
        fields.set_fields(vac);
        auto masses = potential_eff_hessian_evals(fields, params);
        // Make all masses positive
        for (double &mass : masses) {
            mass = abs(mass);
        }
        // Sort masses
        std::sort(masses.begin(), masses.end(), std::less<double>());
        // first three should have masses that are less
        // near zero.
        has_goldstones = has_goldstones && (masses[0] < TOL);
        has_goldstones = has_goldstones && (masses[1] < TOL);
        has_goldstones = has_goldstones && (masses[2] < TOL);

        if (is_cb) {
            has_goldstones = has_goldstones && (masses[3] < TOL);
        }
    } catch (...) {
        has_goldstones = false;
    }

    return has_goldstones;
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

    /**
     * Initialize an empty model.
     */
    Model() = default;

    /**
     * Start a model from know vacuua.
     * @param params THDM parameters.
     * @param nvac The normal vacuum.
     * @param cbvac The charge-breaking vacuum.
     */
    Model(Parameters<double> &params, Vacuum<double> &nvac, Vacuum<double> &cbvac) {
        // Add the known normal and charge-breaking vacuua to list.
        one_loop_vacuua.push_back(nvac);
        one_loop_vacuua.push_back(cbvac);
        // Find all the tree roots.
        tree_vacuua = get_tree_roots(params);
        // Fill in the potential and types for the tree-vacuua.
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

    /**
     * Starting only from the renormalization scale, fill in the model by
     * finding parameters such that there exists a normal and charge-breaking
     * vacuum. Then compute all the tree roots and from the tree roots,
     * attempt to find new minima.
     * @param renorm_scale Renormalization scale.
     */
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
    /**
     * Fill in the potential values and types.
     * @param vac Vacuum
     * @param tree If tree, vacuum is a tree vacuum, else it is an effective
     * vacuum.
     */
    void complete_vacuua(Vacuum<double> &vac, bool tree = false) {
        fields.set_fields(vac);
        if (tree) {
            vac.potential = potential_tree(fields, params);
            vac.extrema_type = determine_single_extrema_type_tree(params, vac);
        } else {
            vac.potential = potential_eff(fields, params);
            vac.extrema_type = determine_single_extrema_type_eff(params, vac);
        }
    }

    /**
     * Solve the effective root equations to determine parameters such that
     * there exists a normal and charge-breaking vacuum.
     * @param renorm_scale Renormalization scale.
     */
    void solve_model(double renorm_scale) {
        bool done = false;
        Vacuum<double> nvac{};
        Vacuum<double> cbvac{};
        while (!done) {
            try {
                auto sol = solve_root_equations_eff(renorm_scale);
                nvac = std::get<0>(sol);
                cbvac = std::get<1>(sol);
                params = std::get<2>(sol);

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

    /**
     * Starting from all the tree roots, minimize the effective potential to
     * find new minima.
     */
    void minimize_from_tree_roots_and_refine() {
        // Starting from tree-roots, try to find a new one-loop min
        for (const auto &vac : tree_vacuua) {
            try {
                auto new_vac = Vacuum<double>(vac);
                minimize_potential_eff(params, new_vac);
                refine_root(params, new_vac);
                // Make sure this new vacuum is really a root.
                bool derivs_zero = verify_derivatives_zero(params, new_vac);
                // Make sure there are goldstones.
                bool is_cb = (abs(new_vac.vevs[2]) > 1e-5);
                bool has_goldstones = verify_goldstones(params, new_vac, is_cb);
                if (derivs_zero && has_goldstones) {
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

    /**
     * Find the deepest vacuum in the lot of vacuua. Additionally, find deepest
     * normal and charge-breaking minima.
     */
    void find_deepest_eff() {
        using std::abs;
        double TOL = 1e-5;
        for (const auto &vac : one_loop_vacuua) {
            if (vac.potential < one_loop_deepest.potential)
                one_loop_deepest = vac;
            if (abs(vac.vevs[2]) < TOL) {
                if (vac.potential < one_loop_deepest_normal.potential) {
                    one_loop_deepest_normal = vac;
                }
            }
            if (abs(vac.vevs[2]) > TOL) {
                if (vac.potential < one_loop_deepest_cb.potential) {
                    one_loop_deepest_cb = vac;
                }
            }
        }
    }

    /**
     * Determine if we have both charge-breaking and normal minima.
     */
    void determine_if_has_normal_cb_min() {
        using std::abs;
        double TOL = 1e-5;
        for (auto vac : one_loop_vacuua) {
            if (abs(vac.vevs[2]) < TOL) {
                if (vac.extrema_type == SingleExtremaType::Minimum)
                    has_normal_min = true;
            }
            if (abs(vac.vevs[2]) > TOL) {
                if (vac.extrema_type == SingleExtremaType::Minimum)
                    has_cb_min = true;
            }
        }
    }
};

}


#endif //THDM_MODEL_HPP
