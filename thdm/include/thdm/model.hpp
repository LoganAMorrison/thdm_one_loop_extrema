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
#include "thdm/validation.hpp"
#include <vector>
#include <tuple>
#include <iostream>

namespace thdm {

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
            int status;
            // Initialize the parameters. Random values will be chosen
            // such that the potential is bounded from below
            params = Parameters<double>{renorm_scale};
            // Create random normal and cb vacuua
            nvac = generate_normal_vac(renorm_scale);
            cbvac = generate_cb_vac(renorm_scale);
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
                if (is_vacuum_valid(params, new_vac)) {
                    // Make sure root isn't duplicate
                    bool should_add = true;
                    for (const auto &one_loop_vac : one_loop_vacuua) {
                        if (are_vacuua_approx_equal(new_vac, one_loop_vac)) {
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
        for (const auto &vac : one_loop_vacuua) {
            if (vac.potential < one_loop_deepest.potential)
                one_loop_deepest = vac;
            if (is_vacuum_normal(vac)) {
                if (vac.potential < one_loop_deepest_normal.potential) {
                    one_loop_deepest_normal = vac;
                }
            }
            if (!is_vacuum_normal(vac)) {
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
        has_normal_min = false;
        has_cb_min = false;
        for (auto &vac : one_loop_vacuua) {
            if (is_vacuum_normal(vac) &&
                    vac.extrema_type == SingleExtremaType::Minimum) {
                has_normal_min = true;
            }
            if (!is_vacuum_normal(vac) &&
                    vac.extrema_type == SingleExtremaType::Minimum) {
                has_cb_min = true;
            }
        }
    }
};

}


#endif //THDM_MODEL_HPP
