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
#include <random>

namespace thdm {

struct Point {
    Parameters<double> params;
    Vacuum<double> nvac;
    Vacuum<double> cbvac;
};

class Model {

public:
    Fields<double> fields{};
    Parameters<double> params;

    std::vector<Vacuum<double>> one_loop_vacuua{};
    std::vector<Vacuum<double>> tree_vacuua{};

    /**
     * Initialize an empty model.
     */
    Model() = default;

    /**
     * Start a model from known parameters and vacuua.
     * @param Point A point consisting of parameters,
     */
    explicit Model(const Point &point) {
        params = point.params;
        auto nvac = point.nvac;
        auto cbvac = point.cbvac;
        set_top_yukawa(params, nvac);
        // Make sure vacuua are filled in.
        complete_vacuua(nvac, false);
        complete_vacuua(cbvac, false);
        // Add the known normal and charge-breaking vacuua to list.
        one_loop_vacuua.push_back(nvac);
        one_loop_vacuua.push_back(cbvac);
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
    }

    /**
     * Starting from all the tree roots, minimize the effective potential to
     * find new minima.
     */
    void minimize_from_tree_roots_and_refine() {
        // Find all tree roots:
        // Find all tree roots
        tree_vacuua = get_tree_roots(params);
        // complete tree-roots
        for (auto &vac : tree_vacuua) {
            complete_vacuua(vac, true);
        }
        // Starting from tree-roots, try to find a new one-loop min
        for (const auto &vac : tree_vacuua) {
            try {
                auto new_vac = vac;
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
            } catch (THDMException &e) {
                // that one didn't work! don't add it.
            }
        }
    }

    /**
     * Starting from a number of random vacuua, minimize the effective potential
     * and add unique results to vacuum list.
     * @param num_vacs
     */
    void minimize_from_random_vacuum_and_refine(int num_vacs) {
        static std::random_device rd{};
        static std::mt19937 engine{rd()};
        static std::uniform_real_distribution<double> dist{0.0, 1.0};

        // Generate random new vacuua
        std::vector<Vacuum<double>> random_vacs(num_vacs, Vacuum<double>{});
        for (auto &vac: random_vacs) {
            vac.vevs[0] = 2.0 * params.mu * (dist(engine) - 0.5);
            vac.vevs[1] = 2.0 * params.mu * (dist(engine) - 0.5);
            vac.vevs[2] = 2.0 * params.mu * (dist(engine) - 0.5);
        }
        // Starting from random vacuua, minimize and refine roots
        for (auto &vac: random_vacs) {
            try {
                minimize_potential_eff(params, vac);
                refine_root(params, vac);
                // Check that the vacuum is unique and valid.
                if (is_vacuum_unique(vac) && is_vacuum_valid(params, vac)) {
                    complete_vacuua(vac);
                    one_loop_vacuua.push_back(vac);
                }
            } catch (THDMException &e) {
                // Failed.
            }
        }
    }

    /**
     * Sort the one_loop vacuua so that the deepest is in the fist spot.
     */
    void sort_vacuua() {
        std::sort(one_loop_vacuua.begin(),
                  one_loop_vacuua.end(), std::less<Vacuum<double>>());
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
                done = done && is_vacuum_valid(params, cbvac);
                // Check that root-finder succeeded.
                done = done && (status == 0);
            } catch (THDMException &e) {
                done = false;
            }
        }
        complete_vacuua(nvac);
        complete_vacuua(cbvac);
        one_loop_vacuua.push_back(nvac);
        one_loop_vacuua.push_back(cbvac);
    }

    /**
     * Determine if a vacuum is unique against vacuua in one_loop_vacuua.
     * @param vac Vacuum to check
     * @return True if it is unique.
     */
    bool is_vacuum_unique(const Vacuum<double> &vac) {
        bool is_new = true;
        for (auto &current_vac: one_loop_vacuua) {
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
        return is_new;
    }
};

/**
 * Get the deepest vacuum.
 * @param model
 * @return
 */
Vacuum<double> get_deepest_vacuum(Model &model) {
    model.sort_vacuua();
    return model.one_loop_vacuua[0];
}

/**
 * Get the deepest cb vacuum.
 * @param model THDM vacuum.
 * @param Vacuum The deepest cb vacuum.
 * @return True if cb vacuum was found.
 */
bool get_deepest_cb_vacuum(Model &model, Vacuum<double> &cbvac) {
    model.sort_vacuua();
    bool found = false;
    for (auto &vac: model.one_loop_vacuua) {
        if (!is_vacuum_normal(vac)) {
            cbvac = vac;
            found = true;
            break;
        }
    }
    return found;
}

/**
 * Get the deepest normal vacuum.
 * @param model THDM vacuum.
 * @param Vacuum The deepest normal vacuum.
 * @return True if normal vacuum was found.
 */
bool get_deepest_normal_vacuum(Model &model, Vacuum<double> &nvac) {
    model.sort_vacuua();
    bool found = false;
    for (auto &vac: model.one_loop_vacuua) {
        if (is_vacuum_normal(vac)) {
            nvac = vac;
            found = true;
        }
    }
    return found;
}

/**
 * Determine if the model normal and charge-breaking minima
 * @param model Model to check
 * @param nvac Deepest normal minimum (if model has one.)
 * @param cbvac Deepest cb minimum (if model has one.)
 * @return
 */
bool has_cb_and_normal_minima(Model &model, Vacuum<double> &nvac,
                              Vacuum<double> &cbvac) {
    model.sort_vacuua();
    bool has_cb_min = false;
    bool has_n_min = false;
    for (auto &vac: model.one_loop_vacuua) {
        if (vac.extrema_type == SingleExtremaType::Minimum) {
            if (is_vacuum_normal(vac) && !has_n_min) {
                nvac = vac;
                has_n_min = true;
            } else if (!is_vacuum_normal(vac) && !has_cb_min) {
                cbvac = vac;
                has_cb_min = true;
            }
        }
    }
    return has_cb_min && has_n_min;
}


}


#endif //THDM_MODEL_HPP
