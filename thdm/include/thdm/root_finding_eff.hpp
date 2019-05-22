#ifndef ROOT_FINDING_EFF_HPP
#define ROOT_FINDING_EFF_HPP

#include "thdm/parameters.hpp"
#include "thdm/potentials.hpp"
#include "thdm/fields.hpp"
#include "thdm/vacuua.hpp"
#include "thdm/fermion_masses.hpp"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multiroots.h>
#include <vector>
#include <random>
#include <algorithm>
#include <map>
#include <iostream>
#include <tuple>

namespace thdm {


/**
 * Create a random injective key from [1,5] to [1,8] which
 * shuffles the order of the THDM parameters.
 * @return std::map<int,int> key
 */
std::map<int, int> create_key() {
    // Create random number generator that will be used
    // for all calls to this function.
    static std::random_device PShufflerRD{};
    static std::mt19937 PShufflerRng{PShufflerRD()};
    // Create the list of indices. This will be static as well.
    static std::vector<int> shuffled_indices = {0, 1, 2, 3, 4, 5, 6, 7};

    std::shuffle(std::begin(shuffled_indices),
                 std::end(shuffled_indices),
                 PShufflerRng);

    std::map<int, int> key;
    for (size_t i = 0; i < shuffled_indices.size(); i++) {
        key[i] = shuffled_indices[i];
        //key.insert(std::make_tuple(i, shuffled_indices[i]));
    }
    return key;
}

/**
 * Struct to hold parameters needed for GSL multiroot solver.
 */
struct eff_root_params {
    Parameters<double> &params;
    Fields<double> &fields;
    Vacuum<double> &vac_normal;
    Vacuum<double> &vac_cb;
    std::map<int, int> &key;

    void update_params(const gsl_vector *x) {
        for (size_t i = 0; i < 5; i++) {
            params[key[i]] = gsl_vector_get(x, i);
        }
    }
};

/**
 * Root equations for the effective potential.
 *
 * There are five equations:
 *      dV/dr1 = 0, dV/dr2 = 0
 *      dV/dr1' = 0, dV/dr2' = 0, dV/dc1' = 0
 * where the `'` is for CB and non-primed variables
 * are normal.
 * @param x
 * @param input_params
 * @param f
 * @return
 */
int root_functions_eff(const gsl_vector *x, void *input_params, gsl_vector *f) {
    // Pull in the parameters
    auto *rparams = (struct eff_root_params *) input_params;
    rparams->update_params(x);

    rparams->fields.set_fields(rparams->vac_normal);
    gsl_vector_set(f, 0, potential_eff_deriv(rparams->fields, rparams->params, 1));
    gsl_vector_set(f, 1, potential_eff_deriv(rparams->fields, rparams->params, 2));
    rparams->fields.set_fields(rparams->vac_cb);
    gsl_vector_set(f, 2, potential_eff_deriv(rparams->fields, rparams->params, 1));
    gsl_vector_set(f, 3, potential_eff_deriv(rparams->fields, rparams->params, 2));
    gsl_vector_set(f, 4, potential_eff_deriv(rparams->fields, rparams->params, 3));


    return GSL_SUCCESS;
}

/**
 * Jacobian of equations for the effective potential.
 * @param x
 * @param input_params
 * @param f
 * @return
 */
int root_functions_eff_jacobian(const gsl_vector *x, void *input_params, gsl_matrix *J) {
    // Pull in the parameters
    auto *rparams = (struct eff_root_params *) input_params;
    rparams->update_params(x);

    // Compute the Jacobian for the first two root equations
    rparams->fields.set_fields(rparams->vac_normal);
    for (int i = 0; i < 5; i++) {
        gsl_matrix_set(J, 0, i, potential_eff_deriv_fld_par(
                rparams->fields, rparams->params, 1, rparams->key[i] + 1));
        gsl_matrix_set(J, 1, i, potential_eff_deriv_fld_par(
                rparams->fields, rparams->params, 2, rparams->key[i] + 1));
    }

    // Compute the Jacobian for the second three root equations
    rparams->fields.set_fields(rparams->vac_cb);
    for (int i = 0; i < 5; i++) {
        gsl_matrix_set(J, 2, i, potential_eff_deriv_fld_par(
                rparams->fields, rparams->params, 1, rparams->key[i] + 1));
        gsl_matrix_set(J, 3, i, potential_eff_deriv_fld_par(
                rparams->fields, rparams->params, 2, rparams->key[i] + 1));
        gsl_matrix_set(J, 4, i, potential_eff_deriv_fld_par(
                rparams->fields, rparams->params, 3, rparams->key[i] + 1));
    }
    return GSL_SUCCESS;
}

int root_functions_eff_fdf(const gsl_vector *x, void *params, gsl_vector *f, gsl_matrix *J) {
    root_functions_eff(x, params, f);
    root_functions_eff_jacobian(x, params, J);

    return GSL_SUCCESS;
}

int try_solve_root_equations_eff(Vacuum<double> &nvac, Vacuum<double> &cbvac, Parameters<double> &params) {
    //Create GSL multi-root solver type
    const gsl_multiroot_fdfsolver_type *T;
    // Create the state variable for GSL multidimensional root solver
    gsl_multiroot_fdfsolver *s;

    int status;
    size_t iter = 0;

    // Set up multi-root solver parameters
    const size_t n = 5;
    Fields<double> fields{};
    auto key = create_key();
    struct eff_root_params rparams = {params, fields,
            nvac, cbvac,
            key};

    // Crete GSL root finding functions
    gsl_multiroot_function_fdf f = {&root_functions_eff,
            &root_functions_eff_jacobian,
            &root_functions_eff_fdf,
            n, &rparams};

    // Create GSL root finding variables and initialize
    gsl_vector *x = gsl_vector_alloc(n);
    // Initialize the search parameters
    for (int i = 0; i < 5; i++) {
        gsl_vector_set(x, i, params[key[i]]);
    }

    // Chose hybrid method for root finding
    T = gsl_multiroot_fdfsolver_hybridsj;
    s = gsl_multiroot_fdfsolver_alloc(T, 5);
    gsl_multiroot_fdfsolver_set(s, &f, x);

    do {
        iter++;
        status = gsl_multiroot_fdfsolver_iterate(s);
        // Check if solver is stuck
        if (status)
            break;

        status = gsl_multiroot_test_residual(s->f, 1e-8);

    } while (status == GSL_CONTINUE && iter < 1000);

    // Determine if all the squared scalar masses are positive
    if (!are_sqrd_masses_positive_semi_definite(nvac, cbvac, params))
        status = -1;

    // Free up memory used by root solver.
    gsl_multiroot_fdfsolver_free(s);
    gsl_vector_free(x);

    return status;
}

} // namespace thdm


#endif //ROOT_FINDING_EFF_HPP