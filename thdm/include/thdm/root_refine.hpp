//
// Created by Logan Morrison on 2019-04-26.
//

#ifndef THDM_ROOT_REFINE_HPP
#define THDM_ROOT_REFINE_HPP

#include "thdm/parameters.hpp"
#include "thdm/potentials.hpp"
#include "thdm/fields.hpp"
#include "thdm/vacuua.hpp"
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
 * Struct to hold parameters needed for GSL multiroot solver.
 */
struct roof_refine_params {
    Parameters<double> &params;
    Fields<double> &fields;

    void update_fields(const gsl_vector *x) {
        for (size_t i = 0; i < 3; i++) {
            fields[i] = gsl_vector_get(x, i);
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
int roof_refine_func(const gsl_vector *x, void *input_params, gsl_vector *f) {
    // Pull in the parameters
    auto *rparams = (struct roof_refine_params *) input_params;
    rparams->update_fields(x);

    gsl_vector_set(f, 0, potential_eff_deriv(rparams->fields, rparams->params, 1));
    gsl_vector_set(f, 1, potential_eff_deriv(rparams->fields, rparams->params, 2));
    gsl_vector_set(f, 2, potential_eff_deriv(rparams->fields, rparams->params, 3));


    return GSL_SUCCESS;
}

/**
 * Jacobian of equations for the effective potential.
 * @param x
 * @param input_params
 * @param f
 * @return
 */
int roof_refine_jac(const gsl_vector *x, void *input_params, gsl_matrix *J) {
    // Pull in the parameters
    auto *rparams = (struct roof_refine_params *) input_params;
    rparams->update_fields(x);

    // Compute the Jacobian for the second three root equations
    for (int i = 0; i < 3; i++) {
        gsl_matrix_set(J, 0, i, potential_eff_deriv_fld_par(
                rparams->fields, rparams->params, 1, i + 1));
        gsl_matrix_set(J, 1, i, potential_eff_deriv_fld_par(
                rparams->fields, rparams->params, 2, i + 1));
        gsl_matrix_set(J, 2, i, potential_eff_deriv_fld_par(
                rparams->fields, rparams->params, 3, i + 1));
    }

    return GSL_SUCCESS;
}

int roof_refine_fdf(const gsl_vector *x, void *params, gsl_vector *f, gsl_matrix *J) {
    roof_refine_func(x, params, f);
    roof_refine_jac(x, params, J);

    return GSL_SUCCESS;
}

int refine_root(Parameters<double> &params, Vacuum<double> &vac) {
    //Create GSL multi-root solver type
    const gsl_multiroot_fdfsolver_type *T;
    // Create the state variable for GSL multidimensional root solver
    gsl_multiroot_fdfsolver *s;

    int status;
    size_t iter = 0;

    // Set up multi-root solver parameters
    const size_t n = 3;
    Fields<double> fields{};
    struct roof_refine_params rparams = {params, fields};

    // Crete GSL root finding functions
    gsl_multiroot_function_fdf f = {&roof_refine_func,
            &roof_refine_jac,
            &roof_refine_fdf,
            n, &rparams};

    // Create GSL root finding variables and initialize
    gsl_vector *x = gsl_vector_alloc(n);
    // Initialize the search parameters
    for (size_t i = 0; i < n; i++) {
        gsl_vector_set(x, i, vac.vevs[i]);
    }

    // Chose hybrid method for root finding
    T = gsl_multiroot_fdfsolver_hybridsj;
    s = gsl_multiroot_fdfsolver_alloc(T, 3);
    gsl_multiroot_fdfsolver_set(s, &f, x);

    do {
        iter++;
        status = gsl_multiroot_fdfsolver_iterate(s);
        // Check if solver is stuck
        if (status)
            break;

        status = gsl_multiroot_test_residual(s->f, 1e-8);

    } while (status == GSL_CONTINUE && iter < 1000);

    vac.vevs[0] = gsl_vector_get(s->x, 0);
    vac.vevs[1] = gsl_vector_get(s->x, 1);
    vac.vevs[2] = gsl_vector_get(s->x, 2);

    // Determine if all the squared scalar masses are positive
    if (!are_sqrd_masses_positive_semi_definite(vac, params))
        status = -1;

    // Free up memory used by root solver.
    gsl_multiroot_fdfsolver_free(s);
    gsl_vector_free(x);

    return status;
}


/**
 * Struct to hold parameters needed for GSL multiroot solver.
 */
struct roof_refine_params_n {
    Parameters<double> &params;
    Fields<double> &fields;

    void update_fields(const gsl_vector *x) {
        fields[0] = gsl_vector_get(x, 0);
        fields[1] = gsl_vector_get(x, 1);
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
int roof_refine_func_n(const gsl_vector *x, void *input_params, gsl_vector *f) {
    // Pull in the parameters
    auto *rparams = (struct roof_refine_params_n *) input_params;
    rparams->update_fields(x);

    gsl_vector_set(f, 0, potential_eff_deriv(rparams->fields, rparams->params, 1));
    gsl_vector_set(f, 1, potential_eff_deriv(rparams->fields, rparams->params, 2));

    return GSL_SUCCESS;
}

/**
 * Jacobian of equations for the effective potential.
 * @param x
 * @param input_params
 * @param f
 * @return
 */
int roof_refine_jac_n(const gsl_vector *x, void *input_params, gsl_matrix *J) {
    // Pull in the parameters
    auto *rparams = (struct roof_refine_params_n *) input_params;
    rparams->update_fields(x);

    // Compute the Jacobian for the second three root equations
    for (int i = 0; i < 2; i++) {
        gsl_matrix_set(J, 0, i, potential_eff_deriv_fld_par(
                rparams->fields, rparams->params, 1, i + 1));
        gsl_matrix_set(J, 1, i, potential_eff_deriv_fld_par(
                rparams->fields, rparams->params, 2, i + 1));
    }

    return GSL_SUCCESS;
}

int roof_refine_fdf_n(const gsl_vector *x, void *params, gsl_vector *f, gsl_matrix *J) {
    roof_refine_func_n(x, params, f);
    roof_refine_jac_n(x, params, J);

    return GSL_SUCCESS;
}

int refine_root_n(Parameters<double> &params, Vacuum<double> &vac) {
    //Create GSL multi-root solver type
    const gsl_multiroot_fdfsolver_type *T;
    // Create the state variable for GSL multidimensional root solver
    gsl_multiroot_fdfsolver *s;

    int status;
    size_t iter = 0;

    // Set up multi-root solver parameters
    const size_t n = 2;
    Fields<double> fields{};
    struct roof_refine_params_n rparams = {params, fields};

    // Crete GSL root finding functions
    gsl_multiroot_function_fdf f = {&roof_refine_func_n,
            &roof_refine_jac_n,
            &roof_refine_fdf_n,
            n, &rparams};

    // Create GSL root finding variables and initialize
    gsl_vector *x = gsl_vector_alloc(n);
    // Initialize the search parameters
    for (size_t i = 0; i < n; i++) {
        gsl_vector_set(x, i, vac.vevs[i]);
    }

    // Chose hybrid method for root finding
    T = gsl_multiroot_fdfsolver_hybridsj;
    s = gsl_multiroot_fdfsolver_alloc(T, 2);
    gsl_multiroot_fdfsolver_set(s, &f, x);

    do {
        iter++;
        status = gsl_multiroot_fdfsolver_iterate(s);
        // Check if solver is stuck
        if (status)
            break;

        status = gsl_multiroot_test_residual(s->f, 1e-8);

    } while (status == GSL_CONTINUE && iter < 1000);

    vac.vevs[0] = gsl_vector_get(s->x, 0);
    vac.vevs[1] = gsl_vector_get(s->x, 1);

    // Determine if all the squared scalar masses are positive
    if (!are_sqrd_masses_positive_semi_definite(vac, params))
        status = -1;

    // Free up memory used by root solver.
    gsl_multiroot_fdfsolver_free(s);
    gsl_vector_free(x);

    return status;
}


} // namespace thdm



#endif //THDM_ROOT_REFINE_HPP
