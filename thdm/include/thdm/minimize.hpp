//
// Created by Logan Morrison on 2019-04-26.
//

#ifndef THDM_MINIMIZE_HPP
#define THDM_MINIMIZE_HPP

#include "thdm/parameters.hpp"
#include "thdm/vacuua.hpp"
#include "thdm/fields.hpp"
#include "thdm/potentials.hpp"
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>

namespace thdm {

struct minimization_params {
    Parameters<double> &params;
    Fields<double> &fields;
    Vacuum<double> &vac;
};

double minimize_f(const gsl_vector *v, void *input_params) {
    auto *mparams = (struct minimization_params *) input_params;

    mparams->vac.vevs[0] = gsl_vector_get(v, 0);
    mparams->vac.vevs[1] = gsl_vector_get(v, 1);
    mparams->vac.vevs[2] = gsl_vector_get(v, 2);
    mparams->fields.set_fields(mparams->vac);

    return potential_eff(mparams->fields, mparams->params);
}

/* The gradient of f, df = (df/dx, df/dy). */
void minimize_df(const gsl_vector *v, void *input_params, gsl_vector *df) {
    auto *mparams = (struct minimization_params *) input_params;

    mparams->vac.vevs[0] = gsl_vector_get(v, 0);
    mparams->vac.vevs[1] = gsl_vector_get(v, 1);
    mparams->vac.vevs[2] = gsl_vector_get(v, 2);
    mparams->fields.set_fields(mparams->vac);

    gsl_vector_set(df, 0, potential_eff_deriv(mparams->fields, mparams->params, 1));
    gsl_vector_set(df, 1, potential_eff_deriv(mparams->fields, mparams->params, 2));
    gsl_vector_set(df, 2, potential_eff_deriv(mparams->fields, mparams->params, 3));
}

/* Compute both f and df together. */
void minimize_fdf(const gsl_vector *x, void *input_params, double *f, gsl_vector *df) {
    *f = minimize_f(x, input_params);
    minimize_df(x, input_params, df);
}

int minimize_potential_eff(Parameters<double> &params, Vacuum<double> &vac) {
    size_t iter = 0;
    int status;

    const gsl_multimin_fdfminimizer_type *T;
    gsl_multimin_fdfminimizer *s;

    Fields<double> fields{};
    struct minimization_params mparams = {
            params, fields, vac};

    gsl_vector *x;
    gsl_multimin_function_fdf minimize_func = {&minimize_f,
            &minimize_df,
            &minimize_fdf,
            3,
            &mparams};

    /* Starting point, x = (5,7) */
    x = gsl_vector_alloc(3);
    gsl_vector_set(x, 0, vac.vevs[0]);
    gsl_vector_set(x, 1, vac.vevs[1]);
    gsl_vector_set(x, 2, vac.vevs[2]);

    T = gsl_multimin_fdfminimizer_conjugate_fr;
    s = gsl_multimin_fdfminimizer_alloc(T, 3);

    gsl_multimin_fdfminimizer_set(s, &minimize_func, x, 0.01, 1e-7);

    do {
        iter++;
        status = gsl_multimin_fdfminimizer_iterate(s);

        if (status)
            break;

        status = gsl_multimin_test_gradient(s->gradient, 1e-7);

    } while (status == GSL_CONTINUE && iter < 1000);

    vac.vevs[0] = gsl_vector_get(s->x, 0);
    vac.vevs[1] = gsl_vector_get(s->x, 1);
    vac.vevs[2] = gsl_vector_get(s->x, 2);

    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(x);

    return status;
}


}

#endif //THDM_MINIMIZE_HPP
