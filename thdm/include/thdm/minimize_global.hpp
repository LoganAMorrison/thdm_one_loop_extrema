//
// Created by Logan Morrison on 2019-05-03.
//

#ifndef THDM_MINIMIZE_GLOBAL_HPP
#define THDM_MINIMIZE_GLOBAL_HPP

#include "thdm/potentials.hpp"
#include "thdm/fields.hpp"
#include "thdm/parameters.hpp"
#include <iomanip>
#include <iostream>
#include <nlopt.hpp>
#include <vector>

namespace thdm {

typedef struct {
    Fields<double> fields;
    Parameters<double> params;
} one_loop_global_minimization_data;

double one_loop_global_minimization_func(unsigned n, const double *x, double *grad, void *data) {
    auto *d = (one_loop_global_minimization_data *) data;
    d->fields.r1 = x[0];
    d->fields.r2 = x[1];
    d->fields.c1 = x[2];
    if (grad) {
        grad[0] = potential_eff_deriv(d->fields, d->params, 1);
        grad[1] = potential_eff_deriv(d->fields, d->params, 2);
        grad[2] = potential_eff_deriv(d->fields, d->params, 3);
    }
    return potential_eff(d->fields, d->params);
}

void one_loop_global_minimization(Parameters<double> &params) {
    double mu = params.mu;
    nlopt::opt opt(nlopt::G_MLSL, 3);/* algorithm and dimensionality */
    nlopt::opt lopt(nlopt::LD_MMA, 3);/* algorithm and dimensionality */
    std::vector<double> lb{-5 * mu, -5 * mu, -5 * mu}; /* lower bounds */
    std::vector<double> ub{5 * mu, 5 * mu, 5 * mu}; /* upper bounds */

    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);
    lopt.set_lower_bounds(lb);
    lopt.set_upper_bounds(ub);

    Fields<double> fields{};
    one_loop_global_minimization_data data{fields, params};
    opt.set_min_objective(one_loop_global_minimization_func, &data);
    lopt.set_min_objective(one_loop_global_minimization_func, &data);
    opt.set_local_optimizer(lopt);

    std::vector<double> x{mu, mu, mu};
    double minf;

    nlopt::result result = opt.optimize(x, minf);
    std::cout << "found minimum at f(" << x[0] << "," << x[1] << ") = "
              << std::setprecision(10) << minf << std::endl;
}


}

#endif //THDM_MINIMIZE_GLOBAL_HPP
