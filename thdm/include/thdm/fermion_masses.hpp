//
// Created by Logan Morrison on 2019-05-09.
//

#ifndef THDM_FERMION_MASSES_HPP
#define THDM_FERMION_MASSES_HPP

#include "thdm/fields.hpp"
#include "thdm/parameters.hpp"
#include "thdm/vacuua.hpp"
#include "thdm/constants.hpp"
#include <vector>
#include <cmath>

namespace thdm {

double top_mass_squared(const Fields<double> &fields, const Parameters<double> &params) {
    double yt = params.yt;
    double r2 = fields.r2;
    double i2 = fields.i2;
    double c3 = fields.c3;
    double c4 = fields.c4;

    return 1. / 2. * (pow(c3, 2) + pow(c4, 2) + pow(i2, 2) + pow(r2, 2)) * pow(yt, 2);
}

double top_mass_squared_deriv(const Fields<double> &fields, const Parameters<double> &params, int fld) {
    double yt = params.yt;
    double r2 = fields.r2;
    double i2 = fields.i2;
    double c3 = fields.c3;
    double c4 = fields.c4;

    if (fld == 2) {
        return r2 * pow(yt, 2);
    } else if (fld == 5) {
        return c3 * pow(yt, 2);
    } else if (fld == 6) {
        return c4 * pow(yt, 2);
    } else if (fld == 8) {
        return i2 * pow(yt, 2);
    } else {
        return 0.0;
    }
}

double top_mass_squared_deriv(const Fields<double> &, const Parameters<double> &params, int fld1, int fld2) {

    if (fld1 == 2 && fld2 == 2) {
        return pow(params.yt, 2);
    } else if (fld1 == 5 && fld2 == 5) {
        return pow(params.yt, 2);
    } else if (fld1 == 6 && fld2 == 6) {
        return pow(params.yt, 2);
    } else if (fld1 == 8 && fld2 == 8) {
        return pow(params.yt, 2);
    } else {
        return 0.0;
    }

}

void set_top_yukawa(Parameters<double> &params, const Vacuum<double> &nvac) {
    double r2 = nvac.vevs[1];
    params.yt = sqrt(2) * M_TOP / fabs(r2);
}

}

#endif //THDM_FERMION_MASSES_HPP
