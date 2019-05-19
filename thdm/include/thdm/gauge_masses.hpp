//
// Created by Logan Morrison on 2019-05-09.
//

#ifndef THDM_GAUGE_MASSES_HPP
#define THDM_GAUGE_MASSES_HPP

#include "thdm/fields.hpp"
#include "thdm/parameters.hpp"
#include <armadillo>
#include <cmath>
#include <vector>
#include <tuple>

namespace thdm {

arma::mat gauge_sqaured_mass_matrix(const Fields<double> &fields,
                                    const Parameters<double> &params) {

    double r1 = fields.r1;
    double r2 = fields.r2;
    double c1 = fields.c1;
    double c2 = fields.c2;
    double c3 = fields.c3;
    double c4 = fields.c4;
    double i1 = fields.i1;
    double i2 = fields.i2;

    double x = pow(r1, 2) + pow(r2, 2) + pow(i1, 2) + pow(i2, 2);
    double y = pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2);
    double z = c2 * i1 + c4 * i2 + c1 * r1 + c3 * r2;
    double w = c1 * i1 + c3 * i2 - c2 * r1 - c4 * r2;
    double tw = params.gp / params.g;

    return pow(params.g, 2) / 4. *
            arma::mat{
                    {x + y, 0., 0., 2. * tw * z},
                    {0., x + y, 0., 2. * tw * w},
                    {0., 0., x + y, tw * (y - x)},
                    {2. * tw * z, 2. * tw * w, tw * (y - x), pow(tw, 2) * (x + y)}};
}


std::vector<double> gauge_squared_masses(const Fields<double> &fields,
                                         const Parameters<double> &params) {
    double r1 = fields.r1;
    double r2 = fields.r2;
    double c1 = fields.c1;
    double c2 = fields.c2;
    double c3 = fields.c3;
    double c4 = fields.c4;
    double i1 = fields.i1;
    double i2 = fields.i2;

    double x = pow(r1, 2) + pow(r2, 2) + pow(i1, 2) + pow(i2, 2);
    double y = pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2);
    double z = c2 * i1 + c4 * i2 + c1 * r1 + c3 * r2;
    double w = c1 * i1 + c3 * i2 - c2 * r1 - c4 * r2;
    double tw = params.gp / params.g;

    double sw = params.gp / sqrt(pow(params.gp, 2) + pow(params.g, 2));
    double cw = params.g / sqrt(pow(params.gp, 2) + pow(params.g, 2));

    double sqrt_fac = sqrt(pow(x + y, 2) + 16. * pow(sw, 2) * pow(cw, 2) *
            (pow(w, 2) - x * y + pow(z, 2)));

    double mW = pow(params.g, 2) / 4. * (x + y);
    double mZ = pow(params.g, 2) / 8. * (1. + pow(tw, 2)) * (x + y + sqrt_fac);
    double mA = pow(params.g, 2) / 8. * (1. + pow(tw, 2)) * (x + y - sqrt_fac);

    return std::vector<double>{mW, mW, mZ, mA};
}

std::vector<std::tuple<double, double>>
gauge_squared_masses_deriv(const Fields<double> &fields,
                           const Parameters<double> &params, int fld) {

    double r1 = fields.r1;
    double r2 = fields.r2;
    double c1 = fields.c1;
    double c2 = fields.c2;
    double c3 = fields.c3;
    double c4 = fields.c4;
    double i1 = fields.i1;
    double i2 = fields.i2;

    double sw = params.gp / sqrt(pow(params.gp, 2) + pow(params.g, 2));
    double cw = params.g / sqrt(pow(params.gp, 2) + pow(params.g, 2));

    auto masses = gauge_squared_masses(fields, params);
    std::vector<double> deriv(4, 1.0);

    if (fld == 1) {
        deriv[0] = ((pow(params.g, 2) * r1) / 2.);
        deriv[1] = ((pow(params.g, 2) * r1) / 2.);
        deriv[2] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) * (r1 +
                (pow(c1, 2) * r1 + pow(c2, 2) * r1 +
                        8 * c1 * pow(cw, 2) * (c4 * i2 + c3 * r2) * pow(sw, 2) +
                        8 * c2 * pow(cw, 2) * (-(c3 * i2) + c4 * r2) * pow(sw, 2) +
                        r1 * (pow(c3, 2) + pow(c4, 2) + pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2) -
                                8 * pow(c3, 2) * pow(cw, 2) * pow(sw, 2) -
                                8 * pow(c4, 2) * pow(cw, 2) * pow(sw, 2))) /
                        sqrt(pow(pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2) +
                                         pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2), 2) + 16 * pow(cw, 2) *
                                (pow(c2 * i1 + c4 * i2 + c1 * r1 + c3 * r2, 2) +
                                        pow(c1 * i1 + c3 * i2 - c2 * r1 - c4 * r2, 2) -
                                        (pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2)) *
                                                (pow(i1, 2) + pow(i2, 2) + pow(r1, 2) +
                                                        pow(r2, 2))) * pow(sw, 2)))) / 4.);
        deriv[3] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) *
                (2 * r1 -
                        (2 * (pow(c1, 2) * r1 + pow(c2, 2) * r1 +
                                8 * c1 * pow(cw, 2) * (c4 * i2 + c3 * r2) * pow(sw, 2) +
                                8 * c2 * pow(cw, 2) * (-(c3 * i2) + c4 * r2) * pow(sw, 2) +
                                r1 * (pow(c3, 2) + pow(c4, 2) + pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2) -
                                        8 * pow(c3, 2) * pow(cw, 2) * pow(sw, 2) -
                                        8 * pow(c4, 2) * pow(cw, 2) * pow(sw, 2)))) /
                                sqrt(pow(pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2) +
                                                 pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2), 2) +
                                             16 * pow(cw, 2) * (pow(c2 * i1 + c4 * i2 + c1 * r1 + c3 * r2, 2) +
                                                     pow(c1 * i1 + c3 * i2 - c2 * r1 - c4 * r2, 2) -
                                                     (pow(c1, 2) + pow(c2, 2) + pow(c3, 2) +
                                                             pow(c4, 2)) *
                                                             (pow(i1, 2) + pow(i2, 2) + pow(r1, 2) +
                                                                     pow(r2, 2))) *
                                                     pow(sw, 2)))) / 8.);

    } else if (fld == 2) {
        deriv[0] = ((pow(params.g, 2) * r2) / 2.);
        deriv[1] = ((pow(params.g, 2) * r2) / 2.);
        deriv[2] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) * (r2 +
                (r2 * (pow(c3, 2) + pow(c4, 2) + pow(i1, 2) + pow(i2, 2) +
                        pow(r1, 2) + pow(r2, 2)) +
                        8 * c1 * pow(cw, 2) * (-(c4 * i1) + c3 * r1) * pow(sw, 2) +
                        8 * c2 * pow(cw, 2) * (c3 * i1 + c4 * r1) * pow(sw, 2) +
                        pow(c1, 2) * (r2 - 8 * pow(cw, 2) * r2 * pow(sw, 2)) +
                        pow(c2, 2) * (r2 - 8 * pow(cw, 2) * r2 * pow(sw, 2))) /
                        sqrt(pow(pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2) +
                                         pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2),
                                 2) +
                                     16 * pow(cw, 2) *
                                             (pow(c2 * i1 + c4 * i2 + c1 * r1 + c3 * r2, 2) +
                                                     pow(c1 * i1 + c3 * i2 - c2 * r1 - c4 * r2, 2) -
                                                     (pow(c1, 2) + pow(c2, 2) + pow(c3, 2) +
                                                             pow(c4, 2)) *
                                                             (pow(i1, 2) + pow(i2, 2) + pow(r1, 2) +
                                                                     pow(r2, 2))) *
                                             pow(sw, 2)))) /
                4.);
        deriv[3] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) *
                (2 * r2 -
                        (2 * (r2 * (pow(c3, 2) + pow(c4, 2) + pow(i1, 2) + pow(i2, 2) +
                                pow(r1, 2) + pow(r2, 2)) +
                                8 * c1 * pow(cw, 2) * (-(c4 * i1) + c3 * r1) * pow(sw, 2) +
                                8 * c2 * pow(cw, 2) * (c3 * i1 + c4 * r1) * pow(sw, 2) +
                                pow(c1, 2) * (r2 - 8 * pow(cw, 2) * r2 * pow(sw, 2)) +
                                pow(c2, 2) * (r2 - 8 * pow(cw, 2) * r2 * pow(sw, 2)))) /
                                sqrt(pow(pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2) +
                                                 pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2),
                                         2) +
                                             16 * pow(cw, 2) *
                                                     (pow(c2 * i1 + c4 * i2 + c1 * r1 + c3 * r2, 2) +
                                                             pow(c1 * i1 + c3 * i2 - c2 * r1 - c4 * r2, 2) -
                                                             (pow(c1, 2) + pow(c2, 2) + pow(c3, 2) +
                                                                     pow(c4, 2)) *
                                                                     (pow(i1, 2) + pow(i2, 2) + pow(r1, 2) +
                                                                             pow(r2, 2))) *
                                                     pow(sw, 2)))) / 8.);

    } else if (fld == 3) {
        deriv[0] = ((c1 * pow(params.g, 2)) / 2.);
        deriv[1] = ((c1 * pow(params.g, 2)) / 2.);
        deriv[2] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) * (c1 + (pow(c1, 3) +
                8 * pow(cw, 2) * (c3 * i1 * i2 + c4 * i2 * r1 - c4 * i1 * r2 + c3 * r1 * r2) * pow(sw, 2) +
                c1 * (pow(c2, 2) + pow(c3, 2) + pow(c4, 2) + pow(i1, 2) +
                        pow(i2, 2) + pow(r1, 2) + pow(r2, 2) -
                        8 * pow(cw, 2) * pow(i2, 2) * pow(sw, 2) -
                        8 * pow(cw, 2) * pow(r2, 2) * pow(sw, 2))) /
                sqrt(pow(pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2) +
                                 pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2),
                         2) +
                             16 * pow(cw, 2) *
                                     (pow(c2 * i1 + c4 * i2 + c1 * r1 + c3 * r2, 2) +
                                             pow(c1 * i1 + c3 * i2 - c2 * r1 - c4 * r2, 2) -
                                             (pow(c1, 2) + pow(c2, 2) + pow(c3, 2) +
                                                     pow(c4, 2)) *
                                                     (pow(i1, 2) + pow(i2, 2) + pow(r1, 2) +
                                                             pow(r2, 2))) *
                                     pow(sw, 2)))) /
                4.);
        deriv[3] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) *
                (2 * c1 - (2 * (pow(c1, 3) + 8 * pow(cw, 2) *
                        (c3 * i1 * i2 + c4 * i2 * r1 - c4 * i1 * r2 + c3 * r1 * r2) * pow(sw, 2) +
                        c1 * (pow(c2, 2) + pow(c3, 2) + pow(c4, 2) + pow(i1, 2) +
                                pow(i2, 2) + pow(r1, 2) + pow(r2, 2) -
                                8 * pow(cw, 2) * pow(i2, 2) * pow(sw, 2) -
                                8 * pow(cw, 2) * pow(r2, 2) * pow(sw, 2)))) /
                        sqrt(pow(pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2) +
                                         pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2), 2) + 16 * pow(cw, 2) *
                                (pow(c2 * i1 + c4 * i2 + c1 * r1 + c3 * r2, 2) +
                                        pow(c1 * i1 + c3 * i2 - c2 * r1 - c4 * r2, 2) -
                                        (pow(c1, 2) + pow(c2, 2) + pow(c3, 2) +
                                                pow(c4, 2)) * (pow(i1, 2) + pow(i2, 2) + pow(r1, 2) +
                                                pow(r2, 2))) * pow(sw, 2)))) / 8.);

    } else if (fld == 4) {
        deriv[0] = ((c2 * pow(params.g, 2)) / 2.);
        deriv[1] = ((c2 * pow(params.g, 2)) / 2.);
        deriv[2] = (
                ((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) * (c2 + (pow(c1, 2) * c2 + pow(c2, 3) +
                        8 * pow(cw, 2) * (c4 * i1 * i2 - c3 * i2 * r1 + c3 * i1 * r2 + c4 * r1 * r2) * pow(sw, 2) +
                        c2 * (pow(c3, 2) + pow(c4, 2) + pow(i1, 2) + pow(i2, 2) +
                                pow(r1, 2) + pow(r2, 2) -
                                8 * pow(cw, 2) * pow(i2, 2) * pow(sw, 2) -
                                8 * pow(cw, 2) * pow(r2, 2) * pow(sw, 2))) /
                        sqrt(pow(pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2) +
                                         pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2), 2) + 16 * pow(cw, 2) *
                                (pow(c2 * i1 + c4 * i2 + c1 * r1 + c3 * r2, 2) +
                                        pow(c1 * i1 + c3 * i2 - c2 * r1 - c4 * r2, 2) -
                                        (pow(c1, 2) + pow(c2, 2) + pow(c3, 2) +
                                                pow(c4, 2)) *
                                                (pow(i1, 2) + pow(i2, 2) + pow(r1, 2) +
                                                        pow(r2, 2))) * pow(sw, 2)))) / 4.);
        deriv[3] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) *
                (2 * c2 - (2 * (pow(c1, 2) * c2 + pow(c2, 3) +
                        8 * pow(cw, 2) * (c4 * i1 * i2 - c3 * i2 * r1 + c3 * i1 * r2 + c4 * r1 * r2) * pow(sw, 2) +
                        c2 * (pow(c3, 2) + pow(c4, 2) + pow(i1, 2) + pow(i2, 2) +
                                pow(r1, 2) + pow(r2, 2) -
                                8 * pow(cw, 2) * pow(i2, 2) * pow(sw, 2) -
                                8 * pow(cw, 2) * pow(r2, 2) * pow(sw, 2)))) /
                        sqrt(pow(pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2) +
                                         pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2), 2) + 16 * pow(cw, 2) *
                                (pow(c2 * i1 + c4 * i2 + c1 * r1 + c3 * r2, 2) +
                                        pow(c1 * i1 + c3 * i2 - c2 * r1 - c4 * r2, 2) -
                                        (pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2)) *
                                                (pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2))) * pow(sw, 2)))) /
                8.);

    } else if (fld == 5) {
        deriv[0] = ((c3 * pow(params.g, 2)) / 2.);
        deriv[1] = ((c3 * pow(params.g, 2)) / 2.);
        deriv[2] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) *
                (c3 + (pow(c1, 2) * c3 + pow(c2, 2) * c3 +
                        8 * c2 * pow(cw, 2) * (-(i2 * r1) + i1 * r2) * pow(sw, 2) +
                        8 * c1 * pow(cw, 2) * (i1 * i2 + r1 * r2) * pow(sw, 2) +
                        c3 * (pow(c3, 2) + pow(c4, 2) + pow(i1, 2) + pow(i2, 2) +
                                pow(r1, 2) + pow(r2, 2) -
                                8 * pow(cw, 2) * pow(i1, 2) * pow(sw, 2) -
                                8 * pow(cw, 2) * pow(r1, 2) * pow(sw, 2))) /
                        sqrt(pow(pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2) +
                                         pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2), 2) + 16 * pow(cw, 2) *
                                (pow(c2 * i1 + c4 * i2 + c1 * r1 + c3 * r2, 2) +
                                        pow(c1 * i1 + c3 * i2 - c2 * r1 - c4 * r2, 2) -
                                        (pow(c1, 2) + pow(c2, 2) + pow(c3, 2) +
                                                pow(c4, 2)) *
                                                (pow(i1, 2) + pow(i2, 2) + pow(r1, 2) +
                                                        pow(r2, 2))) * pow(sw, 2)))) / 4.);
        deriv[3] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) * (2 * c3 -
                (2 * (pow(c1, 2) * c3 + pow(c2, 2) * c3 +
                        8 * c2 * pow(cw, 2) * (-(i2 * r1) + i1 * r2) * pow(sw, 2) +
                        8 * c1 * pow(cw, 2) * (i1 * i2 + r1 * r2) * pow(sw, 2) +
                        c3 * (pow(c3, 2) + pow(c4, 2) + pow(i1, 2) + pow(i2, 2) +
                                pow(r1, 2) + pow(r2, 2) -
                                8 * pow(cw, 2) * pow(i1, 2) * pow(sw, 2) -
                                8 * pow(cw, 2) * pow(r1, 2) * pow(sw, 2)))) /
                        sqrt(pow(pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2) +
                                         pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2), 2) + 16 * pow(cw, 2) *
                                (pow(c2 * i1 + c4 * i2 + c1 * r1 + c3 * r2, 2) +
                                        pow(c1 * i1 + c3 * i2 - c2 * r1 - c4 * r2, 2) -
                                        (pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2)) *
                                                (pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2))) *
                                pow(sw, 2)))) / 8.);

    } else if (fld == 6) {
        deriv[0] = ((c4 * pow(params.g, 2)) / 2.);
        deriv[1] = ((c4 * pow(params.g, 2)) / 2.);
        deriv[2] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) *
                (c4 + (pow(c1, 2) * c4 + pow(c2, 2) * c4 +
                        8 * c1 * pow(cw, 2) * (i2 * r1 - i1 * r2) * pow(sw, 2) +
                        8 * c2 * pow(cw, 2) * (i1 * i2 + r1 * r2) * pow(sw, 2) +
                        c4 * (pow(c3, 2) + pow(c4, 2) + pow(i1, 2) + pow(i2, 2) +
                                pow(r1, 2) + pow(r2, 2) -
                                8 * pow(cw, 2) * pow(i1, 2) * pow(sw, 2) -
                                8 * pow(cw, 2) * pow(r1, 2) * pow(sw, 2))) /
                        sqrt(pow(pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2) +
                                         pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2), 2) +
                                     16 * pow(cw, 2) * (pow(c2 * i1 + c4 * i2 + c1 * r1 + c3 * r2, 2) +
                                             pow(c1 * i1 + c3 * i2 - c2 * r1 - c4 * r2, 2) -
                                             (pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2)) *
                                                     (pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2))) *
                                             pow(sw, 2)))) / 4.);
        deriv[3] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) * (2 * c4 -
                (2 * (pow(c1, 2) * c4 + pow(c2, 2) * c4 +
                        8 * c1 * pow(cw, 2) * (i2 * r1 - i1 * r2) * pow(sw, 2) +
                        8 * c2 * pow(cw, 2) * (i1 * i2 + r1 * r2) * pow(sw, 2) +
                        c4 * (pow(c3, 2) + pow(c4, 2) + pow(i1, 2) + pow(i2, 2) +
                                pow(r1, 2) + pow(r2, 2) -
                                8 * pow(cw, 2) * pow(i1, 2) * pow(sw, 2) -
                                8 * pow(cw, 2) * pow(r1, 2) * pow(sw, 2)))) /
                        sqrt(pow(pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2) +
                                         pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2), 2) +
                                     16 * pow(cw, 2) * (pow(c2 * i1 + c4 * i2 + c1 * r1 + c3 * r2, 2) +
                                             pow(c1 * i1 + c3 * i2 - c2 * r1 - c4 * r2, 2) -
                                             (pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2)) *
                                                     (pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2))) *
                                             pow(sw, 2)))) / 8.);

    } else if (fld == 7) {
        deriv[0] = ((pow(params.g, 2) * i1) / 2.);
        deriv[1] = ((pow(params.g, 2) * i1) / 2.);
        deriv[2] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) *
                (i1 + (pow(c1, 2) * i1 + pow(c2, 2) * i1 +
                        8 * c2 * pow(cw, 2) * (c4 * i2 + c3 * r2) * pow(sw, 2) +
                        8 * c1 * pow(cw, 2) * (c3 * i2 - c4 * r2) * pow(sw, 2) +
                        i1 * (pow(c3, 2) + pow(c4, 2) + pow(i1, 2) + pow(i2, 2) +
                                pow(r1, 2) + pow(r2, 2) -
                                8 * pow(c3, 2) * pow(cw, 2) * pow(sw, 2) -
                                8 * pow(c4, 2) * pow(cw, 2) * pow(sw, 2))) /
                        sqrt(pow(pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2) +
                                         pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2),
                                 2) +
                                     16 * pow(cw, 2) *
                                             (pow(c2 * i1 + c4 * i2 + c1 * r1 + c3 * r2, 2) +
                                                     pow(c1 * i1 + c3 * i2 - c2 * r1 - c4 * r2, 2) -
                                                     (pow(c1, 2) + pow(c2, 2) + pow(c3, 2) +
                                                             pow(c4, 2)) *
                                                             (pow(i1, 2) + pow(i2, 2) + pow(r1, 2) +
                                                                     pow(r2, 2))) *
                                             pow(sw, 2)))) /
                4.);
        deriv[3] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) * (2 * i1 -
                (2 * (pow(c1, 2) * i1 + pow(c2, 2) * i1 +
                        8 * c2 * pow(cw, 2) * (c4 * i2 + c3 * r2) * pow(sw, 2) +
                        8 * c1 * pow(cw, 2) * (c3 * i2 - c4 * r2) * pow(sw, 2) +
                        i1 * (pow(c3, 2) + pow(c4, 2) + pow(i1, 2) + pow(i2, 2) +
                                pow(r1, 2) + pow(r2, 2) -
                                8 * pow(c3, 2) * pow(cw, 2) * pow(sw, 2) -
                                8 * pow(c4, 2) * pow(cw, 2) * pow(sw, 2)))) /
                        sqrt(pow(pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2) +
                                         pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2), 2) + 16 * pow(cw, 2) *
                                (pow(c2 * i1 + c4 * i2 + c1 * r1 + c3 * r2, 2) +
                                        pow(c1 * i1 + c3 * i2 - c2 * r1 - c4 * r2, 2) -
                                        (pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2)) *
                                                (pow(i1, 2) + pow(i2, 2) + pow(r1, 2) +
                                                        pow(r2, 2))) * pow(sw, 2)))) / 8.);

    } else if (fld == 8) {
        deriv[0] = ((pow(params.g, 2) * i2) / 2.);
        deriv[1] = ((pow(params.g, 2) * i2) / 2.);
        deriv[2] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) * (i2 +
                (i2 * (pow(c3, 2) + pow(c4, 2) + pow(i1, 2) + pow(i2, 2) +
                        pow(r1, 2) + pow(r2, 2)) +
                        8 * c2 * pow(cw, 2) * (c4 * i1 - c3 * r1) * pow(sw, 2) +
                        8 * c1 * pow(cw, 2) * (c3 * i1 + c4 * r1) * pow(sw, 2) +
                        pow(c1, 2) * (i2 - 8 * pow(cw, 2) * i2 * pow(sw, 2)) +
                        pow(c2, 2) * (i2 - 8 * pow(cw, 2) * i2 * pow(sw, 2))) /
                        sqrt(pow(pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2) +
                                         pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2), 2) + 16 * pow(cw, 2) *
                                (pow(c2 * i1 + c4 * i2 + c1 * r1 + c3 * r2, 2) +
                                        pow(c1 * i1 + c3 * i2 - c2 * r1 - c4 * r2, 2) -
                                        (pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2)) *
                                                (pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2))) *
                                pow(sw, 2)))) / 4.);
        deriv[3] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) * (2 * i2 -
                (2 * (i2 * (pow(c3, 2) + pow(c4, 2) + pow(i1, 2) + pow(i2, 2) +
                        pow(r1, 2) + pow(r2, 2)) +
                        8 * c2 * pow(cw, 2) * (c4 * i1 - c3 * r1) * pow(sw, 2) +
                        8 * c1 * pow(cw, 2) * (c3 * i1 + c4 * r1) * pow(sw, 2) +
                        pow(c1, 2) * (i2 - 8 * pow(cw, 2) * i2 * pow(sw, 2)) +
                        pow(c2, 2) * (i2 - 8 * pow(cw, 2) * i2 * pow(sw, 2)))) /
                        sqrt(pow(pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2) +
                                         pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2), 2) + 16 * pow(cw, 2) *
                                (pow(c2 * i1 + c4 * i2 + c1 * r1 + c3 * r2, 2) +
                                        pow(c1 * i1 + c3 * i2 - c2 * r1 - c4 * r2, 2) -
                                        (pow(c1, 2) + pow(c2, 2) + pow(c3, 2) + pow(c4, 2)) *
                                                (pow(i1, 2) + pow(i2, 2) + pow(r1, 2) + pow(r2, 2))) *
                                pow(sw, 2)))) / 8.);
    }

    std::vector<std::tuple<double, double>> ret_val(4);

    for (size_t i = 0; i < 4; i++) {
        ret_val[i] = std::make_tuple(masses[i], deriv[i]);
    }

    return ret_val;
}

std::vector<std::tuple<double, double>> gauge_squared_masses_deriv(
        const Fields<double> &fields, const Parameters<double> &params, int fld1, int fld2) {
    auto masses = gauge_squared_masses(fields, params);
    std::vector<double> deriv(4);

    double r1 = fields.r1;
    double r2 = fields.r2;
    double c1 = fields.c1;

    double sw = params.gp / sqrt(pow(params.gp, 2) + pow(params.g, 2));
    double cw = params.g / sqrt(pow(params.gp, 2) + pow(params.g, 2));

    if (fld1 == 1) {
        if (fld2 == 1) {
            deriv[0] = (pow(params.g, 2) / 2.);
            deriv[1] = (pow(params.g, 2) / 2.);
            deriv[2] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) * (2 -
                    (4 * pow(r1, 2) * pow(pow(c1, 2) + pow(r1, 2) + pow(r2, 2), 2)) /
                            pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) *
                                    (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5) +
                    (2 * (pow(c1, 2) + 3 * pow(r1, 2) + pow(r2, 2))) /
                            sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) * (pow(r1, 2) +
                                    pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))))) / 8.);
            deriv[3] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) * (2 +
                    (4 * pow(r1, 2) * pow(pow(c1, 2) + pow(r1, 2) + pow(r2, 2), 2)) /
                            pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                        2 * pow(c1, 2) *
                                                (pow(r1, 2) +
                                                        pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))),
                                1.5) -
                    (2 * (pow(c1, 2) + 3 * pow(r1, 2) + pow(r2, 2))) /
                            sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                         2 * pow(c1, 2) *
                                                 (pow(r1, 2) +
                                                         pow(r2, 2) *
                                                                 (1 - 8 * pow(cw, 2) * pow(sw, 2)))))) /
                    8.);

        } else if (fld2 == 2) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = ((4 * pow(c1, 2) * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 * r2 *
                    (pow(c1, 2) + pow(r1, 2) - pow(r2, 2)) * pow(sw, 2)) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                2 * pow(c1, 2) *
                                        (pow(r1, 2) +
                                                pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))),
                        1.5));
            deriv[3] = ((-4 * pow(c1, 2) * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 * r2 *
                    (pow(c1, 2) + pow(r1, 2) - pow(r2, 2)) * pow(sw, 2)) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                2 * pow(c1, 2) *
                                        (pow(r1, 2) +
                                                pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))),
                        1.5));

        } else if (fld2 == 3) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = ((4 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 * pow(r2, 2) *
                    (-pow(c1, 2) + pow(r1, 2) + pow(r2, 2)) * pow(sw, 2)) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                2 * pow(c1, 2) *
                                        (pow(r1, 2) +
                                                pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))),
                        1.5));
            deriv[3] = ((-4 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 * pow(r2, 2) *
                    (-pow(c1, 2) + pow(r1, 2) + pow(r2, 2)) * pow(sw, 2)) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                2 * pow(c1, 2) *
                                        (pow(r1, 2) +
                                                pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))),
                        1.5));

        } else if (fld2 == 4) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 5) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = ((2 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r2 * pow(sw, 2) *
                    (pow(c1, 4) - pow(r1, 4) + pow(r2, 4) +
                            2 * pow(c1, 2) * pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                2 * pow(c1, 2) *
                                        (pow(r1, 2) +
                                                pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))),
                        1.5));
            deriv[3] = ((-2 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r2 * pow(sw, 2) *
                    (pow(c1, 4) - pow(r1, 4) + pow(r2, 4) +
                            2 * pow(c1, 2) * pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                2 * pow(c1, 2) *
                                        (pow(r1, 2) +
                                                pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))),
                        1.5));

        } else if (fld2 == 6) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 7) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 8) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);
        }

    } else if (fld1 == 2) {
        if (fld2 == 1) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = ((4 * pow(c1, 2) * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 * r2 *
                    (pow(c1, 2) + pow(r1, 2) - pow(r2, 2)) * pow(sw, 2)) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                2 * pow(c1, 2) * (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5));
            deriv[3] = ((-4 * pow(c1, 2) * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 * r2 *
                    (pow(c1, 2) + pow(r1, 2) - pow(r2, 2)) * pow(sw, 2)) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) *
                            (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5));

        } else if (fld2 == 2) {
            deriv[0] = (pow(params.g, 2) / 2.);
            deriv[1] = (pow(params.g, 2) / 2.);
            deriv[2] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) * (2 - (4 * pow(r2, 2) *
                    pow(pow(r1, 2) + pow(r2, 2) + pow(c1, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)), 2)) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) * (pow(r1, 2) +
                            pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5) +
                    (2 * (pow(r1, 2) + 3 * pow(r2, 2) +
                            pow(c1, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))) /
                            sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                         2 * pow(c1, 2) * (pow(r1, 2) + pow(r2, 2) *
                                                 (1 - 8 * pow(cw, 2) * pow(sw, 2)))))) / 8.);
            deriv[3] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) * (2 + (4 * pow(r2, 2) *
                    pow(pow(r1, 2) + pow(r2, 2) + pow(c1, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)), 2)) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) * (pow(r1, 2) +
                            pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5) -
                    (2 * (pow(r1, 2) + 3 * pow(r2, 2) + pow(c1, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))) /
                            sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) * (pow(r1, 2) +
                                    pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))))) / 8.);

        } else if (fld2 == 3) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = ((-4 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r2 * pow(sw, 2) *
                    (pow(r1, 2) * (pow(r1, 2) + pow(r2, 2)) + pow(c1, 2) * (pow(r1, 2) +
                            2 * pow(r2, 2) * (1 - 4 * pow(cw, 2) * pow(sw, 2))))) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) *
                            (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5));
            deriv[3] = ((4 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r2 * pow(sw, 2) *
                    (pow(r1, 2) * (pow(r1, 2) + pow(r2, 2)) + pow(c1, 2) * (pow(r1, 2) +
                            2 * pow(r2, 2) * (1 - 4 * pow(cw, 2) * pow(sw, 2))))) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) *
                            (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5));

        } else if (fld2 == 4) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 5) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = ((2 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 *
                    (pow(c1, 4) + 2 * pow(c1, 2) * pow(r1, 2) + pow(r1, 4) -
                            pow(r2, 4)) * pow(sw, 2)) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) *
                            (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5));
            deriv[3] = ((-2 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 *
                    (pow(c1, 4) + 2 * pow(c1, 2) * pow(r1, 2) + pow(r1, 4) - pow(r2, 4)) *
                    pow(sw, 2)) / pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) *
                    (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5));

        } else if (fld2 == 6) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 7) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 8) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);
        }

    } else if (fld1 == 3) {
        if (fld2 == 1) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = ((4 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 * pow(r2, 2) *
                    (-pow(c1, 2) + pow(r1, 2) + pow(r2, 2)) * pow(sw, 2)) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                2 * pow(c1, 2) * (pow(r1, 2) +
                                        pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5));
            deriv[3] = ((-4 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 * pow(r2, 2) *
                    (-pow(c1, 2) + pow(r1, 2) + pow(r2, 2)) * pow(sw, 2)) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                2 * pow(c1, 2) * (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5));

        } else if (fld2 == 2) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = ((-4 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r2 * pow(sw, 2) *
                    (pow(r1, 2) * (pow(r1, 2) + pow(r2, 2)) + pow(c1, 2) * (pow(r1, 2) +
                            2 * pow(r2, 2) * (1 - 4 * pow(cw, 2) * pow(sw, 2))))) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) * (pow(r1, 2) +
                            pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5));
            deriv[3] = ((4 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r2 * pow(sw, 2) *
                    (pow(r1, 2) * (pow(r1, 2) + pow(r2, 2)) + pow(c1, 2) * (pow(r1, 2) +
                            2 * pow(r2, 2) * (1 - 4 * pow(cw, 2) * pow(sw, 2))))) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) * (pow(r1, 2) +
                            pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5));

        } else if (fld2 == 3) {
            deriv[0] = (pow(params.g, 2) / 2.);
            deriv[1] = (pow(params.g, 2) / 2.);
            deriv[2] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) * (2 - (4 * pow(c1, 2) *
                    pow(pow(c1, 2) + pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)), 2)) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) * (pow(r1, 2) +
                            pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5) +
                    (2 * (3 * pow(c1, 2) + pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))) /
                            sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) *
                                    (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))))) / 8.);
            deriv[3] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) * (2 + (4 * pow(c1, 2) *
                    pow(pow(c1, 2) + pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)), 2)) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) *
                            (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5) -
                    (2 * (3 * pow(c1, 2) + pow(r1, 2) +
                            pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))) /
                            sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                         2 * pow(c1, 2) * (pow(r1, 2) + pow(r2, 2) *
                                                 (1 - 8 * pow(cw, 2) * pow(sw, 2)))))) / 8.);

        } else if (fld2 == 4) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 5) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = ((2 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 * r2 *
                    (-pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2)) * pow(sw, 2)) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) *
                            (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5));
            deriv[3] = ((-2 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 * r2 *
                    (-pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2)) * pow(sw, 2)) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                2 * pow(c1, 2) * (pow(r1, 2) +
                                        pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5));

        } else if (fld2 == 6) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 7) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 8) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);
        }

    } else if (fld1 == 4) {
        if (fld2 == 1) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 2) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 3) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 4) {
            deriv[0] = (pow(params.g, 2) / 2.);
            deriv[1] = (pow(params.g, 2) / 2.);
            deriv[2] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) *
                    (2 + (2 * (pow(c1, 2) + pow(r1, 2) +
                            pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))) /
                            sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) * (pow(r1, 2) +
                                    pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))))) / 8.);
            deriv[3] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) *
                    (2 - (2 * (pow(c1, 2) + pow(r1, 2) +
                            pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))) /
                            sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) *
                                    (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))))) / 8.);

        } else if (fld2 == 5) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 6) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = ((2 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 * r2 * pow(sw, 2)) /
                    sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) *
                            (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))));
            deriv[3] = ((-2 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 * r2 * pow(sw, 2)) /
                    sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) *
                            (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))));

        } else if (fld2 == 7) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 8) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);
        }

    } else if (fld1 == 5) {
        if (fld2 == 1) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = ((2 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r2 * pow(sw, 2) *
                    (pow(c1, 4) - pow(r1, 4) + pow(r2, 4) +
                            2 * pow(c1, 2) * pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) *
                            (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5));
            deriv[3] = ((-2 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r2 * pow(sw, 2) *
                    (pow(c1, 4) - pow(r1, 4) + pow(r2, 4) +
                            2 * pow(c1, 2) * pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) *
                            (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5));

        } else if (fld2 == 2) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = ((2 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 *
                    (pow(c1, 4) + 2 * pow(c1, 2) * pow(r1, 2) + pow(r1, 4) - pow(r2, 4)) *
                    pow(sw, 2)) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) * (pow(r1, 2) +
                            pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5));
            deriv[3] = ((-2 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 *
                    (pow(c1, 4) + 2 * pow(c1, 2) * pow(r1, 2) + pow(r1, 4) - pow(r2, 4)) *
                    pow(sw, 2)) / pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                              2 * pow(c1, 2) * (pow(r1, 2) +
                                                      pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5));

        } else if (fld2 == 3) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = ((2 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 * r2 *
                    (-pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2)) * pow(sw, 2)) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                2 * pow(c1, 2) * (pow(r1, 2) +
                                        pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5));
            deriv[3] = ((-2 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 * r2 *
                    (-pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2)) * pow(sw, 2)) /
                    pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                2 * pow(c1, 2) *
                                        (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5));

        } else if (fld2 == 4) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 5) {
            deriv[0] = (pow(params.g, 2) / 2.);
            deriv[1] = (pow(params.g, 2) / 2.);
            deriv[2] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) *
                    (2 - (256 * pow(c1, 2) * pow(cw, 4) * pow(r1, 2) * pow(r2, 2) * pow(sw, 4)) /
                            pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) *
                                    (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5) +
                            (2 * (pow(c1, 2) + pow(r2, 2) +
                                    pow(r1, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))) /
                                    sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) *
                                            (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))))) / 8.);
            deriv[3] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) *
                    (2 + (256 * pow(c1, 2) * pow(cw, 4) * pow(r1, 2) * pow(r2, 2) * pow(sw, 4)) /
                            pow(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) *
                                    (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2))), 1.5) -
                            (2 * (pow(c1, 2) + pow(r2, 2) +
                                    pow(r1, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))) /
                                    sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) *
                                            (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))))) / 8.);

        } else if (fld2 == 6) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 7) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 8) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);
        }

    } else if (fld1 == 6) {
        if (fld2 == 1) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 2) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 3) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 4) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = ((2 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 * r2 * pow(sw, 2)) /
                    sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                 2 * pow(c1, 2) *
                                         (pow(r1, 2) +
                                                 pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))));
            deriv[3] = ((-2 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 * r2 *
                    pow(sw, 2)) /
                    sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                 2 * pow(c1, 2) *
                                         (pow(r1, 2) +
                                                 pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))));

        } else if (fld2 == 5) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 6) {
            deriv[0] = (pow(params.g, 2) / 2.);
            deriv[1] = (pow(params.g, 2) / 2.);
            deriv[2] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) *
                    (2 + (2 * (pow(c1, 2) + pow(r2, 2) +
                            pow(r1, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))) /
                            sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                         2 * pow(c1, 2) *
                                                 (pow(r1, 2) +
                                                         pow(r2, 2) *
                                                                 (1 - 8 * pow(cw, 2) * pow(sw, 2)))))) / 8.);
            deriv[3] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) *
                    (2 - (2 * (pow(c1, 2) + pow(r2, 2) +
                            pow(r1, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))) /
                            sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                         2 * pow(c1, 2) * (pow(r1, 2) + pow(r2, 2) *
                                                 (1 - 8 * pow(cw, 2) * pow(sw, 2)))))) / 8.);

        } else if (fld2 == 7) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = ((-2 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r2 *
                    pow(sw, 2)) /
                    sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                 2 * pow(c1, 2) *
                                         (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))));
            deriv[3] = ((2 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r2 * pow(sw, 2)) /
                    sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                 2 * pow(c1, 2) *
                                         (pow(r1, 2) +
                                                 pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))));

        } else if (fld2 == 8) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = ((2 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 * pow(sw, 2)) /
                    sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                 2 * pow(c1, 2) *
                                         (pow(r1, 2) +
                                                 pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))));
            deriv[3] = ((-2 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 *
                    pow(sw, 2)) /
                    sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                 2 * pow(c1, 2) *
                                         (pow(r1, 2) +
                                                 pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))));
        }

    } else if (fld1 == 7) {
        if (fld2 == 1) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 2) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 3) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 4) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 5) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 6) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = ((-2 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r2 *
                    pow(sw, 2)) /
                    sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                 2 * pow(c1, 2) *
                                         (pow(r1, 2) +
                                                 pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))));
            deriv[3] = ((2 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r2 * pow(sw, 2)) /
                    sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                 2 * pow(c1, 2) *
                                         (pow(r1, 2) +
                                                 pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))));

        } else if (fld2 == 7) {
            deriv[0] = (pow(params.g, 2) / 2.);
            deriv[1] = (pow(params.g, 2) / 2.);
            deriv[2] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) * (2 +
                    (2 * (pow(c1, 2) + pow(r1, 2) + pow(r2, 2))) /
                            sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                         2 * pow(c1, 2) * (pow(r1, 2) + pow(r2, 2) *
                                                 (1 - 8 * pow(cw, 2) * pow(sw, 2)))))) / 8.);
            deriv[3] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) * (2 -
                    (2 * (pow(c1, 2) + pow(r1, 2) + pow(r2, 2))) /
                            sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) + 2 * pow(c1, 2) *
                                    (pow(r1, 2) + pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))))) / 8.);

        } else if (fld2 == 8) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);
        }

    } else if (fld1 == 8) {
        if (fld2 == 1) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 2) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 3) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 4) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 5) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 6) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = ((2 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 * pow(sw, 2)) /
                    sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                 2 * pow(c1, 2) *
                                         (pow(r1, 2) +
                                                 pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))));
            deriv[3] = ((-2 * c1 * pow(cw, 2) * (pow(params.gp, 2) + pow(params.g, 2)) * r1 *
                    pow(sw, 2)) /
                    sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                 2 * pow(c1, 2) *
                                         (pow(r1, 2) +
                                                 pow(r2, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))));

        } else if (fld2 == 7) {
            deriv[0] = (0);
            deriv[1] = (0);
            deriv[2] = (0);
            deriv[3] = (0);

        } else if (fld2 == 8) {
            deriv[0] = (pow(params.g, 2) / 2.);
            deriv[1] = (pow(params.g, 2) / 2.);
            deriv[2] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) *
                    (2 + (2 * (pow(r1, 2) + pow(r2, 2) +
                            pow(c1, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))) /
                            sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                         2 * pow(c1, 2) * (pow(r1, 2) + pow(r2, 2) *
                                                 (1 - 8 * pow(cw, 2) * pow(sw, 2)))))) / 8.);
            deriv[3] = (((1 + pow(params.gp, 2) / pow(params.g, 2)) * pow(params.g, 2) *
                    (2 - (2 * (pow(r1, 2) + pow(r2, 2) +
                            pow(c1, 2) * (1 - 8 * pow(cw, 2) * pow(sw, 2)))) /
                            sqrt(pow(c1, 4) + pow(pow(r1, 2) + pow(r2, 2), 2) +
                                         2 * pow(c1, 2) * (pow(r1, 2) + pow(r2, 2) *
                                                 (1 - 8 * pow(cw, 2) * pow(sw, 2)))))) / 8.);
        }
    }

    std::vector<std::tuple<double, double>> ret_val(4);

    for (size_t i = 0; i < 4; i++) {
        ret_val[i] = std::make_tuple(masses[i], deriv[i]);
    }

    return ret_val;
}

} // namespace thdm

#endif // THDM_GAUGE_MASSES_HPP
