//
// Created by Logan Morrison on 2019-05-20.
//

#ifndef THDM_VALIDATION_HPP
#define THDM_VALIDATION_HPP

#include "thdm/potentials.hpp"
#include "thdm/fields.hpp"
#include "thdm/parameters.hpp"
#include "thdm/vacuua.hpp"
#include "thdm/constants.hpp"

namespace thdm {

/**
 * Check that all the derivatives at the vacuum are near zero.
 * @param params THDM parameters
 * @param vac Vacuum to check
 * @return bool true if derivatives are small.
 */
bool verify_derivatives_zero(Parameters<double> &params, Vacuum<double> &vac) {
    bool isvalid;

    Fields<double> fields{};
    try {
        fields.set_fields(vac);
        double quad = 0.0;
        for (int i = 1; i <= 8; i++) {
            quad += pow(potential_eff_deriv(fields, params, i), 2);
        }
        isvalid = sqrt(quad) < DERIVATIVE_TOL;

    } catch (THDMException &e) {
        isvalid = false;
    }
    return isvalid;
}

/**
 * Check that there are the correct number of goldstone
 * bosons. For normal extrema, there should be three,
 * for CB, there should be four.
 * @param params THDM parameters
 * @param vac Vacuum to check
 * @param is_cb
 * @return
 */
bool verify_goldstones(Parameters<double> &params,
                       Vacuum<double> &vac, bool is_cb = false) {
    bool has_goldstones = true;

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
        has_goldstones = has_goldstones && (masses[0] < ZERO_MASS_TOL);
        has_goldstones = has_goldstones && (masses[1] < ZERO_MASS_TOL);
        has_goldstones = has_goldstones && (masses[2] < ZERO_MASS_TOL);

        if (is_cb) {
            has_goldstones = has_goldstones && (masses[3] < ZERO_MASS_TOL);
        }
    } catch (THDMException &e) {
        has_goldstones = false;
    }

    return has_goldstones;
}

/**
 * Check that there are the correct number of goldstones and that the
 * derivatives of the effective potential are small.
 * @param params THDM parameters
 * @param vac Vacuum to check
 * @return
 */
bool is_vacuum_valid(Parameters<double> &params, Vacuum<double> &vac) {
    using std::abs;
    bool is_cb = (abs(vac.vevs[2]) > ZERO_CB_VEV_TOL);
    bool is_valid = verify_derivatives_zero(params, vac);
    return is_valid && verify_goldstones(params, vac, is_cb);
}
}


#endif //THDM_VALIDATION_HPP
