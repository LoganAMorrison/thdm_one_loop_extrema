//
// Created by Logan Morrison on 2019-05-20.
//

#ifndef THDM_CONSTANTS_HPP
#define THDM_CONSTANTS_HPP

namespace thdm {

/* Top quark mass */
static constexpr double const M_TOP = 173.0;
/* A derivative is considered zero if less than this tolerance. */
static constexpr double const DERIVATIVE_TOL = 1e-8;
/* A mass is considered zero if less than this tolerance. */
static constexpr double const ZERO_MASS_TOL = 1e-7;
/* U(1) coupling constant */
static constexpr double const U1Y_COUP = 0.3497;
/* SU(2) coupling constant */
static constexpr double const SU2_COUP = 0.652954;
}

#endif //THDM_CONSTANTS_HPP
