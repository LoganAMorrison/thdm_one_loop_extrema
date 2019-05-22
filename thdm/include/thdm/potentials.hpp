#ifndef POTENTIALS_HPP
#define POTENTIALS_HPP

#include "thdm/parameters.hpp"
#include "thdm/fields.hpp"
#include "thdm/scalar_masses.hpp"
#include "thdm/gauge_masses.hpp"
#include "thdm/fermion_masses.hpp"
#include <cmath>
#include <tuple>
#include <iostream>
#include <armadillo>
#include <assert.h>

namespace thdm {

/**
 * Compute the tree-level scalar potential for a given set of fields and
 * parameters.
 * @tparam T a double, Dual<double> or Dual<Dual<double>>
 * @param fields 2hdm fields
 * @param params 2hdm parameter
 * @return tree-level scalar potential
 */
template<class T>
T potential_tree(const Fields<T> &fields, const Parameters<T> &params) {
    T r1 = fields.r1;
    T r2 = fields.r2;
    T c1 = fields.c1;
    T c2 = fields.c2;
    T c3 = fields.c3;
    T c4 = fields.c4;
    T i1 = fields.i1;
    T i2 = fields.i2;

    T m112 = params.m112;
    T m122 = params.m122;
    T m222 = params.m222;
    T lam1 = params.lam1;
    T lam2 = params.lam2;
    T lam3 = params.lam3;
    T lam4 = params.lam4;
    T lam5 = params.lam5;

    return (pow(c1, 4) * lam1) / 8.0 + (pow(c1, 2) *
            pow(c2, 2) * lam1) / 4.0 +
            (pow(c2, 4) * lam1) / 8.0 + (pow(c1, 2) *
            pow(i1, 2) * lam1) / 4.0 +
            (pow(c2, 2) * pow(i1, 2) * lam1) / 4.0 +
            (pow(i1, 4) * lam1) / 8.0 +
            (pow(c3, 4) * lam2) / 8.0 + (pow(c3, 2) *
            pow(c4, 2) * lam2) / 4.0 +
            (pow(c4, 4) * lam2) / 8.0 + (pow(c3, 2) *
            pow(i2, 2) * lam2) / 4.0 +
            (pow(c4, 2) * pow(i2, 2) * lam2) / 4.0 +
            (pow(i2, 4) * lam2) / 8.0 +
            (pow(c1, 2) * pow(c3, 2) * lam3) / 4.0 +
            (pow(c2, 2) * pow(c3, 2) * lam3) / 4.0 +
            (pow(c1, 2) * pow(c4, 2) * lam3) / 4.0 +
            (pow(c2, 2) * pow(c4, 2) * lam3) / 4.0 +
            (pow(c3, 2) * pow(i1, 2) * lam3) / 4.0 +
            (pow(c4, 2) * pow(i1, 2) * lam3) / 4.0 +
            (pow(c1, 2) * pow(i2, 2) * lam3) / 4.0 +
            (pow(c2, 2) * pow(i2, 2) * lam3) / 4.0 +
            (pow(i1, 2) * pow(i2, 2) * lam3) / 4.0 +
            (pow(c1, 2) * pow(c3, 2) * lam4) / 4.0 +
            (pow(c2, 2) * pow(c3, 2) * lam4) / 4.0 +
            (pow(c1, 2) * pow(c4, 2) * lam4) / 4.0 +
            (pow(c2, 2) * pow(c4, 2) * lam4) / 4.0 +
            (c1 * c3 * i1 * i2 * lam4) / 2.0 +
            (c2 * c4 * i1 * i2 * lam4) / 2.0 +
            (pow(i1, 2) * pow(i2, 2) * lam4) / 4.0 +
            (pow(c1, 2) * pow(c3, 2) * lam5) / 4.0 -
            (pow(c2, 2) * pow(c3, 2) * lam5) / 4.0 +
            c1 * c2 * c3 * c4 * lam5 -
            (pow(c1, 2) * pow(c4, 2) * lam5) / 4.0 +
            (pow(c2, 2) * pow(c4, 2) * lam5) / 4.0 +
            (c1 * c3 * i1 * i2 * lam5) / 2.0 +
            (c2 * c4 * i1 * i2 * lam5) / 2.0 +
            (pow(i1, 2) * pow(i2, 2) * lam5) / 4.0 +
            (pow(c1, 2) * m112) / 2.0 + (pow(c2, 2) * m112) / 2.0 +
            (pow(i1, 2) * m112) / 2.0 -
            c1 * c3 * m122 - c2 * c4 * m122 - i1 * i2 * m122 +
            (pow(c3, 2) * m222) / 2.0 +
            (pow(c4, 2) * m222) / 2.0 + (pow(i2, 2) * m222) / 2.0 -
            (c2 * c3 * i2 * lam4 * r1) / 2.0 +
            (c1 * c4 * i2 * lam4 * r1) / 2.0 +
            (c2 * c3 * i2 * lam5 * r1) / 2.0 -
            (c1 * c4 * i2 * lam5 * r1) / 2.0 +
            (pow(c1, 2) * lam1 * pow(r1, 2)) / 4.0 +
            (pow(c2, 2) * lam1 * pow(r1, 2)) / 4.0 +
            (pow(i1, 2) * lam1 * pow(r1, 2)) / 4.0 +
            (pow(c3, 2) * lam3 * pow(r1, 2)) / 4.0 +
            (pow(c4, 2) * lam3 * pow(r1, 2)) / 4.0 +
            (pow(i2, 2) * lam3 * pow(r1, 2)) / 4.0 +
            (pow(i2, 2) * lam4 * pow(r1, 2)) / 4.0 -
            (pow(i2, 2) * lam5 * pow(r1, 2)) / 4.0 +
            (m112 * pow(r1, 2)) / 2.0 + (lam1 * pow(r1, 4)) / 8.0 +
            (c2 * c3 * i1 * lam4 * r2) / 2.0 -
            (c1 * c4 * i1 * lam4 * r2) / 2.0 -
            (c2 * c3 * i1 * lam5 * r2) / 2.0 +
            (c1 * c4 * i1 * lam5 * r2) / 2.0 +
            (c1 * c3 * lam4 * r1 * r2) / 2.0 +
            (c2 * c4 * lam4 * r1 * r2) / 2.0 +
            (c1 * c3 * lam5 * r1 * r2) / 2.0 +
            (c2 * c4 * lam5 * r1 * r2) / 2.0 +
            i1 * i2 * lam5 * r1 * r2 - m122 * r1 * r2 +
            (pow(c3, 2) * lam2 * pow(r2, 2)) / 4.0 +
            (pow(c4, 2) * lam2 * pow(r2, 2)) / 4.0 +
            (pow(i2, 2) * lam2 * pow(r2, 2)) / 4.0 +
            (pow(c1, 2) * lam3 * pow(r2, 2)) / 4.0 +
            (pow(c2, 2) * lam3 * pow(r2, 2)) / 4.0 +
            (pow(i1, 2) * lam3 * pow(r2, 2)) / 4.0 +
            (pow(i1, 2) * lam4 * pow(r2, 2)) / 4.0 -
            (pow(i1, 2) * lam5 * pow(r2, 2)) / 4.0 +
            (m222 * pow(r2, 2)) / 2.0 +
            (lam3 * pow(r1, 2) * pow(r2, 2)) / 4.0 +
            (lam4 * pow(r1, 2) * pow(r2, 2)) / 4.0 +
            (lam5 * pow(r1, 2) * pow(r2, 2)) / 4.0 +
            (lam2 * pow(r2, 4)) / 8.;
}

/**
 * Compute the first derivative of the tree-level scalar potential wrt a field
 * @tparam T a double, Dual<double> or Dual<Dual<double>>
 * @param fields 2HDM fields
 * @param params 2HDM parameters
 * @param fld index of field to take derivative of
 * @return derivative of the tree-level scalar potential
 */
template<class T>
T potential_tree_deriv(
        const Fields<T> &fields, const Parameters<T> &params, int fld) {
    if (fld < 1 || 8 < fld)
        throw THDMException(THDMExceptionCode::ScalarMassesInvalidFldIndex);

    T r1 = fields.r1;
    T r2 = fields.r2;
    T c1 = fields.c1;
    T c2 = fields.c2;
    T c3 = fields.c3;
    T c4 = fields.c4;
    T i1 = fields.i1;
    T i2 = fields.i2;

    T m112 = params.m112;
    T m122 = params.m122;
    T m222 = params.m222;
    T lam1 = params.lam1;
    T lam2 = params.lam2;
    T lam3 = params.lam3;
    T lam4 = params.lam4;
    T lam5 = params.lam5;


    if (fld == 1) {
        return -(c2 * c3 * i2 * lam4) / 2. + (c1 * c4 * i2 * lam4) / 2. +
                (c2 * c3 * i2 * lam5) / 2. -
                (c1 * c4 * i2 * lam5) / 2. + (pow(c1, 2) * lam1 * r1) / 2. +
                (pow(c2, 2) * lam1 * r1) / 2. + (pow(i1, 2) * lam1 * r1) / 2. +
                (pow(c3, 2) * lam3 * r1) / 2. +
                (pow(c4, 2) * lam3 * r1) / 2. +
                (pow(i2, 2) * lam3 * r1) / 2. + (pow(i2, 2) * lam4 * r1) / 2. -
                (pow(i2, 2) * lam5 * r1) / 2. +
                m112 * r1 + (lam1 * pow(r1, 3)) / 2. +
                (c1 * c3 * lam4 * r2) / 2. + (c2 * c4 * lam4 * r2) / 2. +
                (c1 * c3 * lam5 * r2) / 2. +
                (c2 * c4 * lam5 * r2) / 2. + i1 * i2 * lam5 * r2 - m122 * r2 +
                (lam3 * r1 * pow(r2, 2)) / 2. + (lam4 * r1 * pow(r2, 2)) / 2. +
                (lam5 * r1 * pow(r2, 2)) / 2.;
    } else if (fld == 2) {
        return (c2 * c3 * i1 * lam4) / 2. - (c1 * c4 * i1 * lam4) / 2. -
                (c2 * c3 * i1 * lam5) / 2. +
                (c1 * c4 * i1 * lam5) / 2. + (c1 * c3 * lam4 * r1) / 2. +
                (c2 * c4 * lam4 * r1) / 2. + (c1 * c3 * lam5 * r1) / 2. +
                (c2 * c4 * lam5 * r1) / 2. +
                i1 * i2 * lam5 * r1 - m122 * r1 + (pow(c3, 2) * lam2 * r2) / 2. +
                (pow(c4, 2) * lam2 * r2) / 2. + (pow(i2, 2) * lam2 * r2) / 2. +
                (pow(c1, 2) * lam3 * r2) / 2. +
                (pow(c2, 2) * lam3 * r2) / 2. +
                (pow(i1, 2) * lam3 * r2) / 2. + (pow(i1, 2) * lam4 * r2) / 2. -
                (pow(i1, 2) * lam5 * r2) / 2. +
                m222 * r2 + (lam3 * pow(r1, 2) * r2) / 2. +
                (lam4 * pow(r1, 2) * r2) / 2. + (lam5 * pow(r1, 2) * r2) / 2. +
                (lam2 * pow(r2, 3)) / 2.;
    } else if (fld == 3) {
        return (pow(c1, 3) * lam1) / 2. + (c1 * pow(c2, 2) * lam1) / 2. +
                (c1 * pow(i1, 2) * lam1) / 2. +
                (c1 * pow(c3, 2) * lam3) / 2. +
                (c1 * pow(c4, 2) * lam3) / 2. + (c1 * pow(i2, 2) * lam3) / 2. +
                (c1 * pow(c3, 2) * lam4) / 2. +
                (c1 * pow(c4, 2) * lam4) / 2. +
                (c3 * i1 * i2 * lam4) / 2. + (c1 * pow(c3, 2) * lam5) / 2. +
                c2 * c3 * c4 * lam5 -
                (c1 * pow(c4, 2) * lam5) / 2. + (c3 * i1 * i2 * lam5) / 2. +
                c1 * m112 -
                c3 * m122 + (c4 * i2 * lam4 * r1) / 2. -
                (c4 * i2 * lam5 * r1) / 2. + (c1 * lam1 * pow(r1, 2)) / 2. -
                (c4 * i1 * lam4 * r2) / 2. + (c4 * i1 * lam5 * r2) / 2. +
                (c3 * lam4 * r1 * r2) / 2. + (c3 * lam5 * r1 * r2) / 2. +
                (c1 * lam3 * pow(r2, 2)) / 2.;
    } else if (fld == 4) {
        return (pow(c1, 2) * c2 * lam1) / 2. + (pow(c2, 3) * lam1) / 2. +
                (c2 * pow(i1, 2) * lam1) / 2. +
                (c2 * pow(c3, 2) * lam3) / 2. +
                (c2 * pow(c4, 2) * lam3) / 2. + (c2 * pow(i2, 2) * lam3) / 2. +
                (c2 * pow(c3, 2) * lam4) / 2. +
                (c2 * pow(c4, 2) * lam4) / 2. +
                (c4 * i1 * i2 * lam4) / 2. - (c2 * pow(c3, 2) * lam5) / 2. +
                c1 * c3 * c4 * lam5 +
                (c2 * pow(c4, 2) * lam5) / 2. + (c4 * i1 * i2 * lam5) / 2. +
                c2 * m112 -
                c4 * m122 - (c3 * i2 * lam4 * r1) / 2. +
                (c3 * i2 * lam5 * r1) / 2. + (c2 * lam1 * pow(r1, 2)) / 2. +
                (c3 * i1 * lam4 * r2) / 2. - (c3 * i1 * lam5 * r2) / 2. +
                (c4 * lam4 * r1 * r2) / 2. + (c4 * lam5 * r1 * r2) / 2. +
                (c2 * lam3 * pow(r2, 2)) / 2.;
    } else if (fld == 5) {
        return (pow(c3, 3) * lam2) / 2. + (c3 * pow(c4, 2) * lam2) / 2. +
                (c3 * pow(i2, 2) * lam2) / 2. +
                (pow(c1, 2) * c3 * lam3) / 2. +
                (pow(c2, 2) * c3 * lam3) / 2. + (c3 * pow(i1, 2) * lam3) / 2. +
                (pow(c1, 2) * c3 * lam4) / 2. +
                (pow(c2, 2) * c3 * lam4) / 2. +
                (c1 * i1 * i2 * lam4) / 2. + (pow(c1, 2) * c3 * lam5) / 2. -
                (pow(c2, 2) * c3 * lam5) / 2. +
                c1 * c2 * c4 * lam5 + (c1 * i1 * i2 * lam5) / 2. - c1 * m122 +
                c3 * m222 - (c2 * i2 * lam4 * r1) / 2. +
                (c2 * i2 * lam5 * r1) / 2. + (c3 * lam3 * pow(r1, 2)) / 2. +
                (c2 * i1 * lam4 * r2) / 2. - (c2 * i1 * lam5 * r2) / 2. +
                (c1 * lam4 * r1 * r2) / 2. + (c1 * lam5 * r1 * r2) / 2. +
                (c3 * lam2 * pow(r2, 2)) / 2.;
    } else if (fld == 6) {
        return (pow(c3, 2) * c4 * lam2) / 2. + (pow(c4, 3) * lam2) / 2. +
                (c4 * pow(i2, 2) * lam2) / 2. +
                (pow(c1, 2) * c4 * lam3) / 2. +
                (pow(c2, 2) * c4 * lam3) / 2. + (c4 * pow(i1, 2) * lam3) / 2. +
                (pow(c1, 2) * c4 * lam4) / 2. +
                (pow(c2, 2) * c4 * lam4) / 2. +
                (c2 * i1 * i2 * lam4) / 2. + c1 * c2 * c3 * lam5 -
                (pow(c1, 2) * c4 * lam5) / 2. +
                (pow(c2, 2) * c4 * lam5) / 2. + (c2 * i1 * i2 * lam5) / 2. -
                c2 * m122 +
                c4 * m222 + (c1 * i2 * lam4 * r1) / 2. -
                (c1 * i2 * lam5 * r1) / 2. + (c4 * lam3 * pow(r1, 2)) / 2. -
                (c1 * i1 * lam4 * r2) / 2. + (c1 * i1 * lam5 * r2) / 2. +
                (c2 * lam4 * r1 * r2) / 2. + (c2 * lam5 * r1 * r2) / 2. +
                (c4 * lam2 * pow(r2, 2)) / 2.;
    } else if (fld == 7) {
        return (pow(c1, 2) * i1 * lam1) / 2. + (pow(c2, 2) * i1 * lam1) / 2. +
                (pow(i1, 3) * lam1) / 2. +
                (pow(c3, 2) * i1 * lam3) / 2. +
                (pow(c4, 2) * i1 * lam3) / 2. + (i1 * pow(i2, 2) * lam3) / 2. +
                (c1 * c3 * i2 * lam4) / 2. +
                (c2 * c4 * i2 * lam4) / 2. +
                (i1 * pow(i2, 2) * lam4) / 2. + (c1 * c3 * i2 * lam5) / 2. +
                (c2 * c4 * i2 * lam5) / 2. +
                (i1 * pow(i2, 2) * lam5) / 2. + i1 * m112 - i2 * m122 +
                (i1 * lam1 * pow(r1, 2)) / 2. + (c2 * c3 * lam4 * r2) / 2. -
                (c1 * c4 * lam4 * r2) / 2. -
                (c2 * c3 * lam5 * r2) / 2. + (c1 * c4 * lam5 * r2) / 2. +
                i2 * lam5 * r1 * r2 + (i1 * lam3 * pow(r2, 2)) / 2. +
                (i1 * lam4 * pow(r2, 2)) / 2. - (i1 * lam5 * pow(r2, 2)) / 2.;
    } else if (fld == 8) {
        return (pow(c3, 2) * i2 * lam2) / 2. + (pow(c4, 2) * i2 * lam2) / 2. +
                (pow(i2, 3) * lam2) / 2. +
                (pow(c1, 2) * i2 * lam3) / 2. +
                (pow(c2, 2) * i2 * lam3) / 2. + (pow(i1, 2) * i2 * lam3) / 2. +
                (c1 * c3 * i1 * lam4) / 2. +
                (c2 * c4 * i1 * lam4) / 2. +
                (pow(i1, 2) * i2 * lam4) / 2. + (c1 * c3 * i1 * lam5) / 2. +
                (c2 * c4 * i1 * lam5) / 2. +
                (pow(i1, 2) * i2 * lam5) / 2. - i1 * m122 + i2 * m222 -
                (c2 * c3 * lam4 * r1) / 2. + (c1 * c4 * lam4 * r1) / 2. +
                (c2 * c3 * lam5 * r1) / 2. -
                (c1 * c4 * lam5 * r1) / 2. + (i2 * lam3 * pow(r1, 2)) / 2. +
                (i2 * lam4 * pow(r1, 2)) / 2. - (i2 * lam5 * pow(r1, 2)) /
                2. + i1 * lam5 * r1 * r2 + (i2 * lam2 * pow(r2, 2)) / 2.;
    } else
        throw THDMException(THDMExceptionCode::ScalarMassesInvalidFldIndex);
}

/**
 * Compute the second derivative of the tree-level scalar potential wrt a
 * field and a parameter
 * @param fields 2HDM fields
 * @param fld index of field to take derivative of
 * @param par index of parameter to take derivative of
 * @return d^2V/dphi_i dp_j
 */
double potential_tree_deriv_fld_par(
        Fields<double> &fields, Parameters<double> &, int fld, int par) {

    if (fld < 1 || 8 < fld)
        throw THDMException(THDMExceptionCode::ScalarMassesInvalidFldIndex);
    if (par < 1 || 8 < par)
        throw THDMException(THDMExceptionCode::ScalarMassesInvalidParIndex);

    double r1 = fields.r1;
    double r2 = fields.r2;
    double c1 = fields.c1;
    double c2 = fields.c2;
    double c3 = fields.c3;
    double c4 = fields.c4;
    double i1 = fields.i1;
    double i2 = fields.i2;

    if (fld == 1) {
        if (par == 1) {
            return r1;
        } else if (par == 2) {
            return -r2;
        } else if (par == 3) {
            return 0;
        } else if (par == 4) {
            return (pow(c1, 2) * r1) / 2. + (pow(c2, 2) * r1) / 2. +
                    (pow(i1, 2) * r1) / 2. + pow(r1, 3) / 2.;
        } else if (par == 5) {
            return 0;
        } else if (par == 6) {
            return (pow(c3, 2) * r1) / 2. + (pow(c4, 2) * r1) / 2. +
                    (pow(i2, 2) * r1) / 2. + (r1 * pow(r2, 2)) / 2.;
        } else if (par == 7) {
            return -(c2 * c3 * i2) / 2. + (c1 * c4 * i2) / 2. +
                    (pow(i2, 2) * r1) / 2. + (c1 * c3 * r2) / 2. +
                    (c2 * c4 * r2) / 2. + (r1 * pow(r2, 2)) / 2.;
        } else if (par == 8) {
            return (c2 * c3 * i2) / 2. - (c1 * c4 * i2) / 2. -
                    (pow(i2, 2) * r1) / 2. + (c1 * c3 * r2) / 2. +
                    (c2 * c4 * r2) / 2. + i1 * i2 * r2 + (r1 * pow(r2, 2)) / 2.;
        }
    } else if (fld == 2) {
        if (par == 1) {
            return 0;
        } else if (par == 2) {
            return -r1;
        } else if (par == 3) {
            return r2;
        } else if (par == 4) {
            return 0;
        } else if (par == 5) {
            return (pow(c3, 2) * r2) / 2. + (pow(c4, 2) * r2) / 2. +
                    (pow(i2, 2) * r2) / 2. + pow(r2, 3) / 2.;
        } else if (par == 6) {
            return (pow(c1, 2) * r2) / 2. + (pow(c2, 2) * r2) / 2. +
                    (pow(i1, 2) * r2) / 2. + (pow(r1, 2) * r2) / 2.;
        } else if (par == 7) {
            return (c2 * c3 * i1) / 2. - (c1 * c4 * i1) / 2. +
                    (c1 * c3 * r1) / 2. + (c2 * c4 * r1) / 2. +
                    (pow(i1, 2) * r2) / 2. + (pow(r1, 2) * r2) / 2.;
        } else if (par == 8) {
            return -(c2 * c3 * i1) / 2. + (c1 * c4 * i1) / 2. +
                    (c1 * c3 * r1) / 2. + (c2 * c4 * r1) / 2. +
                    i1 * i2 * r1 - (pow(i1, 2) * r2) / 2. +
                    (pow(r1, 2) * r2) / 2.;
        }
    } else if (fld == 3) {
        if (par == 1) {
            return c1;
        } else if (par == 2) {
            return -c3;
        } else if (par == 3) {
            return 0;
        } else if (par == 4) {
            return pow(c1, 3) / 2. + (c1 * pow(c2, 2)) / 2. +
                    (c1 * pow(i1, 2)) / 2. + (c1 * pow(r1, 2)) / 2.;
        } else if (par == 5) {
            return 0;
        } else if (par == 6) {
            return (c1 * pow(c3, 2)) / 2. + (c1 * pow(c4, 2)) / 2. +
                    (c1 * pow(i2, 2)) / 2. + (c1 * pow(r2, 2)) / 2.;
        } else if (par == 7) {
            return (c1 * pow(c3, 2)) / 2. + (c1 * pow(c4, 2)) / 2. +
                    (c3 * i1 * i2) / 2. + (c4 * i2 * r1) / 2. -
                    (c4 * i1 * r2) / 2. + (c3 * r1 * r2) / 2.;
        } else if (par == 8) {
            return (c1 * pow(c3, 2)) / 2. + c2 * c3 * c4 -
                    (c1 * pow(c4, 2)) / 2. + (c3 * i1 * i2) / 2. -
                    (c4 * i2 * r1) / 2. + (c4 * i1 * r2) / 2. +
                    (c3 * r1 * r2) / 2.;
        }
    } else if (fld == 4) {
        if (par == 1) {
            return c2;
        } else if (par == 2) {
            return -c4;
        } else if (par == 3) {
            return 0;
        } else if (par == 4) {
            return (pow(c1, 2) * c2) / 2. + pow(c2, 3) / 2. +
                    (c2 * pow(i1, 2)) / 2. + (c2 * pow(r1, 2)) / 2.;
        } else if (par == 5) {
            return 0;
        } else if (par == 6) {
            return (c2 * pow(c3, 2)) / 2. + (c2 * pow(c4, 2)) / 2. +
                    (c2 * pow(i2, 2)) / 2. + (c2 * pow(r2, 2)) / 2.;
        } else if (par == 7) {
            return (c2 * pow(c3, 2)) / 2. + (c2 * pow(c4, 2)) / 2. +
                    (c4 * i1 * i2) / 2. - (c3 * i2 * r1) / 2. +
                    (c3 * i1 * r2) / 2. + (c4 * r1 * r2) / 2.;
        } else if (par == 8) {
            return -(c2 * pow(c3, 2)) / 2. + c1 * c3 * c4 +
                    (c2 * pow(c4, 2)) / 2. + (c4 * i1 * i2) / 2. +
                    (c3 * i2 * r1) / 2. - (c3 * i1 * r2) / 2. +
                    (c4 * r1 * r2) / 2.;
        }
    } else if (fld == 5) {
        if (par == 1) {
            return 0;
        } else if (par == 2) {
            return -c1;
        } else if (par == 3) {
            return c3;
        } else if (par == 4) {
            return 0;
        } else if (par == 5) {
            return pow(c3, 3) / 2. + (c3 * pow(c4, 2)) / 2. +
                    (c3 * pow(i2, 2)) / 2. + (c3 * pow(r2, 2)) / 2.;
        } else if (par == 6) {
            return (pow(c1, 2) * c3) / 2. + (pow(c2, 2) * c3) / 2. +
                    (c3 * pow(i1, 2)) / 2. + (c3 * pow(r1, 2)) / 2.;
        } else if (par == 7) {
            return (pow(c1, 2) * c3) / 2. + (pow(c2, 2) * c3) / 2. +
                    (c1 * i1 * i2) / 2. - (c2 * i2 * r1) / 2. +
                    (c2 * i1 * r2) / 2. + (c1 * r1 * r2) / 2.;
        } else if (par == 8) {
            return (pow(c1, 2) * c3) / 2. - (pow(c2, 2) * c3) / 2. +
                    c1 * c2 * c4 + (c1 * i1 * i2) / 2. +
                    (c2 * i2 * r1) / 2. - (c2 * i1 * r2) / 2. +
                    (c1 * r1 * r2) / 2.;
        }
    } else if (fld == 6) {
        if (par == 1) {
            return 0;
        } else if (par == 2) {
            return -c2;
        } else if (par == 3) {
            return c4;
        } else if (par == 4) {
            return 0;
        } else if (par == 5) {
            return (pow(c3, 2) * c4) / 2. + pow(c4, 3) / 2. +
                    (c4 * pow(i2, 2)) / 2. + (c4 * pow(r2, 2)) / 2.;
        } else if (par == 6) {
            return (pow(c1, 2) * c4) / 2. + (pow(c2, 2) * c4) / 2. +
                    (c4 * pow(i1, 2)) / 2. + (c4 * pow(r1, 2)) / 2.;
        } else if (par == 7) {
            return (pow(c1, 2) * c4) / 2. + (pow(c2, 2) * c4) / 2. +
                    (c2 * i1 * i2) / 2. + (c1 * i2 * r1) / 2. -
                    (c1 * i1 * r2) / 2. + (c2 * r1 * r2) / 2.;
        } else if (par == 8) {
            return c1 * c2 * c3 - (pow(c1, 2) * c4) / 2. +
                    (pow(c2, 2) * c4) / 2. + (c2 * i1 * i2) / 2. -
                    (c1 * i2 * r1) / 2. + (c1 * i1 * r2) / 2. +
                    (c2 * r1 * r2) / 2.;
        }
    } else if (fld == 7) {
        if (par == 1) {
            return i1;
        } else if (par == 2) {
            return -i2;
        } else if (par == 3) {
            return 0;
        } else if (par == 4) {
            return (pow(c1, 2) * i1) / 2. + (pow(c2, 2) * i1) / 2. +
                    pow(i1, 3) / 2. + (i1 * pow(r1, 2)) / 2.;
        } else if (par == 5) {
            return 0;
        } else if (par == 6) {
            return (pow(c3, 2) * i1) / 2. + (pow(c4, 2) * i1) / 2. +
                    (i1 * pow(i2, 2)) / 2. + (i1 * pow(r2, 2)) / 2.;
        } else if (par == 7) {
            return (c1 * c3 * i2) / 2. + (c2 * c4 * i2) / 2. +
                    (i1 * pow(i2, 2)) / 2. + (c2 * c3 * r2) / 2. -
                    (c1 * c4 * r2) / 2. + (i1 * pow(r2, 2)) / 2.;
        } else if (par == 8) {
            return (c1 * c3 * i2) / 2. + (c2 * c4 * i2) / 2. +
                    (i1 * pow(i2, 2)) / 2. - (c2 * c3 * r2) / 2. +
                    (c1 * c4 * r2) / 2. + i2 * r1 * r2 -
                    (i1 * pow(r2, 2)) / 2.;
        }
    } else if (fld == 8) {
        if (par == 1) {
            return 0;
        } else if (par == 2) {
            return -i1;
        } else if (par == 3) {
            return i2;
        } else if (par == 4) {
            return 0;
        } else if (par == 5) {
            return (pow(c3, 2) * i2) / 2. + (pow(c4, 2) * i2) / 2. +
                    pow(i2, 3) / 2. + (i2 * pow(r2, 2)) / 2.;
        } else if (par == 6) {
            return (pow(c1, 2) * i2) / 2. + (pow(c2, 2) * i2) / 2. +
                    (pow(i1, 2) * i2) / 2. + (i2 * pow(r1, 2)) / 2.;
        } else if (par == 7) {
            return (c1 * c3 * i1) / 2. + (c2 * c4 * i1) / 2. +
                    (pow(i1, 2) * i2) / 2. - (c2 * c3 * r1) / 2. +
                    (c1 * c4 * r1) / 2. + (i2 * pow(r1, 2)) / 2.;
        } else if (par == 8) {
            return (c1 * c3 * i1) / 2. + (c2 * c4 * i1) / 2. +
                    (pow(i1, 2) * i2) / 2. + (c2 * c3 * r1) / 2. -
                    (c1 * c4 * r1) / 2. - (i2 * pow(r1, 2)) / 2. +
                    i1 * r1 * r2;
        }
    }
    return 0;
}

/**
 * Compute the one-loop correction to the scalar potential.
 * @param fields 2HDM fields
 * @param params 2HDM parameters
 * @return one-loop correction to the scalar potential
 */
double potential_one_loop(
        Fields<double> &fields, Parameters<double> &params) {
    double loop = 0.0;
    double mu2 = pow(params.mu, 2);
    // Scalar contributions
    auto scalar_masses = scalar_squared_masses(fields, params);
    for (auto lam : scalar_masses) {
        if (fabs(lam) > 1e-8)
            loop += pow(lam, 2) * (log(fabs(lam / mu2)) - 1.5);
    }
    #ifdef SCALAR_ONLY
    return loop / (64.0 * pow(M_PI, 2));
    #endif
    // Gauge contributions
    auto gauge_masses = gauge_squared_masses(fields, params);
    for (auto lam : gauge_masses) {
        if (lam != 0) {
            loop += 3.0 * pow(lam, 2) * (log(fabs(lam / mu2)) - 2.5);
        }
    }
    // Top contribution
    double top_mass = top_mass_squared(fields, params);

    if (top_mass != 0) {
        loop -= 6.0 * pow(top_mass, 2) * (log(fabs(top_mass) / mu2) - 1.5);
    }

    return loop / (64.0 * pow(M_PI, 2));
}

/**
 * Compute the effective scalar potential to one-loop order.
 * @param fields 2HDM fields
 * @param params 2HDM parameters
 * @return effective scalar potential to one-loop order
 */
double potential_eff(Fields<double> &fields, Parameters<double> &params) {
    return potential_one_loop(fields, params) + potential_tree(fields, params);
}

/**
 * Compute the first derivative wrt a field of the one-loop correction to the
 * scalar potential
 * @param fields 2HDM fields
 * @param params 2HDM parameters
 * @param fld index of field to take derivative of
 * @return derivative wrt a field of the one-loop correction to the scalar
 * potential
 */
double potential_one_loop_deriv(
        Fields<double> &fields, Parameters<double> &params, int fld) {
    double mu2 = pow(params.mu, 2);

    double loop = 0.0;

    // Sum of scalar contributions
    auto scalar_masses = scalar_squared_masses_deriv_fld(fields, params, fld);
    for (auto tup : scalar_masses) {
        double lam = std::get<0>(tup);
        double dlam = std::get<1>(tup);

        if (fabs(lam) > 1e-8)
            loop += lam * dlam * (log(fabs(lam) / mu2) - 1);
    }
    #ifdef SCALAR_ONLY
    return loop / (32.0 * pow(M_PI, 2));
    #endif
    // Gauge contributions
    auto gauge_masses = gauge_squared_masses_deriv(fields, params, fld);
    for (auto tup : gauge_masses) {
        double lam = std::get<0>(tup);
        double dlam = std::get<1>(tup);

        if (fabs(lam) > 1e-8)
            loop += 3.0 * lam * dlam * (log(fabs(lam) / mu2) - 2);
    }
    // Top contribution
    double mtop2 = top_mass_squared(fields, params);
    double mtop2_deriv = top_mass_squared_deriv(fields, params, fld);
    // Factor of 3 for colors and 2 for spins
    if (mtop2 != 0) {
        loop -= 6.0 * mtop2 * mtop2_deriv * (log(fabs(mtop2) / mu2) - 1);
    }

    return loop / (32.0 * pow(M_PI, 2));
}

/**
 * Compute the second derivative wrt a field of the one-loop correction to the
 * scalar potential
 * @param fields 2HDM fields
 * @param params 2HDM parameters
 * @param fld1 index of first field to take derivative of
 * @param fld2 index of second field to take derivative of
 * @return second derivative wrt a field of the one-loop correction to the
 * scalar potential
 */
double potential_one_loop_deriv(Fields<double> &fields,
                                Parameters<double> &params,
                                int fld1, int fld2) {
    double mu2 = pow(params.mu, 2);

    double loop = 0.0;

    // Sum of scalar contributions
    // Seems like we need to use determinant method for this case when
    // fld1 == fld2 and fields[fld1] == 0
    auto scalar_masses = scalar_squared_masses_deriv_fld(fields, params, fld1, fld2);
    for (auto tup: scalar_masses) {
        double lam = std::get<0>(tup);
        double dlam1 = std::get<1>(tup);
        double dlam2 = std::get<2>(tup);
        double d2lam = std::get<3>(tup);

        loop += lam * d2lam * (log(fabs(lam) / mu2) - 1);
        loop += dlam1 * dlam2 * log(fabs(lam) / mu2);
    }
    #ifdef SCALAR_ONLY
    return loop / (32.0 * pow(M_PI, 2));
    #endif
    // Gauge contributions
    auto gauge_masses1 = gauge_squared_masses_deriv(fields, params, fld1);
    auto gauge_masses2 = gauge_squared_masses_deriv(fields, params, fld2);
    auto gauge_masses3 = gauge_squared_masses_deriv(fields, params, fld1, fld2);
    for (int i = 0; i < 4; i++) {
        double lam = std::get<0>(gauge_masses1[i]);
        double dlam1 = std::get<1>(gauge_masses1[i]);
        double dlam2 = std::get<1>(gauge_masses2[i]);
        double d2lam = std::get<1>(gauge_masses3[i]);

        if (lam != 0.0) {
            loop += 3.0 * lam * d2lam * (log(fabs(lam) / mu2) - 2);
            loop += 3.0 * dlam1 * dlam2 * (log(fabs(lam) / mu2) - 1);
        }
    }
    // Top contribution
    double mtop2 = top_mass_squared(fields, params);
    double mtop2_deriv1 = top_mass_squared_deriv(fields, params, fld1);
    double mtop2_deriv2 = top_mass_squared_deriv(fields, params, fld2);
    double mtop2_deriv3 = top_mass_squared_deriv(fields, params, fld1, fld2);
    // Factor of 3 for colors and 2 for spins
    if (mtop2 != 0) {
        loop -= 6.0 * mtop2 * mtop2_deriv3 * (log(fabs(mtop2) / mu2) - 1);
        loop -= 6.0 * mtop2_deriv1 * mtop2_deriv2 * log(fabs(mtop2) / mu2);
    }

    return loop / (32.0 * pow(M_PI, 2));
}

/**
 * Compute the first derivative wrt a parameter of the one-loop correction to
 * the scalar potential
 * @param fields 2HDM fields
 * @param params 2HDM parameters
 * @param fld index of field to take derivative of
 * @param par index of parameter to take derivative of
 * @return first derivative wrt a parameter of the one-loop correction to the
 * scalar potential
 */
double potential_one_loop_deriv_fld_par(
        Fields<double> &fields, Parameters<double> &params,
        int fld, int par) {
    double mu2 = pow(params.mu, 2);

    double loop = 0.0;

    // Sum of scalar contributions
    auto tups = scalar_squared_masses_deriv_fld_par(fields, params, fld, par);
    for (auto tup: tups) {
        double lam = std::get<0>(tup);
        double dlam1 = std::get<1>(tup);
        double dlam2 = std::get<2>(tup);
        double d2lam = std::get<3>(tup);

        loop += lam * d2lam * (log(fabs(lam) / mu2) - 1);
        loop += dlam1 * dlam2 * log(fabs(lam) / mu2);
    }
    // No gauge contribution because gauge masses do not depend on parameters.

    return loop / (32.0 * pow(M_PI, 2));
}

/**
 * Compute the first derivative wrt a field of the effective scalar potential
 * @param fields 2HDM fields
 * @param params 2HDM parameters
 * @param fld index of field to take derivative of
 * @return first derivative wrt a field of the effective scalar potential
 */
double potential_eff_deriv(Fields<double> &fields, Parameters<double> &params,
                           int fld) {
    return potential_one_loop_deriv(fields, params, fld) +
            potential_tree_deriv(fields, params, fld);
}

/**
 * Compute the second derivative wrt fields of the effective scalar potential
 * @param fields 2HDM fields
 * @param params 2HDM parameters
 * @param fld1 index of the first field to take derivative of
 * @param fld2 index of the second field to take derivative of
 * @return second derivative wrt fields of the effective scalar potential
 */
double potential_eff_deriv(Fields<double> &fields, Parameters<double> &params,
                           int fld1, int fld2) {
    auto mat = scalar_squared_mass_matrix(fields, params);
    return potential_one_loop_deriv(fields, params, fld1, fld2) +
            mat[fld1 - 1][fld2 - 1];
}

/**
 * Compute the second derivative wrt a field and a parameter of the effective
 * scalar potential
 * @param fields 2HDM fields
 * @param params 2HDM parameters
 * @param fld index of the field to take derivative of
 * @param par index of the parameter to take derivative of
 * @return
 */
double potential_eff_deriv_fld_par(
        Fields<double> &fields, Parameters<double> &params,
        int fld, int par) {
    return (potential_one_loop_deriv_fld_par(fields, params, fld, par) +
            potential_tree_deriv_fld_par(fields, params, fld, par));
}

/**
 * Compute the hessian matrix for the effective potential
 * @param fields THDM fields
 * @param params THDM params
 * @return std::vector<std::vector<double>> hessian
 */
std::vector<std::vector<double>> potential_eff_hessian(
        Fields<double> &fields, Parameters<double> &params) {

    std::vector<std::vector<double>> hessian(8, std::vector<double>(8));
    auto mat = scalar_squared_mass_matrix(fields, params);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            hessian[i][j] =
                    potential_one_loop_deriv(fields, params, i + 1, j + 1) +
                            mat[i][j];
        }
    }
    return hessian;
}

/**
 * Compute the eigenvalues of the hessian of the effective potential.
 * @param fields THDM fields
 * @param params THDM parameters
 * @return std::vector<double> hessian_evals.
 */
std::vector<double> potential_eff_hessian_evals(
        Fields<double> &fields, Parameters<double> &params) {

    auto hessian = potential_eff_hessian(fields, params);
    return jacobi(hessian);
}
} // namespace thdm

#endif //POTENTIALS_HPP