#ifndef MASSES_HPP
#define MASSES_HPP

#include "thdm/dual.hpp"
#include "thdm/fields.hpp"
#include "thdm/jacobi.hpp"
#include "thdm/parameters.hpp"
#include "thdm/errors.hpp"
#include <cmath>
#include <exception>
#include <iostream>
#include <tuple>
#include <vector>
#include <assert.h>
#include <armadillo>

namespace thdm {


/**
 * Computes the scalar mass matrix for a given set of fields and
 * parameters.
 * @tparam T type of the fields and params: should be a double or dual.
 * @param fields Fields<T> containing 2hdm fields.
 * @param params Parameters<T> containing 2hdm parameters.
 * @return boost matrix<T> containing the scalar mass matrix.
 */
template<class T>
std::vector<std::vector<T>> scalar_squared_mass_matrix(
        Fields<T> &fields, Parameters<T> &params) {
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
    return std::vector<std::vector<T>>{
            {(c1 * c1 * lam1 + c2 * c2 * lam1 + i1 * i1 * lam1 + c3 * c3 * lam3 +
                    c4 * c4 * lam3 + i2 * i2 * lam3 + i2 * i2 * lam4 - i2 * i2 * lam5 +
                    2 * m112 + 3 * lam1 * r1 * r1 + lam3 * r2 * r2 + lam4 * r2 * r2 +
                    lam5 * r2 * r2) /
                    2.,
                    (2 * i1 * i2 * lam5 + c1 * c3 * (lam4 + lam5) + c2 * c4 * (lam4 + lam5) -
                            2 * m122 + 2 * (lam3 + lam4 + lam5) * r1 * r2) /
                            2.,
                    (c4 * i2 * (lam4 - lam5) + 2 * c1 * lam1 * r1 +
                            c3 * (lam4 + lam5) * r2) /
                            2.,
                    -(c3 * i2 * (lam4 - lam5)) / 2. + c2 * lam1 * r1 +
                            (c4 * (lam4 + lam5) * r2) / 2.,
                    (c2 * i2 * (-lam4 + lam5) + 2 * c3 * lam3 * r1 +
                            c1 * (lam4 + lam5) * r2) /
                            2.,
                    (c1 * i2 * (lam4 - lam5) + 2 * c4 * lam3 * r1 +
                            c2 * (lam4 + lam5) * r2) /
                            2.,
                    i1 * lam1 * r1 + i2 * lam5 * r2,
                    (c1 * c4 * (lam4 - lam5) + c2 * c3 * (-lam4 + lam5) +
                            2 * i2 * (lam3 + lam4 - lam5) * r1 + 2 * i1 * lam5 * r2) /
                            2.},
            {(2 * i1 * i2 * lam5 + c1 * c3 * (lam4 + lam5) + c2 * c4 * (lam4 + lam5) -
                    2 * m122 + 2 * (lam3 + lam4 + lam5) * r1 * r2) /
                    2.,
                    (c3 * c3 * lam2 + c4 * c4 * lam2 + i2 * i2 * lam2 + c1 * c1 * lam3 +
                            c2 * c2 * lam3 + i1 * i1 * lam3 + i1 * i1 * lam4 - i1 * i1 * lam5 +
                            2 * m222 + lam3 * r1 * r1 + lam4 * r1 * r1 + lam5 * r1 * r1 +
                            3 * lam2 * r2 * r2) /
                            2.,
                    (c4 * i1 * (-lam4 + lam5) + c3 * (lam4 + lam5) * r1 +
                            2 * c1 * lam3 * r2) /
                            2.,
                    (c3 * i1 * (lam4 - lam5) + c4 * (lam4 + lam5) * r1 +
                            2 * c2 * lam3 * r2) /
                            2.,
                    (c2 * i1 * (lam4 - lam5) + c1 * (lam4 + lam5) * r1 +
                            2 * c3 * lam2 * r2) /
                            2.,
                    -(c1 * i1 * (lam4 - lam5)) / 2. + (c2 * (lam4 + lam5) * r1) / 2. +
                            c4 * lam2 * r2,
                    (c2 * c3 * (lam4 - lam5) + c1 * c4 * (-lam4 + lam5) +
                            2 * i2 * lam5 * r1 + 2 * i1 * (lam3 + lam4 - lam5) * r2) /
                            2.,
                    i1 * lam5 * r1 + i2 * lam2 * r2},
            {(c4 * i2 * (lam4 - lam5) + 2 * c1 * lam1 * r1 +
                    c3 * (lam4 + lam5) * r2) /
                    2.,
                    (c4 * i1 * (-lam4 + lam5) + c3 * (lam4 + lam5) * r1 +
                            2 * c1 * lam3 * r2) /
                            2.,
                    (3 * c1 * c1 * lam1 + c2 * c2 * lam1 + i1 * i1 * lam1 + c3 * c3 * lam3 +
                            c4 * c4 * lam3 + i2 * i2 * lam3 + c3 * c3 * lam4 + c4 * c4 * lam4 +
                            c3 * c3 * lam5 - c4 * c4 * lam5 + 2 * m112 + lam1 * r1 * r1 +
                            lam3 * r2 * r2) /
                            2.,
                    c1 * c2 * lam1 + c3 * c4 * lam5,
                    (2 * c2 * c4 * lam5 + i1 * i2 * (lam4 + lam5) +
                            2 * c1 * c3 * (lam3 + lam4 + lam5) - 2 * m122 + lam4 * r1 * r2 +
                            lam5 * r1 * r2) /
                            2.,
                    c1 * c4 * (lam3 + lam4 - lam5) + c2 * c3 * lam5 +
                            ((lam4 - lam5) * (i2 * r1 - i1 * r2)) / 2.,
                    (2 * c1 * i1 * lam1 + c3 * i2 * (lam4 + lam5) +
                            c4 * (-lam4 + lam5) * r2) /
                            2.,
                    c1 * i2 * lam3 +
                            (c3 * i1 * (lam4 + lam5) + c4 * (lam4 - lam5) * r1) / 2.},
            {-(c3 * i2 * (lam4 - lam5)) / 2. + c2 * lam1 * r1 +
                    (c4 * (lam4 + lam5) * r2) / 2.,
                    (c3 * i1 * (lam4 - lam5) + c4 * (lam4 + lam5) * r1 +
                            2 * c2 * lam3 * r2) /
                            2.,
                    c1 * c2 * lam1 + c3 * c4 * lam5,
                    (c1 * c1 * lam1 + 3 * c2 * c2 * lam1 + i1 * i1 * lam1 + c4 * c4 * lam3 +
                            i2 * i2 * lam3 + c4 * c4 * lam4 + c3 * c3 * (lam3 + lam4 - lam5) +
                            c4 * c4 * lam5 + 2 * m112 + lam1 * r1 * r1 + lam3 * r2 * r2) /
                            2.,
                    c2 * c3 * (lam3 + lam4 - lam5) + c1 * c4 * lam5 -
                            ((lam4 - lam5) * (i2 * r1 - i1 * r2)) / 2.,
                    (2 * c1 * c3 * lam5 + i1 * i2 * (lam4 + lam5) +
                            2 * c2 * c4 * (lam3 + lam4 + lam5) - 2 * m122 + lam4 * r1 * r2 +
                            lam5 * r1 * r2) /
                            2.,
                    (2 * c2 * i1 * lam1 + c4 * i2 * (lam4 + lam5) +
                            c3 * (lam4 - lam5) * r2) /
                            2.,
                    c2 * i2 * lam3 +
                            (c4 * i1 * (lam4 + lam5) + c3 * (-lam4 + lam5) * r1) / 2.},
            {(c2 * i2 * (-lam4 + lam5) + 2 * c3 * lam3 * r1 +
                    c1 * (lam4 + lam5) * r2) /
                    2.,
                    (c2 * i1 * (lam4 - lam5) + c1 * (lam4 + lam5) * r1 +
                            2 * c3 * lam2 * r2) /
                            2.,
                    (2 * c2 * c4 * lam5 + i1 * i2 * (lam4 + lam5) +
                            2 * c1 * c3 * (lam3 + lam4 + lam5) - 2 * m122 + lam4 * r1 * r2 +
                            lam5 * r1 * r2) /
                            2.,
                    c2 * c3 * (lam3 + lam4 - lam5) + c1 * c4 * lam5 -
                            ((lam4 - lam5) * (i2 * r1 - i1 * r2)) / 2.,
                    (3 * c3 * c3 * lam2 + c4 * c4 * lam2 + i2 * i2 * lam2 + c1 * c1 * lam3 +
                            c2 * c2 * lam3 + i1 * i1 * lam3 + c1 * c1 * lam4 + c2 * c2 * lam4 +
                            c1 * c1 * lam5 - c2 * c2 * lam5 + 2 * m222 + lam3 * r1 * r1 +
                            lam2 * r2 * r2) /
                            2.,
                    c3 * c4 * lam2 + c1 * c2 * lam5,
                    (2 * c3 * i1 * lam3 + c1 * i2 * (lam4 + lam5) +
                            c2 * (lam4 - lam5) * r2) /
                            2.,
                    c3 * i2 * lam2 +
                            (c1 * i1 * (lam4 + lam5) + c2 * (-lam4 + lam5) * r1) / 2.},
            {(c1 * i2 * (lam4 - lam5) + 2 * c4 * lam3 * r1 +
                    c2 * (lam4 + lam5) * r2) /
                    2.,
                    -(c1 * i1 * (lam4 - lam5)) / 2. + (c2 * (lam4 + lam5) * r1) / 2. +
                            c4 * lam2 * r2,
                    c1 * c4 * (lam3 + lam4 - lam5) + c2 * c3 * lam5 +
                            ((lam4 - lam5) * (i2 * r1 - i1 * r2)) / 2.,
                    (2 * c1 * c3 * lam5 + i1 * i2 * (lam4 + lam5) +
                            2 * c2 * c4 * (lam3 + lam4 + lam5) - 2 * m122 + lam4 * r1 * r2 +
                            lam5 * r1 * r2) /
                            2.,
                    c3 * c4 * lam2 + c1 * c2 * lam5,
                    (c3 * c3 * lam2 + 3 * c4 * c4 * lam2 + i2 * i2 * lam2 + c1 * c1 * lam3 +
                            c2 * c2 * lam3 + i1 * i1 * lam3 + c1 * c1 * lam4 + c2 * c2 * lam4 -
                            c1 * c1 * lam5 + c2 * c2 * lam5 + 2 * m222 + lam3 * r1 * r1 +
                            lam2 * r2 * r2) /
                            2.,
                    c4 * i1 * lam3 +
                            (c2 * i2 * (lam4 + lam5) + c1 * (-lam4 + lam5) * r2) / 2.,
                    (2 * c4 * i2 * lam2 + c2 * i1 * (lam4 + lam5) +
                            c1 * (lam4 - lam5) * r1) /
                            2.},
            {i1 * lam1 * r1 + i2 * lam5 * r2,
                    (c2 * c3 * (lam4 - lam5) + c1 * c4 * (-lam4 + lam5) +
                            2 * i2 * lam5 * r1 + 2 * i1 * (lam3 + lam4 - lam5) * r2) /
                            2.,
                    (2 * c1 * i1 * lam1 + c3 * i2 * (lam4 + lam5) +
                            c4 * (-lam4 + lam5) * r2) /
                            2.,
                    (2 * c2 * i1 * lam1 + c4 * i2 * (lam4 + lam5) +
                            c3 * (lam4 - lam5) * r2) /
                            2.,
                    (2 * c3 * i1 * lam3 + c1 * i2 * (lam4 + lam5) +
                            c2 * (lam4 - lam5) * r2) /
                            2.,
                    c4 * i1 * lam3 +
                            (c2 * i2 * (lam4 + lam5) + c1 * (-lam4 + lam5) * r2) / 2.,
                    (c1 * c1 * lam1 + c2 * c2 * lam1 + 3 * i1 * i1 * lam1 + c3 * c3 * lam3 +
                            c4 * c4 * lam3 + i2 * i2 * lam3 + i2 * i2 * lam4 + i2 * i2 * lam5 +
                            2 * m112 + lam1 * r1 * r1 + lam3 * r2 * r2 + lam4 * r2 * r2 -
                            lam5 * r2 * r2) /
                            2.,
                    (c2 * c4 * lam4 + c2 * c4 * lam5 + c1 * c3 * (lam4 + lam5) +
                            2 * i1 * i2 * (lam3 + lam4 + lam5) - 2 * m122 + 2 * lam5 * r1 * r2) /
                            2.},
            {(c1 * c4 * (lam4 - lam5) + c2 * c3 * (-lam4 + lam5) +
                    2 * i2 * (lam3 + lam4 - lam5) * r1 + 2 * i1 * lam5 * r2) /
                    2.,
                    i1 * lam5 * r1 + i2 * lam2 * r2,
                    c1 * i2 * lam3 +
                            (c3 * i1 * (lam4 + lam5) + c4 * (lam4 - lam5) * r1) / 2.,
                    c2 * i2 * lam3 +
                            (c4 * i1 * (lam4 + lam5) + c3 * (-lam4 + lam5) * r1) / 2.,
                    c3 * i2 * lam2 +
                            (c1 * i1 * (lam4 + lam5) + c2 * (-lam4 + lam5) * r1) / 2.,
                    (2 * c4 * i2 * lam2 + c2 * i1 * (lam4 + lam5) +
                            c1 * (lam4 - lam5) * r1) /
                            2.,
                    (c2 * c4 * lam4 + c2 * c4 * lam5 + c1 * c3 * (lam4 + lam5) +
                            2 * i1 * i2 * (lam3 + lam4 + lam5) - 2 * m122 + 2 * lam5 * r1 * r2) /
                            2.,
                    (c3 * c3 * lam2 + c4 * c4 * lam2 + 3 * i2 * i2 * lam2 + c1 * c1 * lam3 +
                            c2 * c2 * lam3 + i1 * i1 * lam3 + i1 * i1 * lam4 + i1 * i1 * lam5 +
                            2 * m222 + lam3 * r1 * r1 + lam4 * r1 * r1 - lam5 * r1 * r1 +
                            lam2 * r2 * r2) /
                            2.}};
}

/**
 * Computes the scalar mass matrix for a given set of fields and
 * parameters.
 * @param fields THDM fields
 * @param params THDM params
 * @return scalar mass matrix
 */
arma::mat scalar_squared_mass_matrix_arma(
        Fields<double> &fields, Parameters<double> &params) {
    double r1 = fields.r1;
    double r2 = fields.r2;
    double c1 = fields.c1;
    double c2 = fields.c2;
    double c3 = fields.c3;
    double c4 = fields.c4;
    double i1 = fields.i1;
    double i2 = fields.i2;

    double m112 = params.m112;
    double m122 = params.m122;
    double m222 = params.m222;
    double lam1 = params.lam1;
    double lam2 = params.lam2;
    double lam3 = params.lam3;
    double lam4 = params.lam4;
    double lam5 = params.lam5;


    return arma::mat{{(pow(c1, 2) * lam1 + pow(c2, 2) * lam1 + pow(i1, 2) * lam1 +
            pow(c3, 2) * lam3 + pow(c4, 2) * lam3 + pow(i2, 2) * lam3 +
            pow(i2, 2) * lam4 - pow(i2, 2) * lam5 + 2 * m112 + 3 * lam1 * pow(r1, 2) +
            lam3 * pow(r2, 2) + lam4 * pow(r2, 2) + lam5 * pow(r2, 2)) / 2.,
            (2 * i1 * i2 * lam5 + c1 * c3 * (lam4 + lam5) + c2 * c4 * (lam4 + lam5) - 2 * m122 +
                    2 * (lam3 + lam4 + lam5) * r1 * r2) / 2., (c4 * i2 * (lam4 - lam5) + 2 * c1 * lam1 * r1
                    + c3 * (lam4 + lam5) * r2) / 2., -(c3 * i2 * (lam4 - lam5)) / 2. + c2 * lam1 * r1 +
                    (c4 * (lam4 + lam5) * r2) / 2., (c2 * i2 * (-lam4 + lam5) + 2 * c3 * lam3 * r1 +
                    c1 * (lam4 + lam5) * r2) / 2., (c1 * i2 * (lam4 - lam5) + 2 * c4 * lam3 * r1 +
                    c2 * (lam4 + lam5) * r2) / 2., i1 * lam1 * r1 + i2 * lam5 * r2, (c1 * c4 * (lam4 -
                    lam5) + c2 * c3 * (-lam4 + lam5) + 2 * i2 * (lam3 + lam4 - lam5) * r1 +
                    2 * i1 * lam5 * r2) / 2.},
            {(2 * i1 * i2 * lam5 + c1 * c3 * (lam4 + lam5) + c2 * c4 * (lam4 + lam5) - 2 * m122 +
                    2 * (lam3 + lam4 + lam5) * r1 * r2) / 2., (pow(c3, 2) * lam2 +
                    pow(c4, 2) * lam2 + pow(i2, 2) * lam2 + pow(c1, 2) * lam3 +
                    pow(c2, 2) * lam3 + pow(i1, 2) * lam3 + pow(i1, 2) * lam4 -
                    pow(i1, 2) * lam5 + 2 * m222 + lam3 * pow(r1, 2) + lam4 * pow(r1, 2) +
                    lam5 * pow(r1, 2) + 3 * lam2 * pow(r2, 2)) / 2., (c4 * i1 * (-lam4 + lam5) +
                    c3 * (lam4 + lam5) * r1 + 2 * c1 * lam3 * r2) / 2., (c3 * i1 * (lam4 - lam5) +
                    c4 * (lam4 + lam5) * r1 + 2 * c2 * lam3 * r2) / 2., (c2 * i1 * (lam4 - lam5) +
                    c1 * (lam4 + lam5) * r1 + 2 * c3 * lam2 * r2) / 2., -(c1 * i1 * (lam4 - lam5)) / 2. +
                    (c2 * (lam4 + lam5) * r1) / 2. + c4 * lam2 * r2, (c2 * c3 * (lam4 - lam5) +
                    c1 * c4 * (-lam4 + lam5) + 2 * i2 * lam5 * r1 + 2 * i1 * (lam3 + lam4 -
                    lam5) * r2) / 2., i1 * lam5 * r1 + i2 * lam2 * r2},
            {(c4 * i2 * (lam4 - lam5) + 2 * c1 * lam1 * r1 + c3 * (lam4 + lam5) * r2) / 2.,
                    (c4 * i1 * (-lam4 + lam5) + c3 * (lam4 + lam5) * r1 + 2 * c1 * lam3 * r2) / 2.,
                    (3 * pow(c1, 2) * lam1 + pow(c2, 2) * lam1 + pow(i1, 2) * lam1 +
                            pow(c3, 2) * lam3 + pow(c4, 2) * lam3 + pow(i2, 2) * lam3 +
                            pow(c3, 2) * lam4 + pow(c4, 2) * lam4 + pow(c3, 2) * lam5 -
                            pow(c4, 2) * lam5 + 2 * m112 + lam1 * pow(r1, 2) + lam3 * pow(r2, 2)) / 2.,
                    c1 * c2 * lam1 + c3 * c4 * lam5, (2 * c2 * c4 * lam5 + i1 * i2 * (lam4 + lam5) +
                    2 * c1 * c3 * (lam3 + lam4 + lam5) - 2 * m122 + lam4 * r1 * r2 + lam5 * r1 * r2) / 2.,
                    c1 * c4 * (lam3 + lam4 - lam5) + c2 * c3 * lam5 + ((lam4 - lam5) * (i2 * r1 -
                            i1 * r2)) / 2., (2 * c1 * i1 * lam1 + c3 * i2 * (lam4 + lam5) + c4 * (-lam4 +
                    lam5) * r2) / 2., c1 * i2 * lam3 + (c3 * i1 * (lam4 + lam5) + c4 * (lam4 -
                    lam5) * r1) / 2.},
            {-(c3 * i2 * (lam4 - lam5)) / 2. + c2 * lam1 * r1 + (c4 * (lam4 + lam5) * r2) / 2.,
                    (c3 * i1 * (lam4 - lam5) + c4 * (lam4 + lam5) * r1 + 2 * c2 * lam3 * r2) / 2.,
                    c1 * c2 * lam1 + c3 * c4 * lam5, (pow(c1, 2) * lam1 + 3 * pow(c2, 2) * lam1 +
                    pow(i1, 2) * lam1 + pow(c4, 2) * lam3 + pow(i2, 2) * lam3 +
                    pow(c4, 2) * lam4 + pow(c3, 2) * (lam3 + lam4 - lam5) +
                    pow(c4, 2) * lam5 + 2 * m112 + lam1 * pow(r1, 2) + lam3 * pow(r2, 2)) / 2.,
                    c2 * c3 * (lam3 + lam4 - lam5) + c1 * c4 * lam5 - ((lam4 - lam5) * (i2 * r1 -
                            i1 * r2)) / 2., (2 * c1 * c3 * lam5 + i1 * i2 * (lam4 + lam5) + 2 * c2 * c4 * (lam3 +
                    lam4 + lam5) - 2 * m122 + lam4 * r1 * r2 + lam5 * r1 * r2) / 2., (2 * c2 * i1 * lam1 +
                    c4 * i2 * (lam4 + lam5) + c3 * (lam4 - lam5) * r2) / 2., c2 * i2 * lam3 +
                    (c4 * i1 * (lam4 + lam5) + c3 * (-lam4 + lam5) * r1) / 2.},
            {(c2 * i2 * (-lam4 + lam5) + 2 * c3 * lam3 * r1 + c1 * (lam4 + lam5) * r2) / 2.,
                    (c2 * i1 * (lam4 - lam5) + c1 * (lam4 + lam5) * r1 + 2 * c3 * lam2 * r2) / 2.,
                    (2 * c2 * c4 * lam5 + i1 * i2 * (lam4 + lam5) + 2 * c1 * c3 * (lam3 + lam4 + lam5) -
                            2 * m122 + lam4 * r1 * r2 + lam5 * r1 * r2) / 2., c2 * c3 * (lam3 + lam4 - lam5) +
                    c1 * c4 * lam5 - ((lam4 - lam5) * (i2 * r1 - i1 * r2)) / 2., (3 * pow(c3, 2) * lam2
                    + pow(c4, 2) * lam2 + pow(i2, 2) * lam2 + pow(c1, 2) * lam3 +
                    pow(c2, 2) * lam3 + pow(i1, 2) * lam3 + pow(c1, 2) * lam4 +
                    pow(c2, 2) * lam4 + pow(c1, 2) * lam5 - pow(c2, 2) * lam5 + 2 * m222 +
                    lam3 * pow(r1, 2) + lam2 * pow(r2, 2)) / 2., c3 * c4 * lam2 + c1 * c2 * lam5,
                    (2 * c3 * i1 * lam3 + c1 * i2 * (lam4 + lam5) + c2 * (lam4 - lam5) * r2) / 2.,
                    c3 * i2 * lam2 + (c1 * i1 * (lam4 + lam5) + c2 * (-lam4 + lam5) * r1) / 2.},
            {(c1 * i2 * (lam4 - lam5) + 2 * c4 * lam3 * r1 + c2 * (lam4 + lam5) * r2) / 2.,
                    -(c1 * i1 * (lam4 - lam5)) / 2. + (c2 * (lam4 + lam5) * r1) / 2. + c4 * lam2 * r2,
                    c1 * c4 * (lam3 + lam4 - lam5) + c2 * c3 * lam5 + ((lam4 - lam5) * (i2 * r1 -
                            i1 * r2)) / 2., (2 * c1 * c3 * lam5 + i1 * i2 * (lam4 + lam5) + 2 * c2 * c4 * (lam3 +
                    lam4 + lam5) - 2 * m122 + lam4 * r1 * r2 + lam5 * r1 * r2) / 2., c3 * c4 * lam2 +
                    c1 * c2 * lam5, (pow(c3, 2) * lam2 + 3 * pow(c4, 2) * lam2 + pow(i2, 2) * lam2
                    + pow(c1, 2) * lam3 + pow(c2, 2) * lam3 + pow(i1, 2) * lam3 +
                    pow(c1, 2) * lam4 + pow(c2, 2) * lam4 - pow(c1, 2) * lam5 +
                    pow(c2, 2) * lam5 + 2 * m222 + lam3 * pow(r1, 2) + lam2 * pow(r2, 2)) / 2.,
                    c4 * i1 * lam3 + (c2 * i2 * (lam4 + lam5) + c1 * (-lam4 + lam5) * r2) / 2.,
                    (2 * c4 * i2 * lam2 + c2 * i1 * (lam4 + lam5) + c1 * (lam4 - lam5) * r1) / 2.},
            {i1 * lam1 * r1 + i2 * lam5 * r2, (c2 * c3 * (lam4 - lam5) + c1 * c4 * (-lam4 + lam5)
                    + 2 * i2 * lam5 * r1 + 2 * i1 * (lam3 + lam4 - lam5) * r2) / 2., (2 * c1 * i1 * lam1 +
                    c3 * i2 * (lam4 + lam5) + c4 * (-lam4 + lam5) * r2) / 2., (2 * c2 * i1 * lam1 +
                    c4 * i2 * (lam4 + lam5) + c3 * (lam4 - lam5) * r2) / 2., (2 * c3 * i1 * lam3 +
                    c1 * i2 * (lam4 + lam5) + c2 * (lam4 - lam5) * r2) / 2., c4 * i1 * lam3 +
                    (c2 * i2 * (lam4 + lam5) + c1 * (-lam4 + lam5) * r2) / 2., (pow(c1, 2) * lam1 +
                    pow(c2, 2) * lam1 + 3 * pow(i1, 2) * lam1 + pow(c3, 2) * lam3 +
                    pow(c4, 2) * lam3 + pow(i2, 2) * lam3 + pow(i2, 2) * lam4 +
                    pow(i2, 2) * lam5 + 2 * m112 + lam1 * pow(r1, 2) + lam3 * pow(r2, 2) +
                    lam4 * pow(r2, 2) - lam5 * pow(r2, 2)) / 2., (c2 * c4 * lam4 + c2 * c4 * lam5 +
                    c1 * c3 * (lam4 + lam5) + 2 * i1 * i2 * (lam3 + lam4 + lam5) - 2 * m122 +
                    2 * lam5 * r1 * r2) / 2.},
            {(c1 * c4 * (lam4 - lam5) + c2 * c3 * (-lam4 + lam5) + 2 * i2 * (lam3 + lam4 -
                    lam5) * r1 + 2 * i1 * lam5 * r2) / 2., i1 * lam5 * r1 + i2 * lam2 * r2, c1 * i2 * lam3 +
                    (c3 * i1 * (lam4 + lam5) + c4 * (lam4 - lam5) * r1) / 2., c2 * i2 * lam3 +
                    (c4 * i1 * (lam4 + lam5) + c3 * (-lam4 + lam5) * r1) / 2., c3 * i2 * lam2 +
                    (c1 * i1 * (lam4 + lam5) + c2 * (-lam4 + lam5) * r1) / 2., (2 * c4 * i2 * lam2 +
                    c2 * i1 * (lam4 + lam5) + c1 * (lam4 - lam5) * r1) / 2., (c2 * c4 * lam4 +
                    c2 * c4 * lam5 + c1 * c3 * (lam4 + lam5) + 2 * i1 * i2 * (lam3 + lam4 + lam5) -
                    2 * m122 + 2 * lam5 * r1 * r2) / 2., (pow(c3, 2) * lam2 + pow(c4, 2) * lam2 +
                    3 * pow(i2, 2) * lam2 + pow(c1, 2) * lam3 + pow(c2, 2) * lam3 +
                    pow(i1, 2) * lam3 + pow(i1, 2) * lam4 + pow(i1, 2) * lam5 + 2 * m222 +
                    lam3 * pow(r1, 2) + lam4 * pow(r1, 2) - lam5 * pow(r1, 2) +
                    lam2 * pow(r2, 2)) / 2.}};
}


/**
 * Computes the scalar squared masses for a given set of fields and
 * parameters.
 * @tparam T type of the fields and params: should be a double or dual.
 * @param fields Fields<T> containing 2hdm fields.
 * @param params Parameters<T> containing 2hdm parameters.
 * @return boost vector<T> containing the squared scalar masses.
 */
std::vector<double> scalar_squared_masses(
        Fields<double> &fields, Parameters<double> &params) {
    auto M = scalar_squared_mass_matrix_arma(fields, params);
    arma::vec evals;
    arma::eig_sym(evals, M);
    std::vector<double> lams;
    for (auto lam: evals) {
        lams.push_back(lam);
    }
    return lams;
}

/**
 * Determine if the tree-level scalar masses at the given vacuua are
 * positive
 * @param nvac Normal vacuum.
 * @param cbvac Charge-breaking vacuum.
 * @param params THDM parameters.
 * @return true if all masses are positive, false if not.
 */
bool are_sqrd_masses_positive_semi_definite(const Vacuum<double> &nvac,
                                            const Vacuum<double> &cbvac,
                                            Parameters<double> &params) {
    Fields<double> fields;
    fields.set_fields(nvac);
    auto masses_n = scalar_squared_masses(fields, params);
    fields.set_fields(cbvac);
    auto masses_cb = scalar_squared_masses(fields, params);

    bool positive_masses = true;

    for (auto m : masses_n)
        positive_masses *= (m > -1e-7);
    for (auto m: masses_cb)
        positive_masses *= (m > -1e-7);

    return positive_masses;
}

/**
 * Determine if the tree-level scalar masses at the given vacuum are
 * positive
 * @param nvac Normal vacuum.
 * @param cbvac Charge-breaking vacuum.
 * @param params THDM parameters.
 * @return true if all masses are positive, false if not.
 */
bool are_sqrd_masses_positive_semi_definite(const Vacuum<double> &vac,
                                            Parameters<double> &params) {
    Fields<double> fields;
    fields.set_fields(vac);
    auto masses = scalar_squared_masses(fields, params);

    bool positive_masses = true;

    for (auto m: masses)
        positive_masses *= (m > -1e-7);

    return positive_masses;
}


/**
 * Computes the first derivative of the scalar squared masses wrt a
 * field for a given set of fields and parameters.
 * @param fields Fields<double> containing 2hdm fields.
 * @param params Parameters<double> containing 2hdm parameters.
 * @param fld integer index of the field to take derivative of.
 * @return boost vector<std::pair<double,double>> containing the squared
 * scalar masses and derivatives.
 */

std::vector<std::tuple<double, double>>
scalar_squared_masses_deriv_fld(Fields<double> &fields,
                                Parameters<double> &params, int fld) {
    if (fld < 1 || fld > 8) {
        throw THDMException(THDMExceptionCode::ScalarMassesInvalidFldIndex);
    }
    // Make a new set of fields and parameters.
    Fields<Dual<double>> _fields{};
    Parameters<Dual<double>> _params{};
    // Fill up the fields and parameters will dual numbers who's value
    // agrees with old fields and parameters. if i == fld - 1, then set
    // that field's eps to 1;
    for (int i = 0; i < 8; i++) {
        // Need to do fld -1 since fld in [1, 8] and i in [0,7]
        _fields[i].eps = (i == fld - 1) ? 1.0 : 0.0;
        _fields[i].val = fields[i];
        _params[i].val = params[i];
    }
    // Lastly, set mu for params. Not that it really matters
    _params.mu.val = params.mu;

    // Compute the scalar mass matrix:
    auto mat = scalar_squared_mass_matrix(_fields, _params);
    // Compute the eigenvalues and their derivatives:
    auto evals = jacobi(mat);
    // create new vector with eigenvalues and derivatives:
    std::vector<std::tuple<double, double>> lam_dlams(8);
    for (size_t i = 0; i < 8; i++) {
        lam_dlams[i] = std::make_tuple(evals[i].val, evals[i].eps);
    }
    return lam_dlams;
}

/**
 * Computes the second derivative of the scalar squared masses wrt two
 * fields for a given set of fields and parameters.
 * @param fields Fields<double> containing 2hdm fields.
 * @param params Parameters<double> containing 2hdm parameters.
 * @param fld1 integer index of the first field to take derivative of.
 * @param fld2 integer index of the second field to take derivative of.
 * @return vector of tuples containing the eigenvalues, the derivative
 * of the evals wrt first field, deriv of evals wrt second field and
 * mixed derivatives.
 */

std::vector<std::tuple<double, double, double, double>>
scalar_squared_masses_deriv_fld(Fields<double> &fields,
                                Parameters<double> &params, int fld1,
                                int fld2) {
    if ((fld1 < 1 || fld1 > 8) || (fld2 < 1 || fld2 > 8)) {
        throw THDMException(THDMExceptionCode::ScalarMassesInvalidFldIndex);
    }

    Fields<Dual<Dual<double >>> _fields{};
    Parameters<Dual<Dual<double >>> _params{};

    for (int i = 0; i < 8; i++) {
        // Need to do fld -1 since fld in [1, 8] and i in [0,7]
        _fields[i].val.eps = (i == fld1 - 1) ? 1.0 : 0.0;
        _fields[i].eps.val = (i == fld2 - 1) ? 1.0 : 0.0;

        // Jacobi seems to get second derivative wrong if value is exactly 0.0
        // It appears like the issue is only if fld1 == fld2 and fields[fld1] == 0.0
        // so, if this is the case, we will shift the field by a small amount: 1e-10
        if ((i == fld1 - 1 || i == fld2 - 1) && fields[i] == 0.0)
            _fields[i].val.val = 1e-10;
        else
            _fields[i].val.val = fields[i];
        _params[i].val.val = params[i];
    }
    _params.mu.val.val = params.mu;

    auto mat = scalar_squared_mass_matrix(_fields, _params);
    auto evals = jacobi(mat);

    std::vector<std::tuple<double, double, double, double>> lam_dlams(8);
    for (size_t i = 0; i < 8; i++) {
        lam_dlams[i] = std::make_tuple(evals[i].val.val, evals[i].val.eps,
                                       evals[i].eps.val, evals[i].eps.eps);
    }
    return lam_dlams;
}


/**
 * Computes the second derivative of the scalar squared masses wrt a
 * field and a parameter for a given set of fields
 * and parameters.
 * @param fields Fields<double> containing 2hdm fields.
 * @param params Parameters<double> containing 2hdm parameters.
 * @param fld integer index of the field to take derivative of.
 * @param par integer index of the parameter to take derivative of.
 * @return boost vector<std::pair<double,double>> containing the squared
 * scalar masses and derivatives.
 */
std::vector<std::tuple<double, double, double, double>>
scalar_squared_masses_deriv_fld_par(Fields<double> &fields,
                                    Parameters<double> &params, int fld,
                                    int par) {
    if ((fld < 1 || fld > 8)) {
        throw THDMException(THDMExceptionCode::ScalarMassesInvalidFldIndex);
    }
    if (par < 1 || par > 8) {
        throw THDMException(THDMExceptionCode::ScalarMassesInvalidParIndex);
    }

    Fields<Dual<Dual<double >>> _fields{};
    Parameters<Dual<Dual<double >>> _params{};

    for (int i = 0; i < 8; i++) {
        // Need to do fld -1 since fld in [1, 8] and i in [0,7]
        _fields[i].val.eps = (i == par - 1) ? 1.0 : 0.0;
        _params[i].eps.val = (i == fld - 1) ? 1.0 : 0.0;

        _fields[i].val.val = fields[i];
        _params[i].val.val = params[i];
    }
    _params.mu.val.val = params.mu;

    auto mat = scalar_squared_mass_matrix(_fields, _params);
    auto evals = jacobi(mat);

    std::vector<std::tuple<double, double, double, double>> lam_dlams(8);
    for (size_t i = 0; i < 8; i++) {
        lam_dlams[i] = std::make_tuple(evals[i].val.val, evals[i].val.eps,
                                       evals[i].eps.val, evals[i].eps.eps);
    }
    return lam_dlams;
}

} // namespace thdm

#endif // MASSES_HPP