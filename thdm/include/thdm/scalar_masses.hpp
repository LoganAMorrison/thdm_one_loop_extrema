#ifndef MASSES_HPP
#define MASSES_HPP

#include "thdm/dual.hpp"
#include "thdm/fields.hpp"
#include "thdm/jacobi.hpp"
#include "thdm/parameters.hpp"
#include <cmath>
#include <exception>
#include <iostream>
#include <tuple>
#include <vector>
#include <assert.h>
#include <armadillo>

namespace thdm {

struct InvalidFldIndexException : public std::exception {
    const char *what() const noexcept override {
        return "Field must be in range 1...8 in call to "
               "'scalar_squared_masses_deriv'.";
    }
};

struct InvalidParIndexException : public std::exception {
    const char *what() const noexcept override {
        return "Parameter must be in range 1...8 in call to "
               "'scalar_squared_masses_deriv'.";
    }
};

/**
 * Computes the scalar mass matrix for a given set of fields and
 * parameters.
 * @tparam T type of the fields and params: should be a double or dual.
 * @param fields Fields<T> containing 2hdm fields.
 * @param params Parameters<T> containing 2hdm parameters.
 * @return boost matrix<T> containing the scalar mass matrix.
 */
template<class T>
std::vector<std::vector<T>> scalar_squared_mass_matrix(Fields<T> &fields,
                                                       Parameters<T> &params) {
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
 * Compute the first derivative of the scalar
 * mass matrix wrt a field
 * @param fields THDM fields
 * @param params THDM params
 * @param fld field to take derivative wrt.
 * @return derivative of scalar mass matrix.
 */
arma::mat scalar_squared_mass_matrix_deriv(Fields<double> &fields,
                                           Parameters<double> &params,
                                           int fld) {
    double r1 = fields.r1;
    double r2 = fields.r2;
    double c1 = fields.c1;
    double c2 = fields.c2;
    double c3 = fields.c3;
    double c4 = fields.c4;
    double i1 = fields.i1;
    double i2 = fields.i2;

    double lam1 = params.lam1;
    double lam2 = params.lam2;
    double lam3 = params.lam3;
    double lam4 = params.lam4;
    double lam5 = params.lam5;

    assert(1 <= fld && fld <= 8);

    arma::mat mat;
    if (fld == 1) {
        return arma::mat{{3 * lam1 * r1, (lam3 + lam4 + lam5) * r2, c1 * lam1, c2 * lam1, c3 * lam3,
                c4 * lam3, i1 * lam1, i2 * (lam3 + lam4 - lam5)},
                {(lam3 + lam4 + lam5) * r2, (2 * lam3 * r1 + 2 * lam4 * r1 + 2 * lam5 * r1) / 2.,
                        (c3 * (lam4 + lam5)) / 2., (c4 * (lam4 + lam5)) / 2., (c1 * (lam4 + lam5)) / 2.,
                        (c2 * (lam4 + lam5)) / 2., i2 * lam5, i1 * lam5},
                {c1 * lam1, (c3 * (lam4 + lam5)) / 2., lam1 * r1, 0, (lam4 * r2 +
                        lam5 * r2) / 2., (i2 * (lam4 - lam5)) / 2., 0, (c4 * (lam4 - lam5)) / 2.},
                {c2 * lam1, (c4 * (lam4 + lam5)) / 2., 0, lam1 * r1, -(i2 * (lam4 -
                        lam5)) / 2., (lam4 * r2 + lam5 * r2) / 2., 0, (c3 * (-lam4 + lam5)) / 2.},
                {c3 * lam3, (c1 * (lam4 + lam5)) / 2., (lam4 * r2 + lam5 * r2) / 2., -(i2 * (lam4
                        - lam5)) / 2., lam3 * r1, 0, 0, (c2 * (-lam4 + lam5)) / 2.},
                {c4 * lam3, (c2 * (lam4 + lam5)) / 2., (i2 * (lam4 - lam5)) / 2., (lam4 * r2 +
                        lam5 * r2) / 2., 0, lam3 * r1, 0, (c1 * (lam4 - lam5)) / 2.},
                {i1 * lam1, i2 * lam5, 0, 0, 0, 0, lam1 * r1, lam5 * r2},
                {i2 * (lam3 + lam4 - lam5), i1 * lam5, (c4 * (lam4 - lam5)) / 2.,
                        (c3 * (-lam4 + lam5)) / 2., (c2 * (-lam4 + lam5)) / 2., (c1 * (lam4 -
                        lam5)) / 2., lam5 * r2, (2 * lam3 * r1 + 2 * lam4 * r1 - 2 * lam5 * r1) / 2.}};
    } else if (fld == 2) {
        return arma::mat{{(2 * lam3 * r2 + 2 * lam4 * r2 + 2 * lam5 * r2) / 2., (lam3 + lam4 + lam5) * r1,
                (c3 * (lam4 + lam5)) / 2., (c4 * (lam4 + lam5)) / 2., (c1 * (lam4 + lam5)) / 2.,
                (c2 * (lam4 + lam5)) / 2., i2 * lam5, i1 * lam5},
                {(lam3 + lam4 + lam5) * r1, 3 * lam2 * r2, c1 * lam3, c2 * lam3, c3 * lam2,
                        c4 * lam2, i1 * (lam3 + lam4 - lam5), i2 * lam2},
                {(c3 * (lam4 + lam5)) / 2., c1 * lam3, lam3 * r2, 0, (lam4 * r1 +
                        lam5 * r1) / 2., -(i1 * (lam4 - lam5)) / 2., (c4 * (-lam4 + lam5)) / 2., 0},
                {(c4 * (lam4 + lam5)) / 2., c2 * lam3, 0, lam3 * r2, (i1 * (lam4 - lam5)) / 2.,
                        (lam4 * r1 + lam5 * r1) / 2., (c3 * (lam4 - lam5)) / 2., 0},
                {(c1 * (lam4 + lam5)) / 2., c3 * lam2, (lam4 * r1 + lam5 * r1) / 2., (i1 * (lam4
                        - lam5)) / 2., lam2 * r2, 0, (c2 * (lam4 - lam5)) / 2., 0},
                {(c2 * (lam4 + lam5)) / 2., c4 * lam2, -(i1 * (lam4 - lam5)) / 2., (lam4 * r1 +
                        lam5 * r1) / 2., 0, lam2 * r2, (c1 * (-lam4 + lam5)) / 2., 0},
                {i2 * lam5, i1 * (lam3 + lam4 - lam5), (c4 * (-lam4 + lam5)) / 2.,
                        (c3 * (lam4 - lam5)) / 2., (c2 * (lam4 - lam5)) / 2., (c1 * (-lam4 + lam5)) / 2.,
                        (2 * lam3 * r2 + 2 * lam4 * r2 - 2 * lam5 * r2) / 2., lam5 * r1},
                {i1 * lam5, i2 * lam2, 0, 0, 0, 0, lam5 * r1, lam2 * r2}};
    } else if (fld == 3) {
        return arma::mat{{c1 * lam1, (c3 * (lam4 + lam5)) / 2., lam1 * r1, 0, ((lam4 + lam5) * r2) / 2.,
                (i2 * (lam4 - lam5)) / 2., 0, (c4 * (lam4 - lam5)) / 2.},
                {(c3 * (lam4 + lam5)) / 2., c1 * lam3, lam3 * r2, 0, ((lam4 + lam5) * r1) / 2.,
                        -(i1 * (lam4 - lam5)) / 2., (c4 * (-lam4 + lam5)) / 2., 0},
                {lam1 * r1, lam3 * r2, 3 * c1 * lam1, c2 * lam1, c3 * (lam3 + lam4 + lam5),
                        c4 * (lam3 + lam4 - lam5), i1 * lam1, i2 * lam3},
                {0, 0, c2 * lam1, c1 * lam1, c4 * lam5, c3 * lam5, 0, 0},
                {((lam4 + lam5) * r2) / 2., ((lam4 + lam5) * r1) / 2., c3 * (lam3 + lam4 +
                        lam5), c4 * lam5, (2 * c1 * lam3 + 2 * c1 * lam4 + 2 * c1 * lam5) / 2., c2 * lam5,
                        (i2 * (lam4 + lam5)) / 2., (i1 * (lam4 + lam5)) / 2.},
                {(i2 * (lam4 - lam5)) / 2., -(i1 * (lam4 - lam5)) / 2., c4 * (lam3 + lam4 -
                        lam5), c3 * lam5, c2 * lam5, (2 * c1 * lam3 + 2 * c1 * lam4 - 2 * c1 * lam5) / 2.,
                        ((-lam4 + lam5) * r2) / 2., ((lam4 - lam5) * r1) / 2.},
                {0, (c4 * (-lam4 + lam5)) / 2., i1 * lam1, 0, (i2 * (lam4 + lam5)) / 2.,
                        ((-lam4 + lam5) * r2) / 2., c1 * lam1, (c3 * (lam4 + lam5)) / 2.},
                {(c4 * (lam4 - lam5)) / 2., 0, i2 * lam3, 0, (i1 * (lam4 + lam5)) / 2.,
                        ((lam4 - lam5) * r1) / 2., (c3 * (lam4 + lam5)) / 2., c1 * lam3}};
    } else if (fld == 4) {
        return arma::mat{{c2 * lam1, (c4 * (lam4 + lam5)) / 2., 0, lam1 * r1, (i2 * (-lam4 +
                lam5)) / 2., ((lam4 + lam5) * r2) / 2., 0, (c3 * (-lam4 + lam5)) / 2.},
                {(c4 * (lam4 + lam5)) / 2., c2 * lam3, 0, lam3 * r2, (i1 * (lam4 - lam5)) / 2.,
                        ((lam4 + lam5) * r1) / 2., (c3 * (lam4 - lam5)) / 2., 0},
                {0, 0, c2 * lam1, c1 * lam1, c4 * lam5, c3 * lam5, 0, 0},
                {lam1 * r1, lam3 * r2, c1 * lam1, 3 * c2 * lam1, c3 * (lam3 + lam4 - lam5),
                        c4 * (lam3 + lam4 + lam5), i1 * lam1, i2 * lam3},
                {(i2 * (-lam4 + lam5)) / 2., (i1 * (lam4 - lam5)) / 2., c4 * lam5, c3 * (lam3 +
                        lam4 - lam5), (2 * c2 * lam3 + 2 * c2 * lam4 - 2 * c2 * lam5) / 2., c1 * lam5, ((lam4
                        - lam5) * r2) / 2., ((-lam4 + lam5) * r1) / 2.},
                {((lam4 + lam5) * r2) / 2., ((lam4 + lam5) * r1) / 2., c3 * lam5, c4 * (lam3 +
                        lam4 + lam5), c1 * lam5, (2 * c2 * lam3 + 2 * c2 * lam4 + 2 * c2 * lam5) / 2.,
                        (i2 * (lam4 + lam5)) / 2., (i1 * (lam4 + lam5)) / 2.},
                {0, (c3 * (lam4 - lam5)) / 2., 0, i1 * lam1, ((lam4 - lam5) * r2) / 2.,
                        (i2 * (lam4 + lam5)) / 2., c2 * lam1, (c4 * lam4 + c4 * lam5) / 2.},
                {(c3 * (-lam4 + lam5)) / 2., 0, 0, i2 * lam3, ((-lam4 + lam5) * r1) / 2.,
                        (i1 * (lam4 + lam5)) / 2., (c4 * lam4 + c4 * lam5) / 2., c2 * lam3}};
    } else if (fld == 5) {
        return arma::mat{{c3 * lam3, (c1 * (lam4 + lam5)) / 2., ((lam4 + lam5) * r2) / 2., -(i2 * (lam4
                - lam5)) / 2., lam3 * r1, 0, 0, (c2 * (-lam4 + lam5)) / 2.},
                {(c1 * (lam4 + lam5)) / 2., c3 * lam2, ((lam4 + lam5) * r1) / 2., (i1 * (lam4 -
                        lam5)) / 2., lam2 * r2, 0, (c2 * (lam4 - lam5)) / 2., 0},
                {((lam4 + lam5) * r2) / 2., ((lam4 + lam5) * r1) / 2., (2 * c3 * lam3 +
                        2 * c3 * lam4 + 2 * c3 * lam5) / 2., c4 * lam5, c1 * (lam3 + lam4 + lam5), c2 * lam5,
                        (i2 * (lam4 + lam5)) / 2., (i1 * (lam4 + lam5)) / 2.},
                {-(i2 * (lam4 - lam5)) / 2., (i1 * (lam4 - lam5)) / 2., c4 * lam5, c3 * (lam3 +
                        lam4 - lam5), c2 * (lam3 + lam4 - lam5), c1 * lam5, ((lam4 -
                        lam5) * r2) / 2., ((-lam4 + lam5) * r1) / 2.},
                {lam3 * r1, lam2 * r2, c1 * (lam3 + lam4 + lam5), c2 * (lam3 + lam4 -
                        lam5), 3 * c3 * lam2, c4 * lam2, i1 * lam3, i2 * lam2},
                {0, 0, c2 * lam5, c1 * lam5, c4 * lam2, c3 * lam2, 0, 0},
                {0, (c2 * (lam4 - lam5)) / 2., (i2 * (lam4 + lam5)) / 2., ((lam4 -
                        lam5) * r2) / 2., i1 * lam3, 0, c3 * lam3, (c1 * (lam4 + lam5)) / 2.},
                {(c2 * (-lam4 + lam5)) / 2., 0, (i1 * (lam4 + lam5)) / 2., ((-lam4 +
                        lam5) * r1) / 2., i2 * lam2, 0, (c1 * (lam4 + lam5)) / 2., c3 * lam2}};
    } else if (fld == 6) {
        return arma::mat{{c4 * lam3, (c2 * (lam4 + lam5)) / 2., (i2 * (lam4 - lam5)) / 2., ((lam4 +
                lam5) * r2) / 2., 0, lam3 * r1, 0, (c1 * (lam4 - lam5)) / 2.},
                {(c2 * (lam4 + lam5)) / 2., c4 * lam2, (i1 * (-lam4 + lam5)) / 2., ((lam4 +
                        lam5) * r1) / 2., 0, lam2 * r2, (c1 * (-lam4 + lam5)) / 2., 0},
                {(i2 * (lam4 - lam5)) / 2., (i1 * (-lam4 + lam5)) / 2., (2 * c4 * lam3 +
                        2 * c4 * lam4 - 2 * c4 * lam5) / 2., c3 * lam5, c2 * lam5, c1 * (lam3 + lam4 - lam5),
                        ((-lam4 + lam5) * r2) / 2., ((lam4 - lam5) * r1) / 2.},
                {((lam4 + lam5) * r2) / 2., ((lam4 + lam5) * r1) / 2., c3 * lam5, (2 * c4 * lam3
                        + 2 * c4 * lam4 + 2 * c4 * lam5) / 2., c1 * lam5, c2 * (lam3 + lam4 + lam5),
                        (i2 * (lam4 + lam5)) / 2., (i1 * (lam4 + lam5)) / 2.},
                {0, 0, c2 * lam5, c1 * lam5, c4 * lam2, c3 * lam2, 0, 0},
                {lam3 * r1, lam2 * r2, c1 * (lam3 + lam4 - lam5), c2 * (lam3 + lam4 +
                        lam5), c3 * lam2, 3 * c4 * lam2, i1 * lam3, i2 * lam2},
                {0, (c1 * (-lam4 + lam5)) / 2., ((-lam4 + lam5) * r2) / 2., (i2 * (lam4 +
                        lam5)) / 2., 0, i1 * lam3, c4 * lam3, (c2 * lam4 + c2 * lam5) / 2.},
                {(c1 * (lam4 - lam5)) / 2., 0, ((lam4 - lam5) * r1) / 2., (i1 * (lam4 +
                        lam5)) / 2., 0, i2 * lam2, (c2 * lam4 + c2 * lam5) / 2., c4 * lam2}};
    } else if (fld == 7) {
        return arma::mat{{i1 * lam1, i2 * lam5, 0, 0, 0, 0, lam1 * r1, lam5 * r2},
                {i2 * lam5, (2 * i1 * lam3 + 2 * i1 * lam4 - 2 * i1 * lam5) / 2., (c4 * (-lam4 +
                        lam5)) / 2., (c3 * (lam4 - lam5)) / 2., (c2 * (lam4 - lam5)) / 2., -(c1 * (lam4 -
                        lam5)) / 2., (lam3 + lam4 - lam5) * r2, lam5 * r1},
                {0, (c4 * (-lam4 + lam5)) / 2., i1 * lam1, 0, (i2 * (lam4 + lam5)) / 2.,
                        -((lam4 - lam5) * r2) / 2., c1 * lam1, (c3 * (lam4 + lam5)) / 2.},
                {0, (c3 * (lam4 - lam5)) / 2., 0, i1 * lam1, ((lam4 - lam5) * r2) / 2.,
                        (i2 * (lam4 + lam5)) / 2., c2 * lam1, (c4 * (lam4 + lam5)) / 2.},
                {0, (c2 * (lam4 - lam5)) / 2., (i2 * (lam4 + lam5)) / 2., ((lam4 -
                        lam5) * r2) / 2., i1 * lam3, 0, c3 * lam3, (c1 * (lam4 + lam5)) / 2.},
                {0, -(c1 * (lam4 - lam5)) / 2., -((lam4 - lam5) * r2) / 2., (i2 * (lam4 +
                        lam5)) / 2., 0, i1 * lam3, c4 * lam3, (c2 * (lam4 + lam5)) / 2.},
                {lam1 * r1, (lam3 + lam4 - lam5) * r2, c1 * lam1, c2 * lam1, c3 * lam3,
                        c4 * lam3, 3 * i1 * lam1, i2 * (lam3 + lam4 + lam5)},
                {lam5 * r2, lam5 * r1, (c3 * (lam4 + lam5)) / 2., (c4 * (lam4 + lam5)) / 2.,
                        (c1 * (lam4 + lam5)) / 2., (c2 * (lam4 + lam5)) / 2., i2 * (lam3 + lam4 +
                        lam5), (2 * i1 * lam3 + 2 * i1 * lam4 + 2 * i1 * lam5) / 2.}};
    } else if (fld == 8) {
        return arma::mat{{(2 * i2 * lam3 + 2 * i2 * lam4 - 2 * i2 * lam5) / 2., i1 * lam5, (c4 * (lam4 -
                lam5)) / 2., -(c3 * (lam4 - lam5)) / 2., (c2 * (-lam4 + lam5)) / 2., (c1 * (lam4
                - lam5)) / 2., lam5 * r2, (lam3 + lam4 - lam5) * r1},
                {i1 * lam5, i2 * lam2, 0, 0, 0, 0, lam5 * r1, lam2 * r2},
                {(c4 * (lam4 - lam5)) / 2., 0, i2 * lam3, 0, (i1 * (lam4 + lam5)) / 2.,
                        ((lam4 - lam5) * r1) / 2., (c3 * (lam4 + lam5)) / 2., c1 * lam3},
                {-(c3 * (lam4 - lam5)) / 2., 0, 0, i2 * lam3, -((lam4 - lam5) * r1) / 2.,
                        (i1 * (lam4 + lam5)) / 2., (c4 * (lam4 + lam5)) / 2., c2 * lam3},
                {(c2 * (-lam4 + lam5)) / 2., 0, (i1 * (lam4 + lam5)) / 2., -((lam4 -
                        lam5) * r1) / 2., i2 * lam2, 0, (c1 * (lam4 + lam5)) / 2., c3 * lam2},
                {(c1 * (lam4 - lam5)) / 2., 0, ((lam4 - lam5) * r1) / 2., (i1 * (lam4 +
                        lam5)) / 2., 0, i2 * lam2, (c2 * (lam4 + lam5)) / 2., c4 * lam2},
                {lam5 * r2, lam5 * r1, (c3 * (lam4 + lam5)) / 2., (c4 * (lam4 + lam5)) / 2.,
                        (c1 * (lam4 + lam5)) / 2., (c2 * (lam4 + lam5)) / 2., (2 * i2 * lam3 + 2 * i2 * lam4
                        + 2 * i2 * lam5) / 2., i1 * (lam3 + lam4 + lam5)},
                {(lam3 + lam4 - lam5) * r1, lam2 * r2, c1 * lam3, c2 * lam3, c3 * lam2,
                        c4 * lam2, i1 * (lam3 + lam4 + lam5), 3 * i2 * lam2}};
    }

    return mat;
}

/**
 * Compute the second derivative of the scalar
 * mass matrix wrt  fields.
 * @param fields THDM fields
 * @param params THDM params
 * @param fld1 first field to take derivative wrt.
 * @param fld2 second field to take derivative wrt.
 * @return second derivative of scalar mass matrix.
 */
arma::mat scalar_squared_mass_matrix_deriv(Fields<double> &,
                                           Parameters<double> &params, int fld1, int fld2) {

    double lam1 = params.lam1;
    double lam2 = params.lam2;
    double lam3 = params.lam3;
    double lam4 = params.lam4;
    double lam5 = params.lam5;

    assert(1 <= fld1 && fld1 <= 8);
    assert(1 <= fld2 && fld2 <= 8);

    if (fld1 == 1) {
        if (fld2 == 1) {
            return arma::mat{{3 * lam1, 0, 0, 0, 0, 0, 0, 0},
                    {0, (2 * lam3 + 2 * lam4 + 2 * lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {0, 0, lam1, 0, 0, 0, 0, 0},
                    {0, 0, 0, lam1, 0, 0, 0, 0},
                    {0, 0, 0, 0, lam3, 0, 0, 0},
                    {0, 0, 0, 0, 0, lam3, 0, 0},
                    {0, 0, 0, 0, 0, 0, lam1, 0},
                    {0, 0, 0, 0, 0, 0, 0, (2 * lam3 + 2 * lam4 - 2 * lam5) / 2.}};
        } else if (fld2 == 2) {
            return arma::mat{{0, lam3 + lam4 + lam5, 0, 0, 0, 0, 0, 0},
                    {lam3 + lam4 + lam5, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0},
                    {0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0},
                    {0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0},
                    {0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, lam5},
                    {0, 0, 0, 0, 0, 0, lam5, 0}};
        } else if (fld2 == 3) {
            return arma::mat{{0, 0, lam1, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0},
                    {lam1, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, (lam4 - lam5) / 2.},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, (lam4 - lam5) / 2., 0, 0}};
        } else if (fld2 == 4) {
            return arma::mat{{0, 0, 0, lam1, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {lam1, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, (-lam4 + lam5) / 2.},
                    {0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, (-lam4 + lam5) / 2., 0, 0, 0}};
        } else if (fld2 == 5) {
            return arma::mat{{0, 0, 0, 0, lam3, 0, 0, 0},
                    {0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0},
                    {0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, (-lam4 + lam5) / 2.},
                    {lam3, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, (-lam4 + lam5) / 2., 0, 0, 0, 0}};
        } else if (fld2 == 6) {
            return arma::mat{{0, 0, 0, 0, 0, lam3, 0, 0},
                    {0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, (lam4 - lam5) / 2.},
                    {0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {lam3, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, (lam4 - lam5) / 2., 0, 0, 0, 0, 0}};
        } else if (fld2 == 7) {
            return arma::mat{{0, 0, 0, 0, 0, 0, lam1, 0},
                    {0, 0, 0, 0, 0, 0, 0, lam5},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {lam1, 0, 0, 0, 0, 0, 0, 0},
                    {0, lam5, 0, 0, 0, 0, 0, 0}};
        } else if (fld2 == 8) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, lam3 + lam4 - lam5},
                    {0, 0, 0, 0, 0, 0, lam5, 0},
                    {0, 0, 0, 0, 0, (lam4 - lam5) / 2., 0, 0},
                    {0, 0, 0, 0, (-lam4 + lam5) / 2., 0, 0, 0},
                    {0, 0, 0, (-lam4 + lam5) / 2., 0, 0, 0, 0},
                    {0, 0, (lam4 - lam5) / 2., 0, 0, 0, 0, 0},
                    {0, lam5, 0, 0, 0, 0, 0, 0},
                    {lam3 + lam4 - lam5, 0, 0, 0, 0, 0, 0, 0}};
        }

    } else if (fld1 == 2) {
        if (fld2 == 1) {
            return arma::mat{{0, lam3 + lam4 + lam5, 0, 0, 0, 0, 0, 0},
                    {lam3 + lam4 + lam5, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0},
                    {0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0},
                    {0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0},
                    {0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, lam5},
                    {0, 0, 0, 0, 0, 0, lam5, 0}};
        } else if (fld2 == 2) {
            return arma::mat{{(2 * lam3 + 2 * lam4 + 2 * lam5) / 2., 0, 0, 0, 0, 0, 0, 0},
                    {0, 3 * lam2, 0, 0, 0, 0, 0, 0},
                    {0, 0, lam3, 0, 0, 0, 0, 0},
                    {0, 0, 0, lam3, 0, 0, 0, 0},
                    {0, 0, 0, 0, lam2, 0, 0, 0},
                    {0, 0, 0, 0, 0, lam2, 0, 0},
                    {0, 0, 0, 0, 0, 0, (2 * lam3 + 2 * lam4 - 2 * lam5) / 2., 0},
                    {0, 0, 0, 0, 0, 0, 0, lam2}};
        } else if (fld2 == 3) {
            return arma::mat{{0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0},
                    {0, 0, lam3, 0, 0, 0, 0, 0},
                    {0, lam3, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {(lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, (-lam4 + lam5) / 2., 0},
                    {0, 0, 0, 0, 0, (-lam4 + lam5) / 2., 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (fld2 == 4) {
            return arma::mat{{0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0},
                    {0, 0, 0, lam3, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, lam3, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, (lam4 - lam5) / 2., 0},
                    {(lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, (lam4 - lam5) / 2., 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (fld2 == 5) {
            return arma::mat{{0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, lam2, 0, 0, 0},
                    {(lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, (lam4 - lam5) / 2., 0},
                    {0, lam2, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, (lam4 - lam5) / 2., 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (fld2 == 6) {
            return arma::mat{{0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, lam2, 0, 0},
                    {0, 0, 0, 0, 0, 0, (-lam4 + lam5) / 2., 0},
                    {(lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, lam2, 0, 0, 0, 0, 0, 0},
                    {0, 0, (-lam4 + lam5) / 2., 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (fld2 == 7) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, lam5},
                    {0, 0, 0, 0, 0, 0, lam3 + lam4 - lam5, 0},
                    {0, 0, 0, 0, 0, (-lam4 + lam5) / 2., 0, 0},
                    {0, 0, 0, 0, (lam4 - lam5) / 2., 0, 0, 0},
                    {0, 0, 0, (lam4 - lam5) / 2., 0, 0, 0, 0},
                    {0, 0, (-lam4 + lam5) / 2., 0, 0, 0, 0, 0},
                    {0, lam3 + lam4 - lam5, 0, 0, 0, 0, 0, 0},
                    {lam5, 0, 0, 0, 0, 0, 0, 0}};
        } else if (fld2 == 8) {
            return arma::mat{{0, 0, 0, 0, 0, 0, lam5, 0},
                    {0, 0, 0, 0, 0, 0, 0, lam2},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {lam5, 0, 0, 0, 0, 0, 0, 0},
                    {0, lam2, 0, 0, 0, 0, 0, 0}};
        }

    } else if (fld1 == 3) {
        if (fld2 == 1) {
            return arma::mat{{0, 0, lam1, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0},
                    {lam1, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, (lam4 - lam5) / 2.},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, (lam4 - lam5) / 2., 0, 0}};
        } else if (fld2 == 2) {
            return arma::mat{{0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0},
                    {0, 0, lam3, 0, 0, 0, 0, 0},
                    {0, lam3, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {(lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, (-lam4 + lam5) / 2., 0},
                    {0, 0, 0, 0, 0, (-lam4 + lam5) / 2., 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (fld2 == 3) {
            return arma::mat{{lam1, 0, 0, 0, 0, 0, 0, 0},
                    {0, lam3, 0, 0, 0, 0, 0, 0},
                    {0, 0, 3 * lam1, 0, 0, 0, 0, 0},
                    {0, 0, 0, lam1, 0, 0, 0, 0},
                    {0, 0, 0, 0, (2 * lam3 + 2 * lam4 + 2 * lam5) / 2., 0, 0, 0},
                    {0, 0, 0, 0, 0, (2 * lam3 + 2 * lam4 - 2 * lam5) / 2., 0, 0},
                    {0, 0, 0, 0, 0, 0, lam1, 0},
                    {0, 0, 0, 0, 0, 0, 0, lam3}};
        } else if (fld2 == 4) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, lam1, 0, 0, 0, 0},
                    {0, 0, lam1, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, lam5, 0, 0},
                    {0, 0, 0, 0, lam5, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (fld2 == 5) {
            return arma::mat{{0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {(lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, lam3 + lam4 + lam5, 0, 0, 0},
                    {0, 0, 0, 0, 0, lam5, 0, 0},
                    {0, 0, lam3 + lam4 + lam5, 0, 0, 0, 0, 0},
                    {0, 0, 0, lam5, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2.},
                    {0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0}};
        } else if (fld2 == 6) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, (lam4 - lam5) / 2.},
                    {0, 0, 0, 0, 0, 0, (-lam4 + lam5) / 2., 0},
                    {0, 0, 0, 0, 0, lam3 + lam4 - lam5, 0, 0},
                    {0, 0, 0, 0, lam5, 0, 0, 0},
                    {0, 0, 0, lam5, 0, 0, 0, 0},
                    {0, 0, lam3 + lam4 - lam5, 0, 0, 0, 0, 0},
                    {0, (-lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {(lam4 - lam5) / 2., 0, 0, 0, 0, 0, 0, 0}};
        } else if (fld2 == 7) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, (-lam4 + lam5) / 2., 0, 0},
                    {0, 0, 0, 0, 0, 0, lam1, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2.},
                    {0, (-lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {0, 0, lam1, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0}};
        } else if (fld2 == 8) {
            return arma::mat{{0, 0, 0, 0, 0, (lam4 - lam5) / 2., 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, lam3},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0},
                    {(lam4 - lam5) / 2., 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0},
                    {0, 0, lam3, 0, 0, 0, 0, 0}};
        }

    } else if (fld1 == 4) {
        if (fld2 == 1) {
            return arma::mat{{0, 0, 0, lam1, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {lam1, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, (-lam4 + lam5) / 2.},
                    {0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, (-lam4 + lam5) / 2., 0, 0, 0}};
        } else if (fld2 == 2) {
            return arma::mat{{0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0},
                    {0, 0, 0, lam3, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, lam3, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, (lam4 - lam5) / 2., 0},
                    {(lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, (lam4 - lam5) / 2., 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (fld2 == 3) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, lam1, 0, 0, 0, 0},
                    {0, 0, lam1, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, lam5, 0, 0},
                    {0, 0, 0, 0, lam5, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (fld2 == 4) {
            return arma::mat{{lam1, 0, 0, 0, 0, 0, 0, 0},
                    {0, lam3, 0, 0, 0, 0, 0, 0},
                    {0, 0, lam1, 0, 0, 0, 0, 0},
                    {0, 0, 0, 3 * lam1, 0, 0, 0, 0},
                    {0, 0, 0, 0, (2 * lam3 + 2 * lam4 - 2 * lam5) / 2., 0, 0, 0},
                    {0, 0, 0, 0, 0, (2 * lam3 + 2 * lam4 + 2 * lam5) / 2., 0, 0},
                    {0, 0, 0, 0, 0, 0, lam1, 0},
                    {0, 0, 0, 0, 0, 0, 0, lam3}};
        } else if (fld2 == 5) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, (-lam4 + lam5) / 2.},
                    {0, 0, 0, 0, 0, 0, (lam4 - lam5) / 2., 0},
                    {0, 0, 0, 0, 0, lam5, 0, 0},
                    {0, 0, 0, 0, lam3 + lam4 - lam5, 0, 0, 0},
                    {0, 0, 0, lam3 + lam4 - lam5, 0, 0, 0, 0},
                    {0, 0, lam5, 0, 0, 0, 0, 0},
                    {0, (lam4 - lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {(-lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0, 0}};
        } else if (fld2 == 6) {
            return arma::mat{{0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {(lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, lam5, 0, 0, 0},
                    {0, 0, 0, 0, 0, lam3 + lam4 + lam5, 0, 0},
                    {0, 0, lam5, 0, 0, 0, 0, 0},
                    {0, 0, 0, lam3 + lam4 + lam5, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2.},
                    {0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0}};
        } else if (fld2 == 7) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, (lam4 - lam5) / 2., 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, lam1, 0},
                    {0, (lam4 - lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2.},
                    {0, 0, 0, lam1, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0}};
        } else if (fld2 == 8) {
            return arma::mat{{0, 0, 0, 0, (-lam4 + lam5) / 2., 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, lam3},
                    {(-lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0},
                    {0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0},
                    {0, 0, 0, lam3, 0, 0, 0, 0}};
        }

    } else if (fld1 == 5) {
        if (fld2 == 1) {
            return arma::mat{{0, 0, 0, 0, lam3, 0, 0, 0},
                    {0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0},
                    {0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, (-lam4 + lam5) / 2.},
                    {lam3, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, (-lam4 + lam5) / 2., 0, 0, 0, 0}};
        } else if (fld2 == 2) {
            return arma::mat{{0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, lam2, 0, 0, 0},
                    {(lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, (lam4 - lam5) / 2., 0},
                    {0, lam2, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, (lam4 - lam5) / 2., 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (fld2 == 3) {
            return arma::mat{{0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {(lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, lam3 + lam4 + lam5, 0, 0, 0},
                    {0, 0, 0, 0, 0, lam5, 0, 0},
                    {0, 0, lam3 + lam4 + lam5, 0, 0, 0, 0, 0},
                    {0, 0, 0, lam5, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2.},
                    {0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0}};
        } else if (fld2 == 4) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, (-lam4 + lam5) / 2.},
                    {0, 0, 0, 0, 0, 0, (lam4 - lam5) / 2., 0},
                    {0, 0, 0, 0, 0, lam5, 0, 0},
                    {0, 0, 0, 0, lam3 + lam4 - lam5, 0, 0, 0},
                    {0, 0, 0, lam3 + lam4 - lam5, 0, 0, 0, 0},
                    {0, 0, lam5, 0, 0, 0, 0, 0},
                    {0, (lam4 - lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {(-lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0, 0}};
        } else if (fld2 == 5) {
            return arma::mat{{lam3, 0, 0, 0, 0, 0, 0, 0},
                    {0, lam2, 0, 0, 0, 0, 0, 0},
                    {0, 0, (2 * lam3 + 2 * lam4 + 2 * lam5) / 2., 0, 0, 0, 0, 0},
                    {0, 0, 0, lam3 + lam4 - lam5, 0, 0, 0, 0},
                    {0, 0, 0, 0, 3 * lam2, 0, 0, 0},
                    {0, 0, 0, 0, 0, lam2, 0, 0},
                    {0, 0, 0, 0, 0, 0, lam3, 0},
                    {0, 0, 0, 0, 0, 0, 0, lam2}};
        } else if (fld2 == 6) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, lam5, 0, 0, 0, 0},
                    {0, 0, lam5, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, lam2, 0, 0},
                    {0, 0, 0, 0, lam2, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (fld2 == 7) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, (lam4 - lam5) / 2., 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2.},
                    {0, (lam4 - lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, lam3, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, lam3, 0, 0, 0},
                    {0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0}};
        } else if (fld2 == 8) {
            return arma::mat{{0, 0, 0, (-lam4 + lam5) / 2., 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0},
                    {(-lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, lam2},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, lam2, 0, 0, 0}};
        }

    } else if (fld1 == 6) {
        if (fld2 == 1) {
            return arma::mat{{0, 0, 0, 0, 0, lam3, 0, 0},
                    {0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, (lam4 - lam5) / 2.},
                    {0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {lam3, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, (lam4 - lam5) / 2., 0, 0, 0, 0, 0}};
        } else if (fld2 == 2) {
            return arma::mat{{0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, lam2, 0, 0},
                    {0, 0, 0, 0, 0, 0, (-lam4 + lam5) / 2., 0},
                    {(lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, lam2, 0, 0, 0, 0, 0, 0},
                    {0, 0, (-lam4 + lam5) / 2., 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (fld2 == 3) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, (lam4 - lam5) / 2.},
                    {0, 0, 0, 0, 0, 0, (-lam4 + lam5) / 2., 0},
                    {0, 0, 0, 0, 0, lam3 + lam4 - lam5, 0, 0},
                    {0, 0, 0, 0, lam5, 0, 0, 0},
                    {0, 0, 0, lam5, 0, 0, 0, 0},
                    {0, 0, lam3 + lam4 - lam5, 0, 0, 0, 0, 0},
                    {0, (-lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {(lam4 - lam5) / 2., 0, 0, 0, 0, 0, 0, 0}};
        } else if (fld2 == 4) {
            return arma::mat{{0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {(lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, lam5, 0, 0, 0},
                    {0, 0, 0, 0, 0, lam3 + lam4 + lam5, 0, 0},
                    {0, 0, lam5, 0, 0, 0, 0, 0},
                    {0, 0, 0, lam3 + lam4 + lam5, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2.},
                    {0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0}};
        } else if (fld2 == 5) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, lam5, 0, 0, 0, 0},
                    {0, 0, lam5, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, lam2, 0, 0},
                    {0, 0, 0, 0, lam2, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (fld2 == 6) {
            return arma::mat{{lam3, 0, 0, 0, 0, 0, 0, 0},
                    {0, lam2, 0, 0, 0, 0, 0, 0},
                    {0, 0, (2 * lam3 + 2 * lam4 - 2 * lam5) / 2., 0, 0, 0, 0, 0},
                    {0, 0, 0, (2 * lam3 + 2 * lam4 + 2 * lam5) / 2., 0, 0, 0, 0},
                    {0, 0, 0, 0, lam2, 0, 0, 0},
                    {0, 0, 0, 0, 0, 3 * lam2, 0, 0},
                    {0, 0, 0, 0, 0, 0, lam3, 0},
                    {0, 0, 0, 0, 0, 0, 0, lam2}};
        } else if (fld2 == 7) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, (-lam4 + lam5) / 2., 0, 0, 0, 0, 0},
                    {0, (-lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2.},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, lam3, 0},
                    {0, 0, 0, 0, 0, lam3, 0, 0},
                    {0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0}};
        } else if (fld2 == 8) {
            return arma::mat{{0, 0, (lam4 - lam5) / 2., 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {(lam4 - lam5) / 2., 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, lam2},
                    {0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, lam2, 0, 0}};
        }

    } else if (fld1 == 7) {
        if (fld2 == 1) {
            return arma::mat{{0, 0, 0, 0, 0, 0, lam1, 0},
                    {0, 0, 0, 0, 0, 0, 0, lam5},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {lam1, 0, 0, 0, 0, 0, 0, 0},
                    {0, lam5, 0, 0, 0, 0, 0, 0}};
        } else if (fld2 == 2) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, lam5},
                    {0, 0, 0, 0, 0, 0, lam3 + lam4 - lam5, 0},
                    {0, 0, 0, 0, 0, (-lam4 + lam5) / 2., 0, 0},
                    {0, 0, 0, 0, (lam4 - lam5) / 2., 0, 0, 0},
                    {0, 0, 0, (lam4 - lam5) / 2., 0, 0, 0, 0},
                    {0, 0, (-lam4 + lam5) / 2., 0, 0, 0, 0, 0},
                    {0, lam3 + lam4 - lam5, 0, 0, 0, 0, 0, 0},
                    {lam5, 0, 0, 0, 0, 0, 0, 0}};
        } else if (fld2 == 3) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, (-lam4 + lam5) / 2., 0, 0},
                    {0, 0, 0, 0, 0, 0, lam1, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2.},
                    {0, (-lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {0, 0, lam1, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0}};
        } else if (fld2 == 4) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, (lam4 - lam5) / 2., 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, lam1, 0},
                    {0, (lam4 - lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2.},
                    {0, 0, 0, lam1, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0}};
        } else if (fld2 == 5) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, (lam4 - lam5) / 2., 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2.},
                    {0, (lam4 - lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, lam3, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, lam3, 0, 0, 0},
                    {0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0}};
        } else if (fld2 == 6) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, (-lam4 + lam5) / 2., 0, 0, 0, 0, 0},
                    {0, (-lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2.},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, lam3, 0},
                    {0, 0, 0, 0, 0, lam3, 0, 0},
                    {0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0}};
        } else if (fld2 == 7) {
            return arma::mat{{lam1, 0, 0, 0, 0, 0, 0, 0},
                    {0, (2 * lam3 + 2 * lam4 - 2 * lam5) / 2., 0, 0, 0, 0, 0, 0},
                    {0, 0, lam1, 0, 0, 0, 0, 0},
                    {0, 0, 0, lam1, 0, 0, 0, 0},
                    {0, 0, 0, 0, lam3, 0, 0, 0},
                    {0, 0, 0, 0, 0, lam3, 0, 0},
                    {0, 0, 0, 0, 0, 0, 3 * lam1, 0},
                    {0, 0, 0, 0, 0, 0, 0, (2 * lam3 + 2 * lam4 + 2 * lam5) / 2.}};
        } else if (fld2 == 8) {
            return arma::mat{{0, lam5, 0, 0, 0, 0, 0, 0},
                    {lam5, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0},
                    {0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0},
                    {0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0},
                    {0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, lam3 + lam4 + lam5},
                    {0, 0, 0, 0, 0, 0, lam3 + lam4 + lam5, 0}};
        }

    } else if (fld1 == 8) {
        if (fld2 == 1) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, lam3 + lam4 - lam5},
                    {0, 0, 0, 0, 0, 0, lam5, 0},
                    {0, 0, 0, 0, 0, (lam4 - lam5) / 2., 0, 0},
                    {0, 0, 0, 0, (-lam4 + lam5) / 2., 0, 0, 0},
                    {0, 0, 0, (-lam4 + lam5) / 2., 0, 0, 0, 0},
                    {0, 0, (lam4 - lam5) / 2., 0, 0, 0, 0, 0},
                    {0, lam5, 0, 0, 0, 0, 0, 0},
                    {lam3 + lam4 - lam5, 0, 0, 0, 0, 0, 0, 0}};
        } else if (fld2 == 2) {
            return arma::mat{{0, 0, 0, 0, 0, 0, lam5, 0},
                    {0, 0, 0, 0, 0, 0, 0, lam2},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {lam5, 0, 0, 0, 0, 0, 0, 0},
                    {0, lam2, 0, 0, 0, 0, 0, 0}};
        } else if (fld2 == 3) {
            return arma::mat{{0, 0, 0, 0, 0, (lam4 - lam5) / 2., 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, lam3},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0},
                    {(lam4 - lam5) / 2., 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0},
                    {0, 0, lam3, 0, 0, 0, 0, 0}};
        } else if (fld2 == 4) {
            return arma::mat{{0, 0, 0, 0, (-lam4 + lam5) / 2., 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, lam3},
                    {(-lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0},
                    {0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0},
                    {0, 0, 0, lam3, 0, 0, 0, 0}};
        } else if (fld2 == 5) {
            return arma::mat{{0, 0, 0, (-lam4 + lam5) / 2., 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0},
                    {(-lam4 + lam5) / 2., 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, lam2},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, lam2, 0, 0, 0}};
        } else if (fld2 == 6) {
            return arma::mat{{0, 0, (lam4 - lam5) / 2., 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {(lam4 - lam5) / 2., 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, lam2},
                    {0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, lam2, 0, 0}};
        } else if (fld2 == 7) {
            return arma::mat{{0, lam5, 0, 0, 0, 0, 0, 0},
                    {lam5, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0},
                    {0, 0, 0, 0, 0, (lam4 + lam5) / 2., 0, 0},
                    {0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0, 0},
                    {0, 0, 0, (lam4 + lam5) / 2., 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, lam3 + lam4 + lam5},
                    {0, 0, 0, 0, 0, 0, lam3 + lam4 + lam5, 0}};
        } else if (fld2 == 8) {
            return arma::mat{{(2 * lam3 + 2 * lam4 - 2 * lam5) / 2., 0, 0, 0, 0, 0, 0, 0},
                    {0, lam2, 0, 0, 0, 0, 0, 0},
                    {0, 0, lam3, 0, 0, 0, 0, 0},
                    {0, 0, 0, lam3, 0, 0, 0, 0},
                    {0, 0, 0, 0, lam2, 0, 0, 0},
                    {0, 0, 0, 0, 0, lam2, 0, 0},
                    {0, 0, 0, 0, 0, 0, (2 * lam3 + 2 * lam4 + 2 * lam5) / 2., 0},
                    {0, 0, 0, 0, 0, 0, 0, 3 * lam2}};
        }

    }
}

/**
 * Compute the second derivative of the scalar
 * mass matrix wrt  fields.
 * @param fields THDM fields
 * @param params THDM params
 * @param fld1 first field to take derivative wrt.
 * @param fld2 second field to take derivative wrt.
 * @return second derivative of scalar mass matrix.
 */
arma::mat scalar_squared_mass_matrix_deriv_par(Fields<double> &fields,
                                               Parameters<double> &, int par) {

    double r1 = fields.r1;
    double r2 = fields.r2;
    double c1 = fields.c1;
    double c2 = fields.c2;
    double c3 = fields.c3;
    double c4 = fields.c4;
    double i1 = fields.i1;
    double i2 = fields.i2;

    assert(1 <= par && par <= 8);

    if (par == 1) {
        return arma::mat{{1, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 1, 0, 0, 0, 0, 0},
                {0, 0, 0, 1, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 1, 0},
                {0, 0, 0, 0, 0, 0, 0, 0}};
    } else if (par == 2) {
        return arma::mat{{0, -1, 0, 0, 0, 0, 0, 0},
                {-1, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, -1, 0, 0, 0},
                {0, 0, 0, 0, 0, -1, 0, 0},
                {0, 0, -1, 0, 0, 0, 0, 0},
                {0, 0, 0, -1, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, -1},
                {0, 0, 0, 0, 0, 0, -1, 0}};
    } else if (par == 3) {
        return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                {0, 1, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 1, 0, 0, 0},
                {0, 0, 0, 0, 0, 1, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 1}};
    } else if (par == 4) {
        return arma::mat{{(pow(c1, 2) + pow(c2, 2) + pow(i1, 2) + 3 * pow(r1, 2)) / 2., 0,
                c1 * r1, c2 * r1, 0, 0, i1 * r1, 0},
                {0, 0, 0, 0, 0, 0, 0, 0},
                {c1 * r1, 0, (3 * pow(c1, 2) + pow(c2, 2) + pow(i1, 2) +
                        pow(r1, 2)) / 2., c1 * c2, 0, 0, c1 * i1, 0},
                {c2 * r1, 0, c1 * c2, (pow(c1, 2) + 3 * pow(c2, 2) + pow(i1, 2) +
                        pow(r1, 2)) / 2., 0, 0, c2 * i1, 0},
                {0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0},
                {i1 * r1, 0, c1 * i1, c2 * i1, 0, 0, (pow(c1, 2) + pow(c2, 2) +
                        3 * pow(i1, 2) + pow(r1, 2)) / 2., 0},
                {0, 0, 0, 0, 0, 0, 0, 0}};
    } else if (par == 5) {
        return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                {0, (pow(c3, 2) + pow(c4, 2) + pow(i2, 2) + 3 * pow(r2, 2)) / 2.,
                        0, 0, c3 * r2, c4 * r2, 0, i2 * r2},
                {0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0},
                {0, c3 * r2, 0, 0, (3 * pow(c3, 2) + pow(c4, 2) + pow(i2, 2) +
                        pow(r2, 2)) / 2., c3 * c4, 0, c3 * i2},
                {0, c4 * r2, 0, 0, c3 * c4, (pow(c3, 2) + 3 * pow(c4, 2) + pow(i2, 2)
                        + pow(r2, 2)) / 2., 0, c4 * i2},
                {0, 0, 0, 0, 0, 0, 0, 0},
                {0, i2 * r2, 0, 0, c3 * i2, c4 * i2, 0, (pow(c3, 2) + pow(c4, 2) +
                        3 * pow(i2, 2) + pow(r2, 2)) / 2.}};
    } else if (par == 6) {
        return arma::mat{{(pow(c3, 2) + pow(c4, 2) + pow(i2, 2) + pow(r2, 2)) / 2., r1 * r2,
                0, 0, c3 * r1, c4 * r1, 0, i2 * r1},
                {r1 * r2, (pow(c1, 2) + pow(c2, 2) + pow(i1, 2) + pow(r1, 2)) / 2.,
                        c1 * r2, c2 * r2, 0, 0, i1 * r2, 0},
                {0, c1 * r2, (pow(c3, 2) + pow(c4, 2) + pow(i2, 2) +
                        pow(r2, 2)) / 2., 0, c1 * c3, c1 * c4, 0, c1 * i2},
                {0, c2 * r2, 0, (pow(c3, 2) + pow(c4, 2) + pow(i2, 2) +
                        pow(r2, 2)) / 2., c2 * c3, c2 * c4, 0, c2 * i2},
                {c3 * r1, 0, c1 * c3, c2 * c3, (pow(c1, 2) + pow(c2, 2) + pow(i1, 2) +
                        pow(r1, 2)) / 2., 0, c3 * i1, 0},
                {c4 * r1, 0, c1 * c4, c2 * c4, 0, (pow(c1, 2) + pow(c2, 2) +
                        pow(i1, 2) + pow(r1, 2)) / 2., c4 * i1, 0},
                {0, i1 * r2, 0, 0, c3 * i1, c4 * i1, (pow(c3, 2) + pow(c4, 2) +
                        pow(i2, 2) + pow(r2, 2)) / 2., i1 * i2},
                {i2 * r1, 0, c1 * i2, c2 * i2, 0, 0, i1 * i2, (pow(c1, 2) + pow(c2, 2) +
                        pow(i1, 2) + pow(r1, 2)) / 2.}};
    } else if (par == 7) {
        return arma::mat{{(pow(i2, 2) + pow(r2, 2)) / 2., (c1 * c3 + c2 * c4 + 2 * r1 * r2) / 2.,
                (c4 * i2 + c3 * r2) / 2., -(c3 * i2) / 2. + (c4 * r2) / 2., (-(c2 * i2) + c1 * r2) / 2.,
                (c1 * i2 + c2 * r2) / 2., 0, (-(c2 * c3) + c1 * c4 + 2 * i2 * r1) / 2.},
                {(c1 * c3 + c2 * c4 + 2 * r1 * r2) / 2., (pow(i1, 2) + pow(r1, 2)) / 2.,
                        (-(c4 * i1) + c3 * r1) / 2., (c3 * i1 + c4 * r1) / 2., (c2 * i1 + c1 * r1) / 2.,
                        -(c1 * i1) / 2. + (c2 * r1) / 2., (c2 * c3 - c1 * c4 + 2 * i1 * r2) / 2., 0},
                {(c4 * i2 + c3 * r2) / 2., (-(c4 * i1) + c3 * r1) / 2., (pow(c3, 2) +
                        pow(c4, 2)) / 2., 0, (2 * c1 * c3 + i1 * i2 + r1 * r2) / 2., c1 * c4 + (i2 * r1 -
                        i1 * r2) / 2., (c3 * i2 - c4 * r2) / 2., (c3 * i1 + c4 * r1) / 2.},
                {-(c3 * i2) / 2. + (c4 * r2) / 2., (c3 * i1 + c4 * r1) / 2., 0, (pow(c3, 2) +
                        pow(c4, 2)) / 2., c2 * c3 + (-(i2 * r1) + i1 * r2) / 2., (2 * c2 * c4 + i1 * i2 +
                        r1 * r2) / 2., (c4 * i2 + c3 * r2) / 2., (c4 * i1 - c3 * r1) / 2.},
                {(-(c2 * i2) + c1 * r2) / 2., (c2 * i1 + c1 * r1) / 2., (2 * c1 * c3 + i1 * i2 +
                        r1 * r2) / 2., c2 * c3 + (-(i2 * r1) + i1 * r2) / 2., (pow(c1, 2) +
                        pow(c2, 2)) / 2., 0, (c1 * i2 + c2 * r2) / 2., (c1 * i1 - c2 * r1) / 2.},
                {(c1 * i2 + c2 * r2) / 2., -(c1 * i1) / 2. + (c2 * r1) / 2., c1 * c4 + (i2 * r1 -
                        i1 * r2) / 2., (2 * c2 * c4 + i1 * i2 + r1 * r2) / 2., 0, (pow(c1, 2) +
                        pow(c2, 2)) / 2., (c2 * i2 - c1 * r2) / 2., (c2 * i1 + c1 * r1) / 2.},
                {0, (c2 * c3 - c1 * c4 + 2 * i1 * r2) / 2., (c3 * i2 - c4 * r2) / 2., (c4 * i2 +
                        c3 * r2) / 2., (c1 * i2 + c2 * r2) / 2., (c2 * i2 - c1 * r2) / 2., (pow(i2, 2) +
                        pow(r2, 2)) / 2., (c1 * c3 + c2 * c4 + 2 * i1 * i2) / 2.},
                {(-(c2 * c3) + c1 * c4 + 2 * i2 * r1) / 2., 0, (c3 * i1 + c4 * r1) / 2., (c4 * i1 -
                        c3 * r1) / 2., (c1 * i1 - c2 * r1) / 2., (c2 * i1 + c1 * r1) / 2., (c1 * c3 + c2 * c4 +
                        2 * i1 * i2) / 2., (pow(i1, 2) + pow(r1, 2)) / 2.}};
    } else if (par == 8) {
        return arma::mat{{(-pow(i2, 2) + pow(r2, 2)) / 2., (c1 * c3 + c2 * c4 + 2 * i1 * i2 +
                2 * r1 * r2) / 2., (-(c4 * i2) + c3 * r2) / 2., (c3 * i2) / 2. + (c4 * r2) / 2., (c2 * i2 +
                c1 * r2) / 2., (-(c1 * i2) + c2 * r2) / 2., i2 * r2, (c2 * c3 - c1 * c4 - 2 * i2 * r1 +
                2 * i1 * r2) / 2.},
                {(c1 * c3 + c2 * c4 + 2 * i1 * i2 + 2 * r1 * r2) / 2., (-pow(i1, 2) +
                        pow(r1, 2)) / 2., (c4 * i1 + c3 * r1) / 2., (-(c3 * i1) + c4 * r1) / 2., (-(c2 * i1)
                        + c1 * r1) / 2., (c1 * i1) / 2. + (c2 * r1) / 2., (-(c2 * c3) + c1 * c4 + 2 * i2 * r1 -
                        2 * i1 * r2) / 2., i1 * r1},
                {(-(c4 * i2) + c3 * r2) / 2., (c4 * i1 + c3 * r1) / 2., (pow(c3, 2) -
                        pow(c4, 2)) / 2., c3 * c4, (2 * c1 * c3 + 2 * c2 * c4 + i1 * i2 + r1 * r2) / 2., c2 * c3
                        - c1 * c4 + (-(i2 * r1) + i1 * r2) / 2., (c3 * i2 + c4 * r2) / 2., (c3 * i1 -
                        c4 * r1) / 2.},
                {(c3 * i2) / 2. + (c4 * r2) / 2., (-(c3 * i1) + c4 * r1) / 2., c3 * c4,
                        (-pow(c3, 2) + pow(c4, 2)) / 2., -(c2 * c3) + c1 * c4 + (i2 * r1 -
                        i1 * r2) / 2., (2 * c1 * c3 + 2 * c2 * c4 + i1 * i2 + r1 * r2) / 2., (c4 * i2 -
                        c3 * r2) / 2., (c4 * i1 + c3 * r1) / 2.},
                {(c2 * i2 + c1 * r2) / 2., (-(c2 * i1) + c1 * r1) / 2., (2 * c1 * c3 + 2 * c2 * c4 +
                        i1 * i2 + r1 * r2) / 2., -(c2 * c3) + c1 * c4 + (i2 * r1 - i1 * r2) / 2.,
                        (pow(c1, 2) - pow(c2, 2)) / 2., c1 * c2, (c1 * i2 - c2 * r2) / 2., (c1 * i1 +
                        c2 * r1) / 2.},
                {(-(c1 * i2) + c2 * r2) / 2., (c1 * i1) / 2. + (c2 * r1) / 2., c2 * c3 - c1 * c4 +
                        (-(i2 * r1) + i1 * r2) / 2., (2 * c1 * c3 + 2 * c2 * c4 + i1 * i2 + r1 * r2) / 2., c1 * c2,
                        (-pow(c1, 2) + pow(c2, 2)) / 2., (c2 * i2 + c1 * r2) / 2., (c2 * i1 -
                        c1 * r1) / 2.},
                {i2 * r2, (-(c2 * c3) + c1 * c4 + 2 * i2 * r1 - 2 * i1 * r2) / 2., (c3 * i2 +
                        c4 * r2) / 2., (c4 * i2 - c3 * r2) / 2., (c1 * i2 - c2 * r2) / 2., (c2 * i2 +
                        c1 * r2) / 2., (pow(i2, 2) - pow(r2, 2)) / 2., (c1 * c3 + c2 * c4 + 2 * i1 * i2 +
                        2 * r1 * r2) / 2.},
                {(c2 * c3 - c1 * c4 - 2 * i2 * r1 + 2 * i1 * r2) / 2., i1 * r1, (c3 * i1 - c4 * r1) / 2.,
                        (c4 * i1 + c3 * r1) / 2., (c1 * i1 + c2 * r1) / 2., (c2 * i1 - c1 * r1) / 2., (c1 * c3 +
                        c2 * c4 + 2 * i1 * i2 + 2 * r1 * r2) / 2., (pow(i1, 2) - pow(r1, 2)) / 2.}};
    }
}

/**
 * Compute the second derivative of the scalar
 * mass matrix wrt  fields.
 * @param fields THDM fields
 * @param params THDM params
 * @param fld1 first field to take derivative wrt.
 * @param fld2 second field to take derivative wrt.
 * @return second derivative of scalar mass matrix.
 */
arma::mat scalar_squared_mass_matrix_deriv_fld_par(Fields<double> &fields,
                                                   Parameters<double> &, int fld, int par) {

    double r1 = fields.r1;
    double r2 = fields.r2;
    double c1 = fields.c1;
    double c2 = fields.c2;
    double c3 = fields.c3;
    double c4 = fields.c4;
    double i1 = fields.i1;
    double i2 = fields.i2;

    assert(1 <= fld && fld <= 8);
    assert(1 <= par && par <= 8);

    if (fld == 1) {
        if (par == 1) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 2) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 3) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 4) {
            return arma::mat{{3 * r1, 0, c1, c2, 0, 0, i1, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {c1, 0, r1, 0, 0, 0, 0, 0},
                    {c2, 0, 0, r1, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {i1, 0, 0, 0, 0, 0, r1, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 5) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 6) {
            return arma::mat{{0, r2, 0, 0, c3, c4, 0, i2},
                    {r2, r1, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {c3, 0, 0, 0, r1, 0, 0, 0},
                    {c4, 0, 0, 0, 0, r1, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {i2, 0, 0, 0, 0, 0, 0, r1}};
        } else if (par == 7) {
            return arma::mat{{0, r2, 0, 0, 0, 0, 0, i2},
                    {r2, r1, c3 / 2., c4 / 2., c1 / 2., c2 / 2., 0, 0},
                    {0, c3 / 2., 0, 0, r2 / 2., i2 / 2., 0, c4 / 2.},
                    {0, c4 / 2., 0, 0, -i2 / 2., r2 / 2., 0, -c3 / 2.},
                    {0, c1 / 2., r2 / 2., -i2 / 2., 0, 0, 0, -c2 / 2.},
                    {0, c2 / 2., i2 / 2., r2 / 2., 0, 0, 0, c1 / 2.},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {i2, 0, c4 / 2., -c3 / 2., -c2 / 2., c1 / 2., 0, r1}};
        } else if (par == 8) {
            return arma::mat{{0, r2, 0, 0, 0, 0, 0, -i2},
                    {r2, r1, c3 / 2., c4 / 2., c1 / 2., c2 / 2., i2, i1},
                    {0, c3 / 2., 0, 0, r2 / 2., -i2 / 2., 0, -c4 / 2.},
                    {0, c4 / 2., 0, 0, i2 / 2., r2 / 2., 0, c3 / 2.},
                    {0, c1 / 2., r2 / 2., i2 / 2., 0, 0, 0, c2 / 2.},
                    {0, c2 / 2., -i2 / 2., r2 / 2., 0, 0, 0, -c1 / 2.},
                    {0, i2, 0, 0, 0, 0, 0, r2},
                    {-i2, i1, -c4 / 2., c3 / 2., c2 / 2., -c1 / 2., r2, -r1}};
        }

    } else if (fld == 2) {
        if (par == 1) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 2) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 3) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 4) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 5) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 3 * r2, 0, 0, c3, c4, 0, i2},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, c3, 0, 0, r2, 0, 0, 0},
                    {0, c4, 0, 0, 0, r2, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, i2, 0, 0, 0, 0, 0, r2}};
        } else if (par == 6) {
            return arma::mat{{r2, r1, 0, 0, 0, 0, 0, 0},
                    {r1, 0, c1, c2, 0, 0, i1, 0},
                    {0, c1, r2, 0, 0, 0, 0, 0},
                    {0, c2, 0, r2, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, i1, 0, 0, 0, 0, r2, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 7) {
            return arma::mat{{r2, r1, c3 / 2., c4 / 2., c1 / 2., c2 / 2., 0, 0},
                    {r1, 0, 0, 0, 0, 0, i1, 0},
                    {c3 / 2., 0, 0, 0, r1 / 2., -i1 / 2., -c4 / 2., 0},
                    {c4 / 2., 0, 0, 0, i1 / 2., r1 / 2., c3 / 2., 0},
                    {c1 / 2., 0, r1 / 2., i1 / 2., 0, 0, c2 / 2., 0},
                    {c2 / 2., 0, -i1 / 2., r1 / 2., 0, 0, -c1 / 2., 0},
                    {0, i1, -c4 / 2., c3 / 2., c2 / 2., -c1 / 2., r2, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 8) {
            return arma::mat{{r2, r1, c3 / 2., c4 / 2., c1 / 2., c2 / 2., i2, i1},
                    {r1, 0, 0, 0, 0, 0, -i1, 0},
                    {c3 / 2., 0, 0, 0, r1 / 2., i1 / 2., c4 / 2., 0},
                    {c4 / 2., 0, 0, 0, -i1 / 2., r1 / 2., -c3 / 2., 0},
                    {c1 / 2., 0, r1 / 2., -i1 / 2., 0, 0, -c2 / 2., 0},
                    {c2 / 2., 0, i1 / 2., r1 / 2., 0, 0, c1 / 2., 0},
                    {i2, -i1, c4 / 2., -c3 / 2., -c2 / 2., c1 / 2., -r2, r1},
                    {i1, 0, 0, 0, 0, 0, r1, 0}};
        }

    } else if (fld == 3) {
        if (par == 1) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 2) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 3) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 4) {
            return arma::mat{{c1, 0, r1, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {r1, 0, 3 * c1, c2, 0, 0, i1, 0},
                    {0, 0, c2, c1, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, i1, 0, 0, 0, c1, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 5) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 6) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, c1, r2, 0, 0, 0, 0, 0},
                    {0, r2, 0, 0, c3, c4, 0, i2},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, c3, 0, c1, 0, 0, 0},
                    {0, 0, c4, 0, 0, c1, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, i2, 0, 0, 0, 0, c1}};
        } else if (par == 7) {
            return arma::mat{{0, c3 / 2., 0, 0, r2 / 2., i2 / 2., 0, c4 / 2.},
                    {c3 / 2., 0, 0, 0, r1 / 2., -i1 / 2., -c4 / 2., 0},
                    {0, 0, 0, 0, c3, c4, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {r2 / 2., r1 / 2., c3, 0, c1, 0, i2 / 2., i1 / 2.},
                    {i2 / 2., -i1 / 2., c4, 0, 0, c1, -r2 / 2., r1 / 2.},
                    {0, -c4 / 2., 0, 0, i2 / 2., -r2 / 2., 0, c3 / 2.},
                    {c4 / 2., 0, 0, 0, i1 / 2., r1 / 2., c3 / 2., 0}};
        } else if (par == 8) {
            return arma::mat{{0, c3 / 2., 0, 0, r2 / 2., -i2 / 2., 0, -c4 / 2.},
                    {c3 / 2., 0, 0, 0, r1 / 2., i1 / 2., c4 / 2., 0},
                    {0, 0, 0, 0, c3, -c4, 0, 0},
                    {0, 0, 0, 0, c4, c3, 0, 0},
                    {r2 / 2., r1 / 2., c3, c4, c1, c2, i2 / 2., i1 / 2.},
                    {-i2 / 2., i1 / 2., -c4, c3, c2, -c1, r2 / 2., -r1 / 2.},
                    {0, c4 / 2., 0, 0, i2 / 2., r2 / 2., 0, c3 / 2.},
                    {-c4 / 2., 0, 0, 0, i1 / 2., -r1 / 2., c3 / 2., 0}};
        }

    } else if (fld == 4) {
        if (par == 1) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 2) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 3) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 4) {
            return arma::mat{{c2, 0, 0, r1, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, c2, c1, 0, 0, 0, 0},
                    {r1, 0, c1, 3 * c2, 0, 0, i1, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, i1, 0, 0, c2, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 5) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 6) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, c2, 0, r2, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, r2, 0, 0, c3, c4, 0, i2},
                    {0, 0, 0, c3, c2, 0, 0, 0},
                    {0, 0, 0, c4, 0, c2, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, i2, 0, 0, 0, c2}};
        } else if (par == 7) {
            return arma::mat{{0, c4 / 2., 0, 0, -i2 / 2., r2 / 2., 0, -c3 / 2.},
                    {c4 / 2., 0, 0, 0, i1 / 2., r1 / 2., c3 / 2., 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, c3, c4, 0, 0},
                    {-i2 / 2., i1 / 2., 0, c3, c2, 0, r2 / 2., -r1 / 2.},
                    {r2 / 2., r1 / 2., 0, c4, 0, c2, i2 / 2., i1 / 2.},
                    {0, c3 / 2., 0, 0, r2 / 2., i2 / 2., 0, c4 / 2.},
                    {-c3 / 2., 0, 0, 0, -r1 / 2., i1 / 2., c4 / 2., 0}};
        } else if (par == 8) {
            return arma::mat{{0, c4 / 2., 0, 0, i2 / 2., r2 / 2., 0, c3 / 2.},
                    {c4 / 2., 0, 0, 0, -i1 / 2., r1 / 2., -c3 / 2., 0},
                    {0, 0, 0, 0, c4, c3, 0, 0},
                    {0, 0, 0, 0, -c3, c4, 0, 0},
                    {i2 / 2., -i1 / 2., c4, -c3, -c2, c1, -r2 / 2., r1 / 2.},
                    {r2 / 2., r1 / 2., c3, c4, c1, c2, i2 / 2., i1 / 2.},
                    {0, -c3 / 2., 0, 0, -r2 / 2., i2 / 2., 0, c4 / 2.},
                    {c3 / 2., 0, 0, 0, r1 / 2., i1 / 2., c4 / 2., 0}};
        }

    } else if (fld == 5) {
        if (par == 1) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 2) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 3) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 4) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 5) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, c3, 0, 0, r2, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, r2, 0, 0, 3 * c3, c4, 0, i2},
                    {0, 0, 0, 0, c4, c3, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, i2, 0, 0, c3}};
        } else if (par == 6) {
            return arma::mat{{c3, 0, 0, 0, r1, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, c3, 0, c1, 0, 0, 0},
                    {0, 0, 0, c3, c2, 0, 0, 0},
                    {r1, 0, c1, c2, 0, 0, i1, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, i1, 0, c3, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 7) {
            return arma::mat{{0, c1 / 2., r2 / 2., -i2 / 2., 0, 0, 0, -c2 / 2.},
                    {c1 / 2., 0, r1 / 2., i1 / 2., 0, 0, c2 / 2., 0},
                    {r2 / 2., r1 / 2., c3, 0, c1, 0, i2 / 2., i1 / 2.},
                    {-i2 / 2., i1 / 2., 0, c3, c2, 0, r2 / 2., -r1 / 2.},
                    {0, 0, c1, c2, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, c2 / 2., i2 / 2., r2 / 2., 0, 0, 0, c1 / 2.},
                    {-c2 / 2., 0, i1 / 2., -r1 / 2., 0, 0, c1 / 2., 0}};
        } else if (par == 8) {
            return arma::mat{{0, c1 / 2., r2 / 2., i2 / 2., 0, 0, 0, c2 / 2.},
                    {c1 / 2., 0, r1 / 2., -i1 / 2., 0, 0, -c2 / 2., 0},
                    {r2 / 2., r1 / 2., c3, c4, c1, c2, i2 / 2., i1 / 2.},
                    {i2 / 2., -i1 / 2., c4, -c3, -c2, c1, -r2 / 2., r1 / 2.},
                    {0, 0, c1, -c2, 0, 0, 0, 0},
                    {0, 0, c2, c1, 0, 0, 0, 0},
                    {0, -c2 / 2., i2 / 2., -r2 / 2., 0, 0, 0, c1 / 2.},
                    {c2 / 2., 0, i1 / 2., r1 / 2., 0, 0, c1 / 2., 0}};
        }

    } else if (fld == 6) {
        if (par == 1) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 2) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 3) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 4) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 5) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, c4, 0, 0, 0, r2, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, c4, c3, 0, 0},
                    {0, r2, 0, 0, c3, 3 * c4, 0, i2},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, i2, 0, c4}};
        } else if (par == 6) {
            return arma::mat{{c4, 0, 0, 0, 0, r1, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, c4, 0, 0, c1, 0, 0},
                    {0, 0, 0, c4, 0, c2, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {r1, 0, c1, c2, 0, 0, i1, 0},
                    {0, 0, 0, 0, 0, i1, c4, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 7) {
            return arma::mat{{0, c2 / 2., i2 / 2., r2 / 2., 0, 0, 0, c1 / 2.},
                    {c2 / 2., 0, -i1 / 2., r1 / 2., 0, 0, -c1 / 2., 0},
                    {i2 / 2., -i1 / 2., c4, 0, 0, c1, -r2 / 2., r1 / 2.},
                    {r2 / 2., r1 / 2., 0, c4, 0, c2, i2 / 2., i1 / 2.},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, c1, c2, 0, 0, 0, 0},
                    {0, -c1 / 2., -r2 / 2., i2 / 2., 0, 0, 0, c2 / 2.},
                    {c1 / 2., 0, r1 / 2., i1 / 2., 0, 0, c2 / 2., 0}};
        } else if (par == 8) {
            return arma::mat{{0, c2 / 2., -i2 / 2., r2 / 2., 0, 0, 0, -c1 / 2.},
                    {c2 / 2., 0, i1 / 2., r1 / 2., 0, 0, c1 / 2., 0},
                    {-i2 / 2., i1 / 2., -c4, c3, c2, -c1, r2 / 2., -r1 / 2.},
                    {r2 / 2., r1 / 2., c3, c4, c1, c2, i2 / 2., i1 / 2.},
                    {0, 0, c2, c1, 0, 0, 0, 0},
                    {0, 0, -c1, c2, 0, 0, 0, 0},
                    {0, c1 / 2., r2 / 2., i2 / 2., 0, 0, 0, c2 / 2.},
                    {-c1 / 2., 0, -r1 / 2., i1 / 2., 0, 0, c2 / 2., 0}};
        }

    } else if (fld == 7) {
        if (par == 1) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 2) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 3) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 4) {
            return arma::mat{{i1, 0, 0, 0, 0, 0, r1, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, i1, 0, 0, 0, c1, 0},
                    {0, 0, 0, i1, 0, 0, c2, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {r1, 0, c1, c2, 0, 0, 3 * i1, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 5) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 6) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, i1, 0, 0, 0, 0, r2, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, i1, 0, c3, 0},
                    {0, 0, 0, 0, 0, i1, c4, 0},
                    {0, r2, 0, 0, c3, c4, 0, i2},
                    {0, 0, 0, 0, 0, 0, i2, i1}};
        } else if (par == 7) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, i1, -c4 / 2., c3 / 2., c2 / 2., -c1 / 2., r2, 0},
                    {0, -c4 / 2., 0, 0, i2 / 2., -r2 / 2., 0, c3 / 2.},
                    {0, c3 / 2., 0, 0, r2 / 2., i2 / 2., 0, c4 / 2.},
                    {0, c2 / 2., i2 / 2., r2 / 2., 0, 0, 0, c1 / 2.},
                    {0, -c1 / 2., -r2 / 2., i2 / 2., 0, 0, 0, c2 / 2.},
                    {0, r2, 0, 0, 0, 0, 0, i2},
                    {0, 0, c3 / 2., c4 / 2., c1 / 2., c2 / 2., i2, i1}};
        } else if (par == 8) {
            return arma::mat{{0, i2, 0, 0, 0, 0, 0, r2},
                    {i2, -i1, c4 / 2., -c3 / 2., -c2 / 2., c1 / 2., -r2, r1},
                    {0, c4 / 2., 0, 0, i2 / 2., r2 / 2., 0, c3 / 2.},
                    {0, -c3 / 2., 0, 0, -r2 / 2., i2 / 2., 0, c4 / 2.},
                    {0, -c2 / 2., i2 / 2., -r2 / 2., 0, 0, 0, c1 / 2.},
                    {0, c1 / 2., r2 / 2., i2 / 2., 0, 0, 0, c2 / 2.},
                    {0, -r2, 0, 0, 0, 0, 0, i2},
                    {r2, r1, c3 / 2., c4 / 2., c1 / 2., c2 / 2., i2, i1}};
        }

    } else if (fld == 8) {
        if (par == 1) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 2) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 3) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 4) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0}};
        } else if (par == 5) {
            return arma::mat{{0, 0, 0, 0, 0, 0, 0, 0},
                    {0, i2, 0, 0, 0, 0, 0, r2},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, i2, 0, 0, c3},
                    {0, 0, 0, 0, 0, i2, 0, c4},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, r2, 0, 0, c3, c4, 0, 3 * i2}};
        } else if (par == 6) {
            return arma::mat{{i2, 0, 0, 0, 0, 0, 0, r1},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, i2, 0, 0, 0, 0, c1},
                    {0, 0, 0, i2, 0, 0, 0, c2},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, i2, i1},
                    {r1, 0, c1, c2, 0, 0, i1, 0}};
        } else if (par == 7) {
            return arma::mat{{i2, 0, c4 / 2., -c3 / 2., -c2 / 2., c1 / 2., 0, r1},
                    {0, 0, 0, 0, 0, 0, 0, 0},
                    {c4 / 2., 0, 0, 0, i1 / 2., r1 / 2., c3 / 2., 0},
                    {-c3 / 2., 0, 0, 0, -r1 / 2., i1 / 2., c4 / 2., 0},
                    {-c2 / 2., 0, i1 / 2., -r1 / 2., 0, 0, c1 / 2., 0},
                    {c1 / 2., 0, r1 / 2., i1 / 2., 0, 0, c2 / 2., 0},
                    {0, 0, c3 / 2., c4 / 2., c1 / 2., c2 / 2., i2, i1},
                    {r1, 0, 0, 0, 0, 0, i1, 0}};
        } else if (par == 8) {
            return arma::mat{{-i2, i1, -c4 / 2., c3 / 2., c2 / 2., -c1 / 2., r2, -r1},
                    {i1, 0, 0, 0, 0, 0, r1, 0},
                    {-c4 / 2., 0, 0, 0, i1 / 2., -r1 / 2., c3 / 2., 0},
                    {c3 / 2., 0, 0, 0, r1 / 2., i1 / 2., c4 / 2., 0},
                    {c2 / 2., 0, i1 / 2., r1 / 2., 0, 0, c1 / 2., 0},
                    {-c1 / 2., 0, -r1 / 2., i1 / 2., 0, 0, c2 / 2., 0},
                    {r2, r1, c3 / 2., c4 / 2., c1 / 2., c2 / 2., i2, i1},
                    {-r1, 0, 0, 0, 0, 0, i1, 0}};
        }

    }
}

/**
 * Computes the scalar squared masses for a given set of fields and
 * parameters.
 * @tparam T type of the fields and params: should be a double or dual.
 * @param fields Fields<T> containing 2hdm fields.
 * @param params Parameters<T> containing 2hdm parameters.
 * @return boost vector<T> containing the squared scalar masses.
 */
std::vector<double> scalar_squared_masses(Fields<double> &fields, Parameters<double> &params) {
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
 * @param fld integrer index of the field to take derivative of.
 * @return boost vector<std::pair<double,double>> containing the squared
 * scalar masses and derivatives.
 */

std::vector<std::tuple<double, double>>
scalar_squared_masses_deriv_fld(Fields<double> &fields,
                                Parameters<double> &params, int fld) {
    if (fld < 1 || fld > 8) {
        throw InvalidFldIndexException();
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
 * @param fld1 integrer index of the first field to take derivative of.
 * @param fld2 integrer index of the second field to take derivative of.
 * @return vector of tuples containing the eigenvalues, the derivative
 * of the evals wrt first field, deriv of evals wrt second field and
 * mixed derivatives.
 */

std::vector<std::tuple<double, double, double, double>>
scalar_squared_masses_deriv_fld(Fields<double> &fields,
                                Parameters<double> &params, int fld1,
                                int fld2) {
    if ((fld1 < 1 || fld1 > 8) || (fld2 < 1 || fld2 > 8)) {
        throw InvalidFldIndexException();
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
 * Computes the first derivative of the scalar squared masses wrt a
 * parameter for a given set of fields and parameters.
 * @param fields Fields<double> containing 2hdm fields.
 * @param params Parameters<double> containing 2hdm parameters.
 * @param par integrer index of the parameter to take derivative of.
 * @return boost vector<std::pair<double,double>> containing the squared
 * scalar masses and derivatives.
 */
std::vector<std::tuple<double, double>>
scalar_squared_masses_deriv_par(Fields<double> &fields,
                                Parameters<double> &params, int par) {
    if (par < 1 || par > 8) {
        throw InvalidParIndexException();
    }
    // Make a new set of fields and parameters.
    Fields<Dual<double>> _fields{};
    Parameters<Dual<double>> _params{};
    // Fill up the fields and parameters will dual numbers who's value
    // agrees
    // with old fields and parameters. if i == par - 1, then set that
    // parameters's eps to 1;
    for (int i = 0; i < 8; i++) {
        // Need to do par -1 since par in [1, 8] and i in [0,7]
        _params[i].eps = (i == par - 1) ? 1.0 : 0.0;

        _params[i].val = params[i];
        _fields[i].val = fields[i];
    }
    // Lastly, set mu for params. Not that it really matters
    _params.mu.val = params.mu;

    // Compute the scalar mass matrix:
    auto mat = scalar_squared_mass_matrix(_fields, _params);
    // Compute the eigenvalues and their derivatives:
    auto evals = jacobi(mat);
    // create new vector with derivatives:
    std::vector<std::tuple<double, double>> lam_dlams(8);
    for (size_t i = 0; i < 8; i++) {
        lam_dlams[i] = std::make_tuple(evals[i].val, evals[i].eps);
    }
    return lam_dlams;
}

/**
 * Computes the second derivative of the scalar squared masses wrt
 * parameters for a given set of fields and parameters.
 * @param fields Fields<double> containing 2hdm fields.
 * @param params Parameters<double> containing 2hdm parameters.
 * @param par1 integrer index of the first parameter to take derivative
 * of.
 * @param par2 integrer index of the second parameter to take derivative
 * of.
 * @return boost vector<std::pair<double,double>> containing the squared
 * scalar masses and derivatives.
 */
std::vector<std::tuple<double, double, double, double>>
scalar_squared_masses_deriv_par(Fields<double> &fields,
                                Parameters<double> &params, int par1,
                                int par2) {
    if ((par1 < 1 || par1 > 8) || (par2 < 1 || par2 > 8)) {
        throw InvalidParIndexException();
    }

    Fields<Dual<Dual<double >>> _fields{};
    Parameters<Dual<Dual<double >>> _params{};

    for (int i = 0; i < 8; i++) {
        // Need to do fld -1 since fld in [1, 8] and i in [0,7]
        _params[i].val.eps = (i == par1 - 1) ? 1.0 : 0.0;
        _params[i].eps.val = (i == par2 - 1) ? 1.0 : 0.0;

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
 * @param fld integrer index of the field to take derivative of.
 * @param par integrer index of the parameter to take derivative of.
 * @return boost vector<std::pair<double,double>> containing the squared
 * scalar masses and derivatives.
 */
std::vector<std::tuple<double, double, double, double>>
scalar_squared_masses_deriv_fld_par(Fields<double> &fields,
                                    Parameters<double> &params, int fld,
                                    int par) {
    if ((fld < 1 || fld > 8)) {
        throw InvalidFldIndexException();
    }
    if (par < 1 || par > 8) {
        throw InvalidParIndexException();
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