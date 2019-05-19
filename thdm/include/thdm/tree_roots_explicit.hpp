//
// Created by Logan Morrison on 2019-05-04.
//

#ifndef THDM_TREE_ROOTS_EXPLICIT_HPP
#define THDM_TREE_ROOTS_EXPLICIT_HPP

#include "thdm/extrema_type.hpp"
#include "thdm/fields.hpp"
#include "thdm/parameters.hpp"
#include "thdm/potentials.hpp"
#include "thdm/vacuua.hpp"
#include <Eigen/Dense>
#include <assert.h>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>
#include <tuple>

namespace thdm {

void fill_vacuum(Parameters<double> &params, Vacuum<double> &vac) {
    Fields<double> fields{};
    fields.set_fields(vac);
    vac.potential = potential_tree(fields, params);
    vac.extrema_type = determine_single_extrema_type_tree(params, vac);
}

/**
 * Compute the roots of a univaraiate polynomial.
 *
 * To compute the roots, we construct a companion matrix
 * `C` such that det(C-x * I) = p(x). If the polynomial is
 * given by:
 *      p(x) = a_d x^d + a_{d-1}x^{d-1} +...+a_1 x + a_0
 * then, the companion matrix is:
 *      C = {{0, 0, ..., 0, -a_0/a_d},
 *           {1, 0, ..., 0, -a_1/a_d},
 *           .
 *           .
 *           .
 *           {0, 0, ..., 1, -a_{d-1}/a_{d}}
 *
 * @param coeffs array of the coefficents of the polynomial
 * starting with a_d and ending with a_0.
 * @return roots of the polynomial.
 */
Eigen::VectorXcd univariate_polynomial_root_finder(std::vector<double> coeffs) {
    // Make sure that the leading order coefficient is non-zero.
    assert(coeffs[0] != 0.0);
    // The matrix will be the order of the polynomial, which
    // is one less that the size of the coefficient list.
    int d = coeffs.size() - 1;
    Eigen::MatrixXd companion_matrix(d, d);

    // Fill the companion matrix with zeros.
    // TODO: Check initialization of Eigen matrices
    for (int row = 0; row < d; row++) {
        for (int col = 0; col < d; col++) {
            companion_matrix(row, col) = 0.0;
        }
    }
    // Fill last column
    for (int row = 0; row < d; row++) {
        companion_matrix(row, d - 1) = -coeffs[d - row] / coeffs[0];
    }
    // Fill in ones
    for (int row = 1; row < d; row++) {
        companion_matrix(row, row - 1) = 1.0;
    }
    Eigen::EigenSolver<Eigen::MatrixXd> eigensolver(companion_matrix);
    Eigen::VectorXcd evals = eigensolver.eigenvalues();
    return evals;
}

// Groebner Basis functions
std::vector<double> get_groebner_basis_coeffs_c1(Parameters<double> &params) {
    double m112 = params.m112;
    double m122 = params.m122;
    double m222 = params.m222;
    double lam1 = params.lam1;
    double lam2 = params.lam2;
    double lam3 = params.lam3;
    double lam4 = params.lam4;
    double lam5 = params.lam5;

    std::vector<double> g1{
            lam1 * (lam1 * lam2 - pow(lam3, 2)) * pow(lam4 + lam5, 2) * pow(m122, 2) *
                    (-(lam3 * m112) + lam1 * m222),
            0,
            -2 * pow(m122, 2) *
                    (2 * lam1 * lam2 * lam3 * pow(lam4 + lam5, 2) * pow(m112, 2) -
                            pow(lam3, 3) * pow(lam4 + lam5, 2) * pow(m112, 2) +
                            pow(lam1, 3) * pow(lam2, 2) * pow(m122, 2) +
                            lam1 * pow(lam3, 4) * pow(m122, 2) +
                            pow(lam1, 2) * (-2 * lam2 * pow(lam3, 2) * pow(m122, 2) -
                                    2 * lam2 * pow(lam4 + lam5, 2) * m112 * m222 +
                                    lam3 * pow(lam4 + lam5, 2) * pow(m222, 2))),
            0,
            -4 * m112 * pow(m122, 2) *
                    (pow(lam1, 2) * pow(lam2, 2) * pow(m122, 2) +
                            lam2 * lam3 *
                                    (pow(lam4 + lam5, 2) * pow(m112, 2) -
                                            2 * lam1 * lam3 * pow(m122, 2)) -
                            lam1 * lam2 * pow(lam4 + lam5, 2) * m112 * m222 +
                            lam3 * (pow(lam3, 3) * pow(m122, 2) -
                                    lam3 * pow(lam4 + lam5, 2) * m112 * m222 +
                                    lam1 * pow(lam4 + lam5, 2) * pow(m222, 2))),
            0};

    return g1;
}

std::vector<std::vector<double>> get_groebner_basis_coeffs_r2(Parameters<double> &params, double c1) {
    double m112 = params.m112;
    double m122 = params.m122;
    double m222 = params.m222;
    double lam1 = params.lam1;
    double lam2 = params.lam2;
    double lam3 = params.lam3;
    double lam4 = params.lam4;
    double lam5 = params.lam5;

    std::vector<std::vector<double>> gs;

    gs.push_back(std::vector<double>{
            c1 * m122 *
                    (pow(c1, 2) * (lam1 * lam2 - pow(lam3, 2)) * pow(lam4 + lam5, 2) *
                            (-(lam3 * m112) + lam1 * m222) -
                            2 * (pow(lam1, 2) * pow(lam2, 2) * pow(m122, 2) +
                                    lam2 * (lam3 * pow(lam4 + lam5, 2) * pow(m112, 2) -
                                            2 * lam1 * pow(lam3, 2) * pow(m122, 2) -
                                            lam1 * pow(lam4 + lam5, 2) * m112 * m222) +
                                    lam3 * (pow(lam3, 3) * pow(m122, 2) -
                                            lam3 * pow(lam4 + lam5, 2) * m112 * m222 +
                                            lam1 * pow(lam4 + lam5, 2) * pow(m222, 2)))), 0});
    gs.push_back(std::vector<double>{
            c1 * pow(m122, 2) *
                    (pow(lam3, 3) * pow(lam4 + lam5, 2) * pow(m112, 2) +
                            pow(lam1, 3) * pow(lam2, 2) * pow(m122, 2) +
                            lam1 * pow(lam3, 4) * pow(m122, 2) +
                            pow(lam1, 2) * lam3 * pow(lam4 + lam5, 2) * pow(m222, 2) -
                            2 * lam1 * pow(lam3, 2) *
                                    (lam1 * lam2 * pow(m122, 2) +
                                            pow(lam4 + lam5, 2) * m112 * m222)),
            0,
            c1 * pow(lam4 + lam5, 2) * (pow(c1, 2) * lam1 + 2 * m112) * pow(m122, 2) *
                    pow(lam3 * m112 - lam1 * m222, 2)});
    gs.push_back(std::vector<double>{
            c1 * pow(m122, 2) * (lam2 * m112 - lam3 * m222) *
                    (pow(c1, 2) * pow(lam3, 3) * pow(lam4 + lam5, 2) * m112 -
                            2 * (pow(lam1, 2) * pow(lam2, 2) * pow(m122, 2) +
                                    pow(lam3, 2) * (pow(lam3, 2) * pow(m122, 2) -
                                            2 * pow(lam4 + lam5, 2) * m112 * m222) +
                                    lam1 * lam3 *
                                            (-2 * lam2 * lam3 * pow(m122, 2) +
                                                    pow(lam4 + lam5, 2) * pow(m222, 2)))),
            0,
            c1 * pow(m122, 2) *
                    (pow(c1, 4) * lam1 * lam2 * lam3 * pow(lam4 + lam5, 2) * m112 *
                            (lam3 * m112 - lam1 * m222) +
                            2 * pow(c1, 2) *
                                    (pow(lam1, 2) * pow(lam2, 2) * lam3 * m112 * pow(m122, 2) +
                                            lam1 * lam3 * pow(lam4 + lam5, 2) * pow(m222, 2) *
                                                    (-(lam3 * m112) + lam1 * m222) +
                                            lam2 * m112 *
                                                    (pow(lam3, 2) * pow(lam4 + lam5, 2) * pow(m112, 2) -
                                                            lam1 * pow(lam3, 3) * pow(m122, 2) -
                                                            pow(lam1, 2) * pow(lam4 + lam5, 2) * pow(m222, 2))) +
                            4 * m112 *
                                    (lam1 * (pow(lam2, 2) * lam3 * m112 * pow(m122, 2) -
                                            lam2 * pow(lam4 + lam5, 2) * m112 * pow(m222, 2) +
                                            lam3 * pow(lam4 + lam5, 2) * pow(m222, 3)) -
                                            lam3 * m112 *
                                                    (lam3 * pow(lam4 + lam5, 2) * pow(m222, 2) +
                                                            lam2 * (pow(lam3, 2) * pow(m122, 2) -
                                                                    pow(lam4 + lam5, 2) * m112 * m222))))});
    gs.push_back(std::vector<double>{
            c1 * pow(m122, 2) *
                    (-(pow(c1, 2) * pow(lam3, 2) * pow(lam4 + lam5, 2) *
                            (lam3 * m112 - lam1 * m222)) +
                            2 * (pow(lam1, 2) * pow(lam2, 2) * pow(m122, 2) +
                                    pow(lam3, 2) * (pow(lam3, 2) * pow(m122, 2) -
                                            pow(lam4 + lam5, 2) * m112 * m222) +
                                    lam1 * lam3 *
                                            (-2 * lam2 * lam3 * pow(m122, 2) +
                                                    pow(lam4 + lam5, 2) * pow(m222, 2)))),
            0,
            c1 * (pow(c1, 2) * lam1 + 2 * m112) * pow(m122, 2) *
                    (-(pow(c1, 2) * lam3 * pow(lam4 + lam5, 2) *
                            (lam3 * m112 - lam1 * m222)) +
                            2 * lam3 *
                                    (pow(lam3, 2) * pow(m122, 2) -
                                            pow(lam4 + lam5, 2) * m112 * m222) +
                            2 * lam1 *
                                    (-(lam2 * lam3 * pow(m122, 2)) +
                                            pow(lam4 + lam5, 2) * pow(m222, 2)))});
    gs.push_back(std::vector<double>{
            c1 * pow(lam4 + lam5, 2) * (pow(c1, 2) * lam1 + 2 * m112) * pow(m122, 2) *
                    (lam2 * m112 - lam3 * m222),
            0,
            -(c1 * (pow(c1, 2) * lam1 + 2 * m112) * pow(m122, 2) *
                    (2 * (-(lam1 * lam2) + pow(lam3, 2)) * pow(m122, 2) -
                            pow(c1, 2) * pow(lam4 + lam5, 2) * (lam3 * m112 - lam1 * m222)))});
    gs.push_back(std::vector<double>{
            pow(c1, 3) * (lam1 * lam2 - pow(lam3, 2)) * pow(lam4 + lam5, 2) *
                    (-(lam3 * m112) + lam1 * m222) -
                    2 * c1 *
                            (pow(lam1, 2) * pow(lam2, 2) * pow(m122, 2) +
                                    lam2 * (lam3 * pow(lam4 + lam5, 2) * pow(m112, 2) -
                                            2 * lam1 * pow(lam3, 2) * pow(m122, 2) -
                                            lam1 * pow(lam4 + lam5, 2) * m112 * m222) +
                                    lam3 * (pow(lam3, 3) * pow(m122, 2) -
                                            lam3 * pow(lam4 + lam5, 2) * m112 * m222 +
                                            lam1 * pow(lam4 + lam5, 2) * pow(m222, 2))),
            0, 0});
    gs.push_back(std::vector<double>{c1 * (lam1 * lam2 - pow(lam3, 2)) *
            (pow(c1, 2) * lam1 + 2 * m112) * pow(m122, 2),
            0,
            2 * c1 * (pow(c1, 2) * lam1 + 2 * m112) *
                    pow(m122, 2) * (-(lam3 * m112) + lam1 * m222)});
    gs.push_back(std::vector<double>{
            c1 * lam3 * pow(lam4 + lam5, 2) * (pow(c1, 2) * lam1 + 2 * m112) *
                    pow(m122, 2) * (-(lam3 * m112) + lam1 * m222),
            0,
            c1 * (pow(c1, 2) * lam1 + 2 * m112) * pow(m122, 2) *
                    (-2 * lam3 * pow(lam4 + lam5, 2) * pow(m112, 2) +
                            2 * lam1 * pow(lam3, 2) * pow(m122, 2) +
                            pow(c1, 2) * lam1 * pow(lam4 + lam5, 2) *
                                    (-(lam3 * m112) + lam1 * m222) +
                            2 * lam1 *
                                    (-(lam1 * lam2 * pow(m122, 2)) +
                                            pow(lam4 + lam5, 2) * m112 * m222))});
    gs.push_back(std::vector<double>{
            pow(c1, 5) * lam3 * (lam1 * lam2 - pow(lam3, 2)) * pow(lam4 + lam5, 2) *
                    m112 +
                    4 * c1 * m112 *
                            (-(lam1 * pow(lam2, 2) * pow(m122, 2)) -
                                    lam3 * pow(lam4 + lam5, 2) * pow(m222, 2) +
                                    lam2 * (pow(lam3, 2) * pow(m122, 2) +
                                            pow(lam4 + lam5, 2) * m112 * m222)) +
                    2 * pow(c1, 3) *
                            (pow(lam3, 2) * (pow(lam3, 2) * pow(m122, 2) -
                                    2 * pow(lam4 + lam5, 2) * m112 * m222) +
                                    lam2 * (lam3 * pow(lam4 + lam5, 2) * pow(m112, 2) -
                                            lam1 * pow(lam3, 2) * pow(m122, 2) +
                                            lam1 * pow(lam4 + lam5, 2) * m112 * m222)),
            0,
            4 * c1 * lam3 * (pow(c1, 2) * lam1 + 2 * m112) * pow(m122, 2) *
                    (lam2 * m112 - lam3 * m222)});
    gs.push_back(std::vector<double>{c1 * pow(lam4 + lam5, 2) *
            (pow(c1, 2) * lam1 + 2 * m112) *
            (pow(c1, 2) * (lam1 * lam2 - pow(lam3, 2)) +
                    2 * lam2 * m112 - 2 * lam3 * m222),
            0,
            4 * c1 * (lam1 * lam2 - pow(lam3, 2)) *
                    (pow(c1, 2) * lam1 + 2 * m112) * pow(m122, 2)});
    gs.push_back(std::vector<double>{
            c1 * pow(lam4 + lam5, 2) * m122 * (lam2 * m112 - lam3 * m222), 0,
            2 * c1 * (lam1 * lam2 - pow(lam3, 2)) * pow(m122, 3) +
                    pow(c1, 3) * pow(lam4 + lam5, 2) * m122 * (lam3 * m112 - lam1 * m222),
            0});
    gs.push_back(std::vector<double>{c1 * (lam1 * lam2 - pow(lam3, 2)) * m122, 0,
            2 * c1 * m122 * (-(lam3 * m112) + lam1 * m222), 0});
    gs.push_back(std::vector<double>{
            -(c1 * lam3 * pow(lam4 + lam5, 2) * m122 * (lam3 * m112 - lam1 * m222)),
            0,
            c1 * m122 *
                    (-2 * lam3 * pow(lam4 + lam5, 2) * pow(m112, 2) +
                            2 * lam1 * pow(lam3, 2) * pow(m122, 2) +
                            pow(c1, 2) * lam1 * pow(lam4 + lam5, 2) *
                                    (-(lam3 * m112) + lam1 * m222) +
                            2 * lam1 *
                                    (-(lam1 * lam2 * pow(m122, 2)) +
                                            pow(lam4 + lam5, 2) * m112 * m222)),
            0});
    gs.push_back(std::vector<double>{
            c1 * pow(lam4 + lam5, 2) *
                    (pow(c1, 2) * (lam1 * lam2 - pow(lam3, 2)) + 2 * lam2 * m112 -
                            2 * lam3 * m222),
            0, 4 * c1 * (lam1 * lam2 - pow(lam3, 2)) * pow(m122, 2), 0});
    gs.push_back(std::vector<double>{
            c1 * pow(lam4 + lam5, 2) * (lam2 * m112 - lam3 * m222), 0,
            2 * c1 * (lam1 * lam2 - pow(lam3, 2)) * pow(m122, 2) +
                    pow(c1, 3) * pow(lam4 + lam5, 2) * (lam3 * m112 - lam1 * m222),
            0, 0});
    gs.push_back(std::vector<double>{c1 * (lam1 * lam2 - pow(lam3, 2)), 0,
            c1 * (-2 * lam3 * m112 + 2 * lam1 * m222), 0, 0});
    gs.push_back(std::vector<double>{
            -(c1 * lam3 * pow(lam4 + lam5, 2) * (lam3 * m112 - lam1 * m222)), 0,
            c1 * (-2 * lam3 * pow(lam4 + lam5, 2) * pow(m112, 2) +
                    2 * lam1 * pow(lam3, 2) * pow(m122, 2) +
                    pow(c1, 2) * lam1 * pow(lam4 + lam5, 2) *
                            (-(lam3 * m112) + lam1 * m222) +
                    2 * lam1 *
                            (-(lam1 * lam2 * pow(m122, 2)) +
                                    pow(lam4 + lam5, 2) * m112 * m222)),
            0, 0});
    gs.push_back(std::vector<double>{
            c1 * lam3 * pow(lam4 + lam5, 2) * (pow(c1, 2) * lam3 + 2 * m222), 0,
            pow(c1, 5) * lam1 * lam3 * pow(lam4 + lam5, 2) +
                    2 * pow(c1, 3) * pow(lam4 + lam5, 2) * (lam3 * m112 + lam1 * m222) +
                    4 * c1 *
                            (-(lam1 * lam2 * pow(m122, 2)) + pow(lam3, 2) * pow(m122, 2) +
                                    pow(lam4 + lam5, 2) * m112 * m222),
            0, 4 * c1 * lam3 * (pow(c1, 2) * lam1 + 2 * m112) * pow(m122, 2)});
    gs.push_back(std::vector<double>{
            c1 * lam3 * pow(lam4 + lam5, 2) * (pow(c1, 2) * lam1 + 2 * m112), 0,
            c1 * pow(lam4 + lam5, 2) * pow(pow(c1, 2) * lam1 + 2 * m112, 2), 0,
            4 * c1 * lam1 * (pow(c1, 2) * lam1 + 2 * m112) * pow(m122, 2)});
    gs.push_back(std::vector<double>{c1 * lam2 * pow(lam4 + lam5, 2),
            0,
            c1 * pow(lam4 + lam5, 2) *
                    (pow(c1, 2) * lam3 + 2 * m222),
            0,
            4 * c1 * lam3 * pow(m122, 2),
            0});
    gs.push_back(std::vector<double>{c1 * lam3 * pow(lam4 + lam5, 2),
            0,
            c1 * pow(lam4 + lam5, 2) *
                    (pow(c1, 2) * lam1 + 2 * m112),
            0,
            4 * c1 * lam1 * pow(m122, 2),
            0});
    gs.push_back(std::vector<double>{
            lam2 * pow(-(lam1 * lam2) + pow(lam3 + lam4 + lam5, 2), 2),
            0,
            2 * (lam1 * lam2 - pow(lam3 + lam4 + lam5, 2)) *
                    (-(pow(lam3 + lam4 + lam5, 2) * m222) -
                            lam2 * (2 * lam3 * m112 + 2 * lam4 * m112 + 2 * lam5 * m112 -
                                    3 * lam1 * m222)),
            0,
            -(pow(c1, 4) * lam1 * pow(lam4 + lam5, 4)) -
                    2 * pow(c1, 2) * pow(lam4 + lam5, 4) * m112 +
                    8 * pow(lam3 + lam4 + lam5, 2) * m222 *
                            (lam3 * m112 + lam4 * m112 + lam5 * m112 - lam1 * m222) +
                    4 * lam2 *
                            (pow(lam3, 2) * pow(m112, 2) + pow(lam4, 2) * pow(m112, 2) +
                                    pow(lam5, 2) * pow(m112, 2) + 4 * lam1 * lam5 * pow(m122, 2) -
                                    4 * lam1 * lam5 * m112 * m222 + 3 * pow(lam1, 2) * pow(m222, 2) +
                                    2 * lam4 *
                                            (lam5 * pow(m112, 2) + 2 * lam1 * pow(m122, 2) -
                                                    2 * lam1 * m112 * m222) +
                                    2 * lam3 *
                                            (lam4 * pow(m112, 2) + lam5 * pow(m112, 2) +
                                                    2 * lam1 * pow(m122, 2) - 2 * lam1 * m112 * m222)),
            0,
            4 * (pow(c1, 6) * pow(lam1, 2) * pow(lam4 + lam5, 2) *
                    (lam3 + lam4 + lam5) +
                    pow(c1, 4) * lam1 * pow(lam4 + lam5, 2) *
                            (2 * lam3 + 3 * (lam4 + lam5)) * m112 +
                    pow(c1, 2) * (-4 * lam1 * pow(lam3, 2) * pow(m122, 2) +
                            lam3 * (lam4 + lam5) *
                                    (lam4 * pow(m112, 2) + lam5 * pow(m112, 2) -
                                            4 * lam1 * pow(m122, 2)) +
                            2 * pow(lam4 + lam5, 2) *
                                    (lam4 * pow(m112, 2) + lam5 * pow(m112, 2) -
                                            lam1 * pow(m122, 2))) +
                    2 * (lam1 * lam2 * m112 * pow(m122, 2) +
                            pow(lam1, 2) * pow(m222, 3) -
                            2 * lam1 * (lam3 + lam4 + lam5) * m222 *
                                    (-2 * pow(m122, 2) + m112 * m222) +
                            pow(lam3 + lam4 + lam5, 2) * m112 *
                                    (-pow(m122, 2) + m112 * m222))),
            0,
            16 * lam1 * pow(m122, 2) *
                    (pow(c1, 4) * lam1 * (lam3 + lam4 + lam5) +
                            pow(c1, 2) * (lam4 + lam5) * m112 - pow(m122, 2) + m112 * m222),
            0});

    return gs;
}

std::vector<std::vector<double>> get_groebner_basis_coeffs_r1(Parameters<double> &params, double c1, double r2) {
    double m112 = params.m112;
    double m122 = params.m122;
    double m222 = params.m222;
    double lam1 = params.lam1;
    double lam2 = params.lam2;
    double lam3 = params.lam3;
    double lam4 = params.lam4;
    double lam5 = params.lam5;

    std::vector<std::vector<double>> gs;

    gs.push_back(std::vector<double>{8 * lam1 * m122 * (pow(lam1, 2) * pow(lam2, 2) *
            pow(m122, 2) + pow(lam3 + lam4 + lam5, 4) * (pow(m122, 2) - m112 * m222) +
            lam1 * pow(lam3 + lam4 + lam5, 2) * (2 * (lam3 + lam4 + lam5) * pow(m222, 2) +
                    lam2 * (2 * pow(m122, 2) - m112 * m222))), r2 * (pow(c1, 4) * lam1 * pow(lam4 +
                                                                                                     lam5, 2) *
            pow(2 * lam3 + lam4 + lam5, 2) * (2 * pow(lam3, 3) +
            4 * pow(lam3, 2) * (lam4 + lam5) + 3 * lam3 * pow(lam4 + lam5, 2) + pow(lam4 +
                                                                                            lam5, 3)) * pow(r2, 2) -
            pow(lam1, 3) * pow(lam2, 2) * (2 * m222 +
                    lam2 * pow(r2, 2)) * (4 * pow(m122, 2) + (lam3 + lam4 +
                    lam5) * pow(r2, 2) * (2 * m222 + lam2 * pow(r2, 2))) - pow(lam3 + lam4 +
                                                                                       lam5, 5) *
            (-8 * m112 * pow(m122, 2) + 4 * pow(m112, 2) * (2 * m222 +
                    lam2 * pow(r2, 2)) + 4 * (lam3 + lam4 + lam5) * m112 * pow(r2, 2) * (2 * m222 +
                    lam2 * pow(r2, 2)) + pow(lam3 + lam4 + lam5, 2) * pow(r2, 4) * (2 * m222 +
                    lam2 * pow(r2, 2))) + pow(lam1, 2) * pow(lam3 + lam4 + lam5, 2) * (-16 * (lam3 +
            lam4 + lam5) * pow(m222, 3) - 2 * pow(lam2, 2) * pow(r2, 2) * (8 * pow(m122, 2) -
            6 * m112 * m222 + (lam3 + lam4 + lam5) * m222 * pow(r2, 2)) -
            8 * lam2 * m222 * (2 * pow(m122, 2) - m112 * m222 + 2 * (lam3 + lam4 +
                    lam5) * m222 * pow(r2, 2)) + pow(lam2, 3) * (4 * m112 * pow(r2, 4) + (lam3 + lam4
            + lam5) * pow(r2, 6))) + lam1 * pow(lam3 + lam4 + lam5, 3) * (12 * (lam3 + lam4
            + lam5) * m222 * (-2 * pow(m122, 2) + 2 * m112 * m222 + (lam3 + lam4 +
            lam5) * m222 * pow(r2, 2)) + pow(lam2, 2) * (-4 * pow(m112, 2) * pow(r2, 2) +
            pow(lam3 + lam4 + lam5, 2) * pow(r2, 6)) + 4 * lam2 * (-2 * pow(m112, 2) * m222 +
            (lam3 + lam4 + lam5) * pow(r2, 2) * (-3 * pow(m122, 2) + 2 * (lam3 + lam4 +
                    lam5) * m222 * pow(r2, 2)) + m112 * (2 * pow(m122, 2) + 3 * (lam3 + lam4 +
            lam5) * m222 * pow(r2, 2)))) +
            2 * pow(c1, 2) * (2 * pow(lam1, 3) * pow(lam2, 2) * (lam4 + lam5) * pow(m122, 2) +
                    pow(lam4 + lam5, 2) * pow(lam3 + lam4 + lam5, 5) * m112 * pow(r2, 2) +
                    pow(lam1, 2) * (lam4 + lam5) * (lam3 + lam4 + lam5) * (4 * pow(lam3 + lam4 +
                                                                                           lam5, 2) * pow(m222, 2) -
                            pow(lam2, 2) * (lam4 + lam5) * m112 * pow(r2, 2) +
                            2 * lam2 * (lam4 + lam5) * (2 * pow(m122, 2) - m112 * m222 + (lam4 +
                                    lam5) * m222 * pow(r2, 2)) + lam2 * lam3 * (4 * pow(m122, 2) - 2 * m112 * m222 +
                            3 * (lam4 + lam5) * m222 * pow(r2, 2))) + lam1 * (lam3 + lam4 +
                    lam5) * (16 * pow(lam3, 4) * pow(m122, 2) + pow(lam3, 3) * (lam4 +
                    lam5) * (34 * pow(m122, 2) - 2 * m112 * m222 + 5 * (lam4 + lam5) * m222 * pow(r2, 2))
                    + pow(lam4 + lam5, 4) * (4 * pow(m122, 2) - m112 * (2 * m222 + lam2 * pow(r2, 2)))
                    + 2 * lam3 * pow(lam4 + lam5, 3) * (9 * pow(m122, 2) + (lam4 +
                    lam5) * m222 * pow(r2, 2) - m112 * (3 * m222 + 2 * lam2 * pow(r2, 2))) +
                    2 * pow(lam3, 2) * pow(lam4 + lam5, 2) * (17 * pow(m122, 2) + 3 * (lam4 +
                            lam5) * m222 * pow(r2, 2) - m112 * (3 * m222 + 2 * lam2 * pow(r2, 2))))))});
    gs.push_back(std::vector<double>{2 * c1 * lam3 * pow(m122, 2), c1 * (lam4 +
            lam5) * m122 * r2 * (pow(c1, 2) * lam3 + 2 * m222 + lam2 * pow(r2, 2))});
    gs.push_back(std::vector<double>{2 * c1 * lam1 * m122,
            c1 * r2 * (pow(c1, 2) * lam1 * (lam4 + lam5) + 2 * lam3 * m112 + 2 * lam4 * m112 +
                    2 * lam5 * m112 - 2 * lam1 * m222 - lam1 * lam2 * pow(r2, 2) +
                    pow(lam3, 2) * pow(r2, 2) + lam3 * (lam4 + lam5) * pow(r2, 2))});
    gs.push_back(std::vector<double>{2 * c1 * lam3 * m122 * r2, c1 * (lam4 +
            lam5) * pow(r2, 2) * (pow(c1, 2) * lam3 + 2 * m222 + lam2 * pow(r2, 2))});
    gs.push_back(std::vector<double>{4 * c1 * (lam4 + lam5) * m112 * m122 * r2,
            -(c1 * (8 * m112 * pow(m122, 2) + pow(c1, 4) * lam1 * pow(lam4 +
                                                                              lam5, 2) * pow(r2, 2) +
                    pow(c1, 2) * (4 * lam1 * pow(m122, 2) + pow(lam4 +
                                                                        lam5, 2) * pow(r2, 2) *
                            (2 * m112 + lam3 * pow(r2, 2)))))});
    gs.push_back(std::vector<double>{c1 * (lam4 +
            lam5) * (pow(c1, 2) * lam1 * (lam4 + lam5) + 2 * (lam3 * m112 + lam4 * m112 +
            lam5 * m112 - lam1 * m222)) * r2, -2 * c1 * m122 * (pow(c1, 2) * lam1 * (lam4 + lam5)
            + 2 * lam4 * m112 + 2 * lam5 * m112 + lam1 * lam2 * pow(r2, 2) -
            pow(lam3, 2) * pow(r2, 2))});
    gs.push_back(std::vector<double>{8 * m122 * (pow(lam1, 2) * pow(lam2, 2) * pow(
            m122, 2) + lam1 * pow(lam3 + lam4 + lam5, 2) * (2 * (lam3 + lam4 +
            lam5) * pow(m222, 2) + lam2 * (pow(m122, 2) - m112 * m222)) + pow(lam3 + lam4
            + lam5, 5) * m222 * pow(r2, 2)), r2 * (pow(c1, 4) * pow(lam3, 2) * pow(lam4 +
                                                                                           lam5, 2) *
            (lam3 + lam4 + lam5) * pow(2 * lam3 + lam4 + lam5, 2) * pow(r2, 2) -
            pow(lam1, 2) * pow(lam2, 4) * (lam3 + lam4 + lam5) * pow(r2, 6) +
            2 * lam1 * pow(lam2, 3) * pow(r2, 2) * (pow(lam3 + lam4 +
                                                                lam5, 2) * pow(r2, 2) *
                    (2 * m112 + (lam3 + lam4 + lam5) * pow(r2, 2)) -
                    2 * lam1 * (pow(m122, 2) + (lam3 + lam4 + lam5) * m222 * pow(r2, 2))) -
            pow(lam2, 2) * (4 * pow(lam1, 2) * m222 * (2 * pow(m122, 2) + (lam3 + lam4 +
                    lam5) * m222 * pow(r2, 2)) - 2 * lam1 * pow(lam3 + lam4 +
                                                                        lam5, 2) * pow(r2, 2) *
                    (-6 * pow(m122, 2) + 6 * m112 * m222 + (lam3 + lam4 +
                            lam5) * m222 * pow(r2, 2)) + pow(lam3 + lam4 + lam5, 3) * pow(2 * m112 * r2 +
                                                                                                  (lam3 + lam4 + lam5) *
                                                                                                          pow(r2, 3),
                                                                                          2)) + 8 * pow(lam3 + lam4 +
                                                                                                                lam5,
                                                                                                        3) * m222 *
            (-2 * lam3 * pow(m122, 2) - 2 * lam5 * pow(m122, 2) +
                    2 * lam5 * m112 * m222 - 2 * lam1 * pow(m222, 2) + pow(lam3, 2) * m222 * pow(r2, 2) +
                    pow(lam4, 2) * m222 * pow(r2, 2) + pow(lam5, 2) * m222 * pow(r2, 2) +
                    2 * lam3 * m222 * (m112 + (lam4 + lam5) * pow(r2, 2)) + 2 * lam4 * (-pow(m122, 2) +
                    m222 * (m112 + lam5 * pow(r2, 2)))) + 2 * lam2 * pow(lam3 + lam4 +
                                                                                 lam5, 2) *
            (4 * lam5 * m112 * pow(m122, 2) - 4 * lam5 * pow(m112, 2) * m222 -
                    4 * lam1 * pow(m122, 2) * m222 + 4 * lam1 * m112 * pow(m222, 2) -
                    6 * lam1 * lam5 * pow(m222, 2) * pow(r2, 2) + pow(lam3, 3) * m222 * pow(r2, 4) +
                    pow(lam4, 3) * m222 * pow(r2, 4) + 3 * pow(lam4, 2) * lam5 * m222 * pow(r2, 4) +
                    pow(lam5, 3) * m222 * pow(r2, 4) + 3 * pow(lam3, 2) * (lam4 +
                    lam5) * m222 * pow(r2, 4) + lam4 * (4 * m112 * pow(m122, 2) - 4 * pow(m112, 2) * m222
                    - 6 * lam1 * pow(m222, 2) * pow(r2, 2) + 3 * pow(lam5, 2) * m222 * pow(r2, 4)) +
                    lam3 * (4 * m112 * pow(m122, 2) - 4 * pow(m112, 2) * m222 +
                            3 * m222 * pow(r2, 2) * (-2 * lam1 * m222 + pow(lam4 + lam5, 2) * pow(r2, 2)))) +
            2 * pow(c1, 2) * (2 * pow(lam3, 5) * (4 * pow(m122, 2) + (lam4 +
                    lam5) * m222 * pow(r2, 2)) + 4 * pow(lam3, 4) * (lam4 + lam5) * (4 * pow(m122, 2) +
                    3 * (lam4 + lam5) * m222 * pow(r2, 2)) + pow(lam3, 3) * (lam4 +
                    lam5) * (10 * lam4 * pow(m122, 2) + 10 * lam5 * pow(m122, 2) + 4 * lam1 * pow(m222, 2)
                    - 3 * lam2 * lam5 * m112 * pow(r2, 2) + 20 * pow(lam4, 2) * m222 * pow(r2, 2) +
                    20 * pow(lam5, 2) * m222 * pow(r2, 2) + lam4 * (-3 * lam2 * m112 +
                    40 * lam5 * m222) * pow(r2, 2)) + lam1 * (lam4 +
                    lam5) * (2 * lam1 * pow(lam2, 2) * pow(m122, 2) + pow(lam4 + lam5, 2) * (4 * (lam4 +
                    lam5) * pow(m222, 2) - pow(lam2, 2) * m112 * pow(r2, 2) + 2 * lam2 * (pow(m122, 2)
                    - m112 * m222 + (lam4 + lam5) * m222 * pow(r2, 2)))) + pow(lam3, 2) * (lam4 +
                    lam5) * (pow(lam4 + lam5, 2) * (2 * pow(m122, 2) + (-5 * lam2 * m112 + 13 * (lam4 +
                    lam5) * m222) * pow(r2, 2)) + lam1 * (12 * (lam4 + lam5) * pow(m222, 2) +
                    lam2 * (2 * pow(m122, 2) - 2 * m112 * m222 + 3 * (lam4 + lam5) * m222 * pow(r2, 2))))
                    + lam3 * pow(lam4 + lam5, 2) * (pow(lam4 + lam5, 2) * (-2 * lam2 * m112 + 3 * (lam4
                    + lam5) * m222) * pow(r2, 2) + lam1 * (12 * (lam4 + lam5) * pow(m222, 2) -
                    pow(lam2, 2) * m112 * pow(r2, 2) + lam2 * (4 * pow(m122, 2) - 4 * m112 * m222 +
                    5 * (lam4 + lam5) * m222 * pow(r2, 2))))))});
    gs.push_back(std::vector<double>{2 * m122 * (pow(lam3 + lam4 +
                                                             lam5, 2) * pow(r2, 2) +
            lam1 * (2 * m222 + lam2 * pow(r2, 2))),
            r2 * (-4 * lam4 * pow(m122, 2) - 4 * lam5 * pow(m122, 2) +
                    2 * pow(c1, 2) * lam1 * lam5 * m222 + 4 * lam5 * m112 * m222 - 4 * lam1 * pow(m222, 2) +
                    pow(c1, 2) * lam1 * lam2 * lam5 * pow(r2, 2) + 2 * lam2 * lam5 * m112 * pow(r2, 2) -
                    4 * lam1 * lam2 * m222 * pow(r2, 2) + 2 * pow(lam5, 2) * m222 * pow(r2, 2) -
                    lam1 * pow(lam2, 2) * pow(r2, 4) + lam2 * pow(lam5, 2) * pow(r2, 4) +
                    pow(lam3, 2) * pow(r2, 2) * (pow(c1, 2) * (lam4 + lam5) + 2 * m222 +
                            lam2 * pow(r2, 2)) + pow(lam4, 2) * (2 * m222 * pow(r2, 2) + lam2 * pow(r2, 4)) +
                    lam3 * (-4 * pow(m122, 2) + 2 * m112 * (2 * m222 + lam2 * pow(r2, 2)) + (lam4 +
                            lam5) * pow(r2, 2) * (pow(c1, 2) * (lam4 + lam5) + 4 * m222 +
                            2 * lam2 * pow(r2, 2))) + lam4 * (2 * m222 + lam2 * pow(r2, 2)) * (pow(c1, 2) * lam1 +
                    2 * (m112 + lam5 * pow(r2, 2))))});
    gs.push_back(std::vector<double>{-8 * lam1 * m122 * (lam1 * lam2 * pow(m122, 2)
            - pow(lam3 + lam4 + lam5, 2) * (-pow(m122, 2) + m222 * (m112 + (lam3 + lam4
            + lam5) * pow(r2, 2)))), r2 * (-8 * pow(lam4, 3) * m112 * pow(m122, 2) -
            24 * pow(lam4, 2) * lam5 * m112 * pow(m122, 2) -
            24 * lam4 * pow(lam5, 2) * m112 * pow(m122, 2) - 8 * pow(lam5, 3) * m112 * pow(m122, 2)
            + 8 * pow(lam4, 3) * pow(m112, 2) * m222 +
            24 * pow(lam4, 2) * lam5 * pow(m112, 2) * m222 +
            24 * lam4 * pow(lam5, 2) * pow(m112, 2) * m222 + 8 * pow(lam5, 3) * pow(m112, 2) * m222
            + 8 * pow(lam1, 2) * lam2 * pow(m122, 2) * m222 +
            8 * lam1 * pow(lam4, 2) * pow(m122, 2) * m222 +
            16 * lam1 * lam4 * lam5 * pow(m122, 2) * m222 +
            8 * lam1 * pow(lam5, 2) * pow(m122, 2) * m222 -
            8 * lam1 * pow(lam4, 2) * m112 * pow(m222, 2) -
            16 * lam1 * lam4 * lam5 * m112 * pow(m222, 2) -
            8 * lam1 * pow(lam5, 2) * m112 * pow(m222, 2) - pow(c1, 4) * lam1 * pow(lam4 +
                                                                                            lam5, 2) *
            (lam3 + lam4 + lam5) * pow(2 * lam3 + lam4 + lam5, 2) * pow(r2, 2) +
            4 * lam2 * pow(lam4, 3) * pow(m112, 2) * pow(r2, 2) +
            12 * lam2 * pow(lam4, 2) * lam5 * pow(m112, 2) * pow(r2, 2) +
            12 * lam2 * lam4 * pow(lam5, 2) * pow(m112, 2) * pow(r2, 2) +
            4 * lam2 * pow(lam5, 3) * pow(m112, 2) * pow(r2, 2) +
            4 * pow(lam1, 2) * pow(lam2, 2) * pow(m122, 2) * pow(r2, 2) +
            12 * lam1 * lam2 * pow(lam4, 2) * pow(m122, 2) * pow(r2, 2) +
            24 * lam1 * lam2 * lam4 * lam5 * pow(m122, 2) * pow(r2, 2) +
            12 * lam1 * lam2 * pow(lam5, 2) * pow(m122, 2) * pow(r2, 2) -
            12 * lam1 * lam2 * pow(lam4, 2) * m112 * m222 * pow(r2, 2) +
            8 * pow(lam4, 4) * m112 * m222 * pow(r2, 2) -
            24 * lam1 * lam2 * lam4 * lam5 * m112 * m222 * pow(r2, 2) +
            32 * pow(lam4, 3) * lam5 * m112 * m222 * pow(r2, 2) -
            12 * lam1 * lam2 * pow(lam5, 2) * m112 * m222 * pow(r2, 2) +
            48 * pow(lam4, 2) * pow(lam5, 2) * m112 * m222 * pow(r2, 2) +
            32 * lam4 * pow(lam5, 3) * m112 * m222 * pow(r2, 2) +
            8 * pow(lam5, 4) * m112 * m222 * pow(r2, 2) +
            4 * pow(lam1, 2) * lam2 * lam4 * pow(m222, 2) * pow(r2, 2) -
            4 * lam1 * pow(lam4, 3) * pow(m222, 2) * pow(r2, 2) +
            4 * pow(lam1, 2) * lam2 * lam5 * pow(m222, 2) * pow(r2, 2) -
            12 * lam1 * pow(lam4, 2) * lam5 * pow(m222, 2) * pow(r2, 2) -
            12 * lam1 * lam4 * pow(lam5, 2) * pow(m222, 2) * pow(r2, 2) -
            4 * lam1 * pow(lam5, 3) * pow(m222, 2) * pow(r2, 2) -
            4 * lam1 * pow(lam2, 2) * pow(lam4, 2) * m112 * pow(r2, 4) +
            4 * lam2 * pow(lam4, 4) * m112 * pow(r2, 4) -
            8 * lam1 * pow(lam2, 2) * lam4 * lam5 * m112 * pow(r2, 4) +
            16 * lam2 * pow(lam4, 3) * lam5 * m112 * pow(r2, 4) -
            4 * lam1 * pow(lam2, 2) * pow(lam5, 2) * m112 * pow(r2, 4) +
            24 * lam2 * pow(lam4, 2) * pow(lam5, 2) * m112 * pow(r2, 4) +
            16 * lam2 * lam4 * pow(lam5, 3) * m112 * pow(r2, 4) +
            4 * lam2 * pow(lam5, 4) * m112 * pow(r2, 4) +
            4 * pow(lam1, 2) * pow(lam2, 2) * lam4 * m222 * pow(r2, 4) -
            6 * lam1 * lam2 * pow(lam4, 3) * m222 * pow(r2, 4) + 2 * pow(lam4, 5) * m222 * pow(r2, 4)
            + 4 * pow(lam1, 2) * pow(lam2, 2) * lam5 * m222 * pow(r2, 4) -
            18 * lam1 * lam2 * pow(lam4, 2) * lam5 * m222 * pow(r2, 4) +
            10 * pow(lam4, 4) * lam5 * m222 * pow(r2, 4) -
            18 * lam1 * lam2 * lam4 * pow(lam5, 2) * m222 * pow(r2, 4) +
            20 * pow(lam4, 3) * pow(lam5, 2) * m222 * pow(r2, 4) -
            6 * lam1 * lam2 * pow(lam5, 3) * m222 * pow(r2, 4) +
            20 * pow(lam4, 2) * pow(lam5, 3) * m222 * pow(r2, 4) +
            10 * lam4 * pow(lam5, 4) * m222 * pow(r2, 4) + 2 * pow(lam5, 5) * m222 * pow(r2, 4) +
            pow(lam1, 2) * pow(lam2, 3) * lam4 * pow(r2, 6) -
            2 * lam1 * pow(lam2, 2) * pow(lam4, 3) * pow(r2, 6) + lam2 * pow(lam4, 5) * pow(r2, 6)
            + pow(lam1, 2) * pow(lam2, 3) * lam5 * pow(r2, 6) -
            6 * lam1 * pow(lam2, 2) * pow(lam4, 2) * lam5 * pow(r2, 6) +
            5 * lam2 * pow(lam4, 4) * lam5 * pow(r2, 6) -
            6 * lam1 * pow(lam2, 2) * lam4 * pow(lam5, 2) * pow(r2, 6) +
            10 * lam2 * pow(lam4, 3) * pow(lam5, 2) * pow(r2, 6) -
            2 * lam1 * pow(lam2, 2) * pow(lam5, 3) * pow(r2, 6) +
            10 * lam2 * pow(lam4, 2) * pow(lam5, 3) * pow(r2, 6) +
            5 * lam2 * lam4 * pow(lam5, 4) * pow(r2, 6) + lam2 * pow(lam5, 5) * pow(r2, 6) +
            pow(lam3, 4) * pow(r2, 2) * (2 * m222 + lam2 * pow(r2, 2)) * (4 * m112 + 5 * (lam4 +
                    lam5) * pow(r2, 2)) + pow(lam3, 5) * (2 * m222 * pow(r2, 4) + lam2 * pow(r2, 6)) +
            2 * pow(lam3, 3) * (-4 * m112 * pow(m122, 2) + 2 * pow(m112, 2) * (2 * m222 +
                    lam2 * pow(r2, 2)) + 8 * (lam4 + lam5) * m112 * pow(r2, 2) * (2 * m222 +
                    lam2 * pow(r2, 2)) - pow(r2, 2) * (2 * m222 + lam2 * pow(r2, 2)) * (-5 * pow(lam4 +
                                                                                                         lam5, 2) *
                    pow(r2, 2) + lam1 * (m222 + lam2 * pow(r2, 2)))) +
            2 * pow(lam3, 2) * (-12 * lam5 * m112 * pow(m122, 2) + 6 * lam5 * pow(m112, 2) * (2 * m222
                    + lam2 * pow(r2, 2)) + 12 * pow(lam5, 2) * m112 * pow(r2, 2) * (2 * m222 +
                    lam2 * pow(r2, 2)) + 2 * lam1 * pow(m122, 2) * (2 * m222 + 3 * lam2 * pow(r2, 2)) +
                    3 * pow(lam4, 2) * pow(r2, 2) * (2 * m222 + lam2 * pow(r2, 2)) * (4 * m112 +
                            5 * lam5 * pow(r2, 2)) - 2 * lam1 * m112 * (2 * pow(m222, 2) +
                    3 * lam2 * m222 * pow(r2, 2) + pow(lam2, 2) * pow(r2, 4)) -
                    3 * lam1 * lam5 * pow(r2, 2) * (2 * pow(m222, 2) + 3 * lam2 * m222 * pow(r2, 2) +
                            pow(lam2, 2) * pow(r2, 4)) + 5 * pow(lam4, 3) * (2 * m222 * pow(r2, 4) +
                    lam2 * pow(r2, 6)) + 5 * pow(lam5, 3) * (2 * m222 * pow(r2, 4) + lam2 * pow(r2, 6)) +
                    3 * lam4 * (2 * pow(m112, 2) * (2 * m222 + lam2 * pow(r2, 2)) - pow(r2, 2) * (2 * m222 +
                            lam2 * pow(r2, 2)) * (lam1 * m222 + lam1 * lam2 * pow(r2, 2) -
                            5 * pow(lam5, 2) * pow(r2, 2)) - 4 * m112 * (pow(m122, 2) -
                            4 * lam5 * m222 * pow(r2, 2) - 2 * lam2 * lam5 * pow(r2, 4)))) +
            lam3 * (16 * pow(lam5, 3) * m112 * pow(r2, 2) * (2 * m222 + lam2 * pow(r2, 2)) +
                    pow(lam1, 2) * lam2 * pow(r2, 2) * pow(2 * m222 + lam2 * pow(r2, 2), 2) +
                    4 * pow(lam4, 3) * pow(r2, 2) * (2 * m222 + lam2 * pow(r2, 2)) * (4 * m112 +
                            5 * lam5 * pow(r2, 2)) + 5 * pow(lam4, 4) * (2 * m222 * pow(r2, 4) + lam2 * pow(r2, 6))
                    + 5 * pow(lam5, 4) * (2 * m222 * pow(r2, 4) + lam2 * pow(r2, 6)) +
                    8 * lam1 * lam5 * (pow(m122, 2) * (2 * m222 + 3 * lam2 * pow(r2, 2)) -
                            m112 * (2 * pow(m222, 2) + 3 * lam2 * m222 * pow(r2, 2) + pow(lam2, 2) * pow(r2, 4)))
                            + 6 * pow(lam5, 2) * (-4 * m112 * pow(m122, 2) + 2 * pow(m112, 2) * (2 * m222 +
                    lam2 * pow(r2, 2)) - lam1 * pow(r2, 2) * (2 * pow(m222, 2) +
                    3 * lam2 * m222 * pow(r2, 2) + pow(lam2, 2) * pow(r2, 4))) +
                    6 * pow(lam4, 2) * (2 * pow(m112, 2) * (2 * m222 + lam2 * pow(r2, 2)) -
                            pow(r2, 2) * (2 * m222 + lam2 * pow(r2, 2)) * (lam1 * m222 + lam1 * lam2 * pow(r2, 2)
                                    - 5 * pow(lam5, 2) * pow(r2, 2)) - 4 * m112 * (pow(m122, 2) -
                            4 * lam5 * m222 * pow(r2, 2) - 2 * lam2 * lam5 * pow(r2, 4))) +
                    4 * lam4 * (-12 * lam5 * m112 * pow(m122, 2) + 6 * lam5 * pow(m112, 2) * (2 * m222 +
                            lam2 * pow(r2, 2)) + 12 * pow(lam5, 2) * m112 * pow(r2, 2) * (2 * m222 +
                            lam2 * pow(r2, 2)) + 2 * lam1 * pow(m122, 2) * (2 * m222 + 3 * lam2 * pow(r2, 2)) -
                            2 * lam1 * m112 * (2 * pow(m222, 2) + 3 * lam2 * m222 * pow(r2, 2) +
                                    pow(lam2, 2) * pow(r2, 4)) - 3 * lam1 * lam5 * pow(r2, 2) * (2 * pow(m222, 2) +
                            3 * lam2 * m222 * pow(r2, 2) + pow(lam2, 2) * pow(r2, 4)) +
                            5 * pow(lam5, 3) * (2 * m222 * pow(r2, 4) + lam2 * pow(r2, 6)))) -
            2 * pow(c1, 2) * (2 * pow(lam1, 2) * lam2 * (lam4 + lam5) * pow(m122, 2) + pow(lam4
                                                                                                   + lam5, 2) *
                    pow(lam3 + lam4 + lam5, 3) * m112 * pow(r2, 2) - lam1 * (lam3 +
                    lam4 + lam5) * (2 * pow(lam3, 2) * (-4 * pow(m122, 2) + (lam4 +
                    lam5) * m222 * pow(r2, 2)) + lam3 * (lam4 + lam5) * (-10 * pow(m122, 2) +
                    2 * m112 * m222 + (lam4 + lam5) * m222 * pow(r2, 2)) - pow(lam4 +
                                                                                       lam5, 2) *
                    (4 * pow(m122, 2) - m112 * (2 * m222 + lam2 * pow(r2, 2))))))});
    gs.push_back(std::vector<double>{c1 * (lam4 + lam5) * pow(r2, 2),
            -2 * c1 * m122 * r2});
    gs.push_back(std::vector<double>{-(c1 * r2 * (pow(c1, 2) * lam1 * (lam4 +
            lam5) + 2 * lam3 * m112 + 2 * lam4 * m112 + 2 * lam5 * m112 - 2 * lam1 * m222 -
            lam1 * lam2 * pow(r2, 2) + pow(lam3, 2) * pow(r2, 2))), 2 * c1 * (pow(c1, 2) * lam1 +
            2 * m112) * m122});
    gs.push_back(std::vector<double>{8 * lam1 * pow(m122, 3) + 4 * pow(lam3 +
                                                                               lam4 + lam5, 2) * m122 * pow(r2, 2) *
            (m112 + (lam3 + lam4 +
                    lam5) * pow(r2, 2)), -(pow(c1, 4) * lam1 * pow(lam4 + lam5, 2) * (3 * lam3 + lam4
            + lam5) * pow(r2, 3)) - lam1 * r2 * (2 * m222 + lam2 * pow(r2, 2)) * (4 * pow(m122, 2)
            + (lam3 + lam4 + lam5) * pow(r2, 2) * (2 * m222 + lam2 * pow(r2, 2))) -
            2 * pow(c1, 2) * (6 * lam1 * lam3 * pow(m122, 2) * r2 + (lam4 +
                    lam5) * pow(r2, 3) * (-(pow(lam3, 2) * m112) + 2 * lam3 * (lam4 + lam5) * m112 +
                    pow(lam4 + lam5, 2) * m112 - pow(lam3, 3) * pow(r2, 2))) + pow(lam3 + lam4 +
                                                                                           lam5, 2) * pow(r2, 3) *
            (-8 * pow(m122, 2) + (2 * m222 + lam2 * pow(r2, 2)) * (2 * m112
                    + (lam3 + lam4 + lam5) * pow(r2, 2)))});
    gs.push_back(std::vector<double>{-(pow(lam3 + lam4 +
                                                   lam5, 2) * pow(r2, 2) *
            (2 * m112 + (lam3 + lam4 + lam5) * pow(r2, 2))) +
            lam1 * (-4 * pow(m122, 2) + (lam3 + lam4 + lam5) * pow(r2, 2) * (2 * m222 +
                    lam2 * pow(r2, 2))), 2 * m122 * r2 * (-(pow(c1, 2) * lam1 * (lam4 + lam5)) +
            pow(lam3 + lam4 + lam5, 2) * pow(r2, 2) + lam1 * (2 * m222 +
            lam2 * pow(r2, 2)))});
    gs.push_back(std::vector<double>{4 * (lam3 + lam4 +
            lam5) * m122 * pow(r2, 2) * (-2 * pow(m122, 2) + (2 * m222 +
            lam2 * pow(r2, 2)) * (m112 + (lam3 + lam4 + lam5) * pow(r2, 2))),
            r2 * (16 * pow(m122, 4) + pow(c1, 6) * lam1 * lam3 * pow(lam4 + lam5, 2) * pow(r2, 2)
                    - 4 * pow(m122, 2) * (2 * m222 + lam2 * pow(r2, 2)) * (2 * m112 + 3 * (lam3 + lam4 +
                    lam5) * pow(r2, 2)) + 2 * pow(c1, 4) * lam3 * (2 * lam1 * pow(m122, 2) + pow(lam4 +
                                                                                                         lam5, 2) *
                    m112 * pow(r2, 2)) + pow(2 * m222 * r2 +
                                                     lam2 * pow(r2, 3), 2) *
                    (2 * lam5 * m112 - 2 * lam1 * m222 - lam1 * lam2 * pow(r2, 2) +
                            pow(lam3, 2) * pow(r2, 2) + pow(lam4, 2) * pow(r2, 2) + pow(lam5, 2) * pow(r2, 2)
                            + 2 * lam4 * (m112 + lam5 * pow(r2, 2)) + 2 * lam3 * (m112 + (lam4 +
                            lam5) * pow(r2, 2))) + 2 * pow(c1, 2) * lam3 * pow(r2, 2) * (-((lam4 +
                    lam5) * (2 * pow(m122, 2) - m112 * (2 * m222 + lam2 * pow(r2, 2)))) +
                    lam3 * (-2 * pow(m122, 2) + (lam4 + lam5) * pow(r2, 2) * (2 * m222 +
                            lam2 * pow(r2, 2)))))});
    gs.push_back(std::vector<double>{-(pow(r2, 2) * (-4 * lam3 * pow(m122, 2) -
            4 * lam4 * pow(m122, 2) - 4 * lam5 * pow(m122, 2) + 4 * lam5 * m112 * m222 -
            4 * lam1 * pow(m222, 2) + 2 * lam2 * lam5 * m112 * pow(r2, 2) -
            4 * lam1 * lam2 * m222 * pow(r2, 2) + 2 * pow(lam5, 2) * m222 * pow(r2, 2) -
            lam1 * pow(lam2, 2) * pow(r2, 4) + lam2 * pow(lam5, 2) * pow(r2, 4) +
            2 * lam4 * (2 * m222 + lam2 * pow(r2, 2)) * (m112 + lam5 * pow(r2, 2)) +
            2 * lam3 * (2 * m222 + lam2 * pow(r2, 2)) * (m112 + (lam4 + lam5) * pow(r2, 2)) +
            pow(lam3, 2) * (2 * m222 * pow(r2, 2) + lam2 * pow(r2, 4)) +
            pow(lam4, 2) * (2 * m222 * pow(r2, 2) + lam2 * pow(r2, 4)))),
            2 * m122 * r2 * (-4 * pow(m122, 2) + 2 * m112 * (2 * m222 + lam2 * pow(r2, 2)) +
                    pow(r2, 2) * (pow(c1, 2) * lam3 * (lam4 + lam5) + 2 * (lam3 + lam4 +
                            lam5) * (2 * m222 + lam2 * pow(r2, 2))))});
    gs.push_back(std::vector<double>{2 * lam1 * m122,
            r2 * (pow(c1, 2) * lam1 * (lam4 + lam5) + 2 * lam4 * m112 + 2 * lam5 * m112 -
                    2 * lam1 * m222 - lam1 * lam2 * pow(r2, 2) + pow(lam3, 2) * pow(r2, 2) +
                    pow(lam4, 2) * pow(r2, 2) + 2 * lam4 * lam5 * pow(r2, 2) + pow(lam5, 2) * pow(r2, 2)
                    + 2 * lam3 * (m112 + (lam4 + lam5) * pow(r2, 2))), -2 * (lam3 + lam4 +
                    lam5) * m122 * pow(r2, 2)});
    gs.push_back(std::vector<double>{c1 * lam1, 0, c1 * (pow(c1, 2) * lam1 +
            2 * m112 + lam3 * pow(r2, 2))});
    gs.push_back(std::vector<double>{(lam3 + lam4 + lam5) * r2, -2 * m122,
            r2 * (pow(c1, 2) * lam3 + 2 * m222 + lam2 * pow(r2, 2))});
    gs.push_back(std::vector<double>{c1 * (lam4 + lam5) * m122 * r2,
            -2 * c1 * pow(m122, 2), 0});
    gs.push_back(std::vector<double>{lam1 * r2 * (2 * m222 + lam2 * pow(r2, 2)),
            2 * (lam3 + lam4 + lam5) * m122 * pow(r2, 2), r2 * (-4 * pow(m122, 2) + (2 * m222 +
                    lam2 * pow(r2, 2)) * (2 * m112 + (lam3 + lam4 + lam5) * pow(r2, 2)) +
                    pow(c1, 2) * (lam3 * (lam4 + lam5) * pow(r2, 2) + lam1 * (2 * m222 +
                            lam2 * pow(r2, 2))))});
    gs.push_back(std::vector<double>{lam1, 0, pow(c1, 2) * lam1 + 2 * m112 +
            (lam3 + lam4 + lam5) * pow(r2, 2), -2 * m122 * r2});

    return gs;
}


// Solving for c1, r1, and r2
std::vector<double> solve_for_c1s(Parameters<double> &params) {
    using std::abs;
    std::vector<double> c1s;
    auto coeffs = get_groebner_basis_coeffs_c1(params);
    auto correct_coeffs = coeffs;
    while (correct_coeffs[0] == 0.0) {
        correct_coeffs.erase(correct_coeffs.begin());
        if (correct_coeffs.empty())break;
    }
    auto complex_c1s = univariate_polynomial_root_finder(correct_coeffs);

    for (int i = 0; i < complex_c1s.rows(); i++) {
        auto cplx = complex_c1s[i];
        if (abs(cplx.imag()) < 1e-5) {
            c1s.push_back(cplx.real());
        }
    }
    return c1s;
}

std::vector<std::tuple<double, double>> solve_for_r2s(Parameters<double> &params, const std::vector<double> &c1s) {
    using std::abs;
    std::vector<std::tuple<double, double>> r2sandc1s;

    for (double c1 : c1s) {
        auto all_coeffs = get_groebner_basis_coeffs_r2(params, c1);
        for (const auto &coeffs : all_coeffs) {
            auto correct_coeffs = coeffs;
            while (correct_coeffs[0] == 0.0) {
                correct_coeffs.erase(correct_coeffs.begin());
                if (correct_coeffs.empty())break;
            }
            if (correct_coeffs.size() > 1) {
                auto complex_r1s = univariate_polynomial_root_finder(correct_coeffs);
                for (int i = 0; i < complex_r1s.rows(); i++) {
                    auto cplx = complex_r1s[i];
                    if (abs(cplx.imag()) < 1e-5) {
                        r2sandc1s.emplace_back(c1, cplx.real());
                    }
                }
            }

        }
    }
    return r2sandc1s;
}

std::vector<std::tuple<double, double, double>> solve_for_r1s(
        Parameters<double> &params, const std::vector<std::tuple<double, double>> &r1sandc1s) {

    std::vector<std::tuple<double, double, double>> sols;
    for (auto pair: r1sandc1s) {
        double c1 = std::get<0>(pair);
        double r2 = std::get<1>(pair);
        auto all_coeffs = get_groebner_basis_coeffs_r1(params, c1, r2);
        for (const auto &coeffs : all_coeffs) {
            auto correct_coeffs = coeffs;
            while (correct_coeffs[0] == 0.0) {
                correct_coeffs.erase(correct_coeffs.begin());
                if (correct_coeffs.empty()) break;
            }
            if (correct_coeffs.size() > 1) {
                auto complex_r2s = univariate_polynomial_root_finder(correct_coeffs);
                for (int i = 0; i < complex_r2s.rows(); i++) {
                    auto cplx = complex_r2s[i];
                    if (abs(cplx.imag()) < 1e-5) {
                        sols.emplace_back(c1, r2, cplx.real());
                    }
                }
            }
        }
    }

    return sols;
}

// Tadpole equations
double r1_equation(double r1, double r2, double c1, Parameters<double> &params) {
    double m112 = params.m112;
    double m122 = params.m122;
    double lam1 = params.lam1;
    double lam3 = params.lam3;
    double lam4 = params.lam4;
    double lam5 = params.lam5;

    return (pow(c1, 2) * lam1 * r1) / 2. + m112 * r1 + (lam1 * pow(r1, 3)) / 2. - m122 * r2 +
            (lam3 * r1 * pow(r2, 2)) / 2. + (lam4 * r1 * pow(r2, 2)) / 2. +
            (lam5 * r1 * pow(r2, 2)) / 2.;
}

double r2_equation(double r1, double r2, double c1, Parameters<double> &params) {
    double m122 = params.m122;
    double m222 = params.m222;
    double lam2 = params.lam2;
    double lam3 = params.lam3;
    double lam4 = params.lam4;
    double lam5 = params.lam5;

    return -(m122 * r1) + (pow(c1, 2) * lam3 * r2) / 2. + m222 * r2 + (lam3 * pow(r1, 2) * r2) / 2. +
            (lam4 * pow(r1, 2) * r2) / 2. +
            (lam5 * pow(r1, 2) * r2) / 2. + (lam2 * pow(r2, 3)) / 2.;
}

double c1_equation(double r1, double r2, double c1, Parameters<double> &params) {
    double m112 = params.m112;
    double lam1 = params.lam1;
    double lam3 = params.lam3;

    return (pow(c1, 3) * lam1) / 2. + c1 * m112 + (c1 * lam1 * pow(r1, 2)) / 2. + (c1 * lam3 * pow(r2, 2)) / 2.;
}

// Get all tree roots
std::vector<std::tuple<double, double, double>> get_real_tree_roots(Parameters<double> &params) {
    using std::abs;
    auto c1s = solve_for_c1s(params);
    auto c1sandr1s = solve_for_r2s(params, c1s);
    auto sols = solve_for_r1s(params, c1sandr1s);

    std::vector<std::tuple<double, double, double>> real_sols;
    std::vector<std::tuple<double, double, double>> real_sols_no_dup;

    for (auto tup: sols) {
        double r1 = std::get<2>(tup);
        double r2 = std::get<1>(tup);
        double c1 = std::get<0>(tup);
        double eqn1 = r1_equation(r1, r2, c1, params);
        double eqn2 = r2_equation(r1, r2, c1, params);
        double eqn3 = c1_equation(r1, r2, c1, params);

        if (abs(eqn1) < 1e-7 && abs(eqn2) < 1e-7 && abs(eqn3) < 1e-7) {
            real_sols.emplace_back(r1, r2, c1);
        }
    }

    // Get unique roots
    for (auto root : real_sols) {
        if (real_sols_no_dup.empty()) {
            real_sols_no_dup.push_back(root);
        } else {
            bool should_add = true;
            for (auto unique_root : real_sols_no_dup) {
                if ((abs(std::get<0>(root) - std::get<0>(unique_root)) < 1e-5) &&
                        (abs(std::get<1>(root) - std::get<1>(unique_root)) < 1e-5) &&
                        (abs(std::get<2>(root) - std::get<2>(unique_root)) < 1e-5)) {
                    should_add = false;
                    break;
                }
            }
            if (should_add)
                real_sols_no_dup.push_back(root);
        }
    }
    return real_sols_no_dup;
}

} // namespace thdm

#endif // THDM_TREE_ROOTS_EXPLICIT_HPP
