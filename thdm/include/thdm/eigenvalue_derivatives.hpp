//
// Created by Logan Morrison on 2019-04-29.
//

#ifndef THDM_EIGENVALUE_DERIVATIVES_HPP
#define THDM_EIGENVALUE_DERIVATIVES_HPP

#include <armadillo>
#include <cmath>
#include <tuple>
#include <assert.h>
#include <vector>

namespace thdm {

static double dF_dlam(const arma::mat &M, const double lam) {
    // Create identity matrix
    arma::mat I = arma::mat(M.n_rows, M.n_cols, arma::fill::eye);
    // Create matrix that goes into determinant equation:
    //      det(A) = det(M - lam I) = 0
    arma::mat A = M - lam * I;

    double deriv = 0.0;

    // Replace the ith column with -I_{:i}
    for (size_t i = 0; i < M.n_rows; i++) {
        arma::mat Atemp = A;
        Atemp.col(i) = -I.col(i);
        deriv += arma::det(Atemp);
    }

    return deriv;
}

static double d2F_dlamd2(const arma::mat &M, const double lam) {
    // Create identity matrix
    arma::mat I = arma::mat(M.n_rows, M.n_cols, arma::fill::eye);
    // Create matrix that goes into determinant equation:
    //      det(A) = det(M - lam I) = 0
    arma::mat A = M - lam * I;

    double deriv = 0.0;

    // Replace the ith column with -I_{:i} and jth column
    // with dM[:j]/dp compute determinant. Skip i=j since we would be taking
    // dI/dp = 0.
    for (size_t i = 0; i < M.n_rows; i++) {
        for (size_t j = 0; j < M.n_cols; j++) {
            if (i != j) {
                arma::mat Atemp = A;
                Atemp.col(i) = -I.col(i);
                Atemp.col(j) = -I.col(j);
                deriv += arma::det(Atemp);
            }
        }
    }

    return deriv;
}

static double d2F_dlamdp(const arma::mat &M, const arma::mat &dM,
                         const double lam) {
    // Create identity matrix
    arma::mat I = arma::mat(M.n_rows, M.n_cols, arma::fill::eye);
    // Create matrix that goes into determinant equation:
    //      det(A) = det(M - lam I) = 0
    arma::mat A = M - lam * I;

    double deriv = 0.0;

    // Replace the ith column with -I_{:i} and jth column
    // with dM[:j]/dp compute determinant. Skip i=j since we would be taking
    // dI/dp = 0.
    for (size_t i = 0; i < M.n_rows; i++) {
        for (size_t j = 0; j < M.n_cols; j++) {
            if (i != j) {
                arma::mat Atemp = A;
                Atemp.col(i) = dM.col(i);
                Atemp.col(j) = -I.col(j);
                deriv += arma::det(Atemp);
            }
        }
    }

    return deriv;
}

static double d2F_dpidpj(const arma::mat &M,
                         const arma::mat &dM1,
                         const arma::mat &dM2,
                         const arma::mat &d2M,
                         const double lam) {
    // Create identity matrix
    arma::mat I = arma::mat(M.n_rows, M.n_cols, arma::fill::eye);
    // Create matrix that goes into determinant equation:
    //      det(A) = det(M - lam I) = 0
    arma::mat A = M - lam * I;

    double deriv = 0.0;

    // Replace the ith column with dM[:j]/dp_1 and jth column
    // with dM[:j]/dp_2 compute determinant. If i=j, we take the second
    // derivative: d^2M/dp_1 dp_2.
    for (size_t i = 0; i < M.n_rows; i++) {
        for (size_t j = 0; j < M.n_cols; j++) {
            if (i != j) {
                arma::mat Atemp = A;
                Atemp.col(i) = dM1.col(i);
                Atemp.col(j) = dM2.col(j);
                deriv += arma::det(Atemp);
            }
            if (i == j) {
                arma::mat Atemp = A;
                Atemp.col(i) = d2M.col(i);
                deriv += arma::det(Atemp);
            }
        }
    }

    return deriv;
}

/**
 * Compute the derivatives of the eigenvalues and eigenvectors
 * of a given matrix with respect to a variable
 * @param M Input matrix
 * @param dM Derivative of input matrix
 * @return Derivatives of eigenvalues and eigenvectors.
 */
std::vector<std::tuple<double, double>> eigenvalue_first_derivative(const arma::mat &M, const arma::mat &dM) {
    assert(M.n_cols == M.n_rows);
    arma::vec evals;
    arma::mat evecs;
    arma::eig_sym(evals, evecs, M);
    std::vector<std::tuple<double, double>> res;

    for (size_t i = 0; i < M.n_cols; i++) {
        res.emplace_back(evals(i), arma::dot(evecs.col(i), dM * evecs.col(i)));
    }
    return res;
}

/**
 * Compute the derivatives of the eigenvalues and eigenvectors
 * of a given matrix with respect to two variables
 * @param M Input matrix
 * @param dM1 1st derivative of input matrix wrt var 1
 * @param dM2 1st derivative of input matrix wrt var 2
 * @param d2M 2nd derivative of input matrix wrt var 1 and var 2
 * @return
 */
std::vector<std::tuple<double, double, double, double>> eigenvalue_first_second_derivative(
        const arma::mat &M, const arma::mat &dM1,
        const arma::mat &dM2, const arma::mat &d2M) {
    assert(M.n_cols == M.n_rows);
    // First derivatives
    auto res1 = eigenvalue_first_derivative(M, dM1);
    auto res2 = eigenvalue_first_derivative(M, dM2);
    std::vector<std::tuple<double, double, double, double >> res;

    for (size_t i = 0; i < res1.size(); i++) {
        double lam = std::get<0>(res1[i]);
        double dlam1 = std::get<1>(res1[i]);
        double dlam2 = std::get<1>(res2[i]);
        double d2lam;

        double df_dlam = dF_dlam(M, lam);
        // Seconds derivatives
        double d2f_dlam2 = d2F_dlamd2(M, lam);
        double d2f_dlamdp1 = d2F_dlamdp(M, dM1, lam);
        double d2f_dlamdp2 = d2F_dlamdp(M, dM2, lam);
        double d2f_dp1dp2 = d2F_dpidpj(M, dM1, dM2, d2M, lam);

        d2lam = -pow(df_dlam, -1) * (d2f_dlam2 * dlam1 * dlam2 +
                d2f_dlamdp2 * dlam1 +
                d2f_dlamdp1 * dlam2 +
                d2f_dp1dp2);
        res.emplace_back(lam, dlam1, dlam2, d2lam);
    }
    return res;
}


}
#endif //THDM_EIGENVALUE_DERIVATIVES_HPP
