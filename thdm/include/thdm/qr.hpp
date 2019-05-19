//
// Created by Logan Morrison on 2019-05-01.
//

#ifndef THDM_QR_HPP
#define THDM_QR_HPP

#include "thdm/dual.hpp"
#include <Eigen/Dense>
#include <tuple>
#include <exception>
#include <iostream>
#include <cmath>

using namespace Eigen;

namespace thdm {

template<class T>
inline T SIGN(const T &a, const T &b) {
    return b >= static_cast<T>(0) ? (a >= static_cast<T>(0) ? a : -a) : (a >= static_cast<T>(0) ? -a : a);
}

template<class T, int N>
class QRdcmp {
public:
    Matrix<T, N, N> qt, r;
    bool sing;

    QRdcmp(Matrix<T, N, N> &a)
            : qt(N, N), r(a), sing(false) {
        int i, j, k;
        Matrix<T, N, 1> c, d;
        T scale, sigma, sum, tau;
        for (k = 0; k < N - 1; k++) {
            scale = static_cast<T>(0);
            for (i = k; i < N; i++)
                scale = (scale > abs(r(i, k))) ? scale : abs(r(i, k));
            if (scale == static_cast<T>(0)) {
                sing = true;
                c(k) = d(k) = static_cast<T>(0);
            } else {
                for (i = k; i < N; i++)
                    r(i, k) /= scale;
                for (sum = 0.0, i = k; i < N; i++)
                    sum += r(i, k) * r(i, k);
                sigma = SIGN(sqrt(sum), r(k, k));
                r(k, k) += sigma;
                c(k) = sigma * r(k, k);
                d(k) = -scale * sigma;
                for (j = k + 1; j < N; j++) {
                    for (sum = 0.0, i = k; i < N; i++) sum += r(i, k) * r(i, j);
                    tau = sum / c(k);
                    for (i = k; i < N; i++) r(i, j) -= tau * r(i, k);
                }
            }
        }
        d(N - 1) = r(N - 1, N - 1);
        if (d(N - 1) == static_cast<T>(0)) sing = true;
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++)
                qt(i, j) = static_cast<T>(0);
            qt(i, i) = 1.0;
        }
        for (k = 0; k < N - 1; k++) {
            if (c(k) != 0.0) {
                for (j = 0; j < N; j++) {
                    sum = 0.0;
                    for (i = k; i < N; i++)
                        sum += r(i, k) * qt(i, j);
                    sum /= c(k);
                    for (i = k; i < N; i++)
                        qt(i, j) -= sum * r(i, k);
                }
            }
        }
        for (i = 0; i < N; i++) {
            r(i, i) = d(i);
            for (j = 0; j < i; j++) r(i, j) = static_cast<T>(0);
        }
    }

    void solve(Matrix<T, N, 1> &b, Matrix<T, N, 1> &x) {
        qtmult(b, x);
        rsolve(x, x);
    }

    void qtmult(Matrix<T, N, 1> &b, Matrix<T, N, 1> &x) {
        int i, j;
        T sum;
        for (i = 0; i < N; i++) {
            sum = 0.;
            for (j = 0; j < N; j++) sum += qt(i, j) * b(j);
            x(i) = sum;
        }
    }

    void rsolve(Matrix<T, N, 1> &b, Matrix<T, N, 1> &x) {
        int i, j;
        T sum;
        if (sing) throw ("attempting solve in a singular QR");
        for (i = N - 1; i >= 0; i--) {
            sum = b(i);
            for (j = i + 1; j < N; j++) sum -= r(i, j) * x(j);
            x(i) = sum / r(i, i);
        }
    }

    void rotate(const int i, const double a, const T b) {
        int j;
        T c, fact, s, w, y;
        if (a == 0.0) {
            c = 0.0;
            s = (b >= 0.0 ? 1.0 : -1.0);
        } else if (abs(a) > abs(b)) {
            fact = b / a;
            c = SIGN(1.0 / sqrt(1.0 + (fact * fact)), a);
            s = fact * c;
        } else {
            fact = a / b;
            s = SIGN(1.0 / sqrt(1.0 + (fact * fact)), b);
            c = fact * s;
        }
        for (j = i; j < N; j++) {
            y = r(i, j);
            w = r(i + 1, j);
            r(i, j) = c * y - s * w;
            r(i + 1)(j) = s * y + c * w;
        }
        for (j = 0; j < N; j++) {
            y = qt(i, j);
            w = qt(i + 1, j);
            qt(i, j) = c * y - s * w;
            qt(i + 1, j) = s * y + c * w;
        }
    }
};


template<class T, int N, int M>
Matrix<T, N, 1> eigenvalues_qr(const Matrix<T, N, M> &mat) {
    Matrix<T, N, M> A(mat);
    Matrix<T, N, M> Aold(mat);

    for (int i = 0; i < 100; i++) {
        QRdcmp<T, N> comp(Aold);
        std::cout << i << std::endl;
        auto Q = comp.qt.transpose();
        auto R = comp.r;
        A = R * Q;
        if (i > 5) {
            T normsqr;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    normsqr += (A(i, j) - Aold(i, j)) * (A(i, j) - Aold(i, j));
                }
            }
            if (normsqr == static_cast<T>(0)) {
                return A.diagonal();
            } else if (sqrt(normsqr) < 1e-12) {
                return A.diagonal();
            }
        }
        Aold = A;
    }
    return A.diagonal();
}
}

#endif //THDM_QR_HPP
