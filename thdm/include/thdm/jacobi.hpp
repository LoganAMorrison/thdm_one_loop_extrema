#ifndef JACOBI_HPP
#define JACOBI_HPP

#include "thdm/errors.hpp"
#include <numeric>
#include <cmath>
#include <vector>
#include <limits>

namespace thdm {
/* Maximum iterations for Jacobi algorithm */
static constexpr int const JACOBI_MAX_ITER = 100;


/**
 * Rotate two of the components of a matrix.
 * @tparam T
 * @param mat
 * @param s
 * @param tau
 * @param i
 * @param j
 * @param k
 * @param l
 */
template<class T>
inline void rot(std::vector<std::vector<T>> &mat, const T s, const T tau,
                const int i, const int j, const int k, const int l) {
    T g = mat[i][j];
    T h = mat[k][l];
    mat[i][j] = g - s * (h + g * tau);
    mat[k][l] = h + s * (g - h * tau);
}

/**
 * Compute the eigenvalues of a symmetric matrix using the jacobi
 * algorithm.
 * @tparam T double, Dual<double>, or Dual<Dual<double>>
 * @param mmat matrix to compute dual of.
 * @return eigenvalues.
 */
template<class T>
std::vector<T> jacobi(const std::vector<std::vector<T>> &mmat) {
    // Copy the matrix so we don't destroy original
    std::vector<std::vector<T>> mat(mmat);
    int n = mat.size();

    std::vector<std::vector<T>> v(n, std::vector<T>(n, static_cast<T>(0)));
    std::vector<T> d(n, static_cast<T>(0));
    std::vector<T> b(n, static_cast<T>(0));
    std::vector<T> z(n, static_cast<T>(0));

    int nrot = 0;
    const double EPS = std::numeric_limits<double>::epsilon();

    int i, j, ip, iq;
    T tresh, theta, tau, t, sm, s, h, g, c;

    for (ip = 0; ip < n; ip++) {
        for (iq = 0; iq < n; iq++)
            v[ip][iq] = static_cast<T>(0);
        v[ip][ip] = static_cast<T>(1);
    }
    for (ip = 0; ip < n; ip++) {
        b[ip] = d[ip] = mat[ip][ip];
        z[ip] = static_cast<T>(0);
    }
    for (i = 1; i <= JACOBI_MAX_ITER; i++) {
        sm = static_cast<T>(0);
        for (ip = 0; ip < n - 1; ip++) {
            for (iq = ip + 1; iq < n; iq++)
                sm += abs(mat[ip][iq]);
        }
        if (sm == 0) {
            return d;
        }
        if (i < 4)
            tresh = 0.2 * sm / (n * n);
        else
            tresh = static_cast<T>(0);
        for (ip = 0; ip < n - 1; ip++) {
            for (iq = ip + 1; iq < n; iq++) {
                g = 100 * abs(mat[ip][iq]);
                if (i > 4 && g <= EPS * abs(d[ip]) && g <= EPS * abs(d[iq]))
                    mat[ip][iq] = static_cast<T>(0);
                else if (abs(mat[ip][iq]) > tresh) {
                    h = d[iq] - d[ip];
                    if (g <= EPS * abs(h))
                        t = (mat[ip][iq]) / h;
                    else {
                        theta = 0.5 * h / (mat[ip][iq]);
                        t = 1 / (abs(theta) + sqrt(1 + theta * theta));
                        if (theta < 0)
                            t = -t;
                    }
                    c = 1 / sqrt(1 + t * t);
                    s = t * c;
                    tau = s / (1 + c);
                    h = t * mat[ip][iq];
                    z[ip] -= h;
                    z[iq] += h;
                    d[ip] -= h;
                    d[iq] += h;
                    mat[ip][iq] = static_cast<T>(0);
                    for (j = 0; j < ip; j++)
                        rot(mat, s, tau, j, ip, j, iq);
                    for (j = ip + 1; j < iq; j++)
                        rot(mat, s, tau, ip, j, j, iq);
                    for (j = iq + 1; j < n; j++)
                        rot(mat, s, tau, ip, j, iq, j);
                    for (j = 0; j < n; j++)
                        rot(v, s, tau, j, ip, j, iq);
                    ++nrot;
                }
            }
        }
        for (ip = 0; ip < n; ip++) {
            b[ip] += z[ip];
            d[ip] = b[ip];
            z[ip] = static_cast<T>(0);
        }
    }
    throw THDMException(THDMExceptionCode::JacobiTooManyIterations);
}

} // namespace thdm

#endif // JACOBI_HPP