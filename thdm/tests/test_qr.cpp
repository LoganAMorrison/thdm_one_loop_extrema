#include "thdm/dual.hpp"
#include "thdm/qr.hpp"
#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include <random>
#include <functional>
#include <vector>
#include <numeric>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace thdm;
using namespace Eigen;
//using namespace duals;

template<class T, int N, int M> using MatrixXT = Matrix<T, N, M>;
template<class T, int N> using VectorXT = Matrix<T, N, 1>;

template<class T, int N, int M>
Matrix<T, 2, 2> test_matrix(T a, T b) {
    Matrix<T, 2, 2> mat(2, 2);
    mat(0, 0) = exp(exp(sin(a)));
    mat(0, 1) = log(sqrt(b));
    mat(1, 0) = log(sqrt(b));
    mat(1, 1) = sin(cos(exp(cos(a * b))));
    return mat;
}

template<class T>
Matrix<T, 2, 2> test_matrix2(T a, T b) {
    Matrix<T, 2, 2> mat;
    mat(0, 0) = a * a;
    mat(0, 1) = a * b;
    mat(1, 0) = a * b;
    mat(1, 1) = b * b * b;
    return mat;
}


TEST(QRTest, Test1) {
    Dual<double> a{1.0, 1.0};
    Dual<double> b{2.0, 0.0};

    auto mat = test_matrix<Dual<double>, 2, 2>(a, b);

    auto res = eigenvalues_qr(mat);

    std::cout << std::setprecision(15) << std::endl;
    std::cout << res << std::endl;

}

TEST(QRTest, Test2) {
    Dual<Dual<double>> a{Dual<double>{1e-10, 1.0}, Dual<double>{1.0, 0.0}};
    Dual<Dual<double>> b{Dual<double>{3.0, 0.0}, Dual<double>{0.0, 0.0}};

    auto mat = test_matrix2<Dual<Dual<double>>>(a, b);

    auto res = eigenvalues_qr(mat);

    std::cout << std::setprecision(15) << std::endl;
    std::cout << res << std::endl;

}


int main(int argc, char *argv[]) {

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}