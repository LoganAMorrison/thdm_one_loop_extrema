#include "thdm/jacobi.hpp"
#include "thdm/dual.hpp"
#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include <random>
#include <functional>
#include <vector>
#include <numeric>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

using namespace thdm;
using namespace Eigen;
using namespace thdm;


template<class T>
std::vector<std::vector<T>> test_matrix(T a, T b) {
    std::vector<std::vector<T>> mat(2, std::vector<T>(2));
    mat[0][0] = exp(exp(sin(a)));
    mat[0][1] = log(sqrt(b));
    mat[1][0] = log(sqrt(b));
    mat[1][1] = sin(cos(exp(cos(a * b))));
    return mat;
}


TEST(DualTest, FirstDerivativeTest1) {
    Dual<double> a{2.0, 0.0};
    Dual<double> b{3.0, 0.0};

    a.eps = 1.0;
    std::vector<std::vector<Dual<double>>> mat1 = test_matrix(a, b);
    auto res1 = jacobi(mat1);
    std::cout << res1[0] << std::endl;
    std::cout << res1[1] << std::endl;

    a.eps = 0.0;
    b.eps = 1.0;
    std::vector<std::vector<Dual<double>>> mat2 = test_matrix(a, b);
    auto res2 = jacobi(mat2);
    std::cout << res2[0] << std::endl;
    std::cout << res2[1] << std::endl;
}

TEST(DualTest, FirstDerivativeTest2) {
    Dual<double> a{0.0, 0.0};
    Dual<double> b{3.0, 0.0};

    a.eps = 1.0;
    std::vector<std::vector<Dual<double>>> mat1 = test_matrix(a, b);
    auto res1 = jacobi(mat1);
    std::cout << res1[0] << std::endl;
    std::cout << res1[1] << std::endl;

    a.eps = 0.0;
    b.eps = 1.0;
    std::vector<std::vector<Dual<double>>> mat2 = test_matrix(a, b);
    auto res2 = jacobi(mat2);
    std::cout << res2[0] << std::endl;
    std::cout << res2[1] << std::endl;
}

TEST(DualTest, HessianTest1) {
    Dual<Dual<double>> a;
    Dual<Dual<double>> b;
    a.val.val = 2.0;
    b.val.val = 3.0;

    a.val.eps = 0.0;
    a.eps.val = 0.0;
    a.eps.eps = 0.0;
    b.val.eps = 0.0;
    b.eps.val = 0.0;
    b.eps.eps = 0.0;

    std::cout << std::endl;
    // d^2lam/da^2
    a.val.eps = 1.0;
    a.eps.val = 1.0;
    std::vector<std::vector<Dual<Dual<double>>>> mat1 = test_matrix(a, b);
    auto res1 = jacobi(mat1);
    std::cout << res1[0] << std::endl;
    std::cout << res1[1] << std::endl;

    // d^2lam/da db
    a.val.eps = 1.0;
    a.eps.val = 0.0;
    b.val.eps = 0.0;
    b.eps.val = 1.0;
    std::vector<std::vector<Dual<Dual<double>>>> mat2 = test_matrix(a, b);
    auto res2 = jacobi(mat2);
    std::cout << res2[0] << std::endl;
    std::cout << res2[1] << std::endl;

    // d^2lam/db da
    a.val.eps = 0.0;
    a.eps.val = 1.0;
    b.val.eps = 1.0;
    b.eps.val = 0.0;
    std::vector<std::vector<Dual<Dual<double>>>> mat3 = test_matrix(a, b);
    auto res3 = jacobi(mat3);
    std::cout << res3[0] << std::endl;
    std::cout << res3[1] << std::endl;

    // d^2lam/db^2
    a.val.eps = 0.0;
    a.eps.val = 0.0;
    b.val.eps = 1.0;
    b.eps.val = 1.0;
    std::vector<std::vector<Dual<Dual<double>>>> mat4 = test_matrix(a, b);
    auto res4 = jacobi(mat4);
    std::cout << res4[0] << std::endl;
    std::cout << res4[1] << std::endl;
}


int main(int argc, char *argv[]) {

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}