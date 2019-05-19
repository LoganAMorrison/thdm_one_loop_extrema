#include "thdm/dual.hpp"
#include <cmath>
#include <functional>
#include <gtest/gtest.h>
#include <iostream>
#include <random>

using namespace thdm;

TEST(DualTest, FirstDerivativeTest) {
    std::random_device rd{};
    std::mt19937 engine{rd()};
    std::uniform_real_distribution<double> dist{0.0, 100.0};

    for (int i = 0; i < 100; i++) {
        double x = dist(engine);
        double y = dist(engine);
        Dual<double> z = Dual<double>(x, 1.0);
        auto powz1 = pow(z, y);
        auto powz2 = pow(y, z);
        auto sqrtz = sqrt(z);
        auto sinz = sin(z);
        auto cosz = cos(z);
        auto tanz = tan(z);

        // Test derivative and value of pow(x)
        ASSERT_NEAR(powz1.val, pow(x, y), 1e-10);
        ASSERT_NEAR(powz1.eps, y * pow(x, y - 1), 1e-10);
        // Test derivative and value of pow(x)
        ASSERT_NEAR(powz2.val, pow(y, x), 1e-10);
        ASSERT_NEAR(powz2.eps, log(y) * pow(y, x), 1e-10);
        // Test derivative and value of sqrt(x)
        ASSERT_NEAR(sqrtz.val, sqrt(x), 1e-10);
        ASSERT_NEAR(sqrtz.eps, 1.0 / (2.0 * sqrt(x)), 1e-10);
        // Test derivative and value of sin(x)
        ASSERT_NEAR(sinz.val, sin(x), 1e-10);
        ASSERT_NEAR(sinz.eps, cos(x), 1e-10);
        // Test derivative and value of cos(x)
        ASSERT_NEAR(cosz.val, cos(x), 1e-10);
        ASSERT_NEAR(cosz.eps, -sin(x), 1e-10);
        // Test derivative and value of tan(x)
        ASSERT_NEAR(tanz.val, tan(x), 1e-10);
        ASSERT_NEAR(tanz.eps, pow(cos(x), -2), 1e-10);
    }
}

TEST(DualTest, NestedDuals) {
    std::random_device rd{};
    std::mt19937 engine{rd()};
    std::uniform_real_distribution<double> dist{0.0, 100.0};

    double x = dist(engine);
    // double y = dist(engine);
    auto z = Dual<Dual<double >>{Dual<double>{x, 1.0}, Dual<double>{1.0, 0.0}};
    auto sqrtz = sqrt(z);
    auto sinz = sin(z);
    auto cosz = cos(z);
    auto tanz = tan(z);
    auto expz = exp(z);

    // Test derivative and value of sqrt(x)
    ASSERT_NEAR(sqrtz.val.val, sqrt(x), 1e-10);
    ASSERT_NEAR(sqrtz.val.eps, 1.0 / (2.0 * sqrt(x)), 1e-10);
    ASSERT_NEAR(sqrtz.eps.val, 1.0 / (2.0 * sqrt(x)), 1e-10);
    ASSERT_NEAR(sqrtz.eps.eps, -1.0 / (4.0 * pow(x, 1.5)), 1e-10);
    // Test derivative and value of sin(x)
    ASSERT_NEAR(sinz.val.val, sin(x), 1e-10);
    ASSERT_NEAR(sinz.val.eps, cos(x), 1e-10);
    ASSERT_NEAR(sinz.eps.val, cos(x), 1e-10);
    ASSERT_NEAR(sinz.eps.eps, -sin(x), 1e-10);
    // Test derivative and value of cos(x)
    ASSERT_NEAR(cosz.val.val, cos(x), 1e-10);
    ASSERT_NEAR(cosz.val.eps, -sin(x), 1e-10);
    ASSERT_NEAR(cosz.eps.val, -sin(x), 1e-10);
    ASSERT_NEAR(cosz.eps.eps, -cos(x), 1e-10);
    // Test derivative and value of cos(x)
    ASSERT_NEAR(tanz.val.val, tan(x), 1e-10);
    ASSERT_NEAR(tanz.val.eps, 1.0 / pow(cos(x), 2), 1e-10);
    ASSERT_NEAR(tanz.eps.val, 1.0 / pow(cos(x), 2), 1e-10);
    ASSERT_NEAR(tanz.eps.eps, 2.0 * tan(x) / pow(cos(x), 2), 1e-10);
    // Test derivative and value of exp(x)
    ASSERT_NEAR(expz.val.val, exp(x), 1e-10);
    ASSERT_NEAR(expz.val.eps, exp(x), 1e-10);
    ASSERT_NEAR(expz.eps.val, exp(x), 1e-10);
    ASSERT_NEAR(expz.eps.eps, exp(x), 1e-10);
}

// Test duals to integer powers
TEST(DualTest, TestIntegerPower) {
    auto x = Dual<double>{2.0, 1.0};

    std::cout << "x^0 = " << pow(x, 0) << std::endl;
    std::cout << "x^1 = " << pow(x, 1) << std::endl;
    std::cout << "x^2 = " << pow(x, 2) << std::endl;
    std::cout << "x^3 = " << pow(x, 3) << std::endl;
    std::cout << "x^-1 = " << pow(x, -1) << std::endl;
    std::cout << "x^-2 = " << pow(x, -2) << std::endl;
    std::cout << "x^-3 = " << pow(x, -3) << std::endl;

    auto y = Dual<Dual<double >>{Dual<double>{2.0, 1.0}, Dual<double>{1.0, 0.0}};
    std::cout << "y^0 = " << pow(y, 0) << std::endl;
    std::cout << "y^1 = " << pow(y, 1) << std::endl;
    std::cout << "y^2 = " << pow(y, 2) << std::endl;
    std::cout << "y^3 = " << pow(y, 3) << std::endl;
    std::cout << "y^-1 = " << pow(y, -1) << std::endl;
    std::cout << "y^-2 = " << pow(y, -2) << std::endl;
    std::cout << "y^-3 = " << pow(y, -3) << std::endl;
}

TEST(DualTest, TestFABS) {
    auto x = Dual<double>{2.0, 1.0};

    std::cout << std::endl << "abs(x) " << fabs(x) << std::endl;

    auto y = Dual<Dual<double >>{Dual<double>{2.0, 1.0}, Dual<double>{1.0, 0.0}};
    std::cout << "abs(y) " << fabs(y) << std::endl;

    auto w = Dual<Dual<double >>{Dual<double>{-2.0, 1.0}, Dual<double>{1.0, 0.0}};
    std::cout << "abs(w) " << fabs(w) << std::endl;
}

TEST(DualTest, TestSqrt) {
    auto x = Dual<double>{2.0, 1.0};
    std::cout << std::endl << "1 / (fabs(x) + sqrt(1 + x*x) " << 1 / (fabs(x) + sqrt(1 + x * x)) << std::endl;

    auto y = Dual<Dual<double >>{Dual<double>{2.0, 1.0}, Dual<double>{1.0, 0.0}};
    std::cout << "1 / (fabs(y) + sqrt(1 + y*y)) " << 1 / (fabs(y) + sqrt(1 + y * y)) << std::endl;
}

TEST(DualTest, TestMult) {

    double x = 0.0;
    // double y = dist(engine);
    auto z = Dual<Dual<double >>{Dual<double>{x, 1.0}, Dual<double>{1.0, 0.0}};
    auto sqrtz = sqrt(1 + z);
    auto sinz = sin(z);
    auto cosz = cos(z);
    auto tanz = tan(z);
    auto expz = exp(z);

    // Test derivative and value of sqrt(x)
    ASSERT_NEAR(sqrtz.val.val, sqrt(x + 1), 1e-10);
    ASSERT_NEAR(sqrtz.val.eps, 1.0 / (2.0 * sqrt(1 + x)), 1e-10);
    ASSERT_NEAR(sqrtz.eps.val, 1.0 / (2.0 * sqrt(1 + x)), 1e-10);
    ASSERT_NEAR(sqrtz.eps.eps, -1.0 / (4.0 * pow(x + 1, 1.5)), 1e-10);
    // Test derivative and value of sin(x)
    ASSERT_NEAR(sinz.val.val, sin(x), 1e-10);
    ASSERT_NEAR(sinz.val.eps, cos(x), 1e-10);
    ASSERT_NEAR(sinz.eps.val, cos(x), 1e-10);
    ASSERT_NEAR(sinz.eps.eps, -sin(x), 1e-10);
    // Test derivative and value of cos(x)
    ASSERT_NEAR(cosz.val.val, cos(x), 1e-10);
    ASSERT_NEAR(cosz.val.eps, -sin(x), 1e-10);
    ASSERT_NEAR(cosz.eps.val, -sin(x), 1e-10);
    ASSERT_NEAR(cosz.eps.eps, -cos(x), 1e-10);
    // Test derivative and value of cos(x)
    ASSERT_NEAR(tanz.val.val, tan(x), 1e-10);
    ASSERT_NEAR(tanz.val.eps, 1.0 / pow(cos(x), 2), 1e-10);
    ASSERT_NEAR(tanz.eps.val, 1.0 / pow(cos(x), 2), 1e-10);
    ASSERT_NEAR(tanz.eps.eps, 2.0 * tan(x) / pow(cos(x), 2), 1e-10);
    // Test derivative and value of exp(x)
    ASSERT_NEAR(expz.val.val, exp(x), 1e-10);
    ASSERT_NEAR(expz.val.eps, exp(x), 1e-10);
    ASSERT_NEAR(expz.eps.val, exp(x), 1e-10);
    ASSERT_NEAR(expz.eps.eps, exp(x), 1e-10);
}


int main(int argc, char *argv[]) {

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}