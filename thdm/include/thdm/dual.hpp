#ifndef Dual_HPP
#define Dual_HPP

#include "thdm/errors.hpp"
#include <iostream>
#include <cmath>
#include <string>
#include <Eigen/Core>

namespace thdm {


/**
 * Class to implement dual numbers in c++
 *
 * A dual number `z` is defined as: z = a + b eps, where `a` and `b`
 * are real numbers and `eps` is defined such that eps * eps = 0.
 * These are useful because, if we evaluate a function at a dual
 * number, we get: f(z) = f(a) + b * f'(a) * eps. That is, we can
 * obtain the derivative of the function by grabbing the eps
 * component of the result of f(z). We can also get higher-order
 * derivative by nesting dual numbers. For example, if x = z + w * eps
 * and z = x0 + eps and w = 1 + 0 * eps, then
 * f(z) = ((f(x0), f'(x0)), (f'(x0), f''(x0))). By templating the class,
 * we can nest dual numbers.
 *
 * @tparam T
 */
template<class T>
class Dual {
public:
    /**
     * The real component of the dual number.
     */
    T val;
    /**
     * The infinitesimal component of the dual number.
     * This component is define such that eps * eps = 0.
     */
    T eps;

    /**
     * Default construct for a dual number. Both `val` and `eps` are set
     * to zero.
     */
    Dual() : val(static_cast<T>(0)), eps(static_cast<T>(0)) {
    }

    /**
     * Constructor to set real component of the dual number. `eps` is
     * set to zero.
     * @param val real component of the dual number.
     */
    template<class U>
    explicit Dual(U val) : val(static_cast<T>(val)), eps(static_cast<T>(0)) {}

    explicit Dual(T val) : val(val), eps(static_cast<T>(0)) {}

    /**
     * Full constructor for a dual number.
     * @param val real component of the dual number.
     * @param eps infinitesimal component of the dual number.
     */
    template<class U1, class U2>
    Dual(U1 val, U2 eps) : val(static_cast<T>(val)), eps(static_cast<T>(eps)) {}

    /**
     * Copy constructor for a dual number.
     * @param d the dual number to copy.
     */
    template<class U>
    explicit Dual(const Dual<U> &d) : val(static_cast<T>(d.val)), eps(static_cast<T>(d.eps)) {}

    Dual(const Dual<T> &d) : val(d.val), eps(d.eps) {}

    /**
     * Default destructor.
     */
    ~Dual() = default;

    Dual<T> &operator=(const Dual<T> &z) {
        if (this == &z)
            return *this;
        val = z.val;
        eps = z.eps;
        return *this;
    }

    template<class U>
    Dual<T> &operator=(const U &z) {
        auto res = static_cast<Dual<T>>(z);
        val = res.val;
        eps = res.eps;
        return *this;
    }

    /**
     * Overload of the stream operator to print dual number.
     * @param os stream.
     * @param z dual number.
     * @return stream.
     */
    friend std::ostream &operator<<(std::ostream &os, const Dual<T> &z) {
        os << "(" << z.val << ", " << z.eps << ")";
        return os;
    }


    /**
     * Comparison operator for two dual numbers.
     *
     * Two duals are equal if their real and infinitesimal components
     * are equal.
     *
     * @param z first dual number.
     * @param w second dual number.
     * @return true if the duals are equal, false if not.
     */
    friend bool operator==(const Dual<T> &z, const Dual<T> &w) {
        return z.val == w.val; // && z.eps == w.eps;
    }

    /**
     * Comparison operator for a dual number and a real number.
     *
     * A dual number is equal to a real number if its real part
     * is equal to the number we are comparing to and its
     * infinitesimal component is zero.
     *
     * @param z dual number.
     * @param x real number.
     * @return true if equal, false if not.
     */
    template<class U>
    friend bool operator==(const Dual<T> &z, const U &x) {
        return static_cast<Dual<T>>(x) == z;
    }


    /**
     * Comparison operator for a dual number and a real number.
     *
     * A dual number is equal to a real number if its real part
     * is equal to the number we are comparing to and its
     * infinitesimal component is zero.
     *
     * @param z dual number.
     * @param x real number.
     * @return true if equal, false if not.
     */
    template<class U>
    friend bool operator==(const U &x, const Dual<T> &z) {
        return static_cast<Dual<T>>(x) == z;
    }

    /**
    * Comparison operator for two dual numbers.
    *
    * Dual numbers differ if their values differ or if
     * their infinitesimal components differ.
    *
    * @param z first dual number.
    * @param w second dual number.
    * @return true if not equal, false if equal.
    */
    friend bool operator!=(const Dual<T> &z, const Dual<T> &w) {
        return z.val != w.val; //|| z.eps != w.eps;
    }

    /**
    * Comparison operator for a dual number and a real.
    *
    * A dual numbers differs from a real if its values differs
    * from the real or if its infinitesimal component is non-zero.
    *
    * @param z first dual number.
    * @param x real number.
    * @return true if not equal, false if equal.
    */
    template<class U>
    friend bool operator!=(const Dual<T> &z, const U &x) {
        return static_cast<Dual<T>>(x) != z;
    }

    /**
    * Comparison operator for a dual number and a real.
    *
    * A dual numbers differs from a real if its values differs
    * from the real or if its infinitesimal component is non-zero.
    *
    * @param z first dual number.
    * @param x real number.
    * @return true if not equal, false if equal.
    */
    template<class U>
    friend bool operator!=(const U &x, const Dual<T> &z) {
        return static_cast<Dual<T>>(x) != z;
    }

    /**
    * Comparison less-than operator for two dual numbers.
    *
    * Dual numbers are compared based on their real components.
    *
    * @param z first dual number.
    * @param w second dual number.
    * @return true if z.val < w.val.
    */
    friend bool operator<(const Dual<T> &z, const Dual<T> &w) {
        return z.val < w.val;
    }

    /**
    * Comparison less-than operator for a dual number and a real.
    *
    * A dual and a real are compared based on the real components.
    *
    * @param z dual number.
    * @param x real number.
    * @return true if z.val < x.
    */
    template<class U>
    friend bool operator<(const Dual<T> &z, const U &x) {
        return z < static_cast<Dual<T>>(x);
    }

    /**
    * Comparison less-than operator for a dual number and a real.
    *
    * A dual and a real are compared based on the real components.
    *
    * @param z dual number.
    * @param x real number.
    * @return true if z.val > x.
    */
    template<class U>
    friend bool operator<(const U &x, const Dual<T> &z) {
        return static_cast<Dual<T>>(x) < z;
    }

    /**
    * Comparison greater-than operator for two dual numbers.
    *
    * Dual numbers are compared based on the real components.
    *
    * @param z first dual number.
    * @param w second dual number.
    * @return true if z.val > w.val.
    */
    friend bool operator>(const Dual<T> &z, const Dual<T> &w) {
        return z.val > w.val;
    }

    /**
    * Comparison greater-than operator for a dual and a real.
    *
    * A dual and real are compared based on the real components.
    *
    * @param z dual number.
    * @param x real number.
    * @return true if z.val > x.
    */
    template<class U>
    friend bool operator>(const Dual<T> &z, const U &x) {
        return z > static_cast<Dual<T>>(x);
    }

    /**
    * Comparison greater-than operator for a dual and a real.
    *
    * A dual and real are compared based on the real components.
    *
    * @param z dual number.
    * @param x real number.
    * @return true if z.val < x.
    */
    template<class U>
    friend bool operator>(const U &x, const Dual<T> &z) {
        return static_cast<Dual<T>>(x) > z;
    }

    /**
    * Comparison greater-than or equal operator for two duals.
    *
    * Duals are compared based on the real components.
    *
    * @param z first dual number.
    * @param w second dual number.
    * @return true if z.val >= w.val.
    */
    friend bool operator>=(const Dual<T> &z, const Dual<T> &w) {
        return z > w || z == w;
    }

    /**
    * Comparison greater-than or equal operator for a dual and real.
    *
    * A dual and a real are compared based on the real components.
    *
    * @param z dual number.
    * @param x real number.
    * @return true if z.val >= x.
    */
    template<class U>
    friend bool operator>=(const Dual<T> &z, const U &x) {
        return z >= static_cast<Dual<T>>(x);
    }

    /**
    * Comparison greater-than or equal operator for a dual and real.
    *
    * A dual and a real are compared based on the real components.
    *
    * @param x real number.
    * @param z dual number.
    * @return true if z.val < x.
    */
    template<class U>
    friend bool operator>=(const U &x, const Dual<T> &z) {
        return static_cast<Dual<T>>(x) >= z;
    }

    /**
    * Comparison less-than or equal operator for two duals.
    *
    * Duals are compared based on the real components.
    *
    * @param z first dual number.
    * @param w second dual number.
    * @return true if z.val <= w.val.
    */
    friend bool operator<=(const Dual<T> &z, const Dual<T> &w) {
        return z < w || z == w;
    }

    /**
    * Comparison less-than or equal operator for a dual and real.
    *
    * A dual and a real are compared based on the real components.
    *
    * @param z dual number.
    * @param x real number.
    * @return true if z.val <= x.
    */
    template<class U>
    friend bool operator<=(const Dual<T> &z, const U &x) {
        return z <= static_cast<Dual<T>>(x);
    }

    /**
    * Comparison less-than or equal operator for a dual and real.
    *
    * A dual and a real are compared based on the real components.
    *
    * @param x real number.
    * @param z dual number.
    * @return true if z.val > x.
    */
    template<class U>
    friend bool operator<=(const U &x, const Dual<T> &z) {
        return static_cast<Dual<T>>(x) <= z;
    }

    /**
     * Increment overload for duals.
     *
     * Duals are added component-wise
     *
     * @param z first dual
     * @param w second dual
     */
    friend void operator+=(Dual<T> &z, const Dual<T> &w) {
        z.val += w.val;
        z.eps += w.eps;
    }

    /**
     * Increment overload for duals.
     *
     * Duals are subtracted component-wise
     *
     * @param z first dual
     * @param w second dual
     */
    friend void operator-=(Dual<T> &z, const Dual<T> &w) {
        z.val -= w.val;
        z.eps -= w.eps;
    }

    /**
     * Increment overload for dual and real.
     *
     * Addes to real compoenent of dual.
     *
     * @param z first dual
     * @param x real number.
     */
    template<class U>
    friend void operator+=(Dual<T> &z, const U &x) {
        z.val += static_cast<Dual<T>>(x).val;
    }

    /**
     * Decrement overload for dual and real.
     *
     * Subtracts to real compoenent of dual.
     *
     * @param z first dual
     * @param x real number.
     */
    template<class U>
    friend void operator-=(Dual<T> &z, const U &x) {
        z.val -= static_cast<Dual<T>>(x).val;
    }

    /**
     * Addition overload for duals.
     *
     * Adds component-wise
     *
     * @param z first dual
     * @param w second dual.
     */
    friend Dual<T> operator+(const Dual<T> &z, const Dual<T> &w) {
        return Dual<T>(w.val + z.val, w.eps + z.eps);
    }

    /**
     * Addition overload for dual and real.
     *
     * Adds real components
     *
     * @param z first dual
     * @param w second dual.
     */
    template<class U>
    friend Dual<T> operator+(const Dual<T> &z, const U &x) {
        return static_cast<Dual<T>>(x) + z;
    }

    /**
     * Addition overload for dual and real.
     *
     * Adds real components
     *
     * @param w second dual.
     * @param z first dual
     */
    template<class U>
    friend Dual<T> operator+(const U &x, const Dual<T> &z) {
        return static_cast<Dual<T>>(x) + z;
    }

    /**
     * Subtraction overload for duals.
     *
     * Subtracts component-wise
     *
     * @param z first dual
     * @param w second dual.
     */
    friend Dual<T> operator-(const Dual<T> &z, const Dual<T> &w) {
        return Dual<T>(-w.val + z.val, -w.eps + z.eps);
    }

    /**
     * Subtraction overload for dual and real.
     *
     * Subtracts real components
     *
     * @param w second dual.
     * @param x real
     */
    template<class U>
    friend Dual<T> operator-(const Dual<T> &z, const U &x) {
        return z - static_cast<Dual<T>>(x);
    }

    /**
     * Subtraction overload for dual and real.
     *
     * Subtracts real components
     *
     * @param x real
     * @param w second dual.
     */
    template<class U>
    friend Dual<T> operator-(const U &x, const Dual<T> &z) {
        return static_cast<Dual<T>>(x) - z;
    }

    /**
     * Negation operator for duals.
     *
     * Change signs of both components of dual.
     *
     * @param z dual number
     * @return negated dual.
     */
    friend Dual<T> operator-(const Dual<T> &z) {
        return Dual<T>(-z.val, -z.eps);
    }

    /**
     * Multiplication of duals
     *
     * Multiplication follows product rule.
     *
     * @param z first dual
     * @param w second dual
     * @return multiplied dual
     */
    friend Dual<T> operator*(const Dual<T> &z, const Dual<T> &w) {
        return Dual<T>(w.val * z.val, w.eps * z.val + w.val * z.eps);
    }

    /**
     * Overload for *=
     * @param z Dual number
     * @param w Dual number
     */
    friend void operator*=(Dual<T> &z, const Dual<T> &w) {
        z.val = w.val * z.val;
        z.eps = w.eps * z.val + w.val * z.eps;
    }

    /**
     * Multiplication of dual and real
     *
     * Multiplication is done component wise.
     *
     * @param z first dual
     * @param x real
     * @return scaled dual
     */
    template<class U>
    friend Dual<T> operator*(const Dual<T> &z, const U &x) {
        return static_cast<Dual<T>>(x) * z;
    }


    /**
     * Multiplication of dual and real
     *
     * Multiplication is done component wise.
     *
     * @param z first dual
     * @param x real
     * @return scaled dual
     */
    template<class U>
    friend Dual<T> operator*(const U &x, const Dual<T> &z) {
        return static_cast<Dual<T>>(x) * z;
    }

    /**
     * Overload for *= with dual and real.
     * @tparam U type of real
     * @param z Dual number
     * @param x real
     */
    template<class U>
    friend void operator*=(Dual<T> &z, const U &x) {
        z = z * x;
    }


    /**
     * Division of two duals.
     *
     * Division follows the quotient rule.
     *
     * @param z first dual
     * @param w second dual
     * @return quotent of duals.
     */
    friend Dual<T> operator/(const Dual<T> &z, const Dual<T> &w) {
        if (w.val == 0) {
            std::cout << "division by zero with two duals: " << std::endl;
            std::cout << z << std::endl;
            std::cout << w << std::endl;
            throw THDMException(THDMExceptionCode::DualDivisionByZero);
        }
        return Dual<T>(z.val / w.val, -w.eps * z.val / pow(w.val, 2) + z.eps / w.val);
    }


    /**
     * Overload for /= with two duals
     * @param z Dual
     * @param w Dual
     */
    friend void operator/=(Dual<T> &z, const Dual<T> &w) {
        if (w.val == 0) {
            std::cout << "division by zero with dual and dual: " << std::endl;
            std::cout << z << std::endl;
            std::cout << w << std::endl;
            throw THDMException(THDMExceptionCode::DualDivisionByZero);
        }
        z = z / w;
    }

    /**
     * Divide dual by real.
     *
     * Division component-wise
     *
     * @param z first dual
     * @param x real
     * @return scaled dual
     */
    template<class U>
    friend Dual<T> operator/(const Dual<T> &z, const U &x) {
        if (x == 0) {
            std::cout << "division by zero with dual and other: " << std::endl;
            std::cout << z << std::endl;
            std::cout << x << std::endl;
            throw THDMException(THDMExceptionCode::DualDivisionByZero);
        }
        return z / static_cast<Dual<T>>(x);
    }

    template<class U>
    friend void operator/=(Dual<T> &z, const U &x) {
        if (x == 0) {
            std::cout << "division by zero with dual and other: " << std::endl;
            std::cout << z << std::endl;
            std::cout << x << std::endl;
            throw THDMException(THDMExceptionCode::DualDivisionByZero);
        }
        z = z / static_cast<Dual<T>>(x);
    }

    /**
     * Divide real by dual.
     *
     * Division follow quotent rule.
     *
     * @param x real
     * @param z dual
     * @return quotient
     */
    template<class U>
    friend Dual<T> operator/(const U &x, const Dual<T> &z) {
        if (z.val == 0) {
            std::cout << "division by zero with other and dual: " << std::endl;
            std::cout << z << std::endl;
            std::cout << x << std::endl;
            throw THDMException(THDMExceptionCode::DualDivisionByZero);
        }
        return static_cast<Dual<T>>(x) / z;
    }

    /**
     * Raise dual to a dual power.
     *
     * @param z first dual
     * @param w second dual
     * @return first dual to power of second dual
     */
    friend Dual<T> pow(const Dual<T> &z, const Dual<T> &w) {
        if (z.val == 0)
            return Dual<T>{0};
        if (z.val < 0) {
            std::cout << "error in pow(const Dual<T> &z, const Dual<T> &w)" << std::endl;
            throw THDMException(THDMExceptionCode::DualInvalidLogArgument);
        }
        return Dual<T>(pow(z.val, w.val),
                       w.val * pow(z.val, w.val - 1) * z.eps +
                               w.eps * pow(z.val, w.val) * log(z.val));
    }

    /**
     * Raise dual to real power.
     *
     * @param z first dual
     * @param x real
     * @return first dual to power of real
     */
    template<class U>
    friend Dual<T> pow(const Dual<T> &z, const U &x) {
        return pow(z, static_cast<Dual<T>>(x));
    }

    friend Dual<T> pow(const Dual<T> &z, double x) {
        if (z.val == 0)
            return Dual<T>{0};
        return Dual<T>(pow(z.val, x), x * pow(z.val, x - 1) * z.eps);
    }

    /**
     * Raise dual to integrer power.
     *
     * @param z first dual
     * @param x integer
     * @return first dual to power of integer
     */
    friend Dual<T> pow(const Dual<T> &z, int n) {
        /*
         * Logic:
         * - If n == 0, we return 1
         * - If n == 1, we return z
         * - If n > 1, then call function recursively
         *   using z^n = z * z^n-1. We should eventually hit
         *   z^n = z * z * ... * z * z^1.
         * - If n == -1, then just return 1 / z;
         * - If n < -1, call recursively using
         *   z^(-n) = z^(1-n) / z. Should eventual hit
         *   z^(-n) = 1 / z * 1 /z * ... * z^-1.
         */
        if (n == 0) {
            return Dual<T>(1);
        } else if (n == 1) {
            return z;
        } else if (n > 1) {
            return z * pow(z, n - 1);
        } else if (n == -1) {
            return 1 / z;
        } else if (n < -1) {
            return pow(z, n + 1) / z;
        }
    }

    /**
     * Raise real to dual power.
     *
     * @param z first dual
     * @param x real
     * @return real to power of dual
     */
    template<class U>
    friend Dual<T> pow(const U &x, const Dual<T> &z) {
        return pow(static_cast<Dual<T>>(x), z);
    }

    /**
     * Square root of a dual number.
     *
     * Derivative of sqrt(x) is 1 / 2sqrt(x)
     *
     * @param z dual number
     * @return dual with (sqrt(x), 1 / 2sqrt(x))
     */
    friend Dual<T> sqrt(const Dual<T> &z) {
        if (z.val <= static_cast<T>(0))
            throw THDMException(THDMExceptionCode::DualInvalidSqrtArgument);
        return Dual<T>(sqrt(z.val), z.eps / (2 * sqrt(z.val)));
    }

    /**
     * Absolute value of a dual.
     * @param z dual number
     * @return abs(z.val)
     */
    friend Dual<T> fabs(const Dual<T> &z) {
        if (z.val < 0)
            return Dual<T>(-z);
        return Dual<T>(z);
    }

    /**
     * Absolute value of a dual.
     * @param z dual number
     * @return abs(z.val)
     */
    friend Dual<T> abs(const Dual<T> &z) {
        if (z.val < static_cast<T>(0))
            return Dual<T>(-z);
        return Dual<T>(z);
    }

    /**
     * Sine of a dual number.
     *
     * If z = a + b eps, then sin(z) = sin(a) + b cos(a) eps
     *
     * @param z dual number
     * @return dual w with w.val = sin(z.val) and
     * w.eps = z.eps * cos(z.val)
     */
    friend Dual<T> sin(const Dual<T> &z) {
        return Dual<T>(sin(z.val), z.eps * cos(z.val));
    }

    /**
     * Cosine of a dual number.
     *
     * If z = a + b eps, then cos(z) = cos(a) - b sin(a) eps
     *
     * @param z dual number
     * @return dual w with w.val = cos(z.val) and
     * w.eps = -z.eps * sin(z.val)
     */
    friend Dual<T> cos(const Dual<T> &z) {
        return Dual<T>(cos(z.val), -(z.eps * sin(z.val)));
    }

    /**
     * Tangent of a dual number.
     *
     * If z = a + b eps, then tan(z) = tan(a) + b sec(a)^2 eps
     *
     * @param z dual number
     * @return dual w with w.val = tan(z.val) and
     * w.eps = z.eps * sec(z.val)^2
     */
    friend Dual<T> tan(const Dual<T> &z) {
        return Dual<T>(tan(z.val), z.eps + z.eps * pow(tan(z.val), 2));
    }

    /**
     * Exponential of a dual number.
     *
     * If z = a + b eps, then exp(z) = exp(a) + b exp(a) eps
     *
     * @param z dual number
     * @return dual w with w.val = exp(z.val) and
     * w.eps = z.eps * exp(z.val)
     */
    friend Dual<T> exp(const Dual<T> &z) {
        return Dual<T>(exp(z.val), exp(z.val) * z.eps);
    }

    /**
     * Natural log of a dual number.
     *
     * If z = a + b eps, then log(z) = log(a) + (b / a) eps
     *
     * @param z dual number
     * @return dual w with w.val = log(z.val) and
     * w.eps = z.eps / z.val
     */
    friend Dual<T> log(const Dual<T> &z) {
        if (z.val <= 0)
            throw THDMException(THDMExceptionCode::DualInvalidLogArgument);
        return Dual<T>(log(z.val), z.eps / z.val);
    }
};

} // namespace thdm

namespace Eigen {
template<class T>
struct NumTraits<thdm::Dual<T>> : GenericNumTraits<T> {
    typedef typename NumTraits<T>::Real ReallyReal;
    typedef thdm::Dual<T> Real;
    typedef thdm::Dual<T> NonInteger;
    typedef thdm::Dual<T> Nested;
    enum {
        IsInteger = NumTraits<T>::IsInteger,
        IsSigned = NumTraits<T>::IsSigned,
        IsComplex = 0,
        RequireInitialization = NumTraits<T>::RequireInitialization,
        ReadCost = 2 * NumTraits<T>::ReadCost,
        AddCost = 2 * NumTraits<T>::AddCost,
        MulCost = 3 * NumTraits<T>::MulCost + 1 * NumTraits<T>::AddCost
    };


};
}

#endif // Dual_HPP