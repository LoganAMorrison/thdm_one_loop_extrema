#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include "thdm/errors.hpp"
#include "thdm/constants.hpp"
#include <random>
#include <cmath>
#include <exception>
#include <vector>
#include <iostream>
#include <ostream>

namespace thdm {


/**
 * Class for the parameters of the THDM scalar potential.
 *
 * The THDM potential is given by: m112 (H1^* H1) + m222 (H2^* H2) -
 * m122 (H1^* H2 + H2^* H1) + lam1/2 (H1^* H1)^2 + lam1/2 (H2^* H2)^2
 * + lam3 (H1^* H1)(H2^* H2) + lam4 (H1^* H2)(H2^* H1) +
 * lam5 ((H1^* H2)^2 + (H2^* H1)^2).
 *
 * @tparam T
 */
template<class T>
class Parameters {

public:
    /**
     * coefficient of H1^* H1
     */
    T m112;
    /**
     * coefficient of -(H1^* H2 + H2^* H1)
     */
    T m122;
    /**
     * coefficient of H2^* H2
     */
    T m222;
    /**
     * coefficient of (H1^* H1)^2 / 2
     */
    T lam1;
    /**
     * coefficient of (H2^* H2)^2 / 2
     */
    T lam2;
    /**
     * coefficient of (H1^* H1) (H2^* H2)
     */
    T lam3;
    /**
     * coefficient of (H1^* H2) (H2^* H1)
     */
    T lam4;
    /**
     * coefficient of ((H1^* H2)^2 + (H2^* H1)^2) / 2
     */
    T lam5;
    /**
     * Renormalization scale.
     */
    T mu;
    /**
     * Top yukawa
     */
    T yt;
    /*
     * hyper-charge gauge coupling
     */
    T gp;
    /*
     * SU(2) gauge coupling
     */
    T g;


    /**
     * Default constructor for Parameters<T>. All parameters are set to zero.
     */
    Parameters() : m112(static_cast<T>(0)), m122(static_cast<T>(0)),
                   m222(static_cast<T>(0)), lam1(static_cast<T>(0)),
                   lam2(static_cast<T>(0)), lam3(static_cast<T>(0)),
                   lam4(static_cast<T>(0)), lam5(static_cast<T>(0)),
                   mu(static_cast<T>(0)), yt(static_cast<T>(0)),
                   gp(static_cast<T>(U1Y_COUP)), g(static_cast<T>(SU2_COUP)) {}

    Parameters(const Parameters<T> &params)
            : m112(params.m112), m122(params.m122), m222(params.m222),
              lam1(params.lam1), lam2(params.lam2),
              lam3(params.lam3), lam4(params.lam4), lam5(params.lam5),
              mu(params.mu), yt(params.yt), gp(params.gp), g(params.g) {}

    /**
     * Output stream for parameters.
     * @param os stream
     * @param fields THDM parameters
     * @return stream
     */
    friend std::ostream &operator<<(std::ostream &os, const Parameters<T> &params) {
        os << "Parameters(" << params.m112 << ", " << params.m122 << ", "
           << params.m222 << ", " << params.lam1 << ", " << params.lam2 << ", "
           << params.lam3 << ", " << params.lam4 << ", " << params.lam5 << ", "
           << params.mu << ", " << params.yt << ", "
           << params.gp << ", " << params.g << ")" << std::endl;
        return os;
    }

    /**
     * Constructor for Parameters<T> to set renormalization scale and other
     * parameters to random values that leave the potential bounded from below.
     *
     * We choose the parameters such that the following bounded-from-below
     * conditions are satisfied: lam1 >= 0, lam2 >= 0, lam3 >= -sqrt(lam1 lam2),
     * lam3 + lam4 - abs(lam5) >= -sqrt(lam1 lam2). The range or the random
     * numbers are:
     * - mass squared parameters: m112, m122, m222: [-mu^2,  mu^2]
     * - lam1 and lam2: [0, 10]
     * - lam3: [-sqrt(lam1 lam2), 10]
     * - lam4 and lam5: [-1, 1]
     *
     * @param mu renormalization scale.
     */
    explicit Parameters(T mu) {
        static std::random_device rd{};
        static std::mt19937 engine{rd()};
        std::uniform_real_distribution<double> dist{0.0, 1.0};

        this->mu = mu;
        bool bounded = false;

        while (!bounded) {
            m112 = static_cast<T>(2 * pow(mu, 2) * dist(engine) - pow(mu, 2));
            m122 = static_cast<T>(2 * pow(mu, 2) * dist(engine) - pow(mu, 2));
            m222 = static_cast<T>(2 * pow(mu, 2) * dist(engine) - pow(mu, 2));

            lam1 = static_cast<T>(10 * dist(engine));
            lam2 = static_cast<T>(10 * dist(engine));

            T geo_mean = static_cast<T>(sqrt(lam1 * lam1));

            lam3 = static_cast<T>((10 + geo_mean) * dist(engine) - geo_mean);

            lam4 = static_cast<T>(2 * dist(engine) - 1);
            lam5 = static_cast<T>(2 * dist(engine) - 1);

            bounded = is_bounded();
        }

        gp = U1Y_COUP;
        g = SU2_COUP;
    }

    /**
     * Default destructor for Parameters<T>.
     */
    ~Parameters() = default;

    /**
     * Determine is the scalar potential is bounded from below.
     *
     * The bounded from below conditions are: lam1 >= 0,
     * lam2 >= 0, lam3 >= -sqrt(lam1 lam2),
     * lam3 + lam4 - abs(lam5) >= -sqrt(lam1 lam2)
     *
     * @return true if bounded, false if not.
     */
    bool is_bounded() {
        bool bounded = false;

        if (lam1 > static_cast<T>(0) && lam2 > static_cast<T>(0)) {
            T neg_geo_mean = -sqrt(lam1 * lam2);

            if (lam3 >= neg_geo_mean) {
                if (lam3 + lam4 - fabs(lam5) >= neg_geo_mean)
                    bounded = true;
            }
        }
        return bounded;
    }

    /**
     * Access operator overload for Parameters<T>. Throws
     * error if out of range.
     *
     * par[i] = {m112, m122, m222, lam1, lam2, lam3,
     * lam4, lam5, mu, yt}[i].
     *
     * @param par index of parameter to access.
     * @return parameter at `par` index.
     */
    T &operator[](size_t par) {
        if (par == 0)
            return m112;
        else if (par == 1)
            return m122;
        else if (par == 2)
            return m222;
        else if (par == 3)
            return lam1;
        else if (par == 4)
            return lam2;
        else if (par == 5)
            return lam3;
        else if (par == 6)
            return lam4;
        else if (par == 7)
            return lam5;
        else if (par == 8)
            return mu;
        else if (par == 9)
            return yt;
        else if (par == 10)
            return gp;
        else if (par == 11)
            return g;
        else
            throw THDMException(THDMExceptionCode::ParametersIndexOutOfRange);
    }
};

} // namespace thdm

#endif //PARAMETERS_HPP