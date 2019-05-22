#ifndef VACUUA_HPP
#define VACUUA_HPP

#include <cmath>
#include <random>
#include <iostream>
#include <ostream>

namespace thdm {

/* A vacuum is considered normal c1 is greater than this tolerance. */
const double ZERO_CB_VEV_TOL = 1e-2;
/* Two vacuua are considered the same if sqrt of sum of
 * squares of components are less than this tolerance
 */
const double CLOSE_VEVS_TOL = 1e-5;

enum SingleExtremaType {
    Minimum,
    Maximum,
    Saddle,
    Undefined,
};

enum DoubleExtremaType {
    MinMin,
    MinMax,
    MinSad,
    MinUnd,
    MaxMin,
    MaxMax,
    MaxSad,
    MaxUnd,
    SadMin,
    SadMax,
    SadSad,
    SadUnd,
    UndMin,
    UndMax,
    UndSad,
    UndUnd
};

std::string single_extrema_type_to_string(SingleExtremaType type) {
    if (type == SingleExtremaType::Minimum) {
        return "Minimum";
    } else if (type == SingleExtremaType::Maximum) {
        return "Maximum";
    } else if (type == SingleExtremaType::Saddle) {
        return "Saddle";
    } else {
        return "Undefined";
    }
}


template<class T>
class Vacuum {
private:
public:
    T potential;
    std::vector<T> vevs;
    SingleExtremaType extrema_type;

    Vacuum() : potential(0.0), vevs(std::vector<T>{0.0, 0.0, 0.0}),
               extrema_type(SingleExtremaType::Undefined) {}

    explicit Vacuum(std::vector<T> vevs)
            : potential(0.0), vevs(std::move(vevs)), extrema_type(SingleExtremaType::Undefined) {}

    Vacuum(double potential, std::vector<T> vevs)
            : potential(potential), vevs(vevs),
              extrema_type(SingleExtremaType::Undefined) {}

    ~Vacuum() = default;

    Vacuum(const Vacuum<T> &vac)
            : potential(vac.potential),
              vevs(vac.vevs), extrema_type(vac.extrema_type) {}

    Vacuum<T> &operator=(const Vacuum<T> &vac) {
        potential = vac.potential;
        vevs = vac.vevs;
        extrema_type = vac.extrema_type;
        return *this;
    }

    friend bool operator<=(const Vacuum<T> &vac1, const Vacuum<T> &vac2) {
        return vac1.potential <= vac2.potential;
    }

    friend bool operator<(const Vacuum<T> &vac1, const Vacuum<T> &vac2) {
        return vac1.potential < vac2.potential;
    }

    friend bool operator>=(const Vacuum<T> &vac1, const Vacuum<T> &vac2) {
        return vac1.potential >= vac2.potential;
    }

    friend bool operator>(const Vacuum<T> &vac1, const Vacuum<T> &vac2) {
        return vac1.potential > vac2.potential;
    }


    /**
     * Output stream for vacuua.
     * @param os stream
     * @param fields THDM vacuum
     * @return stream
     */
    friend std::ostream &operator<<(std::ostream &os, const Vacuum<T> &vac) {
        os << "Vacuum(" << vac.potential << ", " << vac.vevs[0] << ", "
           << vac.vevs[1] << ", " << vac.vevs[2] << ", "
           << single_extrema_type_to_string(vac.extrema_type) << ")" << std::endl;
        return os;
    }
};


/**
 * Generate a random, normal vacuum.
 * @param mu renormalization scale.
 * @return normal vacuum.
 */

Vacuum<double> generate_normal_vac(double mu) {
    static std::random_device rd{};
    static std::mt19937 engine{rd()};
    std::uniform_real_distribution<double> dist{-1, 1};

    std::vector<double> _vevs(3, 0.0);
    double beta = M_PI * dist(engine) - M_PI_2;
    _vevs[0] = (cos(beta) * mu);
    _vevs[1] = (sin(beta) * mu);
    return Vacuum<double>(_vevs);
}

/**
 * Generate a random, charge-breaking vacuum.
 * @param mu renormalization scale.
 * @return normal vacuum.
 */
Vacuum<double> generate_cb_vac(double mu) {
    static std::random_device rd{};
    static std::mt19937 engine{rd()};
    std::uniform_real_distribution<double> dist{0, 1};

    std::vector<double> _vevs(3, 0.0);
    _vevs[0] = (2.0 * mu * dist(engine) - mu);
    _vevs[1] = (2.0 * mu * dist(engine) - mu);
    _vevs[2] = (2.0 * mu * dist(engine) - mu);
    return Vacuum<double>(_vevs);
}

/**
 * Determine if vacuum is normal.
 * @param vac THDM vacuum.
 * @return bool True is vacuum is normal.
 */
bool is_vacuum_normal(const Vacuum<double> &vac) {
    using std::abs;
    return (abs(vac.vevs[2]) < ZERO_CB_VEV_TOL);
}

/**
 * Determine if the vacuua are approximately the same.
 * @param vac1 THDM vacuum.
 * @param vac2 THDM vacuum.
 * @return True if vacuua are approximately equal.
 */
bool are_vacuua_approx_equal(const Vacuum<double> &vac1,
                             const Vacuum<double> &vac2) {
    using std::abs;
    double quad = 0.0;
    quad += pow(vac1.vevs[0] - vac2.vevs[0], 2);
    quad += pow(vac1.vevs[1] - vac2.vevs[1], 2);
    quad += pow(vac1.vevs[2] - vac2.vevs[2], 2);
    return sqrt(quad) < CLOSE_VEVS_TOL;
}


}

#endif //VACUUA_HPP