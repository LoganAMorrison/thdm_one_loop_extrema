#ifndef VACUUA_HPP
#define VACUUA_HPP

#include <cmath>
#include <random>

namespace thdm {

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


std::string double_extema_type_to_string(DoubleExtremaType type) {
    if (type == DoubleExtremaType::MinMin) {
        return "MinMin";
    } else if (type == DoubleExtremaType::MinMax) {
        return "MinMax";
    } else if (type == DoubleExtremaType::MinSad) {
        return "MinSad";
    } else if (type == DoubleExtremaType::MinUnd) {
        return "MinUnd";
    } else if (type == DoubleExtremaType::MaxMin) {
        return "MaxMin";
    } else if (type == DoubleExtremaType::MaxMax) {
        return "MaxMax";
    } else if (type == DoubleExtremaType::MaxSad) {
        return "MaxSad";
    } else if (type == DoubleExtremaType::MaxUnd) {
        return "MaxUnd";
    } else if (type == DoubleExtremaType::SadMin) {
        return "SadMin";
    } else if (type == DoubleExtremaType::SadMax) {
        return "SadMax";
    } else if (type == DoubleExtremaType::SadSad) {
        return "SadSad";
    } else if (type == DoubleExtremaType::SadUnd) {
        return "SadUnd";
    } else if (type == DoubleExtremaType::UndMin) {
        return "UndMin";
    } else if (type == DoubleExtremaType::UndMax) {
        return "UndMax";
    } else if (type == DoubleExtremaType::UndSad) {
        return "UndSad";
    } else {
        return "UndUnd";
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


}

#endif //VACUUA_HPP