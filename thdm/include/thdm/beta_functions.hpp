//
// Created by Logan Morrison on 2019-05-10.
//

#ifndef THDM_BETA_FUNCTIONS_HPP
#define THDM_BETA_FUNCTIONS_HPP

#include "thdm/parameters.hpp"
#include <cmath>
#include <boost/numeric/odeint.hpp>
#include <iostream>

namespace thdm {

class RGESystem {
    Parameters<double> params;

public:
    explicit RGESystem(const Parameters<double> &params) : params(params) {}

    ~RGESystem() = default;

    double gs(double mu) {
        return 8.88577 / sqrt(21.612 + 7.0 * log(mu));
    }

    double gamma_1() {
        return (9.0 / 4.0 * pow(params.g, 2) + 3.0 / 4.0 * pow(params.gp, 2));
    }

    double gamma_2() {
        return (9.0 / 4.0 * pow(params.g, 2) + 3.0 / 4.0 * pow(params.gp, 2) -
                3.0 * pow(params.yt, 2));
    }

    double beta_g() {
        return -3.0 * pow(params.g, 3) / (16.0 * pow(M_PI, 2));
    }

    double beta_gp() {
        return 7.0 * pow(params.gp, 3) / (16.0 * pow(M_PI, 2));
    }

    double beta_lam1() {
        double lam1 = params.lam1;
        double lam3 = params.lam3;
        double lam4 = params.lam4;
        double lam5 = params.lam5;
        double gp = params.gp;
        double g = params.g;

        return (12.0 * pow(lam1, 2) +
                4.0 * pow(lam3, 2) +
                4.0 * lam3 * lam4 +
                2.0 * pow(lam4, 2) +
                2.0 * pow(lam5, 2) +
                9.0 / 4.0 * pow(g, 4) +
                3.0 / 2.0 * pow(g, 2) * pow(gp, 2) +
                3.0 / 4.0 * pow(gp, 4) -
                4.0 * gamma_1() * lam1) / (16.0 * pow(M_PI, 2));
    }

    double beta_lam2() {
        double lam2 = params.lam2;
        double lam3 = params.lam3;
        double lam4 = params.lam4;
        double lam5 = params.lam5;
        double yt = params.yt;
        double gp = params.gp;
        double g = params.g;

        return (12.0 * pow(lam2, 2) +
                4.0 * pow(lam3, 2) +
                4.0 * lam3 * lam4 +
                2.0 * pow(lam4, 2) +
                2.0 * pow(lam5, 2) +
                9.0 / 4.0 * pow(g, 4) +
                3.0 / 2.0 * pow(g, 2) * pow(gp, 2) +
                3.0 / 4.0 * pow(gp, 4) -
                4.0 * gamma_2() * lam2 -
                12.0 * pow(yt, 4)) / (16.0 * pow(M_PI, 2));
    }

    double beta_lam3() {
        double lam1 = params.lam1;
        double lam2 = params.lam2;
        double lam3 = params.lam3;
        double lam4 = params.lam4;
        double lam5 = params.lam5;
        double gp = params.gp;
        double g = params.g;

        return ((lam1 + lam2) * (6.0 * lam3 + 2.0 * lam4) +
                4.0 * pow(lam3, 2) +
                2.0 * pow(lam4, 2) +
                2.0 * pow(lam5, 2) +
                9.0 / 4.0 * pow(g, 4) -
                3.0 / 2.0 * pow(g, 2) * pow(gp, 2) +
                3.0 / 4.0 * pow(gp, 4) -
                2.0 * (gamma_1() + gamma_2()) * lam3) /
                (16.0 * pow(M_PI, 2));
    }

    double beta_lam4() {
        double lam1 = params.lam1;
        double lam2 = params.lam2;
        double lam3 = params.lam3;
        double lam4 = params.lam4;
        double lam5 = params.lam5;
        double gp = params.gp;
        double g = params.g;

        return (2.0 * (lam1 + lam2) * lam4 +
                8.0 * lam3 * lam4 +
                4.0 * pow(lam4, 2) +
                8.0 * pow(lam5, 2) -
                2.0 * (gamma_1() + gamma_2()) * lam4 +
                3.0 * pow(g, 2) * pow(gp, 2)) /
                (16.0 * pow(M_PI, 2));
    }

    double beta_lam5() {
        double lam1 = params.lam1;
        double lam2 = params.lam2;
        double lam3 = params.lam3;
        double lam4 = params.lam4;
        double lam5 = params.lam5;

        return lam5 * (lam1 + lam2 + 4.0 * lam3 + 6.0 * lam4 -
                gamma_1() - gamma_2()) /
                (8.0 * pow(M_PI, 2));
    }

    double beta_m112() {
        double lam1 = params.lam1;
        double lam3 = params.lam3;
        double lam4 = params.lam4;
        double m112 = params.m112;
        double m222 = params.m222;

        return (6.0 * lam1 * m112 +
                (4.0 * lam3 + 2.0 * lam4) * m222 -
                2.0 * gamma_1() * m112) / (16.0 * pow(M_PI, 2));
    }

    double beta_m222() {
        double lam2 = params.lam2;
        double lam3 = params.lam3;
        double lam4 = params.lam4;
        double m112 = params.m112;
        double m222 = params.m222;

        return (6.0 * lam2 * m222 +
                (4.0 * lam3 + 2.0 * lam4) * m112 -
                2.0 * gamma_2() * m222) / (16.0 * pow(M_PI, 2));
    }

    double beta_m122() {
        double lam3 = params.lam3;
        double lam4 = params.lam4;
        double lam5 = params.lam5;
        double m122 = params.m122;

        return m122 * (2.0 * lam3 + 4.0 * lam4 + 6.0 * lam5 -
                gamma_1() - gamma_2()) / (16.0 * pow(M_PI, 2));
    }

    double beta_yt() {
        double yt = params.yt;
        double g = params.g;
        double gp = params.gp;
        double au = -8.0 * pow(gs(params.mu), 2) - 9.0 / 4.0 * pow(g, 2) -
                17.0 / 12.0 * pow(gp, 2);

        return au * yt + 9.0 / 2.0 * pow(yt, 3);
    }

    void update_params(const std::vector<double> &x, const double mu) {
        params.m112 = x[0];
        params.m122 = x[1];
        params.m222 = x[2];
        params.lam1 = x[3];
        params.lam2 = x[4];
        params.lam3 = x[5];
        params.lam4 = x[6];
        params.lam5 = x[7];
        params.yt = x[8];
        params.gp = x[9];
        params.g = x[10];
        params.mu = mu;
    }

    Parameters<double> get_params() {
        return params;
    }

    void operator()(const std::vector<double> &x,
                    std::vector<double> &dxdt, const double mu) {

        update_params(x, mu);

        dxdt[0] = beta_m112() / mu;
        dxdt[1] = beta_m122() / mu;
        dxdt[2] = beta_m222() / mu;
        dxdt[3] = beta_lam1() / mu;
        dxdt[4] = beta_lam2() / mu;
        dxdt[5] = beta_lam3() / mu;
        dxdt[6] = beta_lam4() / mu;
        dxdt[7] = beta_lam5() / mu;
        dxdt[8] = beta_yt() / mu;
        dxdt[9] = beta_gp() / mu;
        dxdt[10] = beta_g() / mu;
    }
};


Parameters<double> run_parameters(const Parameters<double> &params,
                                  double mu1, double mu2) {

    using namespace boost::numeric::odeint;

    RGESystem rge_system(params);
    std::vector<double> x(11);
    x[0] = params.m112;
    x[1] = params.m122;
    x[2] = params.m222;
    x[3] = params.lam1;
    x[4] = params.lam2;
    x[5] = params.lam3;
    x[6] = params.lam4;
    x[7] = params.lam5;
    x[8] = params.yt;
    x[9] = params.gp;
    x[10] = params.g;

    double initial_step = 0.5;
    if (mu1 > mu2)
        initial_step = -initial_step;

    integrate(rge_system, x, mu1, mu2, initial_step);

    rge_system.update_params(x, mu2);

    return rge_system.get_params();
}


}

#endif //THDM_BETA_FUNCTIONS_HPP
