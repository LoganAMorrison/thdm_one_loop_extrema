//
// Created by Logan Morrison on 2019-05-11.
//

#include "thdm/beta_functions.hpp"
#include "thdm/fermion_masses.hpp"
#include "thdm/fields.hpp"
#include "thdm/gauge_masses.hpp"
#include "thdm/model.hpp"
#include "thdm/parameters.hpp"
#include "thdm/potentials.hpp"
#include "thdm/root_refine.hpp"
#include "thdm/vacuua.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace thdm;

int main() {

    int num_mus = 1000;
    double mu1 = 246.0;
    double mu2 = 50.0;
    double mu_step = (mu2 - mu1) / (num_mus - 1);
    double v = 246.0;

    // Set up the model.
    Fields<double> fields{};
    Parameters<double> params{};
    params.m112 = -24387.2798158948;
    params.m122 = -306.105737762432;
    params.m222 = -35015.1441785317;
    params.lam1 = 0.595501797126108;
    params.lam2 = 1.18188716752982;
    params.lam3 = 0.848217876952806;
    params.lam4 = 0.559145266584235;
    params.lam5 = -0.511092299548673;
    params.mu = v;
    params.g = SU2_COUP;
    params.gp = U1Y_COUP;
    Vacuum<double> nvac(
            std::vector<double>{-54.393055758293, 239.911224175267, 0});
    Vacuum<double> cbvac(std::vector<double>{-187.586188286565, 72.1879089873015,
            -206.800486744671});
    set_top_yukawa(params, nvac);

    /* Do some checks */
    fields.set_fields(nvac);
    for (int i = 0; i < 8; i++)
        std::cout << potential_eff_deriv(fields, params, i + 1) << ", ";
    std::cout << std::endl;

    auto evals = potential_eff_hessian_evals(fields, params);
    for (int i = 0; i < 8; i++)
        std::cout << evals[i] << ", ";
    std::cout << std::endl;

    fields.set_fields(cbvac);
    for (int i = 0; i < 8; i++)
        std::cout << potential_eff_deriv(fields, params, i + 1) << ", ";
    std::cout << std::endl;

    evals = potential_eff_hessian_evals(fields, params);
    for (int i = 0; i < 8; i++)
        std::cout << evals[i] << ", ";
    std::cout << std::endl;
    std::cout << std::endl;

    std::vector<double> potential_eff_normal(num_mus);
    std::vector<double> potential_eff_cb(num_mus);
    std::vector<double> mus(num_mus);
    std::vector<Parameters<double>> params_vec(num_mus);

    // Fill in the values for mus
    for (int i = 0; i < num_mus; i++) {
        mus[i] = i * mu_step + mu1;
    }

    // Set initial values
    fields.set_fields(nvac);
    potential_eff_normal[0] = potential_eff(fields, params);
    fields.set_fields(cbvac);
    potential_eff_cb[0] = potential_eff(fields, params);
    params_vec[0] = params;

    for (int i = 1; i < num_mus; i++) {
        params = run_parameters(params, mus[i - 1], mus[i]);
        params_vec[i] = params;
        refine_root(params, nvac);
        refine_root(params, cbvac);

        fields.set_fields(nvac);
        potential_eff_normal[i] = potential_eff(fields, params);
        auto treemassesn = scalar_squared_masses(fields, params);
        auto effmassesn = potential_eff_hessian_evals(fields, params);
        fields.set_fields(cbvac);
        auto treemassescb = scalar_squared_masses(fields, params);
        auto effmassescb = potential_eff_hessian_evals(fields, params);
        potential_eff_cb[i] = potential_eff(fields, params);

        int counter = 0;
        std::cout << "Normal" << nvac << std::endl;
        for (auto m : treemassesn) {
            if (std::abs(m) < 1e-3) {
                counter++;
            }
        }
        counter = 0;
        std::cout << "Number normal tree zero masses =" << counter << std::endl;
        for (auto m : effmassesn) {
            if (std::abs(m) < 1e-3) {
                counter++;
            }
        }
        std::cout << "Number normal eff zero masses =" << counter << std::endl;
        std::cout << std::endl;
        std::cout << "CB" << cbvac << std::endl;
        counter = 0;
        for (auto m : treemassescb) {
            if (std::abs(m) < 1e-3) {
                counter++;
            }
        }
        std::cout << "CB normal tree zero masses =" << counter << std::endl;
        counter = 0;
        for (auto m : effmassescb) {
            if (std::abs(m) < 1e-3) {
                counter++;
            }
        }
        std::cout << "CB normal eff zero masses =" << counter << std::endl;
        std::cout << std::endl;
    }

    /*std::cout << std::setprecision(15) << std::endl;
    for (int i = 0; i < num_mus; i++) {
        std::cout << mus[i] << ", " << potential_eff_normal[i] << ","
                  << potential_eff_cb[i] << std::endl;
    }
    std::cout << std::endl;*/



    /*std::cout << std::setprecision(15) << std::endl;
    for (int i = 0; i < num_mus; i++) {
        auto _params = params_vec[i];
        std::cout << _params.mu << ","
                  << _params.m112 << ", "
                  << _params.m122 << ","
                  << _params.m222 << ","
                  << _params.lam1 << ","
                  << _params.lam2 << ","
                  << _params.lam3 << ","
                  << _params.lam4 << ","
                  << _params.lam5 << ","
                  << _params.yt << ","
                  << _params.gp << ","
                  << _params.g << std::endl;
    }*/

    return 0;
}