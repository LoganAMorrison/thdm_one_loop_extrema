//
// Created by Logan Morrison on 2019-05-05.
//

/*
 * The goal here will be to start from one of configurations
 * where there is a global charge-breaking minimum and a local
 * normal minimum and randomly adjust the parameters in an
 * attempt to find new configurations which also have a
 * global CB minimum and local minimum.
 *
 * To achieve this, we will use a monte carlo Metropolis
 * algorithm. We will randomly adust all the parameters,
 * then re-minimize starting at the normal and charge
 * breaking vacuua. If we obtain the situation we want,
 * then we repeat starting at the new point. If we fail,
 * then we randomly choose a number between [0, 1] and
 * if the number is less some tolerance, then we move to
 * the new point and repeat. If the random number is
 * greater than the tolerance, then we will remain at
 * the old points and try again.
 */

#include "thdm/parameters.hpp"
#include "thdm/fields.hpp"
#include "thdm/vacuua.hpp"
#include "thdm/minimize.hpp"
#include "thdm/potentials.hpp"
#include "thdm/extrema_type.hpp"
#include "thdm/model.hpp"
#include <random>
#include <tuple>
#include <iostream>
#include <iomanip>

using namespace thdm;

static std::random_device rd{};
static std::mt19937 engine{rd()};
std::uniform_real_distribution<double> dist{-1, 1};

double perturb_strength = 0.005; // perturb by at most 5%
double tolerance = -2.0;

/*
 * Generate new parameters by perturbing the old parameters
 * by at most `perturb_strength`.
 *
 * @params params
 */
Parameters<double> get_new_parameters(const Parameters<double> &params) {
    Parameters<double> new_params{};

    new_params.m112 = params.m112 * (1.0 + perturb_strength * dist(engine));
    new_params.m122 = params.m122 * (1.0 + perturb_strength * dist(engine));
    new_params.m222 = params.m222 * (1.0 + perturb_strength * dist(engine));

    new_params.lam1 = params.lam1 * (1.0 + perturb_strength * dist(engine));
    new_params.lam2 = params.lam2 * (1.0 + perturb_strength * dist(engine));
    new_params.lam3 = params.lam3 * (1.0 + perturb_strength * dist(engine));
    new_params.lam4 = params.lam4 * (1.0 + perturb_strength * dist(engine));
    new_params.lam5 = params.lam5 * (1.0 + perturb_strength * dist(engine));
    new_params.mu = params.mu;

    return new_params;
}

/**
 * Check that all derivative of the effective potential
 * are zero at vacuum and the goldstones are present.
 * @param params THDM params
 * @param vac THDM vacuum
 * @param is_cb true if vacuum is charge breaking
 * @return true if vacuum passes checks.
 */
bool check_vacuum(Parameters<double> &params, Vacuum<double> &vac, bool is_cb = false) {
    Fields<double> fields{};

    fields.set_fields(vac);
    auto nmasses = potential_eff_hessian_evals(fields, params);

    bool isgood = verify_derivatives_zero(params, vac);
    isgood = isgood && verify_goldstones(params, vac, is_cb);
    isgood = isgood && are_sqrd_masses_positive_semi_definite(vac, params);

    return isgood;
}

/**
 * Generate a new model from the old one. Model must have
 * a global charge breaking minimum and a local normal
 * minimum.
 * @param model old model
 * @return new model. If no new model was found, old one
 * is returned.
 */
Model get_new_model(const Model &model) {
    // Extract the good vacuua from old model.
    Vacuum<double> nvac = model.one_loop_vacuua[0];
    Vacuum<double> cbvac = model.one_loop_vacuua[1];
    // Perturb the old parameters.
    Parameters<double> _params = get_new_parameters(model.params);

    // Perform a minimization on given the new parameters.
    int nsuccess = minimize_potential_eff(_params, nvac);
    int cbsuccess = minimize_potential_eff(_params, cbvac);

    // If we succeeded during minimization, proceed to check
    // other properties.
    if (nsuccess == 0 && cbsuccess == 0) {

        try {
            // First, verify that the new vacuua are actually
            // vacuua and that they have goldstones.
            bool ngood = check_vacuum(_params, nvac);
            bool cbgood = check_vacuum(_params, cbvac);

            nvac.extrema_type = determine_single_extrema_type_eff(_params, nvac);
            cbvac.extrema_type = determine_single_extrema_type_eff(_params, cbvac);

            if (nvac.extrema_type == SingleExtremaType::Minimum &&
                    cbvac.extrema_type == SingleExtremaType::Minimum &&
                    ngood && cbgood) {
                // So far so good. But let us generate the model then check
                // that CB is really the global minimum
                auto new_model = Model(_params, nvac, cbvac);
                if (new_model.is_deepest_cb &&
                        new_model.has_normal_min &&
                        new_model.has_cb_min) {
                    // All good!
                    return new_model;
                } else {
                    // Not good!
                    return model;
                }
            } else {
                return model;
            }
        } catch (...) {
            // Well, something when wrong...
            return model;
        }
    }


}

int main() {

    Parameters<double> params{};
    Vacuum<double> nvac{};
    Vacuum<double> cbvac{};

    // This is one of the good ones.
    params.m112 = -2998.11648903296;
    params.m122 = 116.371364202366;
    params.m222 = -24992.9774290268;
    params.lam1 = 0.059372288964707;
    params.lam2 = 4.27011212987043;
    params.lam3 = 0.489518733649848;
    params.lam4 = 0.394870514329584;
    params.lam5 = -0.337647937509538;
    params.mu = 246.0;
    nvac.vevs[0] = 331.111654906292;
    nvac.vevs[1] = 13.2406185441237;
    nvac.vevs[2] = 0;
    cbvac.vevs[0] = -47.193186477208;
    cbvac.vevs[1] = -89.6238218708166;
    cbvac.vevs[2] = 208.276489491324;

    Model model(params, nvac, cbvac);

    std::cout << std::setprecision(15) << std::endl;

    int counter = 0;

    while (counter < 100) {
        try {
            auto new_model = get_new_model(model);
            std::cout << "new nvac = " << new_model.one_loop_deepest_normal << std::endl;
            std::cout << "new cbvac = " << new_model.one_loop_deepest_cb << std::endl;
            std::cout << std::endl;
            model = new_model;
        } catch (...) {

        }

        counter++;
    }


    return 0;
}

