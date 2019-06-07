//
// Created by Logan Morrison on 2019-05-30.
//


#include "thdm/potentials.hpp"
#include "thdm/fields.hpp"
#include "thdm/parameters.hpp"
#include "thdm/scalar_masses.hpp"
#include "thdm/gauge_masses.hpp"
#include "thdm/fermion_masses.hpp"
#include "thdm/model.hpp"
#include <tuple>
#include <vector>
#include <cmath>
#include <string>

using namespace thdm;

/**
 * From the file stream, read in the next 'point' and return a Point object.
 * @param file file stream
 * @return Point containing parameters, normal and charge breaking vacuua.
 */
Point read_in_next_point(std::ifstream &file) {
    std::string value;
    Parameters<double> params{};
    // Read in all parameters
    getline(file, value, ',');
    params.m112 = std::stod(value);
    getline(file, value, ',');
    params.m122 = std::stod(value);
    getline(file, value, ',');
    params.m222 = std::stod(value);
    getline(file, value, ',');
    params.lam1 = std::stod(value);
    getline(file, value, ',');
    params.lam2 = std::stod(value);
    getline(file, value, ',');
    params.lam3 = std::stod(value);
    getline(file, value, ',');
    params.lam4 = std::stod(value);
    getline(file, value, ',');
    params.lam5 = std::stod(value);
    getline(file, value, ',');
    params.mu = std::stod(value);
    // Read in normal vacuum
    Vacuum<double> nvac{};
    getline(file, value, ',');
    nvac.vevs[0] = std::stod(value);
    getline(file, value, ',');
    nvac.vevs[1] = std::stod(value);
    getline(file, value, ',');
    nvac.vevs[2] = std::stod(value);
    // Read in charge-breaking vacuum
    Vacuum<double> cbvac{};
    getline(file, value, ',');
    cbvac.vevs[0] = std::stod(value);
    getline(file, value, ',');
    cbvac.vevs[1] = std::stod(value);
    getline(file, value);
    cbvac.vevs[2] = std::stod(value);

    return Point{params, nvac, cbvac};
}

int main() {


    Parameters<double> params{};
    Vacuum<double> nvac{};
    Vacuum<double> cbvac{};
    Fields<double> fields{};
    params.m112 = -56769.1625189425;
    params.m122 = 11298.9441043711;
    params.m222 = -21800.5075577879;

    params.lam1 = 5.08772715231133;
    params.lam2 = 0.657186787219266;
    params.lam3 = 1.94804091410591;
    params.lam4 = 2.20182701539137;
    params.lam5 = -0.570051230985773;
    params.yt = 1.02110482180414;
    params.mu = 246;
    nvac.vevs[0] = -55.7386132154807;
    nvac.vevs[1] = -239.602184874878;
    nvac.vevs[2] = 0;
    cbvac.vevs[0] = 116.453931975248;
    cbvac.vevs[1] = 118.024105034008;
    cbvac.vevs[2] = -70.3895466120712;

    fields.set_fields(nvac);
    auto deriv_scalar_masses_r1 = scalar_squared_masses_deriv_fld(fields, params, 1);
    auto deriv_scalar_masses_r2 = scalar_squared_masses_deriv_fld(fields, params, 2);
    auto deriv_gauge_masses_r1 = gauge_squared_masses_deriv(fields, params, 1);
    auto deriv_gauge_masses_r2 = gauge_squared_masses_deriv(fields, params, 2);
    auto top_masses = top_mass_squared(fields, params);
    auto deriv_top_masses_r1 = top_mass_squared_deriv(fields, params, 1);
    auto deriv_top_masses_r2 = top_mass_squared_deriv(fields, params, 2);
    auto deriv_tree_pot_r1 = potential_tree_deriv(fields, params, 1);
    auto deriv_tree_pot_r2 = potential_tree_deriv(fields, params, 2);

    std::cout << "Masses and derivs r1: \n";
    for (auto tup: deriv_scalar_masses_r1) {
        std::cout << "mass, deriv = " << sqrt(std::get<0>(tup)) << ", " << std::get<1>(tup) << "\n";
    }
    for (auto tup: deriv_gauge_masses_r1) {
        std::cout << "mass, deriv = " << sqrt(std::get<0>(tup)) << ", " << std::get<1>(tup) << "\n";
    }
    std::cout << "mass, deriv = " << sqrt(top_masses) << ", " << deriv_top_masses_r1 << "\n";
    std::cout << "Masses and derivs r2: \n";
    for (auto tup: deriv_scalar_masses_r2) {
        std::cout << "mass, deriv = " << sqrt(std::get<0>(tup)) << ", " << std::get<1>(tup) << "\n";
    }
    for (auto tup: deriv_gauge_masses_r2) {
        std::cout << "mass, deriv = " << sqrt(std::get<0>(tup)) << ", " << std::get<1>(tup) << "\n";
    }
    std::cout << "mass, deriv = " << sqrt(top_masses) << ", " << deriv_top_masses_r2 << "\n";


    std::cout << "tree deriv r1: " << deriv_tree_pot_r1 << std::endl;
    std::cout << "tree deriv r2: " << deriv_tree_pot_r2 << std::endl;

    fields.set_fields(nvac);
    std::cout << "potential eff derivs: " << std::endl;
    for (int i = 1; i <= 8; i++) {
        std::cout << potential_eff_deriv(fields, params, i) << std::endl;
    }

    fields.set_fields(cbvac);
    std::cout << "potential eff derivs: " << std::endl;
    for (int i = 1; i <= 8; i++) {
        std::cout << potential_eff_deriv(fields, params, i) << std::endl;
    }


    return 0;
}