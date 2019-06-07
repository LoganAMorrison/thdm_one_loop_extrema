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
    params.m112 = -68565.417571077;
    params.m122 = 212.879551931601;
    params.m222 =-25472.8371733382;

    params.lam1 = 6.82033686923365;
    params.lam2 = 0.84217650859847;
    params.lam3 = 2.52068856888243;
    params.lam4 = 0.795114747809148;
    params.lam5 = -0.640248307167735;
    params.yt = 0.995310366281959;
    params.mu = 246;
    nvac.vevs[0] = 9.62297558376271;
    nvac.vevs[1] = 245.811713189006;
    nvac.vevs[2] = 0;
    cbvac.vevs[0] = -83.4733492946056;
    cbvac.vevs[1] = -33.086784045566;
    cbvac.vevs[2] = 116.283964204495;

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