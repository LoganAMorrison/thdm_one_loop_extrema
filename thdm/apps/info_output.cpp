//
// Created by Logan Morrison on 2019-05-19.
//

/*
 * This file is for computing the effective potential along a line connenting
 * a normal and charge-breaking extrema. We read in data from 'type_a1.csv',
 * the file where there exists parameters and vacuua such that there is a
 * global charge-breaking minimum and local normal minimum, We then evaluate
 * the effective potential along a line connecting these two vacuua, i.e.:
 *      V_eff(t) = n_vac * (1-t) * cb_vac * t
 * where 0 < t < 1 (actually, we use a slightly larger window to captue the
 * behavior beyond the minima.)
 */

#include "thdm/potentials.hpp"
#include "thdm/fields.hpp"
#include "thdm/parameters.hpp"
#include "thdm/vacuua.hpp"
#include "thdm/fermion_masses.hpp"
#include "thdm/gauge_masses.hpp"
#include "thdm/validation.hpp"
#include "thdm/model.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <tuple>
#include <iomanip>

using namespace thdm;

/**
 * Struct to hold the t's (the parametric values), the effective potentials
 * and the tree-level potentials.
 */
struct PotentialData {
    std::vector<double> ts;
    std::vector<double> v_effs;
    std::vector<double> v_trees;
};

/**
 * Read in all parameters, normal vacuua and CB vacuua for the type A1 data,
 * which has a global charge-breaking minimum and a local normal minimum.
 * @return vector of the parameters, normal vacuua and charge-breaking vacuua.
 */
std::vector<Point> read_data_from_file(const std::string &file_name) {

    std::ifstream infile(file_name);
    std::vector<Point> data;

    std::string value;
    // First line will be a header. This is a flag to tell if we are on first line.
    bool in_header_line = true;
    while (infile.good()) {
        if (in_header_line) {
            getline(infile, value);
            in_header_line = false;
            continue;
        }
        Parameters<double> params{};
        // Read in all parameters
        getline(infile, value, ',');
        params.m112 = std::stod(value);
        getline(infile, value, ',');
        params.m122 = std::stod(value);
        getline(infile, value, ',');
        params.m222 = std::stod(value);
        getline(infile, value, ',');
        params.lam1 = std::stod(value);
        getline(infile, value, ',');
        params.lam2 = std::stod(value);
        getline(infile, value, ',');
        params.lam3 = std::stod(value);
        getline(infile, value, ',');
        params.lam4 = std::stod(value);
        getline(infile, value, ',');
        params.lam5 = std::stod(value);
        getline(infile, value, ',');
        params.yt = std::stod(value);
        getline(infile, value, ',');
        params.mu = std::stod(value);
        // Read in normal vacuum
        Vacuum<double> nvac{};
        getline(infile, value, ',');
        nvac.vevs[0] = std::stod(value);
        getline(infile, value, ',');
        nvac.vevs[1] = std::stod(value);
        getline(infile, value, ',');
        nvac.vevs[2] = std::stod(value);
        // Read in charge-breaking vacuum
        Vacuum<double> cbvac{};
        getline(infile, value, ',');
        cbvac.vevs[0] = std::stod(value);
        getline(infile, value, ',');
        cbvac.vevs[1] = std::stod(value);
        getline(infile, value);
        cbvac.vevs[2] = std::stod(value);

        data.push_back(Point{params, nvac, cbvac});
    }

    infile.close();
    return data;
}


int main() {
    static std::string project_path = "/Users/loganmorrison/CLionProjects/thdm_one_loop_extrema";
    static std::string type_a1_path = project_path + "/run_data/type_a1.csv";
    static std::string type_a2_path = project_path + "/run_data/type_a2.csv";

    // Gather all the data
    auto a1_points = read_data_from_file(type_a1_path);
    auto a2_points = read_data_from_file(type_a2_path);

    Model model{a1_points[1]};

    model.sort_vacuua();
    std::cout << model.params << std::endl;
    std::cout << model.one_loop_vacuua[0] << std::endl;
    std::cout << model.one_loop_vacuua[1] << std::endl;

    model.fields.set_fields(model.one_loop_vacuua[0]);
    std::cout << "Tree-level masses at deepest." << std::endl;
    for (auto m2 : scalar_squared_masses(model.fields, model.params)) {
        std::cout << m2 << ", ";
    }
    std::cout << std::endl;

    std::cout << "one-loop masses at deepest." << std::endl;
    for (auto m2 : potential_eff_hessian_evals(model.fields, model.params)) {
        std::cout << m2 << ", ";
    }
    std::cout << std::endl;

    model.fields.set_fields(model.one_loop_vacuua[1]);
    std::cout << "Tree-level masses at next deepest." << std::endl;
    for (auto m2 : scalar_squared_masses(model.fields, model.params)) {
        std::cout << m2 << ", ";
    }
    std::cout << std::endl;

    std::cout << "one-loop masses at next deepest." << std::endl;
    for (auto m2 : potential_eff_hessian_evals(model.fields, model.params)) {
        std::cout << m2 << ", ";
    }
    std::cout << std::endl;

    return 0;
}