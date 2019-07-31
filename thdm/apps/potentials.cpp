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
std::vector<Point> read_data_from_file() {
    static std::string project_path = "/Users/loganmorrison/CLionProjects/thdm_one_loop_extrema";
    static std::string type_a1_path = project_path + "/run_data/type_a2.csv";

    std::ifstream infile(type_a1_path);
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

        std::cout << params << std::endl;
        std::cout << nvac << std::endl;
        std::cout << cbvac << std::endl;

        data.push_back(Point{params, nvac, cbvac});
    }

    infile.close();
    return data;
}

/**
 * Compute the effective and tree-level potential along a line connecting the
 * normal and charge-breaking vacuua.
 * @param point Point struct holding the THDM parameters, the normal and
 * charge-breaking vacuua.
 * @return A data structure holding vectors of the parametric parameters (ts),
 * and the effective and tree-level potentials evaluated at the ts.
 */
PotentialData compute_potential_data(const Point &point) {
    size_t NUM_POINTS = 200;
    std::vector<double> ts(NUM_POINTS);
    std::vector<double> v_effs(NUM_POINTS);
    std::vector<double> v_trees(NUM_POINTS);
    PotentialData data{ts, v_effs, v_trees};
    double t0 = -0.2;
    double t1 = 1.2;
    double t_step = (t1 - t0) / (double) (NUM_POINTS - 1);

    auto nvac = point.nvac;
    auto cbvac = point.cbvac;
    auto params = point.params;
    Fields<double> fields{};
    auto interp_vac = nvac;

    // Fill ts
    for (size_t i = 0; i < data.ts.size(); i++) {
        data.ts[i] = t0 + i * t_step;
    }
    // Fill potentials
    for (size_t i = 0; i < data.ts.size(); i++) {
        // Interpolate the vacuua
        double t = data.ts[i];
        interp_vac.vevs[0] = nvac.vevs[0] * (1 - t) + cbvac.vevs[0] * t;
        interp_vac.vevs[1] = nvac.vevs[1] * (1 - t) + cbvac.vevs[1] * t;
        interp_vac.vevs[2] = nvac.vevs[2] * (1 - t) + cbvac.vevs[2] * t;
        fields.set_fields(interp_vac);
        data.v_effs[i] = potential_eff(fields, params);
        data.v_trees[i] = potential_tree(fields, params);
    }
    return data;
}

/**
 * Save the for the potentials and parametric values to a data file in the
 * 'run_data/potentials' directory. The naming conversion for the files
 * corresponding to which point in the 'type_a1.csv' file we are using. For
 * example, 'potentials_0.csv' has all the potential data for the first point
 * in the 'type_a1.csv' data file.
 * @param data The potential data to save.
 * @param file File name of file we would like to save to.
 */
void save_data_to_file(const PotentialData &data, const std::string &file) {
    std::ofstream out_file;
    out_file.open(file);
    out_file << std::setprecision(15);

    // Write header
    out_file << "t" << "," << "veff" << "," << "vtree" << std::endl;

    // Write data
    for (size_t i = 0; i < data.ts.size(); i++) {
        out_file << data.ts[i] << ",";
        out_file << data.v_effs[i] << ",";
        out_file << data.v_trees[i] << std::endl;
    }

    out_file.close();
}


int main() {
    // Create counter for the naming files
    int counter = 0;

    // Gather all the data
    auto points = read_data_from_file();
    // Generate potential data and save
    for (auto &point: points) {
        // Create file name
        std::string project_path = "/Users/loganmorrison/CLionProjects/thdm_one_loop_extrema";
        std::string file = project_path + "/run_data/potentials/type_a2/potential_" +
                std::to_string(counter) + ".csv";
        auto data = compute_potential_data(point);
        save_data_to_file(data, file);
        counter++;
    }
    return 0;
}