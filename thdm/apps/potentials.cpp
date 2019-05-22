//
// Created by Logan Morrison on 2019-05-19.
//

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

struct PotentialData {
    std::vector<double> ts;
    std::vector<double> v_effs;
    std::vector<double> v_trees;
};

/**
 * Read in all parameters, normal vacuua and CB vacuua for the type A1 data.
 * @return vector of the parameters.
 */
std::vector<Point> read_data_from_file() {
    static std::string project_path = "/Users/loganmorrison/CLionProjects/thdm_one_loop_extrema";
    static std::string type_a1_path = project_path + "/run_data/type_a1.csv";

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
        set_top_yukawa(params, nvac);
        std::cout << params << std::endl;
        std::cout << nvac << std::endl;
        std::cout << cbvac << std::endl;

        data.push_back(Point{params, nvac, cbvac});
    }

    infile.close();
    return data;
}

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
        std::string file = project_path + "/run_data/potentials/potential_" +
                std::to_string(counter) + ".csv";
        auto data = compute_potential_data(point);
        save_data_to_file(data, file);
        counter++;
    }
    return 0;
}