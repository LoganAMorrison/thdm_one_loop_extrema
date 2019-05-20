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
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>

using namespace thdm;

std::string project_path = "/Users/loganmorrison/CLionProjects/thdm_one_loop_extrema";

struct Point {
    Parameters<double> params;
    Vacuum<double> nvac;
    Vacuum<double> cbvac;
};

struct RGEData {
    std::vector<double> mus;
    std::vector<Parameters<double>> params_vec;
    std::vector<double> potential_eff_normal;
    std::vector<double> potential_eff_cb;
};

/**
 * Read in all parameters, normal vacuua and CB vacuua for the type A1 data.
 * @return vector of the parameters.
 */
std::vector<Point> read_data_from_file() {
    std::string type_a1_path = project_path + "/run_data/type_a1.csv";

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

        data.push_back(Point{params, nvac, cbvac});
    }

    infile.close();
    return data;
}

RGEData run_point(const Point &point) {
    int num_mus = 100;
    double mu1 = 246.0;
    double mu2 = 200.0;
    double mu_step = (mu2 - mu1) / (num_mus - 1);

    auto params = point.params;
    auto nvac = point.nvac;
    auto cbvac = point.cbvac;
    Fields<double> fields{};

    params.gp = U1Y_COUP;
    params.g = SU2_COUP;
    set_top_yukawa(params, nvac);

    RGEData rge_data{std::vector<double>(num_mus), // mus
            std::vector<Parameters<double>>(num_mus), // parameters
            std::vector<double>(num_mus), // effective potential eff normal
            std::vector<double>(num_mus) // effective potential eff cb
    };

    // Fill in the values for mus
    for (int i = 0; i < num_mus; i++) {
        rge_data.mus[i] = i * mu_step + mu1;
    }

    // Set initial values
    fields.set_fields(nvac);
    rge_data.potential_eff_normal[0] = potential_eff(fields, params);
    fields.set_fields(cbvac);
    rge_data.potential_eff_cb[0] = potential_eff(fields, params);
    rge_data.params_vec[0] = params;

    // run data from old mu to new mu
    for (int i = 1; i < num_mus; i++) {
        params = run_parameters(params, rge_data.mus[i - 1], rge_data.mus[i]);
        rge_data.params_vec[i] = params;
        // Root solve for the new vacuua starting at the old vacuua.
        refine_root(params, nvac);
        refine_root(params, cbvac);

        fields.set_fields(nvac);
        rge_data.potential_eff_normal[i] = potential_eff(fields, params);
        fields.set_fields(cbvac);
        rge_data.potential_eff_cb[i] = potential_eff(fields, params);
    }

    return rge_data;
}

void save_rge_data(RGEData data, const std::string &save_file) {
    std::ofstream out_file;
    out_file.open(save_file);
    out_file << std::setprecision(15);

    // Write header
    out_file << "mu" << "," << "VN" << "," << "VCB" << ","
             << "m112" << "," << "m122" << "," << "m222" << ","
             << "lam1" << "," << "lam2" << "," << "lam3" << ","
             << "lam4" << "," << "lam5" << "," << "yt" << ","
             << "gp" << "," << "g" << std::endl;

    // Write data
    for (size_t i = 0; i < data.mus.size(); i++) {
        out_file << data.mus[i] << ",";
        out_file << data.potential_eff_normal[i] << ",";
        out_file << data.potential_eff_cb[i] << ",";
        out_file << data.params_vec[i].m112 << ",";
        out_file << data.params_vec[i].m122 << ",";
        out_file << data.params_vec[i].m222 << ",";
        out_file << data.params_vec[i].lam1 << ",";
        out_file << data.params_vec[i].lam2 << ",";
        out_file << data.params_vec[i].lam3 << ",";
        out_file << data.params_vec[i].lam4 << ",";
        out_file << data.params_vec[i].lam5 << ",";
        out_file << data.params_vec[i].yt << ",";
        out_file << data.params_vec[i].gp << ",";
        out_file << data.params_vec[i].g << std::endl;
    }

    out_file.close();
}

int main() {


    // Read in all data
    auto data = read_data_from_file();

    // For each data point, run and save data to file.
    int counter = 0; // Counter for naming the files.
    for (auto point: data) {
        std::string save_file = (project_path +
                "/run_data/RGE/rge_" + std::to_string(counter) + ".csv");
        try {
            auto rge_data = run_point(point);
            save_rge_data(rge_data, save_file);
            counter++;
        } catch (...) {
            // Oops! something went wrong..
        }
    }

    return 0;
}