//
// Created by Logan Morrison on 2019-04-27.
//

#define ARMA_DONT_PRINT_ERRORS

#include "thdm/fields.hpp"
#include "thdm/vacuua.hpp"
#include "thdm/parameters.hpp"
#include "thdm/potentials.hpp"
#include "thdm/jacobi.hpp"
#include "thdm/model.hpp"
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <tuple>
#include <algorithm>

using namespace thdm;

void print_masses(Model &model, const Vacuum<double> &vac) {
    Fields<double> fields{};
    fields.set_fields(vac);
    auto masses = potential_eff_hessian_evals(fields, model.params);
    std::sort(masses.begin(), masses.end(), std::less<double>());
    for (const auto &mass: masses) {
        std::cout << mass << " ";
    }
    std::cout << std::endl;
}

std::vector<Model> get_full_data() {
    std::ifstream in_file;
    std::string file_name;
    #ifdef SCALAR_ONLY
    file_name = "/Users/loganmorrison/CLionProjects"
                "/thdm_one_loop_extrema/run_data/raw_data_scalar_only.csv";
    #else
    file_name = "/Users/loganmorrison/CLionProjects"
                "/thdm_one_loop_extrema/run_data/raw_data.csv";
    #endif
    in_file.open(file_name);

    std::vector<Model> data;

    std::string value;
    while (in_file.good()) {
        Parameters<double> params{};
        // Read in all parameters
        getline(in_file, value, ',');
        params.m112 = std::stod(value);
        getline(in_file, value, ',');
        params.m122 = std::stod(value);
        getline(in_file, value, ',');
        params.m222 = std::stod(value);
        getline(in_file, value, ',');
        params.lam1 = std::stod(value);
        getline(in_file, value, ',');
        params.lam2 = std::stod(value);
        getline(in_file, value, ',');
        params.lam3 = std::stod(value);
        getline(in_file, value, ',');
        params.lam4 = std::stod(value);
        getline(in_file, value, ',');
        params.lam5 = std::stod(value);
        getline(in_file, value, ',');
        params.mu = std::stod(value);
        // Read in normal vacuum
        Vacuum<double> nvac{};
        getline(in_file, value, ',');
        nvac.vevs[0] = std::stod(value);
        getline(in_file, value, ',');
        nvac.vevs[1] = std::stod(value);
        getline(in_file, value, ',');
        nvac.vevs[2] = std::stod(value);
        // Read in charge-breaking vacuum
        Vacuum<double> cbvac{};
        getline(in_file, value, ',');
        cbvac.vevs[0] = std::stod(value);
        getline(in_file, value, ',');
        cbvac.vevs[1] = std::stod(value);
        getline(in_file, value);
        cbvac.vevs[2] = std::stod(value);

        data.emplace_back(Point{params, nvac, cbvac});
    }

    in_file.close();

    return data;
}

std::vector<Point> extract_good_data() {
    std::vector<Point> points;
    auto models = get_full_data();

    for (auto &model :models) {
        model.minimize_from_random_vacuum_and_refine(100);

        auto nvac = Vacuum<double>{};
        auto cbvac = Vacuum<double>{};

        auto deepest = get_deepest_vacuum(model);

        if (has_cb_and_normal_minima(model, nvac, cbvac) &&
                deepest.extrema_type == SingleExtremaType::Minimum) {
            points.push_back(Point{model.params, nvac, cbvac});
        }
    }
    return points;
}

void write_good_data(const Point &point, std::string file_name) {
    std::ofstream out_file;
    out_file.open(file_name, std::ios_base::app);
    out_file << std::setprecision(15);

    auto params = point.params;
    auto nvac = point.nvac;
    auto cbvac = point.cbvac;
    out_file << params.m112 << ",";
    out_file << params.m122 << ",";
    out_file << params.m222 << ",";
    out_file << params.lam1 << ",";
    out_file << params.lam2 << ",";
    out_file << params.lam3 << ",";
    out_file << params.lam4 << ",";
    out_file << params.lam5 << ",";
    out_file << params.mu << ",";
    out_file << nvac.vevs[0] << ",";
    out_file << nvac.vevs[1] << ",";
    out_file << nvac.vevs[2] << ",";
    out_file << cbvac.vevs[0] << ",";
    out_file << cbvac.vevs[1] << ",";
    out_file << cbvac.vevs[2] << std::endl;

    out_file.close();
}

int main() {
    auto data = extract_good_data();
    std::string file_name;
    for (const auto &point : data) {
        if (point.cbvac < point.nvac) {
            file_name = "/Users/loganmorrison/CLionProjects"
                        "/thdm_one_loop_extrema/run_data/type_a1.csv";
        } else {
            file_name = "/Users/loganmorrison/CLionProjects"
                        "/thdm_one_loop_extrema/run_data/type_a2.csv";
        }
        write_good_data(point, file_name);
    }
}