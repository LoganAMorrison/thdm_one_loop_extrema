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
#include <cmath>

using namespace thdm;

/**
 * A debug function to print the one-loop scalar squared masses at a given
 * vacuum.
 * @param model THDM Model object.
 * @param vac Vacuum to evaluate masses at.
 */
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

/**
 * Check that the lambdas are small enough (less than 10 in absolute value)
 * and that the mass parameters don't exceed 10 * mu^2 (to avoid large logs).
 * @param model THDM Model object
 * @return True if parameters satisfy the criteria.
 */
bool are_parameters_good(Model &model) {
    bool are_good = true;
    // Make value of lambda's is 10. Larger than this and we start violating
    // perturbative unitarity
    double lam_max = 10.0;
    // Make value of m112, m122, and m222 is 10 mu^2. This is to avoid large
    // logs.
    double m2_max = pow(10.0 * model.params.mu, 2);
    are_good = are_good && (std::abs(model.params.lam1) < lam_max);
    are_good = are_good && (std::abs(model.params.lam2) < lam_max);
    are_good = are_good && (std::abs(model.params.lam3) < lam_max);
    are_good = are_good && (std::abs(model.params.lam4) < lam_max);
    are_good = are_good && (std::abs(model.params.lam5) < lam_max);
    are_good = are_good && (std::abs(model.params.m112) < m2_max);
    are_good = are_good && (std::abs(model.params.m122) < m2_max);
    are_good = are_good && (std::abs(model.params.m222) < m2_max);
    return are_good;
}

/**
 * Write the desired data to the 'type_a1.csv' and 'type_a2.csv' data files.
 * If the normal vacuum is the deepest, then we write to 'type_a2.csv' and if
 * the charge-breaking vacuum is the deepest, to 'type_a1.csv'.
 * @param point Point struct to write to file.
 * @param file_name File name to write to: either type_a1.csv' or type_a2.csv'
 */
void write_good_data(const Point &point, const std::string &file_name) {
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
    out_file << params.yt << ",";
    out_file << params.mu << ",";
    out_file << nvac.vevs[0] << ",";
    out_file << nvac.vevs[1] << ",";
    out_file << nvac.vevs[2] << ",";
    out_file << cbvac.vevs[0] << ",";
    out_file << cbvac.vevs[1] << ",";
    out_file << cbvac.vevs[2] << std::endl;

    out_file.close();
}

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

/**
 * Extract the 'good' data from the raw data and save to file. By 'good',
 * we mean that there is both a normal and charge-breaking minimum and the
 * THDM parameters satisfy perturbative unitarity and don't have large masses
 * (mass squared values which exceed 10 mu^2, which would lead to large logs).
 * @return A vector of the 'good' points. This is a struct containing the
 * THDM parameters, the normal minimum and charge-breaking minimum.
 */
void extract_good_data_and_save() {
    static std::string project_dir = "/Users/loganmorrison/CLionProjects"
                                     "/thdm_one_loop_extrema";
    static std::string file_name_a1 = project_dir + "/run_data/type_a1.csv";
    static std::string file_name_a2 = project_dir + "/run_data/type_a2.csv";
    static std::string file_name_raw;
    #ifdef SCALAR_ONLY
    file_name_raw = project_dir + "/run_data/raw_data_scalar_only.csv";
    #else
    file_name_raw = project_dir + "/run_data/raw_data.csv";
    #endif

    // File stream for reading in raw data.
    std::ifstream in_file;
    in_file.open(file_name_raw);

    // We will go through each point in the raw data, create a model from
    // the point, try to find the deepest normal and cb vacuua and then
    // save the point to a file if there are normal and charge-breaking
    // minima.
    std::string value;
    // The raw data file has a header, so we skip that line.
    bool on_header_line = true;
    while (in_file.good()) {
        if (on_header_line) {
            getline(in_file, value);
            on_header_line = false;
        }
        // Create the model from the parameters and normal + charge-breaking
        // vacuua, then try to find new vacuua by minimizing the potential
        // starting from 100 different random vacuua.
        Model model(read_in_next_point(in_file));
        model.minimize_from_random_vacuum_and_refine(50);
        auto nvac = Vacuum<double>{};
        auto cbvac = Vacuum<double>{};
        auto deepest = get_deepest_vacuum(model);
        // Check that we a minimum a a normal and charge-breaking vacuum and
        // that the deepest vacuum is a minimum. If its not, the potential
        // seems unstable.
        if (has_cb_and_normal_minima(model, nvac, cbvac) &&
                deepest.extrema_type == SingleExtremaType::Minimum) {
            // Check that parameters obey the criteria of pertubative unitarity
            // and avoid large logs.
            if (are_parameters_good(model)) {
                auto point = Point{model.params, nvac, cbvac};
                if (cbvac < nvac) {
                    write_good_data(point, file_name_a1);
                } else {
                    write_good_data(point, file_name_a2);
                }
            }
        }
    }
    in_file.close();
}

/**
 * Create the headers for the 'type_a1.csv' and 'type_a2.csv' files.
 */
void write_headers() {
    std::string a1_file = "/Users/loganmorrison/CLionProjects"
                          "/thdm_one_loop_extrema/run_data/type_a1.csv";
    std::string a2_file = "/Users/loganmorrison/CLionProjects"
                          "/thdm_one_loop_extrema/run_data/type_a2.csv";
    std::ofstream out_file_a1;
    std::ofstream out_file_a2;
    out_file_a1.open(a1_file);
    out_file_a2.open(a2_file);

    out_file_a1 << "m112" << "," << "m122" << "," << "m222" << ","
                << "lam1" << "," << "lam2" << "," << "lam3" << ","
                << "lam4" << "," << "lam5" << "," << "yt" << ","
                << "mu" << "," << "nvev1" << "," << "nvev2" << ","
                << "nvev3" << "," << "cbvev1" << "," << "cbvev2" << ","
                << "cbvev3" << std::endl;
    out_file_a2 << "m112" << "," << "m122" << "," << "m222" << ","
                << "lam1" << "," << "lam2" << "," << "lam3" << ","
                << "lam4" << "," << "lam5" << "," << "yt" << ","
                << "mu" << "," << "nvev1" << "," << "nvev2" << ","
                << "nvev3" << "," << "cbvev1" << "," << "cbvev2" << ","
                << "cbvev3" << std::endl;

    out_file_a1.close();
    out_file_a2.close();
}

int main() {
    write_headers();
    extract_good_data_and_save();
}