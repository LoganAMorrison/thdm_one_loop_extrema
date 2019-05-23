//
// Created by Logan Morrison on 2019-04-27.
//

#include "thdm/parameters.hpp"
#include "thdm/fields.hpp"
#include "thdm/model.hpp"
#include "thdm/csv_parser.hpp"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <tuple>
#include <boost/progress.hpp>

using namespace thdm;

const size_t NUM_POINTS_TO_GENERATE = 10000;
boost::progress_display progress(NUM_POINTS_TO_GENERATE);

/**
 * Save the model to the 'raw_data.csv' or 'raw_data_scalar_only.csv', depending
 * on whether or not 'SCALAR_ONLY' is defined. I.e., save the THDM parameters,
 * the normal vacuum and charge-breaking vacuum to the respective file..
 * @param model THDM model to save to file.
 */
void save_point(const Model &model) {
    std::ofstream out_file;
    std::string file_name;
    // If SCALAR_ONLY is set, we turn off gauge bosons and the top quark and
    // save to a different file called 'raw_data_scalar_only.csv'.
    #ifdef SCALAR_ONLY
    file_name = "/Users/loganmorrison/CLionProjects"
                "/thdm_one_loop_extrema/run_data/raw_data_scalar_only.csv";
    #else
    file_name = "/Users/loganmorrison/CLionProjects"
                "/thdm_one_loop_extrema/run_data/raw_data.csv";
    #endif
    out_file.open(file_name, std::ios_base::app);
    out_file << std::setprecision(15);

    auto nvac = model.one_loop_vacuua[0];
    auto cbvac = model.one_loop_vacuua[1];
    auto params = model.params;
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
    ++progress;
    out_file.close();
}

/**
 * Generate a new model and save the parameters, normal vacuum and
 * charge-breaking vacuum to the 'raw_data.csv' file.
 * @param mu renormalization scale.
 */
void generate_and_save(double mu) {
    Model model(mu);
    save_point(model);
}


int main() {
    const double MU = 246.0; // Renormalization scale

    for (size_t i = 0; i < NUM_POINTS_TO_GENERATE; i++) {
        generate_and_save(MU);
    }

    return 0;
}

