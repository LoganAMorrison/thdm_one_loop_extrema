//
// Created by Logan Morrison on 2019-04-27.
//

#include "thdm/parameters.hpp"
#include "thdm/fields.hpp"
#include "thdm/model.hpp"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <tuple>
#include <boost/progress.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/bind.hpp>

using namespace thdm;

const size_t pts_per_thread = 2000;
const size_t num_threads = 8;
boost::progress_display progress(pts_per_thread * num_threads);
static boost::mutex mtx;

void save_point(const Model &model) {
    // Lock mutex so only one thread is writing to file at a time.
    boost::lock_guard<boost::mutex> lock(mtx);
    // Create and open file object
    std::ofstream out_file;
    std::string file_name;
    #ifdef SCALAR_ONLY
    file_name = "/Users/loganmorrison/CLionProjects"
                "/thdm_one_loop_extrema/run_data/raw_data_scalar_only.csv";
    #else
    file_name = "/Users/loganmorrison/CLionProjects"
                "/thdm_one_loop_extrema/run_data/raw_data.csv";
    #endif
    out_file.open(file_name, std::ios_base::app);
    out_file << std::setprecision(15);

    // Normal vacuum will be in first place and cb in second.
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

void generate_and_save(double mu) {
    Model model(mu);
    save_point(model);
}

void generate_and_save_many(double mu, size_t num_pts) {
    for (size_t i = 0; i < num_pts; i++)
        generate_and_save(mu);
}

int main() {
    double mu = 246.0;
    //boost::thread_group thread_group;

    for (size_t j = 0; j < num_threads; j++) {
        //thread_group.create_thread(boost::bind(generate_and_save_many, mu, pts_per_thread));
        generate_and_save_many(mu, pts_per_thread);
    }
    //thread_group.join_all();

    return 0;
}

