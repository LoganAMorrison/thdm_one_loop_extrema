//
// Created by Logan Morrison on 2019-04-27.
//

#include "thdm/potentials.hpp"
#include "thdm/parameters.hpp"
#include "thdm/fields.hpp"
#include "thdm/root_finding_eff.hpp"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <tuple>
#include <boost/progress.hpp>
#include <mutex>
#include <thread>

using namespace thdm;

size_t num_threads = 5;
size_t pts_per_thread = 10000;
boost::progress_display progress(pts_per_thread * num_threads);
std::string fname = "/Users/loganmorrison/Documents/research/thdm_cpb_vev/cpp/rundata/initial_data.csv";
std::ofstream ofile;
size_t POINT_COUNTER = 0;

typedef std::tuple<Vacuum<double>, Vacuum<double>, Parameters<double>> Point;
std::mutex mtx;

bool verify_goldstones(Point point) {
    bool hasgoldstones = true;

    Fields<double> fields{};
    auto nvac = std::get<0>(point);
    auto cbvac = std::get<1>(point);
    auto params = std::get<2>(point);

    try {
        fields.set_fields(nvac);
        auto nmasses = potential_eff_hessian_evals(fields, params);
        // Make all masses positive
        for (double &mass : nmasses)
            mass = abs(mass);
        // Sort masses
        std::sort(nmasses.begin(), nmasses.end(), std::less<double>());
        // first three should have masses that are less
        // near zero.
        hasgoldstones = hasgoldstones && (nmasses[0] < 1e-8);
        hasgoldstones = hasgoldstones && (nmasses[1] < 1e-8);
        hasgoldstones = hasgoldstones && (nmasses[2] < 1e-8);

        fields.set_fields(cbvac);
        auto cbmasses = potential_eff_hessian_evals(fields, params);
        for (double &mass : cbmasses)
            mass = abs(mass);
        std::sort(cbmasses.begin(), cbmasses.end(), std::less<double>());
        // Should have 4 goldstones.
        hasgoldstones = hasgoldstones && (cbmasses[0] < 1e-8);
        hasgoldstones = hasgoldstones && (cbmasses[1] < 1e-8);
        hasgoldstones = hasgoldstones && (cbmasses[2] < 1e-8);
        hasgoldstones = hasgoldstones && (cbmasses[3] < 1e-8);
    } catch (...) {
        //std::cout << "Failed at goldstones" << std::endl;
        hasgoldstones = false;
    }

    return hasgoldstones;
}


Point generate_point(double mu) {
    auto res = solve_root_equations_eff(mu);
    return res;
}

void save_point(Point point) {
    // Lock mutex so only one thread is writing to file at a time.
    mtx.lock();
    ofile.open(fname, std::ios_base::app);
    ofile << std::setprecision(15);

    auto nvac = std::get<0>(point);
    auto cbvac = std::get<1>(point);
    auto params = std::get<2>(point);
    ofile << params.m112 << ",";
    ofile << params.m122 << ",";
    ofile << params.m222 << ",";
    ofile << params.lam1 << ",";
    ofile << params.lam2 << ",";
    ofile << params.lam3 << ",";
    ofile << params.lam4 << ",";
    ofile << params.lam5 << ",";
    ofile << nvac.vevs[0] << ",";
    ofile << nvac.vevs[1] << ",";
    ofile << cbvac.vevs[0] << ",";
    ofile << cbvac.vevs[1] << ",";
    ofile << cbvac.vevs[2] << "\n";
    ofile.close();
    mtx.unlock();
}

void generate_and_save(double mu) {
    auto point = generate_point(mu);
    if (verify_goldstones(point)) {
        save_point(point);
        POINT_COUNTER++;
        ++progress;
    }
}

void generate_and_save_many(double mu, size_t num_pts) {
    while (POINT_COUNTER < num_pts) {
        generate_and_save(mu);
    }

}

int main() {
    double mu = 246.0;

    generate_and_save_many(mu, pts_per_thread);

    return 0;
}

