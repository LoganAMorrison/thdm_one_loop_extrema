//
// Created by Logan Morrison on 2019-05-03.
//

/*
 * Type A1: Global CB minimum with local normal minimum
 * Type A2: Global normal minimum with local cb minimum
 * Type B: Global CB minimum with normal saddle
 * Type C: Global normal minimum with CB saddle
 */

/*
 * old:
 else if (is_type_b(model, nvac, cbvac)) {
                save_point(model.params, nvac, cbvac, fname3);
                COUNTER++;
                TYPEB_COUNTER++;
                display_counts();
            } else if (is_type_c(model, nvac, cbvac)) {
                save_point(model.params, nvac, cbvac, fname4);
                COUNTER++;
                TYPEC_COUNTER++;
                display_counts();
            }

 */


#include "thdm/model.hpp"
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
#include <cstdlib>

using namespace thdm;

const size_t TOTAL_NUM_POINTS = 100000;
size_t COUNTER = 0;
size_t TYPEA1_COUNTER = 0;
size_t TYPEA2_COUNTER = 0;
size_t TYPEB_COUNTER = 0;
size_t TYPEC_COUNTER = 0;
std::string fname1 = "/Users/loganmorrison/Documents/research/thdm_cpb_vev/cpp/rundata/type_a1.csv";
std::string fname2 = "/Users/loganmorrison/Documents/research/thdm_cpb_vev/cpp/rundata/type_a2.csv";
//std::string fname3 = "/Users/loganmorrison/Documents/research/thdm_cpb_vev/cpp/rundata/type_b.csv";
//std::string fname4 = "/Users/loganmorrison/Documents/research/thdm_cpb_vev/cpp/rundata/type_c.csv";
std::ofstream ofile;

std::mutex mtx;

void save_point(Parameters<double> &params, Vacuum<double> &nvac,
                Vacuum<double> &cbvac, std::string fname) {
    // Lock mutex so only one thread is writing to file at a time.
    mtx.lock();
    ofile.open(fname, std::ios_base::app);
    ofile << std::setprecision(15);

    ofile << params.m112 << ",";
    ofile << params.m122 << ",";
    ofile << params.m222 << ",";
    ofile << params.lam1 << ",";
    ofile << params.lam2 << ",";
    ofile << params.lam3 << ",";
    ofile << params.lam4 << ",";
    ofile << params.lam5 << ",";
    ofile << params.mu << "\n";
    ofile << nvac.vevs[0] << ",";
    ofile << nvac.vevs[1] << ",";
    ofile << nvac.vevs[2] << ",";
    ofile << cbvac.vevs[0] << ",";
    ofile << cbvac.vevs[1] << ",";
    ofile << cbvac.vevs[2] << "\n";
    ofile.close();
    mtx.unlock();
}

/**
 * Determine if model is type A1. If it is, store the noraml and
 * charge breaking vacuua in the inputs.
 * @param model
 * @param nvac
 * @param cbvac
 * @return
 */
bool is_type_a1(const Model &model, Vacuum<double> &nvac, Vacuum<double> &cbvac) {
    if (model.one_loop_deepest.extrema_type != SingleExtremaType::Minimum) {
        return false;
    }

    if (model.is_cb_deepest && model.has_cb_min && model.has_normal_min) {
        cbvac = model.one_loop_deepest;
        // Find normal min
        for (auto &vac : model.one_loop_vacuua) {
            if (std::abs(vac.vevs[2]) < 1e-5) {
                if (vac.extrema_type == SingleExtremaType::Minimum) {
                    nvac = vac;
                    break;
                }
            }
        }
        return true;
    }
    return false;


}

bool is_type_a2(const Model &model, Vacuum<double> &nvac, Vacuum<double> &cbvac) {
    if (model.one_loop_deepest.extrema_type != SingleExtremaType::Minimum) {
        return false;
    }

    if (!model.is_cb_deepest && model.has_cb_min && model.has_normal_min) {
        nvac = model.one_loop_deepest;
        // Find cb min
        for (auto &vac : model.one_loop_vacuua) {
            if (std::abs(vac.vevs[2]) > 1e-5) {
                if (vac.extrema_type == SingleExtremaType::Minimum) {
                    cbvac = vac;
                    break;
                }
            }
        }
        return true;
    }
    return false;


}

bool is_type_b(const Model &model, Vacuum<double> &nvac, Vacuum<double> &cbvac) {
    if (model.one_loop_deepest.extrema_type != SingleExtremaType::Minimum) {
        return false;
    }
    if (model.is_cb_deepest && model.has_cb_min && !model.has_normal_min) {
        cbvac = model.one_loop_deepest;
        // Find cb min
        for (auto &vac : model.one_loop_vacuua) {
            if (std::abs(vac.vevs[2]) < 1e-5) {
                if (vac.extrema_type == SingleExtremaType::Saddle) {
                    nvac = vac;
                    break;
                }
            }
        }
        return true;
    }
    return false;


}

bool is_type_c(const Model &model, Vacuum<double> &nvac, Vacuum<double> &cbvac) {
    if (model.one_loop_deepest.extrema_type != SingleExtremaType::Minimum) {
        return false;
    }
    if (!model.is_cb_deepest && !model.has_cb_min && model.has_normal_min) {
        nvac = model.one_loop_deepest;
        // Find cb min
        for (auto &vac : model.one_loop_vacuua) {
            if (std::abs(vac.vevs[2]) > 1e-5) {
                if (vac.extrema_type == SingleExtremaType::Saddle) {
                    cbvac = vac;
                    break;
                }
            }
        }
        return true;
    }
    return false;


}

void display_counts() {
    std::cout << "Counts:   " << std::endl;
    std::cout << "----------" << std::endl;
    std::cout << "Type A1 : " << TYPEA1_COUNTER << std::endl;
    std::cout << "Type A2 : " << TYPEA2_COUNTER << std::endl;
    //std::cout << "Type B  : " << TYPEB_COUNTER << std::endl;
    //std::cout << "Type C  : " << TYPEC_COUNTER << std::endl;
    std::cout << "Total   : " << COUNTER << std::endl;
    std::cout << std::endl;
}

int main() {
    double mu = 246.0;
    //boost::progress_display progress(TOTAL_NUM_POINTS);
    while (COUNTER < TOTAL_NUM_POINTS) {
        try {
            Model model(mu);
            Vacuum<double> nvac{};
            Vacuum<double> cbvac{};
            if (is_type_a1(model, nvac, cbvac)) {
                save_point(model.params, nvac, cbvac, fname1);
                COUNTER++;
                TYPEA1_COUNTER++;
                display_counts();
            } else if (is_type_a2(model, nvac, cbvac)) {
                save_point(model.params, nvac, cbvac, fname2);
                COUNTER++;
                TYPEA2_COUNTER++;
                display_counts();
            }
        } catch (...) {
            // Oops! Something went wrong...
        }
    }


    return 0;
}

