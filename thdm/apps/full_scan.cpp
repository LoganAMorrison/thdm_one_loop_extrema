//
// Created by Logan Morrison on 2019-05-03.
//


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
size_t type_a1_counter = 0;
size_t type_a2_counter = 0;

struct Point {
    Parameters<double> params;
    Vacuum<double> nvac;
    Vacuum<double> cbvac;
};

void save_point(const Point &point, const std::string &file_name) {

    std::ofstream out_file;
    out_file.open(file_name, std::ios_base::app);
    out_file << std::setprecision(15);

    out_file << point.params.m112 << ",";
    out_file << point.params.m122 << ",";
    out_file << point.params.m222 << ",";
    out_file << point.params.lam1 << ",";
    out_file << point.params.lam2 << ",";
    out_file << point.params.lam3 << ",";
    out_file << point.params.lam4 << ",";
    out_file << point.params.lam5 << ",";
    out_file << point.params.mu << ",";
    out_file << point.nvac.vevs[0] << ",";
    out_file << point.nvac.vevs[1] << ",";
    out_file << point.nvac.vevs[2] << ",";
    out_file << point.cbvac.vevs[0] << ",";
    out_file << point.cbvac.vevs[1] << ",";
    out_file << point.cbvac.vevs[2] << "\n";
    out_file.close();
}

/**
 * Determine if model is type A1. If it is, store the normal and
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
        // Find the deepest normal min
        nvac.potential = 0.0;
        for (auto &vac : model.one_loop_vacuua) {
            if (std::abs(vac.vevs[2]) < 1e-2) {
                if (vac.extrema_type == SingleExtremaType::Minimum) {
                    if (vac.potential < nvac.potential) {
                        nvac = vac;
                    }
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
        // Find deepest cb min
        cbvac.potential = 0.0;
        for (auto &vac : model.one_loop_vacuua) {
            if (std::abs(vac.vevs[2]) > 1e-2) {
                if (vac.extrema_type == SingleExtremaType::Minimum) {
                    if (vac.potential < cbvac.potential) {
                        cbvac = vac;
                    }
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
    std::cout << "Type A1 : " << type_a1_counter << std::endl;
    std::cout << "Type A2 : " << type_a2_counter << std::endl;
    std::cout << "Total   : " << COUNTER << std::endl;
    std::cout << std::endl;
}

int main() {
    double mu = 246.0;
    std::string project_path = "/Users/loganmorrison/CLionProjects/thdm_one_loop_extrema";
    std::string a1_file_name = project_path + "/run_data/type_a1.csv";
    std::string a2_file_name = project_path + "/run_data/type_a2.csv";
    //boost::progress_display progress(TOTAL_NUM_POINTS);
    while (COUNTER < TOTAL_NUM_POINTS) {
        try {
            Model model(mu);
            Vacuum<double> nvac{};
            Vacuum<double> cbvac{};
            if (is_type_a1(model, nvac, cbvac)) {
                save_point(Point{model.params, nvac, cbvac}, a1_file_name);
                COUNTER++;
                type_a1_counter++;
                display_counts();
            } else if (is_type_a2(model, nvac, cbvac)) {
                save_point(Point{model.params, nvac, cbvac}, a2_file_name);
                COUNTER++;
                type_a2_counter++;
                display_counts();
            }
        } catch (...) {
            // Oops! Something went wrong...
        }
    }


    return 0;
}

