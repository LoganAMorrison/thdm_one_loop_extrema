//
// Created by Logan Morrison on 2019-04-27.
//

#include "thdm/fields.hpp"
#include "thdm/vacuua.hpp"
#include "thdm/parameters.hpp"
#include "thdm/potentials.hpp"
#include "thdm/jacobi.hpp"
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <tuple>
#include <algorithm>

using namespace thdm;

std::ifstream infile;
std::string in_fname = "/Users/loganmorrison/Documents/research/thdm_cpb_vev/cpp/rundata/initial_data.csv";

std::ofstream outfile;
std::string out_fname = "/Users/loganmorrison/Documents/research/thdm_cpb_vev/cpp/rundata/refined_data.csv";

typedef std::tuple<Parameters<double>, Vacuum<double>, Vacuum<double>> Point;
typedef std::tuple<Parameters<double>, Vacuum<double>, Vacuum<double>, DoubleExtremaType> GoodPoint;

/**
 * Check that all the derivatives at the normal and charge
 * breaking vacuums are zero.
 * @param point Point to check.
 * @return bool
 */
bool verify_derivatives_zero(Point point) {
    bool isvalid = true;

    Fields<double> fields{};
    auto params = std::get<0>(point);
    auto nvac = std::get<1>(point);
    auto cbvac = std::get<2>(point);
    try {
        fields.set_fields(nvac);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 1)) < 1e-8);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 2)) < 1e-8);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 3)) < 1e-8);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 4)) < 1e-8);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 5)) < 1e-8);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 6)) < 1e-8);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 7)) < 1e-8);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 8)) < 1e-8);
        fields.set_fields(cbvac);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 1)) < 1e-8);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 2)) < 1e-8);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 3)) < 1e-8);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 4)) < 1e-8);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 5)) < 1e-8);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 6)) < 1e-8);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 7)) < 1e-8);
        isvalid = isvalid && (abs(potential_eff_deriv(fields, params, 8)) < 1e-8);
    } catch (...) {
        isvalid = false;
    }
    if (isvalid) {
        // std::cout << "point is good." << std::endl;
    }
    return isvalid;
}

bool verify_goldstones(Point point) {
    bool hasgoldstones = true;

    Fields<double> fields{};
    auto params = std::get<0>(point);
    auto nvac = std::get<1>(point);
    auto cbvac = std::get<2>(point);
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
        hasgoldstones = hasgoldstones && (nmasses[0] < 1e-7);
        hasgoldstones = hasgoldstones && (nmasses[1] < 1e-7);
        hasgoldstones = hasgoldstones && (nmasses[2] < 1e-7);
        // std::cout << nmasses[0] << " " << nmasses[1] << " " << nmasses[2] << std::endl;
        // std::cout << nmasses[3] << " " << nmasses[4] << " " << nmasses[5] << std::endl;
        // std::cout << nmasses[6] << " " << nmasses[7] << std::endl << std::endl;

        fields.set_fields(cbvac);
        auto cbmasses = potential_eff_hessian_evals(fields, params);
        for (double &mass : cbmasses)
            mass = abs(mass);
        std::sort(cbmasses.begin(), cbmasses.end(), std::less<double>());
        // Should have 4 goldstones.
        hasgoldstones = hasgoldstones && (cbmasses[0] < 1e-7);
        hasgoldstones = hasgoldstones && (cbmasses[1] < 1e-7);
        hasgoldstones = hasgoldstones && (cbmasses[2] < 1e-7);
        hasgoldstones = hasgoldstones && (cbmasses[3] < 1e-7);
        // std::cout << cbmasses[0] << " " << cbmasses[1] << " " << cbmasses[2] << std::endl;
        // std::cout << cbmasses[3] << " " << cbmasses[4] << " " << cbmasses[5] << std::endl;
        // std::cout << cbmasses[6] << " " << cbmasses[7] << std::endl << std::endl;
    } catch (...) {
        //std::cout << "Failed at goldstones" << std::endl;
        hasgoldstones = false;
    }

    return hasgoldstones;
}

SingleExtremaType determine_single_extrema_type(Vacuum<double> &vac, Parameters<double> &params) {
    Fields<double> fields{};
    fields.set_fields(vac);
    auto masses = potential_eff_hessian_evals(fields, params);

    auto min_masses = *std::min_element(masses.begin(), masses.end());
    auto max_masses = *std::max_element(masses.begin(), masses.end());
    SingleExtremaType type;

    // Saddle : [-100, -1e-8, 1e-8, 1e-8, 100]
    // Minimum : [ -1e-8, 1e-8, 1e-8, 100, 100]
    // Maximum : [-100,-100, -1e-8, 1e-8, 1e-8]
    // undefined : [-1e-8, -1e-8, -1e-8, 1e-8, 1e-8]

    if (min_masses < -1e-7) {
        // Either a saddle or maximum
        // Saddle if one large mass
        if (max_masses > 1e-7)
            type = SingleExtremaType::Saddle;
            // Maximum if all masses negative
        else
            type = SingleExtremaType::Maximum;
    }
        // No large negative masses, either a min or undefined
    else {
        // If it has large mass, its a min
        if (max_masses > 1e-7)
            type = SingleExtremaType::Minimum;
            // All masses are small, undefined
        else
            type = SingleExtremaType::Undefined;
    }

    return type;

}

DoubleExtremaType determine_extrema_type(Point point) {
    Fields<double> fields{};
    auto params = std::get<0>(point);
    auto nvac = std::get<1>(point);
    auto cbvac = std::get<2>(point);

    SingleExtremaType ntype = determine_single_extrema_type(nvac, params);
    SingleExtremaType cbtype = determine_single_extrema_type(cbvac, params);
    DoubleExtremaType combined_type;

    if (ntype == SingleExtremaType::Minimum && cbtype == SingleExtremaType::Minimum) {
        combined_type = DoubleExtremaType::MinMin;
    } else if (ntype == SingleExtremaType::Minimum && cbtype == SingleExtremaType::Maximum) {
        combined_type = DoubleExtremaType::MinMax;
    } else if (ntype == SingleExtremaType::Minimum && cbtype == SingleExtremaType::Saddle) {
        combined_type = DoubleExtremaType::MinSad;
    } else if (ntype == SingleExtremaType::Minimum && cbtype == SingleExtremaType::Undefined) {
        combined_type = DoubleExtremaType::MinUnd;
    } else if (ntype == SingleExtremaType::Maximum && cbtype == SingleExtremaType::Minimum) {
        combined_type = DoubleExtremaType::MaxMin;
    } else if (ntype == SingleExtremaType::Maximum && cbtype == SingleExtremaType::Maximum) {
        combined_type = DoubleExtremaType::MaxMax;
    } else if (ntype == SingleExtremaType::Maximum && cbtype == SingleExtremaType::Saddle) {
        combined_type = DoubleExtremaType::MaxSad;
    } else if (ntype == SingleExtremaType::Maximum && cbtype == SingleExtremaType::Undefined) {
        combined_type = DoubleExtremaType::MaxUnd;
    } else if (ntype == SingleExtremaType::Saddle && cbtype == SingleExtremaType::Minimum) {
        combined_type = DoubleExtremaType::SadMin;
    } else if (ntype == SingleExtremaType::Saddle && cbtype == SingleExtremaType::Maximum) {
        combined_type = DoubleExtremaType::SadMax;
    } else if (ntype == SingleExtremaType::Saddle && cbtype == SingleExtremaType::Saddle) {
        combined_type = DoubleExtremaType::SadSad;
    } else if (ntype == SingleExtremaType::Saddle && cbtype == SingleExtremaType::Undefined) {
        combined_type = DoubleExtremaType::SadUnd;
    } else if (ntype == SingleExtremaType::Undefined && cbtype == SingleExtremaType::Minimum) {
        combined_type = DoubleExtremaType::UndMin;
    } else if (ntype == SingleExtremaType::Undefined && cbtype == SingleExtremaType::Maximum) {
        combined_type = DoubleExtremaType::UndMax;
    } else if (ntype == SingleExtremaType::Undefined && cbtype == SingleExtremaType::Saddle) {
        combined_type = DoubleExtremaType::UndSad;
    } else {
        combined_type = DoubleExtremaType::UndUnd;
    }
    return combined_type;
}

std::string type_to_string(DoubleExtremaType type) {
    if (type == DoubleExtremaType::MinMin) {
        return "MinMin";
    } else if (type == DoubleExtremaType::MinMax) {
        return "MinMax";
    } else if (type == DoubleExtremaType::MinSad) {
        return "MinSad";
    } else if (type == DoubleExtremaType::MinUnd) {
        return "MinUnd";
    } else if (type == DoubleExtremaType::MaxMin) {
        return "MaxMin";
    } else if (type == DoubleExtremaType::MaxMax) {
        return "MaxMax";
    } else if (type == DoubleExtremaType::MaxSad) {
        return "MaxSad";
    } else if (type == DoubleExtremaType::MaxUnd) {
        return "MaxUnd";
    } else if (type == DoubleExtremaType::SadMin) {
        return "SadMin";
    } else if (type == DoubleExtremaType::SadMax) {
        return "SadMax";
    } else if (type == DoubleExtremaType::SadSad) {
        return "SadSad";
    } else if (type == DoubleExtremaType::SadUnd) {
        return "SadUnd";
    } else if (type == DoubleExtremaType::UndMin) {
        return "UndMin";
    } else if (type == DoubleExtremaType::UndMax) {
        return "UndMax";
    } else if (type == DoubleExtremaType::UndSad) {
        return "UndSad";
    } else {
        return "UndUnd";
    }
}

std::vector<Point> get_full_data() {
    infile.open(in_fname);
    std::vector<Point> data;

    std::string value;
    while (infile.good()) {
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
        params.mu = 246.0;
        // Read in normal vacuum
        Vacuum<double> nvac{};
        getline(infile, value, ',');
        nvac.vevs[0] = std::stod(value);
        getline(infile, value, ',');
        nvac.vevs[1] = std::stod(value);
        // Read in charge-breaking vacuum
        Vacuum<double> cbvac{};
        getline(infile, value, ',');
        cbvac.vevs[0] = std::stod(value);
        getline(infile, value, ',');
        cbvac.vevs[1] = std::stod(value);
        getline(infile, value);
        cbvac.vevs[2] = std::stod(value);

        data.emplace_back(params, nvac, cbvac);
    }

    infile.close();

    return data;
}

std::vector<GoodPoint> extract_good_data() {
    std::vector<GoodPoint> data;
    auto full_data = get_full_data();

    for (auto data_point :full_data) {
        // Verify derivatives
        if (verify_derivatives_zero(data_point)) {
            // Verify Goldstones
            if (verify_goldstones(data_point)) {
                // Verify tree masses
                auto params = std::get<0>(data_point);
                auto nvac = std::get<1>(data_point);
                auto cbvac = std::get<2>(data_point);
                if (are_sqrd_masses_positive_semi_definite(nvac, cbvac, params)) {
                    auto type = determine_extrema_type(data_point);
                    data.emplace_back(params, nvac, cbvac, type);
                }
            }
        }
    }

    return data;

}

void write_good_data(std::vector<GoodPoint> points) {
    outfile.open(out_fname);
    outfile << std::setprecision(15);

    for (auto point : points) {
        auto params = std::get<0>(point);
        auto nvac = std::get<1>(point);
        auto cbvac = std::get<2>(point);
        auto type = std::get<3>(point);
        outfile << params.m112 << ",";
        outfile << params.m122 << ",";
        outfile << params.m222 << ",";
        outfile << params.lam1 << ",";
        outfile << params.lam2 << ",";
        outfile << params.lam3 << ",";
        outfile << params.lam4 << ",";
        outfile << params.lam5 << ",";
        outfile << nvac.vevs[0] << ",";
        outfile << nvac.vevs[1] << ",";
        outfile << cbvac.vevs[0] << ",";
        outfile << cbvac.vevs[1] << ",";
        outfile << cbvac.vevs[2] << ",";
        outfile << type_to_string(type) << "\n";
    }
    outfile.close();
}

int main() {
    auto good_data = extract_good_data();

    write_good_data(good_data);
}