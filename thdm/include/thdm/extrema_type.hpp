//
// Created by Logan Morrison on 2019-05-02.
//

#ifndef THDM_EXTREMA_TYPE_HPP
#define THDM_EXTREMA_TYPE_HPP

#include "thdm/vacuua.hpp"
#include "thdm/parameters.hpp"
#include "thdm/fields.hpp"
#include "thdm/potentials.hpp"
#include <algorithm>
#include <vector>

namespace thdm {

SingleExtremaType determine_single_extrema_type_tree(Parameters<double> &params, Vacuum<double> &vac) {
    Fields<double> fields{};
    fields.set_fields(vac);
    auto masses = scalar_squared_masses(fields, params);

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

SingleExtremaType determine_single_extrema_type_eff(Parameters<double> &params, Vacuum<double> &vac) {
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

DoubleExtremaType determine_extrema_type_eff(Parameters<double> &params,
                                             Vacuum<double> &nvac,
                                             Vacuum<double> &cbvac) {
    Fields<double> fields{};

    SingleExtremaType ntype = determine_single_extrema_type_eff(params, nvac);
    SingleExtremaType cbtype = determine_single_extrema_type_eff(params, cbvac);
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


}

#endif //THDM_EXTREMA_TYPE_HPP
