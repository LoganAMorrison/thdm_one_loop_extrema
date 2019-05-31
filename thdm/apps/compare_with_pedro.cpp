//
// Created by Logan Morrison on 2019-05-30.
//


#include "thdm/potentials.hpp"
#include "thdm/fields.hpp"
#include "thdm/parameters.hpp"

using namespace thdm;

int main() {
    Parameters<double> params{};
    Vacuum<double> nvac{};
    Vacuum<double> cbvac{};
    Fields<double> fields{};
    params.m112 = -46093.5402300992;
    params.m122 = -899.700385043558;
    params.m222 = -44737.1716396116;
    params.lam1 = 1.63487525949677;
    params.lam2 = 1.48074031509965;
    params.lam3 = 1.58032039140444;
    params.lam4 = 0.100046482470595;
    params.lam5 = 0.0140615017356978;
    params.yt = 1.02362273008221;
    params.mu = 246;
    nvac.vevs[0] = -58.21405413164;
    nvac.vevs[1] = 239.012811166177;
    nvac.vevs[2] = -5.2697087226065e-14;
    cbvac.vevs[0] = -181.801566807722;
    cbvac.vevs[1] = 90.0784953660183;
    cbvac.vevs[2] = 130.449202517779;

    fields.set_fields(nvac);
    std::cout << "dv0dr1 = " << potential_tree_deriv(fields, params, 1) << "\n";
    std::cout << "dv0dr2 = " << potential_tree_deriv(fields, params, 2) << "\n";

    std::cout << "dv1dr1 = " << potential_one_loop_deriv(fields, params, 1) << "\n";
    std::cout << "dv1dr2 = " << potential_one_loop_deriv(fields, params, 2) << "\n";


    return 0;
}