//
// Created by Logan Morrison on 2019-04-26.
//

#ifndef THDM_TREE_ROOT_HPP
#define THDM_TREE_ROOT_HPP

#include "thdm/vacuua.hpp"
#include "thdm/fields.hpp"
#include "thdm/parameters.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <mutex>

namespace thdm {

std::mutex tree_roots_mtx;

typedef std::vector<Vacuum < double>>
VacVec;

template<typename T>
std::string to_string_with_precision(const T a_value, const int n = 15) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

static void write_equations(Parameters<double> &params) {
    static std::string PATH_HOM4PS2 =
            "/Users/loganmorrison/HEP-Tools/HOM4PS2_Mac";
    std::ofstream data_file;
    data_file.open(PATH_HOM4PS2 + "/current_tadpoles.in");

    /* Create strings for tadpole equations to write to file. */
    std::string tadpole_a =
            ("0.5*a^3*" + to_string_with_precision(params.lam1) + "+0.5*a*c^2*" +
                    to_string_with_precision(params.lam1) + "+0.5*a*b^2*" +
                    to_string_with_precision(params.lam3) + "+0.5*a*b^2*" +
                    to_string_with_precision(params.lam4) + "+0.5*a*b^2*" +
                    to_string_with_precision(params.lam5) + "+1.*a*" + std::to_string(params.m112) +
                    "-1.*b*" + to_string_with_precision(params.m122));

    std::string tadpole_b =
            ("0.5*b^3*" + to_string_with_precision(params.lam2) + "+0.5*a^2*b*" +
                    to_string_with_precision(params.lam3) + "+0.5*b*c^2*" +
                    to_string_with_precision(params.lam3) + "+0.5*a^2*b*" +
                    to_string_with_precision(params.lam4) + "+0.5*a^2*b*" +
                    to_string_with_precision(params.lam5) + "-1.*a*" +
                    to_string_with_precision(params.m122) +
                    "+1.*b*" + to_string_with_precision(params.m222));

    std::string tadpole_c =
            ("0.5*a^2*c*" + to_string_with_precision(params.lam1) + "+0.5*c^3*" +
                    to_string_with_precision(params.lam1) + "+0.5*b^2*c*" +
                    to_string_with_precision(params.lam3) + "+1.*c*" +
                    to_string_with_precision(params.m112));

    /* Output these equations to HOME4PS2 file */
    data_file << "{"
              << "\n"
              << tadpole_a << ";"
              << "\n"
              << tadpole_b << ";"
              << "\n"
              << tadpole_c << "\n"
              << ";}";

    data_file.close();
}

static std::vector<std::string> read_data_file() {
    static std::string PATH_HOM4PS2 =
            "/Users/loganmorrison/HEP-Tools/HOM4PS2_Mac";
    std::vector<std::string> data_file_text;

    /* Open file and read in roots */
    std::ifstream root_file;
    root_file.open(PATH_HOM4PS2 + "/data.roots");

    std::string line;

    if (root_file.is_open()) {
        while (getline(root_file, line)) {
            data_file_text.push_back(line);
        }
        root_file.close();
    }

    return data_file_text;
}

static VacVec parse_data_file() {
    // declare variables to use
    double real1, real2, real3;
    double imag1, imag2, imag3;
    int _pos1, _pos2, n1, n2, n3;
    // Get the txt from private member variable
    std::vector<std::string> lines = read_data_file();

    /* 10 lines from the bottom of the HOM4PS2 output file is a line
     * stating the number of roots found. This line looks like:
     *      "The # of roots       =        $num_roots"
     * To determine the number of roots, we need to get this line, find the
     * positions of the "=", remove a characters of the line before the "=",
     * the trim off the whitespace before and after the number representing
     * the number of roots.
     */

    size_t nlines = lines.size();
    // Get line with the root in it.
    std::string str1 =
            lines[nlines - 10].substr(0, lines[nlines - 10].find('\n'));
    // Get ordering of variables
    std::string str_var1 =
            lines[nlines - 15].substr(1, lines[nlines - 15].find('\n'));
    std::string str_var2 =
            lines[nlines - 14].substr(1, lines[nlines - 14].find('\n'));
    std::string str_var3 =
            lines[nlines - 13].substr(1, lines[nlines - 13].find('\n'));
    // Find the positions of the "=" in the line
    int pos = str1.find('=');
    // Removes all text in front of "="
    str1.erase(0, pos + 1);
    // Removes all spaces from the beginning of the string
    while (!str1.empty() && isspace(str1.front()))
        str1.erase(str1.begin());
    // Remove all spaces from the end of the string.
    while (!str1.empty() && isspace(str1.back()))
        str1.pop_back();

    auto num_roots = stoi(str1);

    /* Create array to store roots */
    std::vector<Vacuum<double>> roots;

    for (int i = 0; i < num_roots; i++) {
        /* The HOM4PS2 output file is formatted as follows:
         *      ( root11 , err_root11)
         *      ( root12 , err_root12)
         *      ( root13 , err_root13)
         *
         *      residue = #
         *      condition number = #
         *      ----------------------
         *      ( root21 , err_root21)
         *      .
         *      .
         *      .
         *  Thus, there are three lines for the roots starting from the
         *  top of the file. Then, 7 lines from the first root is the start of
         *  the next three roots. This, we want to parse i * 7 + 0, i * 7 + 1,
         *  and i * 7 + 2 for i in range 0 -> number roots -1.
         */
        // Check the variable order. Make sure n1 is associate with a,
        // n2 is associated with b and n3 is associated with c.
        if (str_var1 == "a" && str_var2 == "b" && str_var3 == "c") {
            n1 = 7 * i + 0;
            n2 = 7 * i + 1;
            n3 = 7 * i + 2;
        } else if (str_var1 == "a" && str_var2 == "c" && str_var3 == "b") {
            n1 = 7 * i + 0;
            n3 = 7 * i + 1;
            n2 = 7 * i + 2;
        } else if (str_var1 == "b" && str_var2 == "a" && str_var3 == "c") {
            n2 = 7 * i + 0;
            n1 = 7 * i + 1;
            n3 = 7 * i + 2;
        } else if (str_var1 == "c" && str_var2 == "b" && str_var3 == "a") {
            n3 = 7 * i + 0;
            n2 = 7 * i + 1;
            n1 = 7 * i + 2;
        } else if (str_var1 == "b" && str_var2 == "c" && str_var3 == "a") {
            n2 = 7 * i + 0;
            n3 = 7 * i + 1;
            n1 = 7 * i + 2;
        } else {
            n3 = 7 * i + 0;
            n1 = 7 * i + 1;
            n2 = 7 * i + 2;
        }

        /* Roots in the HOM4PS2 output file are of the form:
         *  ( root , error )
         * Thus, to parse the root, we need to find position of the "( ",
         * then remove all characters before that, then find the position of
         * " ," and store all characters before that, which will be a number
         * representing the root. We will need to do this three times to obtain
         * the values of v1, v2 and v3.
         */

        _pos1 = lines[n1].find("( ");  // Find position of the "( "
        lines[n1].erase(0, _pos1 + 2); // Erase chars before "( "
        _pos2 = lines[n1].find(')');   // Find position of the ")"
        lines[n1].erase(_pos2, 1);     // Erase chars before "( "

        /* Find position of the " ," and store everything before, which is
         * the root that we are looking for
         */

        std::stringstream(lines[n1].substr(0, lines[n1].find(" ,"))) >> real1;

        /*
         * Get the imaginary part of the number.
         */
        std::stringstream(lines[n1].substr(lines[n1].find(" ,") + 2, _pos2 - 1)) >> imag1;

        /* Repeat what we did above with the roots for v2 and v3*/
        _pos1 = lines[n2].find("( ");
        lines[n2].erase(0, _pos1 + 2);
        _pos2 = lines[n2].find(')');
        lines[n2].erase(_pos2, 1);
        std::stringstream(lines[n2].substr(0, lines[n2].find(" ,"))) >> real2;
        std::stringstream(lines[n2].substr(lines[n2].find(" ,") + 2, _pos2 - 1)) >> imag2;

        _pos1 = lines[n3].find("( ");
        lines[n3].erase(0, _pos1 + 2);
        _pos2 = lines[n3].find(')');
        lines[n3].erase(_pos2, 1);
        std::stringstream(lines[n3].substr(0, lines[n3].find(" ,"))) >> real3;
        std::stringstream(lines[n3].substr(lines[n3].find(" ,") + 2, _pos2 - 1)) >> imag3;

        // If all the imaginary parts are zero, add the point.
        if (std::abs(imag1) < 1e-12 && std::abs(imag2) < 1e-12 &&
                std::abs(imag3) < 1e-12) {
            roots.push_back(Vacuum<double>{std::vector<double>{real1, real2, real3}});
        }
    }

    return roots;
}

VacVec get_tree_roots(Parameters<double> &params) {
    // Lock the function. Only one thread can access this at a time.
    std::lock_guard<std::mutex> lock(tree_roots_mtx);
    // First write the equations to current_tadpoles.in in the HOM4PS2
    // directory
    write_equations(params);

    // Run HOM4PS2 from command line.
    std::system("cd /Users/loganmorrison/HEP-Tools/HOM4PS2_Mac && printf '1' "
                "| ./hom4ps2 current_tadpoles.in > nul");

    // Parse the text and read in roots
    return parse_data_file();
}

} // namespace thdm

#endif // THDM_TREE_ROOT_HPP
