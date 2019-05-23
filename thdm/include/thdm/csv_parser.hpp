#include <utility>

//
// Created by Logan Morrison on 2019-05-22.
//

#ifndef THDM_CSV_PARSER_HPP
#define THDM_CSV_PARSER_HPP

#include <string>
#include <vector>
#include <fstream>
#include <boost/tokenizer.hpp>

namespace thdm {
typedef boost::tokenizer<boost::char_separator<char>> tokenizer;

class CSVReader {
    std::string _file_name;
public:
    CSVReader(std::string file_name) : _file_name(std::move(file_name)) {}

    ~CSVReader() = default;

    std::vector<std::vector<double>> parse(bool header = false) {
        std::ifstream in_file;
        in_file.open(_file_name);

        std::vector<std::vector<double>> data;
        // Separating character for a CSV file.
        boost::char_separator<char> sep(",");

        bool on_header_line = header;
        while (in_file.good()) {
            std::string line; // Store the an entire line of the CSV file.
            // If there is a header, read and skip it.
            if (on_header_line) {
                getline(in_file, line);
                on_header_line = false;
            }
            // Check if the line is empty
            if (line == "") {
                continue;
            } else {
                // Otherwise, split the line at the ',' and put all values
                // between ',' into a vector.
                std::vector<double> row;
                getline(in_file, line);
                tokenizer tokens(line, sep);
                for (auto iter = tokens.begin(); iter != tokens.end(); iter++) {
                    row.push_back(std::stod(*iter));
                }
                data.push_back(row);
            }
        }
        return data;
    }
};
}

#endif //THDM_CSV_PARSER_HPP
