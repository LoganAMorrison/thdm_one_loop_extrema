#include <utility>

//
// Created by Logan Morrison on 2019-05-20.
//

#ifndef THDM_ERRORS_HPP
#define THDM_ERRORS_HPP

#include <exception>
#include <string>

namespace thdm {
enum THDMExceptionCode {
    DualDivisionByZero,
    DualInvalidLogArgument,
    DualInvalidSqrtArgument,
    FieldsInvalidVectorSizeConstructor,
    FieldsInvalidVectorSizeSet,
    FieldsIndexOutOfRange,
    ParametersIndexOutOfRange,
    JacobiTooManyIterations,
    ScalarMassesInvalidFldIndex,
    ScalarMassesInvalidParIndex
};

std::string thdm_exception_code_to_string(const THDMExceptionCode &code) {
    if (code == THDMExceptionCode::DualDivisionByZero) {
        return "Divide by zero in dual function";
    } else if (code == THDMExceptionCode::DualInvalidLogArgument) {
        return "Invalid argument passed to dual log";
    } else if (code == THDMExceptionCode::DualInvalidSqrtArgument) {
        return "Invalid argument passed to dual sqrt";
    } else if (code == THDMExceptionCode::FieldsInvalidVectorSizeConstructor) {
        return "Invalid vector size to initialize Fields object";
    } else if (code == THDMExceptionCode::FieldsInvalidVectorSizeSet) {
        return "Invalid vector size to set Fields object";
    } else if (code == THDMExceptionCode::FieldsIndexOutOfRange) {
        return "Invalid index in call Fields[i]";
    } else if (code == THDMExceptionCode::ParametersIndexOutOfRange) {
        return "Invalid index in call Parameters[i]";
    } else if (code == THDMExceptionCode::JacobiTooManyIterations) {
        return "Too many iterations in jacobi";
    } else if (code == THDMExceptionCode::ScalarMassesInvalidFldIndex) {
        return "Invalid field index in scalar masses derivative";
    } else if (code == THDMExceptionCode::ScalarMassesInvalidParIndex) {
        return "Invalid parameter index in scalar masses derivative";
    }
}

class THDMException : public std::exception {
    std::string _msg;
public:
    THDMExceptionCode code;

    explicit THDMException(const THDMExceptionCode &code)
            : _msg(thdm_exception_code_to_string(code)), code(code) {}

    const char *what() const noexcept override {
        return _msg.c_str();
    }
};

}


#endif //THDM_ERRORS_HPP
