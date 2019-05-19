#ifndef FIELDS_HPP
#define FIELDS_HPP

#include "thdm/vacuua.hpp"
#include <vector>
#include <exception>

namespace thdm {

struct InvalidVectorSizeConstructor : public std::exception {
    const char *what() const noexcept override {
        return "Invalid vector size to initialize Fields.";
    }
};

struct InvalidVectorSizeSet : public std::exception {
    const char *what() const noexcept override {
        return "Invalid vector size to set Fields.";
    }
};

struct FieldIndexOutOfRange : public std::exception {
    const char *what() const noexcept override {
        return "Invalid index in call Fields[i].";
    }
};

/**
 * Fields class for the THDM.
 *
 * The fields of the THDM are defined such that, the first doublet is:
 * H1 = Transpose[{c1 + I c2, r1 + I i1 }]/sqrt(2) and the second
 * doublet is H2 = Transpose[{c3 + I c4, r2 + I i2 }]/sqrt(2)
 *
 * @tparam T double, Dual<double> or Dual<Dual<double>>
 */
template<class T>
class Fields {
private:
    /* data */
public:
    /**
     * Fields or the THDM.
     */
    T r1, r2, c1, c2, c3, c4, i1, i2;

    /**
     * Default constructor for Fields<T>. All fields are set to zero.
     */
    Fields() : r1(static_cast<T>(0)), r2(static_cast<T>(0)), c1(static_cast<T>(0)),
               c2(static_cast<T>(0)), c3(static_cast<T>(0)), c4(static_cast<T>(0)),
               i1(static_cast<T>(0)), i2(static_cast<T>(0)) {}

    /**
     * Full constructor for Fields<T>
     * @param r1 real component of bottom of first higgs doublet.
     * @param r2 real component of bottom of second higgs doublet.
     * @param c1 real component of top of first higgs doublet.
     * @param c2 imaginary component of top of first higgs doublet.
     * @param c3 real component of top of second higgs doublet.
     * @param c4 imaginary component of top of second higgs doublet.
     * @param i1 imaginary component of bottom of first higgs doublet.
     * @param i2 imaginary component of bottom of second higgs doublet.
     */
    Fields(T r1, T r2, T c1, T c2, T c3, T c4, T i1, T i2)
            : r1(r1), r2(r2), c1(c1), c2(c2), c3(c3), c4(c4), i1(i1), i2(i2) {}

    /**
     * Partial constructor for Fields<T>.
     * @param r1 real component of bottom of first higgs doublet.
     * @param r2 real component of bottom of second higgs doublet.
     * @param c1 real component of top of first higgs doublet.
     */
    Fields(T r1, T r2, T c1)
            : r1(r1), r2(r2), c1(c1), c2(static_cast<T>(0)), c3(static_cast<T>(0)),
              c4(static_cast<T>(0)), i1(static_cast<T>(0)), i2(static_cast<T>(0)) {}

    /**
     * Constructor to initialize Fields<T> using a boost vector.
     *
     * If v.size() = 3, we initilize r1, r2, and c1. If v.size() = 8,
     * we initilize all components. Otherwise, we throw an error.
     *
     * @param v boost vector of fields.
     */
    explicit Fields(std::vector<T> &v) {
        if (v.size() == 3) {
            r1 = v[0];
            r2 = v[1];
            c1 = v[2];
        } else if (v.size() == 8) {

            r1 = v[0];
            r2 = v[1];
            c1 = v[2];
            c2 = v[3];
            c3 = v[4];
            c4 = v[5];
            i1 = v[6];
            i2 = v[7];
        } else {
            throw InvalidVectorSizeConstructor();
        }
    }

    /**
     * Output stream for fields.
     * @param os stream
     * @param fields THDM fields
     * @return stream
     */
    friend std::ostream &operator<<(std::ostream &os, const Fields<T> &fields) {
        os << "Fields(" << fields.r1 << ", " << fields.r2 << ", "
           << fields.c1 << ", " << fields.c2 << ", " << fields.c3 << ", "
           << fields.c4 << ", " << fields.i1 << ", " << fields.i2 << ")" << std::endl;
        return os;
    }

    /**
     * Access overload for Fields<T>
     * @param fld index of the field.
     * @return value of field at index `fld`.
     */
    T &operator[](size_t fld) {
        if (fld == 0)
            return r1;
        else if (fld == 1)
            return r2;
        else if (fld == 2)
            return c1;
        else if (fld == 3)
            return c2;
        else if (fld == 4)
            return c3;
        else if (fld == 5)
            return c4;
        else if (fld == 6)
            return i1;
        else if (fld == 7)
            return i2;
        else
            throw FieldIndexOutOfRange();
    }

    /**
     * Defualt destructor for Fields<T>.
     */
    ~Fields() = default;

    /**
     * Set all fields to zero.
     */
    void null_fields() {
        r1 = static_cast<T>(0);
        r2 = static_cast<T>(0);
        c1 = static_cast<T>(0);
        c2 = static_cast<T>(0);
        c3 = static_cast<T>(0);
        c4 = static_cast<T>(0);
        i1 = static_cast<T>(0);
        i2 = static_cast<T>(0);
    }

    /**
     * Set the value of the fields using a boost vector.
     *
     * If v.size() = 3, we set r1, r2, and c1. If v.size() = 8,
     * we set all components. Otherwise, we throw an error.
     *
     * @param v
     */
    void set_fields(std::vector<T> &v) {
        if (v.size() == 3) {
            r1 = v[0];
            r2 = v[1];
            c1 = v[2];
        } else if (v.size() == 8) {
            r1 = v[0];
            r2 = v[1];
            c1 = v[2];
            c2 = v[3];
            c3 = v[4];
            c4 = v[5];
            i1 = v[6];
            i2 = v[7];
        } else {
            throw InvalidVectorSizeSet();
        }
    }

    void set_fields(const Vacuum<T> &vac) {
        null_fields();
        r1 = vac.vevs[0];
        r2 = vac.vevs[1];
        c1 = vac.vevs[2];
    }
};

} // namespace thdm

#endif //FIELDS_HPP