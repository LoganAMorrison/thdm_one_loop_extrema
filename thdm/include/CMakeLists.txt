cmake_minimum_required(VERSION 2.8)
add_subdirectory(thdm)

set(THDM_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/thdm/include/thdm)

set(LIB_HEADERS
        ${THDM_INCLUDE_DIR}/dual.hpp
        ${THDM_INCLUDE_DIR}/fields.hpp
        ${THDM_INCLUDE_DIR}/parameters.hpp
        ${THDM_INCLUDE_DIR}/vacuua.hpp
        ${THDM_INCLUDE_DIR}/potentials.hpp
        ${THDM_INCLUDE_DIR}/jacobi.hpp
        ${THDM_INCLUDE_DIR}/root_finding_eff.hpp
        ${THDM_INCLUDE_DIR}/tree_roots.hpp
        ${THDM_INCLUDE_DIR}/minimize.hpp
        ${THDM_INCLUDE_DIR}/eigenvalue_derivatives.hpp
        ${THDM_INCLUDE_DIR}/extrema_type.hpp
        ${THDM_INCLUDE_DIR}/root_refine.hpp
        ${THDM_INCLUDE_DIR}/tree_roots_explict.hpp
        ${THDM_INCLUDE_DIR}/gauge_masses.hpp
        ${THDM_INCLUDE_DIR}/fermion_masses.hpp
        ${THDM_INCLUDE_DIR}/beta_functions.hpp
        )

# Define two variables in order not to repeat ourselves.
set(LIBRARY_NAME thdm)

# Define the library
add_library(${LIBRARY_NAME} INTERFACE)

target_include_directories(${LIBRARY_NAME} INTERFACE include/)

target_link_libraries(${LIBRARY_NAME})
