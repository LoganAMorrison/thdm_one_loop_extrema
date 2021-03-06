cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)


# Build the Effective Type A Extrema Finder Application
set(INITIAL_DATA_GENERATOR_APP initial_data_generator)


add_executable(${INITIAL_DATA_GENERATOR_APP} initial_data_generator.cpp)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(${INITIAL_DATA_GENERATOR_APP}
        thdm
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
        ${Boost_THREAD_LIBRARY}
        GSL::gsl
        GSL::gslcblas
        Eigen3::Eigen)

set_target_properties(${INITIAL_DATA_GENERATOR_APP} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin
        )

# Build the Effective Type A Extrema Finder Application
set(POST_PROCESSING post_processing)


add_executable(${POST_PROCESSING} post_processing.cpp)
target_link_libraries(${POST_PROCESSING}
        thdm
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
        GSL::gsl
        GSL::gslcblas
        Eigen3::Eigen)

set_target_properties(${POST_PROCESSING} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin
        )

# Build the Effective Type A Extrema Finder Application
set(FULL_SCAN find_double_mins)


add_executable(${FULL_SCAN} full_scan.cpp)
target_link_libraries(${FULL_SCAN}
        thdm
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
        GSL::gsl
        GSL::gslcblas
        Eigen3::Eigen)

set_target_properties(${FULL_SCAN} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin
        )

set(MONTE_CARLO monte_carlo)

add_executable(${MONTE_CARLO} monte_carlo.cpp)
target_link_libraries(${MONTE_CARLO}
        thdm
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
        GSL::gsl
        GSL::gslcblas
        Eigen3::Eigen)

set_target_properties(${MONTE_CARLO} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin
        )

set(RGE rge)
add_executable(${RGE} rge.cpp)
target_link_libraries(${RGE}
        thdm
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
        GSL::gsl
        GSL::gslcblas
        Eigen3::Eigen)

set_target_properties(${RGE} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin
        )

set(POTENTIALS potentials)
add_executable(${POTENTIALS} potentials.cpp)
target_link_libraries(${POTENTIALS}
        thdm
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
        GSL::gsl
        GSL::gslcblas
        Eigen3::Eigen)

set_target_properties(${POTENTIALS} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin
        )

set(COMPARE_WITH_PEDRO compare_with_pedro)
add_executable(${COMPARE_WITH_PEDRO} compare_with_pedro.cpp)
target_link_libraries(${COMPARE_WITH_PEDRO}
        thdm
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
        GSL::gsl
        GSL::gslcblas
        Eigen3::Eigen)

set_target_properties(${COMPARE_WITH_PEDRO} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin
        )

set(INFO_OUT info_output)
add_executable(${INFO_OUT} info_output.cpp)
target_link_libraries(${INFO_OUT}
        thdm
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
        GSL::gsl
        GSL::gslcblas
        Eigen3::Eigen)

set_target_properties(${INFO_OUT} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin
        )