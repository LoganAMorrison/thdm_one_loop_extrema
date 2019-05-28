# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
if (result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif ()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
if (result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif ()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
        ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
        EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
endif ()

find_package(Boost REQUIRED)
find_package(GSL REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${Boost_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS})

# Now simply link against gtest or gtest_main as needed. Eg

add_executable(test_dual test_dual.cpp)
target_link_libraries(test_dual
        gtest_main
        ${Boost_LIBRARIES}
        thdm
        ${ARMADILLO_LIBRARIES}
        GSL::gsl
        GSL::gslcblas
        Eigen3::Eigen)
set_target_properties(test_dual PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin
        )

add_executable(test_potential test_potential.cpp)
target_link_libraries(test_potential gtest_main
        thdm
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
        GSL::gsl
        GSL::gslcblas
        Eigen3::Eigen)
set_target_properties(test_potential PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin
        )

add_executable(test_jacobi test_jacobi.cpp)
target_link_libraries(test_jacobi gtest_main
        thdm
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
        GSL::gsl
        GSL::gslcblas
        Eigen3::Eigen)
set_target_properties(test_jacobi PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin
        )

add_executable(test_scalar_masses test_scalar_masses.cpp)
target_link_libraries(test_scalar_masses
        gtest_main
        thdm
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
        GSL::gsl
        GSL::gslcblas
        Eigen3::Eigen)
set_target_properties(test_scalar_masses PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin
        )

add_executable(test_root_finding test_root_finding.cpp)
target_link_libraries(test_root_finding
        gtest_main
        thdm
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
        GSL::gsl
        GSL::gslcblas
        Eigen3::Eigen)
set_target_properties(test_root_finding PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin
        )

add_executable(test_tree_roots test_tree_roots.cpp)
target_link_libraries(test_tree_roots
        gtest_main
        thdm
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
        GSL::gsl
        GSL::gslcblas
        Eigen3::Eigen)
set_target_properties(test_tree_roots PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin
        )

add_executable(test_minimize test_minimize.cpp)
target_link_libraries(test_minimize
        gtest_main
        thdm
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
        GSL::gsl
        GSL::gslcblas
        Eigen3::Eigen)
set_target_properties(test_minimize PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin
        )


add_executable(test_model test_model.cpp)
target_link_libraries(test_model
        gtest_main
        thdm
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
        GSL::gsl
        GSL::gslcblas
        Eigen3::Eigen)
set_target_properties(test_model PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin
        )


add_executable(test_tree_roots_explict test_tree_roots_explict.cpp)
target_link_libraries(test_tree_roots_explict
        gtest_main
        thdm
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
        GSL::gsl
        GSL::gslcblas
        Eigen3::Eigen)
set_target_properties(test_tree_roots_explict PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin
        )

add_executable(test_gauge_masses test_gauge_masses.cpp)
target_link_libraries(test_gauge_masses
        gtest_main
        thdm
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
        GSL::gsl
        GSL::gslcblas
        Eigen3::Eigen)
set_target_properties(test_gauge_masses PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin
        )

add_executable(test_rge test_rge.cpp)
target_link_libraries(test_rge
        gtest_main
        thdm
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
        GSL::gsl
        GSL::gslcblas
        Eigen3::Eigen)
set_target_properties(test_rge PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin
        )

add_executable(test_with_pedro test_with_pedro.cpp)
target_link_libraries(test_with_pedro
        gtest_main
        thdm
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
        GSL::gsl
        GSL::gslcblas
        Eigen3::Eigen)
set_target_properties(test_with_pedro PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}../../bin
        )
