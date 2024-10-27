enable_testing()
find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(comp-phy-tests
    ode_solvers_tests.cpp
)

target_link_libraries(comp-phy-tests PRIVATE comp-phy GTest::gmock_main)

set_target_properties(comp-phy-tests PROPERTIES CXX_STANDARD 20)

gtest_discover_tests(comp-phy-tests)
