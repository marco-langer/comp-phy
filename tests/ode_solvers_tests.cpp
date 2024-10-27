#include "ml/ode_solvers.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>


struct Params
{
    double h;
    double expectedValue;
};


class EulerFixture : public testing::TestWithParam<Params>
{};

/**
 * dy(t)/dt = f(t, y(t)) = y
 *
 * analytic solution: y(t) = e^t; with y(4) ~= 54.598
 */
TEST_P(EulerFixture, Euler1)
{
    const auto& [h, expectedValue] = GetParam();

    constexpr double y_n = 4.0;
    const std::size_t n = static_cast<std::size_t>(y_n / h);

    const std::vector<double> data =
        ml::ode::euler(n, h, 0.0, 1.0, [](double /* t */, double y_n) { return y_n; });

    constexpr double eps = 10.0e-3;
    EXPECT_THAT(data.back(), testing::DoubleNear(expectedValue, eps));
}


INSTANTIATE_TEST_SUITE_P(
    EulerTest,
    EulerFixture,
    testing::Values(
        Params{ .h = 1.00, .expectedValue = 16.00 },
        Params{ .h = 0.25, .expectedValue = 35.53 },
        Params{ .h = 0.1, .expectedValue = 45.26 },
        Params{ .h = 0.05, .expectedValue = 49.56 }));
