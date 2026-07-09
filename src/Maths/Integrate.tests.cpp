#include "Maths/NumCalc_quadIntegrate.hpp"
#include "catch2/catch.hpp"
#include <iostream>
#include <vector>

TEST_CASE("NumCalc_quadIntegrate", "[num_integrate][unit]") {

  {
    const auto value = NumCalc::num_integrate([](double) { return 1.0; }, 0.0,
                                              10.0, 100, NumCalc::linear);
    REQUIRE(value == 10.0);
  }

  {
    const auto value = NumCalc::num_integrate(
      [](double) { return 1.0; }, 1.0e-6, 10.0, 100, NumCalc::logarithmic);
    REQUIRE(value == Approx(10.0));
  }

  {
    const auto value = NumCalc::num_integrate([](double x) { return x; }, 0.0,
                                              10.0, 100, NumCalc::linear);
    REQUIRE(value == Approx(50.0));
  }

  {
    const auto value = NumCalc::num_integrate(
      [](double x) { return x; }, 1.0e-6, 10.0, 100, NumCalc::logarithmic);
    REQUIRE(value == Approx(50.0));
  }

  {
    const auto value =
      NumCalc::num_integrate([](double x) { return 1.0 / x; }, 1.0,
                             std::exp(3.5), 500, NumCalc::linear);
    REQUIRE(value == Approx(3.5));
  }

  {
    const auto value =
      NumCalc::num_integrate([](double x) { return 1.0 / x; }, 1.0,
                             std::exp(3.5), 500, NumCalc::logarithmic);
    REQUIRE(value == Approx(3.5));
  }
}
