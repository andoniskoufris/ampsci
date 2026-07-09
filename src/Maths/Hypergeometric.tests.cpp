#include "Maths/Hypergeometric.hpp"
#include "catch2/catch.hpp"
#include <complex>

//==============================================================================
TEST_CASE("Maths::Hypergeometric 1F1", "[Hypergeometric][unit]") {

  // bare bones test - OK because tested also elsewhere (contniuum functions)

  // Real version (GSL), vs Mathematica: Hypergeometric1F1[1.0, 2.0, 3.0]
  REQUIRE(Hypergeometric::H1f1(1.0, 2.0, 3.0) ==
          Approx(6.36184564106256).epsilon(1.0e-13));

  if constexpr (Hypergeometric::has_flint) {

    // Complex version with zero imaginary part must match real version:
    for (const double a : {0.5, 1.0, 2.5}) {
      for (const double b : {1.5, 2.0}) {
        for (const double z : {-1.0, 0.5, 3.0}) {
          const auto expected = Hypergeometric::H1f1(a, b, z);
          const auto found = Hypergeometric::H1f1(std::complex<double>{a}, b,
                                                  std::complex<double>{z});
          REQUIRE(found.real() == Approx(expected).epsilon(1.0e-12));
          REQUIRE(found.imag() == Approx(0.0).margin(1.0e-12));
        }
      }
    }

    // vs Mathematica: Hypergeometric1F1[1.0 + I, 2.0, 3.0 + 2.0*I]
    const auto found = Hypergeometric::H1f1(std::complex<double>{1.0, 1.0}, 2.0,
                                            std::complex<double>{3.0, 2.0});
    REQUIRE(found.real() == Approx(-4.10484015745801).epsilon(1.0e-13));
    REQUIRE(found.imag() == Approx(0.42030004004629).epsilon(1.0e-13));
  }
}
