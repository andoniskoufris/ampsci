#include "GreenQED.testdata.hpp"
#include "Maths/Hypergeometric.hpp"
#include "catch2/catch.hpp"
#include "fmt/format.hpp"
#include "qip/Vector.hpp"
#include <iostream>
#include <vector>

//==============================================================================
TEST_CASE("Whittaker function M_{1/2} * W_{1/2} test", "[Whittaker]") {

  fmt::print(" k  Re(x)    Im(x)   Re(M*W)   Im(M*W)   Re(exp.)    Im(exp.)   "
             "d(Re)   d(Im) \n");

  for (auto i = 0ul; i < UnitTest::MpWp_DATA.size(); i++) {

    const auto d = UnitTest::MpWp_DATA[i];

    auto code_value =
      Hypergeometric::Whittaker_WM(0.5, d.k, 0.5, d.k, d.x, d.x2_scale * d.x);

    if (i * 110 % 3 == 0) {
      fmt::print("{:2}  {:.1e}  {:.1e}  {:.6e}  {:.6e}  {:.6e}  {:.6e}  {:.1e} "
                 " {:.1e}\n",
                 d.k, d.x.real(), d.x.imag(), code_value.real(),
                 code_value.imag(), d.value.real(), d.value.imag(),
                 (code_value.real() - d.value.real()) / d.value.real(),
                 (code_value.imag() - d.value.imag()) / d.value.imag());
    }

    //! test also for the imaginary parts
    if (std::abs(d.value.real()) < 1.0e-9) {
      REQUIRE(code_value.real() == Approx(d.value.real()).epsilon(1e-6));
      REQUIRE(code_value.real() == Approx(d.value.real()).margin(1e-12));
    } else if (std::abs(d.value.real()) < 1.0e30) {
      REQUIRE(code_value.real() == Approx(d.value.real()).epsilon(1e-6));
      REQUIRE(code_value.real() == Approx(d.value.real()).margin(1e26));
    } else {
      REQUIRE(code_value.real() == Approx(d.value.real()).epsilon(1e-12));
    }
  }
}
