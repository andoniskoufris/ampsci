#include "DiracODE/AsymptoticSpinor.hpp"
#include "Angular/include.hpp"
#include "DiracODE/include.hpp"
#include "DiracOperator/include.hpp"
#include "Maths/Grid.hpp"
#include "Maths/NumCalc_quadIntegrate.hpp"
#include "Physics/AtomData.hpp"
#include "Physics/DiracContinuum.hpp"
#include "Physics/DiracHydrogen.hpp"
#include "Physics/PhysConst_constants.hpp"
#include "Potentials/NuclearPotentials.hpp"
#include "Wavefunction/DiracSpinor.hpp"
#include "catch2/catch.hpp"
#include "fmt/format.hpp"
#include <algorithm>
#include <array>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

//==============================================================================
//! Unit tests for solving (local) Dirac equation ODE
TEST_CASE("DiracODE: AsymptoticSpinor expansion", "[DiracODE][asymp][unit]") {

  std::cout << "AsymptoticSpinor expansion (large r)\n";

  fmt::print("{:<3s} {:>2s} {:1s} {:1s} {:>5s}  {:16s}  {:17s}   {}\n", "z",
             "k", "l", "n", "r", "f/g (asym)", "f/g (exact)", "eps");

  for (auto z : {0.1, 0.5, 1.0, 10.0, 100.0, 150.0}) {
    for (auto kappa : {-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6}) {
      const auto l = Angular::l_k(kappa);
      double w_eps = -1.0;
      int w_n{0};
      double w_asym{0.0}, w_exact{0.0}, w_r{0.0};

      if (z * PhysConst::alpha > 1.0 && std::abs(kappa) == 1)
        continue;

      const auto r0{10.0 / z};
      const auto rmax{150.0 / z};
      const auto num_grid_points{15ul};
      const auto b{(rmax + r0) / 4};
      const auto grid = std::make_shared<const Grid>(r0, rmax, num_grid_points,
                                                     GridType::loglinear, b);
      for (auto n : {1, 2, 3, 4, 5, 6, 7}) {

        if (l >= n)
          continue;

        const auto e = AtomData::diracen(z, n, kappa, PhysConst::alpha);
        const auto F1s = DiracSpinor::exactHlike(n, kappa, grid, z);
        REQUIRE(F1s.en() == Approx(e));
        DiracODE::AsymptoticSpinor x{kappa, z, e};
        for (std::size_t i = 0; i < num_grid_points; ++i) {
          const auto r = grid->r(i);
          const auto [f, g] = x.fg(r);
          const auto f0 = F1s.f(i);
          const auto g0 = F1s.g(i);
          if (g == 0.0 || f0 == 0.0 || g0 == 0.0) {
            continue;
          }

          const auto ratio_asym = f / g;
          const auto ratio_exact = f0 / g0;
          const auto eps = std::abs(ratio_asym / ratio_exact - 1.0);

          if (eps > w_eps) {
            w_eps = eps;
            w_n = n;
            w_asym = ratio_asym;
            w_exact = ratio_exact;
            w_r = r;
          }

          auto eps_targ = z > 99.0 ? 1.0e-10 :
                          z > 9.0  ? 1.0e-8 :
                          z > 0.9  ? 1.0e-7 :
                                     1.0e-5;

          if (l >= 5)
            eps_targ *= 100;
          if (l >= 5 && z < 1.0)
            eps_targ *= 100;

          REQUIRE(eps < eps_targ);
        }
      }

      // if (l >= 5)
      fmt::print("{:3g} {:2} {:1} {:1} {:5.1f} {:+.10e} [{:+.10e}]  {:.1e}\n",
                 z, kappa, l, w_n, w_r, w_asym, w_exact, w_eps);
    }
  }
}
