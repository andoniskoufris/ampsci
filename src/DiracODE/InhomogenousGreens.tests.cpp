#include "Angular/include.hpp"
#include "DiracODE/AsymptoticSpinor.hpp"
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
// Test inhomogenous (Green's) method:
TEST_CASE("DiracODE: inhomogenous (Green's) method",
          "[DiracODE][inhomog][unit]") {

  const double Zeff = 5.0;

  // Set up radial grid:
  const auto r0{1.0e-7};
  const auto rmax{100.0}; // NB: rmax depends on Zeff
  const auto num_grid_points{2000ul};
  const auto b{10.0};
  const auto grid = std::make_shared<const Grid>(r0, rmax, num_grid_points,
                                                 GridType::loglinear, b);
  // Sperical potential w/ R_nuc = 0.0 is a pointlike potential
  const auto v_nuc = Nuclear::sphericalNuclearPotential(Zeff, 0.0, grid->r());

  {
    // Solve: (Fa and Fb should be equal)
    // (H + v + vp - e)Fa = 0
    // (H + v - e)Fb = -vp*Fa
    std::cout << "Test inhomogenous (Green's) method:   \n"
                 "(H + v + vp - e)Fa = 0     vs.\n"
                 "(H + v - e)Fb = -vp*Fa    (Fa and Fb should be equal)\n";
    auto max_eps_dF = -1.0;
    auto max_eps_orthNorm = -1.0;
    const auto states_new = AtomData::listOfStates_nk("5spdf");

    // This will act as a "non-local" potential
    std::vector<double> vp;
    for (const auto r : grid->r()) {
      vp.push_back(-0.3 / (r * r * r * r + 1.0));
    }
    const auto v_tot = qip::add(v_nuc, vp);

    for (const auto &[n, k, en] : states_new) {

      auto Fa = DiracSpinor(n, k, grid);
      auto Fap1 = DiracSpinor(n + 1, k, grid); // for orthog
      auto Fb = DiracSpinor(n, k, grid);
      const auto en_guess = -(Zeff * Zeff) / (2.0 * n * n);
      const auto en_guess_p1 = -(Zeff * Zeff) / (2.0 * (n + 1) * (n + 1));

      DiracODE::boundState(Fa, en_guess, v_tot, {}, PhysConst::alpha, 1.0e-15);
      DiracODE::boundState(Fap1, en_guess_p1, v_tot, {}, PhysConst::alpha,
                           1.0e-15);

      const auto dvFa = vp * Fa; // "non-local"
      DiracODE::solve_inhomog(Fb, Fa.en(), v_nuc, {}, PhysConst::alpha,
                              -1 * dvFa);

      const auto eps_norm = std::abs(Fb * Fb - 1.0); //<b|b> - norm

      Fb.normalise(); // don't propogate norm error:

      const auto eps_orth = std::abs(Fap1 * Fb);  //<a+1|b> - orthogonality
      const auto eps_1 = std::abs(Fa * Fb - 1.0); // <a|b> - check values
      const auto eps_2 = (Fa - Fb) * (Fa - Fb);   // <a-b>^2 - check values

      const auto eps_dF = std::max(eps_1, eps_2);
      const auto eps_orthNorm = std::max(eps_norm, eps_orth);
      printf("%4s <b|b>-1 = %.0e, <a|b>-1 = %.0e, <a-b>^2 = %.0e, <a+1|b> = "
             "%.0e\n",
             Fa.shortSymbol().c_str(), eps_norm, eps_1, eps_2, eps_orth);

      if (!(eps_dF < max_eps_dF))
        max_eps_dF = eps_dF;
      if (!(eps_orthNorm < max_eps_orthNorm))
        max_eps_orthNorm = eps_orthNorm;
    }
    //"Inhomog (G): orthonorm"
    REQUIRE(std::abs(max_eps_orthNorm) < 1.0e-5);
    //"Inhomog (G): value"
    REQUIRE(std::abs(max_eps_dF) < 1.0e-11);
  }

  { // Test DiracODE HartreeFock method:
    // Solve: (Fa and Fb should be equal)
    // (H + v + vp - e)Fa = 0
    // (H + v - e)Fb + X = 0, where X = VpFa (for now, local Vp)
    std::cout << "Test DiracODE HartreeFock method:   \n"
                 "(H + v + vx - e)Fa = 0     vs.\n"
                 "(H + v - e)Fb = -vx*Fa    (Fa and Fb should be equal)\n";
    auto max_eps_dF = -1.0;
    auto max_eps_en = -1.0;
    auto max_eps_orthNorm = -1.0;
    const auto states_new = AtomData::listOfStates_nk("5spdf");

    // This will act as a "non-local" potential
    std::vector<double> vp;
    for (const auto r : grid->r()) {
      vp.push_back(0.1 / (r * r * r * r + 1.0));
    }
    const auto v_tot = qip::add(v_nuc, vp);

    for (const auto &[n, k, en] : states_new) {

      auto Fa = DiracSpinor(n, k, grid);
      auto Fap1 = DiracSpinor(n + 1, k, grid); // for orthog
      auto Fb = DiracSpinor(n, k, grid);
      const auto en_guess = -(Zeff * Zeff) / (2.0 * n * n);
      const auto en_guess_p1 = -(Zeff * Zeff) / (2.0 * (n + 1) * (n + 1));

      // Solve 'a' version (local)
      DiracODE::boundState(Fa, en_guess, v_tot, {}, PhysConst::alpha, 1.0e-15);
      DiracODE::boundState(Fap1, en_guess_p1, v_tot, {}, PhysConst::alpha,
                           1.0e-15);

      auto dvFa = vp * Fa;
      DiracODE::boundState(Fb, Fa.en(), v_nuc, {}, PhysConst::alpha, 1.0e-15,
                           &dvFa, &Fa, 1);

      const auto eps_norm = std::abs(Fb * Fb - 1.0); //<b|b> - norm

      Fb.normalise(); // don't propogate norm error:

      const auto eps_en = std::abs((Fb.en() - Fa.en()) / (Fa.en()));
      const auto eps_orth = std::abs(Fap1 * Fb);  //<a+1|b> - orthogonality
      const auto eps_1 = std::abs(Fa * Fb - 1.0); // <a|b> - check values
      const auto eps_2 = (Fa - Fb) * (Fa - Fb);   // <a-b>^2 - check values

      const auto eps_dF = std::max(eps_1, eps_2);
      const auto eps_orthNorm = std::max(eps_norm, eps_orth);
      printf("%4s <b|b>-1 = %.0e, <a|b>-1 = %.0e, <a-b>^2 "
             "= %.0e, <a+1|b> = "
             "%.0e, en:%.0e\n",
             Fa.shortSymbol().c_str(), eps_norm, eps_1, eps_2, eps_orth,
             eps_en);

      if (!(eps_dF < max_eps_dF))
        max_eps_dF = eps_dF;
      if (!(eps_orthNorm < max_eps_orthNorm))
        max_eps_orthNorm = eps_orthNorm;
      if (!(eps_en < max_eps_en))
        max_eps_en = eps_en;
    }
    //"Dirac ODE HF: orthonorm"
    REQUIRE(std::abs(max_eps_orthNorm) < 5.0e-9);
    //"Dirac ODE HF: value"
    REQUIRE(std::abs(max_eps_dF) < 1.0e-14);
    //"Dirac ODE HF: energy"
    REQUIRE(std::abs(max_eps_en) < 5.0e-9);
  }
}
