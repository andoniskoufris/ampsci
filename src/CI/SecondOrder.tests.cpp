#include "SecondOrder.hpp"
#include "Angular/include.hpp"
#include "CI_Integrals.hpp"
#include "Coulomb/include.hpp"
#include "DiracOperator/include.hpp"
#include "ExternalField/calcMatrixElements.hpp"
#include "Wavefunction/Wavefunction.hpp"
#include "catch2/catch.hpp"
#include "fmt/format.hpp"
#include <cmath>
#include <vector>

//==============================================================================
// Wigner-Eckart factor for the q=0 component:
// <a,m|h^k_0|b,m> = we_factor * <a||h^k||b>  (cf TensorOperator::rme3js)
static double we_factor(int k, int twoja, int twojb, int two_m) {
  return Angular::neg1pow_2(twoja - two_m) *
         Angular::threej_2(twoja, 2 * k, twojb, -two_m, 0, two_m);
}

//==============================================================================
// Tests the angular coefficients of the second-order amplitude A^K.
// A^K is the reduced ME of the composite operator [t x s]^K, so for each
// intermediate state the sum over K of the z-component factors must reproduce
// the product of the two Wigner-Eckart factors of <b,m|t_0|n,m><n,m|s_0|a,m>
TEST_CASE("CI: second-order angular coefficients", "[CI][SecondOrder][unit]") {

  std::cout << "CI second-order amplitude: angular coefficients\n";

  double worst = 0.0;
  int num_tested = 0;

  for (int kt = 0; kt <= 2; ++kt) {
    for (int ks = 0; ks <= 2; ++ks) {

      for (int twoJa = 0; twoJa <= 6; twoJa += 2) {
        for (int twoJb = 0; twoJb <= 6; twoJb += 2) {
          for (int twoJn = 0; twoJn <= 8; twoJn += 2) {

            if (Angular::triangle(twoJb, 2 * kt, twoJn) == 0)
              continue;
            if (Angular::triangle(twoJn, 2 * ks, twoJa) == 0)
              continue;

            const auto two_m = std::min(twoJa, twoJb);

            // The two factors, for each of the two terms of A^K
            const auto zz_1 = we_factor(kt, twoJb, twoJn, two_m) *
                              we_factor(ks, twoJn, twoJa, two_m);
            const auto zz_2 = we_factor(ks, twoJb, twoJn, two_m) *
                              we_factor(kt, twoJn, twoJa, two_m);

            // The same, summed over the ranks K
            double sum_1 = 0.0;
            double sum_2 = 0.0;
            for (int K = std::abs(kt - ks); K <= kt + ks; ++K) {
              const auto [c1, c2] =
                CI::A_K_coefs(K, kt, ks, twoJb, twoJn, twoJa);
              const auto z = CI::z_component(K, kt, ks, twoJb, twoJa, two_m);
              sum_1 += z * c1;
              sum_2 += z * c2;
            }

            const auto scale = std::max({std::abs(zz_1), std::abs(zz_2), 1.0});
            worst = std::max({worst, std::abs(sum_1 - zz_1) / scale,
                              std::abs(sum_2 - zz_2) / scale});
            ++num_tested;
          }
        }
      }
    }
  }

  fmt::print("{} cases, worst: {:.1e}\n", num_tested, worst);
  REQUIRE(num_tested > 100);
  REQUIRE(worst < 1.0e-12);
}

//==============================================================================
// Reduced ME of the total spin between LS-coupled states of the same L and S:
//   <LSJb||S||LSJa> = (-1)^(L+S+Jb+1) sqrt([Ja][Jb]) {S Jb L; Ja S 1}
//                     * sqrt(S(S+1)(2S+1))
// A reference for CI::sigma_rme, which does not assume L or S are good
// quantum numbers (they are not, for relativistic CI states)
static double spin_rme_LS(int L, int twoS, int twoJb, int twoJa) {
  const auto rme_S = std::sqrt(double(twoS) * (twoS + 2) * (twoS + 1)) / 2;
  return Angular::neg1pow_2(2 * L + twoS + twoJb + 2) *
         std::sqrt(double(twoJa + 1) * (twoJb + 1)) *
         Angular::sixj_2(twoS, twoJb, 2 * L, twoJa, twoS, 2) * rme_S;
}

//==============================================================================
// The LS reference above must reproduce Angular::S_kk for a single electron
// (L = l, S = 1/2, J = j)
TEST_CASE("CI: spin reduced ME", "[CI][SecondOrder][unit]") {

  std::cout << "CI: <LSJ'||S||LSJ> against S_kk\n";

  double worst = 0.0;
  int num_tested = 0;

  for (int l = 0; l <= 4; ++l) {
    // kappa = -l-1 (j=l+1/2) and kappa = l (j=l-1/2)
    const std::vector<int> kappas =
      l == 0 ? std::vector{-1} : std::vector{-l - 1, l};
    for (const auto ka : kappas) {
      for (const auto kb : kappas) {
        const auto twoja = Angular::twoj_k(ka);
        const auto twojb = Angular::twoj_k(kb);
        // nb: S_kk(ka,kb) is <ka||S||kb>, i.e., 'a' is the bra
        const auto expected = Angular::S_kk(ka, kb);
        const auto found = spin_rme_LS(l, 1, twoja, twojb);
        worst = std::max(worst, std::abs(found - expected));
        ++num_tested;
        fmt::print("l={}: <{}/2||S||{}/2> = {:12.8f} [{:12.8f}]\n", l, twoja,
                   twojb, found, expected);
      }
    }
  }

  fmt::print("{} cases, worst: {:.1e}\n", num_tested, worst);
  REQUIRE(num_tested == 17);
  REQUIRE(worst < 1.0e-12);
}

//==============================================================================
// Tests the second-order amplitude for a two-electron CI:
//  - the two routes (mixed states of s, and of t) must agree
//  - the static scalar polarisability of the He ground state, which fixes the
//    overall sign and normalisation
TEST_CASE("CI: second-order amplitudes", "[CI][SecondOrder][unit]") {

  std::cout << "CI second-order amplitudes (not meant to be accurate)\n";

  Wavefunction wf({400, 1.0e-4, 20.0, 0.33 * 20.0, "loglinear"},
                  {"He", -1, "pointlike"}, 1.0);
  wf.solve_core("HartreeFock", "[]", std::nullopt, 1.0e-5);
  wf.formBasis(SplineBasis::Parameters("6spd", 20, 6, 1.0e-2, 1.0e-2, 20.0));

  CI::Integrals ints;
  ints.ci_basis = CI::basis_subset(wf.basis(), "5spd");
  {
    const Coulomb::YkTable yk(ints.ci_basis);
    ints.qk.fill(ints.ci_basis, yk, 8, false);
  }
  ints.h1 = CI::calculate_h1_table(ints.ci_basis, {}, {}, ints.qk, false);
  REQUIRE(ints.availableQ());

  // J=0 even (ground state and first excited), and J=1 even
  CI::PsiJPi Psi_0e(0, +1, ints.ci_basis);
  Psi_0e.solve(CI::construct_Hci(Psi_0e, ints));
  CI::PsiJPi Psi_1e(2, +1, ints.ci_basis);
  Psi_1e.solve(CI::construct_Hci(Psi_1e, ints));

  const DiracOperator::E1 d{wf.grid()};
  const DiracOperator::PNCnsi hpnc{1.0, Nuclear::default_t, wf.grid()};
  const auto d_me = ExternalField::me_table(ints.ci_basis, &d);
  const auto pnc_me = ExternalField::me_table(ints.ci_basis, &hpnc);

  //----------------------------------------------------------------------------
  // The spin matrix element that defines beta. The lowest J=1 even solution is
  // 1s2s 3S_1: two s-electrons coupled to J=1 can only be a triplet S, so it
  // is (almost) pure L=0, S=1, and <b||sigma||a> = 2<LSJ||S||LSJ>
  {
    const auto sigma = CI::sigma_rme(Psi_1e, 0, Psi_1e, 0, ints.ci_basis);
    const auto expected = 2.0 * spin_rme_LS(0, 2, 2, 2);
    fmt::print("\n<b||sigma||a> for 1s2s 3S_1: {:.6f} [{:.6f}]\n", sigma,
               expected);
    REQUIRE(std::abs(sigma - expected) / expected < 1.0e-3);
  }

  //----------------------------------------------------------------------------
  // Static polarisability of the ground state (all K), and the dynamic case
  for (const auto omega : {0.0, 0.05}) {
    fmt::print("\nE1-E1, J=0 -> J=0, omega = {:.2f}\n", omega);
    const auto [A_ket, A_bra] =
      CI::A_K(0, Psi_0e, 0, Psi_0e, 0, &d, d_me, &d, d_me, omega, ints);
    const auto eps = std::abs(A_ket - A_bra) / std::abs(A_ket);
    fmt::print("A^0: {:.8e} (ket), {:.8e} (bra) [{:.1e}]\n", A_ket, A_bra, eps);
    REQUIRE(eps < 1.0e-10);

    if (omega == 0.0) {
      // alpha_0 = +A^0/sqrt(3[J]); exact for He is 1.383 au, but this basis is
      // small: only the sign and rough magnitude are meaningful
      const auto alpha = A_ket / std::sqrt(3.0);
      fmt::print("alpha_0 = {:.4f} au (expect ~1.4)\n", alpha);
      REQUIRE(alpha > 1.0);
      REQUIRE(alpha < 2.0);
    }
  }

  //----------------------------------------------------------------------------
  // PNC-like amplitude: both operators odd, so a and b have the same parity.
  // Tests the case k_t != k_s (the two terms then have different J_n)
  {
    fmt::print("\nE1-pnc, J=0 -> J=1\n");
    const auto omega = Psi_1e.energy(0) - Psi_0e.energy(0);
    const auto [A_ket, A_bra] =
      CI::A_K(1, Psi_1e, 0, Psi_0e, 0, &d, d_me, &hpnc, pnc_me, omega, ints);
    const auto eps = std::abs(A_ket - A_bra) / std::abs(A_ket);
    fmt::print("A^1: {:.8e} (ket), {:.8e} (bra) [{:.1e}]\n", A_ket, A_bra, eps);
    REQUIRE(eps < 1.0e-10);
    REQUIRE(std::abs(A_ket) > 1.0e-12);
  }

  //----------------------------------------------------------------------------
  // Transition amplitude between the two lowest J=0 even states: K = 0 only
  {
    fmt::print("\nE1-E1, transition between two J=0 even states\n");
    const auto omega = Psi_0e.energy(1) - Psi_0e.energy(0);
    const auto [A_ket, A_bra] =
      CI::A_K(0, Psi_0e, 1, Psi_0e, 0, &d, d_me, &d, d_me, omega, ints);
    const auto eps = std::abs(A_ket - A_bra) / std::abs(A_ket);
    fmt::print("A^0: {:.8e} (ket), {:.8e} (bra) [{:.1e}]\n", A_ket, A_bra, eps);
    REQUIRE(eps < 1.0e-10);
  }

  //----------------------------------------------------------------------------
  // Projecting out intermediate levels: the amplitude must drop by exactly
  // their sum-over-states contributions
  {
    fmt::print("\nProjecting out intermediate levels\n");
    const auto A_all =
      CI::A_K(0, Psi_0e, 0, Psi_0e, 0, &d, d_me, &d, d_me, 0.0, ints).first;

    const std::vector<CI::Level> project{{2, -1, 0}, {2, -1, 2}};
    const auto A_cut =
      CI::A_K(0, Psi_0e, 0, Psi_0e, 0, &d, d_me, &d, d_me, 0.0, ints, project)
        .first;

    // The removed levels, by direct sum-over-states
    CI::PsiJPi Psi_1o(2, -1, ints.ci_basis);
    Psi_1o.solve(CI::construct_Hci(Psi_1o, ints));
    double removed = 0.0;
    for (const auto &level : project) {
      const auto ip = level.index;
      const auto dE = Psi_0e.energy(0) - Psi_1o.energy(ip);
      const auto d_bp = CI::ReducedME(Psi_0e, 0, Psi_1o, ip, d_me, 1, -1);
      const auto d_pa = CI::ReducedME(Psi_1o, ip, Psi_0e, 0, d_me, 1, -1);
      const auto [c1, c2] = CI::A_K_coefs(0, 1, 1, 0, 2, 0);
      removed += (c1 + c2) * d_bp * d_pa / dE;
    }

    const auto eps = std::abs(A_cut + removed - A_all) / std::abs(A_all);
    fmt::print("A^0: {:.8e} = {:.8e} + {:.8e} [{:.1e}]\n", A_all, A_cut,
               removed, eps);
    REQUIRE(eps < 1.0e-10);
  }
}
