#include "SecondOrder.hpp"
#include "Angular/include.hpp"
#include "CI_Integrals.hpp"
#include "DiracOperator/include.hpp"
#include "ExternalField/calcMatrixElements.hpp"
#include "MixedStates.hpp"
#include "fmt/format.hpp"
#include "fmt/ostream.hpp"
#include <algorithm>
#include <cmath>

namespace CI {

//==============================================================================
std::pair<double, double> A_K_coefs(int K, int kt, int ks, int twoJb, int twoJn,
                                    int twoJa) {

  const auto f = std::sqrt(2.0 * K + 1.0) * Angular::neg1pow_2(twoJb + twoJa);

  const auto c1 = Angular::neg1pow(K) * f *
                  Angular::sixj_2(2 * K, 2 * ks, 2 * kt, twoJn, twoJb, twoJa);

  const auto c2 = Angular::neg1pow(kt + ks) * f *
                  Angular::sixj_2(2 * K, 2 * kt, 2 * ks, twoJn, twoJb, twoJa);

  return {c1, c2};
}

//==============================================================================
double z_component(int K, int kt, int ks, int twoJb, int twoJa, int two_m) {
  return Angular::cg_2(2 * kt, 0, 2 * ks, 0, 2 * K, 0) *
         Angular::neg1pow_2(twoJb - two_m) *
         Angular::threej_2(twoJb, 2 * K, twoJa, -two_m, 0, two_m);
}

//==============================================================================
int symm_sign(const DiracOperator::TensorOperator *h, int twoJA, int twoJB) {
  const auto s_imaginary = h->imaginaryQ() ? -1 : 1;
  return s_imaginary * Angular::neg1pow_2(twoJA - twoJB);
}

//==============================================================================
double sigma_rme(const PsiJPi &Psi_b, std::size_t ib, const PsiJPi &Psi_a,
                 std::size_t ia, const std::vector<DiracSpinor> &ci_basis) {

  // sigma = 2S, a rank 1, even parity, operator
  const DiracOperator::s spin{};
  const auto spin_me = ExternalField::me_table(ci_basis, &spin);

  return 2.0 * ReducedME(Psi_b, ib, Psi_a, ia, spin_me, 1, 1);
}

//==============================================================================
std::pair<double, double>
A_K(int K, const PsiJPi &Psi_b, std::size_t ib, const PsiJPi &Psi_a,
    std::size_t ia, const DiracOperator::TensorOperator *t,
    const Coulomb::meTable<double> &t_me,
    const DiracOperator::TensorOperator *s,
    const Coulomb::meTable<double> &s_me, double omega, const Integrals &ints,
    const std::vector<Level> &levels_to_remove, std::ostream &outstream) {

  double A_ket{0.0};
  double A_bra{0.0};

  const auto kt = t->rank();
  const auto ks = s->rank();
  const auto twoJa = Psi_a.twoJ();
  const auto twoJb = Psi_b.twoJ();
  const auto Ea = Psi_a.energy(ia);
  const auto Eb = Psi_b.energy(ib);

  fmt::print(outstream, "{:>4} {:>2} {:>7} {:>4}  {:>14} {:>14} {:>8}\n", "2Jn",
             "pi", "CSFs", "term", "from ket", "from bra", "eps");

  // Intermediate states are connected to a by one operator, and to b by the
  // other. The first ('ts') term has s acting on a, so those states have parity
  // pi_a*pi_s; the second ('st') term has t acting on a: pi_a*pi_t
  const auto twoJn_max = std::max(twoJa, twoJb) + 2 * std::max(kt, ks);
  for (int twoJn = 0; twoJn <= twoJn_max; twoJn += 2) {
    for (const auto pi_n : {-1, 1}) {

      const bool do_ts = pi_n == Psi_a.parity() * s->parity() &&
                         Angular::triangle(twoJb, 2 * kt, twoJn) != 0 &&
                         Angular::triangle(twoJn, 2 * ks, twoJa) != 0;

      const bool do_st = pi_n == Psi_a.parity() * t->parity() &&
                         Angular::triangle(twoJb, 2 * ks, twoJn) != 0 &&
                         Angular::triangle(twoJn, 2 * kt, twoJa) != 0;

      if (!do_ts && !do_st)
        continue;

      PsiJPi target(twoJn, pi_n, ints.ci_basis);
      if (target.CSFs().empty())
        continue;
      const auto Hci = construct_Hci(target, ints);

      // Levels of this (J, parity) that are to be removed from the mixed
      // states. The CI problem must be solved (as far as those levels) first
      std::vector<std::size_t> indices;
      for (const auto &level : levels_to_remove) {
        if (level.twoJ == twoJn && level.parity == pi_n) {
          indices.push_back(level.index);
        }
      }
      if (!indices.empty()) {
        const auto num_solutions =
          *std::max_element(indices.cbegin(), indices.cend()) + 1;
        target.solve(Hci, int(num_solutions));
      }

      // The two sums over the intermediate states of this (J, parity), each
      // formed from the ket (mixed state of a) and from the bra (of b)
      double ts_ket{0.0}, ts_bra{0.0}, st_ket{0.0}, st_bra{0.0};

      if (do_ts) {
        // sum_n <b||t||n><n||s||a>/(E_a - E_n)
        const auto da =
          project_out(solve_mixed_state(Psi_a, ia, target, Hci, s_me, ks, 0.0),
                      target, indices);
        ts_ket = ReducedME(Psi_b, ib, da, 0, t_me, kt, t->parity());

        const auto Db = project_out(
          solve_mixed_state(Psi_b, ib, target, Hci, t_me, kt, Ea - Eb), target,
          indices);
        ts_bra = symm_sign(t, twoJb, twoJn) *
                 ReducedME(Db, 0, Psi_a, ia, s_me, ks, s->parity());
      }

      if (do_st) {
        // sum_n <b||s||n><n||t||a>/(E_a + omega - E_n)
        const auto Da = project_out(
          solve_mixed_state(Psi_a, ia, target, Hci, t_me, kt, omega), target,
          indices);
        st_ket = ReducedME(Psi_b, ib, Da, 0, s_me, ks, s->parity());

        const auto db = project_out(
          solve_mixed_state(Psi_b, ib, target, Hci, s_me, ks, Ea + omega - Eb),
          target, indices);
        st_bra = symm_sign(s, twoJb, twoJn) *
                 ReducedME(db, 0, Psi_a, ia, t_me, kt, t->parity());
      }

      // A^K: from the ket, this uses the mixed states of s (ts term) and of t
      // (st term); from the bra, the other way around
      const auto [c1, c2] = A_K_coefs(K, kt, ks, twoJb, twoJn, twoJa);
      A_ket += c1 * ts_ket + c2 * st_bra;
      A_bra += c1 * ts_bra + c2 * st_ket;

      const auto pi_str = pi_n == 1 ? '+' : '-';
      if (do_ts) {
        fmt::print(
          outstream, "{:>4} {:>2} {:>7} {:>4}  {:14.6e} {:14.6e} {:8.1e}\n",
          twoJn, pi_str, target.CSFs().size(), "ts", ts_ket, ts_bra,
          std::abs(ts_ket - ts_bra) / std::max(std::abs(ts_ket), 1.0e-30));
      }
      if (do_st) {
        fmt::print(
          outstream, "{:>4} {:>2} {:>7} {:>4}  {:14.6e} {:14.6e} {:8.1e}\n",
          twoJn, pi_str, target.CSFs().size(), "st", st_ket, st_bra,
          std::abs(st_ket - st_bra) / std::max(std::abs(st_bra), 1.0e-30));
      }
      outstream << std::flush;
    }
  }

  return {A_ket, A_bra};
}

} // namespace CI
