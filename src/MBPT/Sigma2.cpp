#include "Sigma2.hpp"
#include "Angular/include.hpp"
#include "Coulomb/include.hpp"
#include "MBPT/Feynman.hpp"
#include "Wavefunction/DiracSpinor.hpp"

namespace MBPT {

//==============================================================================
// Returns string representation of Denominators enum
std::string parse_Denominators(Denominators d) {
  if (d == Denominators::RS)
    return "RS";
  if (d == Denominators::Fermi)
    return "Fermi";
  return "Fermi0";
}

// Parses string to Denominators enum (case-insensitive); returns Fermi0 if unrecognised
Denominators parse_Denominators(std::string_view s) {
  if (qip::ci_compare(s, "RS"))
    return Denominators::RS;
  if (qip::ci_compare(s, "Fermi"))
    return Denominators::Fermi;
  return Denominators::Fermi0;
}

//==============================================================================
std::pair<std::vector<DiracSpinor>, std::vector<DiracSpinor>>
split_basis(const std::vector<DiracSpinor> &basis, double E_Fermi,
            int min_n_core, int max_n_excited) {

  std::pair<std::vector<DiracSpinor>, std::vector<DiracSpinor>> core_excited;
  auto &[core, excited] = core_excited;
  for (const auto &Fn : basis) {
    if (Fn.en() <= E_Fermi && Fn.n() >= min_n_core) {
      core.push_back(Fn);
    } else if (Fn.en() > E_Fermi && Fn.n() <= max_n_excited) {
      excited.push_back(Fn);
    }
  }
  return core_excited;
}

//==============================================================================
double e_bar(int kappa_v, const std::vector<DiracSpinor> &excited) {
  const auto v_bar = std::find_if(
    excited.cbegin(), excited.cend(),
    [kappa_v](const DiracSpinor &n) { return n.kappa() == kappa_v; });
  if (v_bar == excited.cend()) {
    return 0.0;
  }
  return v_bar->en();
}

//==============================================================================
bool Sk_vwxy_SR(int k, const DiracSpinor &v, const DiracSpinor &w,
                const DiracSpinor &x, const DiracSpinor &y) {
  return Coulomb::sixjTriads({}, {}, k, v, x, {}) &&
         Coulomb::sixjTriads({}, {}, k, w, y, {});
}

//==============================================================================
std::pair<int, int> k_minmax_S(const DiracSpinor &v, const DiracSpinor &w,
                               const DiracSpinor &x, const DiracSpinor &y) {

  // From the 6j part only:
  // |b-d| <= k <=|b+d|
  // |a-c| <= k <=|a+c|
  const auto [lk1, uk1] = Coulomb::k_minmax_tj(v.twoj(), x.twoj());
  const auto [lk2, uk2] = Coulomb::k_minmax_tj(w.twoj(), y.twoj());

  return {std::max({lk1, lk2}), std::min({uk1, uk2})};
}

std::pair<int, int> k_minmax_S(int twojv, int twojw, int twojx, int twojy) {

  // From the 6j part only:
  // |b-d| <= k <=|b+d|
  // |a-c| <= k <=|a+c|
  const auto [lk1, uk1] = Coulomb::k_minmax_tj(twojv, twojx);
  const auto [lk2, uk2] = Coulomb::k_minmax_tj(twojw, twojy);

  return {std::max({lk1, lk2}), std::min({uk1, uk2})};
}

//==============================================================================
double Sk_vwxy(int k, const DiracSpinor &v, const DiracSpinor &w,
               const DiracSpinor &x, const DiracSpinor &y,
               const Coulomb::QkTable &qk, const std::vector<DiracSpinor> &core,
               const std::vector<DiracSpinor> &excited,
               const Angular::SixJTable &SixJ, Denominators denominators) {
  using namespace Sigma2;

  if (!Sk_vwxy_SR(k, v, w, x, y))
    return 0.0;

  return S_Sigma2_ab(k, v, w, x, y, qk, core, excited, SixJ, denominators) +
         S_Sigma2_c1(k, v, w, x, y, qk, core, excited, SixJ, denominators) +
         S_Sigma2_c2(k, v, w, x, y, qk, core, excited, SixJ, denominators) +
         S_Sigma2_d(k, v, w, x, y, qk, core, excited, SixJ, denominators);
}

//==============================================================================
// this just calculates the screening Sk integrals
double Sk_vwxy_screened(
  int k, const DiracSpinor &v, const DiracSpinor &w, const DiracSpinor &x,
  const DiracSpinor &y, const Coulomb::QkTable &qk,
  const std::vector<DiracSpinor> &core, const std::vector<DiracSpinor> &excited,
  const Angular::SixJTable &SixJ, Denominators denominators,
  const std::vector<MBPT::ComplexRMatrix> &dQ_screen,
  const std::vector<LinAlg::Matrix<MBPT::ComplexRMatrix>> &dQ_screen_Fermi) {
  using namespace Sigma2;

  if (!Sk_vwxy_SR(k, v, w, x, y)) {
    return 0.0;
  }

  return S_Sigma2_screen(k, v, w, x, y, excited, denominators, dQ_screen,
                         dQ_screen_Fermi) +
         S_Sigma2_ab(k, v, w, x, y, qk, core, excited, SixJ, denominators,
                     true) +
         S_Sigma2_c1(k, v, w, x, y, qk, core, excited, SixJ, denominators) +
         S_Sigma2_c2(k, v, w, x, y, qk, core, excited, SixJ, denominators) +
         S_Sigma2_d(k, v, w, x, y, qk, core, excited, SixJ, denominators);
}

//==============================================================================
//==============================================================================
//==============================================================================
double Sigma2::S_Sigma2_ab(int k, const DiracSpinor &v, const DiracSpinor &w,
                           const DiracSpinor &x, const DiracSpinor &y,
                           const Coulomb::QkTable &qk,
                           const std::vector<DiracSpinor> &core,
                           const std::vector<DiracSpinor> &excited,
                           const Angular::SixJTable &SixJ,
                           Denominators denominators, const bool &screen) {

  // overall selectrion rule tested outside

  const auto f = Angular::neg1pow(k) / (2.0 * k + 1.0);

  const auto v0 = e_bar(v.kappa(), excited);
  const auto w0 = e_bar(w.kappa(), excited);
  const auto x0 = e_bar(x.kappa(), excited);
  const auto y0 = e_bar(y.kappa(), excited);

  // const auto de_xv = x.en() - v.en();
  const auto de_xv = denominators == Denominators::Fermi0 ?
                       0.0 :
                     denominators == Denominators::Fermi ?
                       0.5 * (x0 - v0 + y0 - w0) :
                       0.5 * (x.en() - v.en() + y.en() - w.en());

  double sum = 0.0;
  for (const auto &a : core) {
    for (const auto &n : excited) {
      const auto de = de_xv + a.en() - n.en();

      // A diagrams:
      const auto qk_vnxa = qk.Q(k, v, n, x, a);
      const auto pk_vnxa = qk.P(k, v, n, x, a, &SixJ);

      const auto qk_awny = qk.Q(k, a, w, n, y);
      const auto pk_awny = qk.P(k, a, w, n, y, &SixJ);
      // _DON'T_ include diagram a1 if we are screening, since this would overcount
      const auto wk_awny = screen ? pk_awny : qk_awny + pk_awny;

      // diagrams a1, a2, a3:
      sum += (qk_vnxa * wk_awny + pk_vnxa * qk_awny) / de;

      // B diagrams: a <-> n
      const auto qk_vaxn = qk_vnxa;
      const auto pk_vaxn = v == x ? pk_vnxa : qk.P(k, v, a, x, n, &SixJ);
      const auto qk_nway = qk_awny;
      const auto pk_nway = w == y ? pk_awny : qk.P(k, n, w, a, y, &SixJ);
      // again, we don't include diagram b1 if we are screening
      const auto wk_nway = screen ? pk_nway : qk_nway + pk_nway;

      // diagrams b1, b2, b3:
      sum += (qk_vaxn * wk_nway + pk_vaxn * qk_nway) / de;
    }
  }

  return f * sum;
}

//==============================================================================
double Sigma2::S_Sigma2_c1(int k, const DiracSpinor &v, const DiracSpinor &w,
                           const DiracSpinor &x, const DiracSpinor &y,
                           const Coulomb::QkTable &qk,
                           const std::vector<DiracSpinor> &core,
                           const std::vector<DiracSpinor> &excited,
                           const Angular::SixJTable &SixJ,
                           Denominators denominators) {

  // overall selectrion rule tested outside

  const auto f =
    Angular::neg1pow_2(v.twoj() + w.twoj() + x.twoj() + y.twoj() + 2 * k) *
    (2.0 * k + 1.0);

  const auto v0 = e_bar(v.kappa(), excited);
  const auto w0 = e_bar(w.kappa(), excited);
  const auto x0 = e_bar(x.kappa(), excited);
  const auto y0 = e_bar(y.kappa(), excited);

  // const auto de_yv = y.en() - v.en();
  const auto de_yv = denominators == Denominators::Fermi0 ?
                       0.0 :
                     denominators == Denominators::Fermi ?
                       0.5 * (y0 - v0 + x0 - w0) :
                       0.5 * (y.en() - v.en() + x.en() - w.en());

  double sum = 0.0;
  for (const auto &a : core) {
    if (!Coulomb::sixjTriads({}, {}, k, v, x, a))
      continue;
    for (const auto &n : excited) {
      const auto [u0, u1] = Coulomb::k_minmax_Q(v, n, a, y);
      const auto [l0, l1] = Coulomb::k_minmax_Q(a, w, x, n);
      if (l0 > l1)
        continue;

      if (!Coulomb::sixjTriads({}, {}, k, y, w, n))
        continue;

      const auto de = de_yv + a.en() - n.en();

      for (int u = u0; u <= u1; u += 2) {
        const auto l0_SixJ = l0; // allow += 2
        const auto l1_SixJ = std::min(l1, std::abs(u + k));
        for (int l = l0_SixJ; l <= l1_SixJ; l += 2) {

          const auto SixJ1 = SixJ.get(l, u, k, v, x, a);
          const auto SixJ2 = SixJ.get(l, u, k, y, w, n);
          const auto s = Angular::neg1pow_2(2 * a.twoj() + 2 * l + 2 * u);

          const auto qk_vnay = qk.Q(u, v, n, a, y);
          const auto qk_awxn = qk.Q(l, a, w, x, n);

          sum += s * SixJ1 * SixJ2 * qk_vnay * qk_awxn / de;
        }
      }
    }
  }
  return f * sum;
}

//==============================================================================
double Sigma2::S_Sigma2_c2(int k, const DiracSpinor &v, const DiracSpinor &w,
                           const DiracSpinor &x, const DiracSpinor &y,
                           const Coulomb::QkTable &qk,
                           const std::vector<DiracSpinor> &core,
                           const std::vector<DiracSpinor> &excited,
                           const Angular::SixJTable &SixJ,
                           Denominators denominators) {

  // overall selectrion rule tested outside

  const auto v0 = e_bar(v.kappa(), excited);
  const auto w0 = e_bar(w.kappa(), excited);
  const auto x0 = e_bar(x.kappa(), excited);
  const auto y0 = e_bar(y.kappa(), excited);

  const auto f =
    Angular::neg1pow_2(v.twoj() + w.twoj() + x.twoj() + y.twoj() + 2 * k) *
    (2.0 * k + 1.0);

  const auto de_yv = denominators == Denominators::Fermi0 ?
                       0.0 :
                     denominators == Denominators::Fermi ?
                       0.5 * (x0 - w0 + y0 - v0) :
                       0.5 * (x.en() - w.en() + y.en() - v.en());

  double sum = 0.0;
  for (const auto &a : core) {
    if (!Coulomb::sixjTriads({}, {}, k, y, w, a))
      continue;
    for (const auto &n : excited) {

      const auto [u0, u1] = Coulomb::k_minmax_Q(v, a, n, y);
      const auto [l0, l1] = Coulomb::k_minmax_Q(n, w, x, a);
      if (l0 > l1)
        continue;

      if (!Coulomb::sixjTriads({}, {}, k, v, x, n))
        continue;

      const auto de = de_yv + a.en() - n.en();

      for (int u = u0; u <= u1; u += 2) {
        const auto l0_SixJ = l0; // allow += 2
        const auto l1_SixJ = std::min(l1, std::abs(u + k));
        for (int l = l0_SixJ; l <= l1_SixJ; l += 2) {

          const auto SixJ1 = SixJ.get(l, u, k, v, x, n);
          const auto SixJ2 = SixJ.get(l, u, k, y, w, a);
          const auto s = Angular::neg1pow_2(2 * a.twoj() + 2 * l + 2 * u);

          const auto qk_vany = qk.Q(u, v, a, n, y);
          const auto qk_nwxa = qk.Q(l, n, w, x, a);

          sum += s * SixJ1 * SixJ2 * qk_vany * qk_nwxa / de;
        }
      }
    }
  }
  return f * sum;
}

//==============================================================================
double Sigma2::S_Sigma2_d(int k, const DiracSpinor &v, const DiracSpinor &w,
                          const DiracSpinor &x, const DiracSpinor &y,
                          const Coulomb::QkTable &qk,
                          const std::vector<DiracSpinor> &core,
                          const std::vector<DiracSpinor> &excited,
                          const Angular::SixJTable &SixJ,
                          Denominators denominators) {

  const auto f = Angular::neg1pow_2(v.twoj() + w.twoj() + x.twoj() + y.twoj()) *
                 (2.0 * k + 1.0);

  const auto v0 = e_bar(v.kappa(), excited);
  const auto w0 = e_bar(w.kappa(), excited);
  const auto x0 = e_bar(x.kappa(), excited);
  const auto y0 = e_bar(y.kappa(), excited);
  const auto e0 = DiracSpinor::min_En(excited);

  // Here, "Fermi0" cancellation doesn't happen
  // So, Fermi and Fermi0 are the same
  const auto de_vw = denominators == Denominators::Fermi0 ?
                       2.0 * e0 :
                     denominators == Denominators::Fermi ?
                       -0.5 * (v0 + w0 + x0 + y0) :
                       -0.5 * (v.en() + w.en() + x.en() + y.en());

  double sum = 0.0;
  for (const auto &a : core) {
    if (!Coulomb::sixjTriads({}, {}, k, v, x, a))
      continue;

    for (const auto &b : core) {

      const auto [u0, u1] = Coulomb::k_minmax_Q(v, w, a, b);
      const auto [l0, l1] = Coulomb::k_minmax_Q(a, b, x, y);
      if (l0 > l1)
        continue;

      if (!Coulomb::sixjTriads({}, {}, k, w, y, b))
        continue;

      const auto de = de_vw + a.en() + b.en();
      const auto s = Angular::neg1pow_2(a.twoj() - b.twoj());

      for (int u = u0; u <= u1; u += 2) {
        const auto l0_SixJ = l0;
        const auto l1_SixJ = std::min(l1, std::abs(u + k));
        for (int l = l0_SixJ; l <= l1_SixJ; l += 2) {

          if (!Coulomb::triangle(l, u, k))
            continue;

          const auto SixJ1 = SixJ.get(l, u, k, v, x, a);
          const auto SixJ2 = SixJ.get(l, u, k, w, y, b);

          const auto qu_vwab = qk.Q(u, v, w, a, b);
          const auto ql_abxy = qk.Q(l, a, b, x, y);

          sum += s * SixJ1 * SixJ2 * qu_vwab * ql_abxy / de;
        }
      }
    }
  }
  return f * sum;
}

//==============================================================================
double Sigma2::S_Sigma2_screen(
  int k, const DiracSpinor &v, const DiracSpinor &w, const DiracSpinor &x,
  const DiracSpinor &y, const std::vector<DiracSpinor> &excited,
  Denominators denominators, const std::vector<MBPT::ComplexRMatrix> &Q_screen,
  const std::vector<LinAlg::Matrix<MBPT::ComplexRMatrix>> &dQ_screen_Fermi) {

  // overall selectrion rule tested outside
  const auto f = Angular::neg1pow_2(2 * k + v.twoj() - w.twoj());
  const auto Ck_vx = Angular::Ck_kk(k, v.kappa(), x.kappa());
  const auto Ck_wy = Angular::Ck_kk(k, w.kappa(), y.kappa());

  const auto ebar_v = e_bar(v.kappa(), excited);
  const auto ebar_w = e_bar(w.kappa(), excited);
  const auto ebar_x = e_bar(x.kappa(), excited);
  const auto ebar_y = e_bar(y.kappa(), excited);

  const auto de_vx = std::abs(ebar_v - ebar_x);
  const auto de_wy = std::abs(ebar_w - ebar_y);

  if (denominators == MBPT::Denominators::Fermi0 ||
      (de_vx <= 0.1 && de_wy <= 0.1)) {
    return f * Ck_vx * Ck_wy * two_body_ME(Q_screen[k], v, w, x, y);
  } else if (de_vx > 0.1 && de_wy <= 0.1) {
    const int kvi = Angular::kappa_to_kindex(v.kappa());
    // const int kwi = Angular::kappa_to_kindex(w.kappa());
    const int kxi = Angular::kappa_to_kindex(x.kappa());
    // const int kyi = Angular::kappa_to_kindex(y.kappa());

    return f * Ck_vx * Ck_wy *
           two_body_ME(
             0.5 * (dQ_screen_Fermi[k](std::max(kvi, kxi), std::min(kvi, kxi)) +
                    Q_screen[k]),
             v, w, x, y);
  } else if (de_vx <= 0.1 && de_wy > 0.1) {
    const int kwi = Angular::kappa_to_kindex(w.kappa());
    const int kyi = Angular::kappa_to_kindex(y.kappa());

    return f * Ck_vx * Ck_wy *
           two_body_ME(
             0.5 * (dQ_screen_Fermi[k](std::max(kwi, kyi), std::min(kwi, kyi)) +
                    Q_screen[k]),
             v, w, x, y);
  } else {
    const int kvi = Angular::kappa_to_kindex(v.kappa());
    const int kwi = Angular::kappa_to_kindex(w.kappa());
    const int kxi = Angular::kappa_to_kindex(x.kappa());
    const int kyi = Angular::kappa_to_kindex(y.kappa());

    return f * Ck_vx * Ck_wy *
           two_body_ME(
             0.5 * (dQ_screen_Fermi[k](std::max(kvi, kxi), std::min(kvi, kxi)) +
                    dQ_screen_Fermi[k](std::max(kwi, kyi), std::min(kwi, kyi))),
             v, w, x, y);
  }
}

//==============================================================================
Coulomb::LkTable calculate_Sk(
  const std::string &filename, const std::vector<DiracSpinor> &external,
  const std::vector<DiracSpinor> &core, const std::vector<DiracSpinor> &excited,
  const Coulomb::QkTable &qk, int max_k, bool exclude_wrong_parity_box,
  Denominators denominators, bool no_new_integrals) {

  Coulomb::LkTable Sk;

  const auto max_twoj =
    std::max({DiracSpinor::max_tj(excited), DiracSpinor::max_tj(core),
              DiracSpinor::max_tj(external)});
  Angular::SixJTable sjt(max_twoj);

  const auto Sk_function = [&](int k, const DiracSpinor &v,
                               const DiracSpinor &w, const DiracSpinor &x,
                               const DiracSpinor &y) {
    return MBPT::Sk_vwxy(k, v, w, x, y, qk, core, excited, sjt, denominators);
  };
  const auto Sk_selection_rule = [&](int k, const DiracSpinor &v,
                                     const DiracSpinor &w, const DiracSpinor &x,
                                     const DiracSpinor &y) {
    return exclude_wrong_parity_box ? Coulomb::Qk_abcd_SR(k, v, w, x, y) :
                                      MBPT::Sk_vwxy_SR(k, v, w, x, y);
  };

  // Try to read from disk (may already have calculated Qk)
  Sk.read(filename);

  const auto existing = Sk.count();
  {
    if (!no_new_integrals)
      Sk.fill(external, Sk_function, Sk_selection_rule, max_k);

    const auto total = Sk.count();
    assert(total >= existing);
    const auto new_integrals = total - existing;
    std::cout << "Calculated " << new_integrals << " new MBPT integrals\n";
    if (new_integrals > 0) {
      Sk.write(filename);
    }
  }
  std::cout << "\n" << std::flush;
  return Sk;
}

Coulomb::LkTable calculate_Sk_screened(
  const std::string &filename, const std::vector<DiracSpinor> &external,
  const std::vector<DiracSpinor> &core, const std::vector<DiracSpinor> &excited,
  const Coulomb::QkTable &qk, int max_k, bool exclude_wrong_parity_box,
  Denominators denominators, const std::vector<MBPT::ComplexRMatrix> &Q_screen,
  const std::vector<LinAlg::Matrix<MBPT::ComplexRMatrix>> &dQ_screen_Fermi,
  bool no_new_integrals) {

  Coulomb::LkTable Sk;

  const auto max_twoj =
    std::max({DiracSpinor::max_tj(excited), DiracSpinor::max_tj(core),
              DiracSpinor::max_tj(external)});
  Angular::SixJTable sjt(max_twoj);

  const auto Sk_function = [&](int k, const DiracSpinor &v,
                               const DiracSpinor &w, const DiracSpinor &x,
                               const DiracSpinor &y) {
    return MBPT::Sk_vwxy_screened(k, v, w, x, y, qk, core, excited, sjt,
                                  denominators, Q_screen, dQ_screen_Fermi);
  };
  const auto Sk_selection_rule = [&](int k, const DiracSpinor &v,
                                     const DiracSpinor &w, const DiracSpinor &x,
                                     const DiracSpinor &y) {
    return exclude_wrong_parity_box ? Coulomb::Qk_abcd_SR(k, v, w, x, y) :
                                      MBPT::Sk_vwxy_SR(k, v, w, x, y);
  };

  // Try to read from disk (may already have calculated Qk)
  Sk.read(filename);

  const auto existing = Sk.count();
  {
    if (!no_new_integrals)
      Sk.fill(external, Sk_function, Sk_selection_rule, max_k);

    const auto total = Sk.count();
    assert(total >= existing);
    const auto new_integrals = total - existing;
    std::cout << "Calculated " << new_integrals << " new MBPT integrals\n";
    if (new_integrals > 0) {
      Sk.write(filename);
    }
  }
  std::cout << "\n" << std::flush;
  return Sk;
}

// we only fill in lower half of the matrix to save on memory allocation
void FillPiMatrix(const int &max_k_Coulomb,
                  const std::vector<DiracSpinor> &cis2_basis,
                  const std::string &cis2_basis_str,
                  const std::vector<int> &kappai_list, const Feynman &feyn,
                  std::vector<LinAlg::Matrix<ComplexRMatrix>> &PiMatrix) {

  qip::ProgressBar bar(kappai_list.size(), true);

#pragma omp parallel for collapse(2)
  for (int kv_i = kappai_list[0]; kv_i < kappai_list.back(); kv_i++) {
    for (int kw_i = 0; kw_i < kv_i; kw_i++) {
      auto kw = Angular::kindex_to_kappa(kw_i);
      auto kv = Angular::kindex_to_kappa(kv_i);
      double ebar_v = MBPT::e_bar(kv, cis2_basis);
      double ebar_w = MBPT::e_bar(kw, cis2_basis);
      double de = std::abs(ebar_v - ebar_w);
      for (int k = 0; k <= max_k_Coulomb; k++) {
        std::size_t sk = k;

        // do not bother calculating the polarisation operator
        // if the matrix element <vx|dQ|wy> will be zero
        if (Angular::Ck_kk(k, kv, kw) == 0.0) {
          continue;
        }

        // if de ~= 0.0, can just use the w = 0.0 matrix element we have evaluated outside
        if (std::abs(de) <= 0.1) {
          // dQ_screen_Fermi[sk](kv_i, kw_i) = dQ_screen[k];
          continue;
        }

        PiMatrix[sk](kv_i, kw_i) = feyn.dQ_screen_k(k, de, true, true);
        // fill symmetric lower half of matrix
        // dQ_screen_Fermi[sk](kw_i, kv_i) =
        //   dQ_screen_Fermi[sk](kv_i, kw_i); // instead will just only use bottom half
      }
      bar.update();
    }
  }
}

} // namespace MBPT