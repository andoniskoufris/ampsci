#include "ComplexDirac.hpp"
#include "AsymptoticSpinor.hpp"
#include "BoundState.hpp"
#include "LinAlg/include.hpp"
#include "Maths/Grid.hpp"
#include "Maths/NumCalc_quadIntegrate.hpp"
#include "Wavefunction/DiracSpinor.hpp"
#include "qip/Maths.hpp"
#include "qip/Vector.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace DiracODE {

using namespace Internal;

//==============================================================================
void regularAtOrigin_C(DiracSpinor &FaR, DiracSpinor &FaI,
                       const std::complex<double> en,
                       const std::vector<double> &v,
                       const std::vector<double> &H_mag, const double alpha) {

  const auto &gr = FaR.grid();
  if (en != 0.0) {
    FaR.en() = en.real();
    FaI.en() = en.imag();
  }
  // Effective real energy with the same decay constant as the complex-energy
  // solution: Re[sqrt(-2en)] = sqrt(2|e_eff|), e_eff = -(|en|+|Re(en)|)/2.
  // (Reduces to en.real() when Im(en)=0; range shrinks as Im(en) grows.)
  const auto e_eff = -0.5 * (std::abs(en) + std::abs(en.real()));
  // 4x usual cALR: the Green's function built from these solutions is not
  // exponentially small along the diagonal r1=r2 (only off it), so a more
  // generous range is required; 4*cALR corresponds to r_c ~ 70/sqrt(2|E|)
  const auto pinf =
    Internal::findPracticalInfinity(e_eff, v, gr.r(), 4.0 * Param::cALR);
  Internal::CDiracDerivative Hd(gr, v, FaR.kappa(), en, alpha, H_mag);

  std::vector<std::complex<double>> f(gr.num_points()), g(gr.num_points());
  Internal::solve_Dirac_outwards_C(f, g, Hd, pinf);

  for (std::size_t i = 0; i < pinf; ++i) {
    FaR.f(i) = f.at(i).real();
    FaI.f(i) = f.at(i).imag();
    FaR.g(i) = g.at(i).real();
    FaI.g(i) = g.at(i).imag();
  }

  FaR.min_pt() = 0;
  FaR.max_pt() = pinf;
  FaI.min_pt() = 0;
  FaI.max_pt() = pinf;
  // for safety: make sure zerod! (I may re-use existing orbitals!)
  FaR.zero_boundaries();
  FaI.zero_boundaries();
}
//==============================================================================
void regularAtInfinity_C(DiracSpinor &FaR, DiracSpinor &FaI,
                         const std::complex<double> en,
                         const std::vector<double> &v,
                         const std::vector<double> &H_mag, const double alpha) {

  const auto &gr = FaR.grid();
  if (en != 0.0) {
    FaR.en() = en.real();
    FaI.en() = en.imag();
  }
  // Same effective-energy range cut as regularAtOrigin_C (see comments there)
  const auto e_eff = -0.5 * (std::abs(en) + std::abs(en.real()));
  const auto pinf =
    Internal::findPracticalInfinity(e_eff, v, gr.r(), 4.0 * Param::cALR);
  Internal::CDiracDerivative Hd(gr, v, FaR.kappa(), en, alpha, H_mag);

  std::vector<std::complex<double>> f(gr.num_points()), g(gr.num_points());
  Internal::solve_Dirac_inwards_C(f, g, Hd, 0, pinf);

  for (std::size_t i = 0; i < pinf; ++i) {
    FaR.f(i) = f.at(i).real();
    FaI.f(i) = f.at(i).imag();
    FaR.g(i) = g.at(i).real();
    FaI.g(i) = g.at(i).imag();
  }

  FaR.min_pt() = 0;
  FaR.max_pt() = pinf;
  FaI.min_pt() = 0;
  FaI.max_pt() = pinf;
  // for safety: make sure zerod! (I may re-use existing orbitals!)
  FaR.zero_boundaries();
  FaI.zero_boundaries();
}

namespace Internal {
//==============================================================================
void solve_Dirac_outwards_C(std::vector<std::complex<double>> &f,
                            std::vector<std::complex<double>> &g,
                            const CDiracDerivative &Hd, std::size_t t_pinf) {

  const auto &r = Hd.pgr->r();
  // const auto &drduor = Hd.pgr->drduor();
  const auto du = Hd.pgr->du();
  const auto &v = *(Hd.v);
  const auto alpha = Hd.alpha;
  const auto ka = Hd.k;
  const auto Z_eff = (-1.0 * v[0] * r[0]);
  const double az0 = Z_eff < 1.0 ? alpha : Z_eff * alpha;
  const auto ka2 = (double)(ka * ka);
  const double ga0 = std::sqrt(ka2 - az0 * az0);
  const auto pinf = t_pinf == 0 ? f.size() : t_pinf;

  // initial wf values

  // Set initial values:
  std::complex<double> f0{0.0}, g0{0.0};
  {
    // Otherwise, use H-like form for f and ratio of f/g
    // f0 is arbitrary, but it's nice to be the correct order-of-magnitude
    const auto g_f_ratio = (ka > 0) ? (ka + ga0) / az0 : az0 / (ka - ga0);
    f0 = 2.0 * std::pow(r[0], ga0);
    g0 = f0 * g_f_ratio;
  }

  AdamsMoulton::ODESolver2D<Param::K_Adams, std::size_t, std::complex<double>>
    ode{du, &Hd};

  ode.solve_initial_K(0, f0, g0);
  for (std::size_t i = 0; i < ode.K_steps(); ++i) {
    f.at(i) = ode.f.at(i);
    g.at(i) = ode.g.at(i);
  }
  for (std::size_t i = ode.K_steps(); i < pinf; ++i) {
    ode.drive(i);
    f.at(i) = ode.last_f();
    g.at(i) = ode.last_g();
  }
}
//==================================================================
void solve_Dirac_inwards_C(std::vector<std::complex<double>> &f,
                           std::vector<std::complex<double>> &g,
                           const CDiracDerivative &Hd, std::size_t nf,
                           std::size_t pinf)
// Program to start the INWARD integration.
// Starts from Pinf, and uses an expansion [WKB approx] to go to (pinf-K_Adams)
// i.e., gets last K_Adams points
// Then, it then call ADAMS-MOULTON, to finish (from num_loops*K_Adams+1
//   to nf = ctp-d_ctp)
{

  // short-cuts
  const auto alpha = Hd.alpha;
  const auto ka = Hd.k;
  const auto en = Hd.en;
  const auto &r = Hd.pgr->r();
  const auto &v = *(Hd.v);
  const auto du = Hd.pgr->du();

  const auto Zeff = -v[pinf - 1] * r[pinf - 1];

  static constexpr std::size_t K_AMO = Param::K_Adams;
  AdamsMoulton::ODESolver2D<K_AMO, std::size_t, std::complex<double>> ode{-du,
                                                                          &Hd};

  ode.S_scale = 0.0;

  // Complex-energy asymptotic start: decaying solution has complex momentum,
  // lambda = sqrt(-2*en*(1 + en*alpha^2/2)); a real-energy start would mix in
  // the growing solution and spoil the whole inward integration
  const auto Rasym = AsymptoticSpinor{ka, Zeff, en, alpha, Param::nx_eps};

  // nb: can use AsymptoticWavefunction for more r values?
  for (std::size_t i0 = pinf - 1, i = 0; i < ode.K_steps(); ++i) {
    const auto [f0, g0] = Rasym.fg(r[i0]);
    ode.f[i] = f0;
    ode.g[i] = g0;
    ode.df[i] = ode.dfdt(f0, g0, i0);
    ode.dg[i] = ode.dgdt(f0, g0, i0);
    ode.t[i] = i0;
    --i0;
  }

  for (std::size_t i = 0; i < ode.K_steps(); ++i) {
    f.at(ode.t.at(i)) = ode.f.at(i);
    g.at(ode.t.at(i)) = ode.g.at(i);
  }
  const auto i_start = ode.last_t();
  for (std::size_t i = i_start - 1; i >= nf; --i) {
    ode.drive(i);
    f.at(i) = ode.last_f();
    g.at(i) = ode.last_g();
    if (i == 0)
      break;
  }
  for (std::size_t i = i_start + ode.K_steps(); i < f.size(); ++i) {
    f.at(i) = 0.0;
    g.at(i) = 0.0;
  }
}

//==============================================================================
CDiracDerivative::CDiracDerivative(const Grid &in_grid,
                                   const std::vector<double> &in_v,
                                   const int in_k,
                                   const std::complex<double> in_en,
                                   const double in_alpha,
                                   const std::vector<double> &V_off_diag)
  : pgr(&in_grid),
    v(&in_v),
    Hmag(V_off_diag.empty() ? nullptr : &V_off_diag),
    k(in_k),
    en(in_en),
    alpha(in_alpha),
    cc(1.0 / in_alpha) {}

std::complex<double> CDiracDerivative::a(std::size_t i) const {
  const auto h_mag = (Hmag == nullptr) ? 0.0 : (*Hmag)[i];
  return (double(-k)) * pgr->drduor(i) + alpha * h_mag * pgr->drdu(i);
}
std::complex<double> CDiracDerivative::b(std::size_t i) const {
  return (alpha * en + 2.0 * cc - alpha * (*v)[i]) * pgr->drdu(i);
}
std::complex<double> CDiracDerivative::c(std::size_t i) const {
  return alpha * ((*v)[i] - en) * pgr->drdu(i);
}
std::complex<double> CDiracDerivative::d(std::size_t i) const { return -a(i); }

} // namespace Internal

} // namespace DiracODE