#include "DiracContinuum.hpp"
#include "Maths/Hypergeometric.hpp"
#include <cassert>
#include <cmath>
#include <complex>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_coulomb.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_result.h>
#include <iostream>

namespace DiracContinuum {

//==============================================================================
namespace Hidden {
// Some helper functions:

double lngamma_mag(double re, double im) {
  // ln|Gamma(re + i*im)|
  // static: set once only (thread safe)
  [[maybe_unused]] static const auto hndl = gsl_set_error_handler_off();
  gsl_sf_result lnr, arg;
  gsl_sf_lngamma_complex_e(re, im, &lnr, &arg);
  return lnr.val;
}

//------------------------------------------------------------------------------
double gamma_arg(double re, double im) {
  // arg Gamma(re + i*im)
  // static: set once only (thread safe)
  [[maybe_unused]] static const auto hndl = gsl_set_error_handler_off();
  gsl_sf_result lnr, arg;
  gsl_sf_lngamma_complex_e(re, im, &lnr, &arg);
  return arg.val;
}

//------------------------------------------------------------------------------
double nu(double en, double zeff, double alpha, double m) {
  // nu = Z * alpha^2 * E / pe, where alpha^2 * E = m + alpha^2 * en
  return zeff * (m + alpha * alpha * en) / pe(en, alpha, m);
}

//------------------------------------------------------------------------------
double mu(double en, double zeff, double alpha, double m) {
  // mu = Z * m / pe; reduces to nu / (alpha^2 * E) = Z / pe for m = 1 (Methods)
  return zeff * m / pe(en, alpha, m);
}

//------------------------------------------------------------------------------
std::complex<double> eta(double en, int kappa, double zeff, double alpha,
                         double m) {
  // eta = (gamma - i*nu) / (-kappa + i*mu). Pure phase: |eta| = 1
  const std::complex<double> I{0.0, 1.0};
  const auto gam = gamma(kappa, zeff, alpha);
  const auto et = (gam - I * nu(en, zeff, alpha, m)) /
                  (-double(kappa) + I * mu(en, zeff, alpha, m));
  // Im(eta) = (kappa*nu - gamma*mu)/(kappa^2 + mu^2), which analytically has
  // the sign of kappa. For kappa > 0, eta ~ -1 with Im(eta) ~ O(alpha^2):
  // if this falls below rounding noise (very small alpha), eta can land on
  // the wrong side of the sqrt/arg branch cut, flipping the overall sign of
  // f and g. Enforce the correct side:
  return {et.real(), std::copysign(et.imag(), double(kappa))};
}

//------------------------------------------------------------------------------
double Delta(double en, int kappa, double zeff, double alpha, double m) {
  // Asymptotic (Coulomb) phase: Delta = sigma_kappa + (pi*gamma + arg(eta))/2,
  // with sigma_kappa = arg Gamma(gamma + 1 + i*nu)
  const auto gam = gamma(kappa, zeff, alpha);
  const auto sigma = gamma_arg(gam + 1.0, nu(en, zeff, alpha, m));
  const auto theta_eta = std::arg(eta(en, kappa, zeff, alpha, m));
  return sigma + 0.5 * (M_PI * gam + theta_eta);
}

} // namespace Hidden

//==============================================================================
double gamma(int kappa, double zeff, double alpha) {
  const auto az = alpha * zeff;
  return std::sqrt(double(kappa * kappa) - az * az);
}

//==============================================================================
double pe(double en, double alpha, double m) {
  return std::sqrt(en * (2.0 * m + en * alpha * alpha));
}

//==============================================================================
std::pair<double, double> fg(double r, double en, int kappa, double zeff,
                             double alpha, double m) {
  using namespace Hidden;
  if (!available) {
    // Warn once only (fg is typically called in tight loops):
    [[maybe_unused]] static const bool warned = []() {
      std::cerr << "\nWARNING: DiracContinuum::fg requires FLINT, but ampsci "
                   "was compiled without FLINT support. Returning NaN\n";
      return true;
    }();
    assert(false && "Required FLINT to compute continuum states. See docs.");
    return {std::nan(""), std::nan("")};
  }

  const std::complex<double> I{0.0, 1.0};
  const auto p = pe(en, alpha, m);
  const auto gam = gamma(kappa, zeff, alpha);
  const auto v = nu(en, zeff, alpha, m);
  const auto et = eta(en, kappa, zeff, alpha, m);
  const auto x = 2.0 * p * r;

  // Normalisation (Methods):
  //   N = |Gamma(gam+1-i*nu)| / Gamma(2*gam+1) * e^{pi*nu/2} / (2*Sqrt[pi])
  //       * e^{i*pi*gam/2} / Sqrt[eta]
  // |Gamma(gam+1-i*nu)| decays like e^{-pi*nu/2}, so this log-magnitude
  // combination is moderate even at large nu (small en and/or large Z):
  const auto lnN =
    lngamma_mag(gam + 1.0, -v) + 0.5 * M_PI * v - std::lgamma(2.0 * gam + 1.0);
  // The magnitude e^{lnN} is folded into the hypergeometric evaluation (see
  // Hypergeometric::H1f1); only the phase part of N is kept here:
  const auto Nphase =
    std::exp(I * (0.5 * M_PI * gam)) / std::sqrt(et) / (2.0 * std::sqrt(M_PI));

  // Phase factor: phi(r) = (-i)^gam * e^{i*p*r}
  const auto phi = std::exp(I * (p * r - 0.5 * M_PI * gam));

  // M_zeta = e^{lnN} * 1F1(gam + zeta - i*nu, 2*gam + 1, -i*x), zeta = 0, 1
  const auto b = 2.0 * gam + 1.0;
  const std::complex<double> a0{gam, -v};
  const std::complex<double> zz{0.0, -x};
  const auto M0 = Hypergeometric::H1f1(a0, b, zz, lnN);
  const auto M1 = Hypergeometric::H1f1(a0 + 1.0, b, zz, lnN);

  const auto common = Nphase * std::pow(x, gam) * phi;
  const auto ff = std::sqrt(p / en) * common * (et * M1 + M0);
  const auto gg = I * alpha * std::sqrt(en / p) * common * (M0 - et * M1);
  // Exactly real with these phase conventions; discard the tiny imaginary
  // part that remains numerically
  return {ff.real(), gg.real()};
}

//==============================================================================
double f(double r, double en, int kappa, double zeff, double alpha, double m) {
  return fg(r, en, kappa, zeff, alpha, m).first;
}

//==============================================================================
double g(double r, double en, int kappa, double zeff, double alpha, double m) {
  return fg(r, en, kappa, zeff, alpha, m).second;
}

//==============================================================================
double f_asymptotic(double r, double en, int kappa, double zeff, double alpha,
                    double m) {
  using namespace Hidden;
  const auto p = pe(en, alpha, m);
  const auto theta = p * r + nu(en, zeff, alpha, m) * std::log(2.0 * p * r);
  return std::sqrt(p / (M_PI * en)) *
         std::cos(theta - Delta(en, kappa, zeff, alpha, m));
}

//==============================================================================
double g_asymptotic(double r, double en, int kappa, double zeff, double alpha,
                    double m) {
  using namespace Hidden;
  const auto p = pe(en, alpha, m);
  const auto theta = p * r + nu(en, zeff, alpha, m) * std::log(2.0 * p * r);
  return -alpha * std::sqrt(en / (M_PI * p)) *
         std::sin(theta - Delta(en, kappa, zeff, alpha, m));
}

//==============================================================================
double P_el(double r, double en, int l, double zeff, double m) {
  // static: set once only (thread safe)
  [[maybe_unused]] static const auto hndl = gsl_set_error_handler_off();
  const auto p = std::sqrt(2.0 * m * en);
  gsl_sf_result F, Fp, G, Gp;
  double expF, expG;
  const auto status = gsl_sf_coulomb_wave_FG_e(
    -zeff * m / p, p * r, double(l), 0, &F, &Fp, &G, &Gp, &expF, &expG);
  if (status != 0) {
    return 0.0;
  }
  return std::sqrt(2.0 * m / (M_PI * p)) * F.val * std::exp(expF);
}

} // namespace DiracContinuum
