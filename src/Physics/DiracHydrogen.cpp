#include "DiracHydrogen.hpp"
#include "Maths/Hypergeometric.hpp"
#include <cmath>

namespace DiracHydrogen {

//==============================================================================
namespace Hidden {
// Some helper functions:

double Gamma(double x) { return std::tgamma(x); }

//------------------------------------------------------------------------------
double nn(double n, int kappa, double zeff, double alpha) {
  const auto ak = double(std::abs(kappa));
  return std::sqrt(n * n - 2.0 * (n - ak) * (ak - gamma(kappa, zeff, alpha)));
}

//------------------------------------------------------------------------------
double Norm(double n, int kappa, double zeff, double alpha) {
  // N_{nk}: normalisation prefactor (Methods).
  // Mass-independent: c1 ~ sqrt(mass) and x = 2*lambda_m*r already account for
  // the full mass-scaling, f_m(r) = sqrt(m) f_1(m*r).
  const auto denom =
    nn(n, kappa, zeff, alpha) * Gamma(2.0 * gamma(kappa, zeff, alpha) + 1);
  const auto argGamma = n + double(1 - std::abs(kappa));
  const auto arg1 = zeff * Gamma(2.0 * gamma(kappa, zeff, alpha) + argGamma);
  const auto arg2 =
    2.0 * Gamma(argGamma) * (nn(n, kappa, zeff, alpha) - double(kappa));
  return std::sqrt(arg1 / arg2) / denom;
}

//------------------------------------------------------------------------------
double lambda(double n, int kappa, double zeff, double alpha, double mass) {
  const auto e = enk(n, kappa, zeff, alpha, mass);
  const auto a2 = alpha * alpha;
  return std::sqrt(-e * (2.0 * mass + a2 * e));
}

//------------------------------------------------------------------------------
double x(double r, double n, int kappa, double zeff, double alpha,
         double mass) {
  return r * 2.0 * lambda(n, kappa, zeff, alpha, mass);
}
} // namespace Hidden

//==============================================================================
double enk(double n, int kappa, double zeff, double alpha, double mass) {
  // Stable form for enk (avoids cancellation between O(c^2) terms):
  //   nbar = gamma + n - |kappa|
  //   s    = sqrt(nbar^2 + (alpha*Z)^2)
  //   enk  = -mass*Z^2 / (s * (nbar + s))
  const auto nbar = gamma(kappa, zeff, alpha) + n - double(std::abs(kappa));
  const auto az = alpha * zeff;
  const auto s = std::sqrt(nbar * nbar + az * az);
  return -mass * zeff * zeff / (s * (nbar + s));
}

//==============================================================================
double Enk(double n, int kappa, double zeff, double alpha, double mass) {
  return mass / (alpha * alpha) + enk(n, kappa, zeff, alpha, mass);
}

//==============================================================================
double gamma(int kappa, double zeff, double alpha) {
  return std::sqrt(double(kappa * kappa) - alpha * alpha * zeff * zeff);
}

//==============================================================================
double f(double r, double n, int kappa, double zeff, double alpha,
         double mass) {
  using namespace Hidden;
  const auto xr = x(r, n, kappa, zeff, alpha, mass);
  const auto gam = gamma(kappa, zeff, alpha);
  const auto kmn = double(std::abs(kappa)) - n;
  const auto a2 = alpha * alpha;
  const auto en = enk(n, kappa, zeff, alpha, mass);
  const auto c1 = std::sqrt(2.0 * mass + a2 * en);
  const auto c2 =
    Norm(n, kappa, zeff, alpha) * std::exp(-0.5 * xr) * std::pow(xr, gam);
  const auto d1 = (nn(n, kappa, zeff, alpha) - double(kappa)) *
                  Hypergeometric::H1f1(kmn, 2.0 * gam + 1.0, xr);
  const auto d2 = kmn * Hypergeometric::H1f1(kmn + 1.0, 2.0 * gam + 1.0, xr);
  const auto sk = kappa < 0 ? 1.0 : -1.0;
  return sk * c1 * c2 * (d1 + d2);
}

//==============================================================================
double g(double r, double n, int kappa, double zeff, double alpha,
         double mass) {
  using namespace Hidden;
  const auto xr = x(r, n, kappa, zeff, alpha, mass);
  const auto gam = gamma(kappa, zeff, alpha);
  const auto kmn = double(std::abs(kappa)) - n;
  const auto a2 = alpha * alpha;
  const auto en = enk(n, kappa, zeff, alpha, mass);
  const auto c1 = std::sqrt(-a2 * en);
  const auto c2 =
    Norm(n, kappa, zeff, alpha) * std::exp(-0.5 * xr) * std::pow(xr, gam);
  const auto d1 = (nn(n, kappa, zeff, alpha) - double(kappa)) *
                  Hypergeometric::H1f1(kmn, 2.0 * gam + 1.0, xr);
  const auto d2 = kmn * Hypergeometric::H1f1(kmn + 1.0, 2.0 * gam + 1.0, xr);
  const auto sk = kappa < 0 ? 1.0 : -1.0;
  return -sk * c1 * c2 * (d1 - d2);
}

//==============================================================================
double P_nl(double r, int n, int l, double zeff, double mass) {
  // Mass scaling: P_m(r) = Sqrt[m] * P_1(m*r)
  const auto mr = mass * r;
  const auto rho = 2.0 * zeff * mr / n;
  const auto norm =
    std::pow(2.0 * zeff / n, 1.5) *
    std::sqrt(std::tgamma(n - l) / (2.0 * n * std::tgamma(n + l + 1.0)));
  return std::sqrt(mass) * mr * norm * std::exp(-0.5 * rho) * std::pow(rho, l) *
         std::assoc_laguerre(unsigned(n - l - 1), unsigned(2 * l + 1), rho);
}

//==============================================================================
double gfratio(double r, int kappa, double zeff, double alpha, double en,
               double mass) {
  using namespace Hidden;

  const auto gam = gamma(kappa, zeff, alpha);
  const auto a2 = alpha * alpha;
  const auto absk = double(std::abs(kappa));

  const auto n =
    ((zeff * (mass + en * a2)) / std::sqrt(-(en * (2.0 * mass + en * a2)))) -
    gam + absk;

  const auto xr = r * 2.0 * std::sqrt(-en * (2.0 * mass + en * a2));
  const auto kmn = absk - n;
  const auto c1_f = std::sqrt(2.0 * mass + a2 * en);
  const auto nn = std::sqrt((n * n) - 2.0 * (n - absk) * (absk - gam));
  const auto d1 =
    (nn - double(kappa)) * Hypergeometric::H1f1(kmn, 2.0 * gam + 1.0, xr);
  const auto d2 = kmn * Hypergeometric::H1f1(kmn + 1.0, 2.0 * gam + 1.0, xr);
  const auto ff = c1_f * (d1 + d2);

  const auto c1_g = std::sqrt(-a2 * en);
  // sign factor s_kappa cancels in the ratio g/f
  const auto gg = c1_g * (d2 - d1);
  return gg / ff;
}

} // namespace DiracHydrogen
