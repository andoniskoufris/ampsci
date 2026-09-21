#include "GreenQED.hpp"
#include "DiracOperator/include.hpp" //For E1 operator
#include "IO/InputBlock.hpp"
#include "MBPT/Feynman.hpp"
#include "MBPT/SpinorMatrix.hpp"
#include "Physics/UnitConv_conversions.hpp"
#include "Wavefunction/Wavefunction.hpp"
#include "fmt/format.hpp"
#include <cassert>
#include <complex>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_legendre.h>
#include <memory>

namespace Module {

double rho(const double &p, const double &E) {
  // dimensionless combination
  // rho = (m^2 - p^2) / m^2
  //     = (m^2 - E^2 + |p|^2) / m^2,
  // where E = e_v is the physical energy and |p| is the magnitude of the 3-momentum
  // note: |p| is an integration variable, and is not on the mass-shell

  // to do things in atomic units, E is assumed already to be in atomic units
  // n.b. m_e = 1 and c = 1/α in a.u.
  //  - E and m_e * c^2 have the same units; this means that E^2 and m^2 must
  //    be added together as e_v^2 + 1/α^2
  //  - E and p * c have the same units (I am letting p be dimensionless); E^2
  //    and p^2 must be added as E^2 + p^2/α^2
  // thus, rho = (α^{-4} - E^2 + p^2 * α^{-2})/α^{-4}
  //           = 1 - α^4 * E^2 + α^2 * p^2
  // note: to compare to Shabaev should add mc^2 to e_v (since ampsci subtracts
  //       rest mass from H), so E = e_v + m_e * c^2 = e_v + 1/α^2 (in a.u.)

  const auto alphan2 = 1.0 / (PhysConst::alpha2);
  const auto alpha4 = PhysConst::alpha2 * PhysConst::alpha2;
  const auto Enu = E + alphan2;
  const auto Enu2 = alpha4 * Enu * Enu;
  const auto pnu = PhysConst::alpha * p;

  return 1.0 - Enu2 + pnu * pnu;
}

//=============================================================================

// could be improved with quadrature?
DiracSpinor FourierTransformF(const DiracSpinor &F,
                              std::shared_ptr<const Grid> pGrid) {
  // initialise Fourier transform to be on momentum space grid
  DiracSpinor FTransform = DiracSpinor(F.n(), F.kappa(), pGrid);

  // lambda for sign
  auto sign = [](const double &x) {
    return x < 0 ? -1.0 : (x > 0 ? 1.0 : 0.0);
  };

  const auto grid = F.grid();
  const auto r = F.grid().r();
  const auto p = pGrid->r();

  const auto l_tilde = F.kappa() < 0 ? F.l() + 1 : F.l() - 1;

  for (auto i = 0ul; i < pGrid->num_points(); i++) {
    // in atomic units, r is in a.u. in which case what r actually is numerically is r/aB
    // in atomic units aB = 1, but we want p * r to be dimensionless. This is only the case if we actually use alpha * p
    const auto p_i = p[i]; // / PhysConst::alpha;

    for (auto j = F.min_pt(); j < F.max_pt(); j++) {
      FTransform.f(i) += r[j] * F.f(j) *
                         SphericalBessel::JL(F.l(), p_i * r[j]) * grid.drdu(j) *
                         grid.du();
      FTransform.g(i) += r[j] * F.g(j) *
                         SphericalBessel::JL(l_tilde, p_i * r[j]) *
                         grid.drdu(j) * grid.du();
    }
    FTransform.f(i) *= 4 * M_PI;
    FTransform.g(i) *= -4 * M_PI * sign(F.kappa());
  }

  return FTransform;
}

//=============================================================================

double FourierTransform_f(const DiracSpinor &F, const double &p) {

  const auto grid = F.grid();
  const auto r = F.grid().r();

  double out = 0.0;

  for (auto j = F.min_pt(); j < F.max_pt(); j++) {
    out += r[j] * F.f(j) * SphericalBessel::JL(F.l(), p * r[j]) * grid.drdu(j) *
           grid.du();
  }
  out *= 4 * M_PI;

  return out;
}

//=============================================================================

double FourierTransform_g(const DiracSpinor &F, const double &p) {

  // lambda for sign
  auto sign = [](const double &x) {
    return x < 0 ? -1.0 : (x > 0 ? 1.0 : 0.0);
  };

  const auto l_tilde = F.kappa() < 0 ? F.l() + 1 : F.l() - 1;

  const auto grid = F.grid();
  const auto r = F.grid().r();

  double out = 0.0;

  for (auto j = F.min_pt(); j < F.max_pt(); j++) {
    out += r[j] * F.g(j) * SphericalBessel::JL(l_tilde, p * r[j]) *
           grid.drdu(j) * grid.du();
  }
  out *= -4 * M_PI * sign(F.kappa());

  return out;
}

//=============================================================================

// could be improved with quadrature?
double p_norm(const DiracSpinor &Fa) {

  const auto pGrid = &Fa.grid();
  const auto p = pGrid->r();

  double out = 0.0;

  for (auto i = Fa.min_pt(); i < Fa.max_pt(); i++) {
    const auto p_i = p[i]; // / PhysConst::alpha;

    out += p_i * p_i * (Fa.f(i) * Fa.f(i) + Fa.g(i) * Fa.g(i)) *
           pGrid->drdu(i) * pGrid->du();
  }
  return out / (8.0 * M_PI * M_PI * M_PI);
}

//=============================================================================

double p_braket(const DiracSpinor &Fa, const DiracSpinor &Fb) {
  const auto pGrid = &Fa.grid();
  const auto p = pGrid->r();

  double out = 0.0;

  const auto [i_min, i_max] = std::pair<std::size_t, std::size_t>{
    std::max(Fa.min_pt(), Fb.min_pt()), std::min(Fa.max_pt(), Fb.max_pt())};

  if (i_min >= i_max) {
    return 0.0;
  }

  for (auto i = i_min; i < i_max; i++) {
    out += p[i] * p[i] * (Fa.f(i) * Fa.f(i) + Fa.g(i) * Fa.g(i)) *
           pGrid->drdu(i) * pGrid->du();
  }
  return out / (8.0 * M_PI * M_PI * M_PI);
}

//=============================================================================

double aTerm(const double &rho, const double &m) {

  return 2.0 * m * (1.0 + 2.0 * (rho / (1.0 - rho)) * log(rho));
}

//=============================================================================

double bTerm(const double &rho) {

  return ((rho - 2.0) / (1.0 - rho)) * (1.0 + (rho / (1.0 - rho)) * log(rho));
}

//=============================================================================
void write_orbitals(const std::string &fname,
                    const std::vector<DiracSpinor> &orbs) {
  if (orbs.empty())
    return;
  const auto &gr = orbs.front().grid();

  std::ofstream of(fname);
  of << "r ";
  for (auto &psi : orbs) {
    // of << "r\'$" << psi.symbol(true) << "$" << "\' ";
    of << "$" << psi.symbol(true) << "$ ";
  }
  of << "\n";

  of << "# f block\n";
  for (std::size_t i = 0; i < gr.num_points(); i++) {
    of << gr.r(i) << " ";
    for (auto &psi : orbs) {
      of << psi.f(i) << " ";
    }
    of << "\n";
  }

  of << "\n# g block\n";
  for (std::size_t i = 0; i < gr.num_points(); i++) {
    of << gr.r(i) << " ";
    for (auto &psi : orbs) {
      of << psi.g(i) << " ";
    }
    of << "\n";
  }

  of.close();
  std::cout << "Orbitals written to file: " << fname << "\n";
}

//=============================================================================

double SE_ZeroPotential(const DiracSpinor &F_p, const double &ev,
                        const double &mec2) {
  const auto pGrid = &F_p.grid();
  const auto p = pGrid->r();
  // to compare to Shabaev, E -> E + m_e * c^2 = E + 1/α^2 (in a.u.)
  const auto En = ev + (1.0 / PhysConst::alpha2);

  const auto f_p = F_p.f();
  const auto g_p = F_p.g();

  double E = 0.0;

  for (auto i = F_p.min_pt(); i < F_p.max_pt(); i++) {
    const auto Rho = rho(p[i], ev);
    const auto a_rho = aTerm(Rho, mec2);
    const auto b_rho = bTerm(Rho);

    //! triple check the sign of the last term
    E += (p[i] * p[i] / PhysConst::alpha2) *
         (a_rho * (f_p[i] * f_p[i] - g_p[i] * g_p[i]) +
          b_rho * (En * (f_p[i] * f_p[i] + g_p[i] * g_p[i]) +
                   2 * (p[i] / PhysConst::alpha) * f_p[i] * g_p[i])) *
         pGrid->drdu(i) * pGrid->du();
  }

  E *= PhysConst::alpha / (32.0 * pow(M_PI, 4));
  // fudge factor (probably a units issue) to get agreement with Shabev paper
  E *= PhysConst::alpha2;

  return E;
}

//=============================================================================

double Feyn_denom(const double &y, const double &ev, const double &q,
                  const double &p, const double &v) {
  const double ev2 = ev * ev;
  const double q2 = q * q;
  // const double p2 = p * p;
  const double ym1 = y - 1.0;

  return ev2 - y * q2 + ym1 * p - y * ym1 * exp(-2.0 * v);
}

//=============================================================================

// double-check that Mathematica gave me the correct expression for this
std::pair<double, double> Feyn_denom_zeros(const double &ev, const double &q,
                                           const double &p, const double &v) {
  const double ev2 = ev * ev;
  const double p2 = p * p;
  const double q2 = q * q;

  std::pair<double, double> out;

  const double sq_root = exp(-4.0 * v) + (p2 - q2) * (p2 - q2) -
                         2.0 * exp(-2.0 * v) * (p2 + q2 - 2.0 * ev2);
  // make sure the zeroes are not in the interval if the discriminant is zero
  // y is being integrated between 0 and 1 so we can safely set the zeros to -1.0
  if (sq_root < 0) {
    out.first = -1.0;
    out.second = -1.0;
    return out;
  }
  const double A = std::abs(exp(2.0 * v) * (p2 - q2 + sq_root));

  // is this really where the zero is?? does not seem right
  if (A == 0.0) {
    out.first = 0.5;
    out.second = out.first;
    return out;
  }

  const auto y1 = (1.0 - A) / 2.0;
  const auto y2 = (1.0 + A) / 2.0;

  out.first = y1;
  out.second = y2;

  return out;
}

//=============================================================================

double Y(const double &y, const double &ev, const double &m, const double &q,
         const double &p, const double &v) {
  const double numerator = m * m + y * q * q + (1.0 - y) * p * p;

  return numerator / Feyn_denom(y, ev, q, p, v);
}

//=============================================================================

double X(const double &y, const double &ev, const double &m, const double &q,
         const double &p, const double &v) {
  const double m2 = m * m;
  const double ev2 = ev * ev;
  const double q2 = q * q;
  const double p2 = p * p;

  const double numerator = m2 + ev2 + y * (1.0 - y) * exp(-2.0 * v);
  const double denominator = m2 + y * q2 + (1.0 - y) * p2;

  return numerator / denominator;
}

//=============================================================================

double C0_u(const double &y, const double &ev, const double &m, const double &q,
            const double &p, const double &v) {
  const double XX = X(y, ev, m, q, p, v);
  const double denom = Feyn_denom(y, ev, q, p, v);

  return -log(XX) / denom;
}

// come back to this
double C0_i(double y, void *params) {

  const double m = ((double *)params)[0];
  const double ev = ((double *)params)[1];
  const double q = ((double *)params)[2];
  const double p = ((double *)params)[3];
  const double v = ((double *)params)[4];

  return C0_u(y, ev, m, q, p, v);
}

//=============================================================================

double C11_u(const double &y, const double &ev, const double &m,
             const double &q, const double &p, const double &v) {
  const double YY = Y(y, ev, m, q, p, v);
  const double XX = X(y, ev, m, q, p, v);
  const double denom = Feyn_denom(y, ev, q, p, v);

  return (1.0 - YY * log(XX)) * y / denom;
}

double C11_i(double y, void *params) {

  const double m = ((double *)params)[0];
  const double ev = ((double *)params)[1];
  const double q = ((double *)params)[2];
  const double p = ((double *)params)[3];
  const double v = ((double *)params)[4];

  return C11_u(y, ev, m, q, p, v);
}

//=============================================================================

double C12_u(const double &y, const double &ev, const double &m,
             const double &q, const double &p, const double &v) {
  const double YY = Y(y, ev, m, q, p, v);
  const double XX = X(y, ev, m, q, p, v);
  const double denom = Feyn_denom(y, ev, q, p, v);

  return (1.0 - YY * log(XX)) * (1.0 - y) / denom;
}

double C12_i(double y, void *params) {

  const double m = ((double *)params)[0];
  const double ev = ((double *)params)[1];
  const double q = ((double *)params)[2];
  const double p = ((double *)params)[3];
  const double v = ((double *)params)[4];

  return C12_u(y, ev, m, q, p, v);
}

//=============================================================================

double C21_u(double y, const double &ev, const double &m, const double &q,
             const double &p, const double &v) {
  const double YY = Y(y, ev, m, q, p, v);
  const double XX = X(y, ev, m, q, p, v);
  const double denom = Feyn_denom(y, ev, q, p, v);

  return (-0.5 + YY * (1.0 - YY * log(XX))) * y * y / denom;
}

double C21_i(double y, void *params) {

  const double m = ((double *)params)[0];
  const double ev = ((double *)params)[1];
  const double q = ((double *)params)[2];
  const double p = ((double *)params)[3];
  const double v = ((double *)params)[4];

  return C21_u(y, ev, m, q, p, v);
}

//=============================================================================

double C22_u(const double &y, const double &ev, const double &m,
             const double &q, const double &p, const double &v) {
  const double YY = Y(y, ev, m, q, p, v);
  const double XX = X(y, ev, m, q, p, v);
  const double denom = Feyn_denom(y, ev, q, p, v);

  return (-0.5 + YY * (1.0 - YY * log(XX))) * (1.0 - y) * (1.0 - y) / denom;
}

double C22_i(double y, void *params) {

  const double m = ((double *)params)[0];
  const double ev = ((double *)params)[1];
  const double q = ((double *)params)[2];
  const double p = ((double *)params)[3];
  const double v = ((double *)params)[4];

  return C22_u(y, ev, m, q, p, v);
}

//=============================================================================

double C23_u(double y, const double &ev, const double &m, const double &q,
             const double &p, const double &v) {
  const double YY = Y(y, ev, m, q, p, v);
  const double XX = X(y, ev, m, q, p, v);
  const double denom = Feyn_denom(y, ev, q, p, v);

  return (-0.5 + YY * (1.0 - YY * log(XX))) * y * (1.0 - y) / denom;
}

double C23_i(double y, void *params) {

  const double m = ((double *)params)[0];
  const double ev = ((double *)params)[1];
  const double q = ((double *)params)[2];
  const double p = ((double *)params)[3];
  const double v = ((double *)params)[4];

  return C23_u(y, ev, m, q, p, v);
}

//=============================================================================

double C24_u(const double &y, const double &m, const double &v) {
  const double k2 = -exp(-2.0 * v);
  const double m2 = m * m;

  // (y^2 * k^2 / m^2) - y * k^2 / m^2 + 1 = [y * (y - 1) * k^2 + m^2] / m^2
  const double x = (y * (y - 1) * k2 + m2) / m2;

  return -log(x);
}

double C24_i(double y, void *params) {

  const double m = ((double *)params)[0];
  const double v = ((double *)params)[4];

  return C24_u(y, m, v);
}

//=============================================================================

double A(const double &q, const double &p, const double &ev, const double &m,
         const double &c0, const double &c11, const double &c12,
         const double &c24, const double &v) {
  const double ev2 = ev * ev;
  const double m2 = m * m;
  // 3-vector dot products denoted with single letters
  const double q2 = q * q;
  const double p2 = p * p;
  const double pq = p * q;
  // 4-vector dot products denoted with double letters
  const double qq2 = ev2 - q2;
  const double pp2 = ev2 - p2;
  const double pp_dot_qq = (2 * ev2 + exp(-2.0 * v) - p2 - q2) / (pq);

  double out = c24 - 2 + qq2 * c11 + pp2 * c12;
  out += 4 * pp_dot_qq * (c0 + c11 + c12) + m2 * (2.0 * c0 + c11 + c12);

  return out;
}

//=============================================================================

inline double B1(const double &c11, const double &c21) {
  return -4.0 * (c11 + c21);
}

//=============================================================================

inline double B2(const double &c0, const double &c11, const double &c12,
                 const double &c23) {
  return -4.0 * (c0 + c11 + c12 + c23);
}

//=============================================================================

inline double C1(const double &c0, const double &c11, const double &c12,
                 const double &c23) {
  return -4.0 * (c0 + c11 + c12 + c23);
}

//=============================================================================

inline double C2(const double &c12, const double &c22) {
  return -4.0 * (c12 + c22);
}

//=============================================================================

inline double D(const double &c0, const double &c11, const double &c12) {
  return 2.0 * (c0 + c11 + c12);
}

//=============================================================================

inline double H1(const double &m, const double &c0, const double &c11) {
  return 4.0 * m * (c0 + 2.0 * c11);
}

//=============================================================================

inline double H2(const double &m, const double &c0, const double &c12) {
  return 4.0 * m * (c0 + 2.0 * c12);
}

//=============================================================================

// takes in a _real_space_ wave function and calculates F1 and F2 i.e. the Fourier components are calculated inside
// should _NOT_ ever be used with Fourier space wave function
double F1(const double &q, const double &q_i, const double &p,
          const double &p_i, const double &a, const double &b1,
          const double &b2, const double &c1, const double &c2, const double &d,
          const double &h1, const double &h2, const DiracSpinor &Fv,
          const double &ev) {
  const double f_q = FourierTransform_f(Fv, q_i);
  const double f_p = FourierTransform_f(Fv, p_i);
  const double g_q = FourierTransform_g(Fv, q_i);
  const double g_p = FourierTransform_g(Fv, p_i);

  double f1 = 0.0;

  // first row
  f1 += a * f_q * f_p + ev * (b1 + b2) * (ev * f_q * f_p + q * g_q * f_p);
  // second row
  f1 += ev * (c1 + c2) * (ev * f_q * f_p + p * f_q * g_p);
  // third row
  f1 += d * (ev * ev * f_q * f_p + ev * q * g_q * f_p + ev * p * f_q * g_p -
             p * q * g_q * g_p);
  // fourth row
  f1 += ev * (h1 + h2) * f_q * f_p;

  return f1;
}

// // overload in case we have the spinor components
// double F1(const double &q, const size_t &q_i, const double &p,
//           const size_t &p_i, const double &a, const double &b1,
//           const double &b2, const double &c1, const double &c2, const double &d,
//           const double &h1, const double &h2, const double &f_q,
//           const double &g_q, const double &f_p, const double &g_p,
//           const double &ev, bool each_iter) {

//   double f1 = 0.0;

//   // first row
//   f1 += a * f_q * f_p + ev * (b1 + b2) * (ev * f_q * f_p + q * g_q * f_p);
//   // second row
//   f1 += ev * (c1 + c2) * (ev * f_q * f_p + p * f_q * g_p);
//   // third row
//   f1 += d * (ev * ev * f_q * f_p + ev * q * g_q * f_p + ev * p * f_q * g_p -
//              p * q * g_q * g_p);
//   // fourth row
//   f1 += ev * (h1 + h2) * f_q * f_p;

//   return f1;
// }

//=============================================================================

// takes in a _real_space_ wave function and calculates F1 and F2
double F2(const double &q, const double &q_i, const double &p,
          const double &p_i, const double &a, const double &b1,
          const double &b2, const double &c1, const double &c2, const double &d,
          const double &h1, const double &h2, const DiracSpinor &Fv,
          const double &ev) {
  const double f_q = FourierTransform_f(Fv, q_i);
  const double f_p = FourierTransform_f(Fv, p_i);
  const double g_q = FourierTransform_g(Fv, q_i);
  const double g_p = FourierTransform_g(Fv, p_i);

  double f2 = 0.0;

  // first row
  f2 += a * g_q * g_p + ev * (b1 + b2) * (ev * g_q * g_p + q * f_q * g_p);
  // second row
  f2 += ev * (c1 + c2) * (ev * g_q * g_p + p * g_q * f_p);
  // third row
  f2 += d * (ev * ev * g_q * g_p + ev * q * f_q * g_p + ev * p * g_q * f_p -
             p * q * f_q * f_p);
  // fourth row
  f2 += -1.0 * ev * (h1 + h2) * g_q * g_p;

  return f2;
}

// // overload in case we have the spinor components
// double F2(const double &q, const double &q_i, const double &p,
//           const double &p_i, const double &a, const double &b1,
//           const double &b2, const double &c1, const double &c2, const double &d,
//           const double &h1, const double &h2, const double &f_q,
//           const double &g_q, const double &f_p, const double &g_p,
//           const double &ev, bool each_iter) {

//   double f2 = 0.0;

//   // first row
//   f2 += a * g_q * g_p + ev * (b1 + b2) * (ev * g_q * g_p + q * f_q * g_p);
//   // second row
//   f2 += ev * (c1 + c2) * (ev * g_q * g_p + p * g_q * f_p);
//   // third row
//   f2 += d * (ev * ev * g_q * g_p + ev * q * f_q * g_p + ev * p * g_q * f_p -
//              p * q * f_q * f_p);
//   // fourth row
//   f2 += -1.0 * ev * (h1 + h2) * g_q * g_p;

//   return f2;
// }

//=============================================================================

// function for integrating with a Gaussian quadrature scheme
// uses fortran QUADPACK QAGS routine
// good for when the function has singularities on the boundary
template <typename F>
std::pair<double, double>
quad_integrate(F func, const std::pair<double, double> &range, void *parameters,
               double epsabs = 1.49e-5, double epsrel = 1.49e-5,
               size_t limit = 2000) {

  gsl_integration_workspace *work = gsl_integration_workspace_alloc(2000);

  gsl_function f;
  f.function = func;
  f.params = parameters;

  double result, error;

  gsl_integration_qags(&f, range.first, range.second, epsabs, epsrel, limit,
                       work, &result, &error);

  std::pair<double, double> out = {result, error};

  gsl_integration_workspace_free(work);

  return out;
}

// overload for when we are integrating a function that happens to take in just {ev, m, q, p, v} (stored in a OnePotentialParams object)
// probably will be useless??
// template <typename F>
// std::pair<double, double>
// quad_integrate(F func, const std::pair<double, double> &range,
//                const v_Params &params, double epsabs = 1.49e-5,
//                double epsrel = 1.49e-5, size_t limit = 2000) {

//   gsl_integration_workspace *work = gsl_integration_workspace_alloc(2000);

//   double parameters[] = {params.ev(), params.m(), params.q(), params.p(),
//                          params.v()};

//   return quad_integrate(func, range, parameters, epsabs, epsrel, limit);
// }

//=============================================================================

// function for integrating a function with a _pair_ of zeros
// will not work for a function with possibly more than two zeros
template <typename F>
std::pair<double, double>
quad_integrate_zeros(F func, const std::pair<double, double> &range,
                     const std::pair<double, double> &zeros, void *parameters,
                     double epsabs = 1.49e-5, double epsrel = 1.49e-5,
                     size_t limit = 2000) {

  gsl_integration_workspace *work = gsl_integration_workspace_alloc(2000);

  gsl_function f;
  f.function = func;
  f.params = parameters;

  const double xmin = range.first;
  const double xmax = range.second;

  const double zero_1 = zeros.first;
  const double zero_2 = zeros.second;

  // goes through and forms a vector with a list of points starting from the initial integration point, to the last integration point and will add any zeros inbetween if they are in the integration range
  std::vector<double> pts;
  pts.push_back(xmin);
  if (zero_1 > xmin && zero_1 < xmax) {
    pts.push_back(zero_1);
  }
  if (zero_2 > xmin && zero_2 < xmax) {
    pts.push_back(zero_2);
  }
  pts.push_back(xmax);

  double result, error;

  gsl_integration_qagp(&f, pts.data(), pts.size(), epsabs, epsrel, limit, work,
                       &result, &error);

  std::pair<double, double> out = {result, error};

  gsl_integration_workspace_free(work);

  return out;
}

// testing the integration function I made
double test_func(double x, void *params) {
  const double a = ((double *)params)[0];
  const double b = ((double *)params)[1];

  return (x - a) / ((x - a) * (x - b));
}

// template <typename F>
// std::pair<double, double>
// quad_integrate_zeros(F func, const std::pair<double, double> &range,
//                      const std::pair<double, double> &zeros, void *params,
//                      double epsabs = 1.49e-5, double epsrel = 1.49e-5,
//                      size_t limit = 2000) {

//   return quad_integrate_zeros(func, range, zeros, params, epsabs, epsrel,
//                               limit);
// }

//=============================================================================

// v_integrand = F_1 * P_l(xi) + F_2 * P_{~l}(xi),
// with xi = (p^2 + q^2 - e^{-2v}) / (2 * p * q)
double v_integrand(double v, void *v_params) {

  // need to calculate F1 and F2 inside this function, as this is the innermost integrand
  // i.e. this function should take in v as the first argument, and then also take in {p,q,p_to_index,m,ev} as parameters

  // cast v_params to v_Params class
  v_Params *params = static_cast<v_Params *>(v_params);

  const auto Fv = params->Fv();
  const auto q = params->q();
  const auto p = params->p();
  const auto ev = params->ev();
  const auto m = params->m();

  // conversion factor from the momentum q to the index q (I seem to need this for the zero potential so I will just have this here too I guess??)
  const auto p_to_index = params->p_to_index();

  const auto l = Fv.l();
  const auto l_tilde = Fv.kappa() < 0 ? l + 1 : l - 1;

  // calculate the F1 and F2 integrals and then form integrand of v integral
  auto OnePIntegrals = OnePotential(Fv, q, p, p_to_index, ev, m, v);

  const double xi = (p * p + q * q - exp(-2.0 * v)) / (2.0 * p * q);

  const double F1 = OnePIntegrals.f1();
  const double F2 = OnePIntegrals.f2();

  return F1 * gsl_sf_legendre_Pl(l, xi) + F2 * gsl_sf_legendre_Pl(l_tilde, xi);
}

//=============================================================================

// g(q, p) = \int_{-ln(p + q)}^{-\ln(|p - q|)}[F_1 * P_l(xi) + F_2 * P_{l_tilde}(xi)]},
double g_qp(double p, void *p_params) {

  // cast p_params (void pointer) to p_Params type
  p_Params *params = static_cast<p_Params *>(p_params);

  const DiracSpinor Fv = params->Fv();
  const double ev = params->ev();
  const double m = params->m();
  const double q = params->q();
  const double p_to_index = params->p_to_index();

  // const double l = Fv.l();
  // const double l_tilde = Fv.kappa() < 0 ? l + 1 : l - 1;

  const auto v_lims =
    std::pair<double, double>(-log(p + q), -log(std::abs(p - q)));

  // form the set of parameters that the v integral needs
  const auto v_params = v_Params(ev, m, q, p, p_to_index, Fv);

  // cast v_params to void pointer to be passed into function for integrating
  void *v_params_void = (void *)&v_params;

  const std::pair<double, double> v_int =
    quad_integrate(v_integrand, v_lims, v_params_void);

  return v_int.first;
}

//=============================================================================

// p_integrand = 2 * g(q, p),
// with g(q, p) = \int_{-ln(p + q)}^{-\ln(|p - q|)}[F_1 * P_l(xi) + F_2 * P_{l_tilde}(xi)]
inline double p_integrand(double p, void *p_params) {
  return 2.0 * g_qp(p, p_params);
}

//=============================================================================

// p_integral = \int_{0}^{q} [2 * g(q, p)] dp,
// with g(q, p) = \int_{-ln(p + q)}^{-\ln(|p - q|)}[F_1 * P_l(xi) + F_2 * P_{l_tilde}(xi)]
double p_integral(double q, void *q_params) {

  // cast q_params (void pointer) to q_Params type
  q_Params *params = static_cast<q_Params *>(q_params);

  const auto p_lims = std::pair<double, double>(0.0, q);

  const auto ev = params->ev();
  const auto m = params->m();
  const auto p_to_index = params->p_to_index();
  const auto Fv = params->Fv();

  const auto p_params = p_Params(ev, m, q, p_to_index, Fv);

  // cast p_params to void pointer to be passed into function for integrating
  void *p_params_void = (void *)&p_params;

  const std::pair<double, double> p_int =
    quad_integrate(p_integrand, p_lims, p_params_void);

  return p_int.first;
}

//=============================================================================

// should have only m, ev and p_to_index as parameters
inline double q_integrand(double q, void *q_params) {
  return p_integral(q, q_params);
}

//=============================================================================

// should have only m and ev as parameters
double q_integral(const DiracSpinor &F_p, void *q_params) {

  // cast q_params (void pointer) to q_Params type
  q_Params *params = static_cast<q_Params *>(q_params);

  const auto p_to_index = params->p_to_index();

  const auto q_lims = std::pair<double, double>(
    double(F_p.min_pt()) / p_to_index, double(F_p.max_pt()) / p_to_index);

  const std::pair<double, double> q_int =
    quad_integrate(q_integrand, q_lims, q_params);

  return q_int.first;
}

//=============================================================================

// function definition for OnePotential constructor
OnePotential::OnePotential(const DiracSpinor &Fv, const double &q,
                           const double &p, const double &p_to_index,
                           const double &ev, const double &m, const double &v)
  : m_c0(0.0),
    m_c11(0.0),
    m_c12(0.0),
    m_c21(0.0),
    m_c22(0.0),
    m_c23(0.0),
    m_c24(0.0),
    m_a(0.0),
    m_b1(0.0),
    m_b2(0.0),
    m_c1(0.0),
    m_c2(0.0),
    m_d(0.0),
    m_h1(0.0),
    m_h2(0.0),
    m_F1(0.0),
    m_F2(0.0) {

  // determine the zeros in the denominator
  const auto zeros = Feyn_denom_zeros(ev, q, p, v);

  const std::pair<double, double> y_lims(0.0, 1.0);

  const double params[] = {m, ev, q, p, v};

  // calculates C_ij integrals
  const std::pair<double, double> c0_int =
    quad_integrate_zeros(C0_i, y_lims, zeros, (void *)params);
  const std::pair<double, double> c11_int =
    quad_integrate_zeros(C11_i, y_lims, zeros, (void *)params);
  const std::pair<double, double> c12_int =
    quad_integrate_zeros(C12_i, y_lims, zeros, (void *)params);
  const std::pair<double, double> c21_int =
    quad_integrate_zeros(C21_i, y_lims, zeros, (void *)params);
  const std::pair<double, double> c22_int =
    quad_integrate_zeros(C22_i, y_lims, zeros, (void *)params);
  const std::pair<double, double> c23_int =
    quad_integrate_zeros(C23_i, y_lims, zeros, (void *)params);
  const std::pair<double, double> c24_int =
    quad_integrate_zeros(C24_i, y_lims, zeros, (void *)params);

  m_c0 = c0_int.first;
  m_c11 = c11_int.first;
  m_c12 = c12_int.first;
  m_c21 = c21_int.first;
  m_c22 = c22_int.first;
  m_c23 = c23_int.first;
  m_c24 = c24_int.first;

  const double q_i = q * p_to_index;
  const double p_i = p * p_to_index;

  // calculates the constants that go into F_1 and F_2
  m_a = A(q, p, ev, m, m_c0, m_c11, m_c12, m_c24, v);
  m_b1 = B1(m_c11, m_c21);
  m_b2 = B2(m_c0, m_c11, m_c12, m_c23);
  m_c1 = C1(m_c0, m_c11, m_c12, m_c23);
  m_c2 = C2(m_c12, m_c22);
  m_d = D(m_c0, m_c11, m_c12);
  m_h1 = H1(m, m_c0, m_c11);
  m_h2 = H2(m, m_c0, m_c12);
  m_F1 =
    F1(q, q_i, p, p_i, m_a, m_b1, m_b2, m_c1, m_c2, m_d, m_h1, m_h2, Fv, ev);
  m_F2 =
    F2(q, q_i, p, p_i, m_a, m_b1, m_b2, m_c1, m_c2, m_d, m_h1, m_h2, Fv, ev);
};

//=============================================================================

double SE_OnePotential(const DiracSpinor &Fv, const DiracSpinor &F_p,
                       const double &ev, const double &mec2,
                       const double &p_to_index) {

  // const double xi_min = -1.0;
  // const double xi_max = 1.0;
  // const double dxi = (xi_max - xi_min) / double(xi_num_points);

  // const auto pGrid = F_p.grid();
  // const auto pr = pGrid.r();
  // const auto dpdu = pGrid.drdu();
  // const auto dp = pGrid.du();
  // const auto dq = dp;

  // std::vector<double> xi_grid(xi_num_points);

  // for (auto i = 0ul; i < xi_num_points; i++) {
  //   xi_grid[i] = xi_min + dxi;
  // }

  // // lambda for sign
  // auto sign = [](const double &x) {
  //   return x < 0 ? -1.0 : (x > 0 ? 1.0 : 0.0);
  // };

  // const auto s_kappa = sign(F_p.kappa()) + 0.001;

  // double out = 0.0;

  // #pragma omp parallel for
  //   for (auto q_i = F_p.min_pt(); q_i < F_p.max_pt(); q_i++) { // q integration
  //     // can multiply this by whatever we want depending on choice of units
  //     const double q = pr[q_i] / PhysConst::alpha;

  //     // v integration
  //     const double v_min = -1.0 * log(std::abs(p - q));
  //     const double v_max = -1.0 * log(p + q);

  //     // can multiply this by whatever we want depending on choice of units
  //     const double p = pr[p_j] / PhysConst::alpha;
  //     const double qpxi_factor =
  //       q * q * p * p / (p * p + q * q - 2.0 * p * q * xi);

  //     OnePotential OnePIntegrals(F_p, q, q_i, p, p_j, xi, ev, mec2, num_y_pts,
  //                                0.05);
  //     const double f1 = OnePIntegrals.f1();
  //     const double f2 = OnePIntegrals.f2();

  //   } // p
  // } // q
  // out += (P_l * out1 + P_lbar * out2) * dxi;
  // std::cout << double(x) / double(xi_num_points) << "\n";

  const auto parameters = q_Params(ev, mec2, p_to_index, Fv);

  // cast parameters to void pointer to be passed into function for integrating
  void *params = (void *)&parameters;

  const double out =
    -(PhysConst::alpha2 / (32.0 * pow(M_PI, 5))) * q_integral(F_p, params);

  return out;
}

//==============================================================================

class test_class {
private:
  int m_int;
  double m_double;
  std::vector<double> m_vec;

public:
  test_class(int in_int, double in_double, const std::vector<double> &in_vec)
    : m_int(in_int), m_double(in_double), m_vec(in_vec) {}

  int getInt_mem() { return m_int; }
  double getDouble_mem() { return m_double; }
  double getSecondVec() { return m_vec[1]; }
};

//==============================================================================

double test_pointers(void *object) {
  // cast q_params to q_Params type
  test_class *test = static_cast<test_class *>(object);

  return test->getSecondVec();
}

//=============================================================================
//=============================================================================

void GreenQED(const IO::InputBlock &input, const Wavefunction &wf) {

  input.check(
    {{"r0", "Minimum r to calculate Green's fn [1.0e-3]"}, //
     {"rmax", "Maximum r to calculate Green's fn [50.0]"},
     {"num_points",
      "Number of radial points for Green's function (used for stride). [150]"},
     {"stride", "Explicitely set stride. Will over-ride num_points if set."},
     {"kappa", "Kappa value for initial test [default: -1]"}});

  // If we are just requesting 'help', don't run module:
  if (input.has_option("help")) {
    return;
  }

  // Determine "sub" grid:
  const auto r0_target = input.get("r0", 1.0e-3);
  const auto rmax_target = input.get("rmax", 50.0);

  const std::size_t i0 = wf.grid().getIndex(r0_target);
  const std::size_t imax = wf.grid().getIndex(rmax_target);
  assert(imax > i0 && "Require rmax > r0");

  const std::size_t in_size = input.get("num_points", 150ul);
  const std::optional<std::size_t> in_stride = input.get<std::size_t>("stride");

  std::size_t size{0}, stride{0};
  if (in_stride.has_value()) {
    assert(*in_stride > 0 && "Cannot have stride of zero");
    stride = *in_stride;
    size = (imax - i0) / stride + 1;
  } else {
    assert(in_size > 1 && "Cannot have num_points <= 1");
    size = in_size;
    stride = std::max(1ul, (imax - i0) / (size - 1));
  }
  assert(size > 1 && stride > 0);

  // actual r0,rmax might be slightly different, due to finite grid, stride
  const auto grid = wf.grid();
  const auto r0 = wf.grid().r(i0);
  const auto rmax = wf.grid().r(i0 + stride * size);
  fmt::print(
    "Grid for Green's function: {:.1e} - {:.1f} with {} points [stride = {}]\n",
    r0, rmax, size, stride);

  // We don't need QPQ (at least for now)
  bool construct_qpq = false;
  bool verbose = false;
  // I think this only affects Sigma anyway:
  bool include_G = true;

  MBPT::Feynman Fy(wf.vHF(), i0, stride, size, {}, 1, include_G, verbose, "",
                   construct_qpq);

  int kappa = input.get("kappa", -1);
  std::complex<double> en{-0.5, 0.1};

  // regular method:
  const auto g1 = Fy.green(kappa, en);

  // Use basis:
  const auto g2 = Fy.green_basis(kappa, en, wf.basis());

  // Solve Dirac eq. with complex energy directly (never checked if this works)
  const auto g3 = Fy.green_v2(kappa, en);

  // Simple quick test:
  // <v|G(e)|v> = sum_n <v|n><v|n> / (e - en) = 1 / (e - ev)

  // ampsci can do: G|v> which returns a DiracSpinor
  // G does not include integration measure, so we must include it
  // If G is complex, returns a pair: G*F = {re(GF), im(GF)}

  std::cout << "\nState, <v|G|>, expected, error\n\n";
  for (const auto &[name, g_ptr] : {std::pair{"Normal Green's fn", &g1},
                                    {"Basis Green's fn", &g2},
                                    {"Complex Schrodinger Green's fn", &g3}}) {

    std::cout << "For " << name << "\n";
    const auto &gt = *g_ptr;
    for (const auto &v : wf.valence()) {

      if (v.kappa() != kappa)
        continue;

      const auto [re_Gv, im_Gv] = gt.drj() * v;
      const std::complex<double> value = {v * re_Gv, v * im_Gv};
      const auto expected = 1.0 / (en - v.en());

      auto eps = std::abs(value - expected) / std::abs(expected);

      std::cout << v << " " << value << " " << expected << " " << eps << "\n";
    }
    std::cout << "\n";
  }

  //===========================================================================
  // Calculating self-energy corrections
  std::cout << std::endl;
  std::cout << "Calculating electron self-energy\n";
  std::cout << "with momentum grid parameters:\n\n";

  // conversion factor from k in whatever units I am doing now to a.u.
  const double p_to_au = 1.0;

  // momentum grid parameters
  // right now these are in some units that I don't know
  const auto p_num_points = 2000;
  const auto p_min = p_to_au * 1.0e-4;
  const auto p_max = p_to_au * 4.0e3; // for U^{91+}, p_max = 4.0e3 is best
  const auto p_b = 4.0;
  const auto p_grid_type = "loglinear";
  const auto p_indu = 0.0; // shouldn't worry about this

  std::cout << "Grid type   = " << p_grid_type << "\n";
  std::cout << "Num. points = " << p_num_points << "\n";
  std::cout << "p minimum   = " << p_min << "\n";
  std::cout << "p maximum   = " << p_max << "\n";
  std::cout << "b           = " << p_b << "\n\n";

  // initialise momentum-space grid
  const auto pGrid = std::make_shared<const Grid>(
    GridParameters{p_num_points, p_min, p_max, p_b, p_grid_type, p_indu});
  const auto p = pGrid->r();

  const auto mec2 = 1.0 / (PhysConst::alpha * PhysConst::alpha);

  std::vector<DiracSpinor> orbs;

  fmt::print("{:<5s} {:>10s} {:>14s} {:>14s} {:>13s} {:>13s}\n", "State",
             "<v|v>", "HF", "\u03A3(0)", "\u03A3(1)", "\u03A3(2)");

  for (const auto &v : wf.valence()) {

    // // if I only want to do the calculations for a particular value of n and l (or any other Q numbers)
    // if (v.n() != 20 || v.kappa() > 0) {
    //   continue;
    // }

    const auto vtild = FourierTransformF(v, pGrid);
    orbs.push_back(vtild);
    const auto vp_norm = p_norm(vtild);
    const auto ev = v.en();
    const auto En = ev + (1.0 / PhysConst::alpha2);

    double E0 = SE_ZeroPotential(vtild, ev, mec2);
    double E1 =
      wf.Znuc() * SE_OnePotential(v, vtild, En, mec2, 1.0 / PhysConst::alpha);
    // double E1 = 0.0;

    fmt::print("{:<5}  {:>+7.6f}  {:>+7.7f}  {:>+7.7f} {:>+7.7f}  {:>+7.7f}  "
               "{:>+7.7f}\n",
               v.shortSymbol(), vp_norm, v.en(), E0, E1, grid.r(v.min_pt()),
               grid.r(v.max_pt() - 1));
  }

  std::cout << std::endl;

  write_orbitals(wf.identity() + "qed.pwf.txt", orbs);

  //! Testing one-potential term
  // std::cout << "Testing the function that finds the zeros of the Feynman parameter denominator:\n"
  // const auto [y1, y2] = Feyn_denom_zeros(-3.0, 3.6, 12.0, 0.27);
  // std::cout << "y1 = " << y1
  //           << " ; D(y1) = " << Feyn_denom(-3.0, y1, 3.6, 12.0, 0.27)
  //           << std::endl;
  // std::cout << "y2 = " << y2
  //           << " ; D(y2) = " << Feyn_denom(-3.0, y2, 3.6, 12.0, 0.27) << "\n";

  // std::cout << "Testing the quadrature integration scheme:\n"
  // const auto a = 5.5;
  // const auto b = 2.5;
  // const auto x1 = -5.5;
  // const auto x2 = -3.0;

  // const auto parameters = OnePotentialParams(a, b, 0.0, 0.0, 0.0);

  // const auto x_lims = std::pair<double, double>(x1, x2);
  // const auto zeros = std::pair<double, double>(a, b);

  // const double expected = log((x2 - b) / (x1 - b));

  // const auto [test_int, test_err] =
  //   quad_integrate_zeros(test_func, x_lims, zeros, parameters);

  // std::cout << "\nTesting the integration function:\n";
  // std::cout << "Expected: " << expected << "\n";
  // std::cout << "Result: " << test_int << "\n";
  // std::cout << "Delta: " << expected - test_int << "\n";
  // std::cout << "eps: " << (expected - test_int) / expected << "\n";

  // std::cout << "#    x    y\n";
  // for (auto )

  //! Testing casting to void pointers and classes and back
  // const auto test_vec = std::vector<double>{2.0, -6.34, 0.983};

  // const auto test_thing = test_class(4, 9.645, test_vec);

  // std::cout << "Testing cast to void pointer\n Expected: " << 9.645
  //           << "\n Result: " << test_pointers((void *)&test_thing);
}

} // namespace Module
