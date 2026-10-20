#include "GreenQED.hpp"
#include "DiracOperator/include.hpp" //For E1 operator
#include "IO/InputBlock.hpp"
#include "MBPT/Feynman.hpp"
#include "MBPT/SpinorMatrix.hpp"
#include "Maths/Hypergeometric.hpp"
#include "Modules/Modules.hpp"
#include "Physics/UnitConv_conversions.hpp"
#include "Wavefunction/Wavefunction.hpp"
#include "fmt/format.hpp"
#include <cassert>
#include <chrono>
#include <complex>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_legendre.h>
#include <memory>

namespace Module {

#if defined(AMPSCI_USE_FLINT3) || defined(AMPSCI_USE_FLINT2)
// FLINT >= 3.0 merged Arb in: headers live under flint/ and link with -lflint
// (AMPSCI_USE_FLINT3). FLINT 2.x keeps acb_hypgeom in the separate Arb library:
// top-level headers, link with -lflint-arb (Debian/Ubuntu) or -larb (upstream)
// (AMPSCI_USE_FLINT2); see configure.sh. The acb_*/arb_*/arf_* API used below is
// identical either way.
#ifdef AMPSCI_USE_FLINT2
#include <acb.h>
#include <acb_hypgeom.h>
#else
#include <flint/acb.h>
#include <flint/acb_hypgeom.h>
#endif
#endif

namespace {
const Register r_GreenQED{
  "GreenQED", "Calculates Bohr-Weisskopf effect and hyperfine anomaly",
  &GreenQED};
} // namespace

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

  // Thus,
  //       rho = (α^{-4} - E^2 + p^2 * α^{-2})/α^{-4}
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
                         SphericalBessel::jL(F.l(), p_i * r[j]) * grid.drdu(j) *
                         grid.du();
      FTransform.g(i) += r[j] * F.g(j) *
                         SphericalBessel::jL(l_tilde, p_i * r[j]) *
                         grid.drdu(j) * grid.du();
    }
  }

  using namespace qip::overloads;
  FTransform.f() *= 4 * M_PI;
  FTransform.g() *= -4 * M_PI * sign(F.kappa());

  return FTransform;
}

//=============================================================================

double FourierTransform_f(const DiracSpinor &F, const double &p) {

  const auto grid = F.grid();
  const auto r = F.grid().r();

  double out = 0.0;

  for (auto j = F.min_pt(); j < F.max_pt(); j++) {
    if (j > F.max_pt()) {
      std::cout << "r index out of range. Something is wrong!\n";
    }
    out += r[j] * F.f(j) * SphericalBessel::jL(F.l(), p * r[j]) * grid.drdu(j) *
           grid.du();
  }

  return 4 * M_PI * out;
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
    if (j > F.max_pt()) {
      std::cout << "r index out of range. Something is wrong!\n";
    }
    out += r[j] * F.g(j) * SphericalBessel::jL(l_tilde, p * r[j]) *
           grid.drdu(j) * grid.du();
  }

  return -4 * M_PI * sign(F.kappa()) * out;
}

//=============================================================================

double LinInterp(const std::vector<double> &x, const std::vector<double> &y,
                 const double &x_i) {

  // returns the index before the point p_i
  // The '-1' here ensures we always get a valid index, even if p > p_grid.back()
  const auto it = std::upper_bound(x.begin(), x.end() - 1, x_i);
  const auto x0_i = std::size_t(std::distance(x.begin(), it));

  // if at end of array then just return the end point
  // ensures y is never called beyond its size
  // hacky but whatever
  if (x0_i == x.size() - 1) {
    return y[x0_i];
  }

  const auto x1_i = x0_i + 1;

  const auto x0 = x[x0_i];
  const auto x1 = x[x1_i];

  const auto y0 = y[x0_i];
  const auto y1 = y[x1_i];

  // lineraly interpolates
  return y0 + (x_i - x0) * ((y1 - y0) / (x1 - x0));
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

// function for integrating with a Gaussian quadrature scheme
// uses fortran QUADPACK QAGS routine
// good for when the function has singularities on the boundary
template <typename F>
std::pair<double, double>
quad_integrate(F func, const std::pair<double, double> &range, void *parameters,
               double epsabs = 1.00e-2, double epsrel = 1.00e-2,
               size_t limit = 2000) {

  gsl_integration_workspace *work = gsl_integration_workspace_alloc(2000);

  gsl_function f;
  f.function = func;
  f.params = parameters;

  double result, error;

  // gsl_set_error_handler_off();

  gsl_integration_qags(&f, range.first, range.second, epsabs, epsrel, limit,
                       work, &result, &error);

  std::pair<double, double> out = {result, error};

  gsl_integration_workspace_free(work);

  return out;
}

//=============================================================================

// function for integrating a function with a _pair_ of zeros
// will not work for a function with possibly more than two zeros
template <typename F>
std::pair<double, double>
quad_integrate_zeros(F func, const std::pair<double, double> &range,
                     const std::pair<double, double> &zeros, void *parameters,
                     double epsabs = 1.0e-2, double epsrel = 1.0e-2,
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

  // gsl_set_error_handler_off();

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

//=============================================================================

double Feyn_denom(const double &y, const double &ev, const double &q,
                  const double &p, const double &v) {
  const double ev2 = ev * ev;
  const double q2 = q * q;
  const double p2 = p * p;
  const double ym1 = y - 1.0;

  return ev2 - y * q2 + ym1 * p2 - y * ym1 * exp(-2.0 * v);
}

//=============================================================================

// double-check that Mathematica gave me the correct expression for this
std::pair<double, double> Feyn_denom_zeros(const double &ev, const double &q,
                                           const double &p, const double &v) {
  const double ev2 = ev * ev;
  const double p2 = p * p;
  const double q2 = q * q;

  std::pair<double, double> out;

  const double sq_root = sqrt(exp(-4.0 * v) + (p2 - q2) * (p2 - q2) -
                              2.0 * exp(-2.0 * v) * (p2 + q2 - 2.0 * ev2));
  // make sure the zeroes are not in the interval if the discriminant is zero
  // y is being integrated between 0 and 1 so we can safely set the zeros to -1.0
  if (sq_root < 0) {
    out.first = -1.0;
    out.second = -1.0;
    return out;
  }

  const double A = (1.0 + exp(2.0 * v) * (p2 - q2));
  const double B = exp(2.0 * v) * sq_root;

  // is this really where the zero is?? does not seem right
  if (B == 0.0) {
    out.first = A / 2.0;
    out.second = out.first;
    return out;
  }

  const auto y1 = (A - B) / 2.0;
  const auto y2 = (A + B) / 2.0;

  out.first = y1;
  out.second = y2;

  return out;
}

//=============================================================================

double Y(const double &y, const double &ev, const double &m, const double &q,
         const double &p, const double &v) {
  const double numerator = m * m - ev * ev + y * q * q + (1.0 - y) * p * p;

  return numerator / Feyn_denom(y, ev, q, p, v);
}

//=============================================================================

double X(const double &y, const double &ev, const double &m, const double &q,
         const double &p, const double &v) {
  const double m2 = m * m;
  const double ev2 = ev * ev;
  const double q2 = q * q;
  const double p2 = p * p;

  const double numerator = m2 + y * (1.0 - y) * exp(-2.0 * v);
  const double denominator = m2 - ev2 + y * q2 + (1.0 - y) * p2;

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
  // 4-vector dot products denoted with double letters
  const double qq2 = ev2 - q2;
  const double pp2 = ev2 - p2;
  const double pp_dot_qq = (2.0 * ev2 + exp(-2.0 * v) - p2 - q2) / 2.0;

  double out = c24 - 2.0 + qq2 * c11 + pp2 * c12;
  out += 4.0 * pp_dot_qq * (c0 + c11 + c12) + m2 * (-2.0 * c0 + c11 + c12);

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
double F1(const double &q, const double &q_i, const double &p, const double &a,
          const double &b1, const double &b2, const double &c1,
          const double &c2, const double &d, const double &h1, const double &h2,
          const double &f_p, const double &g_p, const double &ev,
          const DiracSpinor &Fp) {
  const double f_q = Fp.f(size_t(q_i));
  const double g_q = Fp.g(size_t(q_i));

  double f1 = 0.0;

  // A term
  f1 += a * f_q * f_p;
  // B terms
  f1 += ev * (b1 + b2) * (ev * f_q * f_p + q * g_q * f_p);
  // C terms
  f1 += ev * (c1 + c2) * (ev * f_q * f_p + p * f_q * g_p);
  // D term
  f1 += d * (ev * f_q + q * g_q) * (ev * f_p + p * g_p);
  // H terms
  f1 += ev * (h1 + h2) * f_q * f_p;

  return f1;
}

//=============================================================================

// takes in a _real_space_ wave function and calculates F1 and F2
double F2(const double &q, const double &q_i, const double &p, const double &a,
          const double &b1, const double &b2, const double &c1,
          const double &c2, const double &d, const double &h1, const double &h2,
          const double &f_p, const double &g_p, const double &ev,
          const DiracSpinor &Fp) {
  const double f_q = Fp.f(size_t(q_i));
  const double g_q = Fp.g(size_t(q_i));

  double f2 = 0.0;

  // A term
  f2 += a * g_q * g_p;
  // B terms
  f2 += +ev * (b1 + b2) * (ev * g_q * g_p + q * f_q * g_p);
  // C terms
  f2 += ev * (c1 + c2) * (ev * g_q * g_p + p * g_q * f_p);
  // D term
  f2 += d * (ev * g_q + q * f_q) * (ev * g_p + p * f_p);
  // H terms
  f2 += -1.0 * ev * (h1 + h2) * g_q * g_p;

  return f2;
}

//=============================================================================

// function definition for OnePotential constructor
OnePotIntegrals::OnePotIntegrals(const DiracSpinor &Fv, const double &q,
                                 const double &p, const double &p_to_index,
                                 const double &ev, const double &m,
                                 const double &v, const DiracSpinor &Fp)
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

  const std::pair<double, double> y_lims = {0.0, 1.0};

  const double params[] = {m, ev, q, p, v};

  // check if we have zeros in the Feynman parameter integrals
  bool zeros_in_interval;
  if ((0.0 <= zeros.first && zeros.first <= 1.0) ||
      (0.0 <= zeros.second && zeros.second <= 1.0)) {
    zeros_in_interval = true;
  } else {
    zeros_in_interval = false;
  }

  std::pair<double, double> c0_int, c11_int, c12_int, c21_int, c22_int, c23_int;

  // gsl_set_error_handler_off();

  // calculates C_ij integrals
  if (zeros_in_interval) {
    c0_int = quad_integrate_zeros(C0_i, y_lims, zeros, (void *)&params);
    c11_int = quad_integrate_zeros(C11_i, y_lims, zeros, (void *)&params);
    c12_int = quad_integrate_zeros(C12_i, y_lims, zeros, (void *)&params);
    c21_int = quad_integrate_zeros(C21_i, y_lims, zeros, (void *)&params);
    c22_int = quad_integrate_zeros(C22_i, y_lims, zeros, (void *)&params);
    c23_int = quad_integrate_zeros(C23_i, y_lims, zeros, (void *)&params);
  } else {
    c0_int = quad_integrate(C0_i, y_lims, (void *)&params);
    c11_int = quad_integrate(C11_i, y_lims, (void *)&params);
    c12_int = quad_integrate(C12_i, y_lims, (void *)&params);
    c21_int = quad_integrate(C21_i, y_lims, (void *)&params);
    c22_int = quad_integrate(C22_i, y_lims, (void *)&params);
    c23_int = quad_integrate(C23_i, y_lims, (void *)&params);
  }

  const std::pair<double, double> c24_int =
    quad_integrate(C24_i, y_lims, (void *)&params);

  m_c0 = c0_int.first;
  m_c11 = c11_int.first;
  m_c12 = c12_int.first;
  m_c21 = c21_int.first;
  m_c22 = c22_int.first;
  m_c23 = c23_int.first;
  m_c24 = c24_int.first;

  const double q_i = Fp.grid().getIndex(q * p_to_index);
  const double p_i = p * p_to_index;

  // const double f_p = FourierTransform_f(Fv, p_i);
  // const double g_p = FourierTransform_g(Fv, p_i);

  const double f_p = LinInterp(Fp.grid().r(), Fp.f(), p_i);
  const double g_p = LinInterp(Fp.grid().r(), Fp.g(), p_i);

  // // returns the index before the point p_i
  // // The '-1' here ensures we always get a valid index, even if p > p_grid.back()
  // const auto it =
  //   std::lower_bound(Fp.grid().r().begin(), Fp.grid().r().end() - 1, p_i);
  // const auto p_i_lower = std::size_t(std::distance(Fp.grid().r().begin(), it));
  // const auto p_i_upper =
  //   p_i_lower + 1; //! need to make sure this never calls outside the vector

  // // lineraly interpolates the Fourier transform
  // const double f_p = Fp.f(p_i_lower) + double(p_i - p_i_lower) *
  //                                        ((Fp.f(p_i_upper) - Fp.f(p_i_lower)) /
  //                                         double(p_i_upper - p_i_lower));
  // const double g_p = Fp.g(p_i_lower) + double(p_i - p_i_lower) *
  //                                        ((Fp.g(p_i_upper) - Fp.g(p_i_lower)) /
  //                                         double(p_i_upper - p_i_lower));

  // calculates the constants that go into F_1 and F_2
  m_a = A(q, p, ev, m, m_c0, m_c11, m_c12, m_c24, v);
  m_b1 = B1(m_c11, m_c21);
  m_b2 = B2(m_c0, m_c11, m_c12, m_c23);
  m_c1 = C1(m_c0, m_c11, m_c12, m_c23);
  m_c2 = C2(m_c12, m_c22);
  m_d = D(m_c0, m_c11, m_c12);
  m_h1 = H1(m, m_c0, m_c11);
  m_h2 = H2(m, m_c0, m_c12);
  m_F1 = F1(q, q_i, p, m_a, m_b1, m_b2, m_c1, m_c2, m_d, m_h1, m_h2, f_p, g_p,
            ev, Fp);
  m_F2 = F2(q, q_i, p, m_a, m_b1, m_b2, m_c1, m_c2, m_d, m_h1, m_h2, f_p, g_p,
            ev, Fp);
};

//=============================================================================

// v_integrand = F_1 * P_l(xi) + F_2 * P_{~l}(xi),
// with xi = (p^2 + q^2 - e^{-2v}) / (2 * p * q)
double v_integrand(double v, void *v_params) {

  // need to calculate F1 and F2 inside this function, as this is the innermost integrand
  // i.e. this function should take in v as the first argument, and then also take in {p,q,p_to_index,m,ev} as parameters

  // cast v_params to v_Params class
  v_Params *params = static_cast<v_Params *>(v_params);

  const auto Fr = params->Fr();
  const auto q = params->q();
  const auto p = params->p();
  const auto ev = params->ev();
  const auto m = params->m();
  const auto Fp = params->Fp();

  // conversion factor from the momentum q to the index q
  const auto p_to_index = params->p_to_index();

  const auto l = Fr.l();
  const auto l_tilde = Fr.kappa() < 0 ? l + 1 : l - 1;

  // calculate the F1 and F2 integrals and then form integrand of v integral
  auto OnePIntegrals = OnePotIntegrals(Fr, q, p, p_to_index, ev, m, v, Fp);

  const double xi = (p * p + q * q - exp(-2.0 * v)) / (2.0 * p * q);

  const double F1 = OnePIntegrals.f1();
  const double F2 = OnePIntegrals.f2();

  //gsl_set_error_handler_off();

  return F1 * gsl_sf_legendre_Pl(l, xi) + F2 * gsl_sf_legendre_Pl(l_tilde, xi);
}

//=============================================================================

// g(q, p) = p * q * \int_{-ln(p + q)}^{-\ln(|p - q|)}[F_1 * P_l(xi) + F_2 * P_{l_tilde}(xi)]},
double g_qp(double p, void *p_params) {

  // cast p_params (void pointer) to p_Params type
  p_Params *params = static_cast<p_Params *>(p_params);

  const DiracSpinor Fr = params->Fr();
  const double ev = params->ev();
  const double m = params->m();
  const double q = params->q();
  const double p_to_index = params->p_to_index();
  const DiracSpinor Fp = params->Fp();

  // const double l = Fv.l();
  // const double l_tilde = Fv.kappa() < 0 ? l + 1 : l - 1;

  const std::pair<double, double> v_lims = {-log(p + q), -log(std::abs(p - q))};

  // form the set of parameters that the v integral needs
  const auto v_params = v_Params(ev, m, q, p, p_to_index, Fr, Fp);

  // cast v_params to void pointer to be passed into function for integrating
  void *v_params_void = (void *)&v_params;

  //gsl_set_error_handler_off();

  const std::pair<double, double> v_int =
    quad_integrate(v_integrand, v_lims, v_params_void);

  return p * q * v_int.first;
}

//=============================================================================

// p_integrand = 2 * g(q, p),
// with g(q, p) = p * q * \int_{-ln(p + q)}^{-\ln(|p - q|)}[F_1 * P_l(xi) + F_2 * P_{l_tilde}(xi)]
inline double p_integrand(double p, void *p_params) {
  return 2.0 * g_qp(p, p_params);
}

//=============================================================================

// p_integral = \int_{0}^{q} [2 * g(q, p)] dp,
// with g(q, p) = p * q * \int_{-ln(p + q)}^{-\ln(|p - q|)}[F_1 * P_l(xi) + F_2 * P_{l_tilde}(xi)]
double p_integral(double q, void *q_params) {

  // cast q_params (void pointer) to q_Params type
  q_Params *params = static_cast<q_Params *>(q_params);

  const auto ev = params->ev();
  const auto m = params->m();
  const auto p_to_index = params->p_to_index();
  const auto Fr = params->Fr();
  const auto Fp = params->Fp();

  // testing to see if the problem is when p goes up to q
  const std::pair<double, double> p_lims = {0.0, 1.0 * q};

  const auto p_params = p_Params(ev, m, q, p_to_index, Fr, Fp);

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

  // these integration limits are wrong!!
  const std::pair<double, double> q_lims = {double(F_p.min_pt()) / p_to_index,
                                            double(F_p.max_pt()) / p_to_index};

  const std::pair<double, double> q_int =
    quad_integrate(q_integrand, q_lims, q_params);

  return q_int.first;
}

//=============================================================================

double SE_OnePotential(const DiracSpinor &Fr, const DiracSpinor &Fp,
                       const double &ev, const double &mec2,
                       const double &p_to_index) {

  const auto qGrid = Fp.grid();
  const auto q_vec = qGrid.r();
  const auto dqdu = qGrid.drdu();
  const auto du = qGrid.du();

  double out = 0.0;

#pragma omp parallel for reduction(+ : out)
  for (auto i = Fp.min_pt(); i < Fp.max_pt(); i++) {

    const double q = q_vec[i] / p_to_index;

    if (qGrid.getIndex(q * p_to_index) >= Fp.max_pt()) {
      std::cout << "q index out of range. Something is wrong!\n";
    }

    // std::cout << qGrid.getIndex(q * p_to_index) << "\n";

    const auto params = q_Params(ev, mec2, p_to_index, Fr, Fp);

    out += q_integrand(q, (void *)&params) * dqdu[i] * du;
  }

  out *= -(PhysConst::alpha2 / (32.0 * pow(M_PI, 5)));

  // fudge factor
  out *= PhysConst::alpha2;

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

std::vector<double> log_grid(double r0, double rmax, size_t num_points) {

  double u = 0.0;
  const double du = log(rmax / r0) / double(num_points - 1);

  std::vector<double> r(num_points);

  r[0] = r0;

  for (auto i = 1ul; i < num_points; i++) {
    u += du;
    r[i] = r0 * exp(u);
  }

  return r;
}

//=============================================================================

std::vector<double> log_linear_grid(double r0, double rmax, size_t num_points,
                                    const double b) {

  double u = r0 + b * log(r0);
  const double du = (rmax - r0 + b * log(rmax / r0)) / double(num_points - 1);

  std::vector<double> r(num_points);
  r[0] = r0;

  for (auto i = 1ul; i < num_points; i++) {
    u += du;

    double rn = r[i - 1];
    double rnp1 = rn - (u - rn - b * log(rn)) / (-1.0 - (b / rn));

    while (std::abs(rnp1 - rn) > 0.01 * r0) {
      rn = rnp1;
      rnp1 = rn - ((u - rn - b * log(rn)) / (-1.0 - (b / rn)));
    }

    r[i] = rnp1;
  }

  return r;
}

//=============================================================================

std::vector<double> inv_log_linear_grid(double r0, double rmax,
                                        size_t num_points, const double b) {

  double u = b * log(r0 + b);
  const double du = b * log((rmax + b) / (r0 + b)) / double(num_points - 1);

  std::vector<double> r(num_points);
  r[0] = r0;

  for (auto i = 1ul; i < num_points; i++) {
    u += du;
    r[i] = exp(u / b) - b;
  }

  return r;
}

//=============================================================================

std::vector<double> calculate_p_quad(const double &p_min, const double &p_max,
                                     const std::vector<double> &quad_points) {
  const size_t num_points = quad_points.size();
  std::vector<double> p_vec(num_points);
  const double alpha = 0.5 * (p_max - p_min);
  const double beta = 0.5 * (p_max + p_min);

  for (auto i = 0ul; i < num_points; i++) {
    p_vec[i] = alpha * quad_points[i] + beta;
  }

  return p_vec;
}

//=============================================================================

SpinorTransform FTransform_quad(const DiracSpinor &F,
                                const std::vector<double> &p_grid,
                                const std::vector<double> &quad_grid,
                                const std::vector<double> &weights) {

  auto Fp =
    SpinorTransform(F.n(), F.kappa(), F.en(), p_grid, quad_grid, weights);

  const size_t p_num_pts = p_grid.size();

  // lambda for sign
  auto sign = [](const double &x) {
    return x < 0 ? -1.0 : (x > 0 ? 1.0 : 0.0);
  };

  const auto r_grid = F.grid();
  const auto r = F.grid().r();

  const auto l_tilde = F.kappa() < 0 ? F.l() + 1 : F.l() - 1;

  for (auto i = 0ul; i < p_num_pts; i++) {
    const auto p_i = p_grid[i];

    for (auto j = F.min_pt(); j < F.max_pt(); j++) {
      Fp.f(i) += r[j] * F.f(j) * SphericalBessel::jL(F.l(), p_i * r[j]) *
                 r_grid.drdu(j) * r_grid.du();
      Fp.g(i) += r[j] * F.g(j) * SphericalBessel::jL(l_tilde, p_i * r[j]) *
                 r_grid.drdu(j) * r_grid.du();
    }
  }

  using namespace qip::overloads;
  Fp.f() *= 4 * M_PI;
  Fp.g() *= -4 * M_PI * sign(F.kappa());

  return Fp;
}

//=============================================================================

double quad_norm(const SpinorTransform &F) {

  const auto f_p = F.f();
  const auto g_p = F.g();

  const auto p_min = F.p0();
  const auto p_max = F.pmax();

  // constants in conversion from the Gauss-Legendre points to momentum grid points
  const double A = 0.5 * (p_max - p_min);
  const double B = 0.5 * (p_max + p_min);

  const auto weights = F.weights();
  const auto quad_points = F.quad_grid();
  const auto num_points = F.quad_grid().size();

  double out = 0.0;

  for (auto i = 0ul; i < num_points; i++) {
    const auto p_i = A * quad_points[i] + B;

    auto si = std::size_t(i);

    out += p_i * p_i * (F.f(si) * F.f(si) + F.g(si) * F.g(si)) * weights[i];
  }

  // A is the Jacobian from converting the p integral between p_min and p_max into an integral over [-1, 1]
  return A * out / (8.0 * M_PI * M_PI * M_PI);
}

//=============================================================================
void write_orbitals(const std::string &fname,
                    const std::vector<SpinorTransform> &orbs) {
  if (orbs.empty())
    return;
  const auto gr = orbs[0].p_grid();

  std::ofstream of(fname);
  of << "r ";
  for (auto &psi : orbs) {
    // of << "r\'$" << psi.symbol(true) << "$" << "\' ";
    of << "$" << psi.symbol(true) << "$ ";
  }
  of << "\n";

  of << "# f block\n";
  for (std::size_t i = 0; i < gr.size(); i++) {
    of << gr[i] << " ";
    for (auto &psi : orbs) {
      of << psi.f(i) << " ";
    }
    of << "\n";
  }

  of << "\n# g block\n";
  for (std::size_t i = 0; i < gr.size(); i++) {
    of << gr[i] << " ";
    for (auto &psi : orbs) {
      of << psi.g(i) << " ";
    }
    of << "\n";
  }

  of.close();
  std::cout << "Orbitals written to file: " << fname << "\n";
}

//=============================================================================

double SJ_Jp1(int J, int ka, int kb) {
  const auto CJ_nba = Angular::Ck_kk(J, -kb, ka);

  if (J == 0) {
    return CJ_nba;
  }
  if (CJ_nba == 0.0) {
    return 0.0;
  }

  const double factor = sqrt((J + 1.0) / (2.0 * J + 1.0));

  return factor * (1.0 + double(ka + kb) / (J + 1.0)) * CJ_nba;
}

//=============================================================================

double SJ_J(int J, int ka, int kb) {
  if (J == 0) {
    return 0.0;
  }

  if (kb == ka) {
    return 0.0;
  }

  const double CJ_ba = Angular::Ck_kk(J, kb, ka);
  if (CJ_ba == 0.0) {
    return 0.0;
  }

  const double factor = double(ka - kb) / sqrt(J * (J + 1.0));

  return factor * CJ_ba;
}

//=============================================================================

double SJ_Jm1(int J, int ka, int kb) {
  if (J == 0) {
    return 0.0;
  }

  const auto CJ_nba = Angular::Ck_kk(J, -kb, ka);

  if (CJ_nba == 0.0) {
    return 0.0;
  }

  const double factor = sqrt(double(J) / (2.0 * J + 1.0));

  return factor * (-1.0 + double(ka + kb) / double(J)) * CJ_nba;
}

//=============================================================================

// Technically this is energy-dependent through A so assumes that A is at the right energy
std::complex<double> AI_ab(const MBPT::ComplexGMatrix &A, const DiracSpinor &a,
                           const DiracSpinor &b, size_t i, size_t j) {

  const auto si = A.index_to_fullgrid(i);
  const auto sj = A.index_to_fullgrid(j);

  const auto f_Aff_f = a.f(si) * A.ff(i, j) * b.f(sj);
  const auto f_Afg_g = a.f(si) * A.fg(i, j) * b.g(sj);
  const auto g_Agf_f = a.g(si) * A.gf(i, j) * b.f(sj);
  const auto g_Agg_g = a.g(si) * A.gg(i, j) * b.g(sj);

  return f_Aff_f + f_Afg_g + g_Agf_f + g_Agg_g;
}

//=============================================================================

// Technically this is energy-dependent through A so assumes that A is at the right energy
std::complex<double> AII_ab_JL(const MBPT::ComplexGMatrix &A,
                               const DiracSpinor &a, const DiracSpinor &b,
                               int J, int k1, int k2,
                               double (*SJL)(int, int, int), size_t i,
                               size_t j) {

  const auto si = A.index_to_fullgrid(i);
  const auto sj = A.index_to_fullgrid(j);

  const double SJL_na1 = SJL(J, -a.kappa(), k1);
  const double SJL_an1 = SJL(J, a.kappa(), -k1);
  const double SJL_nb2 = SJL(J, -b.kappa(), k2);
  const double SJL_bn2 = SJL(J, b.kappa(), -k2);

  // note opposite spinor components to the AI_ab function
  const auto g_Aff_g = a.g(si) * A.ff(i, j) * b.g(sj);
  const auto g_Afg_f = a.g(si) * A.fg(i, j) * b.f(sj);
  const auto f_Agf_g = a.f(si) * A.gf(i, j) * b.g(sj);
  const auto f_Agg_f = a.f(si) * A.gg(i, j) * b.f(sj);

  auto out = g_Aff_g * SJL_na1 * SJL_nb2;
  out -= g_Afg_f * SJL_na1 * SJL_bn2;
  out -= f_Agf_g * SJL_an1 * SJL_nb2;
  out += f_Agg_f * SJL_an1 * SJL_bn2;

  return out;
}

//==============================================================================

// radial part of Feynman gauge photon propagator
// should really, really test this
MBPT::ComplexRMatrix g_L(int L, std::complex<double> en,
                         const MBPT::Feynman &feyn) {

#if !defined(AMPSCI_USE_FLINT3) && !defined(AMPSCI_USE_FLINT2)

  assert(false && "Requires FLINT to call g_L with complex args. See docs");
  (void)L;
  (void)en;
  (void)feyn;
  MBPT::ComplexRMatrix out(feyn.i0(), feyn.stride(), feyn.num_subgrid_pts(),
                           feyn.grid());
  return out;
#else

  // maybe relax?
  constexpr long prec = 64;

  MBPT::ComplexRMatrix gL(feyn.i0(), feyn.stride(), feyn.num_subgrid_pts(),
                          feyn.grid());

  const auto Iunit = std::complex<double>(0.0, 1.0);

  for (auto i = 0ul; i < feyn.num_subgrid_pts(); i++) {
    for (auto j = 0ul; j <= i; j++) {
      const auto si = gL.index_to_fullgrid(i);
      const auto sj = gL.index_to_fullgrid(j);

      const auto ri = feyn.grid()->r(si);
      const auto rj = feyn.grid()->r(sj);

      acb_t w, JL_le, JL_gr, YL_gr, Lplushalf;
      arb_t r_gr, r_le;
      acb_init(w);
      acb_init(JL_le);
      acb_init(JL_gr);
      acb_init(YL_gr);
      acb_init(Lplushalf);
      arb_init(r_gr);
      arb_init(r_le);

      acb_set_d(Lplushalf, L + 0.5);
      arb_set_d(r_gr, ri);
      arb_set_d(r_le, rj);
      acb_set_d_d(w, en.real(), en.imag());

      acb_t wr_gr, wr_le;
      acb_init(wr_gr);
      acb_init(wr_le);
      acb_mul_arb(wr_gr, w, r_gr, prec);
      acb_mul_arb(wr_le, w, r_le, prec);
      acb_hypgeom_bessel_j(JL_le, Lplushalf, wr_le, prec); // J_{L+1/2}(w*r_<)
      acb_hypgeom_bessel_j(JL_gr, Lplushalf, wr_gr, prec); // J_{L+1/2}(w*r_>)
      acb_hypgeom_bessel_y(YL_gr, Lplushalf, wr_gr, prec); // Y_{L+1/2}(w*r_>)

      arb_mul(r_gr, r_gr, r_le, prec); // r_> -> r_> * r_<
      arb_inv(r_gr, r_gr, prec);       // r_> -> 1 / (r_> * r_<)

      arb_sqrt(r_gr, r_gr, prec); // r_> -> sqrt{1 / (r_> * r_<)}

      acb_mul_onei(YL_gr,
                   YL_gr); // Y_{L+1/2}(w * r_>) -> i * Y_{L+1/2}(w * r_>)
      acb_add(JL_gr, JL_gr, YL_gr,
              prec); // J_{L+1/2}(w * r_>) + i * Y_{L+1/2}(w * r_>)

      acb_t result;
      acb_init(result);

      // result = i * sqrt(1 / (r_> * r_<)) * J_{L+1/2}(w * r_<) * [J_{L+1/2}(w * r_>) + i * Y_{L+1/2}(w * r_>)]
      acb_mul(result, JL_le, JL_gr, prec);
      acb_mul_arb(result, result, r_gr, prec);
      acb_mul_onei(result, result);

      acb_clear(w), acb_clear(JL_le), acb_clear(JL_gr), acb_clear(YL_gr),
        acb_clear(Lplushalf), acb_clear(wr_gr), acb_clear(wr_le),
        acb_clear(result);
      arb_clear(r_gr), arb_clear(r_le);

      const auto re = arf_get_d(arb_midref(acb_realref(result)), ARF_RND_NEAR);
      const auto im = arf_get_d(arb_midref(acb_imagref(result)), ARF_RND_NEAR);

      gL(i, j) = M_PI_2 * std::complex<double>(re, im);
      // fill in symmetric lower half
      gL(j, i) = gL(i, j);
      // for testing
      // gL(i, j) = 1.0;
    }
  }

  return gL;

#endif
}

//=============================================================================

// For the calculation of the many-potential term for the self-energy
// takes in a list of orbs (valence states) and then works out from the max_kappa_index we are interested in including in the sum over kappa in E_many the maximum J we would ever need
std::pair<int, int> J_min_max(const std::vector<DiracSpinor> &orbs,
                              int max_kappa_index) {

  int J_min = orbs.front().twoj() / 2; // running minimum j
  int J_max = J_min;                   // running maximum j

  for (auto &orb : orbs) {
    auto ka = orb.kappa();
    for (auto i = 0; i <= max_kappa_index; i++) {
      const auto ki = Angular::kindex_to_kappa(i);
      // j = |k|-0.5
      // Jmin = |ja-ji| = | |ka| - |ki| |
      // Jmax = ja+jb   = |ka| + |ki| - 1
      const auto aka = std::abs(ka);
      const auto aki = std::abs(ki);
      const auto j_min = std::abs(aka - aki);
      const auto j_max = aka + aki - 1;
      if (j_min < J_min) {
        J_min = j_min;
      }
      if (j_max > J_max) {
        J_max = j_max;
      }
    }
  }

  // the sum over L requires g_{J-1} unless J=0
  if (J_min != 0) {
    J_min -= 1;
  }

  // the maximum J we return is J_max + 1 since the sum over L requires g_{J+1}
  return {J_min, J_max + 1};
}

//=============================================================================

// For the calculation of the many-potential term for the self-energy
// takes in a single valence state and then works out from the max_kappa_index we are interested in including in the sum over kappa in E_many the max J we would need given that kappa value
std::pair<int, int> J_min_max(const DiracSpinor &a, int max_kappa_index) {

  int J_min = a.twoj() / 2; // running minimum j
  int J_max = J_min;        // running maximum j

  auto ka = a.kappa();

  for (auto i = 0; i <= max_kappa_index; i++) {
    const auto ki = Angular::kindex_to_kappa(i);
    // j = |k|-0.5
    // Jmin = |ja-ji| = | |ka| - |ki| |
    // Jmax = ja+jb   = |ka| + |ki| - 1
    const auto aka = std::abs(ka);
    const auto aki = std::abs(ki);
    const auto j_min = std::abs(aka - aki);
    const auto j_max = aka + aki - 1;
    if (j_min < J_min) {
      J_min = j_min;
    }
    if (j_max > J_max) {
      J_max = j_max;
    }
  }

  // the sum over L requires g_{J-1} unless J=0
  if (J_min != 0) {
    J_min -= 1;
  }

  // the maximum J we return is J_max + 1 since the sum over L requires g_{J+1}
  return {J_min, J_max + 1};
}

//=============================================================================

// needs to know about J_min_max to construct the grid of photon propagators
// J_min_max gets worked out outside of this from the list of valence states we want to calculate the corrections for
std::vector<double>
SE_many_highE(const DiracSpinor &Fv, const MBPT::Feynman &feyn,
              const std::vector<std::vector<MBPT::ComplexGMatrix>> &g_2_plus,
              const std::vector<std::vector<MBPT::ComplexRMatrix>> &g_J,
              const std::vector<std::complex<double>> &w_grid,
              const std::vector<std::complex<double>> &w_weights, int num_k,
              int J_min) {

  const auto Iunit = std::complex<double>(0.0, 1.0);

  const auto factor = Iunit * PhysConst::alpha / (2.0 * M_PI * Fv.twojp1());

  using namespace MBPT;

  const auto rgrid = Fv.grid();
  const auto drdu = rgrid.drdu();
  const auto du = rgrid.du() * g_2_plus[0][0].stride();

  // vector for storing the energy for each value of kappa
  std::vector<double> E_k(num_k, 0.0);

  const auto kv = Fv.kappa();

#pragma omp parallel for collapse(2)
  for (int ki = 0; ki < num_k; ki++) {
    for (auto wi = 0ul; wi < w_grid.size(); wi++) {
      const auto g2 = g_2_plus[ki][wi];
      const auto k = Angular::kindex_to_kappa(ki);
      // j = |k|-0.5
      // Jmin = |jv-ji| = | |kv| - |ki| |
      // Jmax = jv+ji   = |kv| + |ki| - 1
      const auto akv = std::abs(kv);
      const auto aki = std::abs(k);
      const auto J0 = std::abs(akv - aki);
      const auto Jmax = akv + aki - 1;
      auto sum_i = std::complex<double>(0.0, 0.0);

      for (auto i = 0ul; i < feyn.num_subgrid_pts(); i++) {
        const auto si = g2.index_to_fullgrid(i);
        const auto ri = rgrid.r(si);
        auto sum_j = std::complex<double>(0.0, 0.0);

        for (auto j = 0ul; j < feyn.num_subgrid_pts(); j++) {

          const auto sj = g2.index_to_fullgrid(j);
          const auto rj = rgrid.r(sj);

          auto J_sum = std::complex<double>(0.0, 0.0);

          for (auto J = J0; J <= Jmax; J += 1) {

            // shift the index that we use to call the photon propagator
            // photon propagator defined on larger vector of J values that ranges from smallest J we will need to the largest J we will need
            const auto Ji = J - J_min;

            // if J = 0 need to avoid calculating the photon propagator at L = J - 1 so treat this case separately
            if (J == 0) {
              const auto CJ_iv = Angular::Ck_kk(J, k, Fv.kappa());
              const auto gJ = g_J[Ji][wi].at(i, j);
              const auto gJp1 = g_J[Ji + 1][wi].at(i, j);
              J_sum += CJ_iv * CJ_iv * gJ * AI_ab(g2, Fv, Fv, i, j) -
                       gJp1 * AII_ab_JL(g2, Fv, Fv, J, k, k, SJ_Jp1, i, j);
              J_sum *= 2.0 * J + 1.0;
              continue;
            }

            const auto CJ_iv = Angular::Ck_kk(J, k, Fv.kappa());
            const auto gJ = g_J[Ji][wi].at(i, j);
            const auto gJm1 = g_J[Ji - 1][wi].at(i, j);
            const auto gJp1 = g_J[Ji + 1][wi].at(i, j);

            J_sum += CJ_iv * CJ_iv * gJ * AI_ab(g2, Fv, Fv, i, j);
            J_sum -= gJm1 * AII_ab_JL(g2, Fv, Fv, J, k, k, SJ_Jm1, i, j);
            J_sum -= gJ * AII_ab_JL(g2, Fv, Fv, J, k, k, SJ_J, i, j);
            J_sum -= gJp1 * AII_ab_JL(g2, Fv, Fv, J, k, k, SJ_Jp1, i, j);
            J_sum *= 2.0 * J + 1.0;
          } // J
          sum_j += J_sum * drdu[sj] * rj * du;
        } // rj
        sum_i += sum_j * drdu[si] * ri * du;
      } // ri
      E_k[ki] +=
        (factor * sum_i * w_weights[wi]).real(); // take the real part only
    } // w
  } // kappa

  return E_k;
}

//=============================================================================

// creates a grid that is linear at small distances (0 < Im(w) < w0) and logarithmic from w0 < Im(w) < wmax
// takes in w_num_pts but will actually create a grid with 2 * num_pts points
// integral is not symmetric in w so need to separately evaluate it from -infty < Im(w) < 0 and 0 < Im(w) < infty
// returns w_grid and integration weights
std::pair<std::vector<std::complex<double>>, std::vector<std::complex<double>>>
create_w_grid(double w_real, double w0, double wmax, int w_num_pts) {
  const auto Iunit = std::complex<double>(0.0, 1.0);

  // bridge the gap between 0 < w < w0 with a single trapezoidal point
  // If w_traps_num is increased then this can be increased but not needed so long as w0 is small enough
  const size_t w_traps_num = 1;
  const auto dw_traps = Iunit * double(w0 / (w_traps_num));
  // number of points for the logarithmic part of the grid
  const size_t w_log_num = w_num_pts - w_traps_num;

  std::vector<std::complex<double>> w_grid(w_num_pts);
  std::vector<std::complex<double>> w_weights(w_num_pts);

  w_grid[0] = std::complex<double>(w_real, 0.0);
  w_weights[0] = dw_traps;

  for (auto i = 1ul; i < w_traps_num; i++) {
    w_grid[i] = w_grid[i - 1] + dw_traps;
    w_weights[i] = dw_traps;
  }

  // logarithmic part of the grid
  const auto du = log(wmax / w0) / double(w_log_num - 1);
  auto u0 = log(w0);
  w_grid[w_traps_num] = std::complex<double>(w_real, w0);
  w_weights[w_traps_num] = Iunit * w0 * du;

  auto u = u0;
  for (auto i = 1ul; i < w_log_num; i++) {
    const auto wi = i + w_traps_num;
    u += du;
    w_grid[wi] = std::complex<double>(w_real, exp(u));
    w_weights[wi] = std::complex<double>(0.0, w_grid[wi].imag() * du);
  }

  // the above code just defined the grid for all positive values
  // now we define it for negative values by flipping the sign of the grid values
  // for the weights, I think I just keep these the same but I'm not actually entirely sure??
  auto added_i = 0;
  for (auto i = 1ul; i < w_num_pts; i++) {
    const auto wi_pos = i + added_i;
    w_grid.emplace(
      w_grid.begin(),
      std::complex<double>(w_grid[wi_pos].real(), -w_grid[wi_pos].imag()));
    w_weights.emplace(w_weights.begin(), w_weights[wi_pos]);
    added_i += 1;
  }

  return {w_grid, w_weights};
}

//=============================================================================

void has_nans(const MBPT::ComplexGMatrix &G) {
  const auto size = G.size();

  for (auto i = 0ul; i < size; i++) {
    for (auto j = 0ul; j < size; j++) {
      if (G.ff(i, j) != G.ff(i, j) || G.fg(i, j) != G.fg(i, j) ||
          G.gf(i, j) != G.gf(i, j) || G.gg(i, j) != G.gg(i, j)) {
        std::cout << "We've hit a nan!" << std::endl;
        break;
      }
    }
  }
}

//=============================================================================

void has_nans(const MBPT::ComplexRMatrix &G) {
  const auto size = G.size();

  for (auto i = 0ul; i < size; i++) {
    for (auto j = 0ul; j < size; j++) {
      if (G(i, j) != G(i, j)) {
        std::cout << "We've hit a nan!" << std::endl;
        break;
      }
    }
  }
}

//=============================================================================
//=============================================================================

void GreenQED(const IO::InputBlock &input, const Wavefunction &wf) {

  input.check(
    {{"r0", "Minimum r to calculate Green's fn [1.0e-3]"},
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
  const auto r0_target = input.get("r0", 1.0e-5);
  const auto rmax_target = input.get("rmax", 2.0);

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

  // // actual r0,rmax might be slightly different, due to finite grid, stride
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

  // int kappa = input.get("kappa", -1);
  // std::complex<double> en{-0.5, 0.1};

  // // regular method:
  // const auto g1 = Fy.green(kappa, en);

  // // Use basis:
  // const auto g2 = Fy.green_basis(kappa, en, wf.basis());

  // // Solve Dirac eq. with complex energy directly (never checked if this works)
  // const auto g3 = Fy.green_v2(kappa, en);

  // // Simple quick test:
  // // <v|G(e)|v> = sum_n <v|n><v|n> / (e - en) = 1 / (e - ev)

  // // ampsci can do: G|v> which returns a DiracSpinor
  // // G does not include integration measure, so we must include it
  // // If G is complex, returns a pair: G*F = {re(GF), im(GF)}

  // std::cout << "\nState, <v|G|>, expected, error\n\n";
  // for (const auto &[name, g_ptr] : {std::pair{"Normal Green's fn", &g1},
  //                                   {"Basis Green's fn", &g2},
  //                                   {"Complex Schrodinger Green's fn", &g3}}) {

  //   std::cout << "For " << name << "\n";
  //   const auto &gt = *g_ptr;
  //   for (const auto &v : wf.valence()) {

  //     if (v.kappa() != kappa)
  //       continue;

  //     const auto [re_Gv, im_Gv] = gt.drj() * v;
  //     const std::complex<double> value = {v * re_Gv, v * im_Gv};
  //     const auto expected = 1.0 / (en - v.en());

  //     auto eps = std::abs(value - expected) / std::abs(expected);

  //     std::cout << v << " " << value << " " << expected << " " << eps << "\n";
  //   }
  //   std::cout << "\n";
  // }

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
  const auto valence = wf.valence();

  // number of kappa to include in sum over k, i.e. max kappa index
  const int num_k = 4;
  // index of maximum kappa (since indexing starts from 0 need to subtract 1)
  const int max_kappa_index = num_k - 1;
  // kappa value corresponding to max kappa index
  const auto k_max = Angular::kindex_to_kappa(max_kappa_index);
  // // finds minimum and maximum J we will ever need based on the rule |ji - jv| <= J <= ji + jv
  // const auto [J_min, J_max] = J_min_max(valence, max_kappa_index);
  // const auto num_J = J_max - J_min + 1;

  //! Create omega grid and weights
  // Re(w) = e0 = 0.9 * e_{1s} (Shabaev's choice)
  // if we want to always have Re(w) be in the midpoint of the core-excited gap we need
  // Re(w) = [ea(highest) + en(lowest)] / 2.
  //
  // If we don't have a core (H-like) then we can use Shabaev's choice since we don't have a core gap but instead have a gap between the lowest excited state and the branch cut from the free electron Green's function at +mc^2 which is far off to the right w >= (1/alpha)^2
  const auto w_real = 0.9 * valence.front().en();

  // defines number of points from zero to -Re(w) + i\infty
  const size_t w_num_pos_pts = 110;
  const auto w0 = 1.0e-4;
  const auto wmax = 20000.0;
  const auto omega_gw = create_w_grid(w_real, w0, wmax, w_num_pos_pts);
  const auto w_grid = omega_gw.first;
  const auto w_weights = omega_gw.second;
  const auto w_num_pts = w_grid.size();

  std::cout << "Calculating the many-potential contribution with:\n";
  std::cout << "num_kappa =  " << num_k << " (kappa_max = " << k_max << ")\n";
  std::cout << "w0        =  " << w0 << "\n";
  std::cout << "w_max     =  " << wmax << "\n";
  std::cout << "w_num_pts =  " << w_num_pts << "\n";
  std::cout << "Re(w)     =  " << "0.9 * e_{1s}" << "\n\n";

  using namespace MBPT;

  auto g_2_plus = std::vector<std::vector<ComplexGMatrix>>(
    num_k, std::vector<ComplexGMatrix>(
             w_num_pts, ComplexGMatrix(Fy.i0(), Fy.stride(),
                                       Fy.num_subgrid_pts(), true, Fy.grid())));

  // Record starting time
  auto start1 = std::chrono::high_resolution_clock::now();
// create g_2 for the full list of kappas we will include in our sum over kappa
#pragma omp parallel for
  for (auto iw = 0ul; iw < w_num_pts; iw++) {
    for (auto ik = 0ul; ik < num_k; ik++) {
      const auto k = Angular::kindex_to_kappa(ik);
      // physical energy is alpha^2 * w (?)
      const auto w = w_grid[iw];
      g_2_plus[ik][iw] = Fy.green_2plus(k, w);
      // has_nans(g_2_plus[ik][iw]);
    }
  }
  // Record ending time
  auto stop1 = std::chrono::high_resolution_clock::now();
  auto duration1 =
    std::chrono::duration_cast<std::chrono::seconds>(stop1 - start1);
  std::cout << "Time taken to construct Green's functions: "
            << duration1.count() << " seconds\n\n";
  std::cout << std::flush;

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

    // Zero- and one-potential contributions
    // double E0 = SE_ZeroPotential(vtild, ev, mec2);
    // double E1 =
    //   wf.Znuc() * SE_OnePotential(v, vtild, En, mec2, PhysConst::alpha);
    // so that I can speed the calculations up
    double E0 = 0.0;
    double E1 = 0.0;

    // Many-potential contributions
    // really need to put all this into a class like Feynman
    const auto Iunit = std::complex<double>(0.0, 1.0);

    //!! could move all of this into the function for Emany_high
    using namespace MBPT;
    // finds minimum and maximum J we will need for this particular v based on the rule |ji - jv| <= J <= ji + jv
    const auto [J_min, J_max] = J_min_max(v, max_kappa_index);
    const auto num_J = J_max - J_min + 1;

    // index first with J then with omega
    // we have one more than num_J since in the sum over L inside the J summation, we will need L = J+1
    // this could be sped up. Here I am calculating the photon propagators for the minimum and maximum J's we will need for any valence state v. But we won't necessarily need all of these. instead we only need the ones for J_min and J_max given that kv is this particular value. Will save time since the photon propagators need to be calculated from scratch for each v anyway due to energy dependence
    auto g_J = std::vector<std::vector<ComplexRMatrix>>(
      num_J, std::vector<ComplexRMatrix>(
               w_num_pts, ComplexRMatrix(Fy.i0(), Fy.stride(),
                                         Fy.num_subgrid_pts(), Fy.grid())));

    // Record starting time
    auto start2 = std::chrono::high_resolution_clock::now();
// fill a table of g_J(ev - w) for each possible J we could need and each frequency w
#pragma omp parallel for collapse(2)
    for (auto ji = 0ul; ji < num_J; ji++) {
      for (auto wi = 0ul; wi < w_num_pts; wi++) {
        const auto J = J_min + ji;
        // evaluate Bessel functions at alpha * |w|, just like Breit (?)
        const auto w = std::abs((v.en() - w_grid[wi]) * PhysConst::alpha);
        g_J[ji][wi] = g_L(J, w, Fy);
      }
    }
    // Record ending time
    auto stop2 = std::chrono::high_resolution_clock::now();
    auto duration2 =
      std::chrono::duration_cast<std::chrono::seconds>(stop2 - start2);
    std::cout << "\nTime taken to construct photon propagators: "
              << duration2.count() << " seconds\n\n";
    std::cout << std::flush;

    std::vector<double> Emany_high =
      SE_many_highE(v, Fy, g_2_plus, g_J, w_grid, w_weights, num_k, J_min);

    fmt::print("{:<5}  {:>+7.6f}  {:>+7.7f}  {:>+7.7f} {:>+7.7f}  {:>+7.7f}  "
               "{:>+7.7f}\n",
               v.shortSymbol(), vp_norm, v.en(), E0, E1, grid.r(v.min_pt()),
               grid.r(v.max_pt() - 1));

    std::cout << "Many-potential contribution: " << std::endl;
    for (int ki = 0; ki < num_k; ki++) {
      const auto k = Angular::kindex_to_kappa(ki);
      std::cout << "k = " << k << "  :  " << Emany_high[ki] << std::endl;
    }
  }

  std::cout << std::endl;

  write_orbitals(wf.identity() + "qed.pwf.txt", orbs);

  //===== Testing things
  //! Testing one-potential term
  // std::cout << "Testing the function that finds the zeros of the Feynman "
  //              "parameter denominator:\n";
  // const auto [y1, y2] = Feyn_denom_zeros(7.3, 9.5, 10.0, -2.8);
  // std::cout << "y1 = " << y1
  //           << " ; D(y1) = " << Feyn_denom(y1, 7.3, 9.5, 10.0, -2.8)
  //           << std::endl;
  // std::cout << "y2 = " << y2
  //           << " ; D(y2) = " << Feyn_denom(y2, 7.3, 9.5, 10.0, -2.8) << "\n";

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

  //==== Testing different grids

  // const double r0 = 1.0e-4;
  // const double rmax = 1.0e3;
  // const size_t num_points = 1000;
  // // const double b = 4;

  // const auto log_r = log_grid(r0, rmax, num_points);
  // const auto log_linear_b1 = log_linear_grid(r0, rmax, num_points, 1);
  // const auto log_linear_b4 = log_linear_grid(r0, rmax, num_points, 4);
  // const auto log_linear_b10 = log_linear_grid(r0, rmax, num_points, 10);

  // const auto inv_log_linear_b1 = inv_log_linear_grid(r0, rmax, num_points, 100);
  // const auto inv_log_linear_b4 = inv_log_linear_grid(r0, rmax, num_points, 400);
  // const auto inv_log_linear_b10 =
  //   inv_log_linear_grid(r0, rmax, num_points, 1000);

  // std::ofstream GridFile("grid_tests.txt");

  // for (auto i = 0ul; i < num_points; i++) {
  //   GridFile << i << " " << log_r[i] << " " << log_linear_b1[i] << " "
  //            << log_linear_b4[i] << " " << log_linear_b10[i] << " "
  //            << inv_log_linear_b1[i] << " " << inv_log_linear_b4[i] << " "
  //            << inv_log_linear_b10[i] << "\n";
  // }

  // GridFile.close();

  //==== test linear interpolation

  // const auto Fp = FourierTransformF(wf.valence()[0], pGrid);

  // const auto p_i = 1.0e2;

  // // returns the index _before_ the point p_i
  // // The '-1' here ensures we always get a valid index, even if p > p_grid.back()
  // const auto it =
  //   std::upper_bound(pGrid->r().begin(), pGrid->r().end() - 1, p_i);
  // const auto p0_i = std::size_t(std::distance(pGrid->r().begin(), it)) - 1;

  // const auto p1_i = p0_i + 1;

  // const double f0 = Fp.f(p0_i);
  // const double f1 = Fp.f(p1_i);
  // const double fp = LinInterp(pGrid->r(), Fp.f(), p_i);

  // std::cout << "\nTesting linear interpolation on " << Fp.shortSymbol()
  //           << " at momentum p=" << p_i << std::endl;

  // std::cout << p[p0_i] << ":   " << f0 << std::endl;
  // std::cout << p_i << ":   " << fp << std::endl;
  // std::cout << p[p1_i] << ":   " << f1 << std::endl;

  //==== Gauss-Legendre quadrature tests
  // auto test_coeffs = QuadCoeffs(50);
  // const auto test_roots = test_coeffs.roots();
  // const auto test_weights = test_coeffs.weights();

  // const auto p_grid = calculate_p_quad(p_min, p_max, test_roots);

  // // std::cout << std::endl << "Roots:  " << std::endl;
  // // for (auto i = 0ul; i < 50ul; i++) {
  // //   std::cout << test_roots[i] << " ";
  // // }
  // // std::cout << "\n";
  // // std::cout << std::endl << "Weights:  " << std::endl;
  // // for (auto i = 0ul; i < 50ul; i++) {
  // //   std::cout << test_weights[i] << " ";
  // // }

  // std::cout << std::endl << "With the quadrature grid:\n";

  // std::vector<SpinorTransform> orbs2;

  // for (const auto &v : wf.valence()) {
  //   const auto Fp = FTransform_quad(v, p_grid, test_roots, test_weights);
  //   const auto norm = quad_norm(Fp);
  //   orbs2.push_back(Fp);

  //   fmt::print("{:<5}  {:>+7.6f}\n", v.shortSymbol(), norm);
  // }

  // write_orbitals(wf.identity() + "qed.pwf.quad.txt", orbs2);

  //===== Testing Whittaker functions
  // std::cout << std::endl << "Testing the Whittaker functions\n";
  // std::cout << "Whittaker function of the first kind\n";

  // for (auto i = 0ul; i < 20; i++) {
  //   if (i == 1) {
  //     continue;
  //   }

  //   const auto xi = 10000.0 * i;

  //   const std::complex<double> z =
  //     std::sqrt(std::complex<double>(1.0 - xi * xi, 0.0));

  //   const auto M_series = Hypergeometric::Whittaker_M_series(-0.5, 3, z);

  //   std::cout << "z = " << z << "  :  "
  //             << Hypergeometric::Whittaker_M(-0.5, 3, z) << "  "
  //             << M_series.first << "  " << M_series.second << std::endl;
  // }

  // std::cout << "\nWhittaker function of the second kind\n";

  // for (auto i = 0ul; i < 20; i++) {
  //   if (i == 1) {
  //     continue;
  //   }

  //   const auto xi = 10000.0 * i;

  //   const std::complex<double> z =
  //     std::sqrt(std::complex<double>(1.0 - xi * xi, 0.0));

  //   std::cout << "z = " << z << "  :  "
  //             << Hypergeometric::Whittaker_W(-0.5, 3, z) << std::endl;
  // }

  // //==================== testing my quadrature grid for omega

  // std::cout << "Testing the omega grid quadrature: size " << w_grid.size()
  //           << "\n\n";

  // for (auto i = 0ul; i < w_grid.size(); i++) {
  //   std::cout << "w = " << w_grid[i] << ",    weight = " << w_weights[i]
  //             << std::endl;
  // }

  //=============== Testing the photon propagator

  //   std::cout << "\nTesting the 2+ Green's function: " << std::endl;
  //   // Record starting time
  //   auto start_photon = std::chrono::high_resolution_clock::now();
  //   // const auto gJ = g_L(0, std::complex<double>(-4861, 0.0), Fy);
  //   const auto g2 = Fy.Feynman::green_2plus(
  //     1, std::complex<double>(-4861.0, PhysConst::alpha * 10000.0));
  //   has_nans(g2);
  //   auto stop_photon = std::chrono::high_resolution_clock::now();
  //   auto duration_photon = std::chrono::duration_cast<std::chrono::seconds>(
  //     stop_photon - start_photon);
  //   std::cout << "Time taken to construct g2+: " << duration_photon.count()
  //             << " seconds\n\n";
  //   std::cout << std::flush;

  //   for (auto i = 0ul; i < int(size / 2); i += 2) {
  //     for (auto j = 0ul; j < int(size / 2); j += 2) {
  //       std::cout << g2.ff(i, j) << " ";
  //     }
  //     std::cout << std::endl;
  //   }

  //   std::cout << "\nTesting the photon propagator: " << std::endl;
  //   // Record starting time
  //   auto start_photon2 = std::chrono::high_resolution_clock::now();
  //   // const auto gJ = g_L(0, std::complex<double>(-4861, 0.0), Fy);
  //   const auto gJ =
  //     g_L(1, std::complex<double>(-4861.0, PhysConst::alpha * 10000.0), Fy);
  //   has_nans(gJ);
  //   auto stop_photon2 = std::chrono::high_resolution_clock::now();
  //   auto duration_photon2 = std::chrono::duration_cast<std::chrono::seconds>(
  //     stop_photon2 - start_photon2);
  //   std::cout << "Time taken to construct gJ: " << duration_photon2.count()
  //             << " seconds\n\n";
  //   std::cout << std::flush;

  //   for (auto i = 0ul; i < int(size / 2); i += 2) {
  //     for (auto j = 0ul; j < int(size / 2); j += 2) {
  //       std::cout << gJ(i, j) << " ";
  //     }
  //     std::cout << std::endl;
  //   }

  //   std::cout << "\n\n\n"
  //             << "ri = rj = " << Fy.grid()->r(g2.index_to_fullgrid(size - 4));
  //   std::cout << "Photon g_J   : " << gJ(size - 4, size - 4) << std::endl;
  //   std::cout << "Electron g_2 : " << g2.ff(size - 4, size - 4) << std::endl;
}

} // namespace Module
