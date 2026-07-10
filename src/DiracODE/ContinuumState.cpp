#include "DiracODE/ContinuumState.hpp"
#include "DiracODE/BoundState.hpp"
#include "DiracODE/include.hpp"
#include "Maths/Grid.hpp"
#include "Wavefunction/DiracSpinor.hpp"
#include "fmt/color.hpp"
#include <algorithm>
#include <cmath>

namespace DiracODE {
using namespace DiracODE::Internal;

namespace {
// Dirac derivative matrix for a local potential v(r), linearly interpolated
// off the radial grid. Used to solve through the region where the radial
// grid under-resolves the continuum oscillations (fine fixed-step grid).
struct InterpPotentialDerivative
  : AdamsMoulton::DerivativeMatrix<double, double> {

  InterpPotentialDerivative(const Grid &in_gr, const std::vector<double> &in_v,
                            int in_kappa, double in_en, double in_alpha)
    : gr(in_gr),
      pv(in_v),
      kappa(in_kappa),
      en(in_en),
      alpha(in_alpha),
      cc(1.0 / alpha) {}

  const Grid &gr;
  const std::vector<double> &pv;
  int kappa;
  double en, alpha, cc;

  double v(double r) const {
    const auto i0 = gr.getIndex(r);
    const auto i1 = std::min(i0 + 1, gr.num_points() - 1);
    if (i1 == i0) {
      return pv[i0];
    }
    const auto t = (r - gr.r(i0)) / (gr.r(i1) - gr.r(i0));
    return (1.0 - t) * pv[i0] + t * pv[i1];
  }
  double a(double r) const final { return double(-kappa) / r; }
  double b(double r) const final { return alpha * (en - v(r)) + 2.0 * cc; }
  double c(double r) const final { return -alpha * (en - v(r)); }
  double d(double r) const final { return -a(r); }
};
} // namespace

//==============================================================================
void solveContinuum(DiracSpinor &Fa, double en, const std::vector<double> &v,
                    double alpha, const DiracSpinor *const VxFa,
                    const DiracSpinor *const Fa0) {

  Fa.en() = en;
  const auto &gr = Fa.grid();
  const auto num_points = gr.num_points();

  // Rough expression for wavelenth at large r
  // nb: sin(kr + \eta * log(kr)), so not exactly constant
  const double approx_wavelength = 2.0 * M_PI / std::sqrt(2.0 * Fa.en());

  // The solution is only usable where the radial grid resolves the
  // oscillations. The grid spacing dr grows with r, so (for high energy)
  // this bounds the usable radius: solve and store the solution only up to
  // there (max_pt), and zero the remaining tail. For low energies, this is
  // the entire grid. Require:
  //  - at least N_ppw points per wavelength for the stored solution
  //  - a denser N_ppw_norm at the point where the normalisation
  //    integration starts (its accuracy is limited by the accumulated ODE
  //    error at that point)
  const int N_ppw = 10;
  const int N_ppw_norm = 40;
  const auto resolved_until = [&](double dr_max) {
    auto ip = num_points;
    while (ip > 0 && gr.drdu(ip - 1) * gr.du() > dr_max) {
      --ip;
    }
    return ip;
  };
  const auto i_max = resolved_until(approx_wavelength / N_ppw);
  const auto i_norm = resolved_until(approx_wavelength / N_ppw_norm);

  // If even the low-r region is unresolvable, write zeros and return:
  if (i_norm < 2 * Param::K_Adams) {
    Fa *= 0.0;
    Fa.eps() = 1.0;
    return;
  }

  // Solve on regular grid, where it is dense enough (up to i_norm):
  DiracDerivative Hd(Fa.grid(), v, Fa.kappa(), Fa.en(), alpha, {}, VxFa, Fa0);
  solve_Dirac_outwards(Fa.f(), Fa.g(), Hd, i_norm);

  // Values at the start of the normalisation integration (moved to the end
  // of the marginal band below, if there is one):
  auto r_norm = gr.r(i_norm - 1);
  auto f_norm = Fa.f(i_norm - 1);
  auto g_norm = Fa.g(i_norm - 1);

  // Marginal band [i_norm, i_max): the grid resolves the oscillations well
  // enough to store/use the solution (>= N_ppw), but not well enough to
  // integrate the ODE accurately on it (< N_ppw_norm; errors grow quickly,
  // worse for high energy and high kappa). Instead, integrate through this
  // region on a fine fixed-step grid, with the (linearly interpolated)
  // potential, and sample the solution back onto the grid points using
  // cubic Hermite (values + ODE derivatives). The Hermite error,
  // ~(2*pi/N_band)^4/384, sets the pointwise accuracy: ~1e-10 for N=400.
  // nb: exchange (VxFa) and inhomogeneous (Fa0) terms are neglected in the
  // band; it only exists at high energy, where they are negligible.
  if (i_max > i_norm) {
    const auto dr_band = approx_wavelength / 400.0;
    // Start a few grid points early, so the Adams starting block (first
    // K_Adams fine points) lies below the first sampled grid point:
    const auto i_begin = i_norm - 1 - Param::K_Adams;
    InterpPotentialDerivative Hband(gr, v, Fa.kappa(), en, alpha);
    AdamsMoulton::ODESolver2D<Param::K_Adams, double, double> ode{dr_band,
                                                                  &Hband};
    ode.solve_initial_K(gr.r(i_begin), Fa.f(i_begin), Fa.g(i_begin));

    const auto df = [&](double t, double ff, double gg) {
      return Hband.a(t) * ff + Hband.b(t) * gg;
    };
    const auto dg = [&](double t, double ff, double gg) {
      return Hband.c(t) * ff + Hband.d(t) * gg;
    };

    auto t_prev = ode.last_t();
    auto f_prev = ode.last_f();
    auto g_prev = ode.last_g();
    std::size_t i_next = i_norm;
    while (i_next < i_max) {
      ode.drive();
      const auto t = ode.last_t();
      const auto ff = ode.last_f();
      const auto gg = ode.last_g();
      // fill any grid points inside (t_prev, t], cubic Hermite:
      while (i_next < i_max && gr.r(i_next) <= t) {
        const auto h = t - t_prev;
        const auto x = (gr.r(i_next) - t_prev) / h;
        const auto x2 = x * x, x3 = x2 * x;
        const auto h00 = 2.0 * x3 - 3.0 * x2 + 1.0;
        const auto h10 = x3 - 2.0 * x2 + x;
        const auto h01 = -2.0 * x3 + 3.0 * x2;
        const auto h11 = x3 - x2;
        Fa.f()[i_next] = h00 * f_prev + h10 * h * df(t_prev, f_prev, g_prev) +
                         h01 * ff + h11 * h * df(t, ff, gg);
        Fa.g()[i_next] = h00 * g_prev + h10 * h * dg(t_prev, f_prev, g_prev) +
                         h01 * gg + h11 * h * dg(t, ff, gg);
        ++i_next;
      }
      t_prev = t;
      f_prev = ff;
      g_prev = gg;
    }
    // normalisation starts from the band end:
    r_norm = t_prev;
    f_norm = f_prev;
    g_norm = g_prev;
  }

  Fa.max_pt() = i_max;
  std::fill(Fa.f().begin() + long(i_max), Fa.f().end(), 0.0);
  std::fill(Fa.g().begin() + long(i_max), Fa.g().end(), 0.0);

  // Now, normalise the solution.
  // Keep solving ODE outwards from r_norm, on linearly-spaced grid
  // Use "H-like" derivative (local effective charge): assumes the potential
  // is Coulomb-like, and exchange etc. negligable, beyond r_norm.
  // Average the amplitude estimate over full oscillation cycles, extrapolate
  // the cycle means to r -> infinity, and re-scale the wavefunction so the
  // large-r amplitude matches the analytic (energy-normalised) expression

  // Step-size for large-r solution: uses linear grid, with fixed number of
  // points per wavelength (cycle means are sensitive to node location, which
  // is resolved to ~dr^2 by interpolation)
  const auto dr = approx_wavelength / 800.0;

  const auto ztmp = -1.0 * v.at(i_max - 1) * gr.r(i_max - 1);
  const auto Zeff = std::max(1.0, ztmp);

  // Find large-r amplitude:
  const auto [amp, eps_amp] = numerical_f_amplitude(en, Fa.kappa(), alpha, Zeff,
                                                    f_norm, g_norm, r_norm, dr);

  Fa.eps() = eps_amp;

  // Calculate normalisation coeficient, D, and re-scaling factor:
  const auto D = analytic_f_amplitude(en, alpha);
  Fa *= (amp != 0.0 ? (D / amp) : 0.0);
}

//==============================================================================
std::pair<double, double> numerical_f_amplitude(double en, int kappa,
                                                double alpha, double Zeff,
                                                double f_final, double g_final,
                                                double r_final, double dr) {

  // In the asymptotic region (V ~ -Zeff/r) the Dirac ODE reduces to:
  //   f'' = -k^2 f,  k^2 = en * (alpha^2 * en + 2)
  // giving f = A sin(theta), g = A*rho*cos(theta),
  // where rho = sqrt(alpha^2*en / (alpha^2*en + 2)), and
  // theta = kr + nu*ln(2kr) + const is the Coulomb phase.
  // Combining: A^2 = f^2 + c_g^2 * g^2,  c_g^2 = 1 + 2/(alpha^2 * en).
  // This is exact only asymptotically: at finite r the estimator oscillates
  // within each cycle (relative amplitude ~ nu/kr), and its mean approaches
  // the true amplitude algebraically. So:
  //  - average over full cycles, delimited by zero-crossings of f (this
  //    stays phase-exact despite the log-phase, unlike fixed-length windows);
  //    node positions are located to sub-step accuracy by linear
  //    interpolation (cycle means are otherwise noise-limited by the
  //    O(dr) boundary jitter)
  //  - the cycle means converge as A(r) = A_inf + c2/r^2 + c3/r^3 + ...
  //    (cycle-averaging kills the oscillating 1/r term); fit four well
  //    separated cycle means to this form (through c4/r^4) to obtain A_inf
  // Converged when the extrapolated estimates using the full and half
  // integration ranges agree (consecutive estimates share abscissae, so
  // their agreement badly underestimates the remaining common bias).
  const double c_g2 = 1.0 + 2.0 / (alpha * alpha * en);

  DiracContinuumDerivative Heff(Zeff, kappa, en, alpha);
  AdamsMoulton::ODESolver2D<Param::K_Adams, double, double> ode{dr, &Heff};
  ode.solve_initial_K(r_final, f_final, g_final);

  const double eps_target = 1.0e-7;
  const std::size_t min_cycles = 8;
  const std::size_t max_cycles = 256;

  // Hard cap on total steps (guards against node detection failing):
  const double approx_wavelength = 2.0 * M_PI / std::sqrt(2.0 * en);
  const auto steps_per_cycle = std::size_t(approx_wavelength / dr) + 1;
  const auto max_steps = 2 * max_cycles * steps_per_cycle;

  // Cycle means A_j, at cycle centres r_j:
  std::vector<double> Aj, rj;
  Aj.reserve(max_cycles);
  rj.reserve(max_cycles);

  // Extrapolate r -> infinity: fit A(r) = A_inf + c2/r^2 + c3/r^3 + c4/r^4
  // through cycles (2j/5, 3j/5, 4j/5, j). Gaussian elimination with partial
  // pivoting; only A_inf (x[0]) is needed:
  const auto extrapolate = [&](std::size_t j) {
    const std::size_t idx[4] = {(2 * j) / 5, (3 * j) / 5, (4 * j) / 5, j};
    double M[4][5];
    for (int a = 0; a < 4; ++a) {
      const auto r = rj[idx[a]];
      M[a][0] = 1.0;
      M[a][1] = 1.0 / (r * r);
      M[a][2] = M[a][1] / r;
      M[a][3] = M[a][2] / r;
      M[a][4] = Aj[idx[a]];
    }
    for (int c = 0; c < 3; ++c) {
      int pv = c;
      for (int a = c + 1; a < 4; ++a) {
        if (std::abs(M[a][c]) > std::abs(M[pv][c])) {
          pv = a;
        }
      }
      if (pv != c) {
        std::swap_ranges(std::begin(M[c]), std::end(M[c]), std::begin(M[pv]));
      }
      for (int a = c + 1; a < 4; ++a) {
        const auto fac = M[a][c] / M[c][c];
        for (int cc = c; cc < 5; ++cc) {
          M[a][cc] -= fac * M[c][cc];
        }
      }
    }
    double x[4];
    for (int a = 3; a >= 0; --a) {
      double s = M[a][4];
      for (int cc = a + 1; cc < 4; ++cc) {
        s -= M[a][cc] * x[cc];
      }
      x[a] = s / M[a][a];
    }
    return x[0];
  };

  // Extrapolated estimate after each completed cycle (0 until min_cycles):
  std::vector<double> Ej;
  Ej.reserve(max_cycles);

  double amp = 0.0;
  double eps_amp = 1.0;

  // First zero-crossing of f starts the first cycle (discard partial cycle);
  // every second crossing after that completes a cycle. Trapezoid-integrate
  // A over each cycle, locating the cycle boundaries (nodes of f) to
  // sub-step accuracy by linear interpolation:
  bool started = false;
  int crossings = 0;
  double I_cycle = 0.0;
  double len = 0.0;
  double r_start = 0.0;
  double f_prev = ode.last_f();
  double A_prev = 0.0;

  for (std::size_t step = 0; step < max_steps; ++step) {
    ode.drive();
    const double f = ode.last_f();
    const double g = ode.last_g();
    const double A = std::sqrt(f * f + c_g2 * g * g);
    if (started) {
      I_cycle += 0.5 * (A_prev + A) * dr;
      len += dr;
    }
    if (f_prev * f < 0.0) {
      // node at r_node, fraction t through this step:
      const double t = f_prev / (f_prev - f);
      const double A_node = A_prev + t * (A - A_prev);
      const double r_node = ode.last_t() - (1.0 - t) * dr;
      // contribution of this step's segment beyond the node:
      const double I_post = 0.5 * (A_node + A) * (1.0 - t) * dr;
      const double len_post = (1.0 - t) * dr;
      if (!started) {
        started = true;
        I_cycle = I_post;
        len = len_post;
        r_start = r_node;
      } else if (++crossings == 2) {
        Aj.push_back((I_cycle - I_post) / (len - len_post));
        rj.push_back(r_start + 0.5 * (len - len_post));
        I_cycle = I_post;
        len = len_post;
        r_start = r_node;
        crossings = 0;

        const auto j = Aj.size() - 1;
        Ej.push_back(j + 1 >= min_cycles ? extrapolate(j) : 0.0);
        if (Ej[j] != 0.0) {
          amp = Ej[j];
          // compare against the estimate from half the range:
          if (Ej[j / 2] != 0.0) {
            eps_amp =
              std::abs(Ej[j] - Ej[j / 2]) / (0.5 * std::abs(Ej[j] + Ej[j / 2]));
            if (eps_amp < eps_target)
              break;
          }
        }
        if (Aj.size() == max_cycles)
          break;
      }
    }
    f_prev = f;
    A_prev = A;
  }

  if (eps_amp > 1.0e-2)
    amp = 0.0;

  return {amp, eps_amp};
}

//==============================================================================
double analytic_f_amplitude(double en, double alpha) {
  // D = Sqrt[alpha/(pi*eps)] <-- Amplitude of large-r f(r)
  // eps = Sqrt[en/(en+2mc^2)]
  // ceps = c*eps = eps/alpha
  const double al2 = std::pow(alpha, 2);
  const double ceps = std::sqrt(en / (en * al2 + 2.0));
  return 1.0 / std::sqrt(M_PI * ceps);
}

//==============================================================================
double fitQuadratic(double x1, double x2, double x3, double y1, double y2,
                    double y3)
// Takes in three points, and fits them to a quadratic function.
// Returns y-value for vertex of quadratic.
// Used for finding the amplitude of a sine/cosine function, given thee
// points. i.e., will return amplitude of since function. Note: the given 3
// points _MUST_ be close to maximum, otherwise, fit wont work
{
  if (y1 < 0)
    y1 = std::abs(y1);
  if (y2 < 0)
    y2 = std::abs(y2);
  if (y3 < 0)
    y3 = std::abs(y3);

  const auto d = (x1 - x2) * (x1 - x3) * (x2 - x3);
  const auto Ad = x3 * (x2 * (x2 - x3) * y1 + x1 * (-x1 + x3) * y2) +
                  x1 * (x1 - x2) * x2 * y3;
  const auto Bd =
    x3 * x3 * (y1 - y2) + x1 * x1 * (y2 - y3) + x2 * x2 * (-y1 + y3);
  const auto Cd = x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3);
  auto y0 = (Ad / d) - Bd * Bd / (4.0 * Cd * d);

  // Find largest input y:
  const auto ymax = std::max({y1, y2, y3});
  // if (y1 > ymax)
  //   ymax = y1;
  // if (y3 > ymax)
  //   ymax = y3;

  if (ymax > y0)
    y0 = ymax; // y0 can't be less than (y1,y2,y3)

  return y0;
}

} // namespace DiracODE
