#pragma once
#include "AdamsMoulton.hpp"
#include "Physics/PhysConst_constants.hpp"
#include <utility>
#include <vector>
class DiracSpinor;
class Grid;

namespace DiracODE {

/*!
  @brief Solves Dirac equation for a continuum state (en > 0) with energy normalisation.
  @details
  Normalisation is achieved by continuing the ODE integration to very large r and
  comparing the asymptotic amplitude to that of the analytic solution.
  Only the solution on the regular grid is kept; the extended part is discarded.
  The solution is solved and stored only up to the radius where the grid
  resolves the oscillations (at least ~10 points per wavelength); Fa.max_pt()
  is set accordingly and the tail is zeroed. For high energies this may be
  well inside the grid; for low energies it is the entire grid.
  @param Fa     Output spinor (result stored here).
  @param en     Continuum energy (must be > 0).
  @param v      Local potential v(r).
  @param alpha  Fine-structure constant.
  @param VxFa   Optional exchange potential. If nullptr, ignored.
  @param Fa0    Optional inhomogeneous source spinor. If nullptr, ignored.
  @param average_tail  Optionally fill the unresolved (zeroed) tail with a
  local average (see averageTail()).
*/
void solveContinuum(DiracSpinor &Fa, double en, const std::vector<double> &v,
                    double alpha, const DiracSpinor *const VxFa = nullptr,
                    const DiracSpinor *const Fa0 = nullptr,
                    bool average_tail = false);

//! Grid requirements returned by RequiredContinuumGrid
struct GridRequirements {
  //! num_points required, keeping r0, rmax, and b unchanged
  std::size_t num_points;
  //! largest sufficient b, keeping num_points unchanged (clamped to >= 0.05)
  double b;
  //! num_points required at the returned b (== current num_points, unless
  //! b was clamped)
  std::size_t num_points_b;
};

/*!
  @brief Grid parameters required to safely store (pointwise) a continuum state of energy en on the entire grid.
  @details
  A continuum state is stored pointwise only where the grid spacing gives
  at least N_ppw points per wavelength; beyond that solveContinuum() zeroes
  the solution (see Fa.max_pt()). This checks the grid against the largest
  spacing (the last point) - the constraint is always at large r, where
  the grid is coarsest. Each returned value assumes the other grid
  parameters are unchanged. The b formula assumes a loglinear grid; b is
  clamped from below at 0.05 (below that, even a nearly-linear grid is too
  coarse) - when clamped, num_points_b (the num_points required at the
  clamped b) will exceed the current num_points.
  Uses the relativistic wavelength, 2*pi/k with k^2 = en*(2 + alpha^2*en);
  at high energy this is much shorter than the non-relativistic estimate
  (e.g., 1.9x shorter at en = 1e5 au).
  @param en     Continuum energy (should be the largest energy required).
  @param gr     The radial grid.
  @param N_ppw  Required points per wavelength (default 20, as used by
  solveContinuum).
  @param alpha  Fine-structure constant.
*/
GridRequirements RequiredContinuumGrid(double en, const Grid &gr,
                                       double N_ppw = 20.0,
                                       double alpha = PhysConst::alpha);

/*!
  @brief Optionally fills the zeroed high-r tail of a continuum state with a locally-averaged solution, so radial integrals against smooth functions remain accurate.
  @details
  solveContinuum() stores the solution only up to the radius where the grid
  resolves the oscillations (Fa.max_pt()), and zeroes the rest; radial
  integrals then silently omit any contribution from beyond that radius.
  This routine replaces the tail with the local Gaussian average of a
  finely-integrated solution. The average is smooth enough to store on the
  coarse grid, and preserves radial integrals against functions that are
  smooth on the oscillation scale (bound orbitals, r^k, ...), since
  Int[B f_avg] = Int[B_avg f] ~ Int[B f]. It does not preserve integrals
  against co-factors that oscillate on a comparable scale (e.g. jL(qr)
  with q ~ k), and the stored tail is a local average, not the pointwise
  wavefunction. Exchange is neglected in the tail. Sets Fa.max_pt() to
  num_points.
  @param Fa     Continuum state from solveContinuum() (modified in place).
  @param v      Local potential v(r) (as used to solve Fa).
  @param alpha  Fine-structure constant.
  @return Index of the first averaged point; num_points if nothing was done.
*/
std::size_t averageTail(DiracSpinor &Fa, const std::vector<double> &v,
                        double alpha);

/*!
  @brief Analytic amplitude of f(r) at very large r for an H-like Dirac continuum state.
  @param en     Continuum energy.
  @param alpha  Fine-structure constant.
  @return Analytic asymptotic amplitude.
*/
double analytic_f_amplitude(double en, double alpha);

/*!
  @brief Finds the numerical amplitude of f(r) for a continuum Dirac solution at large r.
  @details
  Continues ODE integration beyond the regular grid, assuming an H-like
  potential (-Zeff/r) and a linearly-spaced extension grid with step dr.
  The amplitude estimate sqrt(f^2 + c_g^2 g^2) is averaged over full
  oscillation cycles (delimited by zero-crossings of f, located to sub-step
  accuracy), and the cycle means are extrapolated to r -> infinity by
  fitting A_inf + c2/r^2 + c3/r^3 + c4/r^4. Converged when the extrapolated
  estimates from the full and half integration ranges agree.
  @param en       Continuum energy.
  @param kappa    Orbital kappa quantum number.
  @param alpha    Fine-structure constant.
  @param Zeff     Effective nuclear charge.
  @param f_final  Value of f at the end of the regular grid.
  @param g_final  Value of g at the end of the regular grid.
  @param r_final  Radial position at the end of the regular grid.
  @param dr       Step size for the extended linear grid.
  @return {amplitude, eps} - extrapolated asymptotic amplitude of f(r), and
  relative difference between the last two estimates (convergence measure).
*/
std::pair<double, double> numerical_f_amplitude(double en, int kappa,
                                                double alpha, double Zeff,
                                                double f_final, double g_final,
                                                double r_final, double dr);

/*!
  @brief H-like Dirac derivative matrix for continuum states at large r.
  @details
  Implements AdamsMoulton::DerivativeMatrix<double, double>, using r directly
  as the argument type. Valid for H-like potential (-Zeff/r); used to extend
  continuum integration beyond the regular grid for normalisation.
  @note Non-copyable.
*/
struct DiracContinuumDerivative
  : AdamsMoulton::DerivativeMatrix<double, double> {

  /*!
    @brief Constructs the H-like continuum derivative matrix.
    @param in_Zeff   Effective nuclear charge.
    @param in_kappa  Orbital kappa quantum number.
    @param in_en     Continuum energy.
    @param in_alpha  Fine-structure constant.
  */
  DiracContinuumDerivative(double in_Zeff, const int in_kappa,
                           const double in_en, const double in_alpha)
    : Zeff(in_Zeff),
      kappa(in_kappa),
      en(in_en),
      alpha(in_alpha),
      cc(1.0 / alpha) {}

  double Zeff = 1.0;
  int kappa;
  double en, alpha, cc;

  double a(double r) const final { return double(-kappa) / r; }
  double b(double r) const final {
    return (alpha * en + 2.0 * cc + Zeff * alpha / r);
  }
  double c(double r) const final { return -alpha * (Zeff / r + en); }
  double d(double r) const final { return -a(r); }
};

/*!
  @brief Fits a quadratic to three points and returns the interpolated maximum.
  @details
  Assumes |y2| = max(|y1|, |y2|, |y3|); used to find the amplitude of a
  sinusoidal oscillation. The three points must be close to the maximum.
  @param x1, x2, x3  x-coordinates of the three points.
  @param y1, y2, y3  y-coordinates of the three points.
  @return Interpolated maximum value.
*/
double fitQuadratic(double x1, double x2, double x3, double y1, double y2,
                    double y3);

} // namespace DiracODE
