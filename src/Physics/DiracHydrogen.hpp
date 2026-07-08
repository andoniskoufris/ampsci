#pragma once

/*!
  @brief Exact relativistic hydrogen-like (Coulomb) wavefunctions

  @details In the form
  \f[ 
      \psi_{n\kappa m}(\vb{r}) = \frac{1}{r}
      \begin{pmatrix} f_{n\kappa}(r)\,\Omega_{\kappa m}(\hat n) \\
      i\,g_{n\kappa}(r)\,\Omega_{-\kappa,m}(\hat n) \end{pmatrix}. 
  \f]

  From: H. A. Bethe and E. E. Salpeter, Quantum Mechanics of One-and Two-Electron
  Atoms (1977).

  @note Uses some numerically unstable functions, including Gamma functions and
  confluent hypergeometric functions. So, for some inputs, may be numerically
  unstable. For reasonable inputs (i.e., Zeff=5, up to n=~10), tested good to 
  at least parts in 10^12

  The optional electron mass parameter @p mass defaults to 1 (atomic units).
  The full relativistic energy is E = mass*c^2 + enk = mass/alpha^2 + enk.
*/
namespace DiracHydrogen {

//! Relativistic factor gamma = Sqrt[kappa^2 - (Z*alpha)^2]
double gamma(int kappa, double zeff, double alpha);

/*!
  @brief Energy, without rest mass.
  @details 
  @p n may be non-integer (effective n). 
  @p mass is the electron mass (default 1 a.u.).
*/
double enk(double n, int kappa, double zeff, double alpha, double mass = 1.0);

/*!
  @brief Full energy: Enk = mass/alpha^2 + enk.
  @details @p mass is the electron mass (default 1 a.u.).
*/
double Enk(double n, int kappa, double zeff, double alpha, double mass = 1.0);

/*!
  @brief Upper (large) radial component.
  @details @p mass is the electron mass (default 1 a.u.).
*/
double f(double r, double n, int kappa, double zeff, double alpha,
         double mass = 1.0);

/*!
  @brief Lower (small) radial component.
  @details @p mass is the electron mass (default 1 a.u.).
*/
double g(double r, double n, int kappa, double zeff, double alpha,
         double mass = 1.0);

/*!
  @brief Ratio g/f at radius r.
  @details For given energy @p en (without rest mass) and mass @p mass.
*/
double gfratio(double r, int kappa, double zeff, double alpha, double en,
               double mass = 1.0);

/*!
  @brief Nonrelativistic bound-state radial function: P_nl(r) = r * R_nl(r).
  @details Hydrogen-like, unit normalised: Int[P_nl^2, {r, 0, Infinity}] = 1.
  @p mass is the electron mass (default 1 a.u.).
*/
double P_nl(double r, int n, int l, double zeff, double mass = 1.0);

} // namespace DiracHydrogen
