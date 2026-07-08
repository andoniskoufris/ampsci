#pragma once
#include <utility>

/*!
  @brief Exact relativistic continuum (en > 0) hydrogen-like (Coulomb)
  wavefunctions

  @details In the form
  \f[ 
      \psi_{\en\kappa m}(\vb{r}) = \frac{1}{r}
      \begin{pmatrix} f_{\en\kappa}(r)\,\Omega_{\kappa m}(\hat n) \\
      i\,g_{\en\kappa}(r)\,\Omega_{-\kappa,m}(\hat n) \end{pmatrix}, 
  \f]

  normalised on the energy scale,

  \f[ 
    \int_0^\infty (f_\en f_{\en'} + g_\en g_{\en'})\,dr
      = \delta(\en - \en'). 
  \f]

  See Methods for full definitions.

  @note Requires the FLINT library (for confluent hypergeometric functions of
  complex argument); available at compile time via the constexpr flag
  DiracContinuum::available. 
  
  @warning Without FLINT, calling f and g may abort, or return NaN.
  Use @ref DiracContinuum::available to check if available.

  The optional electron mass parameter @p m defaults to 1 (atomic units).
  The full relativistic energy is E = m*c^2 + en = m/alpha^2 + en, with
  en > 0 for continuum states.
*/
namespace DiracContinuum {

//! True if compiled with FLINT support; f, g, and fg return NaN otherwise
#ifdef AMPSCI_USE_FLINT
constexpr bool available = true;
#else
constexpr bool available = false;
#endif

//! Relativistic factor gamma = Sqrt[kappa^2 - (aZ)^2]
double gamma(int kappa, double zeff, double alpha);

/*!
  @brief Equivalent momentum: pe = Sqrt[en*(2m + en*alpha^2)].
  @details @p en is the energy without rest mass (en > 0); @p m is the
  electron mass (default 1 a.u.).
*/
double pe(double en, double alpha, double m = 1.0);

/*!
  @brief Both radial components {f, g} at radius r.
  @details More efficient than separate calls to f() and g(), since the
  (expensive) hypergeometric factors are shared.
  @return std::pair {f, g}
  @warning Prints a warning and returns {NaN, NaN} if compiled without FLINT
  (see DiracContinuum::available)
*/
std::pair<double, double> fg(double r, double en, int kappa, double zeff,
                             double alpha, double m = 1.0);

/*!
  @brief Upper (large) radial component.
  @details @p m is the electron mass (default 1 a.u.).
*/
double f(double r, double en, int kappa, double zeff, double alpha,
         double m = 1.0);

/*!
  @brief Lower (small) radial component.
  @details @p m is the electron mass (default 1 a.u.).
*/
double g(double r, double en, int kappa, double zeff, double alpha,
         double m = 1.0);

/*!
  @brief Large-r asymptotic form of f.
  @details f ~ Sqrt[pe/(pi*en)] * cos(pe*r + nu*ln(2*pe*r) - Delta).
  Does not require FLINT.
*/
double f_asymptotic(double r, double en, int kappa, double zeff, double alpha,
                    double m = 1.0);

/*!
  @brief Large-r asymptotic form of g.
  @details g ~ -alpha * Sqrt[en/(pi*pe)] * sin(pe*r + nu*ln(2*pe*r) - Delta).
  Does not require FLINT.
*/
double g_asymptotic(double r, double en, int kappa, double zeff, double alpha,
                    double m = 1.0);

/*!
  @brief Nonrelativistic continuum radial function, energy normalised.
  @details P_el = Sqrt[2m/(pi*p)] * F_l(-Z*m/p, p*r), with p = Sqrt[2*m*en]
  and F_l the regular Coulomb function. Does not require FLINT.
*/
double P_el(double r, double en, int l, double zeff, double m = 1.0);

} // namespace DiracContinuum
