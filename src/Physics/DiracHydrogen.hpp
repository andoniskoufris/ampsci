#pragma once

/*!
  @brief Exact relativistic hydrogen-like (Dirac-Coulomb) wavefunctions

  @details
  Analytic solutions of the single-particle Dirac equation for a pointlike
  nuclear charge Z (the Dirac-Coulomb problem). The orbitals have the form

  \f[
      \psi_{n\kappa m}(\vb{r}) = \frac{1}{r}
      \begin{pmatrix} f_{n\kappa}(r)\,\Omega_{\kappa m}(\hat n) \\
      i\,g_{n\kappa}(r)\,\Omega_{-\kappa,m}(\hat n) \end{pmatrix},
  \f]

  with f the large (upper) and g the small (lower) radial component.
  Expressions follow H. A. Bethe and E. E. Salpeter, Quantum Mechanics of
  One- and Two-Electron Atoms (1977).

  The optional lepton mass @p mass defaults to 1 (atomic units). The full
  relativistic energy is \f$ E = mass\,c^2 + \en_{n\kappa} = mass/\alpha^2 +
  \en_{n\kappa} \f$, and the radial functions scale with mass as
  \f$ f_{mass}(r) = \sqrt{mass}\,f_1(mass\,r) \f$.

  \par Non-relativistic limit
  Passing \f$ \alpha \le 0 \f$ selects the non-relativistic (Schrodinger)
  limit directly, rather than evaluating the Dirac expressions at a tiny
  \f$ \alpha \f$ (which is numerically unstable). In this limit the large
  component reduces to the non-relativistic radial function (f = P_nl), the
  small component vanishes (g = 0), and the energy is
  \f$ \en_n = -mass\,Z^2/(2n^2) \f$.

  @note Uses some numerically unstable functions (Gamma functions and
  confluent hypergeometric functions), so may be inaccurate for extreme
  inputs. For reasonable inputs (Zeff = 5, up to n ~ 10) tested good to at
  least parts in 10^12.
*/
namespace DiracHydrogen {

/*!
  @brief Relativistic factor gamma = Sqrt[kappa^2 - (Z*alpha)^2]
  @details
  \f[ \gamma = \sqrt{\kappa^2 - (Z\alpha)^2}. \f]
  @param kappa  Dirac quantum number
  @param zeff   Effective nuclear charge, Z
  @param alpha  Fine-structure constant
  @return gamma
*/
double gamma(int kappa, double zeff, double alpha);

/*!
  @brief Single-particle energy, without rest mass: enk

  @details
  Bound-state Dirac-Coulomb energy, excluding the rest-mass term:

  \f[
    \en_{n\kappa} = mass\,c^2\left[
      \left(1 + \frac{(Z\alpha)^2}{(\gamma + n - |\kappa|)^2}\right)^{-1/2}
      - 1 \right],
  \f]

  evaluated in a numerically stable form (avoids cancellation between the
  \f$ O(c^2) \f$ terms). For \f$ \alpha \le 0 \f$ returns the non-relativistic
  energy \f$ \en_n = -mass\,Z^2/(2n^2) \f$.

  @param n      Principal quantum number (may be non-integer, effective n)
  @param kappa  Dirac quantum number (ignored in the non-relativistic limit)
  @param zeff   Effective nuclear charge, Z
  @param alpha  Fine-structure constant; alpha <= 0 gives non-relativistic
  @param mass   Lepton mass, in units of the electron mass (default 1)
  @return enk (atomic units)
  @note alpha <= 0 selects the non-relativistic limit
*/
double enk(double n, int kappa, double zeff, double alpha, double mass = 1.0);

/*!
  @brief Full energy, including rest mass: Enk = mass/alpha^2 + enk
  @details
  \f[ E_{n\kappa} = mass\,c^2 + \en_{n\kappa} = mass/\alpha^2 + \en_{n\kappa}. \f]
  @param n      Principal quantum number (may be non-integer)
  @param kappa  Dirac quantum number
  @param zeff   Effective nuclear charge, Z
  @param alpha  Fine-structure constant
  @param mass   Lepton mass, in units of the electron mass (default 1)
  @return Enk (atomic units)
  @warning Diverges as alpha -> 0 (rest energy mass/alpha^2); not meaningful in
           the non-relativistic limit.
*/
double Enk(double n, int kappa, double zeff, double alpha, double mass = 1.0);

/*!
  @brief Upper (large) radial component, f(r)

  @details
  Large (upper) component of the Dirac-Coulomb orbital (see the namespace
  description for the full expression). For \f$ \alpha \le 0 \f$ returns the
  non-relativistic large component, \f$ f(r) = P_{nl}(r) \f$ (see @ref P_nl),
  with l = l(kappa).

  @param r      Radial coordinate (atomic units)
  @param n      Principal quantum number (may be non-integer)
  @param kappa  Dirac quantum number
  @param zeff   Effective nuclear charge, Z
  @param alpha  Fine-structure constant; alpha <= 0 gives non-relativistic
  @param mass   Lepton mass, in units of the electron mass (default 1)
  @return f(r)
  @note alpha <= 0 selects the non-relativistic limit: f = P_nl (and g = 0)
*/
double f(double r, double n, int kappa, double zeff, double alpha,
         double mass = 1.0);

/*!
  @brief Lower (small) radial component, g(r)

  @details
  Small (lower) component of the Dirac-Coulomb orbital (see the namespace
  description for the full expression). In the non-relativistic limit
  \f$ \alpha \le 0 \f$ the small component vanishes, g(r) = 0.

  @param r      Radial coordinate (atomic units)
  @param n      Principal quantum number (may be non-integer)
  @param kappa  Dirac quantum number
  @param zeff   Effective nuclear charge, Z
  @param alpha  Fine-structure constant; alpha <= 0 gives g = 0
  @param mass   Lepton mass, in units of the electron mass (default 1)
  @return g(r)
  @note alpha <= 0 selects the non-relativistic limit: g = 0
*/
double g(double r, double n, int kappa, double zeff, double alpha,
         double mass = 1.0);

/*!
  @brief Ratio g/f at radius r
  @details
  Small-to-large component ratio at radius @p r, for a given energy @p en
  (without rest mass). Independent of the overall normalisation.
  @param r      Radial coordinate (atomic units)
  @param kappa  Dirac quantum number
  @param zeff   Effective nuclear charge, Z
  @param alpha  Fine-structure constant
  @param en     Single-particle energy (without rest mass)
  @param mass   Lepton mass, in units of the electron mass (default 1)
  @return g(r)/f(r)
*/
double gfratio(double r, int kappa, double zeff, double alpha, double en,
               double mass = 1.0);

/*!
  @brief Nonrelativistic bound-state radial function: P_nl(r) = r * R_nl(r)
  @details
  Hydrogen-like Schrodinger radial function, unit normalised:
  \f$ \int_0^\infty P_{nl}^2\,\d r = 1 \f$. Mass scales as
  \f$ P_{nl,mass}(r) = \sqrt{mass}\,P_{nl,1}(mass\,r) \f$.

  Evaluated via the confluent hypergeometric form
  \f$ \rho^l e^{-\rho/2}\,{}_1F_1(l+1-n, 2l+2, \rho) \f$, \f$ \rho = 2Z\,mass\,r/n \f$,
  so @p n may be non-integer (effective n). For integer n this reduces to the
  usual associated Laguerre polynomial.

  @param r      Radial coordinate (atomic units)
  @param n      Principal quantum number; may be non-integer (effective n)
  @param l      Orbital angular momentum
  @param zeff   Effective nuclear charge, Z
  @param mass   Lepton mass, in units of the electron mass (default 1)
  @return P_nl(r)
  @warning For non-integer n the series does not terminate and the function
           grows like \f$ e^{+Zr/n} \f$ at large r (not a normalisable bound
           state); it is only meaningful at finite r / near-integer n.
*/
double P_nl(double r, double n, int l, double zeff, double mass = 1.0);

} // namespace DiracHydrogen
