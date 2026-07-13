#pragma once
#include <complex>

//! Hypergeometric functions
namespace Hypergeometric {

//! True if compiled with FLINT support; complex H1f1 returns zero otherwise
constexpr bool has_flint =
#if defined(AMPSCI_USE_FLINT3) || defined(AMPSCI_USE_FLINT2)
  true;
#else
  false;
#endif

/*!
  @brief Scaled confluent hypergeometric function: e^{s} * 1F1(a, b, z)
  @details
  - @p a and @p z may be complex (via template). @p s and @p b must be real

  @p T (template param) may be double or std::complex<double>.

  Real (double) arguments: evaluated with GSL (double precision); the scale
  e^{s} is applied as an ordinary double factor.

  Complex arguments: first attempts a fast double-precision evaluation
  (Maclaurin series at small |z|, asymptotic expansion at large |z|, each
  with a running error estimate); this covers most continuum-state calls at
  ~100x the speed of ball arithmetic. Where the estimated error is too
  large (strong cancellation: large Im(a) with moderate |z|), falls back to
  FLINT ball arithmetic, increasing the working precision until the result
  is accurate to (at least) full double precision. The scale e^{s} is
  applied inside the evaluation, so exponentially small 1F1 values (e.g.
  like e^{-pi*nu/2} for continuum Coulomb functions) can be paired with
  their compensating normalisation factors without under/overflowing
  double.

  @note Requires FLINT library to work with complex values. 
  Compile with `-lflint` and set `-DAMPSCI_USE_FLINT3` (FLINT 3+), or with
  `-lflint-arb -lflint` and `-DAMPSCI_USE_FLINT2` (FLINT 2.x + Arb)
  (done automatically by Makefile/configure.sh)
  
  @warning Uses IFDEF to allow compilation if FLINT is not available.
  However, functions must not be called if FLINT is not installed.
  Will abort, or return nan if called without FLINT.
  Always use `if constexpr (Hypergeometric::has_flint) {}` in code, 
  to check if available; see @ref Hypergeometric::has_flint 
*/
template <typename T>
T H1f1(T a, double b, T z, double s = 0.0);

} // namespace Hypergeometric
