#pragma once
#include "DiracOperator/TensorOperator.hpp"
#include <iostream>
#include <optional>
#include <string>
#include <vector>
class DiracSpinor;
namespace ExternalField {
class CorePolarisation;
}

//! Physical amplitudes and observables (matrix elements, second-order
//! amplitudes); testable functions, callable from any module.
namespace Amplitudes {

/*!
  @brief Result of a single matrix element calculation.
  @details
  Holds \f$ \redmatel{a}{h}{b} \f$: the lowest-order value, the RPA (core
  polarisation) correction, the transition frequency, and the factor that
  converts the reduced matrix element to the requested form (reduced,
  stretched, or hyperfine constant); see DiracOperator::MatrixElementType.

  Everything is stored unscaled; value() and value0() apply the factor.
*/
struct MEdata {
  //! State labels (shortSymbol); a is the bra: <a||h||b>
  std::string a{}, b{};
  //! Transition frequency, e_a - e_b (0 for diagonal)
  double omega{0.0};
  //! Factor converting reduced ME to requested MatrixElementType (1 for Reduced)
  double factor{1.0};
  //! Lowest-order reduced matrix element <a||h||b>
  double t0{0.0};
  //! RPA correction <a||dV||b> (0 if no RPA)
  double dv{0.0};
  //! True if an RPA correction was calculated (distinguishes from dv = 0)
  bool has_rpa{false};

  //! Full value: factor * (t0 + dv)
  double value() const { return factor * (t0 + dv); }
  //! Lowest-order value: factor * t0
  double value0() const { return factor * t0; }
};

/*!
  @brief Options for the matrix_elements() list driver.
  @details
  Frequency handling: a frequency-dependent operator is evaluated at each
  pair's own transition frequency, \f$ \omega_{ab} = \en_a - \en_b \f$,
  which is its physically correct frequency. @p operator_omega overrides
  this, pinning the operator at a fixed frequency (rarely meaningful; mainly
  for comparison with older calculations, or for building tables at a fixed
  external frequency).

  RPA is solved at the fixed frequency @p omega, or re-solved at each
  transition frequency if @p each_omega is true. The RPA frequency dependence
  is often very small, so a fixed frequency is usually adequate; the operator
  frequency matters more, hence the separate treatment.
*/
struct MEoptions {
  //! RPA frequency: dV is solved at this fixed frequency (unless each_omega)
  double omega{0.0};
  //! If true, re-solve RPA at each transition frequency w_ab
  bool each_omega{false};
  //! Override: pin frequency-dependent operator at this fixed frequency.
  //! Default (empty): operator at each pair's transition frequency w_ab.
  std::optional<double> operator_omega{};
  //! Calculate diagonal matrix elements (only for even-parity operators)
  bool diagonal{true};
  //! Calculate off-diagonal matrix elements
  bool off_diagonal{true};
  //! Calculate both <a||h||b> and <b||h||a>
  bool calculate_both{false};
  //! Form of matrix element: Reduced, Stretched, or HFConstant
  DiracOperator::MatrixElementType type{
    DiracOperator::MatrixElementType::Reduced};
  //! Maximum RPA iterations (1 corresponds to first-order RPA)
  int rpa_iterations{128};
  //! Print RPA solve progress
  bool print{true};
};

/*!
  @brief Single matrix element of h between states a and b, with optional RPA.
  @details
  Pure evaluation: assumes @p h, @p h_minus, and @p dV are already at the
  correct frequency. If @p omega is negative and @p h_minus is given,
  @p h_minus is used for the matrix element (sign-sensitive
  frequency-dependent operators, e.g. E1v: h holds \f$ t_+ \f$ at
  \f$ +|\omega| \f$, h_minus holds \f$ t_- \f$ at \f$ -|\omega| \f$).

  The MatrixElementType factor is calculated with @p h (it is purely
  angular), and stored in the returned MEdata rather than applied.

  @param a,b     States: <a||h||b>.
  @param h       The tensor operator.
  @param h_minus Operator at negative frequency; nullptr if not required.
  @param dV      RPA correction, already solved; nullptr for none.
  @param type    Form of matrix element (Reduced, Stretched, HFConstant).
  @param omega   Transition frequency (only selects h vs h_minus, and is
                 recorded in the output).
  @return MEdata holding t0, dv, factor, omega, and labels.
*/
[[nodiscard]] MEdata
matrix_element(const DiracSpinor &a, const DiracSpinor &b,
               const DiracOperator::TensorOperator *h,
               const DiracOperator::TensorOperator *h_minus = nullptr,
               const ExternalField::CorePolarisation *dV = nullptr,
               DiracOperator::MatrixElementType type =
                 DiracOperator::MatrixElementType::Reduced,
               double omega = 0.0);

/*!
  @brief Matrix elements of h for all pairs from a list of orbitals, with
  optional RPA; owns all frequency updates and RPA solves.
  @details
  Calculates \f$ \redmatel{a}{h}{b} \f$ for each pair from @p orbs allowed
  by the selection rules, diagonal first, then off-diagonal.

  Selection rules: pairs with isZero() are skipped; diagonal elements only
  for even-parity operators. For odd-parity operators, only elements with
  the even-parity state on the right are included (unless calculate_both);
  for even-parity operators, each off-diagonal pair is included once
  (unless calculate_both).

  Frequency handling (see MEoptions): a frequency-dependent operator is
  updated at each pair's transition frequency, @p h at
  \f$ +|\omega_{ab}| \f$ and @p h_minus at \f$ -|\omega_{ab}| \f$, unless
  pinned by operator_omega. RPA is solved once at the fixed frequency, or
  re-solved at each transition frequency if each_omega (cleared first
  when poorly converged, or when rpa_iterations is 1 so that first-order
  RPA is not iterated from a previous solution).

  @param orbs      Orbitals; all pairs are considered.
  @param h         The tensor operator; frequency updated internally.
  @param h_minus   Operator at negative frequency (e.g. a clone of @p h for
                   E1v); nullptr if not required.
  @param dV        RPA; solved internally. nullptr for no RPA.
  @param options   See MEoptions.
  @param outstream Stream for progress output.
  @return Vector of MEdata, one per calculated matrix element.

  @note The operator frequency treatment differs from the older
        Module::matrixElements, which tied the operator frequency to the RPA
        frequency. Here the operator is at its correct (transition)
        frequency by default; set operator_omega to reproduce the old
        fixed-frequency behaviour.
*/
[[nodiscard]] std::vector<MEdata> matrix_elements(
  const std::vector<DiracSpinor> &orbs, DiracOperator::TensorOperator *h,
  DiracOperator::TensorOperator *h_minus = nullptr,
  ExternalField::CorePolarisation *dV = nullptr, const MEoptions &options = {},
  std::ostream &outstream = std::cout);

} // namespace Amplitudes
