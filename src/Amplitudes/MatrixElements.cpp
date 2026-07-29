#include "Amplitudes/MatrixElements.hpp"
#include "DiracOperator/TensorOperator.hpp"
#include "ExternalField/CorePolarisation.hpp"
#include "Wavefunction/DiracSpinor.hpp"
#include "fmt/ostream.hpp"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace Amplitudes {

//==============================================================================
MEdata matrix_element(const DiracSpinor &a, const DiracSpinor &b,
                      const DiracOperator::TensorOperator *h,
                      const DiracOperator::TensorOperator *h_minus,
                      const ExternalField::CorePolarisation *dV,
                      DiracOperator::MatrixElementType type, double omega) {

  // t_- (at negative frequency) for sign-sensitive freq-dependent operators
  const auto *h_pm = (omega < 0.0 && h_minus) ? h_minus : h;

  MEdata me;
  me.a = a.shortSymbol();
  me.b = b.shortSymbol();
  me.omega = omega;
  me.factor = h->matel_factor(type, a, b);
  me.t0 = h_pm->reducedME(a, b);
  me.dv = dV ? dV->dV(a, b) : 0.0;
  me.has_rpa = (dV != nullptr);
  return me;
}

//==============================================================================
std::vector<MEdata> matrix_elements(const std::vector<DiracSpinor> &orbs,
                                    DiracOperator::TensorOperator *h,
                                    DiracOperator::TensorOperator *h_minus,
                                    ExternalField::CorePolarisation *dV,
                                    const MEoptions &options,
                                    std::ostream &outstream) {

  std::vector<MEdata> mes;

  const bool freq_dep = h->freqDependantQ();
  const auto pinned = options.operator_omega;

  // Operator at +|w|, h_minus (if given) at -|w|
  const auto update_operator = [&](double w) {
    if (!freq_dep)
      return;
    h->updateFrequency(std::abs(w));
    if (h_minus) {
      h_minus->updateFrequency(-std::abs(w));
    }
  };

  if (pinned) {
    update_operator(*pinned);
  }

  // RPA at fixed frequency, unless re-solving at each transition
  if (dV && !options.each_omega) {
    dV->solve_core(options.omega, options.rpa_iterations, options.print);
  }

  // Re-solve RPA at given frequency; clear first if previous solution poorly
  // converged, or if first-order RPA (must not iterate from prior solution)
  const auto resolve_rpa = [&](double w) {
    if (!dV)
      return;
    if (dV->last_eps() > 1.0e-5 || options.rpa_iterations == 1 ||
        std::isnan(dV->last_eps())) {
      dV->clear();
    }
    dV->solve_core(w, options.rpa_iterations, options.print);
  };

  //----------------------------------------------------
  // Diagonal (only for even-parity operators): w = 0
  if (options.diagonal && h->parity() == 1) {

    if (!pinned) {
      update_operator(0.0);
    }
    if (options.each_omega) {
      resolve_rpa(0.0);
    }

    for (const auto &a : orbs) {
      if (h->isZero(a.kappa(), a.kappa()))
        continue;
      mes.push_back(matrix_element(a, a, h, h_minus, dV, options.type, 0.0));
    }
  }

  //----------------------------------------------------
  // Off-diagonal:
  if (options.off_diagonal) {
    for (std::size_t ib = 0; ib < orbs.size(); ib++) {
      const auto &b = orbs.at(ib);
      for (std::size_t ia = 0; ia < orbs.size(); ia++) {
        const auto &a = orbs.at(ia);

        if (a == b)
          continue;
        if (h->isZero(a.kappa(), b.kappa()))
          continue;

        // Ensure even-parity state on right for odd-parity operators
        if (h->parity() == -1) {
          if (!options.calculate_both && b.parity() == -1)
            continue;
        } else {
          if (!options.calculate_both && ib > ia)
            continue;
        }

        const auto w_ab = a.en() - b.en();

        if (!pinned) {
          update_operator(w_ab);
        }
        if (options.each_omega && dV) {
          if (options.print) {
            fmt::print(outstream,
                       "<{}||t||{}> : w = {:.8f}\n RPA(w) : ", a.shortSymbol(),
                       b.shortSymbol(), w_ab);
          }
          resolve_rpa(w_ab);
        }

        mes.push_back(matrix_element(a, b, h, h_minus, dV, options.type, w_ab));
      }
    }
  }

  return mes;
}

} // namespace Amplitudes
