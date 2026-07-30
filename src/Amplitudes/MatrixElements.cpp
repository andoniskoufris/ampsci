#include "Amplitudes/MatrixElements.hpp"
#include "DiracOperator/TensorOperator.hpp"
#include "ExternalField/CorePolarisation.hpp"
#include "IO/ChronoTimer.hpp"
#include "MBPT/StructureRad.hpp"
#include "Wavefunction/DiracSpinor.hpp"
#include "fmt/ostream.hpp"
#include <cassert>
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

//==============================================================================
std::vector<SRNdata> sr_matrix_elements(const std::vector<DiracSpinor> &orbs,
                                        DiracOperator::TensorOperator *h,
                                        MBPT::StructureRad *sr,
                                        ExternalField::CorePolarisation *dV,
                                        const SRNoptions &options,
                                        std::ostream &outstream) {
  assert(sr != nullptr);

  std::vector<SRNdata> out;

  const bool each = options.each_omega;
  const bool freq_dep = h->freqDependantQ();
  const auto pinned = options.operator_omega;

  const auto update_op = [&](double w) {
    if (freq_dep) {
      h->updateFrequency(w);
    }
  };

  if (pinned) {
    update_op(*pinned);
  }

  // RPA and SR-table frequency: fixed at options.omega (unless each_omega).
  // The operator must be at the table frequency when the tables are built;
  // it is then moved to each pair's transition frequency below
  if (dV && !each) {
    if (!pinned) {
      update_op(options.omega);
    }
    dV->solve_core(options.omega, 100, options.print);
  }

  // With each_omega, the diagonal elements are all at w = 0: solve there
  // first (off-diagonal elements re-solve per transition below)
  if (each && options.diagonal && h->parity() == 1) {
    if (!pinned) {
      update_op(0.0);
    }
    if (dV) {
      dV->solve_core(0.0, 100, options.print);
    }
  }

  // SR matrix element tables (holds <a||h+dV||b> for the internal lines)
  sr->solve_core(h, dV);

  // Diagonal first, then off-diagonal
  for (const auto diag : {true, false}) {

    if (diag && !options.diagonal)
      continue;
    if (!diag && !options.off_diagonal)
      continue;

    for (std::size_t ib = 0; ib < orbs.size(); ib++) {
      const auto &v = orbs.at(ib);
      for (std::size_t ia = 0; ia < orbs.size(); ia++) {
        const auto &w = orbs.at(ia);

        if (h->isZero(w.kappa(), v.kappa()))
          continue;
        // Ensure even-parity state on right for odd-parity operators
        if (h->parity() == -1) {
          if (!options.calculate_both && v.parity() == -1)
            continue;
        } else {
          if (!options.calculate_both && ib > ia)
            continue;
        }
        if (diag != (v == w))
          continue;

        std::optional<IO::ChronoTimer> timer;
        if (options.print) {
          timer.emplace("time");
        }

        // Transition frequency (operator), and the RPA/SR-table frequency
        const auto w_ab = w.en() - v.en();
        const auto ww_sr = each ? w_ab : options.omega;
        if (options.print) {
          fmt::print(outstream, "\n<{}||t||{}>: {:.6f}\n", w.shortSymbol(),
                     v.shortSymbol(), w_ab);
        }

        const auto factor = h->matel_factor(options.type, w, v);

        // The operator itself is at the (physical) transition frequency
        if (!pinned) {
          update_op(w_ab);
        }
        // Off-diagonal each-frequency: re-solve RPA and the SR tables at
        // this transition frequency
        if (each && dV && !diag) {
          if (dV->last_eps() > 1.0e-3 || std::isnan(dV->last_eps())) {
            dV->clear();
          }
          dV->solve_core(w_ab, 100, options.print);
        }
        if (each && (freq_dep || dV) && !diag) {
          sr->solve_core(h, dV);
        }

        SRNdata me;
        me.a = w.shortSymbol();
        me.b = v.shortSymbol();
        me.omega = w_ab;
        me.factor = factor;
        me.t0 = h->reducedME(w, v);
        me.dv = dV ? dV->dV(w, v) : 0.0;
        me.has_rpa = (dV != nullptr);

        if (options.print) {
          fmt::print(outstream, "{:8s}  {:12.5e}\n", "t0", factor * me.t0);
          if (dV) {
            fmt::print(outstream, "{:8s}  {:12.5e}\n", "dV", factor * me.dv);
            fmt::print(outstream, "{:8s}  {:12.5e}\n", "RPA", me.value0());
          }
          outstream << std::flush;
        }

        // "Top" + "Bottom" + "Centre" SR diagrams:
        me.sr = sr->SR(w, v, ww_sr);
        if (options.print) {
          fmt::print(outstream, "{:8s}  {:12.5e}\n", "SR", factor * me.sr);
          outstream << std::flush;
        }

        // Normalisation of states:
        const auto f_norm =
          sr->f_norm(w) + (w == v ? sr->f_norm(w) : sr->f_norm(v));
        me.norm = f_norm * (me.t0 + me.dv);
        if (options.print) {
          fmt::print(outstream, "{:8s}  {:12.5e}\n", "Norm(0)",
                     factor * me.norm);
          fmt::print(outstream, "{:8s}  {:12.5e}\n", "Norm(SR)",
                     factor * f_norm * (me.t0 + me.dv + me.sr));
          outstream << std::flush;
        }

        // Brueckner orbital correction (only when the legs are not already
        // Brueckner orbitals):
        if (options.include_bo) {
          me.bo = sr->BO(w, v);
          if (options.print) {
            fmt::print(outstream, "{:8s}  {:12.5e}\n", "BO", factor * me.bo);
          }

          // Frequency-derivative term: the second-order energy shifts move
          // the transition frequency; for a freq-dependent operator this
          // shifts the matrix element: dT = (dt/dw) * dw^(2)
          if (freq_dep && !pinned && w != v) {
            const auto dw_2 = sr->Sigma_vw(w, w) - sr->Sigma_vw(v, v);
            const auto del = 0.05 * std::abs(w_ab);
            h->updateFrequency(w_ab + del);
            const auto h_plus = h->reducedME(w, v);
            h->updateFrequency(w_ab - del);
            const auto h_minus = h->reducedME(w, v);
            h->updateFrequency(w_ab);
            const auto T_deriv = (h_plus - h_minus) / (2 * del) * dw_2;
            if (options.print) {
              fmt::print(outstream, "+ {:7s} {:12.5e}\n", "Tderiv",
                         factor * T_deriv);
            }
            me.bo += T_deriv;
          }
        }

        if (options.print) {
          fmt::print(outstream, "{:8s}  {:12.5e}\n", "Total", me.total());
          outstream << std::flush;
        }

        out.push_back(me);
      }
    }
  }

  return out;
}

} // namespace Amplitudes
