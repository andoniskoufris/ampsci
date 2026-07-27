#pragma once
#include "CI_Integrals.hpp"
#include "CSF.hpp"
#include "Coulomb/meTable.hpp"
#include "DiracOperator/TensorOperator.hpp"
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

namespace CI {

//==============================================================================
//! Identifies one CI level: 2J, parity (+/-1), and index (in order of energy)
struct Level {
  int twoJ{0};
  int parity{1};
  std::size_t index{0};
};

//==============================================================================
/*!
  @brief Angular coefficients of the two terms of the second-order amplitude
  \f$ A^K \f$.
  @details
  For a transition \f$ a \to b \f$ due to two one-body operators, a dynamic
  \f$ t \f$ (at frequency \f$ \omega \f$) and a static \f$ s \f$, the
  second-order amplitude of rank \f$ K \f$ is

  \f[
    A^K = \sum_n \left[
      c_1(J_n)\,
      \frac{\redmatel{b}{t}{n}\redmatel{n}{s}{a}}{E_a - E_n}
    + c_2(J_n)\,
      \frac{\redmatel{b}{s}{n}\redmatel{n}{t}{a}}{E_a + \omega - E_n}
    \right],
  \f]

  where the coefficients returned here are

  \f[
    c_1 = (-1)^{K}\sqrt{[K]}\,(-1)^{J_b+J_a}
      \begin{Bmatrix} K & k_s & k_t \\ J_n & J_b & J_a \end{Bmatrix},
    \qquad
    c_2 = (-1)^{k_t+k_s}\sqrt{[K]}\,(-1)^{J_b+J_a}
      \begin{Bmatrix} K & k_t & k_s \\ J_n & J_b & J_a \end{Bmatrix}.
  \f]

  With this convention \f$ A^K \f$ is the reduced matrix element of the
  composite operator \f$ [t \times s]^{K} \f$ (both terms), which fixes the
  relation to the z-component; see @ref z_component.

  @param K      Rank of the amplitude.
  @param kt,ks  Ranks of the dynamic and static operators.
  @param twoJb  2J of the final state.
  @param twoJn  2J of the intermediate states.
  @param twoJa  2J of the initial state.
  @return \f$ \{c_1, c_2\} \f$.
*/
[[nodiscard]] std::pair<double, double>
A_K_coefs(int K, int kt, int ks, int twoJb, int twoJn, int twoJa);

/*!
  @brief Converts the reduced amplitude \f$ A^K \f$ to its contribution to the
  z-component of the amplitude.
  @details
  The z-component (\f$ m_a = m_b = m \f$, and \f$ q = 0 \f$ for both operators)
  is the sum over ranks

  \f[
    A_{zz} = \sum_K \braket{k_t 0, k_s 0}{K 0} \,
      (-1)^{J_b-m}\begin{pmatrix} J_b & K & J_a \\ -m & 0 & m \end{pmatrix}
      A^K,
  \f]

  the factor returned here being that of the \f$ K \f$ term. The
  Clebsch-Gordan coefficient comes from
  \f$ t_0 s_0 = \sum_K \braket{k_t 0, k_s 0}{K 0} [t\times s]^K_0 \f$; it is
  unity for \f$ k_s = 0 \f$ (as for a PNC amplitude), but not in general.

  @param K      Rank of the amplitude.
  @param kt,ks  Ranks of the dynamic and static operators.
  @param twoJb,twoJa 2J of the final and initial states.
  @param two_m  Twice the z-component of the angular momentum.
*/
[[nodiscard]] double z_component(int K, int kt, int ks, int twoJb, int twoJa,
                                 int two_m);

//! Relative sign between <A||h||B> and <B||h||A>, for CI states with total
//! angular momenta 2J_A and 2J_B (cf DiracOperator::TensorOperator::symm_sign)
[[nodiscard]] int symm_sign(const DiracOperator::TensorOperator *h, int twoJA,
                            int twoJB);

/*!
  @brief Reduced matrix element of the Pauli spin operator between two CI
  states, \f$ \redmatel{b}{\sigma}{a} \f$.
  @details
  This is the factor that defines the vector transition polarisability, beta:
  the rank-1 part of the amplitude is written

  \f[
    A^{1} = i\,\beta\,(\epsilon^L\times\epsilon^S)\cdot
            \matel{J_bM_b}{\bm\sigma}{J_aM_a},
    \qquad
    \beta = \frac{A^1}{\sqrt{2}\,\redmatel{b}{\bm\sigma}{a}},
  \f]

  the matrix element serving to express the Wigner-Eckart factor of the rank-1
  amplitude (so any operator of rank 1 would do; \f$ \bm\sigma \f$ is the
  conventional choice).

  It is evaluated as the matrix element it is: \f$ \bm\sigma = 2S \f$, with the
  single-particle table of @ref DiracOperator::s, contracted with the CI
  expansions by @ref ReducedME. Nothing is assumed about L and S, which are not
  good quantum numbers for relativistic CI states.

  @param Psi_b,ib  Final CI state (solution @p ib of @p Psi_b).
  @param Psi_a,ia  Initial CI state.
  @param ci_basis  Single-particle basis of the CI expansion.
  @return \f$ \redmatel{b}{\bm\sigma}{a} \f$.

  @note For a single valence electron, the convention is to drop the radial
        overlap, so that \f$ \redmatel{7s}{\bm\sigma}{6s} = 2S_{\kappa\kappa}
        \f$ rather than zero (as in the dcp module, via @ref Angular::S_kk).
        There is no consistent analogue of that for a multi-configuration
        state - forcing the radial overlaps to unity spoils even the diagonal
        matrix element, by adding cross-configuration terms - so the matrix
        element is taken as it stands.

  @note It follows that this vanishes for two states with no configuration in
        common (e.g., 3s3p and 3s4p): beta does not parameterise the rank-1
        amplitude in that case, and the amplitude \f$ A^1 \f$ should be used
        directly.
*/
[[nodiscard]] double sigma_rme(const PsiJPi &Psi_b, std::size_t ib,
                               const PsiJPi &Psi_a, std::size_t ia,
                               const std::vector<DiracSpinor> &ci_basis);

//==============================================================================
/*!
  @brief Second-order amplitude \f$ A^K \f$ between two CI states, evaluated
  with CI mixed states.
  @details
  Evaluates \f$ A^K \f$; see @ref A_K_coefs.
  The sums over the intermediate spectrum are performed with the CI mixed
  states of @ref solve_mixed_state, so they are complete: there is no sum over
  individual CI solutions, and no truncation of the spectrum. All intermediate
  states of a given (J, parity) share the same angular coefficient, so one
  mixed state per (J, parity) and per term is required.

  Each sum is formed in two independent ways, from the ket and from the bra:

  \f[
    A^K \propto \redmatel{b}{t}{\delta a} + \redmatel{\delta b}{t}{a}
    \qquad{\rm and}\qquad
    \redmatel{\Delta b}{s}{a} + \redmatel{b}{s}{\Delta a},
  \f]

  where \f$ \delta \f$ denotes a mixed state due to \f$ s \f$ and \f$ \Delta \f$
  one due to \f$ t \f$. These are returned as the two elements of the pair;
  they must agree, and the difference is a check on the numerics.

  This covers, e.g., static and transition polarisabilities
  (\f$ t = s = E1 \f$) and PNC amplitudes (\f$ s \f$ = PNC operator).

  @param K       Rank of the amplitude. It vanishes unless
                 \f$ |k_t-k_s| \le K \le k_t+k_s \f$ and
                 \f$ (J_b, K, J_a) \f$ satisfy the triangle rule.
  @param Psi_b,ib  Final CI state (solution @p ib of @p Psi_b).
  @param Psi_a,ia  Initial CI state.
  @param t,t_me  Dynamic operator, and its table of single-particle reduced
                 matrix elements (which may include RPA, structure radiation).
                 For a frequency-dependent operator or RPA, the table should
                 have been formed at @p omega.
  @param s,s_me  Static operator, and its table (formed at zero frequency).
  @param omega   Frequency of the dynamic operator. For a real transition this
                 is \f$ E_b - E_a \f$, for which the second denominator above
                 is just \f$ E_b - E_n \f$.
  @param ints    Integral tables, used to construct the CI Hamiltonian of each
                 intermediate (J, parity); e.g., Wavefunction::CI_integrals().
  @param levels_to_remove  CI levels to be removed from the intermediate
                 states (see @ref project_out), so that they may be treated
                 separately - e.g., with experimental energies. Each is
                 identified by (2J, parity, index), and the CI problem for those
                 (J, parity) is solved here, as far as required.
  @param outstream  Stream for progress and the intermediate sums.
  @return \f$ \{{\rm ket}, {\rm bra}\} \f$: the two evaluations of
          \f$ A^K \f$.

  @note The parity selection rule \f$ \pi_a\pi_b = \pi_t\pi_s \f$ must hold,
        else the amplitude is zero.

  @note If a state of an intermediate (J, parity) is degenerate with the
        denominator (\f$ E_a \f$ or \f$ E_a + \omega \f$), the mixed-states
        equation is singular, and its term in \f$ A^K \f$ is divergent. This
        cannot happen for operators of odd parity (as for polarisabilities and
        PNC), since then the intermediate states have the opposite parity to
        \f$ a \f$ and \f$ b \f$.
*/
[[nodiscard]] std::pair<double, double>
A_K(int K, const PsiJPi &Psi_b, std::size_t ib, const PsiJPi &Psi_a,
    std::size_t ia, const DiracOperator::TensorOperator *t,
    const Coulomb::meTable<double> &t_me,
    const DiracOperator::TensorOperator *s,
    const Coulomb::meTable<double> &s_me, double omega, const Integrals &ints,
    const std::vector<Level> &levels_to_remove = {},
    std::ostream &outstream = std::cout);

} // namespace CI
