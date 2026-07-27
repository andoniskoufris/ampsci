#pragma once
#include "CI_Integrals.hpp"
#include "CSF.hpp"
#include "Coulomb/meTable.hpp"
#include "DiracOperator/TensorOperator.hpp"
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

namespace ExternalField {
class CorePolarisation;
}

namespace CI {

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

  the matrix element expressing the Wigner-Eckart factor of the rank-1
  amplitude (any rank-1 operator would do; \f$ \bm\sigma \f$ is conventional).

  Evaluated directly, as \f$ \bm\sigma = 2S \f$: the single-particle table of
  @ref DiracOperator::s, contracted with the CI expansions by @ref ReducedME.
  Nothing is assumed about L and S, which are not good quantum numbers for
  relativistic CI states.

  @param Psi_b,ib  Final CI state (solution @p ib of @p Psi_b).
  @param Psi_a,ia  Initial CI state.
  @param ci_basis  Single-particle basis of the CI expansion.
  @return \f$ \redmatel{b}{\bm\sigma}{a} \f$.

  @note For a single valence electron the convention is to drop the radial
        overlap, so that \f$ \redmatel{7s}{\bm\sigma}{6s} = 2S_{\kappa\kappa}
        \f$ rather than zero (as in the dcp module, via @ref Angular::S_kk).
        There is no consistent analogue for a multi-configuration state:
        forcing the radial overlaps to unity spoils even the diagonal matrix
        element, by adding cross-configuration terms.

  @note So this vanishes for two states with no configuration in common (e.g.,
        3s3p and 3s4p). Then beta does not parameterise the rank-1 amplitude,
        and \f$ A^1 \f$ should be used directly.
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

  Each sum is formed in two independent ways: with the mixed states of
  \f$ s \f$, and with those of \f$ t \f$. Writing \f$ \ket{A_s} \f$ for the
  state \f$ a \f$ plus its mixed state due to \f$ s \f$, these are the
  first-order parts of

  \f[
    \redmatel{B_s}{t}{A_s}
    \qquad{\rm and}\qquad
    \redmatel{B_t}{s}{A_t},
  \f]

  returned as the two elements of the pair. They must agree; the difference is
  a check on the numerics.

  Covers, e.g., static and transition polarisabilities (\f$ t = s = E1 \f$) and
  PNC amplitudes (\f$ s \f$ = PNC operator).

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
                 separately - e.g., with experimental energies. See @ref Level;
                 the CI problem for those (J, parity) is solved here, as far as
                 required.
  @param outstream  Stream for progress and the intermediate sums.
  @return The two evaluations of \f$ A^K \f$: with the mixed states of
          \f$ s \f$, and with those of \f$ t \f$.

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

//==============================================================================
/*!
  @brief Contribution to \f$ A^K \f$ from the polarisation of the closed core.
  @details
  The intermediate states of @ref A_K carry no core hole. This is the missing
  term: a core electron \f$ c \f$ excited by one operator and de-excited by
  the other,

  \f[
    \redmatel{c}{[t\times s]^0}{c} = \sum_m \left[
      c_1\,\frac{\redmatel{c}{t}{m}\redmatel{m}{s}{c}}{\en_c - \en_m}
    + c_2\,\frac{\redmatel{c}{s}{m}\redmatel{m}{t}{c}}{\en_c + \omega - \en_m}
    \right],
  \f]
  \f[
    A^0_{\rm core} = \sqrt{[J]}\sum_c \sqrt{[j_c]}\,
                     \redmatel{c}{[t\times s]^0}{c},
  \f]

  with \f$ c_1, c_2 \f$ from @ref A_K_coefs at \f$ (j_c, j_m, j_c) \f$.

  The core is closed, \f$ J=0 \f$, so this is non-zero only for \f$ K=0 \f$
  (which requires \f$ k_t=k_s \f$) and only for \f$ b=a \f$, the valence factor
  being \f$ \braket{B}{A} \f$. For \f$ t=s=E1 \f$ it is the core
  polarisability. Apart from \f$ \sqrt{[J]} \f$ it is the same for every CI
  level, so it need only be evaluated once.

  @param K       Rank of the amplitude; zero unless \f$ K=0 \f$.
  @param twoJ    2J of the CI state. The diagonal condition is left to the
                 caller: zero unless the final and initial states are the same.
  @param t,s     Dynamic and static operators.
  @param omega   Frequency of the dynamic operator.
  @param core    Hole states \f$ c \f$; e.g., Wavefunction::core().
  @param excited Particle states \f$ m \f$: basis states above the Fermi
                 level. Not restricted to the CI basis, and states occupied by
                 the valence electrons are not removed - see @ref A_K_cv.
  @param dVt,dVs RPA for each operator, solved at the frequency of that
                 operator. May be nullptr.
  @return \f$ A^0_{\rm core} \f$.

  @note RPA enters once: of the two matrix elements, only the one acting on
        the core orbital is dressed. Dressing both counts each RPA chain
        twice, since the sum over \f$ c \f$ already runs over every link of
        the chain. Same counting as the polarisability module.

  @note No structure radiation: both lines here are core lines
*/
[[nodiscard]] double
A_K_core(int K, int twoJ, const DiracOperator::TensorOperator *t,
         const DiracOperator::TensorOperator *s, double omega,
         const std::vector<DiracSpinor> &core,
         const std::vector<DiracSpinor> &excited,
         const ExternalField::CorePolarisation *dVt = nullptr,
         const ExternalField::CorePolarisation *dVs = nullptr);

/*!
  @brief Core-valence contribution to \f$ A^K \f$: the Pauli blocking of the
  core excitations by the valence electrons.
  @details
  @ref A_K_core sums over every particle state, including those occupied by
  the valence electrons. That excitation is blocked; the path that replaces it
  is one operator exciting a core electron to \f$ v' \f$, the other dropping a
  valence electron from \f$ v \f$ into the hole. This is a one-body operator in
  the valence space,

  \f[
    \redmatel{v'}{[t\times s]^K_{cv}}{v} = \sum_c \left[
      c_2\,\frac{\redmatel{v'}{s}{c}\redmatel{c}{t}{v}}{\en_{v'} - \en_c}
    + c_1\,\frac{\redmatel{v'}{t}{c}\redmatel{c}{s}{v}}
                {\en_{v'} - \omega - \en_c}
    \right],
    \qquad
    A^K_{cv} = \redmatel{B}{[t\times s]^K_{cv}}{A},
  \f]

  with \f$ c_1, c_2 \f$ from @ref A_K_coefs at \f$ (j_{v'}, j_c, j_v) \f$. The
  two terms take the coefficient of the opposite ordering, since the hole
  reverses the roles of the operators; its sign is cancelled by the reversed
  energy denominator. The contraction with the CI states is @ref ReducedME.

  For \f$ v'=v \f$ this is the blocking counter term, of weight
  \f$ n_v/[j_v] \f$ for occupation \f$ n_v \f$. The \f$ v' \ne v \f$ terms
  contribute for any \f$ K \f$, and between different CI states.

  @param K       Rank of the amplitude.
  @param Psi_b,ib  Final CI state.
  @param Psi_a,ia  Initial CI state.
  @param t,s     Dynamic and static operators.
  @param omega   Frequency of the dynamic operator.
  @param core    Hole states \f$ c \f$; e.g., Wavefunction::core().
  @param ci_basis  Single-particle basis of the CI expansion.
  @param dVt,dVs RPA for each operator, solved at the frequency of that
                 operator. May be nullptr.
  @return \f$ A^K_{cv} \f$.

  @note RPA enters twice, on both matrix elements - unlike @ref A_K_core. The
        blocked pair is a link of the RPA chain, and the chain continues on
        either side of it: dressing one vertex removes the chains that end on
        the blocked pair, but not those that pass through it.

  @note No structure radiation; cf @ref A_K_core.
*/
[[nodiscard]] double
A_K_cv(int K, const PsiJPi &Psi_b, std::size_t ib, const PsiJPi &Psi_a,
       std::size_t ia, const DiracOperator::TensorOperator *t,
       const DiracOperator::TensorOperator *s, double omega,
       const std::vector<DiracSpinor> &core,
       const std::vector<DiracSpinor> &ci_basis,
       const ExternalField::CorePolarisation *dVt = nullptr,
       const ExternalField::CorePolarisation *dVs = nullptr);

} // namespace CI
