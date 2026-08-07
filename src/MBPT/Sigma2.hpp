#pragma once
#include "Angular/SixJTable.hpp"
#include "Coulomb/QkTable.hpp"
#include "Wavefunction/DiracSpinor.hpp"
#include "qip/String.hpp"
#include <string>
#include <vector>

namespace MBPT {

/*! @brief Type of energy denominators: DFK, RS, Fermi, Fermi0

 - DFK    : Dzuba-Flambaum-Kozlov convention (Brillouin-Wigner-like, with the target-state energy approximated by the lowest configuration). The external leg belonging to the target state is evaluated at the Fermi level (lowest state for its kappa in excited spectrum); the external leg appearing in the intermediate state keeps its actual orbital energy. Retains state dependence, with no danger of accidental enhancement.
 - RS     : Use actual orbital energies for both external legs. May be danger of accidental enhancement.
 - Fermi  : Both external legs evaluated at the Fermi level for their kappa.
 - Fermi0 : As above, but assume Fermi level for all kappas the same. These often cancel, so there is no (excited-excited) term in denominator (except diagram d). Fine, since the remaining core-excited always dominates.

In each case, each diagram is averaged with its bra-ket partner,
0.5*(1/de + 1/de'), so that S^k (and hence the CI matrix) is symmetric.
This is the Hermitian effective Hamiltonian, correct to
this order in PT. For Fermi0 the two partners coincide.

Energy for internal legs (hole-particle) always actual orbtials.
*/
enum class Denominators { RS, Fermi, Fermi0, DFK };

//! Returns string representation of Denominators enum
std::string parse_Denominators(Denominators d);

//! Parses string to Denominators enum (case-insensitive); returns DFK if unrecognised
Denominators parse_Denominators(std::string_view s);

/*!
  @brief Splits the basis into the core (holes) and excited states.
  @details
  States with energy below @p E_Fermi are considered core/holes.
  Only core states with \f$ n \geq \f$ @p min_n_core, and excited states with
  \f$ n \leq \f$ @p max_n_excited are kept.

  @note Negative energy states not dealt with! Assumed not to be present in basis. Fix?

  @note Replace all instances with \ref DiracSpinor::split_by_energy

  @param basis         Full set of single-particle basis states.
  @param E_Fermi       Energy threshold separating core from excited states.
  @param min_n_core    Minimum principal quantum number for core states.
  @param max_n_excited Maximum principal quantum number for excited states.

  @return Pair {core, excited} of DiracSpinor vectors.
*/
std::pair<std::vector<DiracSpinor>, std::vector<DiracSpinor>>
split_basis(const std::vector<DiracSpinor> &basis, double E_Fermi,
            int min_n_core = 1, int max_n_excited = 999);

/*!
  @brief Reduced two-body Sigma (2nd-order correlation) operator matrix element.
  @details
  Computes \f$ S^k_{vwxy} \f$, the reduced matrix element of the two-body
  second-order correlation (Sigma_2) operator, summed over all 9 Goldstone
  diagrams.

  \f[
    \begin{equation*}
      \begin{split}
        \Sigma^2_{vwxy} 
        =& ~
        \frac{g_{vnxa}\widetilde g_{awny}-g_{vnax}g_{awny}}
        {e_{xa}-\varepsilon_{vn}}\quad\text{(diagram 'a')}\\
        &+
        \frac{g_{vaxn}\widetilde g_{nway}-g_{vanx} g_{nway}}{e_{ya}-\varepsilon_{wn}}
        \quad\text{(diagram 'b')}
        \\
        &-\frac{g_{vnay}g_{awxn}}
        {e_{ya}-\varepsilon_{vn}}
        -\frac{g_{vany}g_{nwxa}}
        {e_{xa}-\varepsilon_{wn}} \quad\text{(diagram 'c1+c2')}\\
        &+
        \frac{g_{vwab}g_{abxy}}{\varepsilon_{ab}-\varepsilon_{vw}}\quad\text{(diagram 'd')}.
      \end{split}
    \end{equation*}
  \f]

  The 'reduced' Sk is defined similarly to Coulomb case (\ref Coulomb). 
  The correlation diagrams have the same angular decomposition as the Coulomb integrals:
  \f[
     \Sigma^2_{vwxy} = \sum_k A^k_{vwxy} S^k_{vwxy},
  \f]

  Note: these have fewer symmetries than \f$ Q^k \f$; specifically
  \f$ S^k_{vwxy} = S^k_{wvyx} \f$. We call with the "Lk" symmetry
  (though, we should have called it "Sk").
  Since each diagram is averaged with its bra-ket partner (Hermitised, see
  \ref MBPT::Denominators), the bra-ket symmetry
  \f$ S^k_{vwxy} = S^k_{xyvw} \f$ also holds, for all denominator options.

  @param k            Multipolarity.
  @param v            External spinor.
  @param w            External spinor.
  @param x            External spinor.
  @param y            External spinor.
  @param qk           Coulomb integral table (QkTable).
  @param core         Core (hole) states for internal lines.
  @param excited      Excited (particle) states (internal lines).
  @param SixJ         Precomputed 6-j symbol table.
  @param denominators Energy denominator convention: see \ref MBPT::Denominators.
  @param fk           Screening factors; fk[k] scales the k-th Coulomb line.
                      Missing (or empty) implies 1.0 (no screening).

  @return \f$ S^k_{vwxy} \f$.
*/
double Sk_vwxy(int k, const DiracSpinor &v, const DiracSpinor &w,
               const DiracSpinor &x, const DiracSpinor &y,
               const Coulomb::QkTable &qk, const std::vector<DiracSpinor> &core,
               const std::vector<DiracSpinor> &excited,
               const Angular::SixJTable &SixJ,
               Denominators denominators = Denominators::DFK,
               const std::vector<double> &fk = {});

/*!
  @brief Selection rule for \f$ S^k_{vwxy} \f$.
  @details
  Differs from the \f$ Q^k_{vwxy} \f$ selection rule due to parity.

  @return True if \f$ S^k_{vwxy} \f$ is non-zero by selection rules.
*/
bool Sk_vwxy_SR(int k, const DiracSpinor &v, const DiracSpinor &w,
                const DiracSpinor &x, const DiracSpinor &y);

/*!
  @brief Minimum and maximum \f$ k \f$ allowed by selection rules for \f$ S^k_{vwxy} \f$.
  @details
  Unlike \f$ Q^k \f$, \f$ k \f$ does not step by 2 for Sigma_2 matrix elements.

  @return Pair {k_min, k_max}.
*/
std::pair<int, int> k_minmax_S(const DiracSpinor &v, const DiracSpinor &w,
                               const DiracSpinor &x, const DiracSpinor &y);

//! @brief Overload taking \f$ 2j \f$ values directly.
std::pair<int, int> k_minmax_S(int twoj_v, int twoj_w, int twoj_x, int twoj_y);

/*!
  @brief Matrix element of the 1-body Sigma (2nd-order correlation) operator.
  @details
  Computes \f$ \langle v | \Sigma(E) | w \rangle \f$ by summing over internal
  core and excited states using the provided Coulomb integral table.

  The energy at which Sigma is evaluated:
  - If @p ev is given, it is used directly.
  - Otherwise \f$ E = \frac{1}{2}(\varepsilon_v + \varepsilon_w) \f$ is used.

  @p max_l_internal truncates the angular momentum of internal lines; intended
  for convergence tests only.

  @param v                External bra spinor.
  @param w                External ket spinor.
  @param qk               Coulomb integral table (YkTable or QkTable).
  @param core             Core (hole) states.
  @param excited          Excited (particle) states.
  @param max_l_internal   Maximum \f$ l \f$ for internal lines (default 99).
  @param ev               Optional energy at which Sigma is evaluated.

  @return \f$ \langle v | \Sigma(E) | w \rangle \f$.
*/
template <class CoulombIntegral> // CoulombIntegral may be YkTable or QkTable
double Sigma_vw(const DiracSpinor &v, const DiracSpinor &w,
                const CoulombIntegral &qk, const std::vector<DiracSpinor> &core,
                const std::vector<DiracSpinor> &excited,
                int max_l_internal = 99,
                std::optional<double> ev = std::nullopt);

/*!
  @brief Energy derivative of the one-body correlation correction,
  \f$ d\langle v|\Sigma(E)|w\rangle/dE \f$, evaluated at E = @p ev.
  @details
  Central finite difference of Sigma_vw(), with step @p delta.

  @param v,w            External spinors.
  @param qk             Coulomb integral table (YkTable or QkTable).
  @param core           Core (hole) states.
  @param excited        Excited (particle) states.
  @param ev             Energy at which the derivative is evaluated.
  @param max_l_internal Maximum \f$ l \f$ for internal lines (default 99).
  @param delta          Finite-difference step, in au.
  @return \f$ d\langle v|\Sigma(E)|w\rangle/dE \f$.
*/
template <class CoulombIntegral>
double dSigma_dE_vw(const DiracSpinor &v, const DiracSpinor &w,
                    const CoulombIntegral &qk,
                    const std::vector<DiracSpinor> &core,
                    const std::vector<DiracSpinor> &excited, double ev,
                    int max_l_internal = 99, double delta = 0.01);

/*!
  @brief Returns energy of first excited state matching a given \f$ \kappa \f$.
  @details
  Searches @p excited for the first state with the given @p kappa_v and
  returns its energy. Used to set a representative energy for a partial wave.

  @param kappa_v  Relativistic angular momentum quantum number.
  @param excited  Excited (particle) states.

  @note Assumes excited is sorted by energy (for each kappa); returns *first* (not lowest) energy

  @note If no state with given kappa is present, returns 0. (Matrix element will be zero anyway)

  @return Energy of the matching state, if kappa is present, otherwise 0
*/
double e_bar(int kappa_v, const std::vector<DiracSpinor> &excited);

/*!
  @brief Calculates (or reads in) a table of two-body Sigma_2 matrix elements.

  @details
  Computes \f$ S^k_{vwxy} \f$ for all relevant combinations of states in
  @p external, using the provided core and excited bases and Coulomb table.
  Results are written to / read from @p filename (empty string disables I/O).

  @param filename                 File to read/write the table. (blank for "false" to not write)
  @param external                 Basis states for external legs (all ME between these are computed).
  @param core                     Core (hole) states for internal summations.
  @param excited                  Excited (particle) states for internal summations.
  @param qk                       Precomputed Coulomb integral table (QkTable).
  @param max_k                    Maximum multipolarity to include.
  @param exclude_wrong_parity_box If true, excludes box diagrams with "wrong" parity.
  @param denominators             DFK, RS, Fermi, Fermi0: see \ref MBPT::Denominators
  @param no_new_integrals         If true, only reads existing intergals; no new computation.
  @param fk                       Screening factors; fk[k] scales the k-th Coulomb line.

  @note no_new_integrals - if we _know_ all required integrals are already in the 
  file to be read in, saves time.
  Otherwise, ampsci will check if any new integrals a requred.
  This checking can take a while, particularly for large basis.

  @return LkTable containing all computed \f$ S^k_{vwxy} \f$ matrix elements.
*/
[[nodiscard]] Coulomb::LkTable calculate_Sk(
  const std::string &filename, const std::vector<DiracSpinor> &external,
  const std::vector<DiracSpinor> &core, const std::vector<DiracSpinor> &excited,
  const Coulomb::QkTable &qk, int max_k, bool exclude_wrong_parity_box,
  Denominators denominators, bool no_new_integrals = false,
  const std::vector<double> &fk = {});

/*!
  @brief Average Sigma_2 correction ratios, h_k, for each multipole k.
  @details
  h_k = <S^k/Q^k>, averaged over all stored S^k integrals with
  |Q^k| above a small cut-off.
  Used to extrapolate Sigma_2 to diagrams outside the tabulated set:
  S^k ~ h_k Q^k. Entries with no data are 0.0 (no correction).

  @param Sk        Table of Sigma_2 integrals (see calculate_Sk).
  @param qk        Coulomb integral table.
  @param external  States for external legs (typically the cis2 basis).
  @param max_k     Maximum multipole; if negative, determined from basis.
  @return Vector of average correction ratios, indexed by k.
*/
std::vector<double> average_hk(const Coulomb::LkTable &Sk,
                               const Coulomb::QkTable &qk,
                               const std::vector<DiracSpinor> &external,
                               int max_k = -1);

/*!
  @brief Extrapolates the Sigma_2 table to diagrams outside the calculated set.
  @details
  For each diagram (k and orbital set) allowed by the Coulomb selection rules
  over @p external that has no stored S^k, adds the average-screening
  approximation, S^k = h_k Q^k (see average_hk). Existing (calculated)
  integrals are untouched. The extrapolated entries are added to the
  in-memory table only; they are never written to the sk file (they would be
  indistinguishable from calculated integrals).

  @param Sk        Table of Sigma_2 integrals; extended in place.
  @param qk        Coulomb integral table.
  @param hk        Average correction ratios (see average_hk).
  @param external  States for external legs (typically the full ci basis).
  @param max_k     Maximum multipole.
*/
void extrapolate_Sk(Coulomb::LkTable &Sk, const Coulomb::QkTable &qk,
                    const std::vector<double> &hk,
                    const std::vector<DiracSpinor> &external, int max_k);

//==============================================================================
//==============================================================================

//! Functions for each Sigma2 diagram; called by \ref Sk_vwxy.
//! @details Broken up for computational convenience, not diagram-by-diagram.
namespace Sigma2 {

/*!
  @brief Diagrams a+b contribution to the reduced two-body Sigma.
  @details
  Computes the sum of Goldstone diagrams a and b for \f$ S^k_{vwxy} \f$.

  @param k            Multipolarity.
  @param v            (+ w x y) External spinors.
  @param qk           Coulomb integral table.
  @param core         Core states.
  @param excited      Excited states.
  @param SixJ         6-j symbol table.
  @param denominators Energy denominator convention.
  @param fk           Screening factors; fk[k] scales the k-th Coulomb line.

  @return Diagrams a+b contribution to \f$ S^k_{vwxy} \f$.
*/
double S_Sigma2_ab(int k, const DiracSpinor &v, const DiracSpinor &w,
                   const DiracSpinor &x, const DiracSpinor &y,
                   const Coulomb::QkTable &qk,
                   const std::vector<DiracSpinor> &core,
                   const std::vector<DiracSpinor> &excited,
                   const Angular::SixJTable &SixJ, Denominators denominators,
                   const std::vector<double> &fk = {});

/*!
  @brief Diagram c1 contribution to the reduced two-body Sigma.
  @details
  Computes Goldstone diagram c1 for \f$ S^k_{vwxy} \f$.

  @param k            Multipolarity.
  @param v            (+ w x y) External spinors.
  @param qk           Coulomb integral table.
  @param core         Core states.
  @param excited      Excited states.
  @param SixJ         6-j symbol table.
  @param denominators Energy denominator convention.
  @param fk           Screening factors; fk[k] scales the k-th Coulomb line.

  @return Diagram c1 contribution to \f$ S^k_{vwxy} \f$.
*/
double S_Sigma2_c1(int k, const DiracSpinor &v, const DiracSpinor &w,
                   const DiracSpinor &x, const DiracSpinor &y,
                   const Coulomb::QkTable &qk,
                   const std::vector<DiracSpinor> &core,
                   const std::vector<DiracSpinor> &excited,
                   const Angular::SixJTable &SixJ, Denominators denominators,
                   const std::vector<double> &fk = {});

/*!
  @brief Diagram c2 contribution to the reduced two-body Sigma.
  @details
  Computes Goldstone diagram c2 for \f$ S^k_{vwxy} \f$.

  @param k            Multipolarity.
  @param v            (+ w x y) External spinors.
  @param qk           Coulomb integral table.
  @param core         Core states.
  @param excited      Excited states.
  @param SixJ         6-j symbol table.
  @param denominators Energy denominator convention.
  @param fk           Screening factors; fk[k] scales the k-th Coulomb line.

  @return Diagram c2 contribution to \f$ S^k_{vwxy} \f$.
*/
double S_Sigma2_c2(int k, const DiracSpinor &v, const DiracSpinor &w,
                   const DiracSpinor &x, const DiracSpinor &y,
                   const Coulomb::QkTable &qk,
                   const std::vector<DiracSpinor> &core,
                   const std::vector<DiracSpinor> &excited,
                   const Angular::SixJTable &SixJ, Denominators denominators,
                   const std::vector<double> &fk = {});

/*!
  @brief Diagram d contribution to the reduced two-body Sigma.
  @details
  Computes Goldstone diagram d for \f$ S^k_{vwxy} \f$.

  @param k            Multipolarity.
  @param v            (+ w x y) External spinors.
  @param qk           Coulomb integral table.
  @param core         Core states.
  @param excited      Excited states.
  @param SixJ         6-j symbol table.
  @param denominators Energy denominator convention.
  @param fk           Screening factors; fk[k] scales the k-th Coulomb line.

  @return Diagram d contribution to \f$ S^k_{vwxy} \f$.
*/
double S_Sigma2_d(int k, const DiracSpinor &v, const DiracSpinor &w,
                  const DiracSpinor &x, const DiracSpinor &y,
                  const Coulomb::QkTable &qk,
                  const std::vector<DiracSpinor> &core,
                  const std::vector<DiracSpinor> &excited,
                  const Angular::SixJTable &SixJ, Denominators denominators,
                  const std::vector<double> &fk = {});

} // namespace Sigma2

} // namespace MBPT

//==============================================================================
#include "Sigma2.ipp"
