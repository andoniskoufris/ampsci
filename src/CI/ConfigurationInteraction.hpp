#pragma once
#include "CI_Integrals.hpp"
#include "CSF.hpp"
#include "Coulomb/QkTable.hpp"
#include "Coulomb/meTable.hpp"
#include "IO/InputBlock.hpp"
#include "Wavefunction/DiracSpinor.hpp"
#include "Wavefunction/Wavefunction.hpp"
#include <iostream>
#include <string>
#include <vector>

//! Functions and classes for Configuration Interaction calculations
/*! @details
    Main functions are: 
    - @ref configuration_interaction
    - @ref run_CI
    
    Main Classes are: 
    - @ref CSF2
    - @ref PsiJPi
*/
namespace CI {

/*!
  @brief Runs Configuration Interation: returns CI solutions for all
  requested J and parity values.

  @details
  Reads options from @p input, the CI Input Block, and constructs the CI basis
  from @p wf, computes the required Coulomb (and optionally Breit and two-body 
  MBPT) integrals, then calls run_CI() for each requested (J, parity) pair.

  The returned vector contains one @ref PsiJPi per {J, parity} combination, each
  holding the eigenvalues and CI expansion coefficients for the requested number
  of solutions.

  As of writing, options are:
  @code{.java}
  // Available CI options/blocks
  CI{
    ci_basis;
      // Basis used for CI expansion; must be a sub-set of
      // full ampsci basis [default: 20spdf]
    J;
      // List of total angular momentum J for CI solutions
      // (comma separated). Must be integers (two-electron
      // only). []
    J+;
      // As above, but for EVEN CSFs only (takes precedence
      // over J).
    J-;
      // As above, but for ODD CSFs (takes precedence over J).
    num_solutions;
      // Number of CI solutions to find (for each J/pi) [5]
    all_below_cm;
      // Find all CI solutions for energies below this
      // threshold, in inverse cm. Note that this is the total
      // energy, not the excitation energy. If set,
      // num_solutions is ignored.
    sigma1;
      // Include one-body MBPT correlations? [false]
    sigma2;
      // Include two-body MBPT correlations? [false]
    Brueckner;
      // Use Brueckner (spectrum) states for CI basis? Must
      // have Correlations, Spectrum, and sigma1. [false]
    cis2_basis;
      // The subset of ci_basis for which the two-body MBPT
      // corrections are calculated. Must be a subset of
      // ci_basis. If existing sk file has more integrals,
      // they will be used. [default: Nspdf, where N is
      // maximum n for core + 3]
    Breit2;
      // Include two-body Breit? Default is true if Breit
      // included in HF. Ignored if Breit not included in HF.
      // [true]
    Breit_basis;
      // Subset of ci_basis used to include two-body Breit
      // corrections into CI matrix. Large basis is slow, uses
      // huge memory, and makes small contribution. [default:
      // Nspdf, where N is maximum n for core + 6]
    s1_basis;
      // Usually should be left as default. Basis used for the
      // one-body MBPT diagrams (Sigma^1) internal lines.
      // These are the most important, so in general the
      // default (all basis states) should be used. Must be a
      // subset of full ampsci basis. [default: full basis]
      //  - Note: if CorrelationPotential is available, it
      // will be used instead of calculating the Sigma_1
      // integrals
    s2_basis;
      // Usually should be left blank. Basis used for internal
      // lines of the two-body MBPT diagrams (Sigma^2)
      // internal lines. Must be a subset of s1_basis.
      // [default: s1_basis]
    n_min_core;
      // Minimum n for core to be included in MBPT [1]
    max_k;
      // Maximum k (multipolarity) to include when calculating
      // new Coulomb integrals. Higher k often contribute
      // negligably. Note: if qk file already has higher-k
      // terms, they will be included. Set negative (or very
      // large) to include all k. [8]
    denominators;
      // 'DFK', 'RS', 'Fermi', 'Fermi0'. Denominators used in
      // Sigma2 matrix elements. DFK (Dzuba-Flambaum-Kozlov):
      // target-state legs use the lowest excited state of their
      // kappa, intermediate-state legs use actual energies. RS
      // uses actual energies for all external legs, Fermi uses
      // the lowest excited state for each kappa (both legs),
      // Fermi0 uses lowest excited state for all kappas (and
      // thus cancels in all except diagram 'd'). Applies to
      // Sigma_2 only: Sigma_1 is always evaluated at a fixed
      // energy per kappa (the lowest valence state, or the
      // energy the correlation potential was formed at), which
      // coincides with the DFK/Fermi convention. Applies to
      // *new* integrals only: existing sk file integrals are
      // re-used as-is (the default sk_file name encodes this
      // setting). [DFK]
    qk_file;
      // Filename for storing two-body Coulomb integrals. By
      // default, is ~ At.qk, where At is atomic symbol +
      // 'identity'. Set to 'false' to disable read/write.
    sk_file;
      // Filename for storing two-body Sigma_2 integrals. By
      // default, is At_n_b_k_d[_fk].sk, where At is atomic
      // symbol, n is n_min_core, b is s2 (internal) basis, k is
      // max_k, d is the denominators mode, and _fk is appended
      // if screening factors are set. Set to 'false' to disable
      // read/write. Note: convenience cache only. Stored
      // integrals are re-used as-is; the fk _values_ and
      // exclude_wrong_parity_box are not encoded in the
      // filename - it is the user's responsibility to delete
      // the file when changing those.
    bk_file;
      // Filename for storing two-body Breit integrals. By
      // default, is ~ At.bk, where At is atomic symbol +
      // 'identity'. Set to 'false' to disable read/write.
    no_new_integrals;
      // Usually false. If set to true, ampsci will not
      // calculate any new Coulomb or Sigma_2 integrals, even
      // if they are implied by the above settings. This saves
      // time when we know all required integrals already
      // exist, since the code doesn't need to check. [false]
    exclude_wrong_parity_box;
      // Excludes the Sigma_2 box corrections that have
      // 'wrong' parity when calculating Sigma2 matrix
      // elements. Note: If existing sk file already has
      // these, they will be included [false]
    sort_output;
      // Sort output by energy? Default is to sort by J and Pi
      // first. [false]
    print_details;
      // Condition to print details of each CI solution
      // (otherwise just prints summary) [true]
    fk;
      // Vector of screening factors for Sigma_2 (fk[k] scales
      // the k-th Coulomb line). []
    extrapolate_sigma2;
      // Extrapolate Sigma_2 to diagrams outside cis2_basis,
      // using average correction ratios: S^k ~ hk*Q^k, where
      // hk = <S/Q> is averaged over the calculated Sigma_2
      // integrals. Note: These are stored in the Sk table, but
      // NOT written to disk [true]
  }
  @endcode
  * Always check for up-to-date options from command line: `$ ampsci -i CI`
  * See also @ref run_CI, which this function calls

  @param input   Input block containing CI options.
  @param wf      Fully initialised Wavefunction object supplying the orbital
                 basis and radial grid.
  @return @ref Solutions: one PsiJPi per (J, parity) combination requested,
          together with the integral tables used to build the CI Hamiltonians.

  @note If run with `read_only`, no integrals are calculated, and the returned
        integral tables are empty (see Integrals::availableQ()).
*/
Solutions configuration_interaction(const IO::InputBlock &input,
                                    const Wavefunction &wf);

/*!
  @brief Constructs and solves the CI eigenvalue problem for a single J,pi
  @details
  Builds the CI+MBPT Hamiltonian matrix in the basis of two-electron 
  configuration state functions (CSFs) with total angular momentum 
  @p twoJ /2 and parity @p parity, 
  then solves the eigenvalue problem to obtain CI energies and
  expansion coefficients.

  The Hamiltonian includes:
  - one-body terms from @p h1
  - two-body Coulomb interaction via the \f$ Q^k \f$ integrals in @p qk
  - optionally, two-body Breit corrections via \f$ B^k \f$ integrals in @p Bk
    (used if @p Bk is non-empty)
  - optionally, two-body MBPT \f$ \Sigma_2 \f$ corrections via \f$ S^k \f$
    integrals in @p Sk (used if @p include_Sigma2 is true, and @p Sk is non-empty)

  The number of solutions returned is controlled by @p num_solutions and
  @p all_below: if @p all_below is set it takes precedence and all eigenstates
  with total energy below the threshold are found.

  @param ci_sp_basis  Single-particle basis states spanning the CI space.
  @param twoJ         Twice the total angular momentum, 2J (must be a
                      non-negative even integer for two-electron systems).
  @param parity       Parity of the sector: +1 (even) or -1 (odd).
  @param num_solutions Number of lowest eigenstates to find. Ignored if
                       @p all_below_cm is set. Pass 0 to find all solutions.
  @param all_below_cm If set, find all eigenstates with total energy below this
                      value (in cm^-1). Overrides @p num_solutions.
  @param h1           Table of one-body Hamiltonian matrix elements between
                      single-particle basis states.
  @param qk           Table of two-body Coulomb \f$ Q^k \f$ integrals.
  @param Bk           Table of two-body Breit \f$ B^k \f$ integrals. Ignored
                      (treated as absent) if the table is empty.
  @param Sk           Table of two-body MBPT \f$ \Sigma_2 \f$ (\f$ S^k \f$)
                      integrals. Only used when @p include_Sigma2 is true.
  @param include_Sigma2 If true, add two-body MBPT corrections from @p Sk to
                        the CI Hamiltonian.
  @param print_details  If true, print a breakdown of the leading configurations
                        for each solution. Leads to very large output if
                        @p num_solutions is large
  @param read_only    If true, only the solutions already in the file are read;
                      the eigenvalue problem is not solved, and nothing is
                      written [default: false].
  @param outstream    Output stream for progress and results [default: stdout].
  @param ci_fname     Filename for reading/writing CI solutions ("" disables).
  @param ci_settings_key Settings key stored in the solutions file header:
                      if it does not match the key in an existing file, the
                      file is not read, and is discarded on the next write
                      (see @ref PsiJPi::read_write).
  @return PsiJPi (@ref PsiJPi) containing the CI eigenvalues and expansion
  coefficients for the requested solutions.
*/
PsiJPi run_CI(const std::vector<DiracSpinor> &ci_sp_basis, int twoJ, int parity,
              int num_solutions, std::optional<double> all_below_cm,
              const Coulomb::meTable<double> &h1, const Coulomb::QkTable &qk,
              const Coulomb::WkTable &Bk, const Coulomb::LkTable &Sk,
              bool include_Sigma2, bool print_details, bool read_only = false,
              std::ostream &outstream = std::cout,
              const std::string &ci_fname = "",
              const std::string &ci_settings_key = "");

} // namespace CI