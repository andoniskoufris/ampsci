\page module_secondorder Second-Order Amplitudes

\brief Second-order amplitudes for a valence state/transition: polarisabilities, PNC, etc.

The `secondOrder` module (run as `Module::secondOrder{}`) calculates the
second-order (in the external field) amplitude \f$ A^K \f$
for a single-valence transition \f$ A \to B \f$, for two one-body operators
\f$ t \f$ (at frequency \f$ \omega \f$) and \f$ s \f$ (at
\f$ \omega_s = E_B - E_A - \omega \f$):

\f[
  A^K = \sum_n \left[
    c_1 \frac{\langle B||t||n\rangle\langle n||s||A\rangle}{E_A+\omega_s-E_n}
  + c_2 \frac{\langle B||s||n\rangle\langle n||t||A\rangle}{E_A+\omega-E_n}
  \right].
\f]

This single module covers (with the appropriate operators and rank K):

* Scalar polarisability, \f$ \alpha_0 \f$ (t = s = E1, K = 0; static,
  dynamic, or transition)
* Tensor polarisability, \f$ \alpha_2 \f$ (K = 2)
* Vector transition polarisability, \f$ \beta \f$ (K = 1)
* PNC amplitudes (t = E1, s = pnc, K = 1)

and the relevant physical quantity is printed automatically.

The valence sum is evaluated either by sum-over-states (`method = SOS;`,
over the spectrum or basis), or with mixed states (`method = MS;`, solving
the inhomogeneous equation), which is complete: no truncation of the sum.
Run twice to compare the two. The mixed-states method evaluates the sums two
independent ways (with the mixed states of s, and of t), which must agree:
a strong internal check, printed with their relative difference.

The core polarisation contribution is included (for diagonal, K = 0
amplitudes), and the valence sum is decomposed into core-valence, main
(low-n, see `n_main`), and tail parts, for comparison with other
calculations. Structure radiation + normalisation may be added to the
matrix elements of the sum via the `StructureRadiation{}` block
(sum-over-states method only).

This is the single-valence analogue of the CI_secondOrder module.

Examples -- static polarisability of Cs 6s, and the 6s-7s PNC amplitude:

```java
Module::secondOrder{
  A = 6s+;
}

Module::secondOrder{
  A = 6s+;
  B = 7s+;
  t = E1;
  s = pnc;
}
```

The calculations are performed by the Amplitudes library: see
\ref Amplitudes::sos_valence "Amplitudes::sos_valence()", \ref Amplitudes::ms_valence "Amplitudes::ms_valence()",
\ref Amplitudes::sos_core "Amplitudes::sos_core()", and \ref Amplitudes::ms_core "Amplitudes::ms_core()" for the formulas and conventions (the module,
Module::secondOrder, only parses input and prints).

Available options (from `./ampsci -m secondOrder`):

```java
// Available Module::secondOrder options/blocks
Module::secondOrder{
  // Second-order amplitude A^K for a single-valence
  // transition A -> B, for two operators t and s, at
  // frequencies omega and omega_s:
  // A^K = sum_n [c1 <B||t||n><n||s||A>/(E_A+omega_s-E_n) +
  // c2 <B||s||n><n||t||A>/(E_A+omega-E_n)].
  // Energy conservation fixes omega_s = E_B - E_A - omega,
  // so only omega is an input: at its default, t carries
  // the whole transition frequency and s is static. For a
  // dynamic polarisability, set B = A and omega to the
  // frequency; then omega_s = -omega.
  // The valence sum is evaluated either by sum-over-states
  // (method=SOS), or with mixed states (method=MS), which
  // is complete (no truncation of the sum). Run twice to
  // compare the two.
  A;
    // Initial valence state, e.g., 6s+ [required]
  B;
    // Final valence state [default: same as A]
  t;
    // The operator that carries the frequency omega [E1]
  t_options{}
    // Options for the t operator
  s;
    // The other operator; it carries omega_s = E_b - E_a -
    // omega [E1]
  s_options{}
    // Options for the s operator
  omega;
    // Frequency of t. For a transition, leave as default,
    // so that t carries it all and s is static. For a
    // dynamic polarisability (B = A), set this to the
    // frequency: s then carries -omega. [default: E_b -
    // E_a]
  omega_t;
    // Explicit frequency for t, overriding omega. Energy
    // conservation is then up to you; use with care [rare]
  omega_s;
    // Explicit frequency for s, overriding omega_s = E_b -
    // E_a - omega. Use with care [rare]
  K;
    // Rank K of the amplitude. Requires |kt-ks| <= K <=
    // kt+ks, and the triangle rule for (jb,K,ja) [default:
    // smallest allowed]
  method;
    // SOS (sum-over-states) or MS (mixed states) [MS]
  rpa;
    // Method used for RPA: true(=TDHF), false, TDHF, basis,
    // diagram [true]. MS requires TDHF: the diagram method
    // triggers a warning, and TDHF is used
  use_spectrum;
    // SOS: use the spectrum (which includes Sigma, if it
    // was made with it) for the sum, if available;
    // otherwise the basis is used [true]
  Sigma;
    // MS: include the correlation potential in the mixed
    // states, if it is available (use with Brueckner
    // valence states) [true]
  n_main;
    // The 'main' part of the valence sum: intermediate
    // states up to this n, printed separately (as the pnc
    // module) [max_n_core + 4]
  StructureRadiation{}
    // Options for structure radiation and normalisation. If
    // this block is included, SR+N is added to the
    // single-particle matrix elements of the sum.
    // Sum-over-states method only
}

// Available StructureRadiation options/blocks
StructureRadiation{
  // If this block is included, SR + Normalisation
  // corrections will be included (sum-over-states method
  // only)
  Qk_file;
    // true/false/filename - SR: filename for QkTable file.
    // If blank will not use QkTable; if exists, will read
    // it in; if doesn't exist, will create it and write to
    // disk. If 'true' will use default filename
  n_minmax;
    // list; min,max n for core/excited (internal): [1,inf]
  n_max_legs;
    // SR+N is applied to matrix elements whose states both
    // have n <= this (the valence legs, and the low-n
    // intermediate states). SR+N is only meaningful between
    // physical states: the high-n basis states are cavity
    // states [default: max_n_core + 3]
  norm;
    // Include the normalisation of states? If false, only
    // the structure radiation is included [true]
}
```
