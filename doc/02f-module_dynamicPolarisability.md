\page module_dynamicpol Dynamic Polarisability

\brief Dynamic (frequency-dependent) polarisability, alpha(omega), over a range of frequencies

The `dynamicPolarisability` module (run as
`Module::dynamicPolarisability{}`) calculates the scalar dynamic
polarisability \f$ \alpha_0(\omega) \f$ (and
optionally the tensor part \f$ \alpha_2(\omega) \f$) for a set of valence
states, over a range of frequencies (or wavelengths), writing the results to
screen and to plain-text files for plotting.

* For a _single_ frequency (or a transition, or PNC), prefer the
  \ref module_secondorder module; this module is for frequency scans.
* The valence sum uses sum-over-states (`method = SOS;`) or mixed states
  (`method = MS;`; can be unstable near resonances).
* RPA (TDHF) may be re-solved at each frequency (`rpa_omega`), or held at
  its static value; likewise the core contribution (`core_omega`).
* The spectrum used for the sum-over-states can be adjusted: drop continuum
  states or specific states (spurious resonances), or replace low-lying
  spectrum states with the valence states.
* Structure radiation + normalisation may be added to the matrix elements
  of the sum via the `StructureRadiation{}` block (evaluated at w = 0;
  sum-over-states only).

Example:

```java
Module::dynamicPolarisability{
  states = 6s;
  lambda_minmax = 600, 1800;
  num_steps = 40;
  filename = Cs_alpha.txt;
}
```

The calculations are performed by the Amplitudes library: see
\ref Amplitudes::sos_valence "Amplitudes::sos_valence()", \ref Amplitudes::ms_valence "Amplitudes::ms_valence()",
\ref Amplitudes::sos_core "Amplitudes::sos_core()", and \ref Amplitudes::ms_core "Amplitudes::ms_core()" (the module, Module::dynamicPolarisability,
builds the frequency-dependent matrix element tables and prints).

Available options (from `./ampsci -m dynamicPolarisability`):

```java
// Available Module::dynamicPolarisability options/blocks
Module::dynamicPolarisability{
  states;
    // Which states to calculate? (e.g., '7sp6d'). Must be a
    // subset of valence. By default, all valence states are
    // calculated.
  tensor;
    // Do tensor polarisability a2(w) (as well as a0)
    // [false]
  rpa;
    // Include RPA? [true]
  core_omega;
    // Frequency-dependent core? If true, core part
    // evaluated at each frequency. If false, core evaluated
    // once at w=0 [true]
  rpa_omega;
    // Frequency-dependent RPA? If true, RPA solved at each
    // frequency. If false, RPA solved once at w=0 [true]
  num_steps;
    // number of steps for dynamic polarisability [10]
  omega_minmax;
    // list (frequencies): omega_min, omega_max (in au)
    // [0.01, 0.1]
  lambda_minmax;
    // list (wavelengths, will override omega_minmax):
    // lambda_min, lambda_max (in nm) [600, 1800]
  method;
    // Method used for dynamic pol. for a0(w). Either 'SOS'
    // (sum-over-states) or 'MS' (mixed-states=TDHF). MS can
    // be unstable for dynamic pol. [SOS]
  replace_w_valence;
    // Replace corresponding spectrum states with valence
    // states - circumvents spectrum issue! [false]
  drop_continuum;
    // Discard states from the spectrum with e>0 - these can
    // cause spurious resonances [false]
  drop_states;
    // List. Discard these states from the spectrum for
    // sum-over-states []
  filename;
    // output filename for dynamic polarisability (a0_
    // and/or a2_ will be appended to start of filename)
    // [identity.txt (e.g., CsI.txt)]
  StructureRadiation{}
    // Options for structure radiation and normalisation. If
    // this block is included, SR+N is added to the
    // single-particle matrix elements of the sum-over-states
    // (evaluated at w = 0). Sum-over-states method only
}
```

(The `StructureRadiation{}` block takes the same options as in
\ref module_secondorder, plus `RPA_in_SR`.)
