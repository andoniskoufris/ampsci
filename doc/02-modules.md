\page modules Modules

\brief Descpription of modules system: available modules and ascociated options

# ampsci Module system

\brief The modules system: listing the available modules and their options.

The modules system is how you interact with ampsci after the wavefunction has been calculated.
Any number of modules can be run in the same job to compute matrix elements, polarisabilities,
QED corrections, structure radiation, and more.

* See \ref tutorial_modules for a hands-on introduction to using modules.
* The modules system allows the easy calculation of any atomic properties after the wavefunction has been calculated.
* Any number of _modules_ can be run by adding `Module::moduleName{}` blocks.
* The operators the modules act with (E1, hfs, pnc, ...) are described at \ref using_operators.

Get a list of available modules: `./ampsci -m`

Output will look something like this (example):

```txt
Available modules: 
 * WriteOrbitals
     Write orbitals to disk for plotting
 * matrixElements
     Calculates matrix elements of any operator
 * CI_matrixElements
     Calculates matrix elements of any operator for CI wavefunctions
 * polarisability
     Calculates static polarisabilities
 * dynamicPolarisability
     Calculates dynamic polarisabilities
 * transitionPolarisability
     Calculates transition polarisabilities
 * StructureRadiation
     Structure radiation + normalisation corrections to matrix elements
 * fieldShift
     Calculates field-shift constants (isotope shift)
 * QED
     QED corrections to energies/matrix elements
 * Breit
     Breit corrections to energies
 * Kionisation
     Calculate atomic ionisation form-factors
 * exampleModule
     A short description of the module
```

You can also get most of this information directly from the command-line:

* `./ampsci -m  <ModuleName>`
  * Prints list of available Modules (same as --modules)
  * ModuleName is optional. If given, will list available options for that Module
  * Note the output is in the same format as required by the input file - you can copy+paste this into your input file.

```sh
./ampsci -m matrixElements
```

```java
// Available Module::matrixElements options/blocks
Module::matrixElements{
  // Matrix elements of any operator for HF/Brueckner
  // valence states. Supports RPA, diagonal and off-diagonal
  // elements, core states, and optional use of the spectrum
  // instead of valence states.
  operator;
    // e.g., E1, hfs (see ampsci -o for available operators)
  options{}
    // options specific to operator (see ampsci -o
    // 'operator')
  rpa;
    // Method used for RPA: true(=TDHF), false, TDHF, basis,
    // diagram [true]
  // ... etc.
}
```

See \ref module_matrixelements for the full list of options.

## See also

Basic documentation for the most commonly-used modules:

* \subpage module_matrixelements - \copybrief module_matrixelements
* \subpage module_structurerad - \copybrief module_structurerad
* \subpage module_secondorder - \copybrief module_secondorder
* \subpage module_dynamicpol - \copybrief module_dynamicpol

And writing your own custom modules:

* \subpage modules_custom - \copybrief modules_custom
