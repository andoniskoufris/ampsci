#pragma once
class DiracSpinor;
namespace MBPT {
class CorrelationPotential;
}

namespace Amplitudes {

/*!
  @brief Normalisation correction factor for a valence state, from the
  energy derivative of the correlation potential.
  @details
  The normalisation of a Brueckner orbital differs from unity at second
  order; the correction factor for state \f$ v \f$ is

  \f[
    \frac{{\rm d}\Sigma_v}{{\rm d}\en} = \lambda_v
      \frac{\matel{v}{\Sigma(\en_v+\delta) - \Sigma(\en_v-\delta)}{v}}
           {2\delta},
  \f]

  evaluated by central difference, with \f$ \lambda_v \f$ the scaling
  factor of the correlation potential. The normalisation correction to a
  matrix element is then
  \f$ \delta t^{\rm Norm}_{ab} = \frac{1}{2}(t_{ab} + \delta V_{ab})
  ({\rm d}\Sigma_a/{\rm d}\en + {\rm d}\Sigma_b/{\rm d}\en) \f$.
  This is the non-perturbative alternative to the sum-over-states
  normalisation of MBPT::StructureRad::norm.

  @param v           Valence state.
  @param Sigma0      The correlation potential of the wavefunction (supplies
                     the lambda scaling factor).
  @param Sigma_plus  Correlation potential formed at e_v + delta.
  @param Sigma_minus Correlation potential formed at e_v - delta.
  @param delta       The energy step the potentials were formed at.
  @return d Sigma_v / d e (dimensionless; zero if the potentials hold no
          Sigma of this kappa).

  @note getSigma falls back to the nearest-n Sigma of matching kappa, so
        a state the potentials were not formed for still returns a value
        (evaluated with that Sigma).
*/
[[nodiscard]] double dSigma_dE(const DiracSpinor &v,
                               const MBPT::CorrelationPotential &Sigma0,
                               const MBPT::CorrelationPotential &Sigma_plus,
                               const MBPT::CorrelationPotential &Sigma_minus,
                               double delta);

} // namespace Amplitudes
