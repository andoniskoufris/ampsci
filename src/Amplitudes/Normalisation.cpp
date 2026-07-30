#include "Amplitudes/Normalisation.hpp"
#include "MBPT/CorrelationPotential.hpp"
#include "Wavefunction/DiracSpinor.hpp"

namespace Amplitudes {

//==============================================================================
double dSigma_dE(const DiracSpinor &v, const MBPT::CorrelationPotential &Sigma0,
                 const MBPT::CorrelationPotential &Sigma_plus,
                 const MBPT::CorrelationPotential &Sigma_minus, double delta) {

  const auto *S1 = Sigma_plus.getSigma(v.kappa(), v.n());
  const auto *S2 = Sigma_minus.getSigma(v.kappa(), v.n());
  if (S1 == nullptr || S2 == nullptr)
    return 0.0;

  const auto lambda = Sigma0.getLambda(v.kappa(), v.n());
  return lambda * (v * ((*S1 - *S2) * v)) / (2.0 * delta);
}

} // namespace Amplitudes
