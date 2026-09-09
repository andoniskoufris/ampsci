#pragma once
#include <memory>
#include <vector>

// Forward declare classes:
class Wavefunction;
class DiracSpinor;
class Grid;
namespace IO {
class InputBlock;
}

namespace Module {

DiracSpinor FourierTransformF(const DiracSpinor &F,
                              std::shared_ptr<const Grid> pGrid);

double p_norm(const DiracSpinor &Fa);
double p_braket(const DiracSpinor &Fa, const DiracSpinor &Fb);

double rho(const double &p, const double &E);

double aTerm(const double &rho, const double &m);
double bTerm(const double &rho);

double SE_ZeroPotential(const DiracSpinor &F_p, const double &mec2);

void write_orbitals(const std::string &fname,
                    const std::vector<DiracSpinor> &orbs);

void GreenQED(const IO::InputBlock &input, const Wavefunction &wf);

} // namespace Module
