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
void write_orbitals(const std::string &fname,
                    const std::vector<DiracSpinor> &orbs);

double SE_ZeroPotential(const DiracSpinor &F_p, const double &ev,
                        const double &mec2);

double Feyn_denom(const double &ev, const double &y, const double &q,
                  const double &p, const double &xi);

double Y(const double &m, const double &y, const double &ev, const double &q,
         const double &p, const double &xi);

double X(const double &m, const double &y, const double &ev, const double &q,
         const double &p, const double &xi);

void GreenQED(const IO::InputBlock &input, const Wavefunction &wf);

} // namespace Module
