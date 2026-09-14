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

double SE_OnePotential(const DiracSpinor &F_p, const double &ev,
                       const double &mec2, const size_t &xi_num_points,
                       const size_t &num_y_pts);

double Feyn_denom(const double &ev, const double &y, const double &q,
                  const double &p, const double &xi);

double Y(const double &m, const double &y, const double &ev, const double &q,
         const double &p, const double &xi);

double X(const double &m, const double &y, const double &ev, const double &q,
         const double &p, const double &xi);

void GreenQED(const IO::InputBlock &input, const Wavefunction &wf);

class OnePotential {
private:
  double m_c0;
  double m_c11, m_c12;
  double m_c21, m_c22, m_c23;
  double m_c24;
  double m_a;
  double m_b1, m_b2;
  double m_c1, m_c2;
  double m_d;
  double m_h1, m_h2;
  double m_F1, m_F2;

  //separately pass in q and its index, and same for p, so that we can multiply p and q without messing up the indexing
public:
  OnePotential(const DiracSpinor &Fv, const double &q, const size_t &q_i,
               const double &p, const size_t &p_i, const double &xi,
               const double &ev, const double &m, const size_t &num_y_pts,
               const double &y_delta);

  double f1() { return m_F1; }
  double f2() { return m_F2; }
};

struct OnePotentialParams {
private:
  const double m_me;
  const double m_ev;
  const double m_q;
  const double m_p;
  const double m_xi;

public:
  OnePotentialParams(const double &m, const double &ev, const double &q,
                     const double &p, const double &xi)
    : m_me(m), m_ev(ev), m_q(q), m_p(p), m_xi(xi) {}

  double m() const { return m_me; }
  double ev() const { return m_ev; }
  double q() const { return m_q; }
  double p() const { return m_p; }
  double xi() const { return m_xi; }
};

} // namespace Module
