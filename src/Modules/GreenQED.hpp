#pragma once
#include "Wavefunction/DiracSpinor.hpp"
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

// double SE_OnePotential(const DiracSpinor &F_p, const double &ev,
//                        const double &mec2, const size_t &xi_num_points,
//                        const size_t &num_y_pts);

// double Feyn_denom(const double &y, const double &ev, const double &m,
//                   const double &q, const double &p, const double &v);

// double Y(const double &m, const double &y, const double &ev, const double &q,
//          const double &p, const double &xi);

// double X(const double &m, const double &y, const double &ev, const double &q,
//          const double &p, const double &xi);

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
  OnePotential(const DiracSpinor &Fv, const double &q, const double &p,
               const double &p_to_index, const double &ev, const double &m,
               const double &v);

  OnePotential(const double &f_q, const double &g_q, const double &f_p,
               const double &g_p, const double &q, const size_t &q_i,
               const double &p, const size_t &p_i, const double &ev,
               const double &m, const double &v);

  double f1() { return m_F1; }
  double f2() { return m_F2; }
};

//=============================================================================

struct v_Params {
private:
  const double m_ev;
  const double m_me;
  const double m_q;
  const double m_p;
  const double m_p_to_index;
  const DiracSpinor m_Fv;

public:
  v_Params(const double &ev, const double &m, const double &q, const double &p,
           const double &p_to_index, const DiracSpinor &Fv)
    : m_ev(ev), m_me(m), m_q(q), m_p(p), m_p_to_index(p_to_index), m_Fv(Fv) {}

  const double &m() { return m_me; }
  const double &ev() { return m_ev; }
  const double &q() { return m_q; }
  const double &p() { return m_p; }
  const double &p_to_index() { return m_p_to_index; }
  const DiracSpinor &Fv() { return m_Fv; }
};

//=============================================================================

struct p_Params {
private:
  const double m_ev;
  const double m_me;
  const double m_q;
  const double m_p_to_index;
  const DiracSpinor m_Fv;

public:
  p_Params(const double &ev, const double &m, const double &q,
           const double &p_to_index, const DiracSpinor &Fv)
    : m_ev(ev), m_me(m), m_q(q), m_p_to_index(p_to_index), m_Fv(Fv) {}

  const double &m() { return m_me; }
  const double &ev() { return m_ev; }
  const double &q() { return m_q; }
  const double &p_to_index() { return m_p_to_index; }
  const DiracSpinor &Fv() { return m_Fv; }
};

//=============================================================================

struct q_Params {
private:
  const double m_ev;
  const double m_me;
  const double m_p_to_index;
  const DiracSpinor m_Fv;

public:
  q_Params(const double &ev, const double &m, const double &p_to_index,
           const DiracSpinor &Fv)
    : m_ev(ev), m_me(m), m_p_to_index(p_to_index), m_Fv(Fv) {}

  const double &m() { return m_me; }
  const double &ev() { return m_ev; }
  const double &p_to_index() { return m_p_to_index; }
  const DiracSpinor &Fv() { return m_Fv; }
};

} // namespace Module
