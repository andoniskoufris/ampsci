#pragma once
#include "Angular/Wigner369j.hpp"
#include "Physics/AtomData.hpp"
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

class OnePotIntegrals {
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
  OnePotIntegrals(const DiracSpinor &Fv, const double &q, const double &p,
                  const double &p_to_index, const double &ev, const double &m,
                  const double &v, const DiracSpinor &Fp);

  OnePotIntegrals(const double &f_q, const double &g_q, const double &f_p,
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
  const DiracSpinor m_Fr;
  const DiracSpinor m_Fp;

public:
  v_Params(const double &ev, const double &m, const double &q, const double &p,
           const double &p_to_index, const DiracSpinor &Fr,
           const DiracSpinor &Fp)
    : m_ev(ev),
      m_me(m),
      m_q(q),
      m_p(p),
      m_p_to_index(p_to_index),
      m_Fr(Fr),
      m_Fp(Fp) {}

  const double &m() { return m_me; }
  const double &ev() { return m_ev; }
  const double &q() { return m_q; }
  const double &p() { return m_p; }
  const double &p_to_index() { return m_p_to_index; }
  const DiracSpinor &Fr() { return m_Fr; }
  const DiracSpinor &Fp() { return m_Fp; }
};

//=============================================================================

struct p_Params {
private:
  const double m_ev;
  const double m_me;
  const double m_q;
  const double m_p_to_index;
  const DiracSpinor m_Fr;
  const DiracSpinor m_Fp;

public:
  p_Params(const double &ev, const double &m, const double &q,
           const double &p_to_index, const DiracSpinor &Fr,
           const DiracSpinor &Fp)
    : m_ev(ev), m_me(m), m_q(q), m_p_to_index(p_to_index), m_Fr(Fr), m_Fp(Fp) {}

  const double &m() { return m_me; }
  const double &ev() { return m_ev; }
  const double &q() { return m_q; }
  const double &p_to_index() { return m_p_to_index; }
  const DiracSpinor &Fr() { return m_Fr; }
  const DiracSpinor &Fp() { return m_Fp; }
};

//=============================================================================

struct q_Params {
private:
  const double m_ev;
  const double m_me;
  const double m_p_to_index;
  const DiracSpinor m_Fr;
  const DiracSpinor m_Fp;

public:
  q_Params(const double &ev, const double &m, const double &p_to_index,
           const DiracSpinor &Fr, const DiracSpinor &Fp)
    : m_ev(ev), m_me(m), m_p_to_index(p_to_index), m_Fr(Fr), m_Fp(Fp) {}

  const double &m() { return m_me; }
  const double &ev() { return m_ev; }
  const double &p_to_index() { return m_p_to_index; }
  const DiracSpinor &Fr() { return m_Fr; }
  const DiracSpinor &Fp() { return m_Fp; }
};

class QuadCoeffs {
private:
  int m_n;                       // Number of Gaussian points
  std::vector<double> m_roots;   // vector with roots
  std::vector<double> m_weights; // vector with weights

public:
  QuadCoeffs(int num_points)
    : m_n(num_points),
      m_roots(std::vector<double>(86)),
      m_weights(std::vector<double>(86)) {
    m_roots = {-0.9996135688,  -0.9979644770,  -0.9949999175,  -0.9907231407,
               -0.9851397067,  -0.9782569603,  -0.9700839727,  -0.9606315209,
               -0.9499120709,  -0.9379397599,  -0.9247303783,  -0.9103013480,
               -0.8946716997,  -0.8778620476,  -0.8598945623,  -0.8407929415,
               -0.8205823788,  -0.7992895303,  -0.7769424796,  -0.7535707009,
               -0.7292050197,  -0.7038775725,  -0.6776217642,  -0.6504722243,
               -0.6224647610,  -0.5936363139,  -0.5640249056,  -0.5336695914,
               -0.5026104074,  -0.4708883186,  -0.4385451637,  -0.4056236012,
               -0.3721670519,  -0.3382196425,  -0.3038261471,  -0.2690319281,
               -0.2338828765,  -0.1984253510,  -0.1627061176,  -0.1267722872,
               -0.09067125382, -0.05445063189, -0.01815819372, 0.01815819372,
               0.05445063189,  0.09067125382,  0.1267722872,   0.1627061176,
               0.1984253510,   0.2338828765,   0.2690319281,   0.3038261471,
               0.3382196425,   0.3721670519,   0.4056236012,   0.4385451637,
               0.4708883186,   0.5026104074,   0.5336695914,   0.5640249056,
               0.5936363139,   0.6224647610,   0.6504722243,   0.6776217642,
               0.7038775725,   0.7292050197,   0.7535707009,   0.7769424796,
               0.7992895303,   0.8205823788,   0.8407929415,   0.8598945623,
               0.8778620476,   0.8946716997,   0.9103013480,   0.9247303783,
               0.9379397599,   0.9499120709,   0.9606315209,   0.9700839727,
               0.9782569603,   0.9851397067,   0.9907231407,   0.9949999175,
               0.9979644770,   0.9996135688};
    m_weights = {
      0.0009916432666, 0.002307087489, 0.003621439250, 0.004931184097,
      0.006234459139,  0.007529521612, 0.008814657102, 0.01008816846,
      0.01134837516,   0.01259361468,  0.01382224445,  0.01503264391,
      0.01622321655,   0.01739239208,  0.01853862841,  0.01966041373,
      0.02075626848,   0.02182474732,  0.02286444098,  0.02387397819,
      0.02485202744,   0.02579729877,  0.02670854542,  0.02758456553,
      0.02842420370,   0.02922635251,  0.02998995397,  0.03071400097,
      0.03139753854,   0.03203966513,  0.03263953385,  0.03319635349,
      0.03370938967,   0.03417796573,  0.03460146364,  0.03497932485,
      0.03531105099,   0.03559620454,  0.03583440939,  0.03602535138,
      0.03616877867,   0.03626450208,  0.03631239538,  0.03631239538,
      0.03626450208,   0.03616877867,  0.03602535138,  0.03583440939,
      0.03559620454,   0.03531105099,  0.03497932485,  0.03460146364,
      0.03417796573,   0.03370938967,  0.03319635349,  0.03263953385,
      0.03203966513,   0.03139753854,  0.03071400097,  0.02998995397,
      0.02922635251,   0.02842420370,  0.02758456553,  0.02670854542,
      0.02579729877,   0.02485202744,  0.02387397819,  0.02286444098,
      0.02182474732,   0.02075626848,  0.01966041373,  0.01853862841,
      0.01739239208,   0.01622321655,  0.01503264391,  0.01382224445,
      0.01259361468,   0.01134837516,  0.01008816846,  0.008814657102,
      0.007529521612,  0.006234459139, 0.004931184097, 0.003621439250,
      0.002307087489,  0.0009916432666};
  }

  const std::vector<double> &roots() { return m_roots; }
  const std::vector<double> &weights() { return m_weights; }
};

//=============================================================================

//=============================================================================

class SpinorTransform {
  using Index = uint16_t;

private:
  int m_n;
  int m_kappa;
  double m_en;
  std::vector<double> m_f;
  std::vector<double> m_g;
  std::vector<double> m_p_grid;
  std::vector<double> m_quad_grid;
  std::vector<double> m_weights;
  // 2j, l, pi, kappa_index (for convenience): these are defined by n and kappa
  int m_twoj;
  int m_l;
  int m_parity;
  int m_kappa_index;
  Index m_nkappa_index;

public:
  SpinorTransform(int in_n, int kappa, double en,
                  const std::vector<double> &p_grid,
                  const std::vector<double> &quad_points,
                  const std::vector<double> &weights)
    : m_n(in_n),
      m_kappa(kappa),
      m_en(en),
      m_f(std::vector<double>(quad_points.size(), 0.0)),
      m_g(std::vector<double>(quad_points.size(), 0.0)),
      m_p_grid(p_grid),
      m_quad_grid(quad_points),
      m_weights(weights),
      m_twoj(Angular::twoj_k(kappa)),
      m_l(Angular::l_k(kappa)),
      m_parity(Angular::parity_k(kappa)),
      m_kappa_index(Angular::indexFromKappa(kappa)),
      m_nkappa_index(static_cast<Index>(Angular::nk_to_index(in_n, kappa))) {};

  std::vector<double> &f() { return m_f; }
  std::vector<double> &g() { return m_g; }
  const std::vector<double> &p_grid() const { return m_p_grid; }
  const std::vector<double> &quad_grid() const { return m_quad_grid; }
  const std::vector<double> &weights() const { return m_weights; }

  const std::vector<double> &f() const { return m_f; }
  const std::vector<double> &g() const { return m_g; }

  double &f(std::size_t i) { return m_f[i]; }
  double &g(std::size_t i) { return m_g[i]; }

  double f(std::size_t i) const { return m_f[i]; }
  double g(std::size_t i) const { return m_g[i]; }

  const double &p0() const { return m_p_grid[0]; }
  const double &pmax() const { return m_p_grid.back(); }

  std::string symbol(bool gnuplot) const {
    // Readable symbol (s_1/2, p_{3/2} etc.).
    // gnuplot-firndly '{}' braces optional.
    std::string ostring1 = (m_n != 0) ?
                             std::to_string(m_n) + AtomData::l_symbol(m_l) :
                             AtomData::l_symbol(m_l);
    std::string ostring2 = gnuplot ? "_{" + std::to_string(m_twoj) + "/2}" :
                                     "_" + std::to_string(m_twoj) + "/2";
    return ostring1 + ostring2;
  }

  static std::string shortSymbol(int n, int kappa) {
    const std::string pm = (kappa < 0) ? "+" : "-";
    int l = Angular::l_k(kappa);
    return (n != 0) ? std::to_string(n) + AtomData::l_symbol(l) + pm :
                      AtomData::l_symbol(l) + pm;
  }

  std::string shortSymbol() const { return shortSymbol(m_n, m_kappa); }
};

} // namespace Module
