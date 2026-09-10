#include "GreenQED.hpp"
#include "DiracOperator/include.hpp" //For E1 operator
#include "IO/InputBlock.hpp"
#include "MBPT/Feynman.hpp"
#include "MBPT/SpinorMatrix.hpp"
#include "Physics/UnitConv_conversions.hpp"
#include "Wavefunction/Wavefunction.hpp"
#include "fmt/format.hpp"
#include <complex>
#include <memory>

namespace Module {

// lambda for sign
auto sign = [](const double &x) { return x < 0 ? -1.0 : (x > 0 ? 1.0 : 0.0); };

double rho(const double &p, const double &E) {
  // dimensionless combination
  // rho = (m^2 - p^2) / m^2
  //     = (m^2 - E^2 + |p|^2) / m^2,
  // where E = e_v is the physical energy and |p| is the magnitude of the 3-momentum
  // note: |p| is an integration variable, and is not on the mass-shell

  // to do things in atomic units, E is assumed already to be in atomic units
  // n.b. m_e = 1 and c = 1/α in a.u.
  //  - E and m_e * c^2 have the same units; this means that E^2 and m^2 must
  //    be added together as e_v^2 + 1/α^2
  //  - E and p * c have the same units (I am letting p be dimensionless); E^2
  //    and p^2 must be added as E^2 + p^2/α^2
  // thus, rho = (α^{-4} - E^2 + p^2 * α^{-2})/α^{-4}
  //           = 1 - α^4 * E^2 + α^2 * p^2
  // note: to compare to Shabaev should add mc^2 to e_v (since ampsci subtracts
  //       rest mass from H), so E = e_v + m_e * c^2 = e_v + 1/α^2 (in a.u.)

  const auto alphan2 = 1.0 / (PhysConst::alpha2);
  const auto alpha4 = PhysConst::alpha2 * PhysConst::alpha2;
  const auto Enu = E + alphan2;
  const auto Enu2 = alpha4 * Enu * Enu;
  const auto pnu = PhysConst::alpha * p;

  return 1.0 - Enu2 + pnu * pnu;
}

//=============================================================================

DiracSpinor FourierTransformF(const DiracSpinor &F,
                              std::shared_ptr<const Grid> pGrid) {
  // initialise Fourier transform to be on momentum space grid
  DiracSpinor FTransform = DiracSpinor(F.n(), F.kappa(), pGrid);

  const auto grid = F.grid();
  const auto r = F.grid().r();
  const auto p = pGrid->r();

  for (auto i = 0ul; i < pGrid->num_points(); i++) {
    // in atomic units, r is in a.u. in which case what r actually is numerically is r/aB
    // in atomic units aB = 1, but we want p * r to be dimensionless. This is only the case if we actually use alpha * p
    const auto p_i = p[i]; // / PhysConst::alpha;
    const auto s_kappa = sign(F.kappa()) + 0.001;

    for (auto j = F.min_pt(); j < F.max_pt(); j++) {
      FTransform.f(i) += r[j] * F.f(j) *
                         SphericalBessel::JL(F.l(), p_i * r[j]) * grid.drdu(j) *
                         grid.du();
      FTransform.g(i) += r[j] * F.g(j) *
                         SphericalBessel::JL(F.l() - int(s_kappa), p_i * r[j]) *
                         grid.drdu(j) * grid.du();
    }
    FTransform.f(i) *= 4 * M_PI;
    FTransform.g(i) *= 4 * M_PI;
  }

  return FTransform;
}

//=============================================================================

double p_norm(const DiracSpinor &Fa) {

  const auto pGrid = &Fa.grid();
  const auto p = pGrid->r();

  double out = 0.0;

  for (auto i = Fa.min_pt(); i < Fa.max_pt(); i++) {
    const auto p_i = p[i]; // / PhysConst::alpha;

    out += p_i * p_i * (Fa.f(i) * Fa.f(i) + Fa.g(i) * Fa.g(i)) *
           pGrid->drdu(i) * pGrid->du();
  }
  return out / (8.0 * M_PI * M_PI * M_PI);
}

//=============================================================================

double p_braket(const DiracSpinor &Fa, const DiracSpinor &Fb) {
  const auto pGrid = &Fa.grid();
  const auto p = pGrid->r();

  double out = 0.0;

  const auto [i_min, i_max] = std::pair<std::size_t, std::size_t>{
    std::max(Fa.min_pt(), Fb.min_pt()), std::min(Fa.max_pt(), Fb.max_pt())};

  if (i_min >= i_max) {
    return 0.0;
  }

  for (auto i = i_min; i < i_max; i++) {
    out += p[i] * p[i] * (Fa.f(i) * Fa.f(i) + Fa.g(i) * Fa.g(i)) *
           pGrid->drdu(i) * pGrid->du();
  }
  return out / (8.0 * M_PI * M_PI * M_PI);
}

//=============================================================================

double aTerm(const double &rho, const double &m) {

  return 2.0 * m * (1.0 + 2.0 * (rho / (1.0 - rho)) * log(rho));
}

//=============================================================================

double bTerm(const double &rho) {

  return ((rho - 2.0) / (1.0 - rho)) * (1.0 + (rho / (1.0 - rho)) * log(rho));
}

//=============================================================================
void write_orbitals(const std::string &fname,
                    const std::vector<DiracSpinor> &orbs) {
  if (orbs.empty())
    return;
  const auto &gr = orbs.front().grid();

  std::ofstream of(fname);
  of << "r ";
  for (auto &psi : orbs) {
    // of << "r\'$" << psi.symbol(true) << "$" << "\' ";
    of << "$" << psi.symbol(true) << "$ ";
  }
  of << "\n";

  of << "# f block\n";
  for (std::size_t i = 0; i < gr.num_points(); i++) {
    of << gr.r(i) << " ";
    for (auto &psi : orbs) {
      of << psi.f(i) << " ";
    }
    of << "\n";
  }

  of << "\n# g block\n";
  for (std::size_t i = 0; i < gr.num_points(); i++) {
    of << gr.r(i) << " ";
    for (auto &psi : orbs) {
      of << psi.g(i) << " ";
    }
    of << "\n";
  }

  of.close();
  std::cout << "Orbitals written to file: " << fname << "\n";
}

//=============================================================================

double SE_ZeroPotential(const DiracSpinor &F_p, const double &ev,
                        const double &mec2) {
  const auto pGrid = &F_p.grid();
  const auto p = pGrid->r();
  // to compare to Shabaev, E -> E + m_e * c^2 = E + 1/α^2 (in a.u.)
  const auto En = ev + (1.0 / PhysConst::alpha2);

  const auto f_p = F_p.f();
  const auto g_p = F_p.g();

  double E = 0.0;

  for (auto i = F_p.min_pt(); i < F_p.max_pt(); i++) {
    const auto Rho = rho(p[i], ev);
    const auto a_rho = aTerm(Rho, mec2);
    const auto b_rho = bTerm(Rho);

    //! need to check the sign of the last term
    E += (p[i] * p[i] / PhysConst::alpha2) *
         (a_rho * (f_p[i] * f_p[i] - g_p[i] * g_p[i]) +
          b_rho * (En * (f_p[i] * f_p[i] + g_p[i] * g_p[i]) -
                   2 * sign(F_p.kappa()) * (p[i] / PhysConst::alpha) * f_p[i] *
                     g_p[i])) *
         pGrid->drdu(i) * pGrid->du();
  }

  E *= PhysConst::alpha / (32.0 * pow(M_PI, 4));
  // fudge factor (probably a units issue) to get agreement with Shabev paper
  E *= PhysConst::alpha2;

  return E;
}

//=============================================================================

double Feyn_denom(const double &ev, const double &y, const double &q,
                  const double &p, const double &xi) {
  double denom = 0.0;
  denom += y * y * (ev * ev - q * q);
  denom += 2 * y * (1.0 - y) * (ev * ev - p * q * xi);
  denom += (1.0 - y) * (1.0 - y) * (ev * ev - p * p);

  return denom;
}

//=============================================================================

std::pair<double, double> Feyn_denom_zeros(const double &ev, const double &q,
                                           const double &p, const double &xi) {
  const double ev2 = ev * ev;
  const double pq = p * q;
  const double p2 = p * p;
  const double q2 = q * q;
  const double xi2 = xi * xi;

  std::pair<double, double> out;

  const double discr =
    q2 * ev2 - 2.0 * pq * ev2 * xi + p2 * (ev2 + q2 * (xi2 - 1));
  // make sure the zeroes are not in the interval if the discriminant is zero or if \vec{p} = \vec{q}
  if (discr < 0 || (p == q && xi == 1)) {
    out.first = -1.0;
    out.second = -1.0;
    return out;
  }
  const double A = p * (p - q * xi);
  const double D = p2 + q2 - 2.0 * pq * xi;

  if (discr == 0.0) {
    out.first = A / D;
    out.second = out.first;
    return out;
  }

  const auto y1 = (A - sqrt(discr)) / D;
  const auto y2 = (A + sqrt(discr)) / D;

  out.first = y1;
  out.second = y2;

  return out;
}

//=============================================================================

double Y(const double &m, const double &y, const double &ev, const double &q,
         const double &p, const double &xi) {
  double Y = 0.0;

  Y += m * m - y * (ev * ev - q * q) - (1.0 - y) * (ev * ev - p * p);
  Y *= 1.0 / (Feyn_denom(ev, y, p, q, xi));

  return Y;
}

//=============================================================================

double X(const double &m, const double &y, const double &ev, const double &q,
         const double &p, const double &xi) {
  return 1.0 + 1.0 / Y(m, y, ev, q, p, xi);
}

//=============================================================================
// alternative definition of above
double X(const double &Y) { return 1.0 + 1.0 / Y; }

//=============================================================================

double C0_u(const double &y, const double &m, const double &ev, const double &q,
            const double &p, const double &xi) {
  const double XX = X(m, y, ev, q, p, xi);
  const double denom = Feyn_denom(ev, y, q, p, xi);

  return -log(XX) / denom;
}

//=============================================================================

double C11_u(const double &y, const double &m, const double &ev,
             const double &q, const double &p, const double &xi) {
  const double YY = X(m, y, ev, q, p, xi);
  const double XX = X(YY);
  const double denom = Feyn_denom(ev, y, q, p, xi);

  return (1.0 - YY * log(XX)) * y / denom;
}

//=============================================================================

double C12_u(const double &y, const double &m, const double &ev,
             const double &q, const double &p, const double &xi) {
  const double YY = X(m, y, ev, q, p, xi);
  const double XX = X(YY);
  const double denom = Feyn_denom(ev, y, q, p, xi);

  return (1.0 - YY * log(XX)) * (1.0 - y) / denom;
}

//=============================================================================

double C21_u(const double &y, const double &m, const double &ev,
             const double &q, const double &p, const double &xi) {
  const double YY = X(m, y, ev, q, p, xi);
  const double XX = X(YY);
  const double denom = Feyn_denom(ev, y, q, p, xi);

  return (-0.5 + YY - YY * YY * log(XX)) * y * y / denom;
}

//=============================================================================

double C22_u(const double &y, const double &m, const double &ev,
             const double &q, const double &p, const double &xi) {
  const double YY = X(m, y, ev, q, p, xi);
  const double XX = X(YY);
  const double denom = Feyn_denom(ev, y, q, p, xi);

  return (-0.5 + YY - YY * YY * log(XX)) * (1.0 - y) * (1.0 - y) / denom;
}

//=============================================================================

double C23_u(const double &y, const double &m, const double &ev,
             const double &q, const double &p, const double &xi) {
  const double YY = X(m, y, ev, q, p, xi);
  const double XX = X(YY);
  const double denom = Feyn_denom(ev, y, q, p, xi);

  return (-0.5 + YY - YY * YY * log(XX)) * y * (1.0 - y) / denom;
}

//=============================================================================

double C24_u(const double &y, const double &m, const double &q, const double &p,
             const double &xi) {
  const double k2 = 2 * p * q * xi - p * p - q * q;
  const double m2 = m * m;

  const double x = y * (y - 1) * (k2 / m2) + 1.0;

  return -log(x);
}

//=============================================================================

double Cij(const double &m, const double &ev, const double &q, const double &p,
           const double &xi,
           std::function<double(double, double, double, double, double, double)>
             Cij_func,
           double &delta) {
  const double y_min = 0.0;
  const double y_max = 1.0;
  const size_t num_y_points = 1000;
  const double dy = (y_max - y_min) / double(num_y_points);
  double y = y_min;

  const double abs_delta = abs(delta);

  // determine the zeros in the denominator
  const auto [y1_zero, y2_zero] = Feyn_denom_zeros(ev, q, p, xi);

  double out = 0.0;

  // if both zeroes are below y = 0 or above y = 1 then integrate like normal
  if ((y1_zero < 0.0 && y2_zero < 0.0) || (y1_zero > 1.0 && y2_zero > 1.0)) {
    for (auto i = 0ul; i < num_y_points; i++) {
      y += dy;
      out += Cij_func(y, m, ev, q, p, xi) * dy;
    }
  } else if (0.0 < y1_zero && y1_zero < 1.0 && y2_zero > 1.0) {
    // if we have one zero then we avoid that single zero within some radius
    const double y1_lower = y1_zero - abs_delta;
    const double y1_upper = y1_zero + abs_delta;
    for (auto i = 0ul; i < num_y_points; i++) {
      y += dy;
      if (y1_lower < y && y < y1_upper) {
        continue;
      }
      out += Cij_func(y, m, ev, q, p, xi) * dy;
    }
  } else if (y1_zero < 0.0 && 0.0 < y2_zero && y2_zero < 1.0) {
    // if we have one zero then we avoid that single zero within some radius
    const double y2_lower = y2_zero - abs_delta;
    const double y2_upper = y2_zero + abs_delta;
    for (auto i = 0ul; i < num_y_points; i++) {
      y += dy;
      if (y2_lower < y && y < y2_upper) {
        continue;
      }
      out += Cij_func(y, m, ev, q, p, xi) * dy;
    }
  } else {
    // if we have two zeroes we need to avoid both
    const double y1_lower = y1_zero - abs_delta;
    const double y1_upper = y1_zero + abs_delta;
    const double y2_lower = y1_zero - abs_delta;
    const double y2_upper = y1_zero + abs_delta;
    for (auto i = 0ul; i < num_y_points; i++) {
      y += dy;
      if ((y1_lower < y && y < y1_upper) || (y2_lower < y && y < y2_upper)) {
        continue;
      }
      out += Cij_func(y, m, ev, q, p, xi) * dy;
    }
  }

  return out;
}

//=============================================================================

double C24(const double &m, const double &q, const double &p,
           const double &xi) {
  const double y_min = 0.0;
  const double y_max = 1.0;
  const size_t num_y_points = 1000;
  const double dy = (y_max - y_min) / double(num_y_points);
  double y = y_min;

  double out = 0.0;
  for (auto i = 0ul; i < num_y_points; i++) {
    y += dy;
    out += C24_u(y, m, q, p, xi) * dy;
  }

  return out;
}

//=============================================================================
//=============================================================================

void GreenQED(const IO::InputBlock &input, const Wavefunction &wf) {

  input.check(
    {{"r0", "Minimum r to calculate Green's fn [1.0e-3]"}, //
     {"rmax", "Maximum r to calculate Green's fn [50.0]"},
     {"num_points",
      "Number of radial points for Green's function (used for stride). [150]"},
     {"stride", "Explicitely set stride. Will over-ride num_points if set."},
     {"kappa", "Kappa value for initial test [default: -1]"}});

  // If we are just requesting 'help', don't run module:
  if (input.has_option("help")) {
    return;
  }

  // Determine "sub" grid:
  const auto r0_target = input.get("r0", 1.0e-3);
  const auto rmax_target = input.get("rmax", 50.0);

  const std::size_t i0 = wf.grid().getIndex(r0_target);
  const std::size_t imax = wf.grid().getIndex(rmax_target);
  assert(imax > i0 && "Require rmax > r0");

  const std::size_t in_size = input.get("num_points", 150ul);
  const std::optional<std::size_t> in_stride = input.get<std::size_t>("stride");

  std::size_t size{0}, stride{0};
  if (in_stride.has_value()) {
    assert(*in_stride > 0 && "Cannot have stride of zero");
    stride = *in_stride;
    size = (imax - i0) / stride + 1;
  } else {
    assert(in_size > 1 && "Cannot have num_points <= 1");
    size = in_size;
    stride = std::max(1ul, (imax - i0) / (size - 1));
  }
  assert(size > 1 && stride > 0);

  // actual r0,rmax might be slightly different, due to finite grid, stride
  const auto grid = wf.grid();
  const auto r0 = wf.grid().r(i0);
  const auto rmax = wf.grid().r(i0 + stride * size);
  fmt::print(
    "Grid for Green's function: {:.1e} - {:.1f} with {} points [stride = {}]\n",
    r0, rmax, size, stride);

  // We don't need QPQ (at least for now)
  bool construct_qpq = false;
  bool verbose = false;
  // I think this only affects Sigma anyway:
  bool include_G = true;

  MBPT::Feynman Fy(wf.vHF(), i0, stride, size, {}, 1, include_G, verbose, "",
                   construct_qpq);

  int kappa = input.get("kappa", -1);
  std::complex<double> en{-0.5, 0.1};

  // regular method:
  const auto g1 = Fy.green(kappa, en);

  // Use basis:
  const auto g2 = Fy.green_basis(kappa, en, wf.basis());

  // Solve Dirac eq. with complex energy directly (never checked if this works)
  const auto g3 = Fy.green_v2(kappa, en);

  // Simple quick test:
  // <v|G(e)|v> = sum_n <v|n><v|n> / (e - en) = 1 / (e - ev)

  // ampsci can do: G|v> which returns a DiracSpinor
  // G does not include integration measure, so we must include it
  // If G is complex, returns a pair: G*F = {re(GF), im(GF)}

  std::cout << "\nState, <v|G|>, expected, error\n\n";
  for (const auto &[name, g_ptr] : {std::pair{"Normal Green's fn", &g1},
                                    {"Basis Green's fn", &g2},
                                    {"Complex Schrodinger Green's fn", &g3}}) {

    std::cout << "For " << name << "\n";
    const auto &gt = *g_ptr;
    for (const auto &v : wf.valence()) {

      if (v.kappa() != kappa)
        continue;

      const auto [re_Gv, im_Gv] = gt.drj() * v;
      const std::complex<double> value = {v * re_Gv, v * im_Gv};
      const auto expected = 1.0 / (en - v.en());

      auto eps = std::abs(value - expected) / std::abs(expected);

      std::cout << v << " " << value << " " << expected << " " << eps << "\n";
    }
    std::cout << "\n";
  }

  //===========================================================================
  // Calculating self-energy corrections
  std::cout << std::endl;
  std::cout << "Calculating electron self-energy\n";
  std::cout << "with momentum grid parameters:\n\n";

  // conversion factor from k in whatever units I am doing now to a.u.
  const double p_to_au = 1.0;

  // momentum grid parameters
  // right now these are in some units that I don't know
  const auto p_num_points = 5000;
  const auto p_min = p_to_au * 1.0e-4;
  const auto p_max = p_to_au * 1.0e3;
  const auto p_b = 4.0;
  const auto p_grid_type = "loglinear";
  const auto p_indu = 0.0; // shouldn't worry about this

  std::cout << "Grid type   = " << p_grid_type << "\n";
  std::cout << "Num. points = " << p_num_points << "\n";
  std::cout << "p minimum   = " << p_min << "\n";
  std::cout << "p maximum   = " << p_max << "\n";
  std::cout << "b           = " << p_b << "\n\n";

  // initialise momentum-space grid
  const auto pGrid = std::make_shared<const Grid>(
    GridParameters{p_num_points, p_min, p_max, p_b, p_grid_type, p_indu});
  const auto p = pGrid->r();

  // const auto mec2 = 1.0 / (PhysConst::alpha * PhysConst::alpha);

  // std::vector<DiracSpinor> orbs;

  // fmt::print("{:<5s} {:>10s} {:>14s} {:>14s} {:>13s} {:>13s}\n", "State",
  //            "<v|v>", "HF", "\u03A3(0)", "\u03A3(1)", "\u03A3(2)");

  // for (const auto &v : wf.valence()) {

  //   // // if I only want to do the calculations for a particular value of n and l (or any other Q numbers)
  //   // if (v.n() != 20 || v.kappa() > 0) {
  //   //   continue;
  //   // }

  //   const auto vtild = FourierTransformF(v, pGrid);
  //   orbs.push_back(vtild);
  //   const auto vp_norm = p_norm(vtild);
  //   const auto ev = v.en();
  //   double E = SE_ZeroPotential(vtild, ev, mec2);

  //   // convert to atomic units (I think it's in atomic units already?) and then print
  //   fmt::print("{:<5}  {:>+7.6f}  {:>+7.7f}  {:>+7.7f}  {:>+7.7f}  {:>+7.7f}\n",
  //              v.shortSymbol(), vp_norm, v.en(), E, grid.r(v.min_pt()),
  //              grid.r(v.max_pt() - 1));
  // }

  // std::cout << std::endl;

  // write_orbitals(wf.identity() + "qed.pwf.txt", orbs);

  // Testing one-potential term

  const auto [y1, y2] = Feyn_denom_zeros(-3.0, 3.6, 12.0, 0.27);
  std::cout << "y1 = " << y1
            << " ; D(y1) = " << Feyn_denom(-3.0, y1, 3.6, 12.0, 0.27)
            << std::endl;
  std::cout << "y2 = " << y2
            << " ; D(y2) = " << Feyn_denom(-3.0, y2, 3.6, 12.0, 0.27) << "\n";
}

} // namespace Module
