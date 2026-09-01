#include "GreenQED.hpp"
#include "DiracOperator/include.hpp" //For E1 operator
#include "IO/InputBlock.hpp"
#include "MBPT/Feynman.hpp"
#include "MBPT/SpinorMatrix.hpp"
#include "Physics/UnitConv_conversions.hpp"
#include "Wavefunction/Wavefunction.hpp"
#include "fmt/format.hpp"
#include <complex>

DiracSpinor FourierTransformF(const DiracSpinor &F,
                              std::shared_ptr<const Grid> pGrid);

double p_norm(const DiracSpinor &Fa);
double p_braket(const DiracSpinor &Fa, const DiracSpinor &Fb);

double rho(const double &p, const double &E);
double aTerm(const double &p, const double &E);
double bTerm(const double &p, const double &E);

double aTerm_rho(const double &rho, const double &m);
double bTerm_rho(const double &rho);

// lambda for sign
auto sign = [](const double &x) { return x < 0 ? -1.0 : (x > 0 ? 1.0 : 0.0); };

namespace Module {

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
  //===========================================================================
  // Calculating self-energy corrections
  std::cout << std::endl << std::endl;
  std::cout << "Calculating electron self-energy" << std::endl << std::endl;
  fmt::print("{:<5s} {:>10s} {:>14s} {:>14s} {:>13s} {:>13s}\n", "State",
             "<v|v>", "HF", "\u03A3(0)", "\u03A3(1)", "\u03A3(2)");

  // initialise momentum-space grid
  const auto pGrid = std::make_shared<const Grid>(
    GridParameters{10000, 1.0e-2, 2.0e6, 4.0, "linear", 0.0});
  const auto p = pGrid->r();

  const auto mec2 = 1.0 / (PhysConst::alpha * PhysConst::alpha);

  for (const auto &v : wf.valence()) {

    const auto FourierFv = FourierTransformF(v, pGrid);
    const auto vp_norm = p_norm(FourierFv);
    const auto ev = v.en();
    double E = 0.0;

    for (auto i = 0ul; i < pGrid->num_points(); i++) {
      // to compare to Shabev, E -> E + m_e * c^2 = E + 1/α^2 (in a.u.)
      const auto En = ev + (1.0 / PhysConst::alpha2);
      const auto Rho = rho(p[i], ev);
      const auto a_rho = aTerm_rho(Rho, mec2);
      const auto b_rho = bTerm_rho(Rho);

      E += (p[i] * p[i] / PhysConst::alpha2) *
           (a_rho * (FourierFv.f(i) * FourierFv.f(i) -
                     FourierFv.g(i) * FourierFv.g(i)) +
            b_rho * (En * (FourierFv.f(i) * FourierFv.f(i) +
                           FourierFv.g(i) * FourierFv.g(i)) -
                     2 * sign(v.kappa()) * (p[i] / PhysConst::alpha) *
                       FourierFv.f(i) * FourierFv.g(i))) *
           pGrid->drdu(i) * pGrid->du();
    }

    E *= PhysConst::alpha / (32.0 * pow(M_PI, 4));

    // convert to atomic units (I think it's in atomic units already?) and then print
    fmt::print("{:<5}  {:>+7.6f}  {:>+7.7f}  {:>+7.7f}  {:>+7.7f}  {:>+7.7f}\n",
               v.shortSymbol(), vp_norm, v.en(), E, grid.r(v.min_pt()),
               grid.r(v.max_pt()));
  }

  std::cout << std::endl;
}

} // namespace Module

//=============================================================================
//=============================================================================

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
  DiracSpinor FTransform(F.n(), F.kappa(), pGrid);

  const auto grid = F.grid();
  const auto r = F.grid().r();
  const auto p = pGrid->r();

  for (auto i = 0ul; i < pGrid->num_points(); i++) {
    // in atomic units, r is in a.u. in which case what r actually is numerically is r/aB
    // in atomic units aB = 1, but we want p * r to be dimensionless. This is only the case if we actually use alpha * p
    const auto p_i = p[i]; // / PhysConst::alpha;
    const auto s_kappa = sign(F.kappa());

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
  return out / (8.0 * M_PI_2 * M_PI);
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
  return out / (8.0 * M_PI_2 * M_PI);
}

//=============================================================================

double aTerm(const double &p, const double &E) {
  // put E and p into same units and then add back rest mass energy
  const auto alphan2 = 1.0 / (PhysConst::alpha2);
  const auto Enu = PhysConst::alpha2 * (E + alphan2);
  const auto pnu = PhysConst::alpha * p;

  return 2.0 *
         (1.0 + 2.0 *
                  ((1.0 - Enu * Enu + pnu * pnu) / (Enu * Enu - pnu * pnu)) *
                  log(1.0 - Enu * Enu + pnu * pnu));
}

//=============================================================================

double bTerm(const double &p, const double &E) {
  // put E and p into same units and then add back rest mass energy
  const auto alphan2 = 1.0 / (PhysConst::alpha2);
  const auto Enu = PhysConst::alpha2 * (E + alphan2);
  const auto pnu = PhysConst::alpha * p;

  return ((-1.0 - Enu * Enu + pnu * pnu) / (Enu * Enu - pnu * pnu)) *
         (1.0 + ((1.0 - Enu * Enu + pnu * pnu) / (Enu * Enu - pnu * pnu)) *
                  log(1.0 - Enu * Enu + pnu * pnu));
}

//=============================================================================

double aTerm_rho(const double &rho, const double &m) {

  return 2.0 * m * (1.0 + 2.0 * (rho / (1.0 - rho)) * log(rho));
}

//=============================================================================

double bTerm_rho(const double &rho) {

  return ((rho - 2.0) / (1.0 - rho)) * (1.0 + (rho / (1.0 - rho)) * log(rho));
}