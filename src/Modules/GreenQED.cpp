#include "GreenQED.hpp"
#include "DiracOperator/include.hpp" //For E1 operator
#include "IO/InputBlock.hpp"
#include "MBPT/Feynman.hpp"
#include "MBPT/SpinorMatrix.hpp"
#include "Physics/UnitConv_conversions.hpp"
#include "Wavefunction/Wavefunction.hpp"
#include "fmt/format.hpp"
#include <complex>

namespace Module {

DiracSpinor FourierTransformF(const DiracSpinor &F,
                              std::shared_ptr<const Grid> pGrid);

double aTerm(const double &p, const double &E);
double bTerm(const double &p, const double &E);

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
    stride = stride = std::max(1ul, (imax - i0) / (size - 1));
  }
  assert(size > 1 && stride > 0);

  // actual r0,rmax might be slightly different, due to finite grid, stride
  const auto grid = wf.grid();
  const auto r0 = wf.grid().r(i0);
  const auto rmax = wf.grid().r(i0 + stride * size);
  fmt::print(
    "Grid for Green's function: {:.1e} - {:.1f} with {} points [stide = {}]\n",
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

  //========================================== CALCULATING SELF-ENERGY CORRECTIONS

  std::cout << std::endl << std::endl;

  std::cout << "Calculating electron self-energy" << std::endl << std::endl;
  std::cout << "State " << "  \u03A3(0) " << "   "
            << "  \u03A3(1) " << "    " << "  \u03A3(2+) " << std::endl;

  //! initialise momentum-space grid
  const auto pGrid = std::make_shared<const Grid>(
    GridParameters{2000, 1e-7, 600.0, 4.0, "logarithmic", 0.0});
  const auto p = pGrid->r();

  for (const auto &v : wf.valence()) {

    const auto FourierFv = FourierTransformF(v, pGrid);
    const auto ev = v.en();
    double E = 0.0;

    for (int i = 0; i < pGrid->num_points(); i++) {
      E +=
        PhysConst::alpha * p[i] * PhysConst::alpha * p[i] *
        (aTerm(p[i], ev) *
           (FourierFv.f(i) * FourierFv.f(i) - FourierFv.g(i) * FourierFv.g(i)) +
         bTerm(p[i], ev) *
           ((PhysConst::alpha2 * ev + 1.0) * (FourierFv.f(i) * FourierFv.f(i) +
                                              FourierFv.g(i) * FourierFv.g(i)) +
            2 * PhysConst::alpha * p[i] * FourierFv.f(i) * FourierFv.g(i))) *
        pGrid->drdu(i) * pGrid->du();
    }

    E *= PhysConst::alpha / (32.0 * pow(M_PI, 4));

    // convert to atomic units and then print
    std::cout << v.shortSymbol() << "     " << UnitConv::Energy_invcm_to_au * E
              << std::endl;
  }

  std::cout << std::endl;

  const auto Fv1s = wf.valence()[0];
  const auto FvTransform1s = FourierTransformF(Fv1s, pGrid);

  const auto Fv5g = wf.valence()[22];
  const auto FvTransform5g = FourierTransformF(Fv5g, pGrid);

  //!Fourier wave function tests

  // std::cout << "p         1s+ wf                         5g+ wf" << std::endl;

  // for (int i = 0; i < int(pGrid->num_points() / 10); i++) {
  //   std::cout << p[10 * i] << "  (" << FvTransform1s.f(10 * i) << ", "
  //             << FvTransform1s.g(10 * i) << ")        ("
  //             << FvTransform5g.f(10 * i) << ", " << FvTransform5g.g(10 * i)
  //             << ")" << std::endl;
  // }

  // std::cout << PhysConst::alpha2 * Fv.en() * Fv.en() << std::endl;

  // for (int i = 0; i < int(pGrid->num_points() / 10); i++) {
  //   std::cout << pGrid->r(10 * i) << "   " << FvTransform.f(10 * i) << "   "
  //             << FvTransform.g(10 * i) << "   "
  //             << aTerm(pGrid->r(10 * i), Fv.en()) << "   "
  //             << bTerm(pGrid->r(10 * i), Fv.en()) << std::endl;
  // }
}

//!============================== FUNCTIONS

DiracSpinor FourierTransformF(const DiracSpinor &F,
                              std::shared_ptr<const Grid> pGrid) {
  // initialise Fourier transform to be on momentum space grid
  DiracSpinor FTransform(F.n(), F.kappa(), pGrid);

  const auto grid = F.grid();
  const auto r = F.grid().r();
  const auto p = pGrid->r();

  for (int i = 0; i < pGrid->num_points(); i++) {
    for (int j = F.min_pt(); j < F.max_pt(); j++) {
      FTransform.f(i) += r[j] * F.f(j) *
                         SphericalBessel::JL(F.l(), p[i] * r[j]) *
                         grid.drdu(j) * grid.du();
      if (FTransform.kappa() < 0) {
        FTransform.g(i) += r[j] * F.g(j) *
                           SphericalBessel::JL(F.l() + 1, p[i] * r[j]) *
                           grid.drdu(j) * grid.du();
      } else {
        FTransform.g(i) += r[j] * F.g(j) *
                           SphericalBessel::JL(F.l() - 1, p[i] * r[j]) *
                           grid.drdu(j) * grid.du();
      }
    }
    FTransform.f(i) *= 4 * M_PI;
    FTransform.g(i) *= 4 * M_PI;
  }

  return FTransform;
}

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

double bTerm(const double &p, const double &E) {
  // put E and p into same units and then add back rest mass energy
  const auto alphan2 = 1.0 / (PhysConst::alpha2);
  const auto Enu = PhysConst::alpha2 * (E + alphan2);
  const auto pnu = PhysConst::alpha * p;

  return ((-1.0 - Enu * Enu + pnu * pnu) / (Enu * Enu - pnu * pnu)) *
         (1.0 + ((1.0 - Enu * Enu + pnu * pnu) / (Enu * Enu - pnu * pnu)) *
                  log(1.0 - Enu * Enu + pnu * pnu));
}

} // namespace Module
