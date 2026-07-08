#include "DiracContinuum.hpp"
#include "DiracHydrogen.hpp"
#include "Maths/Grid.hpp"
#include "Maths/NumCalc_quadIntegrate.hpp"
#include "PhysConst_constants.hpp"
#include "catch2/catch.hpp"
#include "fmt/format.hpp"
#include <cmath>
#include <iostream>
#include <vector>

//==============================================================================
TEST_CASE("DiracContinuum: availability", "[DiracContinuum][unit]") {
  // 'available' must be usable at compile time:
  static_assert(DiracContinuum::available ||
                DiracContinuum::available == false);

  if constexpr (!DiracContinuum::available) {
    // Compiled without FLINT: f and g print a warning and return NaN
    std::cout << "FLINT not available: No analytic continuum states\n";
  } else {
    std::cout
      << "FLINT is available: can calculate analytic continuum states\n";
    REQUIRE(DiracContinuum::f(1.0, 0.5, -1, 1.0, PhysConst::alpha) != 0.0);
  }
}

//==============================================================================
TEST_CASE("DiracContinuum: nonrelativistic limit", "[DiracContinuum][unit]") {
  if (!DiracContinuum::available) {
    std::cout << "FLINT not available: No analytic continuum states\n";
    SUCCEED("Compiled without FLINT: skipping numerical tests");
    return;
  }

  // Set true to print detailed (expected vs found) table:
  constexpr bool print_table = false;

  const double alpha = 1.0e-5;

  if (print_table) {
    fmt::print("\nDirac Coulomb continuum, nonrel. limit (alpha = {:.0e}) vs "
               "GSL Coulomb F_l:\n",
               alpha);
    fmt::print("{:>3} {:>2} {:>4} {:>5} {:>6} {:>15} {:>15} {:>9}\n", "kap",
               "l", "Z", "en", "r", "expected", "found", "eps");
  }

  for (const int kappa : {-1, 1, -2, 2, -3}) {
    const int l = kappa > 0 ? kappa : -kappa - 1;
    for (const double z : {1.0, 5.0}) {
      for (const double en : {0.1, 1.0}) {
        for (const double r : {0.5, 3.0, 15.0}) {

          const auto expected = DiracContinuum::P_el(r, en, l, z);

          const auto [ff, gg] = DiracContinuum::fg(r, en, kappa, z, alpha);

          if (print_table) {
            fmt::print("{:>3} {:>2} {:>4.1f} {:>5.2f} {:>6.2f} {:>15.8e} "
                       "{:>15.8e} {:>9.1e}\n",
                       kappa, l, z, en, r, expected, ff, ff - expected);
          }

          REQUIRE(ff == Approx(expected).epsilon(1.0e-7).margin(1.0e-7));
          // Small component vanishes as alpha -> 0:
          REQUIRE(std::abs(gg) < 1.0e-4);
        }
      }
    }
  }
}

//==============================================================================
TEST_CASE("DiracContinuum: bound-continuum nonrel limit",
          "[DiracContinuum][unit]") {
  if (!DiracContinuum::available) {
    std::cout << "FLINT not available: No analytic continuum states\n";
    SUCCEED("Compiled without FLINT: skipping numerical tests");
    return;
  }

  std::cout << "Calculate <ek|r^k|nk> with k=-1,1\n"
               "Compare between relativistic functions with alpha->0 and "
               "non-rel functions\n"
               "For a range of energies, kappas, n\n";

  // Set true to print detailed (expected vs found) table:
  constexpr bool print_table = false;

  // Radial integrals <n kappa| r^pow |en kappa> (bound * continuum, pow =
  // +/-1): the relativistic functions evaluated at tiny alpha must match a
  // fully nonrelativistic calculation, using the analytic hydrogen bound
  // state P_nl and GSL Coulomb function F_l. No stored data: both sides
  // computed here, on the same grid.
  const double alpha = 1.0e-12;
  const double r0 = 1.0e-7;
  const double rmax = 120.0;
  const double b = 1.0;
  const auto npts = 1000;
  const Grid grid(r0, rmax, npts, GridType::loglinear, b);

  if (print_table) {
    fmt::print("\nBound-continuum r^pow integrals: rel (alpha = {:.0e}) vs "
               "nonrel:\n",
               alpha);
    fmt::print("{:>2} {:>3} {:>2} {:>2} {:>5} {:>3} {:>16} {:>16} {:>9}\n", "n",
               "kap", "l", "Z", "en", "pow", "nonrel", "rel", "eps");
  }

  for (const int n : {2, 4}) {
    for (const int kappa : {-1, 1, -2, 2, -3}) {
      const int l = kappa > 0 ? kappa : -kappa - 1;
      if (l > n - 1) {
        continue;
      }
      for (const double z : {1.0, 5.0}) {
        for (const double en : {0.1, 5.0, 100.0}) {

          // ~10 points per continuum wavelength (13-point quadrature):
          // const auto p = DiracContinuum::pe(en, alpha);
          // const auto du = std::min(0.15, 0.2 * M_PI / p);
          // const auto span =
          //   (rmax + b * std::log(rmax)) - (r0 + b * std::log(r0));
          // const auto npts = std::size_t(span / du) + 2;
          // const Grid grid(r0, rmax, npts, GridType::loglinear, b);

          // integrands for pow = -1 (_m) and pow = +1 (_p):
          std::vector<double> rel_m(grid.num_points()), rel_p(rel_m),
            nr_m(rel_m), nr_p(rel_m);
#pragma omp parallel for
          for (std::size_t i = 0; i < grid.num_points(); ++i) {
            const auto r = grid.r(i);
            const auto fb = DiracHydrogen::f(r, n, kappa, z, alpha);
            const auto gb = DiracHydrogen::g(r, n, kappa, z, alpha);
            const auto [fc, gc] = DiracContinuum::fg(r, en, kappa, z, alpha);
            const auto rel = (fb * fc + gb * gc) * grid.drdu(i);
            const auto nr = DiracHydrogen::P_nl(r, n, l, z) *
                            DiracContinuum::P_el(r, en, l, z) * grid.drdu(i);
            rel_m[i] = rel / r;
            rel_p[i] = rel * r;
            nr_m[i] = nr / r;
            nr_p[i] = nr * r;
          }

          for (const int pw : {-1, 1}) {
            const auto found =
              NumCalc::integrate(grid.du(), 0, 0, pw == 1 ? rel_p : rel_m);
            const auto expected =
              NumCalc::integrate(grid.du(), 0, 0, pw == 1 ? nr_p : nr_m);

            if (print_table) {
              fmt::print("{:>2} {:>3} {:>2} {:>2.0f} {:>5.1f} {:>3} "
                         "{:>16.8e} {:>16.8e} {:>9.1e}\n",
                         n, kappa, l, z, en, pw, expected, found,
                         (found - expected) / expected);
            }

            CHECK(found == Approx(expected).epsilon(1.0e-8).margin(1.0e-12));
          }
        }
      }
    }
  }
}

//==============================================================================
TEST_CASE("DiracContinuum: large-r asymptotics", "[DiracContinuum][unit]") {
  if (!DiracContinuum::available) {
    std::cout << "FLINT not available: No analytic continuum states\n";
    SUCCEED("Compiled without FLINT: skipping numerical tests");
    return;
  }

  // Set true to print detailed (expected vs found) table:
  constexpr bool print_table = false;

  // At physical alpha (including high Z), f and g must approach the
  // analytic large-r asymptotic forms. Leading corrections to the 1F1
  // asymptotics are O((nu^2 + gamma^2)/x), so choose r large enough that
  // these are below ~0.2%, and test to 1%
  const double alpha = PhysConst::alpha;

  fmt::print("\nDirac Coulomb continuum vs large-r asymptotic form.\n");
  if (print_table) {
    fmt::print("{:>3} {:>4} {:>5} {:>9} {:>13} {:>13} {:>9}\n", "kap", "Z",
               "en", "r", "expected", "found", "diff");
  }

  for (const int kappa : {-1, 1, -2}) {
    for (const double z : {1.0, 30.0, 90.0}) {
      for (const double en : {0.1, 1.0, 10.0}) {
        const auto p = DiracContinuum::pe(en, alpha);
        const auto gam = DiracContinuum::gamma(kappa, z, alpha);
        const auto nu = z * (1.0 + alpha * alpha * en) / p;

        const auto x = 500.0 * (nu * nu + gam * gam + 10.0);
        // two radii, to sample different phase points:
        for (const double r : {x / (2.0 * p), 1.37 * x / (2.0 * p)}) {

          const auto [ff, gg] = DiracContinuum::fg(r, en, kappa, z, alpha);
          const auto f0 = DiracContinuum::f_asymptotic(r, en, kappa, z, alpha);
          const auto g0 = DiracContinuum::g_asymptotic(r, en, kappa, z, alpha);

          if (print_table) {
            fmt::print("{:>3} {:>4.0f} {:>5.1f} {:>9.2e} {:>13.6e} {:>13.6e} "
                       "{:>9.1e}\n",
                       kappa, z, en, r, f0, ff, ff - f0);
          }

          const auto amp_f = std::sqrt(p / (M_PI * en));
          const auto amp_g = alpha * std::sqrt(en / (M_PI * p));
          REQUIRE(ff == Approx(f0).margin(0.01 * amp_f));
          REQUIRE(gg == Approx(g0).margin(0.01 * amp_g));
        }
      }
    }
  }
}

//==============================================================================
// Bound * continuum radial integral test data, computed independently
// in Mathematica; used by "bound-continuum integrals" test.
// The data itself is listed at the bottom of this file:
struct TestData {
  double z;
  double en;
  int n;
  int kappa;
  int pow;
  // relativistic (Dirac) value
  double I_rel;
  // nonrelativistic value (for reference only; not tested)
  // The mathematica values seem not that great!
  double I_nr;
};
const std::vector<TestData> &integral_data();

//==============================================================================
TEST_CASE("DiracContinuum: bound-continuum integrals",
          "[DiracContinuum][integration]") {
  if (!DiracContinuum::available) {
    std::cout << "FLINT not available: No analytic continuum states\n";
    SUCCEED("Compiled without FLINT: skipping numerical tests");
    return;
  }

  // Set true to print detailed (expected vs found) table:
  const auto print_table = true;
  const auto skip_half = true;

  const double alpha = PhysConst::alpha;

  // Same limits used in Mathematica:
  const double r0 = 1.0e-7;
  const double rmax = 120.0;
  const double b = 1.0;
  const std::size_t npts = 3000;
  const Grid grid(r0, rmax, npts, GridType::loglinear, b);

  fmt::print(
    "\nBound-continuum radial integrals <ek|r^k|nk> vs Mathematica:\n");
  if (print_table) {
    fmt::print("{:>3} {:>5} {:>3} {:>3} {:>16} {:>16} {:>9}\n", "Z", "en",
               "kap", "pow", "expected", "found", "eps");
  }

  int count = 0;
  for (const auto &d : integral_data()) {

    // Simply to save time: only do half the tests
    if (skip_half && ++count % 2 == 0)
      continue;

    std::vector<double> integrand(grid.num_points());
#pragma omp parallel for
    for (std::size_t i = 0; i < grid.num_points(); ++i) {
      const auto r = grid.r(i);
      const auto fb = DiracHydrogen::f(r, d.n, d.kappa, d.z, alpha);
      const auto gb = DiracHydrogen::g(r, d.n, d.kappa, d.z, alpha);
      const auto [fc, gc] = DiracContinuum::fg(r, d.en, d.kappa, d.z, alpha);
      integrand[i] = (fb * fc + gb * gc) * std::pow(r, d.pow) * grid.drdu(i);
    }

    const auto found = NumCalc::integrate(grid.du(), 0, 0, integrand);

    const auto expected = d.I_rel;

    if (print_table) {
      fmt::print("{:>3.0f} {:>5.1f} {:>3} {:>3} {:>16.8e} {:>16.8e} "
                 "{:>9.1e}\n",
                 d.z, d.en, d.kappa, d.pow, expected, found,
                 (found - expected) / expected);
    }

    REQUIRE(found == Approx(expected).epsilon(1.0e-8).margin(1.0e-12));
  }
}

//==============================================================================
// Mathematica reference data:

// Radial integrals of bound * continuum hydrogen-like states:
//   Int[(f_b(r)*f_c(r) + g_b(r)*g_c(r)) * r^pow, {r, 1e-7, 120}]
// where {f_b, g_b} is the bound (n, kappa) state (DiracHydrogen), and
// {f_c, g_c} the continuum state with energy en (DiracContinuum).
// Reference values computed independently in Mathematica; columns:
// {z, en, n, kappa, pow, I_rel, I_nr}
// nb: It is actually Mathematica that appears to be the bottleneck here!
// Numerics seem quite bad for
const std::vector<TestData> &integral_data() {
  static const std::vector<TestData> data{
    {1, 0.1, 4, -1, -1, 0.10787864442207, 0.10787327289014},
    {1, 0.1, 4, -1, 1, -1.6045299554961, -1.6045352269209},
    {1, 0.1, 4, 1, -1, 0.082641112924538, 0.082636207458869},
    {1, 0.1, 4, 1, 1, -1.5150665786563, -1.5150693850767},
    {1, 0.1, 4, -2, -1, 0.082637649402613, 0.082636207458869},
    {1, 0.1, 4, -2, 1, -1.5150571307323, -1.5150693850767},
    {1, 0.1, 4, 2, -1, 0.045755900283617, 0.045754813876191},
    {1, 0.1, 4, 2, 1, -1.0935728475400, -1.0935788553729},
    {1, 0.1, 4, -3, -1, 0.045755079395833, 0.045754813876191},
    {1, 0.1, 4, -3, 1, -1.0935668805127, -1.0935788553729},
    {1, 0.1, 4, 3, -1, 0.015856538241995, 0.015856375426390},
    {1, 0.1, 4, 3, 1, -0.48323784270338, -0.48324191919528},
    {1, 0.1, 4, -4, -1, 0.015856353065187, 0.015856375426390},
    {1, 0.1, 4, -4, 1, -0.48323535843212, -0.48324191919528},
    {1, 1., 4, -1, -1, 0.049339739087740, 0.049339317440802},
    {1, 1., 4, -1, 1, -0.059525831536307, -0.056459622638551},
    {1, 1., 4, 1, -1, 0.020171658048130, 0.020133330401633},
    {1, 1., 4, 1, 1, -0.040461288770562, -0.039703796276177},
    {1, 1., 4, -2, -1, 0.020169276426568, 0.020133330401633},
    {1, 1., 4, -2, 1, -0.040459904548797, -0.039703796276177},
    {1, 1., 4, 2, -1, 0.0047195734922254, 0.0047055372424271},
    {1, 1., 4, 2, 1, -0.013818374925395, -0.013802369369871},
    {1, 1., 4, -3, -1, 0.0047192702478301, 0.0047055372424271},
    {1, 1., 4, -3, 1, -0.013817959780281, -0.013802369369871},
    {1, 1., 4, 3, -1, 0.00061586005273949, 0.00060968996679920},
    {1, 1., 4, 3, 1, -0.0023886096323282, -0.0027142480743326},
    {1, 1., 4, -4, -1, 0.00061582987231710, 0.00060968996679920},
    {1, 1., 4, -4, 1, -0.0023885411025570, -0.0027142480743326},
    {1, 10., 4, -1, -1, 0.015212174515437, 0.015816892516072},
    {1, 10., 4, -1, 1, -0.0015618757685327, -1.6505552236423},
    {1, 10., 4, 1, -1, 0.0021707163337424, 0.0030427395416101},
    {1, 10., 4, 1, 1, -0.00043393221943315, -0.29311897687016},
    {1, 10., 4, -2, -1, 0.0021694421258561, 0.0030427395416101},
    {1, 10., 4, -2, 1, -0.00043383323170499, -0.29311897687016},
    {1, 10., 4, 2, -1, 0.00016728438327946, 0.0010261902435168},
    {1, 10., 4, 2, 1, -0.000049966220646467, 0.41859205146874},
    {1, 10., 4, -3, -1, 0.00016721606913556, 0.0010261902435168},
    {1, 10., 4, -3, 1, -0.000049954469223161, 0.41859205146874},
    {1, 10., 4, 3, -1, 7.0504490406543e-6, 0.000046310398328296},
    {1, 10., 4, 3, 1, -2.8397770847482e-6, 0.11258074509241},
    {1, 10., 4, -4, -1, 7.0479519281811e-6, 0.000046310398328296},
    {1, 10., 4, -4, 1, -2.8390995224470e-6, 0.11258074509241},
    {1, 100., 4, -1, -1, 0.0033632485081422, 0.0091409251673754},
    {1, 100., 4, -1, 1, -0.000033651870474084, -1.3994837652741},
    {1, 100., 4, 1, -1, 0.00015378247625122, 0.0022022136177263},
    {1, 100., 4, 1, 1, -3.0516141963380e-6, 0.48881593358700},
    {1, 100., 4, -2, -1, 0.00015309350470074, 0.0022022136177263},
    {1, 100., 4, -2, 1, -3.0453932565697e-6, 0.48881593358700},
    {1, 100., 4, 2, -1, 3.7561165591153e-6, -0.0029964724018610},
    {1, 100., 4, 2, 1, -9.4285795345893e-8, 0.34004751236817},
    {1, 100., 4, -3, -1, 3.7428839943390e-6, -0.0029964724018610},
    {1, 100., 4, -3, 1, -9.4034081358515e-8, 0.34004751236817},
    {1, 100., 4, 3, -1, 5.0095931148408e-8, 0.000028912619701956},
    {1, 100., 4, 3, 1, -3.4413410326526e-9, 0.38527703183105},
    {1, 100., 4, -4, -1, 4.9934827781578e-8, 0.000028912619701956},
    {1, 100., 4, -4, 1, -3.4357745403803e-9, 0.38527703183105},
    {5, 0.1, 4, -1, -1, 0.16942566737718, 0.16927604873406},
    {5, 0.1, 4, -1, 1, -0.51204735230315, -0.51215468380392},
    {5, 0.1, 4, 1, -1, 0.15316465194322, 0.15302126556570},
    {5, 0.1, 4, 1, 1, -0.50541838147028, -0.50552598801907},
    {5, 0.1, 4, -2, -1, 0.15307247970479, 0.15302126556570},
    {5, 0.1, 4, -2, 1, -0.50547105258865, -0.50552598801907},
    {5, 0.1, 4, 2, -1, 0.11931914263967, 0.11927463353186},
    {5, 0.1, 4, 2, 1, -0.45626605510482, -0.45631883234172},
    {5, 0.1, 4, -3, -1, 0.11929240490751, 0.11927463353186},
    {5, 0.1, 4, -3, 1, -0.45627135789939, -0.45631883234172},
    {5, 0.1, 4, 3, -1, 0.069367825574413, 0.069355578698805},
    {5, 0.1, 4, 3, 1, -0.31477026024926, -0.31480546324387},
    {5, 0.1, 4, -4, -1, 0.069358813448878, 0.069355578698805},
    {5, 0.1, 4, -4, 1, -0.31476600371707, -0.31480546324387},
    {5, 1., 4, -1, -1, 0.13346042809368, 0.13331646291595},
    {5, 1., 4, -1, 1, -0.16967376425937, -0.16968133184414},
    {5, 1., 4, 1, -1, 0.11238416762229, 0.11224865945139},
    {5, 1., 4, 1, 1, -0.16479881815311, -0.16480654300291},
    {5, 1., 4, -2, -1, 0.11229315561996, 0.11224865945139},
    {5, 1., 4, -2, 1, -0.16479092065053, -0.16480654300291},
    {5, 1., 4, 2, -1, 0.074755661391827, 0.074273421750432},
    {5, 1., 4, 2, 1, -0.13532068982741, -0.13533356900950},
    {5, 1., 4, -3, -1, 0.074731390171664, 0.074273421750432},
    {5, 1., 4, -3, 1, -0.13531194515007, -0.13533356900950},
    {5, 1., 4, 3, -1, 0.033526173304364, 0.033518616283470},
    {5, 1., 4, 3, 1, -0.075258135752260, -0.075269875162837},
    {5, 1., 4, -4, -1, 0.033519418489408, 0.033518616283470},
    {5, 1., 4, -4, 1, -0.075252899292643, -0.075269875162837},
    {5, 10., 4, -1, -1, 0.070191325753514, 0.070081444952260},
    {5, 10., 4, -1, 1, -0.0095582487749356, -0.0095605537626818},
    {5, 10., 4, 1, -1, 0.039713728048937, 0.039623934498452},
    {5, 10., 4, 1, 1, -0.0079140138415839, -0.0079156457328349},
    {5, 10., 4, -2, -1, 0.039641178152331, 0.039623934498452},
    {5, 10., 4, -2, 1, -0.0079105276624913, -0.0079156457328349},
    {5, 10., 4, 2, -1, 0.013794897664189, 0.013783244557125},
    {5, 10., 4, 2, 1, -0.0038920294057743, -0.0038942526058386},
    {5, 10., 4, -3, -1, 0.013782559311265, 0.013783244557125},
    {5, 10., 4, -3, 1, -0.0038905897056309, -0.0038942526058386},
    {5, 10., 4, 3, -1, 0.0027505346540692, 0.0027498539008793},
    {5, 10., 4, 3, 1, -0.0010193757235416, -0.0010202357698922},
    {5, 10., 4, -4, -1, 0.0027487802038584, 0.0027498539008793},
    {5, 10., 4, -4, 1, -0.0010190271091445, -0.0010202357698922},
    {5, 100., 4, -1, -1, 0.025618662399803, 0.025566767617765},
    {5, 100., 4, -1, 1, -0.00027107931581216, -0.00052443128935584},
    {5, 100., 4, 1, -1, 0.0057022456386975, 0.026598185132575},
    {5, 100., 4, 1, 1, -0.00011310445729460, 0.00023462635638958},
    {5, 100., 4, -2, -1, 0.0056611026672622, 0.026598185132575},
    {5, 100., 4, -2, 1, -0.00011281565750126, 0.00023462635638958},
    {5, 100., 4, 2, -1, 0.00068696132893514, 0.00068636447796390},
    {5, 100., 4, 2, 1, -0.000020348133771323, 0.00010901789112393},
    {5, 100., 4, -3, -1, 0.00068379079488794, 0.00068636447796390},
    {5, 100., 4, -3, 1, -0.000020297579884741, 0.00010901789112393},
    {5, 100., 4, 3, -1, 0.000045508175293576, 0.000045294353274001},
    {5, 100., 4, 3, 1, -1.7960764828352e-6, 0.000011931938422216},
    {5, 100., 4, -4, -1, 0.000045333147995130, 0.000045294353274001},
    {5, 100., 4, -4, 1, -1.7915916540728e-6, 0.000011931938422216},
    {10, 0.1, 4, -1, -1, 0.17498804650475, 0.17438771379065},
    {10, 0.1, 4, -1, 1, -0.14681924712117, -0.14696185517289},
    {10, 0.1, 4, 1, -1, 0.15928982655550, 0.15871375948232},
    {10, 0.1, 4, 1, 1, -0.14510104686125, -0.14524391950115},
    {10, 0.1, 4, -2, -1, 0.15892079811212, 0.15871375948232},
    {10, 0.1, 4, -2, 1, -0.14517581759889, -0.14524391950115},
    {10, 0.1, 4, 2, -1, 0.12588420478688, 0.12570331272857},
    {10, 0.1, 4, 2, 1, -0.13216476875059, -0.13223071267004},
    {10, 0.1, 4, -3, -1, 0.12577673542528, 0.12570331272857},
    {10, 0.1, 4, -3, 1, -0.13217643685973, -0.13223071267004},
    {10, 0.1, 4, 3, -1, 0.075305568637543, 0.075254445435593},
    {10, 0.1, 4, 3, 1, -0.093297221348925, -0.093338847051868},
    {10, 0.1, 4, -4, -1, 0.075268694520785, 0.075254445435593},
    {10, 0.1, 4, -4, 1, -0.093294813272246, -0.093338847051868},
    {10, 1., 4, -1, -1, 0.16116196628070, 0.16056390029717},
    {10, 1., 4, -1, 1, -0.10018001440063, -0.10024383741902},
    {10, 1., 4, 1, -1, 0.14383536434991, 0.14326374429119},
    {10, 1., 4, 1, 1, -0.098626925975155, -0.098691026778220},
    {10, 1., 4, -2, -1, 0.14346432838022, 0.14326374429119},
    {10, 1., 4, -2, 1, -0.098652561400602, -0.098691026778220},
    {10, 1., 4, 2, -1, 0.10850636918188, 0.10833392886120},
    {10, 1., 4, 2, 1, -0.087565075164233, -0.087601322169904},
    {10, 1., 4, -3, -1, 0.10840052519713, 0.10833392886120},
    {10, 1., 4, -3, 1, -0.087562959619559, -0.087601322169904},
    {10, 1., 4, 3, -1, 0.059748546996505, 0.059703556777822},
    {10, 1., 4, 3, 1, -0.057867687155157, -0.057894358082415},
    {10, 1., 4, -4, -1, 0.059714169763221, 0.059703556777822},
    {10, 1., 4, -4, 1, -0.057861710186322, -0.057894358082415},
    {10, 10., 4, -1, -1, 0.10841253892229, 0.10787325874652},
    {10, 10., 4, -1, 1, -0.016044719920574, -0.016045352322100},
    {10, 10., 4, 1, -1, 0.083128774580677, 0.082636196426088},
    {10, 10., 4, 1, 1, -0.015150164987790, -0.015150693882820},
    {10, 10., 4, -2, -1, 0.082780560506926, 0.082636196426088},
    {10, 10., 4, -2, 1, -0.015140717866869, -0.015150693882820},
    {10, 10., 4, 2, -1, 0.045863596751451, 0.045754813876098},
    {10, 10., 4, 2, 1, -0.010929119498566, -0.010935788554940},
    {10, 10., 4, -3, -1, 0.045781384742108, 0.045754813876098},
    {10, 10., 4, -3, 1, -0.010923159511670, -0.010935788554940},
    {10, 10., 4, 3, -1, 0.015872671820800, 0.015856375426428},
    {10, 10., 4, 3, 1, -0.0048272373780562, -0.0048324191836009},
    {10, 10., 4, -4, -1, 0.015854148597859, 0.015856375426428},
    {10, 10., 4, -4, 1, -0.0048247571236958, -0.0048324191836009},
    {10, 100., 4, -1, -1, 0.049691899380655, 0.049336197812809},
    {10, 100., 4, -1, 1, -0.00059320598284255, -0.00059527481892802},
    {10, 100., 4, 1, -1, 0.020429189461293, 0.020169069160749},
    {10, 100., 4, 1, 1, -0.00040341583347863, -0.00040463011719577},
    {10, 100., 4, -2, -1, 0.020189850202299, 0.020169069160749},
    {10, 100., 4, -2, 1, -0.00040203976466544, -0.00040463011719577},
    {10, 100., 4, 2, -1, 0.0047397033122522, 0.0047193696195405},
    {10, 100., 4, 2, 1, -0.00013742833967024, -0.00013819698104499},
    {10, 100., 4, -3, -1, 0.0047094086256418, 0.0047193696195405},
    {10, 100., 4, -3, 1, -0.00013701670754813, -0.00013819698104499},
    {10, 100., 4, 3, -1, 0.00061608351672348, 0.00061585777997352},
    {10, 100., 4, 3, 1, -0.000023707288422217, -0.000023887816866755},
    {10, 100., 4, -4, -1, 0.00061307722383061, 0.00061585777997352},
    {10, 100., 4, -4, 1, -0.000023639463330421, -0.000023887816866755}};
  return data;
}
