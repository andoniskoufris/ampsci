#include "GreenQED.hpp"
#include "MBPT/Feynman.hpp"
#include "Maths/Hypergeometric.hpp"
#include "Wavefunction/Wavefunction.hpp"
#include "catch2/catch.hpp"
#include "fmt/format.hpp"
#include "qip/Vector.hpp"
#include <iostream>
#include <vector>

// test function to integrate: -i / [(z - z0) * (z + z0)^2]
// - z0 = a + b*i
// - z = 0.9a + t*i, with -\infty < t < \infty
std::complex<double> test_integrand(double t, double a, double b) {
  const auto I = std::complex<double>{0.0, 1.0};
  auto d1 = -0.1 * a + (t - b) * I;
  auto d2 = std::pow(1.9 * a + (t + b) * I, 2);
  auto d = d1 * d2;
  d = 1.0 / d;
  return -I * d;
}

//==============================================================================
TEST_CASE("G2 Green's function", "[GreenQED][G2]") {

  const auto thres = 1.0e3;

  // Create the array of G^{2+} Green's functions on realistic omega and k grid to H-like

  // Test for H-like full grid and subgrid
  Wavefunction wf({2000, 1.0e-6, 10.0, 0.1, "loglinear"},
                  {"U", 1, "point-like"}, 1.0);
  wf.solve_core("HartreeFock", "[]", std::nullopt, 1.0e-12, false);
  wf.solve_valence("1s", false);

  // Determine "sub" grid:
  const auto r0_target = 1.0e-4;
  const auto rmax_target = 2.9;

  const std::size_t i0 = wf.grid().getIndex(r0_target);
  const std::size_t imax = wf.grid().getIndex(rmax_target);
  assert(imax > i0 && "Require rmax > r0");

  const std::size_t size = 50ul;
  const auto stride = std::max(1ul, (imax - i0) / (size - 1));

  assert(size > 1 && stride > 0);

  // // actual r0,rmax might be slightly different, due to finite grid, stride
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

  // number of kappa to include in sum over k, i.e. max kappa index
  const int num_k = 100;
  // index of maximum kappa (since indexing starts from 0 need to subtract 1)
  const int max_kappa_index = num_k - 1;
  // kappa value corresponding to max kappa index
  const auto k_max = Angular::kindex_to_kappa(max_kappa_index);
  // // finds minimum and maximum J we will ever need based on the rule |ji - jv| <= J <= ji + jv
  // const auto [J_min, J_max] = J_min_max(valence, max_kappa_index);
  // const auto num_J = J_max - J_min + 1;

  //! Create omega grid and weights
  // Re(w) = e0 = 0.9 * e_{1s} (Shabaev's choice)
  // if we want to always have Re(w) be in the midpoint of the core-excited gap we need
  // Re(w) = [ea(highest) + en(lowest)] / 2.
  //
  // If we don't have a core (H-like) then we can use Shabaev's choice since we don't have a core gap but instead have a gap between the lowest excited state and the branch cut from the free electron Green's function at +mc^2 which is far off to the right w >= (1/alpha)^2
  const auto w_real = 0.9 * (wf.valence().front().en() +
                             1.0 / PhysConst::alpha2); // add back 1/α^2

  // defines number of points from zero to -Re(w) + i\infty
  const size_t w_num_pos_pts = 60;
  const auto w0 = 1.0e-4;
  const auto wmax = 20000.0;
  const auto omega_gw = Module::create_w_grid(w_real, w0, wmax, w_num_pos_pts);
  const auto w_grid = omega_gw.first;
  const auto w_weights = omega_gw.second;
  const auto w_num_pts = w_grid.size();

  std::cout << "Calculating the many-potential contribution with:\n";
  std::cout << "num_kappa =  " << num_k << " (kappa_max = " << k_max << ")\n";
  std::cout << "w0        =  " << w0 << "\n";
  std::cout << "w_max     =  " << wmax << "\n";
  std::cout << "w_num_pts =  " << w_num_pts << "\n";
  std::cout << "Re(w)     =  " << "0.9 * E_{1s} = " << w_real << "\n\n";

  using namespace MBPT;

  auto g_2_plus = std::vector<std::vector<ComplexGMatrix>>(
    num_k, std::vector<ComplexGMatrix>(
             w_num_pts, ComplexGMatrix(Fy.i0(), Fy.stride(),
                                       Fy.num_subgrid_pts(), true, Fy.grid())));

  // Record starting time
  auto start1 = std::chrono::high_resolution_clock::now();
// create g_2 for the full list of kappas we will include in our sum over kappa
#pragma omp parallel for
  for (auto iw = 0ul; iw < w_num_pts; iw++) {
    for (auto ik = 0ul; ik < num_k; ik++) {
      const auto k = Angular::kindex_to_kappa(ik);
      // physical energy is alpha^2 * w (?)
      const auto w = w_grid[iw];
      g_2_plus[ik][iw] = Fy.green_2plus(k, w);
      // has_nans(g_2_plus[ik][iw]);
    }
  }
  // Record ending time
  auto stop1 = std::chrono::high_resolution_clock::now();
  auto duration1 =
    std::chrono::duration_cast<std::chrono::seconds>(stop1 - start1);
  std::cout << "Time taken to construct Green's functions: "
            << duration1.count() << " seconds\n\n";
  std::cout << std::flush;

  fmt::print(" k  Re(w)  Im(w)  x1  x2  Re(G.ff^(2+))  Im(G.ff^(2+))  "
             "Re(G.fg^(2+))  Im(G.fg^(2+))  Re(G.gf^(2+))  Im(G.gf^(2+))  "
             "Re(G.gg^(2+))  Im(G.gg^(2+))\n");

  auto counter = 0;
  for (auto ik = 0ul; ik < num_k; ik++) {
    const auto k = Angular::kindex_to_kappa(ik);
    for (auto iw = 0ul; iw < w_num_pts; iw++) {
      const auto w = w_grid[iw];
      const auto g2 = g_2_plus[ik][iw];
      for (auto ix = 0ul; ix < size; ix++) {
        const auto x =
          wf.grid().r(g_2_plus.front().front().index_to_fullgrid(ix));
        for (auto iy = 0ul; iy < size; iy++) {
          const auto y =
            wf.grid().r(g_2_plus.front().front().index_to_fullgrid(iy));
          counter += 1;
          if (counter % 40000 == 0) {
            fmt::print("{:2}  {:.3e}  {:.3e}  {:.2e}  {:.2e}  {:.6e}  {:.6e}\n",
                       k, w.real(), w.imag(), x, y, g2.ff(ix, iy).real(),
                       g2.ff(ix, iy).imag());
            if (g2.ff(ix, iy).real() > thres) {
              std::cout << "\nBigger than " << thres << " at k = " << k
                        << " and w = " << w << "  : Re[G_ff(" << x << ", " << y
                        << ")] = " << g2.ff(ix, iy).real() << "\n\n";
            }
            if (g2.ff(ix, iy).imag() > thres) {
              std::cout << "\nBigger than " << thres << " at k = " << k
                        << " and w = " << w << "  : Im[G_ff(" << x << ", " << y
                        << ")] = " << g2.ff(ix, iy) << "\n\n";
            }
            if (g2.fg(ix, iy).real() > thres) {
              std::cout << "\nBigger than " << thres << " at k = " << k
                        << " and w = " << w << "  : Re[G_fg(" << x << ", " << y
                        << ")] = " << g2.fg(ix, iy) << "\n\n";
            }
            if (g2.fg(ix, iy).imag() > thres) {
              std::cout << "\nBigger than " << thres << " at k = " << k
                        << " and w = " << w << "  : Im[G_fg(" << x << ", " << y
                        << ")] = " << g2.fg(ix, iy) << "\n\n";
            }
            if (g2.gf(ix, iy).real() > thres) {
              std::cout << "\nBigger than " << thres << " at k = " << k
                        << " and w = " << w << "  : Re[G_gf(" << x << ", " << y
                        << ")] = " << g2.gf(ix, iy) << "\n\n";
            }
            if (g2.gf(ix, iy).imag() > thres) {
              std::cout << "\nBigger than " << thres << " at k = " << k
                        << " and w = " << w << "  : Im[G_gf(" << x << ", " << y
                        << ")] = " << g2.gf(ix, iy) << "\n\n";
            }
            if (g2.gg(ix, iy).real() > thres) {
              std::cout << "\nBigger than " << thres << " at k = " << k
                        << " and w = " << w << "  : Re[G_gg(" << x << ", " << y
                        << ")] = " << g2.gg(ix, iy) << "\n\n";
            }
            if (g2.gg(ix, iy).imag() > thres) {
              std::cout << "\nBigger than " << thres << " at k = " << k
                        << " and w = " << w << "  : Im[G_gg(" << x << ", " << y
                        << ")] = " << g2.gg(ix, iy) << "\n\n";
            }
          }
        }
      }
    }
  }
}

//==============================================================================
TEST_CASE("Print photon propagator values", "[GreenQED][GPhoton]") {

  const auto thres = 1.0e3;

  // Create the array of photon Green's functions

  // Test for H-like full grid and subgrid
  Wavefunction wf({2000, 1.0e-6, 10.0, 0.1, "loglinear"},
                  {"U", 1, "point-like"}, 1.0);
  wf.solve_core("HartreeFock", "[]", std::nullopt, 1.0e-12, false);
  wf.solve_valence("1s5g", false);

  // evaluate photon propagator at E_{5g+} - w
  const auto Ev = wf.valence().front().en() + 1.0 / PhysConst::alpha2;

  // Determine "sub" grid:
  const auto r0_target = 1.0e-4;
  const auto rmax_target = 2.9;

  const std::size_t i0 = wf.grid().getIndex(r0_target);
  const std::size_t imax = wf.grid().getIndex(rmax_target);
  assert(imax > i0 && "Require rmax > r0");

  const std::size_t size = 100ul;
  const auto stride = std::max(1ul, (imax - i0) / (size - 1));

  assert(size > 1 && stride > 0);

  // // actual r0,rmax might be slightly different, due to finite grid, stride
  const auto grid = wf.grid();
  const auto r0 = wf.grid().r(i0);
  const auto rmax = wf.grid().r(i0 + stride * size);
  fmt::print("Grid for photon Green's function: {:.1e} - {:.1f} with {} points "
             "[stride = {}]\n",
             r0, rmax, size, stride);

  // We don't need QPQ (at least for now)
  bool construct_qpq = false;
  bool verbose = false;
  // I think this only affects Sigma anyway:
  bool include_G = true;

  MBPT::Feynman Fy(wf.vHF(), i0, stride, size, {}, 1, include_G, verbose, "",
                   construct_qpq);

  // number of kappa to include in sum over k, i.e. max kappa index
  const int num_J = 60;
  const auto w_real = 0.9 * (wf.valence().front().en() +
                             1.0 / PhysConst::alpha2); // add back 1/α^2

  // defines number of points from zero to -Re(w) + i\infty
  const size_t w_num_pos_pts = 60;
  const auto w0 = 1.0e-4;
  const auto wmax = 20000.0;
  const auto omega_gw = Module::create_w_grid(w_real, w0, wmax, w_num_pos_pts);
  const auto w_grid = omega_gw.first;
  const auto w_weights = omega_gw.second;
  const auto w_num_pts = w_grid.size();

  using namespace MBPT;

  auto g_photon = std::vector<std::vector<ComplexRMatrix>>(
    num_J, std::vector<ComplexRMatrix>(
             w_num_pts, ComplexRMatrix(Fy.i0(), Fy.stride(),
                                       Fy.num_subgrid_pts(), Fy.grid())));

  // Record starting time
  auto start1 = std::chrono::high_resolution_clock::now();
// create g_2 for the full list of kappas we will include in our sum over kappa
#pragma omp parallel for
  for (auto iw = 0ul; iw < w_num_pts; iw++) {
    for (auto ij = 0ul; ij < num_J; ij++) {
      // physical energy is alpha^2 * w (?)
      const auto w = (Ev - w_grid[iw]) * PhysConst::alpha;
      g_photon[ij][iw] = Module::g_L(int(ij), w, Fy);
      // has_nans(g_2_plus[ik][iw]);
    }
  }
  // Record ending time
  auto stop1 = std::chrono::high_resolution_clock::now();
  auto duration1 =
    std::chrono::duration_cast<std::chrono::seconds>(stop1 - start1);
  std::cout << "Time taken to construct photon propagator: "
            << duration1.count() << " seconds\n\n";
  std::cout << std::flush;

  fmt::print(" J  Re(w)  Im(w)  x1  x2  Re(g_J(x,y))  Im(g_J(x,y))\n");

  auto counter = 0;
  for (auto ij = 0ul; ij < num_J; ij++) {
    for (auto iw = 0ul; iw < w_num_pts; iw++) {
      const auto w = (Ev - w_grid[iw]) * PhysConst::alpha;
      const auto g2 = g_photon[ij][iw];
      for (auto ix = 0ul; ix < size; ix++) {
        const auto x = wf.grid().r(g2.index_to_fullgrid(ix));
        for (auto iy = 0ul; iy < size; iy++) {
          const auto y = wf.grid().r(g2.index_to_fullgrid(iy));
          counter += 1;
          if (counter % 160000 == 0) {
            fmt::print("{:2}  {:.3e}  {:.3e}  {:.2e}  {:.2e}  {:.6e}  {:.6e}\n",
                       ij, w.real(), w.imag(), x, y, g2(ix, iy).real(),
                       g2(ix, iy).imag());
          }
          if (g2(ix, iy).real() > thres) {
            std::cout << "\nBigger than " << thres << " at J = " << ij
                      << " and w = " << w << "  : Re[G(" << x << ", " << y
                      << ")] = " << g2(ix, iy).real() << "\n";
          }
          if (g2(ix, iy).imag() > thres) {
            std::cout << "\nBigger than " << thres << " at J = " << ij
                      << " and w = " << w << "  : Im[G_ff(" << x << ", " << y
                      << ")] = " << g2(ix, iy) << "\n\n";
          }
        }
      }
    }
  }
}

//==============================================================================
TEST_CASE("Test integration of the function -i / [(z - z0) * (z + z0)^2] with "
          "contour integration",
          "[GreenQED][omega_grid]") {

  const auto I = std::complex<double>{0.0, 1.0};
  const auto a = 10.0;
  const auto b = 3.0;

  const auto w_real = 0.9 * a;
  // defines number of points from zero to -Re(w) + i\infty
  const size_t w_num_pos_pts = 150;
  const auto w0 = 1.0e-5;
  const auto wmax = 1000.0;
  const auto omega_gw = Module::create_w_grid(w_real, w0, wmax, w_num_pos_pts);
  const auto w_grid = omega_gw.first;
  const auto w_weights = omega_gw.second;
  const auto w_num_pts = w_grid.size();

  //   for (auto i = 0ul; i < w_num_pts; i++) {
  //     std::cout << "w = " << w_grid[i] << " ; weight = " << w_weights[i] << "\n";
  //   }

  // expected result from the theorem of residues
  const auto exp_res = M_PI * I / (2.0 * std::pow(a + b * I, 2));

  std::complex<double> res = 0.0;

  for (auto iw = 0ul; iw < w_num_pts; iw++) {
    const auto t = w_grid[iw].imag();
    res += test_integrand(t, a, b) * w_weights[iw].imag();
  }

  std::cout << "\nExpected result with z0 = " << a << " + " << b
            << "*i : " << exp_res << "\n";
  std::cout << "Numerical result : " << res << "\n\n";
}