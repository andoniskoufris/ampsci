#include "CSF.hpp"
#include "Angular/include.hpp"
#include "Maths/Grid.hpp"
#include "catch2/catch.hpp"
#include "fmt/format.hpp"
#include <cmath>
#include <string>
#include <vector>

//==============================================================================
// Tests the text form of a CI level reference: J{+,-}:index, and e/oJ:index
TEST_CASE("CI: level references", "[CI][unit]") {

  std::cout << "CI level references\n";

  // {text, 2J, parity, index}
  const std::vector<std::tuple<std::string, int, int, std::size_t>> cases{
    {"2+:3", 4, 1, 3},  {"e2:3", 4, 1, 3}, {"2-:3", 4, -1, 3},
    {"o2:3", 4, -1, 3}, {"0+", 0, 1, 0},   {"e0", 0, 1, 0},
    {"1-", 2, -1, 0},   {"o1", 2, -1, 0},  {"10+:12", 20, 1, 12},
    {"1-:2", 2, -1, 2}};

  for (const auto &[text, twoJ, parity, index] : cases) {
    const auto level = CI::parse_level(text);
    REQUIRE(level.has_value());
    REQUIRE(level->twoJ == twoJ);
    REQUIRE(level->parity == parity);
    REQUIRE(level->index == index);
  }

  // Round trip: the standard form must parse back to itself
  for (const auto &text : {"0+:0", "2-:3", "10+:12"}) {
    const auto level = CI::parse_level(text);
    REQUIRE(level.has_value());
    REQUIRE(CI::to_string(*level) == text);
  }

  // Malformed
  for (const auto &text :
       {"", " ", "2", "+2", "2+:", "2+:x", "x2:3", "2*:3", "-2+", "2+3"}) {
    fmt::print("'{}' -> {}\n", text,
               CI::parse_level(text) ? "parsed" : "rejected");
    REQUIRE(!CI::parse_level(text).has_value());
  }
}

//==============================================================================
// Tests the jj -> LS recoupling amplitudes and <L^2>, <S^2> expectation values
TEST_CASE("CI: LS coupling", "[CI][unit]") {

  std::cout << "CI LS coupling\n";

  // Sum over (L, S) of A^2, for CSF with given quantum numbers
  const auto sum_A2 = [](int n1, int k1, int n2, int k2, int twoJ) {
    const auto l1 = Angular::l_k(k1);
    const auto l2 = Angular::l_k(k2);
    const auto twoj1 = Angular::twoj_k(k1);
    const auto twoj2 = Angular::twoj_k(k2);
    double sum = 0.0;
    for (int S = 0; S <= 1; ++S) {
      for (int L = std::abs(l1 - l2); L <= l1 + l2; ++L) {
        const auto A =
          CI::LS_amplitude(n1, l1, twoj1, n2, l2, twoj2, L, S, twoJ);
        sum += A * A;
      }
    }
    return sum;
  };

  // Unitarity: each CSF carries unit total LS weight.
  // Cases cover: distinct configs, same shell same j, same shell different j,
  // same l different n. {n1, kappa1, n2, kappa2, twoJ}
  const std::vector<std::array<int, 5>> csf_cases{
    {4, -1, 3, 2, 4},  // 4s1/2 3d3/2, J=2
    {4, -1, 3, -3, 4}, // 4s1/2 3d5/2, J=2
    {4, -1, 3, -3, 6}, // 4s1/2 3d5/2, J=3
    {3, 1, 3, -2, 2},  // 3p1/2 3p3/2, J=1
    {3, 1, 3, -2, 4},  // 3p1/2 3p3/2, J=2
    {3, 1, 3, 1, 0},   // (3p1/2)^2,   J=0
    {3, -2, 3, -2, 0}, // (3p3/2)^2,   J=0
    {3, -2, 3, -2, 4}, // (3p3/2)^2,   J=2
    {3, 1, 4, -2, 2},  // 3p1/2 4p3/2, J=1 (different n: distinct config)
    {3, 2, 3, -3, 2},  // 3d3/2 3d5/2, J=1
    {3, 2, 3, -3, 6},  // 3d3/2 3d5/2, J=3
    {3, -3, 3, -3, 8}, // (3d5/2)^2,   J=4
    {3, 2, 4, 1, 2} // 3d3/2 4p1/2, J=1
  };
  for (const auto &[n1, k1, n2, k2, twoJ] : csf_cases) {
    REQUIRE(sum_A2(n1, k1, n2, k2, twoJ) == Approx(1.0).epsilon(1.0e-12));
  }

  // Known values: p^2 (same n), J=0.
  // jj basis: {(p1/2)^2, (p3/2)^2}; LS terms: 1S0, 3P0
  // Weights: (p1/2)^2 = {1/3, 2/3}; (p3/2)^2 = {2/3, 1/3}
  const auto a11 = CI::LS_amplitude(3, 1, 1, 3, 1, 1, 0, 0, 0);
  const auto a12 = CI::LS_amplitude(3, 1, 1, 3, 1, 1, 1, 1, 0);
  const auto a21 = CI::LS_amplitude(3, 1, 3, 3, 1, 3, 0, 0, 0);
  const auto a22 = CI::LS_amplitude(3, 1, 3, 3, 1, 3, 1, 1, 0);
  REQUIRE(a11 * a11 == Approx(1.0 / 3).epsilon(1.0e-12));
  REQUIRE(a12 * a12 == Approx(2.0 / 3).epsilon(1.0e-12));
  REQUIRE(a21 * a21 == Approx(2.0 / 3).epsilon(1.0e-12));
  REQUIRE(a22 * a22 == Approx(1.0 / 3).epsilon(1.0e-12));
  // rows of the transformation matrix are orthogonal:
  REQUIRE(a11 * a21 + a12 * a22 == Approx(0.0).margin(1.0e-12));

  // 3p1/2 3p3/2, J=1: only 3P1 exists (1P1 is Pauli-forbidden: L+S odd)
  REQUIRE(CI::LS_amplitude(3, 1, 1, 3, 1, 3, 1, 0, 2) == 0.0);
  const auto a3P1 = CI::LS_amplitude(3, 1, 1, 3, 1, 3, 1, 1, 2);
  REQUIRE(a3P1 * a3P1 == Approx(1.0).epsilon(1.0e-12));

  //----------------------------------------------------------------------------
  // <L^2>, <S^2> for hand-built states:

  const auto grid = std::make_shared<const Grid>(
    GridParameters{10, 1.0e-4, 10.0, 5.0, GridType::loglinear});
  const DiracSpinor s4{4, -1, grid};
  const DiracSpinor p1{3, 1, grid};
  const DiracSpinor p3{3, -2, grid};
  const DiracSpinor d3{3, 2, grid};
  const DiracSpinor d5{3, -3, grid};

  const auto view = [](const std::vector<double> &c) {
    return LinAlg::View<const double>{c.data(), 0, c.size(), 1};
  };

  {
    // Single CSF {4s, 3d5/2}, J=3: pure 3D3
    const std::vector<CI::CSF2> csfs{{s4, d5}};
    const std::vector<double> c{1.0};
    const auto [L2, S2] = CI::expectation_L2S2(view(c), csfs, 6);
    REQUIRE(L2 == Approx(6.0).epsilon(1.0e-12));
    REQUIRE(S2 == Approx(2.0).epsilon(1.0e-12));
  }

  {
    // Single CSF {3p1/2, 3p3/2}, J=1: pure 3P1 (tests the sqrt(2) factor)
    const std::vector<CI::CSF2> csfs{{p1, p3}};
    const std::vector<double> c{1.0};
    const auto [L2, S2] = CI::expectation_L2S2(view(c), csfs, 2);
    REQUIRE(L2 == Approx(2.0).epsilon(1.0e-12));
    REQUIRE(S2 == Approx(2.0).epsilon(1.0e-12));
  }

  {
    // {4s, 3d3/2} and {4s, 3d5/2}, J=2: mixtures of 1D2 and 3D2.
    // L2 = 6 always; total 3D2 weight across the two CSFs is 1
    const std::vector<CI::CSF2> csfs{{s4, d3}, {s4, d5}};
    const std::vector<double> ca{1.0, 0.0};
    const std::vector<double> cb{0.0, 1.0};
    const auto [L2a, S2a] = CI::expectation_L2S2(view(ca), csfs, 4);
    const auto [L2b, S2b] = CI::expectation_L2S2(view(cb), csfs, 4);
    REQUIRE(L2a == Approx(6.0).epsilon(1.0e-12));
    REQUIRE(L2b == Approx(6.0).epsilon(1.0e-12));
    REQUIRE(S2a + S2b == Approx(2.0).epsilon(1.0e-12));
  }

  {
    // (3p)^2, J=0: CI coefficients equal to the LS amplitudes of 3P0
    // give the pure 3P0 state; those of 1S0 give pure 1S0
    const std::vector<CI::CSF2> csfs{{p1, p1}, {p3, p3}};
    const std::vector<double> c_3P0{a12, a22};
    const std::vector<double> c_1S0{a11, a21};
    const auto [L2a, S2a] = CI::expectation_L2S2(view(c_3P0), csfs, 0);
    const auto [L2b, S2b] = CI::expectation_L2S2(view(c_1S0), csfs, 0);
    REQUIRE(L2a == Approx(2.0).epsilon(1.0e-12));
    REQUIRE(S2a == Approx(2.0).epsilon(1.0e-12));
    REQUIRE(L2b == Approx(0.0).margin(1.0e-12));
    REQUIRE(S2b == Approx(0.0).margin(1.0e-12));
  }
}
