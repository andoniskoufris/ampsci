#include "CI_Integrals.hpp"
#include "ConfigurationInteraction.hpp"
#include "IO/InputBlock.hpp"
#include "Wavefunction/Wavefunction.hpp"
#include "catch2/catch.hpp"
#include "qip/Random.hpp"

//==============================================================================
TEST_CASE("CI: Configuration Interaction, unit tests", "[CI][unit]") {

  std::cout << "CI, unit tests (not meant to be accurate)\n";

  Wavefunction wf({400, 1.0e-4, 20.0, 0.33 * 20.0, "loglinear"},
                  {"He", -1, "pointlike"}, 1.0);
  wf.solve_core("HartreeFock", std::nullopt, "[]", 1.0e-5);
  wf.formBasis(SplineBasis::Parameters("6spd", 20, 6, 1.0e-2, 1.0e-2, 20.0));

  std::string qk_filename = "deleteme_" + qip::random_string(6) + ".qk.abf";

  const IO::InputBlock input{"CI", "ci_basis = 6spd;"
                                   "n_min_core = 1;"
                                   "J + = 0, 1;"
                                   "J - = 0, 1;"
                                   "qk_file = " +
                                     qk_filename +
                                     ";"
                                     "num_solutions = 2;"};

  const auto CIWFs = CI::configuration_interaction(input, wf);

  // Expected values: (not accurate, just simple unit test)
  std::vector J{0, 1, 0, 1};
  std::vector parity{1, 1, -1, -1};
  std::vector energy{std::vector{-2.84834207, -2.13740814},
                     {-2.17164001, -2.0657259},
                     {-2.13100411, -2.05532175},
                     {-2.13100096, -2.12092003}};
  std::vector gj{
    std::vector{0.0, 0.0}, {1.9999, 1.9999}, {0.0, 0.0}, {1.5, 1.0}};

  for (std::size_t i = 0; i < CIWFs.size(); ++i) {
    const auto &ci_wf = CIWFs.at(i);
    REQUIRE(ci_wf.twoJ() == 2 * J[i]);
    REQUIRE(ci_wf.parity() == parity[i]);
    for (std::size_t j = 0ul; j < ci_wf.num_solutions(); ++j) {
      REQUIRE(ci_wf.energy(j) == Approx(energy[i][j]).epsilon(1.0e-2));
      REQUIRE(ci_wf.info(j).gJ == Approx(gj[i][j]).epsilon(1.0e-2));
    }
  }

  // again, this time, should read in qk file
  const auto CIWFs_2 = CI::configuration_interaction(input, wf);

  for (std::size_t i = 0; i < CIWFs.size(); ++i) {
    const auto &ci_wf = CIWFs.at(i);
    const auto &ci_wf_2 = CIWFs_2.at(i);
    REQUIRE(ci_wf.twoJ() == ci_wf_2.twoJ());
    REQUIRE(ci_wf.parity() == ci_wf_2.parity());
    for (std::size_t j = 0ul; j < ci_wf.num_solutions(); ++j) {
      REQUIRE(ci_wf.energy(j) == Approx(ci_wf_2.energy(j)));
      REQUIRE(ci_wf.info(j).gJ == Approx(ci_wf_2.info(j).gJ));
    }
  }

  //-----------------------------------------------------------------------
  // basic/Misc tests

  // Term(int two_J, int L, int two_S, int parity)

  REQUIRE(CI::Term_Symbol(2, 3, 2, +1) == "3^F_1");
  REQUIRE(CI::Term_Symbol(2, 3, 2, -1) == "3^F°_1");
  REQUIRE(CI::Term_Symbol(1, 3, 2, -1) == "3^F°_1/2");
  REQUIRE(CI::Term_Symbol(1, 2, 1, -1) == "2^D°_1/2");
  REQUIRE(CI::Term_Symbol(6, 0, 0, 1) == "1^S_3");
}

// test for the efficiency of function that fills the Pi matrix
TEST_CASE("CI: Constructing Pi matrix test", "[CI][unit][k78]") {
  Wavefunction wf({400, 1.0e-4, 45.0, 0.33 * 20.0, "loglinear"},
                  {"Ba", -1, "pointlike"}, 1.0);
  wf.solve_core("HartreeFock", std::nullopt, "[Xe]", 1.0e-5);
  wf.formBasis(
    SplineBasis::Parameters("30spdfghi", 45, 7, 1.0e-2, 1.0e-4, 50.0));

  const std::size_t i0 = 0;
  const std::size_t num_points = wf.grid().num_points();
  const std::size_t num_points_subgrid =
    num_points / 4; // stride in denominator
  const std::size_t stride = num_points / num_points_subgrid;

  MBPT::Feynman feyn = MBPT::Feynman(wf.vHF(), i0, stride, num_points_subgrid,
                                     {}, 1, true, true, false);

  const std::string cis2_basis_string = "12spdf";
  const std::vector<DiracSpinor> cis2_basis =
    CI::basis_subset(wf.basis(), cis2_basis_string, wf.coreConfiguration());
  int max_k_Coulomb = 7;

  // extract list of kappa from S^2 basis
  const auto kappai_list = AtomData::kappa_index_list(cis2_basis_string);

  std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now();

  std::vector<LinAlg::Matrix<MBPT::ComplexRMatrix>> PiMatrix(
    max_k_Coulomb,
    {kappai_list.size(), kappai_list.size(),
     MBPT::ComplexRMatrix{i0, stride, num_points_subgrid, wf.grid_sptr()}});

  MBPT::FillPiMatrix(max_k_Coulomb, cis2_basis, cis2_basis_string, kappai_list,
                     feyn, PiMatrix);

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  std::cout
    << std::endl
    << std::endl
    << "Time taken to form and fill Pi matrix: "
    << std::chrono::duration_cast<std::chrono::minutes>(end - begin).count();
}