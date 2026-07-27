#include "Angular/include.hpp"
#include "CI/SecondOrder.hpp"
#include "DiracOperator/include.hpp"
#include "ExternalField/CorePolarisation.hpp"
#include "ExternalField/calcMatrixElements.hpp"
#include "IO/ChronoTimer.hpp"
#include "IO/InputBlock.hpp"
#include "MBPT/StructureRad.hpp"
#include "Modules/Modules.hpp"
#include "Wavefunction/Wavefunction.hpp"
#include "fmt/color.hpp"
#include "fmt/format.hpp"
#include <cmath>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

//==============================================================================
// Parses the input list {J, parity, index} that identifies a CI level. The
// index may be omitted, in which case it is zero. Returns parity = 0 if the
// list is malformed
static CI::Level parse_level(const std::vector<int> &input_list) {
  if (input_list.size() < 2 || input_list.size() > 3)
    return {0, 0, 0};
  const auto parity = input_list.at(1);
  if (input_list.at(0) < 0 || (parity != 1 && parity != -1))
    return {0, 0, 0};
  const auto index = input_list.size() == 3 ? input_list.at(2) : 0;
  if (index < 0)
    return {0, 0, 0};
  return {2 * input_list.at(0), parity, std::size_t(index)};
}

//==============================================================================
// Is this rank K allowed: triangle rules for the operators, and for the states
static bool allowed_K(int K, int kt, int ks, int twoJb, int twoJa) {
  return K >= std::abs(kt - ks) && K <= kt + ks &&
         Angular::triangle(twoJb, 2 * K, twoJa) != 0;
}

//==============================================================================
// The smallest rank K allowed for the amplitude; negative if there is none
static int smallest_allowed_K(int kt, int ks, int twoJb, int twoJa) {
  for (int K = std::abs(kt - ks); K <= kt + ks; ++K) {
    if (allowed_K(K, kt, ks, twoJb, twoJa))
      return K;
  }
  return -1;
}

//==============================================================================
// Short label for a CI state, e.g., "J=1,- #0 6s6p 3P"
static std::string label(const CI::PsiJPi &psi, std::size_t index) {
  const auto &info = psi.info(index);
  return fmt::format("J={},{} #{} {} {}", psi.twoJ() / 2,
                     psi.parity() == 1 ? '+' : '-', index, info.config,
                     CI::Term_Symbol(int(std::round(info.L)),
                                     int(std::round(info.twoS)), psi.parity()));
}

//==============================================================================
namespace Module {

// Declare, register, then define below.
void CI_secondOrder(const IO::InputBlock &input, const Wavefunction &wf);
namespace {
const Register r_CI_secondOrder{
  "CI_secondOrder",
  "Second-order amplitudes between CI states (polarisabilities, PNC), via CI "
  "mixed states",
  &CI_secondOrder};
} // namespace

void CI_secondOrder(const IO::InputBlock &input, const Wavefunction &wf) {

  input.check(
    {{"",
      "Second-order amplitude A^K between two CI states, a -> b, for a dynamic "
      "operator t and a static operator s:\n"
      "A^K = sum_n [c1 <b||t||n><n||s||a>/(E_a-E_n) "
      "+ c2 <b||s||n><n||t||a>/(E_a+omega-E_n)].\n"
      "The sums over the intermediate spectrum are evaluated with CI mixed "
      "states, so are complete (no sum over CI solutions). Requires a CI{} "
      "block (which is where the CI options are set)."},
     {"a", "Initial CI state, as J,parity,index - e.g., '0,1,0' is the lowest "
           "J=0 even-parity solution. Parity is +1 or -1; the index counts "
           "from 0, in order of energy [required]"},
     {"b", "Final CI state, as for a [default: same as a]"},
     {"t", "Dynamic operator: the one that carries the frequency omega [E1]"},
     {"t_options{}", "Options for the t operator"},
     {"s", "Static operator [E1]"},
     {"s_options{}", "Options for the s operator"},
     {"omega", "Frequency of t [default: E_b - E_a]"},
     {"K", "Rank K of the amplitude. Requires |kt-ks| <= K <= kt+ks, and the "
           "triangle rule for (Jb,K,Ja) [default: smallest allowed]"},
     {"rpa", "Method used for RPA: true(=TDHF), false, TDHF, basis, diagram "
             "[true]"},
     {"project_out",
      "List of CI states, as J,parity,index triples, that the intermediate "
      "states are forced to be orthogonal to - e.g., '1,-1,0, 1,-1,1' removes "
      "the two lowest J=1 odd solutions from the sums [none]"},
     {"StructureRadiation{}",
      "Options for structure radiation and normalisation. If this block is "
      "included, SR+N is added to every single-particle matrix element used in "
      "the amplitude, including the internal lines. This double counts against "
      "sigma2, and is not a controlled approximation: use with care [not "
      "included]"}});

  // Check for Structure Radiation
  const auto t_SR_input = input.getBlock("StructureRadiation");
  auto SR_input =
    t_SR_input ? *t_SR_input : IO::InputBlock{"StructureRadiation"};
  if (input.has_option("help")) {
    SR_input.add("help;");
  }
  SR_input.check(
    {{"", "If this block is included, SR + Normalisation corrections will be "
          "included"},
     {"Qk_file", "true/false/filename - SR: filename for QkTable file. If "
                 "blank will not use QkTable; if exists, will read it in; if "
                 "doesn't exist, will create it and write to disk. If 'true' "
                 "will use default filename"},
     {"n_minmax", "list; min,max n for core/excited: [1,inf]"}});

  // If we are just requesting 'help', don't run module:
  if (input.has_option("help")) {
    return;
  }

  using namespace std::string_literals;

  IO::ChronoTimer timer("CI_secondOrder");

  //----------------------------------------------------------------------------
  // The CI solutions, and the integrals used to construct them
  const auto &ints = wf.CI_integrals();
  if (wf.CIwfs().empty() || !ints.availableQ()) {
    fmt2::error();
    std::cout << ": Requires CI solutions and the integrals used to construct "
                 "the CI Hamiltonian. Include a CI{} block (and don't run it "
                 "with read_only)\n";
    return;
  }

  //----------------------------------------------------------------------------
  // The two CI states
  const auto input_a = input.get("a", std::vector<int>{});
  const auto level_a = parse_level(input_a);
  const auto level_b = parse_level(input.get("b", input_a));
  const auto Psi_a = wf.CIwf(level_a.twoJ / 2, level_a.parity);
  const auto Psi_b = wf.CIwf(level_b.twoJ / 2, level_b.parity);
  const auto ia = level_a.index;
  const auto ib = level_b.index;
  if (Psi_a == nullptr || Psi_b == nullptr || ia >= Psi_a->num_solutions() ||
      ib >= Psi_b->num_solutions()) {
    fmt2::error();
    std::cout << ": Could not find the requested CI state(s). Give as "
                 "J,parity,index; e.g., a = 0,1,0; b = 1,-1,0;\n"
                 "Available (J,parity):";
    for (const auto &psi : wf.CIwfs()) {
      fmt::print(" ({},{:+})", psi.twoJ() / 2, psi.parity());
    }
    std::cout << "\n";
    return;
  }
  const auto Ea = Psi_a->energy(ia);
  const auto Eb = Psi_b->energy(ib);

  std::cout << "\nSecond-order amplitude, a -> b:\n";
  fmt::print("a: {}  E = {:.8f} au\n", label(*Psi_a, ia), Ea);
  fmt::print("b: {}  E = {:.8f} au\n", label(*Psi_b, ib), Eb);

  //----------------------------------------------------------------------------
  // The two operators
  const auto t_name = input.get("t", "E1"s);
  const auto s_name = input.get("s", "E1"s);
  const auto t_options = input.getBlock("t_options");
  const auto s_options = input.getBlock("s_options");
  const auto ht = DiracOperator::generate(
    t_name, t_options ? *t_options : IO::InputBlock(t_name, {}), wf);
  const auto hs = DiracOperator::generate(
    s_name, s_options ? *s_options : IO::InputBlock(s_name, {}), wf);
  const auto kt = ht->rank();
  const auto ks = hs->rank();

  const auto omega = input.get("omega", Eb - Ea);

  fmt::print("t: {} (rank {}, {} parity), dynamic, at omega = {:.6f}\n",
             ht->name(), kt, ht->parity() == 1 ? "even" : "odd", omega);
  fmt::print("s: {} (rank {}, {} parity), static\n", hs->name(), ks,
             hs->parity() == 1 ? "even" : "odd");
  if (std::abs(omega - (Eb - Ea)) > 1.0e-10) {
    fmt::print("nb: omega is not the transition frequency, {:.6f}\n", Eb - Ea);
  }

  // Overall parity selection rule
  if (Psi_a->parity() * Psi_b->parity() != ht->parity() * hs->parity()) {
    fmt2::warning();
    std::cout << ": Amplitude is zero by parity\n";
    return;
  }

  // Rank K: default is the smallest allowed by the triangle rules
  const auto K_minimum =
    smallest_allowed_K(kt, ks, Psi_b->twoJ(), Psi_a->twoJ());
  if (K_minimum < 0) {
    fmt2::warning();
    std::cout << ": No allowed K. Require |kt-ks| <= K <= kt+ks, and the "
                 "triangle rule for (Jb, K, Ja)\n";
    return;
  }
  const auto K = input.get("K", K_minimum);
  fmt::print("K = {}{}\n", K, K == K_minimum ? " (smallest allowed)" : "");
  if (!allowed_K(K, kt, ks, Psi_b->twoJ(), Psi_a->twoJ())) {
    fmt2::warning();
    std::cout << ": K is not allowed: the amplitude is zero\n";
  }

  //----------------------------------------------------------------------------
  // Levels to remove from the intermediate states
  const auto level_list = input.get("project_out", std::vector<int>{});
  if (level_list.size() % 3 != 0) {
    fmt2::error();
    std::cout << ": project_out must be a list of J,parity,index triples\n";
    return;
  }
  std::vector<CI::Level> levels_to_remove;
  for (std::size_t i = 0; i < level_list.size(); i += 3) {
    levels_to_remove.push_back({2 * level_list.at(i), level_list.at(i + 1),
                                std::size_t(level_list.at(i + 2))});
  }
  if (!levels_to_remove.empty()) {
    std::cout << "\nRemoving from the intermediate states:";
    for (const auto &level : levels_to_remove) {
      fmt::print(" (J={},{} #{})", level.twoJ / 2,
                 level.parity == 1 ? '+' : '-', level.index);
    }
    std::cout << "\n";
  }

  //----------------------------------------------------------------------------
  // RPA
  auto rpa_method = input.get("rpa", "true"s);
  if (wf.core().empty()) {
    rpa_method = "false";
  }
  std::cout << "\n";
  auto rpa_t = ExternalField::make_rpa(rpa_method, ht.get(), wf.vHF(), true,
                                       wf.basis(), wf.identity());
  auto rpa_s = ExternalField::make_rpa(rpa_method, hs.get(), wf.vHF(), false,
                                       wf.basis(), wf.identity());
  if (ht->freqDependantQ()) {
    ht->updateFrequency(omega);
  }
  if (rpa_t) {
    std::cout << "Solve RPA for t at omega = " << omega << "\n";
    rpa_t->solve_core(omega);
  }
  if (rpa_s) {
    std::cout << "Solve RPA for s at omega = 0\n";
    rpa_s->solve_core(0.0);
  }

  //----------------------------------------------------------------------------
  // Structure radiation
  std::optional<MBPT::StructureRad> sr;
  if (t_SR_input) {
    const auto n_minmax = SR_input.get("n_minmax", std::vector{1});
    const auto n_min = n_minmax.size() > 0 ? n_minmax[0] : 1;
    const auto n_max = n_minmax.size() > 1 ? n_minmax[1] : 999;
    const auto Qk_file_t = SR_input.get("Qk_file", "false"s);
    const std::string Qk_file =
      Qk_file_t != "false" ?
        (Qk_file_t == "true" ? wf.identity() + ".qk.abf" : Qk_file_t) :
        "";
    std::cout << "\nIncluding structure radiation and normalisation:\n";
    fmt2::warning();
    std::cout << ": SR+N is added to every single-particle matrix element, "
                 "including the internal lines. This is not a controlled "
                 "approximation\n";
    sr =
      MBPT::StructureRad(wf.basis(), wf.FermiLevel(), {n_min, n_max}, Qk_file);
  }

  //----------------------------------------------------------------------------
  // Single-particle matrix element tables
  std::cout << "\nFill matrix element tables..." << std::flush;
  if (sr) {
    sr->solve_core(ht.get(), rpa_t.get());
  }
  const auto t_me = ExternalField::me_table(
    ints.ci_basis, ht.get(), rpa_t.get(), sr ? &*sr : nullptr, omega);
  if (sr) {
    sr->solve_core(hs.get(), rpa_s.get());
  }
  const auto s_me = ExternalField::me_table(
    ints.ci_basis, hs.get(), rpa_s.get(), sr ? &*sr : nullptr, 0.0);
  std::cout << "done\n" << std::flush;

  //----------------------------------------------------------------------------
  // The amplitude
  std::cout << "\nSums over the intermediate spectrum:\n";
  const auto [A_ket, A_bra] =
    CI::A_K(K, *Psi_b, ib, *Psi_a, ia, ht.get(), t_me, hs.get(), s_me, omega,
            ints, levels_to_remove);

  // The z-component of the rank-K amplitude: m_a = m_b = m, and q = 0 for both
  // operators
  const auto two_m = std::min(Psi_a->twoJ(), Psi_b->twoJ());
  const auto A_K0 =
    A_ket * CI::z_component(K, kt, ks, Psi_b->twoJ(), Psi_a->twoJ(), two_m);

  fmt::print("\nA^{}      = {:.6e}  (from the ket)\n", K, A_ket);
  fmt::print("A^{}      = {:.6e}  (from the bra) [eps = {:.1e}]\n", K, A_bra,
             std::abs(A_ket - A_bra) / std::max(std::abs(A_ket), 1.0e-30));
  fmt::print("A^{}_0    = {:.6e}  (z-component, m = {})\n", K, A_K0, two_m / 2);

  //----------------------------------------------------------------------------
  // Specific quantities, as in the dcp module. Only s is tested: t is E1 in
  // all of these cases
  const auto E1_s = hs->name() == "E1";

  // Scalar polarisability
  if (K == 0 && E1_s) {
    // alpha_0 = (2/3)[J]^-1 sum_n |<a||d||n>|^2/(E_n-E_a), for a = b
    const auto alpha = A_ket / std::sqrt(3.0 * (Psi_b->twoJ() + 1));
    fmt::print("\nalpha (scalar{}polarisability) = {:.6e} au\n",
               Psi_a == Psi_b && ia == ib ? " " : " transition ", alpha);
  }

  // Tensor polarisability. The normalisation comes from the m-dependence of
  // the Stark shift, so it requires J_a = J_b (as the scalar part does, though
  // there K=0 already implies it), and J >= 1
  if (K == 2 && E1_s && Psi_a->twoJ() == Psi_b->twoJ() && Psi_b->twoJ() >= 2) {
    const auto twoJ = double(Psi_b->twoJ());
    const auto factor =
      -std::sqrt(2.0 * twoJ * (twoJ - 1.0) /
                 (3.0 * (twoJ + 1.0) * (twoJ + 2.0) * (twoJ + 3.0)));
    fmt::print("\nalpha_2 (tensor{}polarisability) = {:.6e} au\n",
               Psi_a == Psi_b && ia == ib ? " " : " transition ",
               factor * A_ket);
  }

  // Vector transition polarisability, beta = A^1/(sqrt(2) <b||sigma||a>).
  // The spin matrix element expresses the Wigner-Eckart factor of the rank-1
  // amplitude, so no radial overlap enters it; see CI::sigma_rme
  if (K == 1 && E1_s) {
    const auto sigma = CI::sigma_rme(*Psi_b, ib, *Psi_a, ia, ints.ci_basis);
    fmt::print("\n<b||sigma||a> = {:.6e}\n", sigma);
    if (std::abs(sigma) < 1.0e-12) {
      std::cout << "beta: not defined - the states have no spin-angular "
                   "structure in common\n";
    } else {
      fmt::print("beta (vector{}polarisability) = {:.6e} au\n",
                 Psi_a == Psi_b && ia == ib ? " " : " transition ",
                 A_ket / (std::sqrt(2.0) * sigma));
    }
  }

  // PNC amplitude: the static operator is the PNC interaction
  if (hs->name().substr(0, 3) == "pnc") {
    fmt::print("\nE_pnc = A^{}_0 = {:.6e} {}\n", K, A_K0, hs->units());
  }
}

} // namespace Module
