#include "Angular/include.hpp"
#include "CI/SecondOrder.hpp"
#include "DiracOperator/include.hpp"
#include "ExternalField/CorePolarisation.hpp"
#include "ExternalField/calcMatrixElements.hpp"
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
// Short label for a CI state, e.g., "1-:0  6s6p 3P"
static std::string label(const CI::PsiJPi &psi, std::size_t index) {
  const auto &info = psi.info(index);
  return fmt::format("{:<6} {} {}",
                     CI::to_string({psi.twoJ(), psi.parity(), index}),
                     info.config,
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
      "Second-order amplitude A^K between two CI states, A -> B, for two "
      "operators t and s, at frequencies omega and omega_s:\n"
      "A^K = sum_n [c1 <B||t||n><n||s||A>/(E_A+omega_s-E_n) "
      "+ c2 <B||s||n><n||t||A>/(E_A+omega-E_n)].\n"
      "Energy conservation fixes omega_s = E_B - E_A - omega, so only omega is "
      "an input: at its default, t carries the whole transition frequency and "
      "s is static. For a dynamic polarisability, set B = A and omega to the "
      "frequency; then omega_s = -omega.\n"
      "The sums over the intermediate spectrum are evaluated with CI mixed "
      "states, so are complete (no sum over CI solutions). Requires a CI{} "
      "block (which is where the CI options are set)."},
     {"A", "Initial CI state, as J{+,-}:index - e.g., '0+' is the lowest J=0 "
           "even-parity solution, and '1-:2' the third J=1 odd-parity one (the "
           "index counts from 0, in order of energy, and may be omitted). The "
           "form 'e0', 'o1:2' is also accepted [required]"},
     {"B", "Final CI state, as for A [default: same as A]"},
     {"t", "The operator that carries the frequency omega [E1]"},
     {"t_options{}", "Options for the t operator"},
     {"s", "The other operator; it carries omega_s = E_b - E_a - omega [E1]"},
     {"s_options{}", "Options for the s operator"},
     {"omega", "Frequency of t. For a transition, leave as default, so that t "
               "carries it all and s is static. For a dynamic polarisability "
               "(B = A), set this to the frequency: s then carries -omega. "
               "[default: E_b - E_a]"},
     {"K", "Rank K of the amplitude. Requires |kt-ks| <= K <= kt+ks, and the "
           "triangle rule for (Jb,K,Ja) [default: smallest allowed]"},
     {"rpa", "Method used for RPA: true(=TDHF), false, TDHF, basis, diagram "
             "[true]"},
     {"project_out",
      "List of CI states, as for A, that are removed from the intermediate "
      "states - e.g., '1-:0, 1-:1' removes the two lowest J=1 odd solutions "
      "from the sums [none]"},
     {"StructureRadiation{}",
      "Options for structure radiation. If this block is included, SR is "
      "added to every single-particle matrix element used in the amplitude: "
      "use with care. The normalisation of states is NOT included: it cannot "
      "be done with mixed states, which never resolve the individual "
      "intermediate states. Use the CI_Pol module for that"}});

  // Check for Structure Radiation
  const auto t_SR_input = input.getBlock("StructureRadiation");
  auto SR_input =
    t_SR_input ? *t_SR_input : IO::InputBlock{"StructureRadiation"};
  if (input.has_option("help")) {
    SR_input.add("help;");
  }
  SR_input.check(
    {{"", "If this block is included, structure radiation will be included. "
          "The normalisation of states is not available in this module"},
     {"Qk_file", "true/false/filename - SR: filename for QkTable file. If "
                 "blank will not use QkTable; if exists, will read it in; if "
                 "doesn't exist, will create it and write to disk. If 'true' "
                 "will use default filename"},
     {"n_minmax", "list; min,max n for core/excited (internal): [1,inf]"},
     {"n_max_legs",
      "Max n of the CI basis states that SR is applied to. SR is only "
      "meaningful between physical states, and the high-n states of the CI "
      "basis are cavity states [default: max_n_core + 3]"}});

  // If we are just requesting 'help', don't run module:
  if (input.has_option("help")) {
    return;
  }

  using namespace std::string_literals;

  //----------------------------------------------------------------------------
  // The CI solutions, and the integrals used to construct them
  const auto &ints = wf.CI_integrals();
  if (wf.CIwfs().empty() || !ints.availableQ()) {
    fmt2::error();
    std::cout << ": Requires CI solutions and the integrals used to construct "
                 "the CI Hamiltonian. Include a CI{} block\n";
    return;
  }

  //----------------------------------------------------------------------------
  // The two CI states
  const auto input_A = input.get("A", ""s);
  const auto level_A = CI::parse_level(input_A);
  const auto level_B = CI::parse_level(input.get("B", input_A));
  const auto Psi_a =
    level_A ? wf.CIwf(level_A->twoJ / 2, level_A->parity) : nullptr;
  const auto Psi_b =
    level_B ? wf.CIwf(level_B->twoJ / 2, level_B->parity) : nullptr;
  const auto ia = level_A ? level_A->index : 0;
  const auto ib = level_B ? level_B->index : 0;
  if (Psi_a == nullptr || Psi_b == nullptr || ia >= Psi_a->num_solutions() ||
      ib >= Psi_b->num_solutions()) {
    fmt2::error();
    std::cout << ": Could not find the requested CI state(s). Give as "
                 "J{+,-}:index; e.g., A = 0+; B = 1-:2;\n"
                 "Available (J,parity):";
    for (const auto &psi : wf.CIwfs()) {
      fmt::print(" {}{}", psi.twoJ() / 2, psi.parity() == 1 ? '+' : '-');
    }
    std::cout << "\n";
    return;
  }
  const auto Ea = Psi_a->energy(ia);
  const auto Eb = Psi_b->energy(ib);

  std::cout << "\nSecond-order amplitude, A -> B:\n";
  fmt::print("A: {}  E = {:.8f} au\n", label(*Psi_a, ia), Ea);
  fmt::print("B: {}  E = {:.8f} au\n", label(*Psi_b, ib), Eb);

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

  // s carries whatever frequency t does not, so that energy is conserved:
  // E_b = E_a + omega + omega_s. With omega at its default, s is static, as
  // for a real transition. For a dynamic polarisability, A = B (so E_b = E_a)
  // and omega is set: then omega_s = -omega, giving the usual pair of
  // denominators, E_a -+ omega - E_n
  const auto omega_s = Eb - Ea - omega;
  const auto dynamic = omega != 0.0 && omega_s != 0.0;

  const auto print_op = [](const std::string &label,
                           const DiracOperator::TensorOperator *h, double w) {
    fmt::print("{}: {} (rank {}, {} parity), ", label, h->name(), h->rank(),
               h->parity() == 1 ? "even" : "odd");
    if (w == 0.0) {
      fmt::print("static\n");
    } else {
      fmt::print("at omega = {:.6f}\n", w);
    }
  };
  print_op("t", ht.get(), omega);
  print_op("s", hs.get(), omega_s);

  // Overall parity selection rule
  if (Psi_a->parity() * Psi_b->parity() != ht->parity() * hs->parity()) {
    std::cout << ": Amplitude is zero by parity\n";
    return;
  }

  // Rank K: default is the smallest allowed by the triangle rules
  const auto K_minimum =
    smallest_allowed_K(kt, ks, Psi_b->twoJ(), Psi_a->twoJ());
  if (K_minimum < 0) {
    std::cout << ": No allowed K. Require |kt-ks| <= K <= kt+ks, and the "
                 "triangle rule for (Jb, K, Ja)\n";
    return;
  }
  const auto K = input.get("K", K_minimum);
  fmt::print("K = {}{}\n", K, K == K_minimum ? " (smallest allowed)" : "");
  if (!allowed_K(K, kt, ks, Psi_b->twoJ(), Psi_a->twoJ())) {
    std::cout << ": K is not allowed: the amplitude is zero\n";
    return;
  }

  //----------------------------------------------------------------------------
  // Levels to remove from the intermediate states
  std::vector<CI::Level> levels_to_remove;
  for (const auto &str : input.get("project_out", std::vector<std::string>{})) {
    const auto level = CI::parse_level(str);
    if (!level) {
      fmt2::error();
      fmt::print(
        ": Could not parse '{}' as a CI level: give as J{{+,-}}:index, "
        "e.g., 1-:0\n",
        str);
      return;
    }
    levels_to_remove.push_back(*level);
  }
  if (!levels_to_remove.empty()) {
    std::cout << "\nRemoving from the intermediate states:";
    for (const auto &level : levels_to_remove) {
      fmt::print(" {}", CI::to_string(level));
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
  if (hs->freqDependantQ()) {
    hs->updateFrequency(omega_s);
  }
  if (rpa_t) {
    std::cout << "Solve RPA for t at omega = " << omega << "\n";
    rpa_t->solve_core(omega);
  }
  if (rpa_s) {
    std::cout << "Solve RPA for s at omega = " << omega_s << "\n";
    rpa_s->solve_core(omega_s);
  }

  //----------------------------------------------------------------------------
  // Structure radiation. SR+N is only meaningful between physical states, so
  // it is applied only to the low-n part of the CI basis
  const auto sr_n_max =
    SR_input.get("n_max_legs", DiracSpinor::max_n(wf.core()) + 3);

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
    std::cout << "\nIncluding structure radiation (no normalisation):\n";
    std::cout << "Added to the single-particle matrix elements used in the "
                 "amplitude, including the internal lines\n";
    fmt::print("Applied to CI basis states with n <= {}\n", sr_n_max);
    sr =
      MBPT::StructureRad(wf.basis(), wf.FermiLevel(), {n_min, n_max}, Qk_file);
  }

  //----------------------------------------------------------------------------
  // Single-particle matrix element tables
  std::cout << "\nFill matrix element tables..." << std::flush;
  if (sr) {
    sr->solve_core(ht.get(), rpa_t.get());
  }
  // sr_norm is false: the normalisation of states must not go onto the
  // single-particle matrix elements, which drops the spectator electrons. It
  // is a property of the CI state; see CI::ReducedME_norm. It is not included
  // in this module at all
  const auto t_me =
    ExternalField::me_table(ints.ci_basis, ht.get(), rpa_t.get(),
                            sr ? &*sr : nullptr, omega, sr_n_max, false);
  if (sr) {
    sr->solve_core(hs.get(), rpa_s.get());
  }
  const auto s_me =
    ExternalField::me_table(ints.ci_basis, hs.get(), rpa_s.get(),
                            sr ? &*sr : nullptr, omega_s, sr_n_max, false);

  std::cout << "done\n" << std::flush;

  //----------------------------------------------------------------------------
  // The amplitude
  fmt::print("\nContributions to A^{}, by intermediate J and parity:\n", K);
  const auto [A_s, A_t] =
    CI::A_K(K, *Psi_b, ib, *Psi_a, ia, ht.get(), t_me, hs.get(), s_me, omega,
            omega_s, ints, levels_to_remove);

  //----------------------------------------------------------------------------
  // Intermediate states carrying a core hole. These lie outside the CI space,
  // so the mixed states cannot produce them: the polarisation of the core
  // (K=0 and diagonal only), and the core-valence term, which is the Pauli
  // blocking of the core excitations by the valence electrons
  const auto &spectrum = wf.spectrum().empty() ? wf.basis() : wf.spectrum();
  const auto excited =
    DiracSpinor::split_by_energy(spectrum, wf.FermiLevel()).second;

  // The core is the same in A and B, so its amplitude needs <B|A>
  const auto A_core =
    (Psi_a == Psi_b && ia == ib) ?
      CI::A_K_core(K, Psi_a->twoJ(), ht.get(), hs.get(), omega, omega_s,
                   wf.core(), excited, rpa_t.get(), rpa_s.get()) :
      0.0;
  const auto A_cv =
    CI::A_K_cv(K, *Psi_b, ib, *Psi_a, ia, ht.get(), hs.get(), omega, omega_s,
               wf.core(), ints.ci_basis, rpa_t.get(), rpa_s.get());

  // No normalisation of states anywhere: see the note on the SR block
  const auto A_total = A_s + A_cv + A_core;

  //----------------------------------------------------------------------------
  // The second column of the table is the quantity that was asked for. For the
  // specific cases below (as in the dcp module) that is alpha/beta/E_pnc;
  // otherwise it is the z-component, A^K_0. Only s is tested: t is E1 in all
  // of these cases
  const auto E1_s = hs->name() == "E1";
  const auto diagonal = Psi_a == Psi_b && ia == ib;
  // e.g., "Scalar polarisability", "Scalar dynamic polarisability",
  // "Scalar transition polarisability"
  const auto kind =
    " "s + (dynamic ? "dynamic "s : ""s) + (diagonal ? ""s : "transition "s);

  // The z-component of the rank-K amplitude: m_a = m_b = m, and q = 0 for both
  // operators
  const auto two_m = std::min(Psi_a->twoJ(), Psi_b->twoJ());
  auto title = fmt::format("z-component, m = {}", two_m / 2);
  auto name = fmt::format("A^{}_0", K);
  auto factor = CI::z_component(K, kt, ks, Psi_b->twoJ(), Psi_a->twoJ(), two_m);

  // Scalar polarisability
  if (K == 0 && E1_s) {
    // alpha_0 = (2/3)[J]^-1 sum_n |<a||d||n>|^2/(E_n-E_a), for a = b
    title = fmt::format("Scalar{}polarisability", kind);
    name = "alpha (au)";
    factor = 1.0 / std::sqrt(3.0 * (Psi_b->twoJ() + 1));
  }

  // Tensor polarisability. The normalisation comes from the m-dependence of
  // the Stark shift, so it requires J_a = J_b (as the scalar part does, though
  // there K=0 already implies it), and J >= 1
  if (K == 2 && E1_s && Psi_a->twoJ() == Psi_b->twoJ() && Psi_b->twoJ() >= 2) {
    const auto twoJ = double(Psi_b->twoJ());
    title = fmt::format("Tensor{}polarisability", kind);
    name = "alpha_2 (au)";
    factor = -std::sqrt(2.0 * twoJ * (twoJ - 1.0) /
                        (3.0 * (twoJ + 1.0) * (twoJ + 2.0) * (twoJ + 3.0)));
  }

  // Vector transition polarisability, beta = A^1/(sqrt(2) <b||sigma||a>).
  // The spin matrix element expresses the Wigner-Eckart factor of the rank-1
  // amplitude, so no radial overlap enters it; see CI::sigma_rme
  if (K == 1 && E1_s) {
    const auto sigma = CI::sigma_rme(*Psi_b, ib, *Psi_a, ia, ints.ci_basis);
    fmt::print("\nVector{}polarisability:\n", kind);
    fmt::print("<B||sigma||A> = {:.6e}\n", sigma);
    if (std::abs(sigma) < 1.0e-12) {
      std::cout << "beta: not defined - the states have no spin-angular "
                   "structure in common\n";
    } else {
      title = fmt::format("Vector{}polarisability", kind);
      name = "beta (au)";
      factor = 1.0 / (std::sqrt(2.0) * sigma);
    }
  }

  // PNC amplitude: the static operator is the PNC interaction
  if (hs->name().substr(0, 3) == "pnc") {
    title = "PNC amplitude";
    name = fmt::format("E_pnc ({})", hs->units());
  }

  //----------------------------------------------------------------------------
  fmt::print("\n{}:\n", title);
  fmt::print("{:<14} {:>16} {:>16}\n", "", fmt::format("A^{}", K), name);
  fmt::print("{:<14} {:16.6e} {:16.6e}\n", "valence (CI)", A_s, factor * A_s);
  fmt::print("{:<14} {:16.6e} {:16.6e}\n", "core", A_core, factor * A_core);
  fmt::print("{:<14} {:16.6e} {:16.6e}\n", "core-valence", A_cv, factor * A_cv);
  fmt::print("{:<14} {:16.6e} {:16.6e}\n", "total", A_total, factor * A_total);
}

} // namespace Module
