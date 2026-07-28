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
      "Second-order amplitude A^K between two CI states, A -> B, for a dynamic "
      "operator t and a static operator s:\n"
      "A^K = sum_n [c1 <B||t||n><n||s||A>/(E_A-E_n) "
      "+ c2 <B||s||n><n||t||A>/(E_A+omega-E_n)].\n"
      "The sums over the intermediate spectrum are evaluated with CI mixed "
      "states, so are complete (no sum over CI solutions). Requires a CI{} "
      "block (which is where the CI options are set)."},
     {"A", "Initial CI state, as J{+,-}:index - e.g., '0+' is the lowest J=0 "
           "even-parity solution, and '1-:2' the third J=1 odd-parity one (the "
           "index counts from 0, in order of energy, and may be omitted). The "
           "form 'e0', 'o1:2' is also accepted [required]"},
     {"B", "Final CI state, as for A [default: same as A]"},
     {"t", "Dynamic operator: the one that carries the frequency omega [E1]"},
     {"t_options{}", "Options for the t operator"},
     {"s", "Static operator [E1]"},
     {"s_options{}", "Options for the s operator"},
     {"omega", "Frequency of t. Generally must be transition frequency, and be "
               "left as default. [default: "
               "E_b - E_a]"},
     {"K", "Rank K of the amplitude. Requires |kt-ks| <= K <= kt+ks, and the "
           "triangle rule for (Jb,K,Ja) [default: smallest allowed]"},
     {"rpa", "Method used for RPA: true(=TDHF), false, TDHF, basis, diagram "
             "[true]"},
     {"project_out",
      "List of CI states, as for A, that are removed from the intermediate "
      "states - e.g., '1-:0, 1-:1' removes the two lowest J=1 odd solutions "
      "from the sums [none]"},
     {"StructureRadiation{}",
      "Options for structure radiation and normalisation. If this block is "
      "included, SR+N is added to every single-particle matrix element used in "
      "the amplitude: use with care"}});

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
     {"n_minmax", "list; min,max n for core/excited (internal): [1,inf]"},
     {"n_max_legs",
      "Max n of the CI basis states that SR+N is applied to. SR+N is only "
      "meaningful between physical states, and the high-n states of the CI "
      "basis are cavity states [default: max_n_core + 3]"},
     {"norm", "Include the normalisation of states? If false, only the "
              "structure radiation is included [true]"}});

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
                 "the CI Hamiltonian. Include a CI{} block (and don't run it "
                 "with read_only)\n";
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

  // nb: t is the dynamic operator; it is only static if omega happens to be
  // zero (as for the diagonal, static, polarisabilities)
  if (omega == 0.0) {
    fmt::print("t: {} (rank {}, {} parity), static\n", ht->name(), kt,
               ht->parity() == 1 ? "even" : "odd");
  } else {
    fmt::print("t: {} (rank {}, {} parity), dynamic, at omega = {:.6f}\n",
               ht->name(), kt, ht->parity() == 1 ? "even" : "odd", omega);
  }
  fmt::print("s: {} (rank {}, {} parity), static\n", hs->name(), ks,
             hs->parity() == 1 ? "even" : "odd");

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
  if (rpa_t) {
    std::cout << "Solve RPA for t at omega = " << omega << "\n";
    rpa_t->solve_core(omega);
  }
  if (rpa_s) {
    std::cout << "Solve RPA for s at omega = 0\n";
    rpa_s->solve_core(0.0);
  }

  //----------------------------------------------------------------------------
  // Structure radiation. SR+N is only meaningful between physical states, so
  // it is applied only to the low-n part of the CI basis
  const auto sr_n_max =
    SR_input.get("n_max_legs", DiracSpinor::max_n(wf.core()) + 3);
  const auto sr_norm = SR_input.get("norm", true);

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
    std::cout << "\nIncluding structure radiation";
    std::cout << (sr_norm ? " and normalisation:\n" : " (no normalisation):\n");
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
  // The normalisation is applied to the CI states, not to the single-particle
  // matrix elements: it is a property of the state, so every valence electron
  // contributes, the spectators included. See CI::norm_factor
  const auto t_me =
    ExternalField::me_table(ints.ci_basis, ht.get(), rpa_t.get(),
                            sr ? &*sr : nullptr, omega, sr_n_max, false);
  if (sr) {
    sr->solve_core(hs.get(), rpa_s.get());
  }
  const auto s_me =
    ExternalField::me_table(ints.ci_basis, hs.get(), rpa_s.get(),
                            sr ? &*sr : nullptr, 0.0, sr_n_max, false);

  // One-body norm defect, for the CI states
  const auto f_norm = sr && sr_norm ?
                        CI::f_norm_table(*sr, ints.ci_basis, sr_n_max) :
                        Coulomb::meTable<double>{};
  std::cout << "done\n" << std::flush;

  //----------------------------------------------------------------------------
  // The amplitude
  fmt::print("\nContributions to A^{}, by intermediate J and parity:\n", K);
  const auto [A_s, A_t] =
    CI::A_K(K, *Psi_b, ib, *Psi_a, ia, ht.get(), t_me, hs.get(), s_me, omega,
            ints, levels_to_remove, f_norm);

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
      CI::A_K_core(K, Psi_a->twoJ(), ht.get(), hs.get(), omega, wf.core(),
                   excited, rpa_t.get(), rpa_s.get()) :
      0.0;
  const auto A_cv =
    CI::A_K_cv(K, *Psi_b, ib, *Psi_a, ia, ht.get(), hs.get(), omega, wf.core(),
               ints.ci_basis, rpa_t.get(), rpa_s.get());

  // Normalisation of the external legs: A^K(1 + F_a + F_b). The intermediate
  // states are done inside A_K; the core term has its own normalisation, which
  // is a core-correlation quantity and is not included
  const auto F_a = f_norm.empty() ? 0.0 : CI::norm_factor(*Psi_a, ia, f_norm);
  const auto F_b = f_norm.empty() ? 0.0 : CI::norm_factor(*Psi_b, ib, f_norm);
  const auto legs = 1.0 + F_a + F_b;

  const auto A_total = legs * (A_s + A_cv) + A_core;

  fmt::print("\nA^{}:\n", K);
  fmt::print("valence (CI)  {:16.6e}\n", A_s);
  fmt::print("core          {:16.6e}\n", A_core);
  fmt::print("core-valence  {:16.6e}\n", A_cv);
  if (!f_norm.empty()) {
    fmt::print("norm (legs)   {:16.6e}   [F_a = {:.2e}, F_b = {:.2e}]\n",
               (legs - 1.0) * (A_s + A_cv), F_a, F_b);
  }
  fmt::print("total         {:16.6e}\n", A_total);

  // The z-component of the rank-K amplitude: m_a = m_b = m, and q = 0 for both
  // operators
  const auto two_m = std::min(Psi_a->twoJ(), Psi_b->twoJ());
  const auto A_K0 =
    A_total * CI::z_component(K, kt, ks, Psi_b->twoJ(), Psi_a->twoJ(), two_m);

  fmt::print("\nA^{}_0 = {:.6e}   (z-component, m = {})\n", K, A_K0, two_m / 2);

  //----------------------------------------------------------------------------
  // Specific quantities, as in the dcp module. Only s is tested: t is E1 in
  // all of these cases
  const auto E1_s = hs->name() == "E1";
  const auto diagonal = Psi_a == Psi_b && ia == ib;

  // Scalar polarisability
  if (K == 0 && E1_s) {
    // alpha_0 = (2/3)[J]^-1 sum_n |<a||d||n>|^2/(E_n-E_a), for a = b
    const auto alpha = A_total / std::sqrt(3.0 * (Psi_b->twoJ() + 1));
    fmt::print("\nScalar{}polarisability:\n", diagonal ? " " : " transition ");
    fmt::print("alpha = {:.6e} au\n", alpha);
  }

  // Tensor polarisability. The normalisation comes from the m-dependence of
  // the Stark shift, so it requires J_a = J_b (as the scalar part does, though
  // there K=0 already implies it), and J >= 1
  if (K == 2 && E1_s && Psi_a->twoJ() == Psi_b->twoJ() && Psi_b->twoJ() >= 2) {
    const auto twoJ = double(Psi_b->twoJ());
    const auto factor =
      -std::sqrt(2.0 * twoJ * (twoJ - 1.0) /
                 (3.0 * (twoJ + 1.0) * (twoJ + 2.0) * (twoJ + 3.0)));
    fmt::print("\nTensor{}polarisability:\n", diagonal ? " " : " transition ");
    fmt::print("alpha_2 = {:.6e} au\n", factor * A_total);
  }

  // Vector transition polarisability, beta = A^1/(sqrt(2) <b||sigma||a>).
  // The spin matrix element expresses the Wigner-Eckart factor of the rank-1
  // amplitude, so no radial overlap enters it; see CI::sigma_rme
  if (K == 1 && E1_s) {
    const auto sigma = CI::sigma_rme(*Psi_b, ib, *Psi_a, ia, ints.ci_basis);
    fmt::print("\nVector{}polarisability:\n", diagonal ? " " : " transition ");
    fmt::print("<B||sigma||A> = {:.6e}\n", sigma);
    if (std::abs(sigma) < 1.0e-12) {
      std::cout << "beta: not defined - the states have no spin-angular "
                   "structure in common\n";
    } else {
      fmt::print("beta = {:.6e} au\n", A_total / (std::sqrt(2.0) * sigma));
    }
  }

  // PNC amplitude: the static operator is the PNC interaction
  if (hs->name().substr(0, 3) == "pnc") {
    fmt::print("\nPNC amplitude:\n");
    fmt::print("E_pnc = A^{}_0 = {:.6e} {}\n", K, A_K0, hs->units());
  }
}

} // namespace Module
