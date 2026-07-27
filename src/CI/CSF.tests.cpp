#include "CSF.hpp"
#include "catch2/catch.hpp"
#include "fmt/format.hpp"
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
