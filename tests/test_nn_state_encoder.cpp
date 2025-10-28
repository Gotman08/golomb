#include "core/golomb.hpp"
#include "nn/state_encoder.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace golomb;
using namespace golomb::nn;

TEST_CASE("StateEncoder construction", "[nn][state_encoder]") {
  StateEncoder encoder(100, 8);

  REQUIRE(encoder.ub() == 100);
  REQUIRE(encoder.target_marks() == 8);

  // Default encoding should include all features
  // POSITIONS: 101 (0 to 100)
  // DISTANCES: 101 (0 to 100)
  // METADATA: 4
  size_t expected_size = 101 + 101 + 4;
  REQUIRE(encoder.encoding_size() == expected_size);
}

TEST_CASE("StateEncoder with custom encoding", "[nn][state_encoder]") {
  SECTION("Positions only") {
    StateEncoder encoder(100, 8, StateEncoder::POSITIONS);
    REQUIRE(encoder.encoding_size() == 101);
  }

  SECTION("Distances only") {
    StateEncoder encoder(100, 8, StateEncoder::DISTANCES);
    REQUIRE(encoder.encoding_size() == 101);
  }

  SECTION("Metadata only") {
    StateEncoder encoder(100, 8, StateEncoder::METADATA);
    REQUIRE(encoder.encoding_size() == 4);
  }

  SECTION("Positions and distances") {
    StateEncoder encoder(100, 8, StateEncoder::POSITIONS | StateEncoder::DISTANCES);
    REQUIRE(encoder.encoding_size() == 101 + 101);
  }
}

TEST_CASE("StateEncoder encode positions", "[nn][state_encoder]") {
  int ub = 10;
  StateEncoder encoder(ub, 5, StateEncoder::POSITIONS);

  RuleState state(ub);
  state.marks = {0, 2, 5};

  auto encoded = encoder.encode(state);

  REQUIRE(encoded.size() == 11); // positions 0 to 10

  // Check that positions 0, 2, 5 are marked
  REQUIRE(encoded(0) == 1.0);
  REQUIRE(encoded(2) == 1.0);
  REQUIRE(encoded(5) == 1.0);

  // Check that other positions are not marked
  REQUIRE(encoded(1) == 0.0);
  REQUIRE(encoded(3) == 0.0);
  REQUIRE(encoded(4) == 0.0);
  REQUIRE(encoded(6) == 0.0);
}

TEST_CASE("StateEncoder encode distances", "[nn][state_encoder]") {
  int ub = 10;
  StateEncoder encoder(ub, 5, StateEncoder::DISTANCES);

  RuleState state(ub);
  state.marks = {0, 2, 5}; // Distances: 2 (0-2), 5 (0-5), 3 (2-5)

  auto encoded = encoder.encode(state);

  REQUIRE(encoded.size() == 11); // distances 0 to 10

  // Check that distances 2, 3, 5 are marked
  REQUIRE(encoded(2) == 1.0);
  REQUIRE(encoded(3) == 1.0);
  REQUIRE(encoded(5) == 1.0);

  // Check that other distances are not marked
  REQUIRE(encoded(0) == 0.0);
  REQUIRE(encoded(1) == 0.0);
  REQUIRE(encoded(4) == 0.0);
  REQUIRE(encoded(6) == 0.0);
}

TEST_CASE("StateEncoder encode metadata", "[nn][state_encoder]") {
  int ub = 100;
  int target_marks = 8;
  StateEncoder encoder(ub, target_marks, StateEncoder::METADATA);

  SECTION("Empty state") {
    RuleState state(ub);
    state.marks = {0}; // Just the origin

    auto encoded = encoder.encode(state);

    REQUIRE(encoded.size() == 4);

    // Feature 0: num_marks / target_marks = 1 / 8
    REQUIRE_THAT(encoded(0), Catch::Matchers::WithinAbs(1.0 / 8.0, 1e-6));

    // Feature 1: current_length / ub = 0 / 100
    REQUIRE(encoded(1) == 0.0);

    // Feature 2: progress = 1 / 8
    REQUIRE_THAT(encoded(2), Catch::Matchers::WithinAbs(1.0 / 8.0, 1e-6));

    // Feature 3: density = 0 (length is 0)
    REQUIRE(encoded(3) == 0.0);
  }

  SECTION("Partial state") {
    RuleState state(ub);
    state.marks = {0, 10, 25, 40}; // 4 marks, length 40

    auto encoded = encoder.encode(state);

    // Feature 0: num_marks / target_marks = 4 / 8
    REQUIRE_THAT(encoded(0), Catch::Matchers::WithinAbs(0.5, 1e-6));

    // Feature 1: current_length / ub = 40 / 100
    REQUIRE_THAT(encoded(1), Catch::Matchers::WithinAbs(0.4, 1e-6));

    // Feature 2: progress = 4 / 8
    REQUIRE_THAT(encoded(2), Catch::Matchers::WithinAbs(0.5, 1e-6));

    // Feature 3: density = 4 / 40 = 0.1
    REQUIRE_THAT(encoded(3), Catch::Matchers::WithinAbs(0.1, 1e-6));
  }

  SECTION("Complete state") {
    RuleState state(ub);
    state.marks = {0, 10, 20, 30, 40, 50, 60, 80}; // 8 marks, length 80

    auto encoded = encoder.encode(state);

    // Feature 0: num_marks / target_marks = 8 / 8
    REQUIRE_THAT(encoded(0), Catch::Matchers::WithinAbs(1.0, 1e-6));

    // Feature 1: current_length / ub = 80 / 100
    REQUIRE_THAT(encoded(1), Catch::Matchers::WithinAbs(0.8, 1e-6));

    // Feature 2: progress = 8 / 8
    REQUIRE_THAT(encoded(2), Catch::Matchers::WithinAbs(1.0, 1e-6));

    // Feature 3: density = 8 / 80 = 0.1
    REQUIRE_THAT(encoded(3), Catch::Matchers::WithinAbs(0.1, 1e-6));
  }
}

TEST_CASE("StateEncoder full encoding", "[nn][state_encoder]") {
  int ub = 10;
  int target_marks = 4;
  StateEncoder encoder(ub, target_marks); // Default: all features

  RuleState state(ub);
  state.marks = {0, 2, 5};

  auto encoded = encoder.encode(state);

  // Total size: positions (11) + distances (11) + metadata (4)
  REQUIRE(encoded.size() == 26);

  // Verify positions section (first 11 elements)
  REQUIRE(encoded(0) == 1.0); // Position 0
  REQUIRE(encoded(2) == 1.0); // Position 2
  REQUIRE(encoded(5) == 1.0); // Position 5

  // Verify distances section (next 11 elements, offset 11)
  REQUIRE(encoded(11 + 2) == 1.0); // Distance 2
  REQUIRE(encoded(11 + 3) == 1.0); // Distance 3
  REQUIRE(encoded(11 + 5) == 1.0); // Distance 5

  // Verify metadata section (last 4 elements, offset 22)
  // num_marks = 3, target_marks = 4, length = 5, ub = 10
  double expected_progress = 3.0 / 4.0;     // 0.75
  double expected_length_norm = 5.0 / 10.0; // 0.5
  double expected_density = 3.0 / 5.0;      // 0.6

  REQUIRE_THAT(encoded(22), Catch::Matchers::WithinAbs(expected_progress, 1e-6));
  REQUIRE_THAT(encoded(23), Catch::Matchers::WithinAbs(expected_length_norm, 1e-6));
  REQUIRE_THAT(encoded(24), Catch::Matchers::WithinAbs(expected_progress, 1e-6));
  REQUIRE_THAT(encoded(25), Catch::Matchers::WithinAbs(expected_density, 1e-6));
}

TEST_CASE("StateEncoder consistency", "[nn][state_encoder]") {
  StateEncoder encoder(50, 6);

  // Encode the same state twice
  RuleState state(50);
  state.marks = {0, 5, 12, 23};

  auto encoded1 = encoder.encode(state);
  auto encoded2 = encoder.encode(state);

  // Results should be identical
  REQUIRE(encoded1.size() == encoded2.size());
  for (size_t i = 0; i < encoded1.size(); ++i) {
    REQUIRE(encoded1(i) == encoded2(i));
  }
}

TEST_CASE("StateEncoder different states", "[nn][state_encoder]") {
  StateEncoder encoder(50, 6);

  RuleState state1(50);
  state1.marks = {0, 5, 12};

  RuleState state2(50);
  state2.marks = {0, 5, 15};

  auto encoded1 = encoder.encode(state1);
  auto encoded2 = encoder.encode(state2);

  // Encodings should be different
  bool different = false;
  for (size_t i = 0; i < encoded1.size(); ++i) {
    if (encoded1(i) != encoded2(i)) {
      different = true;
      break;
    }
  }
  REQUIRE(different);
}
