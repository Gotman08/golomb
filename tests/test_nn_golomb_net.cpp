#include "core/golomb.hpp"
#include "nn/golomb_net.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace golomb;
using namespace golomb::nn;

TEST_CASE("GolombNet construction", "[nn][golomb_net]") {
  int ub = 50;
  int target_marks = 6;
  StateEncoder encoder(ub, target_marks);

  // Create network with small hidden layers for testing
  GolombNet net(encoder, ub, 32, 32);

  REQUIRE(net.ub() == ub);
  REQUIRE(net.encoder().ub() == ub);
  REQUIRE(net.num_parameters() > 0);
}

TEST_CASE("GolombNet forward pass dimensions", "[nn][golomb_net]") {
  int ub = 50;
  int target_marks = 6;
  StateEncoder encoder(ub, target_marks);
  GolombNet net(encoder, ub, 32, 32);

  RuleState state(ub);
  state.marks = {0, 5, 12};

  Tensor policy;
  double value;

  net.forward(state, policy, value);

  // Policy should have size ub+1 (one probability for each position)
  REQUIRE(policy.size() == static_cast<size_t>(ub + 1));

  // Policy should sum to 1 (it's a probability distribution)
  double policy_sum = policy.sum();
  REQUIRE_THAT(policy_sum, Catch::Matchers::WithinAbs(1.0, 1e-5));

  // All policy values should be in [0, 1]
  for (size_t i = 0; i < policy.size(); ++i) {
    REQUIRE(policy(i) >= 0.0);
    REQUIRE(policy(i) <= 1.0);
  }

  // Value should be in range [-1, 1] due to tanh
  REQUIRE(value >= -1.0);
  REQUIRE(value <= 1.0);
}

TEST_CASE("GolombNet different states produce different outputs", "[nn][golomb_net]") {
  int ub = 50;
  int target_marks = 6;
  StateEncoder encoder(ub, target_marks);
  GolombNet net(encoder, ub, 64, 64);

  RuleState state1(ub);
  state1.marks = {0, 5, 12};

  RuleState state2(ub);
  state2.marks = {0, 5, 15}; // Different state

  Tensor policy1, policy2;
  double value1, value2;

  net.forward(state1, policy1, value1);
  net.forward(state2, policy2, value2);

  // Policies should be different for different states
  bool policies_different = false;
  for (size_t i = 0; i < policy1.size(); ++i) {
    if (std::abs(policy1(i) - policy2(i)) > 1e-6) {
      policies_different = true;
      break;
    }
  }
  REQUIRE(policies_different);
}

TEST_CASE("GolombNet same state produces same output", "[nn][golomb_net]") {
  int ub = 50;
  int target_marks = 6;
  StateEncoder encoder(ub, target_marks);
  GolombNet net(encoder, ub, 64, 64);

  RuleState state(ub);
  state.marks = {0, 5, 12, 23};

  Tensor policy1, policy2;
  double value1, value2;

  net.forward(state, policy1, value1);
  net.forward(state, policy2, value2);

  // Should produce identical outputs
  REQUIRE(value1 == value2);

  for (size_t i = 0; i < policy1.size(); ++i) {
    REQUIRE(policy1(i) == policy2(i));
  }
}

TEST_CASE("GolombNet backward pass runs without error", "[nn][golomb_net]") {
  int ub = 20;
  int target_marks = 4;
  StateEncoder encoder(ub, target_marks);
  GolombNet net(encoder, ub, 32, 32);

  RuleState state(ub);
  state.marks = {0, 5, 12};

  // Forward pass
  Tensor policy;
  double value;
  net.forward(state, policy, value);

  // Create dummy gradients
  Tensor grad_policy(ub + 1);
  grad_policy.zeros();
  grad_policy(5) = 1.0; // Gradient for position 5

  double grad_value = 1.0;

  // Backward pass should run without error
  net.zero_grad();
  REQUIRE_NOTHROW(net.backward(grad_policy, grad_value));

  // Gradients should be non-zero after backward
  auto grads = net.gradients();
  bool has_nonzero_grad = false;
  for (auto* grad : grads) {
    if (grad->sum() != 0.0) {
      has_nonzero_grad = true;
      break;
    }
  }
  REQUIRE(has_nonzero_grad);
}

TEST_CASE("GolombNet zero_grad clears gradients", "[nn][golomb_net]") {
  int ub = 20;
  int target_marks = 4;
  StateEncoder encoder(ub, target_marks);
  GolombNet net(encoder, ub, 32, 32);

  RuleState state(ub);
  state.marks = {0, 5};

  // Forward and backward to accumulate gradients
  Tensor policy;
  double value;
  net.forward(state, policy, value);

  Tensor grad_policy(ub + 1);
  grad_policy.fill(0.1);
  net.backward(grad_policy, 0.5);

  // Zero gradients
  net.zero_grad();

  // All gradients should be zero
  auto grads = net.gradients();
  for (auto* grad : grads) {
    REQUIRE(grad->sum() == 0.0);
  }
}

TEST_CASE("GolombNet parameter count", "[nn][golomb_net]") {
  int ub = 50;
  int target_marks = 6;
  StateEncoder encoder(ub, target_marks);

  size_t h1 = 64;
  size_t h2 = 64;
  GolombNet net(encoder, ub, h1, h2);

  size_t input_size = encoder.encoding_size();
  size_t output_policy_size = static_cast<size_t>(ub + 1);

  // Expected parameters:
  // hidden1: (input_size * h1) + h1
  // hidden2: (h1 * h2) + h2
  // policy_head: (h2 * output_policy_size) + output_policy_size
  // value_head: (h2 * 1) + 1

  size_t expected = (input_size * h1 + h1) + (h1 * h2 + h2) +
                    (h2 * output_policy_size + output_policy_size) + (h2 * 1 + 1);

  REQUIRE(net.num_parameters() == expected);
}

TEST_CASE("GolombNet initialization methods", "[nn][golomb_net]") {
  int ub = 50;
  int target_marks = 6;
  StateEncoder encoder(ub, target_marks);

  SECTION("He initialization") {
    GolombNet net(encoder, ub, 32, 32);
    net.init_he();

    // Check that parameters are non-zero and reasonable
    auto params = net.parameters();
    bool has_nonzero = false;
    for (auto* param : params) {
      if (param->sum() != 0.0) {
        has_nonzero = true;
        break;
      }
    }
    REQUIRE(has_nonzero);
  }

  SECTION("Xavier initialization") {
    GolombNet net(encoder, ub, 32, 32);
    net.init_xavier();

    auto params = net.parameters();
    bool has_nonzero = false;
    for (auto* param : params) {
      if (param->sum() != 0.0) {
        has_nonzero = true;
        break;
      }
    }
    REQUIRE(has_nonzero);
  }
}

TEST_CASE("GolombNet forward_encoded matches forward", "[nn][golomb_net]") {
  int ub = 30;
  int target_marks = 5;
  StateEncoder encoder(ub, target_marks);
  GolombNet net(encoder, ub, 32, 32);

  RuleState state(ub);
  state.marks = {0, 5, 12, 20};

  // Forward via state
  Tensor policy1;
  double value1;
  net.forward(state, policy1, value1);

  // Forward via encoded tensor
  Tensor encoded = encoder.encode(state);
  Tensor policy2;
  double value2;
  net.forward_encoded(encoded, policy2, value2);

  // Results should be identical
  REQUIRE(value1 == value2);
  for (size_t i = 0; i < policy1.size(); ++i) {
    REQUIRE(policy1(i) == policy2(i));
  }
}

TEST_CASE("GolombNet gradient accumulation", "[nn][golomb_net]") {
  int ub = 20;
  int target_marks = 4;
  StateEncoder encoder(ub, target_marks);
  GolombNet net(encoder, ub, 16, 16);

  RuleState state(ub);
  state.marks = {0, 5};

  // First backward pass
  Tensor policy;
  double value;
  net.forward(state, policy, value);

  Tensor grad_policy(ub + 1);
  grad_policy.fill(0.1);

  net.zero_grad();
  net.backward(grad_policy, 0.5);

  // Get gradient sum after first backward
  auto grads = net.gradients();
  double grad_sum1 = 0.0;
  for (auto* grad : grads) {
    grad_sum1 += std::abs(grad->sum());
  }

  // Second backward pass (without zero_grad)
  net.forward(state, policy, value);
  net.backward(grad_policy, 0.5);

  // Get gradient sum after second backward
  double grad_sum2 = 0.0;
  for (auto* grad : grads) {
    grad_sum2 += std::abs(grad->sum());
  }

  // Gradients should have accumulated (roughly doubled)
  REQUIRE(grad_sum2 > grad_sum1);
  REQUIRE_THAT(grad_sum2, Catch::Matchers::WithinAbs(2.0 * grad_sum1, 0.1 * grad_sum1));
}
