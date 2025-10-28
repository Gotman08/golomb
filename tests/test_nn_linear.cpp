#include "nn/linear.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace golomb::nn;

TEST_CASE("Linear layer construction", "[nn][linear]") {
  SECTION("Basic construction") {
    Linear layer(10, 5);

    REQUIRE(layer.weight().shape()[0] == 5);
    REQUIRE(layer.weight().shape()[1] == 10);
    REQUIRE(layer.has_bias());
    REQUIRE(layer.bias().shape()[0] == 5);
    REQUIRE(layer.num_parameters() == 10 * 5 + 5);
  }

  SECTION("Without bias") {
    Linear layer(10, 5, false);

    REQUIRE(!layer.has_bias());
    REQUIRE(layer.num_parameters() == 10 * 5);
  }
}

TEST_CASE("Linear layer forward pass 1D", "[nn][linear]") {
  Linear layer(3, 2);

  // Set known weights and bias for testing
  layer.weight()(0, 0) = 1.0;
  layer.weight()(0, 1) = 2.0;
  layer.weight()(0, 2) = 3.0;
  layer.weight()(1, 0) = 4.0;
  layer.weight()(1, 1) = 5.0;
  layer.weight()(1, 2) = 6.0;
  layer.bias()(0) = 0.1;
  layer.bias()(1) = 0.2;

  Tensor input{1.0, 2.0, 3.0};
  auto output = layer.forward(input);

  REQUIRE(output.ndim() == 1);
  REQUIRE(output.shape()[0] == 2);

  // output[0] = 1*1 + 2*2 + 3*3 + 0.1 = 1 + 4 + 9 + 0.1 = 14.1
  // output[1] = 4*1 + 5*2 + 6*3 + 0.2 = 4 + 10 + 18 + 0.2 = 32.2
  REQUIRE_THAT(output(0), Catch::Matchers::WithinAbs(14.1, 1e-6));
  REQUIRE_THAT(output(1), Catch::Matchers::WithinAbs(32.2, 1e-6));
}

TEST_CASE("Linear layer forward pass 2D batch", "[nn][linear]") {
  Linear layer(3, 2);

  // Set known weights
  layer.weight()(0, 0) = 1.0;
  layer.weight()(0, 1) = 2.0;
  layer.weight()(0, 2) = 3.0;
  layer.weight()(1, 0) = 4.0;
  layer.weight()(1, 1) = 5.0;
  layer.weight()(1, 2) = 6.0;
  layer.bias().zeros();

  Tensor input(2, 3);
  input(0, 0) = 1.0;
  input(0, 1) = 2.0;
  input(0, 2) = 3.0;
  input(1, 0) = 4.0;
  input(1, 1) = 5.0;
  input(1, 2) = 6.0;

  auto output = layer.forward(input);

  REQUIRE(output.ndim() == 2);
  REQUIRE(output.shape()[0] == 2);
  REQUIRE(output.shape()[1] == 2);

  // Sample 0: [1,2,3]
  // output[0,0] = 1*1 + 2*2 + 3*3 = 14
  // output[0,1] = 1*4 + 2*5 + 3*6 = 32

  // Sample 1: [4,5,6]
  // output[1,0] = 4*1 + 5*2 + 6*3 = 32
  // output[1,1] = 4*4 + 5*5 + 6*6 = 77

  REQUIRE_THAT(output(0, 0), Catch::Matchers::WithinAbs(14.0, 1e-6));
  REQUIRE_THAT(output(0, 1), Catch::Matchers::WithinAbs(32.0, 1e-6));
  REQUIRE_THAT(output(1, 0), Catch::Matchers::WithinAbs(32.0, 1e-6));
  REQUIRE_THAT(output(1, 1), Catch::Matchers::WithinAbs(77.0, 1e-6));
}

TEST_CASE("Linear layer backward pass 1D", "[nn][linear]") {
  Linear layer(3, 2);
  layer.zero_grad();

  // Simple weights for easy verification
  layer.weight().fill(1.0);
  layer.bias().zeros();

  Tensor input{1.0, 2.0, 3.0};
  auto output = layer.forward(input);

  Tensor grad_output{1.0, 1.0};
  auto grad_input = layer.backward(grad_output);

  REQUIRE(grad_input.ndim() == 1);
  REQUIRE(grad_input.shape()[0] == 3);

  // grad_input[j] = sum_i(grad_output[i] * weight[i,j])
  // All weights are 1.0, so grad_input = [2, 2, 2]
  REQUIRE_THAT(grad_input(0), Catch::Matchers::WithinAbs(2.0, 1e-6));
  REQUIRE_THAT(grad_input(1), Catch::Matchers::WithinAbs(2.0, 1e-6));
  REQUIRE_THAT(grad_input(2), Catch::Matchers::WithinAbs(2.0, 1e-6));

  // grad_weight[i,j] = grad_output[i] * input[j]
  auto grads = layer.gradients();
  auto* grad_w = grads[0];
  REQUIRE_THAT((*grad_w)(0, 0), Catch::Matchers::WithinAbs(1.0, 1e-6));
  REQUIRE_THAT((*grad_w)(0, 1), Catch::Matchers::WithinAbs(2.0, 1e-6));
  REQUIRE_THAT((*grad_w)(0, 2), Catch::Matchers::WithinAbs(3.0, 1e-6));

  // grad_bias[i] = grad_output[i]
  auto* grad_b = grads[1];
  REQUIRE_THAT((*grad_b)(0), Catch::Matchers::WithinAbs(1.0, 1e-6));
  REQUIRE_THAT((*grad_b)(1), Catch::Matchers::WithinAbs(1.0, 1e-6));
}

TEST_CASE("Linear layer backward pass 2D batch", "[nn][linear]") {
  Linear layer(3, 2);
  layer.zero_grad();

  layer.weight().fill(1.0);
  layer.bias().zeros();

  Tensor input(2, 3);
  input(0, 0) = 1.0;
  input(0, 1) = 2.0;
  input(0, 2) = 3.0;
  input(1, 0) = 4.0;
  input(1, 1) = 5.0;
  input(1, 2) = 6.0;

  auto output = layer.forward(input);

  Tensor grad_output(2, 2);
  grad_output.fill(1.0);

  auto grad_input = layer.backward(grad_output);

  REQUIRE(grad_input.shape()[0] == 2);
  REQUIRE(grad_input.shape()[1] == 3);

  // Each element gets gradient of 2.0 (sum of 2 outputs with weight 1.0)
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      REQUIRE_THAT(grad_input(i, j), Catch::Matchers::WithinAbs(2.0, 1e-6));
    }
  }

  // grad_weight accumulates over batch
  auto grads = layer.gradients();
  auto* grad_w = grads[0];
  // grad_w[i,j] = sum over batch of (grad_output[b,i] * input[b,j])
  // For example, grad_w[0,0] = 1*1 + 1*4 = 5
  REQUIRE_THAT((*grad_w)(0, 0), Catch::Matchers::WithinAbs(5.0, 1e-6));
  REQUIRE_THAT((*grad_w)(0, 1), Catch::Matchers::WithinAbs(7.0, 1e-6));
  REQUIRE_THAT((*grad_w)(0, 2), Catch::Matchers::WithinAbs(9.0, 1e-6));

  // grad_bias sums over batch
  auto* grad_b = grads[1];
  REQUIRE_THAT((*grad_b)(0), Catch::Matchers::WithinAbs(2.0, 1e-6));
  REQUIRE_THAT((*grad_b)(1), Catch::Matchers::WithinAbs(2.0, 1e-6));
}

TEST_CASE("Linear layer gradient checking", "[nn][linear]") {
  // Numerical gradient checking to verify backprop correctness
  Linear layer(5, 3);
  layer.init_he();
  layer.zero_grad();

  Tensor input{1.0, 2.0, 3.0, 4.0, 5.0};

  // Forward pass
  auto output = layer.forward(input);

  // Assume simple loss: L = sum(output^2) / 2
  Tensor grad_output(3);
  for (size_t i = 0; i < 3; ++i) {
    grad_output(i) = output(i); // dL/dy = y
  }

  // Analytical gradient
  auto grad_input_analytical = layer.backward(grad_output);

  // Numerical gradient for input
  double epsilon = 1e-5;
  Tensor grad_input_numerical(5);

  for (size_t i = 0; i < 5; ++i) {
    // Compute loss with input[i] + epsilon
    Tensor input_plus = input.copy();
    input_plus(i) += epsilon;
    auto output_plus = layer.forward(input_plus);
    double loss_plus = 0.0;
    for (size_t j = 0; j < 3; ++j) {
      loss_plus += output_plus(j) * output_plus(j) / 2.0;
    }

    // Compute loss with input[i] - epsilon
    Tensor input_minus = input.copy();
    input_minus(i) -= epsilon;
    auto output_minus = layer.forward(input_minus);
    double loss_minus = 0.0;
    for (size_t j = 0; j < 3; ++j) {
      loss_minus += output_minus(j) * output_minus(j) / 2.0;
    }

    // Numerical gradient
    grad_input_numerical(i) = (loss_plus - loss_minus) / (2.0 * epsilon);
  }

  // Compare analytical and numerical gradients
  for (size_t i = 0; i < 5; ++i) {
    REQUIRE_THAT(grad_input_analytical(i),
                 Catch::Matchers::WithinAbs(grad_input_numerical(i), 1e-4));
  }
}

TEST_CASE("Linear layer zero_grad", "[nn][linear]") {
  Linear layer(3, 2);

  Tensor input{1.0, 2.0, 3.0};
  layer.forward(input);

  Tensor grad_output{1.0, 1.0};
  layer.backward(grad_output);

  // Gradients should be non-zero after backward
  {
    auto grads = layer.gradients();
    auto* grad_w = grads[0];
    REQUIRE((*grad_w)(0, 0) != 0.0);
  }

  // Zero gradients
  layer.zero_grad();

  // All gradients should be zero
  {
    auto grads = layer.gradients();
    auto* grad_w = grads[0];
    for (size_t i = 0; i < 2; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        REQUIRE((*grad_w)(i, j) == 0.0);
      }
    }
  }
}

TEST_CASE("Linear layer initialization", "[nn][linear]") {
  SECTION("Xavier initialization") {
    Linear layer(100, 50);
    layer.init_xavier();

    // Check that weights are in reasonable range
    double max_weight = 0.0;
    for (size_t i = 0; i < 50; ++i) {
      for (size_t j = 0; j < 100; ++j) {
        max_weight = std::max(max_weight, std::abs(layer.weight()(i, j)));
      }
    }

    // Xavier limit = sqrt(6 / (100 + 50)) ≈ 0.2
    REQUIRE(max_weight > 0.0);
    REQUIRE(max_weight < 1.0);
  }

  SECTION("He initialization") {
    Linear layer(100, 50);
    layer.init_he();

    // Check that weights have reasonable variance
    double mean = layer.weight().mean();
    REQUIRE_THAT(mean, Catch::Matchers::WithinAbs(0.0, 0.1));
  }
}
