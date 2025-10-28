#include "nn/activations.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

using namespace golomb::nn;

TEST_CASE("ReLU activation", "[nn][activations]") {
  SECTION("Forward pass") {
    Tensor x{-2.0, -1.0, 0.0, 1.0, 2.0};
    auto y = relu(x);

    REQUIRE(y(0) == 0.0);
    REQUIRE(y(1) == 0.0);
    REQUIRE(y(2) == 0.0);
    REQUIRE(y(3) == 1.0);
    REQUIRE(y(4) == 2.0);
  }

  SECTION("Backward pass") {
    Tensor x{-2.0, -1.0, 0.0, 1.0, 2.0};
    Tensor grad_out{1.0, 1.0, 1.0, 1.0, 1.0};
    auto grad_in = relu_backward(grad_out, x);

    REQUIRE(grad_in(0) == 0.0); // x < 0
    REQUIRE(grad_in(1) == 0.0); // x < 0
    REQUIRE(grad_in(2) == 0.0); // x == 0
    REQUIRE(grad_in(3) == 1.0); // x > 0
    REQUIRE(grad_in(4) == 1.0); // x > 0
  }
}

TEST_CASE("Leaky ReLU activation", "[nn][activations]") {
  SECTION("Forward pass") {
    Tensor x{-2.0, -1.0, 0.0, 1.0, 2.0};
    auto y = leaky_relu(x, 0.1);

    REQUIRE_THAT(y(0), Catch::Matchers::WithinAbs(-0.2, 1e-6));
    REQUIRE_THAT(y(1), Catch::Matchers::WithinAbs(-0.1, 1e-6));
    REQUIRE(y(2) == 0.0);
    REQUIRE(y(3) == 1.0);
    REQUIRE(y(4) == 2.0);
  }

  SECTION("Backward pass") {
    Tensor x{-2.0, -1.0, 0.0, 1.0, 2.0};
    Tensor grad_out{1.0, 1.0, 1.0, 1.0, 1.0};
    auto grad_in = leaky_relu_backward(grad_out, x, 0.1);

    REQUIRE_THAT(grad_in(0), Catch::Matchers::WithinAbs(0.1, 1e-6));
    REQUIRE_THAT(grad_in(1), Catch::Matchers::WithinAbs(0.1, 1e-6));
    REQUIRE(grad_in(2) == 1.0);
    REQUIRE(grad_in(3) == 1.0);
    REQUIRE(grad_in(4) == 1.0);
  }
}

TEST_CASE("Tanh activation", "[nn][activations]") {
  SECTION("Forward pass") {
    Tensor x{-1.0, 0.0, 1.0};
    auto y = tanh_activation(x);

    REQUIRE_THAT(y(0), Catch::Matchers::WithinAbs(std::tanh(-1.0), 1e-6));
    REQUIRE_THAT(y(1), Catch::Matchers::WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(y(2), Catch::Matchers::WithinAbs(std::tanh(1.0), 1e-6));
  }

  SECTION("Backward pass") {
    Tensor x{0.0};
    auto y = tanh_activation(x);
    Tensor grad_out{1.0};
    auto grad_in = tanh_backward(grad_out, y);

    // At x=0, tanh(0)=0, so derivative is 1 - 0^2 = 1
    REQUIRE_THAT(grad_in(0), Catch::Matchers::WithinAbs(1.0, 1e-6));
  }

  SECTION("Backward pass gradient check") {
    Tensor x{0.5};
    auto y = tanh_activation(x);
    Tensor grad_out{1.0};
    auto grad_in = tanh_backward(grad_out, y);

    // tanh(0.5) ≈ 0.4621, derivative = 1 - 0.4621^2 ≈ 0.7864
    double tanh_val = std::tanh(0.5);
    double expected_grad = 1.0 - tanh_val * tanh_val;
    REQUIRE_THAT(grad_in(0), Catch::Matchers::WithinAbs(expected_grad, 1e-6));
  }
}

TEST_CASE("Sigmoid activation", "[nn][activations]") {
  SECTION("Forward pass") {
    Tensor x{-10.0, 0.0, 10.0};
    auto y = sigmoid(x);

    // sigmoid(-10) ≈ 0, sigmoid(0) = 0.5, sigmoid(10) ≈ 1
    REQUIRE_THAT(y(0), Catch::Matchers::WithinAbs(0.0, 1e-4));
    REQUIRE_THAT(y(1), Catch::Matchers::WithinAbs(0.5, 1e-6));
    REQUIRE_THAT(y(2), Catch::Matchers::WithinAbs(1.0, 1e-4));
  }

  SECTION("Backward pass") {
    Tensor x{0.0};
    auto y = sigmoid(x);
    Tensor grad_out{1.0};
    auto grad_in = sigmoid_backward(grad_out, y);

    // At x=0, sigmoid(0)=0.5, derivative = 0.5 * 0.5 = 0.25
    REQUIRE_THAT(grad_in(0), Catch::Matchers::WithinAbs(0.25, 1e-6));
  }
}

TEST_CASE("Softmax activation 1D", "[nn][activations]") {
  SECTION("Forward pass") {
    Tensor x{1.0, 2.0, 3.0};
    auto y = softmax(x);

    // Check sum to 1
    double sum = y(0) + y(1) + y(2);
    REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(1.0, 1e-6));

    // Check all positive
    REQUIRE(y(0) > 0.0);
    REQUIRE(y(1) > 0.0);
    REQUIRE(y(2) > 0.0);

    // Check monotonic (larger input -> larger output)
    REQUIRE(y(0) < y(1));
    REQUIRE(y(1) < y(2));
  }

  SECTION("Forward pass numerical stability") {
    Tensor x{1000.0, 1001.0, 1002.0};
    auto y = softmax(x);

    // Should not overflow
    REQUIRE(std::isfinite(y(0)));
    REQUIRE(std::isfinite(y(1)));
    REQUIRE(std::isfinite(y(2)));

    double sum = y(0) + y(1) + y(2);
    REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(1.0, 1e-6));
  }

  SECTION("Backward pass") {
    Tensor x{1.0, 2.0, 3.0};
    auto y = softmax(x);
    Tensor grad_out{1.0, 0.0, 0.0};
    auto grad_in = softmax_backward(grad_out, y);

    // Gradient should sum to 0 for softmax
    double grad_sum = grad_in(0) + grad_in(1) + grad_in(2);
    REQUIRE_THAT(grad_sum, Catch::Matchers::WithinAbs(0.0, 1e-6));
  }
}

TEST_CASE("Softmax activation 2D", "[nn][activations]") {
  SECTION("Forward pass row-wise") {
    Tensor x(2, 3);
    x(0, 0) = 1.0;
    x(0, 1) = 2.0;
    x(0, 2) = 3.0;
    x(1, 0) = 3.0;
    x(1, 1) = 2.0;
    x(1, 2) = 1.0;

    auto y = softmax(x);

    // Check each row sums to 1
    double row0_sum = y(0, 0) + y(0, 1) + y(0, 2);
    double row1_sum = y(1, 0) + y(1, 1) + y(1, 2);
    REQUIRE_THAT(row0_sum, Catch::Matchers::WithinAbs(1.0, 1e-6));
    REQUIRE_THAT(row1_sum, Catch::Matchers::WithinAbs(1.0, 1e-6));

    // Row 0: increasing values -> increasing probabilities
    REQUIRE(y(0, 0) < y(0, 1));
    REQUIRE(y(0, 1) < y(0, 2));

    // Row 1: decreasing values -> decreasing probabilities
    REQUIRE(y(1, 0) > y(1, 1));
    REQUIRE(y(1, 1) > y(1, 2));
  }
}

TEST_CASE("Log-softmax activation 1D", "[nn][activations]") {
  SECTION("Forward pass") {
    Tensor x{1.0, 2.0, 3.0};
    auto y = log_softmax(x);

    // Log-softmax values should be negative (since softmax < 1)
    REQUIRE(y(0) < 0.0);
    REQUIRE(y(1) < 0.0);
    REQUIRE(y(2) < 0.0);

    // exp(log_softmax) should equal softmax
    auto softmax_y = softmax(x);
    REQUIRE_THAT(std::exp(y(0)), Catch::Matchers::WithinAbs(softmax_y(0), 1e-6));
    REQUIRE_THAT(std::exp(y(1)), Catch::Matchers::WithinAbs(softmax_y(1), 1e-6));
    REQUIRE_THAT(std::exp(y(2)), Catch::Matchers::WithinAbs(softmax_y(2), 1e-6));
  }

  SECTION("Backward pass") {
    Tensor x{1.0, 2.0, 3.0};
    auto y = log_softmax(x);
    Tensor grad_out{1.0, 0.0, 0.0};
    auto grad_in = log_softmax_backward(grad_out, y);

    // Check gradient is reasonable
    REQUIRE(std::isfinite(grad_in(0)));
    REQUIRE(std::isfinite(grad_in(1)));
    REQUIRE(std::isfinite(grad_in(2)));
  }
}

TEST_CASE("Activation function gradients numerical check", "[nn][activations]") {
  SECTION("ReLU gradient check") {
    // Numerical gradient check using finite differences
    double epsilon = 1e-5;
    Tensor x{0.5};

    // Forward pass
    auto y1 = relu(x);

    // Perturbed forward pass
    Tensor x_perturbed{0.5 + epsilon};
    auto y2 = relu(x_perturbed);

    // Numerical gradient
    double numerical_grad = (y2(0) - y1(0)) / epsilon;

    // Analytical gradient
    Tensor grad_out{1.0};
    auto analytical_grad = relu_backward(grad_out, x);

    REQUIRE_THAT(analytical_grad(0), Catch::Matchers::WithinAbs(numerical_grad, 1e-4));
  }

  SECTION("Tanh gradient check") {
    double epsilon = 1e-5;
    Tensor x{0.5};

    auto y1 = tanh_activation(x);
    Tensor x_perturbed{0.5 + epsilon};
    auto y2 = tanh_activation(x_perturbed);

    double numerical_grad = (y2(0) - y1(0)) / epsilon;

    Tensor grad_out{1.0};
    auto y = tanh_activation(x);
    auto analytical_grad = tanh_backward(grad_out, y);

    REQUIRE_THAT(analytical_grad(0), Catch::Matchers::WithinAbs(numerical_grad, 1e-4));
  }
}
