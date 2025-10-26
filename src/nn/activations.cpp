#include "nn/activations.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace golomb {
namespace nn {

// ============================================================================
// ReLU
// ============================================================================

Tensor relu(const Tensor& x) {
  Tensor output = x.copy();
  for (auto& val : output.data()) {
    val = std::max(0.0, val);
  }
  return output;
}

Tensor relu_backward(const Tensor& grad_output, const Tensor& input) {
  if (grad_output.shape() != input.shape()) {
    throw std::invalid_argument("relu_backward: shape mismatch");
  }

  Tensor grad_input = grad_output.copy();
  const auto& input_data = input.data();
  auto& grad_data = grad_input.data();

  for (size_t i = 0; i < grad_data.size(); ++i) {
    // Gradient is 0 if input <= 0, otherwise pass through
    if (input_data[i] <= 0.0) {
      grad_data[i] = 0.0;
    }
  }

  return grad_input;
}

// ============================================================================
// Leaky ReLU
// ============================================================================

Tensor leaky_relu(const Tensor& x, double alpha) {
  Tensor output = x.copy();
  for (auto& val : output.data()) {
    if (val < 0.0) {
      val *= alpha;
    }
  }
  return output;
}

Tensor leaky_relu_backward(const Tensor& grad_output, const Tensor& input, double alpha) {
  if (grad_output.shape() != input.shape()) {
    throw std::invalid_argument("leaky_relu_backward: shape mismatch");
  }

  Tensor grad_input = grad_output.copy();
  const auto& input_data = input.data();
  auto& grad_data = grad_input.data();

  for (size_t i = 0; i < grad_data.size(); ++i) {
    if (input_data[i] < 0.0) {
      grad_data[i] *= alpha;
    }
  }

  return grad_input;
}

// ============================================================================
// Tanh
// ============================================================================

Tensor tanh_activation(const Tensor& x) {
  Tensor output = x.copy();
  for (auto& val : output.data()) {
    val = std::tanh(val);
  }
  return output;
}

Tensor tanh_backward(const Tensor& grad_output, const Tensor& output) {
  if (grad_output.shape() != output.shape()) {
    throw std::invalid_argument("tanh_backward: shape mismatch");
  }

  Tensor grad_input = grad_output.copy();
  const auto& output_data = output.data();
  auto& grad_data = grad_input.data();

  for (size_t i = 0; i < grad_data.size(); ++i) {
    // d/dx tanh(x) = 1 - tanh²(x)
    double tanh_val = output_data[i];
    grad_data[i] *= (1.0 - tanh_val * tanh_val);
  }

  return grad_input;
}

// ============================================================================
// Sigmoid
// ============================================================================

Tensor sigmoid(const Tensor& x) {
  Tensor output = x.copy();
  for (auto& val : output.data()) {
    // Numerically stable sigmoid
    if (val >= 0.0) {
      double exp_neg = std::exp(-val);
      val = 1.0 / (1.0 + exp_neg);
    } else {
      double exp_pos = std::exp(val);
      val = exp_pos / (1.0 + exp_pos);
    }
  }
  return output;
}

Tensor sigmoid_backward(const Tensor& grad_output, const Tensor& output) {
  if (grad_output.shape() != output.shape()) {
    throw std::invalid_argument("sigmoid_backward: shape mismatch");
  }

  Tensor grad_input = grad_output.copy();
  const auto& output_data = output.data();
  auto& grad_data = grad_input.data();

  for (size_t i = 0; i < grad_data.size(); ++i) {
    // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    double sig_val = output_data[i];
    grad_data[i] *= sig_val * (1.0 - sig_val);
  }

  return grad_input;
}

// ============================================================================
// Softmax
// ============================================================================

Tensor softmax(const Tensor& x) {
  if (x.ndim() == 1) {
    // 1D: single softmax across all elements
    Tensor output = x.copy();
    auto& data = output.data();

    // Find max for numerical stability
    double max_val = *std::max_element(data.begin(), data.end());

    // Compute exp(x - max) and sum
    double sum = 0.0;
    for (auto& val : data) {
      val = std::exp(val - max_val);
      sum += val;
    }

    // Normalize
    for (auto& val : data) {
      val /= sum;
    }

    return output;

  } else if (x.ndim() == 2) {
    // 2D: row-wise softmax
    size_t rows = x.shape()[0];
    size_t cols = x.shape()[1];
    Tensor output = x.copy();

    for (size_t i = 0; i < rows; ++i) {
      // Find max in row
      double max_val = -std::numeric_limits<double>::infinity();
      for (size_t j = 0; j < cols; ++j) {
        max_val = std::max(max_val, output(i, j));
      }

      // Compute exp and sum
      double sum = 0.0;
      for (size_t j = 0; j < cols; ++j) {
        double val = std::exp(output(i, j) - max_val);
        output(i, j) = val;
        sum += val;
      }

      // Normalize
      for (size_t j = 0; j < cols; ++j) {
        output(i, j) /= sum;
      }
    }

    return output;

  } else {
    throw std::runtime_error("softmax: only 1D and 2D tensors supported");
  }
}

Tensor softmax_backward(const Tensor& grad_output, const Tensor& output) {
  if (grad_output.shape() != output.shape()) {
    throw std::invalid_argument("softmax_backward: shape mismatch");
  }

  if (output.ndim() == 1) {
    // 1D softmax Jacobian
    size_t n = output.size();
    Tensor grad_input(n);

    // For softmax, gradient is: grad_input[i] = sum_j (output[i] * (delta_ij - output[j]) * grad_output[j])
    // Simplified: grad_input[i] = output[i] * (grad_output[i] - sum_j(output[j] * grad_output[j]))
    double sum = 0.0;
    for (size_t j = 0; j < n; ++j) {
      sum += output(j) * grad_output(j);
    }

    for (size_t i = 0; i < n; ++i) {
      grad_input(i) = output(i) * (grad_output(i) - sum);
    }

    return grad_input;

  } else if (output.ndim() == 2) {
    // 2D: row-wise
    size_t rows = output.shape()[0];
    size_t cols = output.shape()[1];
    Tensor grad_input(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
      double sum = 0.0;
      for (size_t j = 0; j < cols; ++j) {
        sum += output(i, j) * grad_output(i, j);
      }

      for (size_t j = 0; j < cols; ++j) {
        grad_input(i, j) = output(i, j) * (grad_output(i, j) - sum);
      }
    }

    return grad_input;

  } else {
    throw std::runtime_error("softmax_backward: only 1D and 2D tensors supported");
  }
}

// ============================================================================
// Log-Softmax
// ============================================================================

Tensor log_softmax(const Tensor& x) {
  if (x.ndim() == 1) {
    // 1D version
    Tensor output = x.copy();
    auto& data = output.data();

    // Find max for numerical stability
    double max_val = *std::max_element(data.begin(), data.end());

    // Compute log-sum-exp
    double log_sum_exp = 0.0;
    for (const auto& val : data) {
      log_sum_exp += std::exp(val - max_val);
    }
    log_sum_exp = max_val + std::log(log_sum_exp);

    // Subtract log_sum_exp from each element
    for (auto& val : data) {
      val -= log_sum_exp;
    }

    return output;

  } else if (x.ndim() == 2) {
    // 2D: row-wise
    size_t rows = x.shape()[0];
    size_t cols = x.shape()[1];
    Tensor output = x.copy();

    for (size_t i = 0; i < rows; ++i) {
      // Find max in row
      double max_val = -std::numeric_limits<double>::infinity();
      for (size_t j = 0; j < cols; ++j) {
        max_val = std::max(max_val, output(i, j));
      }

      // Compute log-sum-exp
      double log_sum_exp = 0.0;
      for (size_t j = 0; j < cols; ++j) {
        log_sum_exp += std::exp(output(i, j) - max_val);
      }
      log_sum_exp = max_val + std::log(log_sum_exp);

      // Subtract from each element
      for (size_t j = 0; j < cols; ++j) {
        output(i, j) -= log_sum_exp;
      }
    }

    return output;

  } else {
    throw std::runtime_error("log_softmax: only 1D and 2D tensors supported");
  }
}

Tensor log_softmax_backward(const Tensor& grad_output, const Tensor& output) {
  if (grad_output.shape() != output.shape()) {
    throw std::invalid_argument("log_softmax_backward: shape mismatch");
  }

  if (output.ndim() == 1) {
    // 1D version
    size_t n = output.size();
    Tensor grad_input(n);

    // Compute softmax from log_softmax output
    Tensor prob(n);
    for (size_t i = 0; i < n; ++i) {
      prob(i) = std::exp(output(i));
    }

    // Gradient: grad_input[i] = grad_output[i] - prob[i] * sum_j(grad_output[j])
    double sum = 0.0;
    for (size_t j = 0; j < n; ++j) {
      sum += grad_output(j);
    }

    for (size_t i = 0; i < n; ++i) {
      grad_input(i) = grad_output(i) - prob(i) * sum;
    }

    return grad_input;

  } else if (output.ndim() == 2) {
    // 2D: row-wise
    size_t rows = output.shape()[0];
    size_t cols = output.shape()[1];
    Tensor grad_input(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
      double sum = 0.0;
      for (size_t j = 0; j < cols; ++j) {
        sum += grad_output(i, j);
      }

      for (size_t j = 0; j < cols; ++j) {
        double prob = std::exp(output(i, j));
        grad_input(i, j) = grad_output(i, j) - prob * sum;
      }
    }

    return grad_input;

  } else {
    throw std::runtime_error("log_softmax_backward: only 1D and 2D tensors supported");
  }
}

}  // namespace nn
}  // namespace golomb
