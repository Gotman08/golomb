#include "nn/activations.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

// OPT-1C: AVX2 vectorization (CSAPP 5.9 - SIMD parallelism)
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace golomb {
namespace nn {

// ============================================================================
// ReLU
// ============================================================================

Tensor relu(const Tensor& x) {
  Tensor output = x.copy();
  double* data = output.data().data();
  const size_t n = output.size();

#if defined(__AVX2__)
  // OPT-1C: AVX2 vectorized ReLU - 4x throughput (CSAPP 5.9)
  const __m256d zero_vec = _mm256_setzero_pd();
  size_t i = 0;

  // Process 4 doubles at a time
  for (; i + 4 <= n; i += 4) {
    __m256d vals = _mm256_loadu_pd(&data[i]);
    vals = _mm256_max_pd(vals, zero_vec);  // max(0, x)
    _mm256_storeu_pd(&data[i], vals);
  }

  // Handle remaining elements
  for (; i < n; ++i) {
    data[i] = std::max(0.0, data[i]);
  }
#else
  // Scalar fallback
  for (size_t i = 0; i < n; ++i) {
    data[i] = std::max(0.0, data[i]);
  }
#endif

  return output;
}

Tensor relu_backward(const Tensor& grad_output, const Tensor& input) {
  if (grad_output.shape() != input.shape()) {
    throw std::invalid_argument("relu_backward: shape mismatch");
  }

  Tensor grad_input = grad_output.copy();
  const double* input_data = input.data().data();
  double* grad_data = grad_input.data().data();
  const size_t n = grad_input.size();

#if defined(__AVX2__)
  // OPT-1C: AVX2 vectorized ReLU backward - 4x throughput (CSAPP 5.9)
  const __m256d zero_vec = _mm256_setzero_pd();
  size_t i = 0;

  // Process 4 doubles at a time
  for (; i + 4 <= n; i += 4) {
    __m256d inp = _mm256_loadu_pd(&input_data[i]);
    __m256d grad = _mm256_loadu_pd(&grad_data[i]);

    // Create mask: input > 0 ? 0xFFFF... : 0x0000...
    __m256d mask = _mm256_cmp_pd(inp, zero_vec, _CMP_GT_OQ);

    // Apply mask: zero out gradient where input <= 0
    grad = _mm256_and_pd(grad, mask);
    _mm256_storeu_pd(&grad_data[i], grad);
  }

  // Handle remaining elements
  for (; i < n; ++i) {
    if (input_data[i] <= 0.0) {
      grad_data[i] = 0.0;
    }
  }
#else
  // Scalar fallback
  for (size_t i = 0; i < n; ++i) {
    if (input_data[i] <= 0.0) {
      grad_data[i] = 0.0;
    }
  }
#endif

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
  double* data = output.data().data();
  const size_t n = output.size();

#if defined(__AVX2__)
  // OPT-1C: AVX2 vectorized Tanh using Padé approximation (CSAPP 5.9)
  // tanh(x) ≈ x * (27 + x²) / (27 + 9*x²) for |x| < 3
  // tanh(x) ≈ sign(x) for |x| >= 3

  const __m256d c1 = _mm256_set1_pd(27.0);
  const __m256d c9 = _mm256_set1_pd(9.0);
  const __m256d c3 = _mm256_set1_pd(3.0);
  const __m256d neg_c3 = _mm256_set1_pd(-3.0);
  const __m256d one = _mm256_set1_pd(1.0);
  const __m256d neg_one = _mm256_set1_pd(-1.0);

  size_t i = 0;

  // Process 4 doubles at a time
  for (; i + 4 <= n; i += 4) {
    __m256d x_vec = _mm256_loadu_pd(&data[i]);

    // For |x| >= 3: return sign(x)
    __m256d ge_3 = _mm256_cmp_pd(x_vec, c3, _CMP_GE_OQ);
    __m256d le_neg3 = _mm256_cmp_pd(x_vec, neg_c3, _CMP_LE_OQ);

    // Compute tanh approximation: x * (27 + x²) / (27 + 9*x²)
    __m256d x2 = _mm256_mul_pd(x_vec, x_vec);  // x²
    __m256d num = _mm256_add_pd(c1, x2);       // 27 + x²
    num = _mm256_mul_pd(x_vec, num);           // x * (27 + x²)

    __m256d den = _mm256_mul_pd(c9, x2);       // 9*x²
    den = _mm256_add_pd(c1, den);              // 27 + 9*x²

    __m256d result = _mm256_div_pd(num, den);  // (x * (27 + x²)) / (27 + 9*x²)

    // Apply saturation: if x >= 3, result = 1; if x <= -3, result = -1
    result = _mm256_blendv_pd(result, one, ge_3);
    result = _mm256_blendv_pd(result, neg_one, le_neg3);

    _mm256_storeu_pd(&data[i], result);
  }

  // Handle remaining elements with scalar tanh
  for (; i < n; ++i) {
    data[i] = std::tanh(data[i]);
  }
#else
  // Scalar fallback
  for (size_t i = 0; i < n; ++i) {
    data[i] = std::tanh(data[i]);
  }
#endif

  return output;
}

Tensor tanh_backward(const Tensor& grad_output, const Tensor& output) {
  if (grad_output.shape() != output.shape()) {
    throw std::invalid_argument("tanh_backward: shape mismatch");
  }

  Tensor grad_input = grad_output.copy();
  const double* output_data = output.data().data();
  double* grad_data = grad_input.data().data();
  const size_t n = grad_input.size();

#if defined(__AVX2__)
  // OPT-1C: AVX2 vectorized Tanh backward - 4x throughput (CSAPP 5.9)
  const __m256d one_vec = _mm256_set1_pd(1.0);
  size_t i = 0;

  // Process 4 doubles at a time
  for (; i + 4 <= n; i += 4) {
    __m256d tanh_val = _mm256_loadu_pd(&output_data[i]);
    __m256d grad = _mm256_loadu_pd(&grad_data[i]);

    // d/dx tanh(x) = 1 - tanh²(x)
    __m256d tanh_sq = _mm256_mul_pd(tanh_val, tanh_val);  // tanh²
    __m256d deriv = _mm256_sub_pd(one_vec, tanh_sq);      // 1 - tanh²
    grad = _mm256_mul_pd(grad, deriv);                     // grad * (1 - tanh²)

    _mm256_storeu_pd(&grad_data[i], grad);
  }

  // Handle remaining elements
  for (; i < n; ++i) {
    double tanh_val = output_data[i];
    grad_data[i] *= (1.0 - tanh_val * tanh_val);
  }
#else
  // Scalar fallback
  for (size_t i = 0; i < n; ++i) {
    double tanh_val = output_data[i];
    grad_data[i] *= (1.0 - tanh_val * tanh_val);
  }
#endif

  return grad_input;
}

// ============================================================================
// Sigmoid
// ============================================================================

Tensor sigmoid(const Tensor& x) {
  Tensor output = x.copy();
  double* data = output.data().data();
  const size_t n = output.size();

#if defined(__AVX2__)
  // OPT-1C: AVX2 vectorized Sigmoid using tanh (CSAPP 5.9)
  // sigmoid(x) = 0.5 * (1 + tanh(x/2))

  const __m256d half = _mm256_set1_pd(0.5);
  const __m256d one = _mm256_set1_pd(1.0);
  const __m256d c27 = _mm256_set1_pd(27.0);
  const __m256d c9 = _mm256_set1_pd(9.0);
  const __m256d c6 = _mm256_set1_pd(6.0);
  const __m256d neg_c6 = _mm256_set1_pd(-6.0);

  size_t i = 0;

  // Process 4 doubles at a time
  for (; i + 4 <= n; i += 4) {
    __m256d x_vec = _mm256_loadu_pd(&data[i]);

    // Compute x/2
    __m256d x_half = _mm256_mul_pd(x_vec, half);

    // Clamp to [-6, 6] for numerical stability
    x_half = _mm256_max_pd(x_half, neg_c6);
    x_half = _mm256_min_pd(x_half, c6);

    // Compute tanh(x/2) using Padé approximation
    __m256d x2 = _mm256_mul_pd(x_half, x_half);  // (x/2)²
    __m256d num = _mm256_add_pd(c27, x2);        // 27 + (x/2)²
    num = _mm256_mul_pd(x_half, num);            // (x/2) * (27 + (x/2)²)

    __m256d den = _mm256_mul_pd(c9, x2);         // 9*(x/2)²
    den = _mm256_add_pd(c27, den);               // 27 + 9*(x/2)²

    __m256d tanh_val = _mm256_div_pd(num, den);  // tanh(x/2)

    // sigmoid(x) = 0.5 * (1 + tanh(x/2))
    __m256d result = _mm256_add_pd(one, tanh_val);  // 1 + tanh(x/2)
    result = _mm256_mul_pd(half, result);            // 0.5 * (1 + tanh(x/2))

    _mm256_storeu_pd(&data[i], result);
  }

  // Handle remaining elements with numerically stable scalar sigmoid
  for (; i < n; ++i) {
    double val = data[i];
    if (val >= 0.0) {
      double exp_neg = std::exp(-val);
      data[i] = 1.0 / (1.0 + exp_neg);
    } else {
      double exp_pos = std::exp(val);
      data[i] = exp_pos / (1.0 + exp_pos);
    }
  }
#else
  // Scalar fallback
  for (size_t i = 0; i < n; ++i) {
    double val = data[i];
    if (val >= 0.0) {
      double exp_neg = std::exp(-val);
      data[i] = 1.0 / (1.0 + exp_neg);
    } else {
      double exp_pos = std::exp(val);
      data[i] = exp_pos / (1.0 + exp_pos);
    }
  }
#endif

  return output;
}

Tensor sigmoid_backward(const Tensor& grad_output, const Tensor& output) {
  if (grad_output.shape() != output.shape()) {
    throw std::invalid_argument("sigmoid_backward: shape mismatch");
  }

  Tensor grad_input = grad_output.copy();
  const double* output_data = output.data().data();
  double* grad_data = grad_input.data().data();
  const size_t n = grad_input.size();

#if defined(__AVX2__)
  // OPT-1C: AVX2 vectorized Sigmoid backward - 4x throughput (CSAPP 5.9)
  const __m256d one_vec = _mm256_set1_pd(1.0);
  size_t i = 0;

  // Process 4 doubles at a time
  for (; i + 4 <= n; i += 4) {
    __m256d sig_val = _mm256_loadu_pd(&output_data[i]);
    __m256d grad = _mm256_loadu_pd(&grad_data[i]);

    // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    __m256d one_minus_sig = _mm256_sub_pd(one_vec, sig_val);  // 1 - sigmoid
    __m256d deriv = _mm256_mul_pd(sig_val, one_minus_sig);    // sigmoid * (1 - sigmoid)
    grad = _mm256_mul_pd(grad, deriv);                         // grad * deriv

    _mm256_storeu_pd(&grad_data[i], grad);
  }

  // Handle remaining elements
  for (; i < n; ++i) {
    double sig_val = output_data[i];
    grad_data[i] *= sig_val * (1.0 - sig_val);
  }
#else
  // Scalar fallback
  for (size_t i = 0; i < n; ++i) {
    double sig_val = output_data[i];
    grad_data[i] *= sig_val * (1.0 - sig_val);
  }
#endif

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

    // For softmax, gradient is: grad_input[i] = sum_j (output[i] * (delta_ij - output[j]) *
    // grad_output[j]) Simplified: grad_input[i] = output[i] * (grad_output[i] - sum_j(output[j] *
    // grad_output[j]))
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

} // namespace nn
} // namespace golomb
