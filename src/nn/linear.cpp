#include "nn/linear.hpp"
#include <stdexcept>

namespace golomb {
namespace nn {

Linear::Linear(size_t in_features, size_t out_features, bool use_bias)
    : in_features_(in_features), out_features_(out_features), use_bias_(use_bias),
      weight_(out_features, in_features), bias_(out_features),
      grad_weight_(out_features, in_features), grad_bias_(out_features) {

  // Initialize with He by default (good for ReLU networks)
  init_he();

  if (use_bias_) {
    bias_.zeros();
  }

  zero_grad();
}

Tensor Linear::forward(const Tensor& input) {
  // Cache input for backward pass
  cached_input_ = input.copy();

  if (input.ndim() == 1) {
    // Single sample: input [in_features]
    if (input.shape()[0] != in_features_) {
      throw std::invalid_argument("Linear::forward: input size mismatch");
    }

    // y = W * x (since we store W as [out_features, in_features])
    // We need to compute W * x where x is a column vector
    Tensor output(out_features_);

    for (size_t i = 0; i < out_features_; ++i) {
      double sum = 0.0;
      for (size_t j = 0; j < in_features_; ++j) {
        sum += weight_(i, j) * input(j);
      }
      output(i) = sum;
      if (use_bias_) {
        output(i) += bias_(i);
      }
    }

    return output;

  } else if (input.ndim() == 2) {
    // Batch: input [batch_size, in_features]
    size_t batch_size = input.shape()[0];
    if (input.shape()[1] != in_features_) {
      throw std::invalid_argument("Linear::forward: input feature size mismatch");
    }

    // y = x * W^T + b
    // Since we store W as [out_features, in_features],
    // we compute: output[i,j] = sum_k(input[i,k] * W[j,k])
    Tensor output(batch_size, out_features_);

    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < out_features_; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < in_features_; ++k) {
          sum += input(i, k) * weight_(j, k);
        }
        output(i, j) = sum;
        if (use_bias_) {
          output(i, j) += bias_(j);
        }
      }
    }

    return output;

  } else {
    throw std::runtime_error("Linear::forward: input must be 1D or 2D");
  }
}

Tensor Linear::backward(const Tensor& grad_output) {
  if (!cached_input_.has_value()) {
    throw std::runtime_error("Linear::backward: must call forward() first");
  }

  const Tensor& input = cached_input_.value();

  if (input.ndim() == 1) {
    // Single sample
    if (grad_output.shape()[0] != out_features_) {
      throw std::invalid_argument("Linear::backward: grad_output size mismatch");
    }

    // grad_input = W^T * grad_output
    Tensor grad_input(in_features_);
    for (size_t j = 0; j < in_features_; ++j) {
      double sum = 0.0;
      for (size_t i = 0; i < out_features_; ++i) {
        sum += weight_(i, j) * grad_output(i);
      }
      grad_input(j) = sum;
    }

    // grad_weight += grad_output * input^T
    // grad_weight[i,j] += grad_output[i] * input[j]
    for (size_t i = 0; i < out_features_; ++i) {
      for (size_t j = 0; j < in_features_; ++j) {
        grad_weight_(i, j) += grad_output(i) * input(j);
      }
    }

    // grad_bias += grad_output
    if (use_bias_) {
      for (size_t i = 0; i < out_features_; ++i) {
        grad_bias_(i) += grad_output(i);
      }
    }

    return grad_input;

  } else if (input.ndim() == 2) {
    // Batch
    size_t batch_size = input.shape()[0];
    if (grad_output.shape()[0] != batch_size || grad_output.shape()[1] != out_features_) {
      throw std::invalid_argument("Linear::backward: grad_output shape mismatch");
    }

    // grad_input = grad_output * W
    // grad_input[i,k] = sum_j(grad_output[i,j] * W[j,k])
    Tensor grad_input(batch_size, in_features_);
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t k = 0; k < in_features_; ++k) {
        double sum = 0.0;
        for (size_t j = 0; j < out_features_; ++j) {
          sum += grad_output(i, j) * weight_(j, k);
        }
        grad_input(i, k) = sum;
      }
    }

    // grad_weight += grad_output^T * input
    // grad_weight[j,k] += sum_i(grad_output[i,j] * input[i,k])
    for (size_t j = 0; j < out_features_; ++j) {
      for (size_t k = 0; k < in_features_; ++k) {
        double sum = 0.0;
        for (size_t i = 0; i < batch_size; ++i) {
          sum += grad_output(i, j) * input(i, k);
        }
        grad_weight_(j, k) += sum;
      }
    }

    // grad_bias += sum(grad_output, axis=0)
    if (use_bias_) {
      for (size_t j = 0; j < out_features_; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < batch_size; ++i) {
          sum += grad_output(i, j);
        }
        grad_bias_(j) += sum;
      }
    }

    return grad_input;

  } else {
    throw std::runtime_error("Linear::backward: cached input must be 1D or 2D");
  }
}

std::vector<Tensor*> Linear::parameters() {
  if (use_bias_) {
    return {&weight_, &bias_};
  } else {
    return {&weight_};
  }
}

std::vector<Tensor*> Linear::gradients() {
  if (use_bias_) {
    return {&grad_weight_, &grad_bias_};
  } else {
    return {&grad_weight_};
  }
}

void Linear::zero_grad() {
  grad_weight_.zeros();
  if (use_bias_) {
    grad_bias_.zeros();
  }
}

size_t Linear::num_parameters() const {
  size_t count = in_features_ * out_features_;
  if (use_bias_) {
    count += out_features_;
  }
  return count;
}

void Linear::init_xavier() {
  weight_ = Tensor::xavier(out_features_, in_features_);
  if (use_bias_) {
    bias_.zeros();
  }
}

void Linear::init_he() {
  weight_ = Tensor::he(out_features_, in_features_);
  if (use_bias_) {
    bias_.zeros();
  }
}

const Tensor& Linear::bias() const {
  if (!use_bias_) {
    throw std::runtime_error("Linear::bias: layer has no bias");
  }
  return bias_;
}

Tensor& Linear::bias() {
  if (!use_bias_) {
    throw std::runtime_error("Linear::bias: layer has no bias");
  }
  return bias_;
}

} // namespace nn
} // namespace golomb
