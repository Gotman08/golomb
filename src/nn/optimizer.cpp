#include "nn/optimizer.hpp"
#include <cmath>
#include <stdexcept>

namespace golomb {
namespace nn {

// ================================
// SGD Implementation
// ================================

SGD::SGD(const std::vector<Tensor*>& parameters, const std::vector<Tensor*>& gradients, double lr,
         double momentum, double weight_decay)
    : parameters_(parameters),
      gradients_(gradients),
      lr_(lr),
      momentum_(momentum),
      weight_decay_(weight_decay) {
  if (parameters_.size() != gradients_.size()) {
    throw std::invalid_argument("SGD: parameters and gradients vectors must have same size");
  }

  // Initialize velocity buffers (same shape as parameters)
  velocity_.reserve(parameters_.size());
  for (const auto* param : parameters_) {
    velocity_.emplace_back(param->shape());
    velocity_.back().zeros(); // Initialize to zero
  }
}

void SGD::step() {
  if (parameters_.empty()) {
    return;
  }

  for (size_t i = 0; i < parameters_.size(); ++i) {
    Tensor* param = parameters_[i];
    Tensor* grad = gradients_[i];
    Tensor& vel = velocity_[i];

    // Apply weight decay (L2 regularization)
    // ∇L = ∇L_data + λθ
    Tensor grad_with_decay = *grad;
    if (weight_decay_ > 0.0) {
      grad_with_decay += (*param) * weight_decay_;
    }

    if (momentum_ > 0.0) {
      // Update velocity: v = momentum * v + grad
      vel *= momentum_;
      vel += grad_with_decay;

      // Update parameters: θ = θ - lr * v
      *param -= vel * lr_;
    } else {
      // Simple SGD without momentum: θ = θ - lr * grad
      *param -= grad_with_decay * lr_;
    }
  }
}

// ================================
// Adam Implementation
// ================================

Adam::Adam(const std::vector<Tensor*>& parameters, const std::vector<Tensor*>& gradients, double lr,
           double beta1, double beta2, double epsilon, double weight_decay)
    : parameters_(parameters),
      gradients_(gradients),
      lr_(lr),
      beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon),
      weight_decay_(weight_decay),
      timestep_(0) {
  if (parameters_.size() != gradients_.size()) {
    throw std::invalid_argument("Adam: parameters and gradients vectors must have same size");
  }

  // Initialize first and second moment estimates (same shape as parameters)
  m_.reserve(parameters_.size());
  v_.reserve(parameters_.size());

  for (const auto* param : parameters_) {
    m_.emplace_back(param->shape());
    m_.back().zeros();
    v_.emplace_back(param->shape());
    v_.back().zeros();
  }
}

void Adam::step() {
  if (parameters_.empty()) {
    return;
  }

  timestep_++;

  // Bias correction terms
  double bias_correction1 = 1.0 - std::pow(beta1_, timestep_);
  double bias_correction2 = 1.0 - std::pow(beta2_, timestep_);

  for (size_t i = 0; i < parameters_.size(); ++i) {
    Tensor* param = parameters_[i];
    Tensor* grad = gradients_[i];
    Tensor& m = m_[i];
    Tensor& v = v_[i];

    // Apply weight decay (L2 regularization)
    Tensor grad_with_decay = *grad;
    if (weight_decay_ > 0.0) {
      grad_with_decay += (*param) * weight_decay_;
    }

    // Update biased first moment estimate: m = β1 * m + (1 - β1) * grad
    m *= beta1_;
    m += grad_with_decay * (1.0 - beta1_);

    // Update biased second moment estimate: v = β2 * v + (1 - β2) * grad²
    v *= beta2_;
    Tensor grad_squared = grad_with_decay * grad_with_decay; // Element-wise square
    v += grad_squared * (1.0 - beta2_);

    // Compute bias-corrected moments
    Tensor m_hat = m * (1.0 / bias_correction1);
    Tensor v_hat = v * (1.0 / bias_correction2);

    // Update parameters: θ = θ - lr * m̂ / (√v̂ + ε)
    // Need to compute sqrt(v_hat) + epsilon element-wise
    Tensor denominator = v_hat;
    for (size_t j = 0; j < denominator.size(); ++j) {
      denominator.data()[j] = std::sqrt(denominator.data()[j]) + epsilon_;
    }

    // Element-wise division: m_hat / denominator
    Tensor update = m_hat;
    for (size_t j = 0; j < update.size(); ++j) {
      update.data()[j] /= denominator.data()[j];
    }

    // Apply update
    *param -= update * lr_;
  }
}

} // namespace nn
} // namespace golomb
