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

  // OPT-2C: Loop fusion - single pass over data for cache efficiency (CSAPP 6.4)
  for (size_t i = 0; i < parameters_.size(); ++i) {
    Tensor* param = parameters_[i];
    Tensor* grad = gradients_[i];
    Tensor& m = m_[i];
    Tensor& v = v_[i];

    const size_t n = param->size();
    const double one_minus_beta1 = 1.0 - beta1_;
    const double one_minus_beta2 = 1.0 - beta2_;
    const double inv_bias_corr1 = 1.0 / bias_correction1;
    const double inv_bias_corr2 = 1.0 / bias_correction2;

    // FUSED LOOP: update m, v, and apply parameter update in single pass
    for (size_t j = 0; j < n; ++j) {
      // Get gradient with weight decay
      double g = grad->data()[j];
      if (weight_decay_ > 0.0) {
        g += param->data()[j] * weight_decay_;
      }

      // Update biased first moment: m = β1*m + (1-β1)*g
      m.data()[j] = beta1_ * m.data()[j] + one_minus_beta1 * g;

      // Update biased second moment: v = β2*v + (1-β2)*g²
      v.data()[j] = beta2_ * v.data()[j] + one_minus_beta2 * g * g;

      // Bias-corrected moments
      double m_hat = m.data()[j] * inv_bias_corr1;
      double v_hat = v.data()[j] * inv_bias_corr2;

      // Compute update: lr * m_hat / (sqrt(v_hat) + epsilon)
      double update = lr_ * m_hat / (std::sqrt(v_hat) + epsilon_);

      // Apply update
      param->data()[j] -= update;
    }
  }
}

} // namespace nn
} // namespace golomb
