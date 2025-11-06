#pragma once

#include "nn/tensor.hpp"
#include <memory>
#include <vector>

namespace golomb {
namespace nn {

/**
 * @brief Base class for neural network optimizers.
 *
 * Optimizers update network parameters based on gradients computed during
 * backpropagation. Common algorithms include SGD, SGD with momentum, and Adam.
 */
class Optimizer {
public:
  /**
   * @brief Virtual destructor for polymorphism.
   */
  virtual ~Optimizer() = default;

  /**
   * @brief Perform one optimization step.
   *
   * Updates all parameters using their gradients.
   * Should be called after backward() and before zero_grad().
   */
  virtual void step() = 0;

  /**
   * @brief Get current learning rate.
   * @return Learning rate value.
   */
  [[nodiscard]] virtual double get_lr() const = 0;

  /**
   * @brief Set learning rate.
   * @param lr New learning rate.
   */
  virtual void set_lr(double lr) = 0;
};

/**
 * @brief Stochastic Gradient Descent optimizer with optional momentum.
 *
 * Update rule without momentum:
 *   θ ← θ - lr * ∇θ
 *
 * Update rule with momentum:
 *   v ← momentum * v + ∇θ
 *   θ ← θ - lr * v
 *
 * Momentum helps accelerate SGD in relevant direction and dampens oscillations.
 */
class SGD : public Optimizer {
public:
  /**
   * @brief Construct SGD optimizer.
   *
   * @param parameters Vector of pointers to parameter tensors to optimize.
   * @param gradients Vector of pointers to gradient tensors (must match parameters order).
   * @param lr Learning rate (default 0.01).
   * @param momentum Momentum factor [0, 1) (default 0.0, no momentum).
   * @param weight_decay L2 regularization coefficient (default 0.0).
   */
  SGD(const std::vector<Tensor*>& parameters, const std::vector<Tensor*>& gradients,
      double lr = 0.01, double momentum = 0.0, double weight_decay = 0.0);

  void step() override;

  [[nodiscard]] double get_lr() const override { return lr_; }

  void set_lr(double lr) override { lr_ = lr; }

  /**
   * @brief Get momentum factor.
   * @return Momentum value.
   */
  [[nodiscard]] double get_momentum() const { return momentum_; }

private:
  std::vector<Tensor*> parameters_;  ///< Pointers to parameters.
  std::vector<Tensor*> gradients_;   ///< Pointers to gradients (same order as parameters).
  std::vector<Tensor> velocity_;     ///< Velocity buffers for momentum.
  double lr_;                        ///< Learning rate.
  double momentum_;                  ///< Momentum factor.
  double weight_decay_;              ///< L2 regularization coefficient.
};

/**
 * @brief Adam optimizer (Adaptive Moment Estimation).
 *
 * Update rules:
 *   m ← β1 * m + (1 - β1) * ∇θ              (first moment)
 *   v ← β2 * v + (1 - β2) * ∇θ²             (second moment)
 *   m̂ ← m / (1 - β1^t)                      (bias correction)
 *   v̂ ← v / (1 - β2^t)                      (bias correction)
 *   θ ← θ - lr * m̂ / (√v̂ + ε)
 *
 * Adam adapts learning rate for each parameter based on first and second
 * moments of gradients. Generally works well with default hyperparameters.
 *
 * Reference: Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014)
 */
class Adam : public Optimizer {
public:
  /**
   * @brief Construct Adam optimizer.
   *
   * @param parameters Vector of pointers to parameter tensors to optimize.
   * @param gradients Vector of pointers to gradient tensors (must match parameters order).
   * @param lr Learning rate (default 0.001).
   * @param beta1 Exponential decay rate for first moment (default 0.9).
   * @param beta2 Exponential decay rate for second moment (default 0.999).
   * @param epsilon Small constant for numerical stability (default 1e-8).
   * @param weight_decay L2 regularization coefficient (default 0.0).
   */
  Adam(const std::vector<Tensor*>& parameters, const std::vector<Tensor*>& gradients,
       double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8,
       double weight_decay = 0.0);

  void step() override;

  [[nodiscard]] double get_lr() const override { return lr_; }

  void set_lr(double lr) override { lr_ = lr; }

  /**
   * @brief Get current timestep (number of steps taken).
   * @return Timestep value.
   */
  [[nodiscard]] int get_timestep() const { return timestep_; }

private:
  std::vector<Tensor*> parameters_;  ///< Pointers to parameters.
  std::vector<Tensor*> gradients_;   ///< Pointers to gradients (same order as parameters).
  std::vector<Tensor> m_;            ///< First moment estimates.
  std::vector<Tensor> v_;            ///< Second moment estimates.
  double lr_;                        ///< Learning rate.
  double beta1_;                     ///< First moment decay rate.
  double beta2_;                     ///< Second moment decay rate.
  double epsilon_;                   ///< Numerical stability constant.
  double weight_decay_;              ///< L2 regularization coefficient.
  int timestep_;                     ///< Current timestep (for bias correction).
};

} // namespace nn
} // namespace golomb
