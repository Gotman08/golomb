#pragma once

#include "nn/layer.hpp"
#include "nn/tensor.hpp"
#include <optional>

namespace golomb {
namespace nn {

/**
 * @brief Fully connected (dense) linear layer.
 *
 * Applies affine transformation: y = x * W^T + b
 * where:
 * - x: input [batch_size, in_features] or [in_features]
 * - W: weight matrix [out_features, in_features]
 * - b: bias vector [out_features]
 * - y: output [batch_size, out_features] or [out_features]
 *
 * Supports both single samples (1D) and batches (2D).
 */
class Linear : public Layer {
public:
  /**
   * @brief Construct a linear layer.
   * @param in_features Number of input features.
   * @param out_features Number of output features.
   * @param use_bias Whether to include bias term (default true).
   */
  Linear(size_t in_features, size_t out_features, bool use_bias = true);

  /**
   * @brief Forward pass.
   *
   * Computes y = x * W^T + b
   *
   * @param input Input tensor [batch_size, in_features] or [in_features].
   * @return Output tensor [batch_size, out_features] or [out_features].
   */
  Tensor forward(const Tensor& input) override;

  /**
   * @brief Backward pass.
   *
   * Computes gradients:
   * - grad_input = grad_output * W
   * - grad_weight = grad_output^T * input
   * - grad_bias = sum(grad_output, axis=0)
   *
   * @param grad_output Gradient of loss w.r.t. output.
   * @return Gradient of loss w.r.t. input.
   */
  Tensor backward(const Tensor& grad_output) override;

  /**
   * @brief Get layer parameters.
   * @return Vector containing [weight, bias] (or just [weight] if no bias).
   */
  std::vector<Tensor*> parameters() override;

  /**
   * @brief Get parameter gradients.
   * @return Vector containing [grad_weight, grad_bias].
   */
  std::vector<Tensor*> gradients() override;

  /**
   * @brief Zero out all gradients.
   */
  void zero_grad() override;

  /**
   * @brief Get total number of parameters.
   * @return in_features * out_features (+ out_features if bias).
   */
  size_t num_parameters() const override;

  /**
   * @brief Initialize weights using Xavier initialization.
   *
   * Samples from U[-limit, limit] where limit = sqrt(6 / (fan_in + fan_out)).
   * Good default for sigmoid/tanh activations.
   */
  void init_xavier();

  /**
   * @brief Initialize weights using He initialization.
   *
   * Samples from N(0, sqrt(2 / fan_in)).
   * Recommended for ReLU activations.
   */
  void init_he();

  /**
   * @brief Get weight matrix (const).
   * @return Const reference to weight tensor.
   */
  [[nodiscard]] const Tensor& weight() const { return weight_; }

  /**
   * @brief Get weight matrix (mutable).
   * @return Reference to weight tensor.
   */
  Tensor& weight() { return weight_; }

  /**
   * @brief Get bias vector (const).
   * @return Const reference to bias tensor (throws if no bias).
   */
  [[nodiscard]] const Tensor& bias() const;

  /**
   * @brief Get bias vector (mutable).
   * @return Reference to bias tensor (throws if no bias).
   */
  Tensor& bias();

  /**
   * @brief Check if layer has bias.
   * @return True if bias is used.
   */
  [[nodiscard]] bool has_bias() const { return use_bias_; }

private:
  size_t in_features_;  ///< Input dimension.
  size_t out_features_; ///< Output dimension.
  bool use_bias_;       ///< Whether to use bias.

  Tensor weight_; ///< Weight matrix [out_features, in_features].
  Tensor bias_;   ///< Bias vector [out_features] (optional).

  Tensor grad_weight_; ///< Gradient w.r.t. weight.
  Tensor grad_bias_;   ///< Gradient w.r.t. bias (optional).

  // OPT-4B: Cache pointer instead of copy (CSAPP 9.9 - Eliminate allocations)
  // SAFETY: Input must remain valid until backward() completes
  const Tensor* cached_input_ = nullptr; ///< Cached input pointer from forward pass.
};

} // namespace nn
} // namespace golomb
