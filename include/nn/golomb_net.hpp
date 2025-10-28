#pragma once

#include "core/golomb.hpp"
#include "nn/linear.hpp"
#include "nn/state_encoder.hpp"
#include "nn/tensor.hpp"
#include <memory>
#include <vector>

namespace golomb {
namespace nn {

/**
 * @brief Neural network for Golomb ruler optimization (AlphaGo-style).
 *
 * Architecture:
 *   Input (encoded state)
 *     ↓
 *   Hidden Layer 1 + ReLU
 *     ↓
 *   Hidden Layer 2 + ReLU
 *     ↓
 *   Split into two heads:
 *     ├─→ Policy Head + Softmax → probabilities for each position
 *     └─→ Value Head + Tanh → value estimation [-1, 1]
 *
 * The policy head predicts which position to place the next mark.
 * The value head estimates the quality of the current state.
 */
class GolombNet {
public:
  /**
   * @brief Construct neural network.
   * @param encoder State encoder (determines input size).
   * @param ub Upper bound for positions (determines policy output size).
   * @param hidden1_size Size of first hidden layer (default 256).
   * @param hidden2_size Size of second hidden layer (default 256).
   */
  GolombNet(const StateEncoder& encoder, int ub, size_t hidden1_size = 256,
            size_t hidden2_size = 256);

  /**
   * @brief Forward pass: compute policy and value.
   *
   * @param state Golomb ruler state.
   * @param policy_out Output: policy probabilities [ub+1].
   * @param value_out Output: value estimation (scalar).
   */
  void forward(const RuleState& state, Tensor& policy_out, double& value_out);

  /**
   * @brief Forward pass from encoded tensor.
   *
   * Useful when state is already encoded (e.g., in batch processing).
   *
   * @param encoded_state Encoded state tensor.
   * @param policy_out Output: policy probabilities [ub+1].
   * @param value_out Output: value estimation (scalar).
   */
  void forward_encoded(const Tensor& encoded_state, Tensor& policy_out, double& value_out);

  /**
   * @brief Backward pass: compute gradients.
   *
   * Given gradients of loss w.r.t. policy and value outputs,
   * computes gradients for all network parameters.
   *
   * @param grad_policy Gradient of loss w.r.t. policy output [ub+1].
   * @param grad_value Gradient of loss w.r.t. value output (scalar).
   */
  void backward(const Tensor& grad_policy, double grad_value);

  /**
   * @brief Get all trainable parameters.
   * @return Vector of pointers to parameter tensors.
   */
  std::vector<Tensor*> parameters();

  /**
   * @brief Get all parameter gradients.
   * @return Vector of pointers to gradient tensors.
   */
  std::vector<Tensor*> gradients();

  /**
   * @brief Zero all parameter gradients.
   */
  void zero_grad();

  /**
   * @brief Get total number of trainable parameters.
   * @return Total parameter count.
   */
  size_t num_parameters() const;

  /**
   * @brief Initialize all layers with He initialization.
   *
   * Recommended for ReLU networks.
   */
  void init_he();

  /**
   * @brief Initialize all layers with Xavier initialization.
   */
  void init_xavier();

  /**
   * @brief Get state encoder.
   * @return Reference to encoder.
   */
  [[nodiscard]] const StateEncoder& encoder() const { return encoder_; }

  /**
   * @brief Get upper bound.
   * @return Upper bound for positions.
   */
  [[nodiscard]] int ub() const { return ub_; }

private:
  StateEncoder encoder_; ///< State encoder.
  int ub_;               ///< Upper bound for positions.

  // Network layers
  Linear hidden1_;     ///< First hidden layer.
  Linear hidden2_;     ///< Second hidden layer.
  Linear policy_head_; ///< Policy head (outputs logits for each position).
  Linear value_head_;  ///< Value head (outputs scalar value).

  // Cached activations for backward pass
  Tensor cached_input_;         ///< Cached input (encoded state).
  Tensor cached_hidden1_;       ///< Cached hidden1 output (before activation).
  Tensor cached_hidden1_relu_;  ///< Cached hidden1 after ReLU.
  Tensor cached_hidden2_;       ///< Cached hidden2 output (before activation).
  Tensor cached_hidden2_relu_;  ///< Cached hidden2 after ReLU.
  Tensor cached_policy_logits_; ///< Cached policy logits (before softmax).
  Tensor cached_value_tanh_;    ///< Cached value after tanh.
};

} // namespace nn
} // namespace golomb
