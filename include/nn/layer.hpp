#pragma once

#include "nn/tensor.hpp"
#include <memory>
#include <vector>

namespace golomb {
namespace nn {

/**
 * @brief Abstract base class for neural network layers.
 *
 * Defines interface for forward pass, backward pass, and parameter access.
 * All concrete layers (Linear, etc.) inherit from this.
 */
class Layer {
public:
  virtual ~Layer() = default;

  /**
   * @brief Forward pass: compute output from input.
   * @param input Input tensor.
   * @return Output tensor.
   */
  virtual Tensor forward(const Tensor& input) = 0;

  /**
   * @brief Backward pass: compute gradients.
   *
   * Given gradient of loss w.r.t. output, computes:
   * - Gradient of loss w.r.t. input (returned)
   * - Gradient of loss w.r.t. layer parameters (stored internally)
   *
   * @param grad_output Gradient of loss w.r.t. output.
   * @return Gradient of loss w.r.t. input.
   */
  virtual Tensor backward(const Tensor& grad_output) = 0;

  /**
   * @brief Get layer parameters (weights, biases).
   * @return Vector of pointers to parameter tensors.
   */
  virtual std::vector<Tensor*> parameters() = 0;

  /**
   * @brief Get parameter gradients.
   * @return Vector of pointers to gradient tensors.
   */
  virtual std::vector<Tensor*> gradients() = 0;

  /**
   * @brief Zero out all parameter gradients.
   */
  virtual void zero_grad() = 0;

  /**
   * @brief Get number of trainable parameters.
   * @return Total number of parameters.
   */
  virtual size_t num_parameters() const = 0;
};

} // namespace nn
} // namespace golomb
