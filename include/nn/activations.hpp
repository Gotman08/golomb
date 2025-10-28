#pragma once

#include "nn/tensor.hpp"

namespace golomb {
namespace nn {

/**
 * @brief Activation functions and their derivatives for neural networks.
 *
 * All activation functions operate element-wise on tensors.
 * Each function has a forward pass (compute output) and backward pass
 * (compute gradient with respect to input).
 */

/**
 * @brief ReLU (Rectified Linear Unit) activation.
 *
 * Forward: f(x) = max(0, x)
 * Derivative: f'(x) = 1 if x > 0, else 0
 *
 * Commonly used in hidden layers. Non-saturating, helps with vanishing
 * gradients.
 *
 * @param x Input tensor.
 * @return Output tensor with ReLU applied element-wise.
 */
[[nodiscard]] Tensor relu(const Tensor& x);

/**
 * @brief ReLU backward pass.
 *
 * Computes gradient of loss with respect to input: dL/dx = dL/dy * dy/dx
 *
 * @param grad_output Gradient of loss w.r.t. output (dL/dy).
 * @param input Original input to relu (x).
 * @return Gradient of loss w.r.t. input (dL/dx).
 */
[[nodiscard]] Tensor relu_backward(const Tensor& grad_output, const Tensor& input);

/**
 * @brief Leaky ReLU activation.
 *
 * Forward: f(x) = x if x > 0, else alpha * x
 * Derivative: f'(x) = 1 if x > 0, else alpha
 *
 * Variant of ReLU that allows small negative values. Helps avoid "dead neurons".
 *
 * @param x Input tensor.
 * @param alpha Slope for negative values (default 0.01).
 * @return Output tensor with Leaky ReLU applied.
 */
[[nodiscard]] Tensor leaky_relu(const Tensor& x, double alpha = 0.01);

/**
 * @brief Leaky ReLU backward pass.
 *
 * @param grad_output Gradient of loss w.r.t. output.
 * @param input Original input.
 * @param alpha Slope for negative values (same as forward).
 * @return Gradient of loss w.r.t. input.
 */
[[nodiscard]] Tensor leaky_relu_backward(const Tensor& grad_output, const Tensor& input,
                                         double alpha = 0.01);

/**
 * @brief Tanh (hyperbolic tangent) activation.
 *
 * Forward: f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 * Derivative: f'(x) = 1 - tanh²(x)
 *
 * Outputs in range [-1, 1]. Commonly used for value head in AlphaGo-style networks.
 *
 * @param x Input tensor.
 * @return Output tensor with tanh applied.
 */
[[nodiscard]] Tensor tanh_activation(const Tensor& x);

/**
 * @brief Tanh backward pass.
 *
 * @param grad_output Gradient of loss w.r.t. output.
 * @param output Original output from tanh (NOT input).
 * @return Gradient of loss w.r.t. input.
 */
[[nodiscard]] Tensor tanh_backward(const Tensor& grad_output, const Tensor& output);

/**
 * @brief Sigmoid activation.
 *
 * Forward: f(x) = 1 / (1 + exp(-x))
 * Derivative: f'(x) = f(x) * (1 - f(x))
 *
 * Outputs in range [0, 1]. Can be used for binary classification.
 *
 * @param x Input tensor.
 * @return Output tensor with sigmoid applied.
 */
[[nodiscard]] Tensor sigmoid(const Tensor& x);

/**
 * @brief Sigmoid backward pass.
 *
 * @param grad_output Gradient of loss w.r.t. output.
 * @param output Original output from sigmoid (NOT input).
 * @return Gradient of loss w.r.t. input.
 */
[[nodiscard]] Tensor sigmoid_backward(const Tensor& grad_output, const Tensor& output);

/**
 * @brief Softmax activation (applied along last dimension for 1D/2D tensors).
 *
 * Forward: f(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
 * (numerically stable version with max subtraction)
 *
 * Outputs probability distribution: sum = 1, all values in [0, 1].
 * Used for policy head in AlphaGo-style networks.
 *
 * For 1D tensor: applies softmax across all elements.
 * For 2D tensor: applies softmax row-wise (each row sums to 1).
 *
 * @param x Input tensor (1D or 2D).
 * @return Output tensor with softmax applied.
 */
[[nodiscard]] Tensor softmax(const Tensor& x);

/**
 * @brief Softmax backward pass.
 *
 * For softmax followed by cross-entropy loss, it's more efficient to combine
 * their gradients. This function computes the general Jacobian-vector product.
 *
 * @param grad_output Gradient of loss w.r.t. output.
 * @param output Original output from softmax.
 * @return Gradient of loss w.r.t. input.
 */
[[nodiscard]] Tensor softmax_backward(const Tensor& grad_output, const Tensor& output);

/**
 * @brief Log-softmax activation.
 *
 * Forward: f(x_i) = log(softmax(x_i)) = x_i - max(x) - log(sum_j exp(x_j - max(x)))
 *
 * More numerically stable than log(softmax(x)) for use with NLL loss.
 *
 * @param x Input tensor (1D or 2D).
 * @return Output tensor with log-softmax applied.
 */
[[nodiscard]] Tensor log_softmax(const Tensor& x);

/**
 * @brief Log-softmax backward pass.
 *
 * @param grad_output Gradient of loss w.r.t. output.
 * @param output Original output from log_softmax.
 * @return Gradient of loss w.r.t. input.
 */
[[nodiscard]] Tensor log_softmax_backward(const Tensor& grad_output, const Tensor& output);

} // namespace nn
} // namespace golomb
