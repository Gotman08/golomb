#include "nn/golomb_net.hpp"
#include "nn/activations.hpp"

namespace golomb {
namespace nn {

GolombNet::GolombNet(const StateEncoder& encoder, int ub, size_t hidden1_size, size_t hidden2_size)
    : encoder_(encoder), ub_(ub), hidden1_(encoder.encoding_size(), hidden1_size),
      hidden2_(hidden1_size, hidden2_size), policy_head_(hidden2_size, static_cast<size_t>(ub + 1)),
      value_head_(hidden2_size, 1) {

  // Initialize with He (good for ReLU)
  init_he();
}

void GolombNet::forward(const RuleState& state, Tensor& policy_out, double& value_out) {
  // Encode state
  Tensor encoded = encoder_.encode(state);
  forward_encoded(encoded, policy_out, value_out);
}

void GolombNet::forward_encoded(const Tensor& encoded_state, Tensor& policy_out,
                                double& value_out) {
  // Cache input
  cached_input_ = encoded_state.copy();

  // Hidden layer 1
  cached_hidden1_ = hidden1_.forward(encoded_state);
  cached_hidden1_relu_ = relu(cached_hidden1_);

  // Hidden layer 2
  cached_hidden2_ = hidden2_.forward(cached_hidden1_relu_);
  cached_hidden2_relu_ = relu(cached_hidden2_);

  // Policy head: logits → softmax
  cached_policy_logits_ = policy_head_.forward(cached_hidden2_relu_);
  policy_out = softmax(cached_policy_logits_);

  // Value head: scalar → tanh
  Tensor value_tensor = value_head_.forward(cached_hidden2_relu_);
  cached_value_tanh_ = tanh_activation(value_tensor);
  value_out = cached_value_tanh_(0);
}

void GolombNet::backward(const Tensor& grad_policy, double grad_value) {
  // ========================================================================
  // Policy head backward
  // ========================================================================

  // Gradient through softmax
  // NOTE: We assume grad_policy is already dL/d(softmax_output)
  // For cross-entropy loss, this simplifies to (softmax_output - target)
  Tensor grad_policy_logits = softmax_backward(grad_policy, softmax(cached_policy_logits_));

  // Backprop through policy head layer
  Tensor grad_hidden2_from_policy = policy_head_.backward(grad_policy_logits);

  // ========================================================================
  // Value head backward
  // ========================================================================

  // Gradient w.r.t. value output is a scalar, convert to tensor
  Tensor grad_value_tensor(1);
  grad_value_tensor(0) = grad_value;

  // Gradient through tanh
  Tensor grad_value_pretanh = tanh_backward(grad_value_tensor, cached_value_tanh_);

  // Backprop through value head layer
  Tensor grad_hidden2_from_value = value_head_.backward(grad_value_pretanh);

  // ========================================================================
  // Combine gradients from both heads
  // ========================================================================

  Tensor grad_hidden2 = grad_hidden2_from_policy + grad_hidden2_from_value;

  // ========================================================================
  // Hidden layer 2 backward
  // ========================================================================

  // Gradient through ReLU
  Tensor grad_hidden2_pre_relu = relu_backward(grad_hidden2, cached_hidden2_);

  // Backprop through hidden2 layer
  Tensor grad_hidden1_relu = hidden2_.backward(grad_hidden2_pre_relu);

  // ========================================================================
  // Hidden layer 1 backward
  // ========================================================================

  // Gradient through ReLU
  Tensor grad_hidden1_pre_relu = relu_backward(grad_hidden1_relu, cached_hidden1_);

  // Backprop through hidden1 layer (completes gradient computation)
  hidden1_.backward(grad_hidden1_pre_relu);
}

std::vector<Tensor*> GolombNet::parameters() {
  std::vector<Tensor*> params;

  auto h1_params = hidden1_.parameters();
  auto h2_params = hidden2_.parameters();
  auto policy_params = policy_head_.parameters();
  auto value_params = value_head_.parameters();

  params.insert(params.end(), h1_params.begin(), h1_params.end());
  params.insert(params.end(), h2_params.begin(), h2_params.end());
  params.insert(params.end(), policy_params.begin(), policy_params.end());
  params.insert(params.end(), value_params.begin(), value_params.end());

  return params;
}

std::vector<Tensor*> GolombNet::gradients() {
  std::vector<Tensor*> grads;

  auto h1_grads = hidden1_.gradients();
  auto h2_grads = hidden2_.gradients();
  auto policy_grads = policy_head_.gradients();
  auto value_grads = value_head_.gradients();

  grads.insert(grads.end(), h1_grads.begin(), h1_grads.end());
  grads.insert(grads.end(), h2_grads.begin(), h2_grads.end());
  grads.insert(grads.end(), policy_grads.begin(), policy_grads.end());
  grads.insert(grads.end(), value_grads.begin(), value_grads.end());

  return grads;
}

void GolombNet::zero_grad() {
  hidden1_.zero_grad();
  hidden2_.zero_grad();
  policy_head_.zero_grad();
  value_head_.zero_grad();
}

size_t GolombNet::num_parameters() const {
  return hidden1_.num_parameters() + hidden2_.num_parameters() + policy_head_.num_parameters() +
         value_head_.num_parameters();
}

void GolombNet::init_he() {
  hidden1_.init_he();
  hidden2_.init_he();
  policy_head_.init_he();
  value_head_.init_he();
}

void GolombNet::init_xavier() {
  hidden1_.init_xavier();
  hidden2_.init_xavier();
  policy_head_.init_xavier();
  value_head_.init_xavier();
}

} // namespace nn
} // namespace golomb
