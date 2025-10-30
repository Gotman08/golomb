#include "nn/golomb_net.hpp"
#include "nn/activations.hpp"

namespace golomb {
namespace nn {

namespace {

// Helper: collect tensors from all layers using a member function pointer
template <typename MemberFunc>
std::vector<Tensor*> collect_from_all_layers(Linear& hidden1, Linear& hidden2, Linear& policy_head,
                                              Linear& value_head, MemberFunc func) {
  std::vector<Tensor*> result;

  auto append_from = [&](Linear& layer) {
    auto items = (layer.*func)();
    result.insert(result.end(), items.begin(), items.end());
  };

  append_from(hidden1);
  append_from(hidden2);
  append_from(policy_head);
  append_from(value_head);

  return result;
}

// Helper: apply a void function to all layers
template <typename MemberFunc>
void apply_to_all_layers(Linear& hidden1, Linear& hidden2, Linear& policy_head, Linear& value_head,
                         MemberFunc func) {
  (hidden1.*func)();
  (hidden2.*func)();
  (policy_head.*func)();
  (value_head.*func)();
}

} // namespace

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
  return collect_from_all_layers(hidden1_, hidden2_, policy_head_, value_head_,
                                  &Linear::parameters);
}

std::vector<Tensor*> GolombNet::gradients() {
  return collect_from_all_layers(hidden1_, hidden2_, policy_head_, value_head_,
                                  &Linear::gradients);
}

void GolombNet::zero_grad() {
  apply_to_all_layers(hidden1_, hidden2_, policy_head_, value_head_, &Linear::zero_grad);
}

size_t GolombNet::num_parameters() const {
  return hidden1_.num_parameters() + hidden2_.num_parameters() + policy_head_.num_parameters() +
         value_head_.num_parameters();
}

void GolombNet::init_he() {
  apply_to_all_layers(hidden1_, hidden2_, policy_head_, value_head_, &Linear::init_he);
}

void GolombNet::init_xavier() {
  apply_to_all_layers(hidden1_, hidden2_, policy_head_, value_head_, &Linear::init_xavier);
}

} // namespace nn
} // namespace golomb
