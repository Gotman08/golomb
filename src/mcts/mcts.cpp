#include "mcts/mcts.hpp"
#include "nn/golomb_net.hpp"
#include "utils/random.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace golomb {

namespace {

// Constants for MCTS evaluation
constexpr double INCOMPLETE_MARK_PENALTY = 1000.0;
constexpr double NN_VALUE_SCALE = 1000.0;

// NOTE: get legal actions (positions that can be added without conflicts)
std::vector<int> get_legal_actions(const RuleState& st, int ub) {
  std::vector<int> actions;
  for (int p = 1; p < ub; ++p) {
    // Skip if already placed
    if (std::find(st.marks.begin(), st.marks.end(), p) != st.marks.end()) {
      continue;
    }
    if (st.used.can_add_mark(st.marks, p)) {
      actions.push_back(p);
    }
  }
  return actions;
}

// Uniform policy priors (no neural network)
void compute_policy_priors_uniform(MCTSNode* node, const std::vector<int>& actions) {
  double uniform_prob = actions.empty() ? 0.0 : 1.0 / actions.size();
  for (int a : actions) {
    node->P[a] = uniform_prob;
  }
}

// Neural network policy priors
void compute_policy_priors_nn(MCTSNode* node, const std::vector<int>& actions,
                              nn::GolombNet* network) {
  if (!network || actions.empty()) {
    compute_policy_priors_uniform(node, actions);
    return;
  }

  // Get policy from network
  nn::Tensor policy;
  double value_unused;
  network->forward(node->state, policy, value_unused);

  // Normalize over legal actions only
  double total_prob = 0.0;
  for (int a : actions) {
    total_prob += policy(static_cast<size_t>(a));
  }

  if (total_prob > 0.0) {
    // Set priors from network (normalized)
    for (int a : actions) {
      node->P[a] = policy(static_cast<size_t>(a)) / total_prob;
    }
  } else {
    // Fallback to uniform if network gives zero probability to all legal actions
    compute_policy_priors_uniform(node, actions);
  }
}

// Simple heuristic leaf evaluation (no neural network)
double evaluate_leaf_heuristic(const RuleState& st, int target_n) {
  // Simple heuristic: prefer states with more marks placed and shorter length
  int marks_placed = static_cast<int>(st.marks.size());
  int len = st.marks.empty() ? 0 : st.marks.back();

  if (marks_placed < target_n) {
    return -len - INCOMPLETE_MARK_PENALTY * (target_n - marks_placed);
  }
  return -len;
}

// Neural network leaf evaluation
double evaluate_leaf_nn(const RuleState& st, int target_n, nn::GolombNet* network) {
  if (!network) {
    return evaluate_leaf_heuristic(st, target_n);
  }

  // Get value from network
  nn::Tensor policy_unused;
  double value;
  network->forward(st, policy_unused, value);

  // Network value is in [-1, 1] range (from tanh)
  // Scale to be comparable to heuristic (negative length)
  return value * NN_VALUE_SCALE;
}

// NOTE: PUCT selection formula
int select_action_puct(MCTSNode* node, double c_puct) {
  double sqrt_parent_n = std::sqrt(static_cast<double>(node->N));
  int best_action = -1;
  double best_value = -std::numeric_limits<double>::infinity();

  for (const auto& [action, child] : node->children) {
    double q = child->N > 0 ? child->W / child->N : 0.0;
    double u = c_puct * node->P[action] * sqrt_parent_n / (1.0 + child->N);
    double puct_value = q + u;

    if (puct_value > best_value) {
      best_value = puct_value;
      best_action = action;
    }
  }

  return best_action;
}

// Policy objects for generic MCTS implementation
struct HeuristicPolicy {
  double evaluate_leaf(const RuleState& st, int target_n) const {
    return evaluate_leaf_heuristic(st, target_n);
  }

  void compute_priors(MCTSNode* node, const std::vector<int>& actions) const {
    compute_policy_priors_uniform(node, actions);
  }
};

struct NNPolicy {
  nn::GolombNet* network;

  double evaluate_leaf(const RuleState& st, int target_n) const {
    return evaluate_leaf_nn(st, target_n, network);
  }

  void compute_priors(MCTSNode* node, const std::vector<int>& actions) const {
    compute_policy_priors_nn(node, actions, network);
  }
};

// NOTE: Generic MCTS simulation - unified implementation
template<typename Policy>
double simulate_impl(MCTSNode* node, int target_n, int ub, double c_puct, const Policy& policy,
                     RNG& rng) {
  // Terminal check
  if (static_cast<int>(node->state.marks.size()) >= target_n) {
    node->is_terminal = true;
    return policy.evaluate_leaf(node->state, target_n);
  }

  std::vector<int> actions = get_legal_actions(node->state, ub);
  if (actions.empty()) {
    node->is_terminal = true;
    return policy.evaluate_leaf(node->state, target_n);
  }

  // Expansion: if first visit, initialize children
  if (node->N == 0) {
    policy.compute_priors(node, actions);
    for (int a : actions) {
      auto child = std::make_unique<MCTSNode>(ub);
      child->state = node->state;
      try_add(child->state, a);
      node->children[a] = std::move(child);
    }
  }

  // Selection
  int action = select_action_puct(node, c_puct);
  if (action == -1 || !node->children.count(action)) {
    // Fallback: random action
    action = actions[rng.uniform_int(0, static_cast<int>(actions.size()) - 1)];
  }

  // Recursion
  double value = simulate_impl(node->children[action].get(), target_n, ub, c_puct, policy, rng);

  // Backpropagation
  node->N++;
  node->W += value;

  return value;
}

// NOTE: recursive MCTS simulation (heuristic version)
double simulate(MCTSNode* node, int target_n, int ub, double c_puct, RNG& rng) {
  HeuristicPolicy policy;
  return simulate_impl(node, target_n, ub, c_puct, policy, rng);
}

// NOTE: recursive MCTS simulation with neural network
double simulate_nn(MCTSNode* node, int target_n, int ub, double c_puct, nn::GolombNet* network,
                   RNG& rng) {
  NNPolicy policy{network};
  return simulate_impl(node, target_n, ub, c_puct, policy, rng);
}

// NOTE: extract best complete ruler from tree
std::vector<int> extract_best_rule(MCTSNode* root, int target_n) {
  MCTSNode* current = root;
  while (!current->children.empty()) {
    // Select most visited child
    MCTSNode* best_child = nullptr;
    int best_visits = -1;
    for (auto& [action, child] : current->children) {
      if (child->N > best_visits) {
        best_visits = child->N;
        best_child = child.get();
      }
    }
    if (!best_child) {
      break;
    }
    current = best_child;

    if (static_cast<int>(current->state.marks.size()) >= target_n) {
      return current->state.marks;
    }
  }
  return current->state.marks;
}

} // namespace

std::vector<int> mcts_build(int n, int ub, int iters, double c_puct) {
  auto root = std::make_unique<MCTSNode>(ub);
  root->state.marks.push_back(0);

  RNG rng;

  // Run MCTS iterations
  for (int iter = 0; iter < iters; ++iter) {
    simulate(root.get(), n, ub, c_puct, rng);
  }

  return extract_best_rule(root.get(), n);
}

std::vector<int> mcts_build_nn(int n, int ub, int iters, nn::GolombNet* network, double c_puct) {
  auto root = std::make_unique<MCTSNode>(ub);
  root->state.marks.push_back(0);

  RNG rng;

  // Run MCTS iterations with neural network guidance
  for (int iter = 0; iter < iters; ++iter) {
    simulate_nn(root.get(), n, ub, c_puct, network, rng);
  }

  return extract_best_rule(root.get(), n);
}

} // namespace golomb
