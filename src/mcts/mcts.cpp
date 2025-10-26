#include "mcts/mcts.hpp"
#include "utils/random.hpp"
#include <cmath>
#include <limits>

namespace golomb {

namespace {

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

// TODO: replace with neural network policy
void compute_policy_priors_uniform(MCTSNode* node, const std::vector<int>& actions) {
  double uniform_prob = actions.empty() ? 0.0 : 1.0 / actions.size();
  for (int a : actions) {
    node->P[a] = uniform_prob;
  }
}

// TODO: replace with neural network value
double evaluate_leaf(const RuleState& st, int target_n) {
  // Simple heuristic: prefer states with more marks placed and shorter length
  int marks_placed = static_cast<int>(st.marks.size());
  int len = st.marks.empty() ? 0 : st.marks.back();

  if (marks_placed < target_n) {
    return -len - 1000.0 * (target_n - marks_placed);
  }
  return -len;
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

// NOTE: recursive MCTS simulation
double simulate(MCTSNode* node, int target_n, int ub, double c_puct, RNG& rng) {
  // Terminal check
  if (static_cast<int>(node->state.marks.size()) >= target_n) {
    node->is_terminal = true;
    return evaluate_leaf(node->state, target_n);
  }

  std::vector<int> actions = get_legal_actions(node->state, ub);
  if (actions.empty()) {
    node->is_terminal = true;
    return evaluate_leaf(node->state, target_n);
  }

  // Expansion: if first visit, initialize children
  if (node->N == 0) {
    compute_policy_priors_uniform(node, actions);
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
  double value = simulate(node->children[action].get(), target_n, ub, c_puct, rng);

  // Backpropagation
  node->N++;
  node->W += value;

  return value;
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

}  // namespace

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

}  // namespace golomb
