#include "mcts/mcts.hpp"
#include "nn/golomb_net.hpp"
#include "utils/random.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace golomb {

namespace {

// Constants for MCTS evaluation
constexpr double INCOMPLETE_MARK_PENALTY = 1000.0;
constexpr double NN_VALUE_SCALE = 1000.0;

// Progressive widening parameters
constexpr double PW_C = 1.5;    // Constant multiplier for progressive widening
constexpr double PW_ALPHA = 0.5; // Exponent for progressive widening (sqrt)

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

// Calculate maximum number of children to expand using progressive widening
// Formula: k(n) = ⌊C * n^α⌋
int calculate_progressive_width(int visit_count) {
  if (visit_count == 0) {
    return 1; // Always expand at least one child on first visit
  }
  return static_cast<int>(PW_C * std::pow(static_cast<double>(visit_count), PW_ALPHA));
}

// Select top-k actions by prior probability
// Used for progressive widening to focus on promising actions first
std::vector<int> select_top_k_actions(const std::vector<int>& actions,
                                       const std::unordered_map<int, double>& priors, int k) {
  if (k >= static_cast<int>(actions.size())) {
    return actions; // Return all if k is large enough
  }

  // Sort actions by prior probability (descending)
  std::vector<std::pair<int, double>> action_priors;
  action_priors.reserve(actions.size());
  for (int a : actions) {
    auto it = priors.find(a);
    double prior = (it != priors.end()) ? it->second : 0.0;
    action_priors.emplace_back(a, prior);
  }

  // Partial sort to get top k
  std::partial_sort(action_priors.begin(), action_priors.begin() + k, action_priors.end(),
                    [](const auto& a, const auto& b) { return a.second > b.second; });

  // Extract top k actions
  std::vector<int> top_actions;
  top_actions.reserve(static_cast<size_t>(k));
  for (int i = 0; i < k; ++i) {
    top_actions.push_back(action_priors[i].first);
  }

  return top_actions;
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

// NOTE: PUCT selection formula with virtual loss
// Virtual loss penalizes nodes being explored by other threads
int select_action_puct_vloss(MCTSNode* node, double c_puct) {
  double sqrt_parent_n = std::sqrt(static_cast<double>(node->N));
  int best_action = -1;
  double best_value = -std::numeric_limits<double>::infinity();

  for (const auto& [action, child] : node->children) {
    // Adjust Q with virtual loss to discourage concurrent exploration
    double effective_n = child->N + child->virtual_loss;
    double q = effective_n > 0 ? (child->W - child->virtual_loss) / effective_n : 0.0;
    double u = c_puct * node->P[action] * sqrt_parent_n / (1.0 + effective_n);
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
template <typename Policy>
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

// NOTE: recursive MCTS simulation (heuristic version) - INTERNAL
// This is now just an internal wrapper, public API is mcts_simulate
double simulate(MCTSNode* node, int target_n, int ub, double c_puct, RNG& rng) {
  HeuristicPolicy policy;
  return simulate_impl(node, target_n, ub, c_puct, policy, rng);
}

// NOTE: MCTS simulation with progressive widening
// Limits action expansion based on visit count: k(n) = ⌊C * n^α⌋
template <typename Policy>
double simulate_pw_impl(MCTSNode* node, int target_n, int ub, double c_puct, const Policy& policy,
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

  // Expansion: on first visit or when we should widen
  if (node->N == 0) {
    // First visit: compute priors for all actions
    policy.compute_priors(node, actions);

    // Progressive widening: only expand k(n) best children
    int max_children = calculate_progressive_width(node->N);
    std::vector<int> actions_to_expand = select_top_k_actions(actions, node->P, max_children);

    // Create children for top-k actions only
    for (int a : actions_to_expand) {
      auto child = std::make_unique<MCTSNode>(ub);
      child->state = node->state;
      try_add(child->state, a);
      node->children[a] = std::move(child);
    }
  } else {
    // Check if we should expand more children based on visit count
    int max_children = calculate_progressive_width(node->N);
    if (static_cast<int>(node->children.size()) < max_children &&
        static_cast<int>(node->children.size()) < static_cast<int>(actions.size())) {
      // Expand one more child (the next best unvisited action)
      std::vector<int> unvisited_actions;
      for (int a : actions) {
        if (node->children.find(a) == node->children.end()) {
          unvisited_actions.push_back(a);
        }
      }

      if (!unvisited_actions.empty()) {
        // Select best unvisited action by prior
        auto best_action_it =
            std::max_element(unvisited_actions.begin(), unvisited_actions.end(),
                             [&](int a, int b) { return node->P[a] < node->P[b]; });

        int action_to_expand = *best_action_it;
        auto child = std::make_unique<MCTSNode>(ub);
        child->state = node->state;
        try_add(child->state, action_to_expand);
        node->children[action_to_expand] = std::move(child);
      }
    }
  }

  // Selection: choose from expanded children only
  int action = select_action_puct(node, c_puct);
  if (action == -1 || !node->children.count(action)) {
    // Fallback: select random from expanded children
    if (!node->children.empty()) {
      auto it = node->children.begin();
      std::advance(it, rng.uniform_int(0, static_cast<int>(node->children.size()) - 1));
      action = it->first;
    } else {
      // No children yet, shouldn't happen but handle gracefully
      return policy.evaluate_leaf(node->state, target_n);
    }
  }

  // Recursion
  double value = simulate_pw_impl(node->children[action].get(), target_n, ub, c_puct, policy, rng);

  // Backpropagation
  node->N++;
  node->W += value;

  return value;
}

// Wrapper for progressive widening MCTS with heuristic policy - INTERNAL
// This is now just an internal wrapper, public API is mcts_simulate_pw
double simulate_pw(MCTSNode* node, int target_n, int ub, double c_puct, RNG& rng) {
  HeuristicPolicy policy;
  return simulate_pw_impl(node, target_n, ub, c_puct, policy, rng);
}

// NOTE: recursive MCTS simulation with neural network - INTERNAL
// This is now just an internal wrapper, public API is mcts_simulate_nn
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

// NOTE: Thread-safe simulation with virtual loss for parallel MCTS
// Uses mutex to protect node updates and virtual loss management
template <typename Policy>
double simulate_parallel_impl(MCTSNode* node, int target_n, int ub, double c_puct,
                               double vloss_penalty, const Policy& policy, RNG& rng,
                               std::mutex& tree_mutex) {
  // Lock for reading/updating node state
  std::unique_lock<std::mutex> lock(tree_mutex);

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

  // Selection with virtual loss
  int action = select_action_puct_vloss(node, c_puct);
  if (action == -1 || !node->children.count(action)) {
    // Fallback: random action
    action = actions[rng.uniform_int(0, static_cast<int>(actions.size()) - 1)];
  }

  MCTSNode* child = node->children[action].get();

  // Apply virtual loss to discourage other threads
  child->virtual_loss += vloss_penalty;

  // Unlock before recursion to allow other threads to work
  lock.unlock();

  // Recursive simulation
  double value = simulate_parallel_impl(child, target_n, ub, c_puct, vloss_penalty, policy, rng,
                                        tree_mutex);

  // Relock for backpropagation
  lock.lock();

  // Remove virtual loss and backpropagate
  child->virtual_loss -= vloss_penalty;
  node->N++;
  node->W += value;

  return value;
}

// Wrapper for parallel simulation with heuristic policy
double simulate_parallel(MCTSNode* node, int target_n, int ub, double c_puct, double vloss_penalty,
                         RNG& rng, std::mutex& tree_mutex) {
  HeuristicPolicy policy;
  return simulate_parallel_impl(node, target_n, ub, c_puct, vloss_penalty, policy, rng,
                                tree_mutex);
}

} // namespace

// ============================================================================
// Public API: Single Iteration Functions (for training integration)
// ============================================================================

double mcts_simulate(MCTSNode* node, int target_n, int ub, double c_puct, RNG& rng) {
  return simulate(node, target_n, ub, c_puct, rng);
}

double mcts_simulate_nn(MCTSNode* node, int target_n, int ub, double c_puct,
                        nn::GolombNet* network, RNG& rng) {
  return simulate_nn(node, target_n, ub, c_puct, network, rng);
}

double mcts_simulate_pw(MCTSNode* node, int target_n, int ub, double c_puct, RNG& rng) {
  return simulate_pw(node, target_n, ub, c_puct, rng);
}

// ============================================================================
// High-Level MCTS Builders
// ============================================================================

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

std::vector<int> mcts_build_parallel(int n, int ub, int iters, int num_threads, double c_puct,
                                     double vloss_penalty) {
  // Shared root node for all threads
  auto root = std::make_unique<MCTSNode>(ub);
  root->state.marks.push_back(0);

  // Mutex to protect tree access
  std::mutex tree_mutex;

  // Atomic counter for iteration tracking
  std::atomic<int> iterations_done{0};

  // Worker function for each thread
  auto worker = [&]() {
    // Each thread has its own RNG with different seed
    RNG rng;

    while (true) {
      // Check if all iterations are done
      int current_iter = iterations_done.fetch_add(1, std::memory_order_relaxed);
      if (current_iter >= iters) {
        break;
      }

      // Run one simulation with virtual loss
      simulate_parallel(root.get(), n, ub, c_puct, vloss_penalty, rng, tree_mutex);
    }
  };

  // Launch worker threads
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(num_threads));
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(worker);
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // Extract best result from final tree
  return extract_best_rule(root.get(), n);
}

std::vector<int> mcts_build_pw(int n, int ub, int iters, double c_puct) {
  auto root = std::make_unique<MCTSNode>(ub);
  root->state.marks.push_back(0);

  RNG rng;

  // Run MCTS iterations with progressive widening
  for (int iter = 0; iter < iters; ++iter) {
    simulate_pw(root.get(), n, ub, c_puct, rng);
  }

  return extract_best_rule(root.get(), n);
}

void export_mcts_graphviz(const MCTSNode* root, const std::string& filename, int max_depth,
                          int min_visits) {
  std::ofstream file(filename);
  if (!file) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }

  file << "digraph MCTS {\n";
  file << "  rankdir=TB;\n";
  file << "  node [shape=box, style=rounded];\n";
  file << "\n";

  // Node ID counter
  int node_id = 0;

  // Helper: recursively export nodes
  std::function<void(const MCTSNode*, int, int, int)> export_node;
  export_node = [&](const MCTSNode* node, int id, int depth, int parent_action) {
    // Stop if max depth reached or too few visits
    if (depth > max_depth || node->N < min_visits) {
      return;
    }

    // Node label: show state, N, W/N
    std::ostringstream label;
    label << "State: [";
    for (size_t i = 0; i < node->state.marks.size(); ++i) {
      if (i > 0) label << ",";
      label << node->state.marks[i];
      if (i >= 5) { // Limit marks shown
        label << ",...";
        break;
      }
    }
    label << "]\\n";
    label << "N=" << node->N;

    if (node->N > 0) {
      double avg_value = node->W / node->N;
      label << "\\nV=" << std::fixed << std::setprecision(2) << avg_value;
    }

    // Color based on visit count
    std::string color = "white";
    if (node->N > 100) {
      color = "lightgreen";
    } else if (node->N > 10) {
      color = "lightyellow";
    }

    file << "  node" << id << " [label=\"" << label.str() << "\", fillcolor=" << color
         << ", style=filled];\n";

    // Export children
    for (const auto& [action, child] : node->children) {
      if (child->N < min_visits) {
        continue; // Skip rarely visited children
      }

      int child_id = ++node_id;

      // Edge label: show action and prior
      std::ostringstream edge_label;
      edge_label << "a=" << action;

      auto it = node->P.find(action);
      if (it != node->P.end()) {
        edge_label << "\\nP=" << std::fixed << std::setprecision(3) << it->second;
      }

      // Edge color based on child visits
      std::string edge_color = "black";
      double edge_penwidth = 1.0;
      if (child->N > 50) {
        edge_color = "darkgreen";
        edge_penwidth = 3.0;
      } else if (child->N > 10) {
        edge_color = "blue";
        edge_penwidth = 2.0;
      }

      file << "  node" << id << " -> node" << child_id << " [label=\"" << edge_label.str()
           << "\", color=" << edge_color << ", penwidth=" << edge_penwidth << "];\n";

      // Recursively export child
      export_node(child.get(), child_id, depth + 1, action);
    }
  };

  // Start from root
  export_node(root, node_id, 0, -1);

  file << "}\n";
  file.close();
}

} // namespace golomb
