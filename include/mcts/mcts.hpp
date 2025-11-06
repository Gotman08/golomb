#pragma once

#include "core/golomb.hpp"
#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace golomb {

// Forward declaration
namespace nn {
class GolombNet;
}

/**
 * @brief MCTS node representing a partial Golomb ruler state.
 *
 * Each node tracks visit count (N), total value (W), and prior probabilities (P).
 * Virtual loss is used for parallel MCTS to prevent multiple threads from exploring
 * the same path simultaneously.
 *
 * OPT-3B: Cache-line alignment and atomics to prevent false sharing (CSAPP 6.6)
 */
struct MCTSNode {
  RuleState state;                                             ///< Current partial ruler state.
  std::unordered_map<int, std::unique_ptr<MCTSNode>> children; ///< Child nodes by action.

  // OPT-3B: Atomic counters on separate cache lines to prevent false sharing (CSAPP 6.6)
  // Cache line = 64 bytes, ensure these hot variables don't share lines with other data
  alignas(64) std::atomic<int> N;                              ///< Visit count (atomic for parallel access).
  alignas(64) std::atomic<double> W;                           ///< Total value accumulated (atomic).
  alignas(64) std::atomic<double> virtual_loss;                ///< Virtual loss for parallel MCTS (atomic).

  std::unordered_map<int, double> P;                           ///< Prior probabilities per action.
  bool is_terminal = false;                                    ///< Whether state is complete/dead-end.

  // OPT-2B: Cache legal actions to avoid recomputation (CSAPP 5.8 - Memory Performance)
  std::vector<int> cached_legal_actions;                       ///< Cached legal actions for this state.
  bool actions_cached = false;                                 ///< Whether legal actions have been computed.

  explicit MCTSNode(int max_dist) : state(max_dist), N(0), W(0.0), virtual_loss(0.0) {}
};

/**
 * @brief MCTS-based Golomb ruler construction using PUCT selection.
 *
 * Builds a ruler incrementally using Monte Carlo Tree Search with Upper Confidence
 * bounds applied to Trees (PUCT) for balancing exploration/exploitation.
 *
 * This version uses uniform policy priors and a simple heuristic for leaf evaluation.
 *
 * @param n Number of marks to place.
 * @param ub Upper bound for mark positions.
 * @param iters Number of MCTS iterations.
 * @param c_puct Exploration constant for PUCT formula.
 * @return Best ruler found (sorted marks).
 */
std::vector<int> mcts_build(int n, int ub, int iters, double c_puct = 1.4);

/**
 * @brief Parallel MCTS with virtual loss.
 *
 * Uses multiple threads to explore the search tree in parallel. Virtual loss
 * prevents multiple threads from exploring the same path simultaneously.
 *
 * Virtual loss mechanism:
 * - When a thread selects a node, it temporarily adds a virtual loss penalty
 * - This discourages other threads from selecting the same node
 * - After backpropagation, the virtual loss is removed
 * - This improves parallel efficiency and exploration diversity
 *
 * @param n Number of marks to place.
 * @param ub Upper bound for mark positions.
 * @param iters Total number of MCTS iterations across all threads.
 * @param num_threads Number of parallel worker threads.
 * @param c_puct Exploration constant for PUCT formula.
 * @param vloss_penalty Virtual loss penalty value (default: 1.0).
 * @return Best ruler found (sorted marks).
 */
std::vector<int> mcts_build_parallel(int n, int ub, int iters, int num_threads,
                                     double c_puct = 1.4, double vloss_penalty = 1.0);

/**
 * @brief MCTS with progressive widening.
 *
 * Uses progressive widening to gradually increase the number of explored actions
 * as nodes are visited more frequently. This focuses exploration on the most
 * promising actions first.
 *
 * Progressive widening formula: k(n) = ⌊C * n^α⌋
 * where:
 * - n is the visit count of the node
 * - C = 1.5 (constant multiplier)
 * - α = 0.5 (exponent, typically sqrt)
 *
 * Benefits:
 * - Reduces branching factor early in search
 * - Focuses on high-prior actions first
 * - Improves convergence speed for large action spaces
 *
 * @param n Number of marks to place.
 * @param ub Upper bound for mark positions.
 * @param iters Number of MCTS iterations.
 * @param c_puct Exploration constant for PUCT formula.
 * @return Best ruler found (sorted marks).
 */
std::vector<int> mcts_build_pw(int n, int ub, int iters, double c_puct = 1.4);

/**
 * @brief MCTS with neural network guidance (AlphaGo-style).
 *
 * Uses a trained neural network to provide:
 * - Policy priors P(a|s) for action selection
 * - Value estimates V(s) for leaf evaluation
 *
 * If network is nullptr, falls back to uniform priors and heuristic evaluation.
 *
 * @param n Number of marks to place.
 * @param ub Upper bound for mark positions.
 * @param iters Number of MCTS iterations.
 * @param network Neural network (can be nullptr for pure MCTS).
 * @param c_puct Exploration constant for PUCT formula.
 * @return Best ruler found (sorted marks).
 */
std::vector<int> mcts_build_nn(int n, int ub, int iters, nn::GolombNet* network,
                               double c_puct = 1.4);

/**
 * @brief Export MCTS tree to Graphviz DOT format for visualization.
 *
 * Generates a DOT file that can be rendered with Graphviz tools.
 * Each node shows:
 * - State (marks placed so far)
 * - Visit count (N)
 * - Average value (W/N)
 * - Prior probability (P)
 *
 * Usage:
 * 1. Run MCTS and keep root node
 * 2. Export to DOT file: export_mcts_graphviz(root, "tree.dot")
 * 3. Render with Graphviz: dot -Tpng tree.dot -o tree.png
 *
 * Options:
 * - max_depth: Limit tree depth to avoid large files
 * - min_visits: Only show nodes with N >= min_visits
 *
 * @param root Root node of MCTS tree.
 * @param filename Output DOT file path.
 * @param max_depth Maximum depth to export (default 5).
 * @param min_visits Minimum visits to include node (default 1).
 */
void export_mcts_graphviz(const MCTSNode* root, const std::string& filename, int max_depth = 5,
                          int min_visits = 1);

// ============================================================================
// Single Iteration Functions (for training integration)
// ============================================================================

// Forward declaration of RNG
class RNG;

/**
 * @brief Single MCTS simulation iteration (heuristic policy).
 *
 * Performs one MCTS simulation from the given node using uniform priors
 * and heuristic leaf evaluation. Used for incremental tree building in
 * training scenarios.
 *
 * @param node Current root node to simulate from.
 * @param target_n Target number of marks to place.
 * @param ub Upper bound for mark positions.
 * @param c_puct Exploration constant for PUCT formula.
 * @param rng Random number generator.
 * @return Leaf value from this simulation.
 */
double mcts_simulate(MCTSNode* node, int target_n, int ub, double c_puct, RNG& rng);

/**
 * @brief Single MCTS simulation iteration with neural network guidance.
 *
 * Performs one MCTS simulation using neural network for policy priors
 * and value estimation. This is the key function for AlphaZero-style
 * self-play training.
 *
 * @param node Current root node to simulate from.
 * @param target_n Target number of marks to place.
 * @param ub Upper bound for mark positions.
 * @param c_puct Exploration constant for PUCT formula.
 * @param network Neural network (can be nullptr for heuristic fallback).
 * @param rng Random number generator.
 * @return Leaf value from this simulation.
 */
double mcts_simulate_nn(MCTSNode* node, int target_n, int ub, double c_puct,
                        nn::GolombNet* network, RNG& rng);

/**
 * @brief Single MCTS simulation iteration with progressive widening.
 *
 * Performs one MCTS simulation using progressive widening to limit
 * action expansion. Useful for large action spaces.
 *
 * @param node Current root node to simulate from.
 * @param target_n Target number of marks to place.
 * @param ub Upper bound for mark positions.
 * @param c_puct Exploration constant for PUCT formula.
 * @param rng Random number generator.
 * @return Leaf value from this simulation.
 */
double mcts_simulate_pw(MCTSNode* node, int target_n, int ub, double c_puct, RNG& rng);

} // namespace golomb
