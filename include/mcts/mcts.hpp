#pragma once

#include "core/golomb.hpp"
#include <memory>
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
 */
struct MCTSNode {
  RuleState state;                                             ///< Current partial ruler state.
  std::unordered_map<int, std::unique_ptr<MCTSNode>> children; ///< Child nodes by action.
  int N = 0;                                                   ///< Visit count.
  double W = 0.0;                                              ///< Total value accumulated.
  std::unordered_map<int, double> P;                           ///< Prior probabilities per action.
  bool is_terminal = false; ///< Whether state is complete/dead-end.

  explicit MCTSNode(int max_dist) : state(max_dist) {}
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
 *
 * TODO: Implement virtual loss for parallel MCTS.
 * TODO: Add Graphviz export for tree visualization.
 */
std::vector<int> mcts_build(int n, int ub, int iters, double c_puct = 1.4);

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

} // namespace golomb
