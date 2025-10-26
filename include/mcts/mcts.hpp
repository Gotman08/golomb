#pragma once

#include "core/golomb.hpp"
#include <memory>
#include <unordered_map>
#include <vector>

namespace golomb {

/**
 * @brief MCTS node representing a partial Golomb ruler state.
 *
 * Each node tracks visit count (N), total value (W), and prior probabilities (P).
 */
struct MCTSNode {
  RuleState state;                                        ///< Current partial ruler state.
  std::unordered_map<int, std::unique_ptr<MCTSNode>> children;  ///< Child nodes by action.
  int N = 0;                                              ///< Visit count.
  double W = 0.0;                                         ///< Total value accumulated.
  std::unordered_map<int, double> P;                     ///< Prior probabilities per action.
  bool is_terminal = false;                               ///< Whether state is complete/dead-end.

  explicit MCTSNode(int max_dist) : state(max_dist) {}
};

/**
 * @brief MCTS-based Golomb ruler construction using PUCT selection.
 *
 * Builds a ruler incrementally using Monte Carlo Tree Search with Upper Confidence
 * bounds applied to Trees (PUCT) for balancing exploration/exploitation.
 *
 * @param n Number of marks to place.
 * @param ub Upper bound for mark positions.
 * @param iters Number of MCTS iterations.
 * @param c_puct Exploration constant for PUCT formula.
 * @return Best ruler found (sorted marks).
 *
 * TODO: Integrate neural network for policy priors and value estimation.
 * TODO: Implement virtual loss for parallel MCTS.
 * TODO: Add Graphviz export for tree visualization.
 */
std::vector<int> mcts_build(int n, int ub, int iters, double c_puct = 1.4);

}  // namespace golomb
