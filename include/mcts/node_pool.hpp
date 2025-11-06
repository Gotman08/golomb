#pragma once

#include "mcts/mcts.hpp"
#include <memory>
#include <vector>

namespace golomb {

/**
 * @brief Object pool for MCTS nodes (OPT-4C: CSAPP 9.9 - Reduce allocation overhead).
 *
 * Pre-allocates a large pool of MCTSNode objects to avoid repeated allocations.
 * Uses a free list for O(1) allocation/deallocation.
 *
 * Thread-safety: Each thread should use its own pool (thread_local).
 */
class NodePool {
public:
  /**
   * @brief Construct a node pool with given capacity.
   * @param capacity Number of nodes to pre-allocate.
   * @param max_dist Maximum distance for ruler (passed to MCTSNode constructor).
   */
  explicit NodePool(size_t capacity, int max_dist);

  /**
   * @brief Allocate a node from the pool.
   * @return Unique pointer to allocated node (uses custom deleter to return to pool).
   * @throws std::runtime_error if pool exhausted.
   */
  std::unique_ptr<MCTSNode, std::function<void(MCTSNode*)>> allocate();

  /**
   * @brief Check if pool has available nodes.
   * @return True if nodes available.
   */
  [[nodiscard]] bool has_available() const { return !free_list_.empty(); }

  /**
   * @brief Get number of nodes in use.
   * @return Number of allocated nodes.
   */
  [[nodiscard]] size_t in_use() const { return capacity_ - free_list_.size(); }

  /**
   * @brief Get total capacity.
   * @return Total pool capacity.
   */
  [[nodiscard]] size_t capacity() const { return capacity_; }

private:
  /**
   * @brief Return a node to the pool (called by unique_ptr deleter).
   * @param node Node to return.
   */
  void deallocate(MCTSNode* node);

  /**
   * @brief Reset a node to initial state.
   * @param node Node to reset.
   */
  void reset_node(MCTSNode* node);

  std::vector<MCTSNode> pool_;  ///< Pre-allocated node storage.
  std::vector<MCTSNode*> free_list_; ///< Free list of available nodes.
  size_t capacity_;             ///< Total capacity.
  int max_dist_;                ///< Maximum distance for MCTSNode constructor.
};

} // namespace golomb
