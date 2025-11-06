#include "mcts/node_pool.hpp"
#include <stdexcept>

namespace golomb {

NodePool::NodePool(size_t capacity, int max_dist)
    : pool_(), free_list_(), capacity_(capacity), max_dist_(max_dist) {
  // Pre-allocate all nodes
  pool_.reserve(capacity);
  free_list_.reserve(capacity);

  for (size_t i = 0; i < capacity; ++i) {
    pool_.emplace_back(max_dist);
    free_list_.push_back(&pool_[i]);
  }
}

std::unique_ptr<MCTSNode, std::function<void(MCTSNode*)>> NodePool::allocate() {
  if (free_list_.empty()) {
    throw std::runtime_error("NodePool exhausted: increase capacity");
  }

  // Pop from free list
  MCTSNode* node = free_list_.back();
  free_list_.pop_back();

  // Reset node to clean state
  reset_node(node);

  // Return unique_ptr with custom deleter that returns node to pool
  auto deleter = [this](MCTSNode* n) { this->deallocate(n); };
  return std::unique_ptr<MCTSNode, std::function<void(MCTSNode*)>>(node, deleter);
}

void NodePool::deallocate(MCTSNode* node) {
  if (node == nullptr) {
    return;
  }

  // Return to free list
  free_list_.push_back(node);
}

void NodePool::reset_node(MCTSNode* node) {
  // Clear state
  node->state = RuleState(max_dist_);

  // Clear children (this releases their unique_ptrs)
  node->children.clear();

  // Reset counters (use store() for atomics)
  node->N.store(0, std::memory_order_relaxed);
  node->W.store(0.0, std::memory_order_relaxed);
  node->virtual_loss.store(0.0, std::memory_order_relaxed);

  // Clear prior probabilities
  node->P.clear();

  // Reset flags
  node->is_terminal = false;
  node->actions_cached = false;
  node->cached_legal_actions.clear();
}

} // namespace golomb
