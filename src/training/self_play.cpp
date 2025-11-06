#include "training/self_play.hpp"
#include "utils/random.hpp"
#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace golomb {
namespace training {

// ================================
// ReplayBuffer Implementation
// ================================

ReplayBuffer::ReplayBuffer(size_t max_size) : max_size_(max_size), next_idx_(0) {
  examples_.reserve(max_size);
}

void ReplayBuffer::add(const TrainingExample& example) {
  if (examples_.size() < max_size_) {
    // Buffer not full yet, just append
    examples_.push_back(example);
  } else {
    // Buffer full, overwrite oldest (circular buffer)
    examples_[next_idx_] = example;
    next_idx_ = (next_idx_ + 1) % max_size_;
  }
}

void ReplayBuffer::add_batch(const std::vector<TrainingExample>& examples) {
  for (const auto& example : examples) {
    add(example);
  }
}

std::vector<TrainingExample> ReplayBuffer::sample(size_t batch_size) const {
  if (batch_size > examples_.size()) {
    throw std::runtime_error("ReplayBuffer::sample: batch_size (" + std::to_string(batch_size) +
                             ") exceeds buffer size (" + std::to_string(examples_.size()) + ")");
  }

  // Random sampling without replacement
  RNG rng;
  std::vector<TrainingExample> batch;
  batch.reserve(batch_size);

  // Create indices and shuffle
  std::vector<size_t> indices(examples_.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    indices[i] = i;
  }

  // Fisher-Yates shuffle (sample first batch_size elements)
  for (size_t i = 0; i < batch_size; ++i) {
    size_t j = i + rng.uniform_int(0, static_cast<int>(indices.size() - i - 1));
    std::swap(indices[i], indices[j]);
  }

  // Extract sampled examples
  for (size_t i = 0; i < batch_size; ++i) {
    batch.push_back(examples_[indices[i]]);
  }

  return batch;
}

void ReplayBuffer::clear() {
  examples_.clear();
  next_idx_ = 0;
}

// ================================
// SelfPlayGenerator Implementation
// ================================

SelfPlayGenerator::SelfPlayGenerator(nn::GolombNet* network, int n, int ub, int mcts_iters,
                                     double c_puct)
    : network_(network), n_(n), ub_(ub), mcts_iters_(mcts_iters), c_puct_(c_puct),
      temperature_(1.0) {}

nn::Tensor SelfPlayGenerator::extract_mcts_policy(const MCTSNode* root) const {
  // Create policy tensor (size = ub+1, one slot per possible position)
  nn::Tensor policy(static_cast<size_t>(ub_ + 1));
  policy.zeros();

  // Sum all visit counts
  int total_visits = 0;
  for (const auto& [action, child] : root->children) {
    total_visits += child->N;
  }

  if (total_visits == 0) {
    // No visits (shouldn't happen), return uniform
    double uniform_prob = root->children.empty() ? 0.0 : 1.0 / root->children.size();
    for (const auto& [action, child] : root->children) {
      policy(static_cast<size_t>(action)) = uniform_prob;
    }
  } else {
    // Policy proportional to visit counts (with temperature)
    for (const auto& [action, child] : root->children) {
      double visit_ratio = static_cast<double>(child->N) / total_visits;

      // Apply temperature: visits^(1/τ)
      if (temperature_ != 1.0) {
        visit_ratio = std::pow(visit_ratio, 1.0 / temperature_);
      }

      policy(static_cast<size_t>(action)) = visit_ratio;
    }

    // Renormalize if temperature was applied
    if (temperature_ != 1.0) {
      double sum = policy.sum();
      if (sum > 0.0) {
        policy *= (1.0 / sum);
      }
    }
  }

  return policy;
}

int SelfPlayGenerator::select_action_with_temperature(const nn::Tensor& policy) const {
  RNG rng;

  // Sample from categorical distribution
  double rand_val = rng.uniform_double(0.0, 1.0);
  double cumsum = 0.0;

  for (size_t i = 0; i < policy.size(); ++i) {
    cumsum += policy(i);
    if (rand_val <= cumsum) {
      return static_cast<int>(i);
    }
  }

  // Shouldn't reach here, but return last action if rounding issues
  for (int i = static_cast<int>(policy.size()) - 1; i >= 0; --i) {
    if (policy(static_cast<size_t>(i)) > 0.0) {
      return i;
    }
  }

  return 0; // Fallback
}

std::vector<TrainingExample> SelfPlayGenerator::generate_game() {
  std::vector<TrainingExample> examples;

  // Initialize game state
  RuleState current_state(ub_);
  current_state.marks.push_back(0); // First mark always at 0

  // Play until we have n marks
  while (static_cast<int>(current_state.marks.size()) < n_) {
    // Run MCTS from current state
    auto root = std::make_unique<MCTSNode>(ub_);
    root->state = current_state;

    // Perform MCTS iterations
    RNG rng; // Create RNG once per move (not per iteration)
    for (int iter = 0; iter < mcts_iters_; ++iter) {
      // Use neural network guided MCTS if available, fallback to heuristic
      if (network_) {
        mcts_simulate_nn(root.get(), n_, ub_, c_puct_, network_, rng);
      } else {
        mcts_simulate(root.get(), n_, ub_, c_puct_, rng);
      }
    }

    // Extract MCTS policy from visit counts
    nn::Tensor mcts_policy = extract_mcts_policy(root.get());

    // Select action according to temperature
    int action = select_action_with_temperature(mcts_policy);

    // Record training example (state, policy, value will be filled later)
    examples.emplace_back(current_state, mcts_policy, 0.0);

    // Apply action to state
    if (!try_add(current_state, action)) {
      // Invalid action selected (shouldn't happen with proper MCTS)
      // Try to recover by selecting a valid action
      for (int p = 1; p < ub_; ++p) {
        if (try_add(current_state, p)) {
          break;
        }
      }
    }

    // Break if state is stuck
    if (static_cast<int>(current_state.marks.size()) >= n_) {
      break;
    }
  }

  // Evaluate final outcome
  double final_value = -static_cast<double>(length(current_state.marks));

  // Assign final value to all examples in this game
  for (auto& example : examples) {
    example.value_target = final_value;
  }

  return examples;
}

std::vector<TrainingExample> SelfPlayGenerator::generate_games_parallel(int num_games,
                                                                         int num_threads) {
  std::vector<TrainingExample> all_examples;
  std::mutex examples_mutex;

  // Atomic counter for games completed
  std::atomic<int> games_completed{0};

  // Worker function
  auto worker = [&]() {
    while (true) {
      int game_id = games_completed.fetch_add(1, std::memory_order_relaxed);
      if (game_id >= num_games) {
        break;
      }

      // Generate one game
      std::vector<TrainingExample> game_examples = generate_game();

      // Add to shared collection (thread-safe)
      {
        std::lock_guard<std::mutex> lock(examples_mutex);
        all_examples.insert(all_examples.end(), game_examples.begin(), game_examples.end());
      }
    }
  };

  // Launch threads
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(num_threads));
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(worker);
  }

  // Wait for completion
  for (auto& thread : threads) {
    thread.join();
  }

  return all_examples;
}

} // namespace training
} // namespace golomb
