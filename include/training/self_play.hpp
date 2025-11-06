#pragma once

#include "core/golomb.hpp"
#include "mcts/mcts.hpp"
#include "nn/golomb_net.hpp"
#include "nn/tensor.hpp"
#include <vector>

namespace golomb {
namespace training {

/**
 * @brief Single training example from self-play.
 *
 * Represents one state-action-value triple collected during MCTS self-play.
 */
struct TrainingExample {
  RuleState state;           ///< Game state.
  nn::Tensor policy_target;  ///< Target policy (MCTS visit distribution).
  double value_target;       ///< Target value (game outcome: -length or reward).

  /**
   * @brief Construct training example.
   * @param st Game state.
   * @param policy MCTS policy (visit distribution over actions).
   * @param value Game outcome or value estimate.
   */
  TrainingExample(const RuleState& st, const nn::Tensor& policy, double value)
      : state(st), policy_target(policy), value_target(value) {}
};

/**
 * @brief Replay buffer for storing training examples.
 *
 * Implements a circular buffer with maximum capacity. When full, oldest
 * examples are discarded (FIFO).
 */
class ReplayBuffer {
public:
  /**
   * @brief Construct replay buffer with maximum capacity.
   * @param max_size Maximum number of examples to store.
   */
  explicit ReplayBuffer(size_t max_size);

  /**
   * @brief Add training example to buffer.
   *
   * If buffer is full, removes oldest example first.
   *
   * @param example Training example to add.
   */
  void add(const TrainingExample& example);

  /**
   * @brief Add multiple training examples.
   * @param examples Vector of training examples.
   */
  void add_batch(const std::vector<TrainingExample>& examples);

  /**
   * @brief Sample random batch of examples for training.
   *
   * @param batch_size Number of examples to sample.
   * @return Vector of randomly sampled examples.
   * @throws std::runtime_error if buffer has fewer than batch_size examples.
   */
  std::vector<TrainingExample> sample(size_t batch_size) const;

  /**
   * @brief Get current buffer size.
   * @return Number of examples currently stored.
   */
  [[nodiscard]] size_t size() const { return examples_.size(); }

  /**
   * @brief Check if buffer is empty.
   * @return True if no examples stored.
   */
  [[nodiscard]] bool empty() const { return examples_.empty(); }

  /**
   * @brief Get maximum capacity.
   * @return Maximum buffer size.
   */
  [[nodiscard]] size_t capacity() const { return max_size_; }

  /**
   * @brief Clear all examples from buffer.
   */
  void clear();

private:
  std::vector<TrainingExample> examples_; ///< Stored examples.
  size_t max_size_;                       ///< Maximum buffer capacity.
  size_t next_idx_;                       ///< Next index for circular buffer.
};

/**
 * @brief Self-play game generator for neural network training.
 *
 * Generates games by playing MCTS against itself using the current network.
 * Collects training examples (state, MCTS policy, outcome) for supervised learning.
 */
class SelfPlayGenerator {
public:
  /**
   * @brief Construct self-play generator.
   *
   * @param network Neural network to use for policy/value guidance.
   * @param n Number of marks in Golomb ruler.
   * @param ub Upper bound for mark positions.
   * @param mcts_iters MCTS iterations per move (default 400).
   * @param c_puct PUCT exploration constant (default 1.4).
   */
  SelfPlayGenerator(nn::GolombNet* network, int n, int ub, int mcts_iters = 400,
                    double c_puct = 1.4);

  /**
   * @brief Generate one self-play game and collect training examples.
   *
   * Plays a complete game using MCTS with neural network guidance.
   * At each step, records (state, MCTS_policy, final_value).
   *
   * @return Vector of training examples from one game.
   */
  std::vector<TrainingExample> generate_game();

  /**
   * @brief Generate multiple self-play games in parallel.
   *
   * Uses multiple threads to generate games concurrently for efficiency.
   *
   * @param num_games Number of games to generate.
   * @param num_threads Number of parallel worker threads (default 4).
   * @return Vector of all training examples from all games.
   */
  std::vector<TrainingExample> generate_games_parallel(int num_games, int num_threads = 4);

  /**
   * @brief Set MCTS temperature for action selection.
   *
   * Temperature controls exploration during move selection:
   * - τ = 1: Select proportional to visit counts (stochastic)
   * - τ → 0: Select most-visited action (greedy)
   * - τ > 1: More random exploration
   *
   * @param temperature Temperature value (typically 1.0).
   */
  void set_temperature(double temperature) { temperature_ = temperature; }

  /**
   * @brief Get current temperature.
   * @return Temperature value.
   */
  [[nodiscard]] double get_temperature() const { return temperature_; }

private:
  nn::GolombNet* network_;  ///< Neural network for guidance.
  int n_;                   ///< Number of marks.
  int ub_;                  ///< Upper bound.
  int mcts_iters_;          ///< MCTS iterations per move.
  double c_puct_;           ///< PUCT constant.
  double temperature_;      ///< Temperature for move selection.

  /**
   * @brief Extract MCTS policy from root node visit counts.
   *
   * Converts visit counts to probability distribution over actions.
   *
   * @param root MCTS root node after search.
   * @return Policy tensor with probabilities for each action.
   */
  nn::Tensor extract_mcts_policy(const MCTSNode* root) const;

  /**
   * @brief Select action from MCTS policy using temperature.
   *
   * @param policy MCTS policy probabilities.
   * @return Selected action index.
   */
  int select_action_with_temperature(const nn::Tensor& policy) const;
};

} // namespace training
} // namespace golomb
