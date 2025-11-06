#pragma once

#include "nn/golomb_net.hpp"
#include "nn/optimizer.hpp"
#include "training/self_play.hpp"
#include <memory>
#include <string>
#include <vector>

namespace golomb {
namespace training {

/**
 * @brief Training configuration parameters.
 */
struct TrainingConfig {
  // Self-play parameters
  int n = 8;                         ///< Number of marks in Golomb ruler.
  int ub = 120;                      ///< Upper bound for positions.
  int games_per_iteration = 100;    ///< Self-play games per training iteration.
  int mcts_iters_per_move = 400;    ///< MCTS iterations per move.
  double c_puct = 1.4;               ///< PUCT exploration constant.
  double temperature = 1.0;          ///< Temperature for move selection.

  // Training parameters
  size_t batch_size = 64;            ///< Mini-batch size.
  int training_steps_per_iter = 100; ///< Training steps per iteration.
  double learning_rate = 0.001;      ///< Learning rate.
  double weight_decay = 1e-4;        ///< L2 regularization coefficient.

  // Optimizer
  std::string optimizer_type = "adam"; ///< Optimizer type ("sgd" or "adam").
  double momentum = 0.9;               ///< Momentum for SGD.

  // Replay buffer
  size_t replay_buffer_size = 10000; ///< Maximum replay buffer capacity.

  // Checkpointing
  int checkpoint_interval = 10;    ///< Save checkpoint every N iterations.
  std::string checkpoint_dir = "checkpoints"; ///< Directory for checkpoints.

  // Evaluation
  int eval_interval = 5;           ///< Evaluate every N iterations.
  int eval_games = 20;             ///< Number of games for evaluation.
};

/**
 * @brief Training statistics for one iteration.
 */
struct TrainingStats {
  int iteration = 0;             ///< Iteration number.
  double policy_loss = 0.0;      ///< Average policy loss.
  double value_loss = 0.0;       ///< Average value loss.
  double total_loss = 0.0;       ///< Average total loss.
  double avg_ruler_length = 0.0; ///< Average ruler length from self-play.
  int training_examples = 0;     ///< Number of training examples collected.
};

/**
 * @brief Neural network trainer for Golomb ruler optimization.
 *
 * Implements AlphaZero-style training loop:
 * 1. Generate self-play games using current network
 * 2. Add examples to replay buffer
 * 3. Sample mini-batches and train network
 * 4. Periodically save checkpoints and evaluate
 *
 * Loss function: L = (z - v)² - π^T log(p) + λ||θ||²
 * where:
 * - z: target value (game outcome)
 * - v: network value prediction
 * - π: MCTS policy (visit distribution)
 * - p: network policy prediction
 * - λ: weight decay coefficient
 */
class Trainer {
public:
  /**
   * @brief Construct trainer.
   *
   * @param network Neural network to train.
   * @param config Training configuration.
   */
  Trainer(nn::GolombNet* network, const TrainingConfig& config);

  /**
   * @brief Run one training iteration.
   *
   * Performs:
   * 1. Self-play game generation
   * 2. Replay buffer update
   * 3. Multiple training steps on sampled batches
   *
   * @return Training statistics for this iteration.
   */
  TrainingStats train_iteration();

  /**
   * @brief Run full training loop for multiple iterations.
   *
   * @param num_iterations Number of training iterations.
   */
  void train(int num_iterations);

  /**
   * @brief Save checkpoint to disk.
   *
   * Saves network parameters and training state.
   *
   * @param iteration Current iteration number.
   * @param filepath Checkpoint file path.
   */
  void save_checkpoint(int iteration, const std::string& filepath);

  /**
   * @brief Load checkpoint from disk.
   *
   * Restores network parameters and training state.
   *
   * @param filepath Checkpoint file path.
   * @return Iteration number from checkpoint.
   */
  int load_checkpoint(const std::string& filepath);

  /**
   * @brief Evaluate current network performance.
   *
   * Generates evaluation games and computes average ruler length.
   *
   * @param num_games Number of evaluation games.
   * @return Average ruler length achieved.
   */
  double evaluate(int num_games);

  /**
   * @brief Get replay buffer.
   * @return Reference to replay buffer.
   */
  [[nodiscard]] const ReplayBuffer& get_replay_buffer() const { return replay_buffer_; }

  /**
   * @brief Get training statistics history.
   * @return Vector of statistics for each iteration.
   */
  [[nodiscard]] const std::vector<TrainingStats>& get_stats() const { return stats_history_; }

private:
  nn::GolombNet* network_;          ///< Neural network to train.
  TrainingConfig config_;           ///< Training configuration.
  ReplayBuffer replay_buffer_;      ///< Replay buffer for training examples.
  std::unique_ptr<nn::Optimizer> optimizer_; ///< Optimizer for parameter updates.
  SelfPlayGenerator self_play_gen_; ///< Self-play game generator.
  std::vector<TrainingStats> stats_history_; ///< Training statistics history.
  int current_iteration_;           ///< Current iteration number.

  /**
   * @brief Compute loss for a single training example.
   *
   * Loss = MSE(value) + CrossEntropy(policy)
   *
   * @param example Training example.
   * @param pred_policy Network policy prediction.
   * @param pred_value Network value prediction.
   * @param grad_policy Output: gradient w.r.t. policy.
   * @param grad_value Output: gradient w.r.t. value.
   * @return Total loss value.
   */
  double compute_loss_and_gradients(const TrainingExample& example, const nn::Tensor& pred_policy,
                                     double pred_value, nn::Tensor& grad_policy,
                                     double& grad_value);

  /**
   * @brief Train on one mini-batch.
   *
   * @param batch Training examples batch.
   * @return Batch loss (policy, value, total).
   */
  std::tuple<double, double, double> train_batch(const std::vector<TrainingExample>& batch);
};

} // namespace training
} // namespace golomb
