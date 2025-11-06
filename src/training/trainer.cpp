#include "training/trainer.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace golomb {
namespace training {

Trainer::Trainer(nn::GolombNet* network, const TrainingConfig& config)
    : network_(network),
      config_(config),
      replay_buffer_(config.replay_buffer_size),
      self_play_gen_(network, config.n, config.ub, config.mcts_iters_per_move, config.c_puct),
      current_iteration_(0) {
  // Initialize optimizer
  auto params = network_->parameters();
  auto grads = network_->gradients();

  if (config_.optimizer_type == "sgd") {
    optimizer_ = std::make_unique<nn::SGD>(params, grads, config_.learning_rate, config_.momentum,
                                            config_.weight_decay);
  } else if (config_.optimizer_type == "adam") {
    optimizer_ = std::make_unique<nn::Adam>(params, grads, config_.learning_rate, 0.9, 0.999,
                                             1e-8, config_.weight_decay);
  } else {
    throw std::invalid_argument("Unknown optimizer type: " + config_.optimizer_type);
  }

  // Set temperature for self-play
  self_play_gen_.set_temperature(config_.temperature);

  // Create checkpoint directory
  std::filesystem::create_directories(config_.checkpoint_dir);
}

double Trainer::compute_loss_and_gradients(const TrainingExample& example,
                                            const nn::Tensor& pred_policy, double pred_value,
                                            nn::Tensor& grad_policy, double& grad_value) {
  // Value loss: MSE = (target - prediction)²
  double value_error = example.value_target - pred_value;
  double value_loss = value_error * value_error;

  // Value gradient: ∂MSE/∂v = -2(target - prediction)
  grad_value = -2.0 * value_error;

  // Policy loss: Cross-entropy = -Σ target * log(prediction)
  double policy_loss = 0.0;
  grad_policy = nn::Tensor(pred_policy.shape());
  grad_policy.zeros();

  double epsilon = 1e-8; // For numerical stability

  for (size_t i = 0; i < pred_policy.size(); ++i) {
    double target = example.policy_target(i);
    double pred = std::max(pred_policy(i), epsilon); // Avoid log(0)

    if (target > 0.0) {
      policy_loss -= target * std::log(pred);

      // Gradient: ∂CE/∂p = -target / prediction
      grad_policy(i) = -target / pred;
    }
  }

  // Total loss
  double total_loss = value_loss + policy_loss;

  return total_loss;
}

std::tuple<double, double, double> Trainer::train_batch(const std::vector<TrainingExample>& batch) {
  if (batch.empty()) {
    return {0.0, 0.0, 0.0};
  }

  double total_policy_loss = 0.0;
  double total_value_loss = 0.0;

  // Zero gradients
  network_->zero_grad();

  // Accumulate gradients over batch
  for (const auto& example : batch) {
    // Forward pass
    nn::Tensor pred_policy;
    double pred_value;
    network_->forward(example.state, pred_policy, pred_value);

    // Compute loss and gradients
    nn::Tensor grad_policy;
    double grad_value;
    double loss =
        compute_loss_and_gradients(example, pred_policy, pred_value, grad_policy, grad_value);

    // Backward pass (accumulate gradients)
    network_->backward(grad_policy, grad_value);

    // Track losses
    double value_error = example.value_target - pred_value;
    total_value_loss += value_error * value_error;

    double policy_loss = 0.0;
    double epsilon = 1e-8;
    for (size_t i = 0; i < pred_policy.size(); ++i) {
      double target = example.policy_target(i);
      if (target > 0.0) {
        double pred = std::max(pred_policy(i), epsilon);
        policy_loss -= target * std::log(pred);
      }
    }
    total_policy_loss += policy_loss;
  }

  // Average losses
  double avg_policy_loss = total_policy_loss / batch.size();
  double avg_value_loss = total_value_loss / batch.size();
  double avg_total_loss = avg_policy_loss + avg_value_loss;

  // Gradient averaging (divide by batch size)
  // Note: Gradients are accumulated, so we need to scale them
  auto grads = network_->gradients();
  for (auto* grad : grads) {
    *grad *= (1.0 / static_cast<double>(batch.size()));
  }

  // Optimizer step
  optimizer_->step();

  return {avg_policy_loss, avg_value_loss, avg_total_loss};
}

TrainingStats Trainer::train_iteration() {
  TrainingStats stats;
  stats.iteration = ++current_iteration_;

  std::cout << "\n=== Iteration " << stats.iteration << " ===\n";

  // 1. Generate self-play games
  std::cout << "Generating " << config_.games_per_iteration << " self-play games...\n";
  auto examples = self_play_gen_.generate_games_parallel(config_.games_per_iteration, 4);

  std::cout << "Collected " << examples.size() << " training examples\n";
  stats.training_examples = static_cast<int>(examples.size());

  // Calculate average ruler length from self-play
  double total_length = 0.0;
  for (const auto& ex : examples) {
    total_length += std::abs(ex.value_target); // value_target is -length
  }
  stats.avg_ruler_length = examples.empty() ? 0.0 : total_length / examples.size();
  std::cout << "Average ruler length: " << stats.avg_ruler_length << "\n";

  // 2. Add to replay buffer
  replay_buffer_.add_batch(examples);
  std::cout << "Replay buffer size: " << replay_buffer_.size() << " / " << replay_buffer_.capacity()
            << "\n";

  // 3. Train on mini-batches
  std::cout << "Training for " << config_.training_steps_per_iter << " steps...\n";
  double total_policy_loss = 0.0;
  double total_value_loss = 0.0;
  double total_loss = 0.0;

  for (int step = 0; step < config_.training_steps_per_iter; ++step) {
    // Sample batch from replay buffer
    if (replay_buffer_.size() < config_.batch_size) {
      continue; // Not enough examples yet
    }

    auto batch = replay_buffer_.sample(config_.batch_size);

    // Train on batch
    auto [policy_loss, value_loss, loss] = train_batch(batch);

    total_policy_loss += policy_loss;
    total_value_loss += value_loss;
    total_loss += loss;

    // Print progress every 20 steps
    if ((step + 1) % 20 == 0) {
      std::cout << "Step " << (step + 1) << "/" << config_.training_steps_per_iter
                << " - Loss: " << loss << " (policy: " << policy_loss << ", value: " << value_loss
                << ")\n";
    }
  }

  // Average losses
  int num_steps = std::min(config_.training_steps_per_iter,
                           static_cast<int>(replay_buffer_.size() / config_.batch_size));
  if (num_steps > 0) {
    stats.policy_loss = total_policy_loss / num_steps;
    stats.value_loss = total_value_loss / num_steps;
    stats.total_loss = total_loss / num_steps;
  }

  std::cout << "Iteration " << stats.iteration << " complete\n";
  std::cout << "  Policy Loss: " << stats.policy_loss << "\n";
  std::cout << "  Value Loss: " << stats.value_loss << "\n";
  std::cout << "  Total Loss: " << stats.total_loss << "\n";

  // Store stats
  stats_history_.push_back(stats);

  return stats;
}

void Trainer::train(int num_iterations) {
  std::cout << "Starting training for " << num_iterations << " iterations\n";
  std::cout << "Configuration:\n";
  std::cout << "  Games per iteration: " << config_.games_per_iteration << "\n";
  std::cout << "  MCTS iterations per move: " << config_.mcts_iters_per_move << "\n";
  std::cout << "  Batch size: " << config_.batch_size << "\n";
  std::cout << "  Training steps per iteration: " << config_.training_steps_per_iter << "\n";
  std::cout << "  Learning rate: " << config_.learning_rate << "\n";
  std::cout << "  Optimizer: " << config_.optimizer_type << "\n";

  for (int iter = 0; iter < num_iterations; ++iter) {
    // Train one iteration
    TrainingStats stats = train_iteration();

    // Evaluate periodically
    if (stats.iteration % config_.eval_interval == 0) {
      std::cout << "\nEvaluating network...\n";
      double avg_length = evaluate(config_.eval_games);
      std::cout << "Evaluation: Average ruler length = " << avg_length << "\n";
    }

    // Save checkpoint periodically
    if (stats.iteration % config_.checkpoint_interval == 0) {
      std::string checkpoint_path =
          config_.checkpoint_dir + "/checkpoint_iter_" + std::to_string(stats.iteration) + ".bin";
      std::cout << "\nSaving checkpoint to " << checkpoint_path << "\n";
      save_checkpoint(stats.iteration, checkpoint_path);
    }
  }

  std::cout << "\nTraining complete!\n";
}

void Trainer::save_checkpoint(int iteration, const std::string& filepath) {
  std::ofstream file(filepath, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open checkpoint file for writing: " + filepath);
  }

  // Write iteration number
  file.write(reinterpret_cast<const char*>(&iteration), sizeof(iteration));

  // Get all network parameters
  auto params = network_->parameters();

  // Write number of parameter tensors
  size_t num_params = params.size();
  file.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));

  // Serialize each parameter tensor
  for (const auto* param : params) {
    // Write shape
    size_t ndim = param->ndim();
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));

    const auto& shape = param->shape();
    for (size_t dim : shape) {
      file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }

    // Write data
    const auto& data = param->data();
    size_t data_size = data.size();
    file.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data_size * sizeof(double)));
  }

  file.close();

  std::cout << "Checkpoint saved (iteration: " << iteration << ")\n";
  std::cout << "  Saved " << num_params << " parameter tensors\n";

  // Calculate total parameters
  size_t total_params = 0;
  for (const auto* param : params) {
    total_params += param->size();
  }
  std::cout << "  Total parameters: " << total_params << "\n";
}

int Trainer::load_checkpoint(const std::string& filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open checkpoint file for reading: " + filepath);
  }

  // Read iteration number
  int iteration;
  file.read(reinterpret_cast<char*>(&iteration), sizeof(iteration));

  // Get network parameters (to restore into)
  auto params = network_->parameters();

  // Read number of parameter tensors
  size_t num_params;
  file.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

  // Validate parameter count matches
  if (num_params != params.size()) {
    throw std::runtime_error("Checkpoint parameter count mismatch: expected " +
                             std::to_string(params.size()) + ", got " + std::to_string(num_params));
  }

  // Deserialize each parameter tensor
  for (size_t i = 0; i < num_params; ++i) {
    // Read shape
    size_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));

    std::vector<size_t> shape(ndim);
    for (size_t& dim : shape) {
      file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    }

    // Validate shape matches
    const auto& param_shape = params[i]->shape();
    if (shape != param_shape) {
      throw std::runtime_error("Checkpoint shape mismatch for parameter " + std::to_string(i));
    }

    // Read data
    size_t data_size;
    file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));

    // Validate data size
    if (data_size != params[i]->size()) {
      throw std::runtime_error("Checkpoint data size mismatch for parameter " + std::to_string(i));
    }

    // Read directly into parameter tensor
    auto& data = params[i]->data();
    file.read(reinterpret_cast<char*>(data.data()),
              static_cast<std::streamsize>(data_size * sizeof(double)));
  }

  file.close();

  current_iteration_ = iteration;
  std::cout << "Checkpoint loaded (iteration: " << iteration << ")\n";
  std::cout << "  Loaded " << num_params << " parameter tensors\n";

  // Calculate total parameters
  size_t total_params = 0;
  for (const auto* param : params) {
    total_params += param->size();
  }
  std::cout << "  Total parameters: " << total_params << "\n";

  return iteration;
}

double Trainer::evaluate(int num_games) {
  // Generate evaluation games and compute average ruler length
  auto eval_examples = self_play_gen_.generate_games_parallel(num_games, 4);

  if (eval_examples.empty()) {
    return 0.0;
  }

  // Calculate average ruler length
  double total_length = 0.0;
  for (const auto& ex : eval_examples) {
    total_length += std::abs(ex.value_target); // value_target is -length
  }

  return total_length / eval_examples.size();
}

} // namespace training
} // namespace golomb
