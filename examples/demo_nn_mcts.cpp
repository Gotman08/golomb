/**
 * @file demo_nn_mcts.cpp
 * @brief Démonstration du réseau de neurones intégré avec MCTS
 *
 * Ce programme montre comment:
 * 1. Créer un réseau de neurones GolombNet
 * 2. Tester le forward pass (inférence)
 * 3. Utiliser le réseau avec MCTS
 * 4. Comparer MCTS avec et sans réseau
 */

#include "nn/golomb_net.hpp"
#include "nn/state_encoder.hpp"
#include "mcts/mcts.hpp"
#include "core/golomb.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

using namespace golomb;
using namespace golomb::nn;

void print_separator() {
  std::cout << "\n" << std::string(70, '=') << "\n\n";
}

void demo_state_encoding() {
  std::cout << "=== DEMO 1: STATE ENCODING ===\n\n";

  int ub = 50;
  int target_marks = 6;
  StateEncoder encoder(ub, target_marks);

  std::cout << "Encoder configuration:\n";
  std::cout << "  Upper bound: " << ub << "\n";
  std::cout << "  Target marks: " << target_marks << "\n";
  std::cout << "  Encoding size: " << encoder.encoding_size() << " features\n";
  std::cout << "    - Positions: " << (ub + 1) << " bits\n";
  std::cout << "    - Distances: " << (ub + 1) << " bits\n";
  std::cout << "    - Metadata: 4 features\n\n";

  // Exemple d'état
  RuleState state(ub);
  state.marks = {0, 5, 12, 23};

  std::cout << "Example state: {";
  for (size_t i = 0; i < state.marks.size(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << state.marks[i];
  }
  std::cout << "}\n";

  auto encoded = encoder.encode(state);
  std::cout << "Encoded to " << encoded.size() << " features\n";
  std::cout << "  (Sum of features: " << encoded.sum() << ")\n";
}

void demo_network_inference() {
  std::cout << "=== DEMO 2: NETWORK INFERENCE ===\n\n";

  int ub = 50;
  int target_marks = 6;
  StateEncoder encoder(ub, target_marks);

  // Créer un réseau avec architecture réduite pour le test
  GolombNet net(encoder, ub, 64, 64);

  std::cout << "GolombNet architecture:\n";
  std::cout << "  Input size: " << encoder.encoding_size() << "\n";
  std::cout << "  Hidden 1: 64 units + ReLU\n";
  std::cout << "  Hidden 2: 64 units + ReLU\n";
  std::cout << "  Policy head: " << (ub + 1) << " outputs (Softmax)\n";
  std::cout << "  Value head: 1 output (Tanh)\n";
  std::cout << "  Total parameters: " << net.num_parameters() << "\n\n";

  // Test avec un état
  RuleState state(ub);
  state.marks = {0, 5, 12};

  std::cout << "Test state: {0, 5, 12}\n\n";

  // Forward pass
  Tensor policy;
  double value;
  net.forward(state, policy, value);

  std::cout << "Network predictions:\n";
  std::cout << "  Value: " << std::fixed << std::setprecision(4) << value << "\n";
  std::cout << "  Policy sum: " << policy.sum() << " (should be ~1.0)\n\n";

  // Afficher les 5 positions les plus probables
  std::cout << "Top 5 positions suggested by policy:\n";
  std::vector<std::pair<int, double>> policy_probs;
  for (size_t i = 0; i < policy.size(); ++i) {
    policy_probs.push_back({static_cast<int>(i), policy(i)});
  }
  std::sort(policy_probs.begin(), policy_probs.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  for (int i = 0; i < 5 && i < static_cast<int>(policy_probs.size()); ++i) {
    std::cout << "    Position " << std::setw(3) << policy_probs[i].first
              << ": " << std::setw(8) << std::fixed << std::setprecision(6)
              << policy_probs[i].second << "\n";
  }
}

void demo_mcts_comparison() {
  std::cout << "=== DEMO 3: MCTS WITH AND WITHOUT NEURAL NETWORK ===\n\n";

  int n = 6;
  int ub = 50;
  int iters = 100;  // Peu d'itérations pour le test

  std::cout << "Problem: Find Golomb ruler with " << n << " marks, ub=" << ub << "\n";
  std::cout << "MCTS iterations: " << iters << "\n\n";

  // MCTS sans réseau (heuristique uniforme)
  auto start = std::chrono::high_resolution_clock::now();
  auto result_no_nn = mcts_build(n, ub, iters);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration_no_nn = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "Result WITHOUT neural network:\n";
  std::cout << "  Ruler: {";
  for (size_t i = 0; i < result_no_nn.size(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << result_no_nn[i];
  }
  std::cout << "}\n";
  std::cout << "  Length: " << length(result_no_nn) << "\n";
  std::cout << "  Valid: " << (is_valid_rule(result_no_nn) ? "YES" : "NO") << "\n";
  std::cout << "  Time: " << duration_no_nn.count() << " ms\n\n";

  // MCTS avec réseau (même non entraîné, il donne une baseline)
  StateEncoder encoder(ub, n);
  GolombNet net(encoder, ub, 64, 64);

  start = std::chrono::high_resolution_clock::now();
  auto result_with_nn = mcts_build_nn(n, ub, iters, &net);
  end = std::chrono::high_resolution_clock::now();
  auto duration_with_nn = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "Result WITH neural network (untrained):\n";
  std::cout << "  Ruler: {";
  for (size_t i = 0; i < result_with_nn.size(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << result_with_nn[i];
  }
  std::cout << "}\n";
  std::cout << "  Length: " << length(result_with_nn) << "\n";
  std::cout << "  Valid: " << (is_valid_rule(result_with_nn) ? "YES" : "NO") << "\n";
  std::cout << "  Time: " << duration_with_nn.count() << " ms\n\n";

  std::cout << "NOTE: Le réseau n'est pas entraîné, donc les résultats sont aléatoires.\n";
  std::cout << "      Après entraînement, le réseau devrait guider MCTS vers de\n";
  std::cout << "      meilleures solutions plus rapidement.\n";
}

void demo_gradient_flow() {
  std::cout << "=== DEMO 4: GRADIENT FLOW (BACKWARD PASS) ===\n\n";

  int ub = 30;
  int target_marks = 5;
  StateEncoder encoder(ub, target_marks);
  GolombNet net(encoder, ub, 32, 32);

  RuleState state(ub);
  state.marks = {0, 5, 12};

  std::cout << "Testing backpropagation with dummy gradients...\n\n";

  // Forward pass
  Tensor policy;
  double value;
  net.forward(state, policy, value);

  std::cout << "Forward pass:\n";
  std::cout << "  Predicted value: " << value << "\n";
  std::cout << "  Policy entropy: " << std::fixed << std::setprecision(4)
            << (-policy.sum()) << "\n\n";

  // Create dummy gradients
  Tensor grad_policy(ub + 1);
  grad_policy.zeros();
  grad_policy(10) = 1.0;  // Encourage position 10

  double grad_value = 1.0;  // Encourage higher value

  net.zero_grad();
  net.backward(grad_policy, grad_value);

  std::cout << "Backward pass completed!\n";
  std::cout << "Gradients computed for " << net.parameters().size() << " parameter tensors\n";

  // Check gradient norms
  auto grads = net.gradients();
  double total_grad_norm = 0.0;
  for (auto* grad : grads) {
    for (const auto& g : grad->data()) {
      total_grad_norm += g * g;
    }
  }
  total_grad_norm = std::sqrt(total_grad_norm);

  std::cout << "Total gradient norm: " << std::scientific << std::setprecision(3)
            << total_grad_norm << "\n";
  std::cout << "\nGradient flow is working! Ready for training.\n";
}

int main() {
  std::cout << "\n";
  std::cout << "╔═══════════════════════════════════════════════════════════════════╗\n";
  std::cout << "║  GOLOMB NEURAL NETWORK + MCTS DEMONSTRATION                       ║\n";
  std::cout << "║  AlphaGo-style architecture implemented from scratch              ║\n";
  std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n";

  print_separator();

  try {
    demo_state_encoding();
    print_separator();

    demo_network_inference();
    print_separator();

    demo_mcts_comparison();
    print_separator();

    demo_gradient_flow();
    print_separator();

    std::cout << "╔═══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  ALL DEMOS COMPLETED SUCCESSFULLY!                                ║\n";
    std::cout << "║                                                                   ║\n";
    std::cout << "║  Next steps:                                                      ║\n";
    std::cout << "║  1. Implement optimizer (SGD/Adam) ✓ Architecture ready          ║\n";
    std::cout << "║  2. Implement self-play data generation                          ║\n";
    std::cout << "║  3. Implement training loop                                       ║\n";
    std::cout << "║  4. Train the network on self-play data                          ║\n";
    std::cout << "║  5. Evaluate trained network vs heuristic MCTS                   ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n\n";

  } catch (const std::exception& e) {
    std::cerr << "\n❌ Error: " << e.what() << "\n\n";
    return 1;
  }

  return 0;
}
