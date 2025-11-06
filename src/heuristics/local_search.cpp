#include "heuristics/local_search.hpp"
#include "core/golomb.hpp"
#include "utils/mutations.hpp"
#include "utils/random.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <unordered_set>

namespace golomb {

std::vector<int> hill_climb(const std::vector<int>& start, int ub, int budget) {
  if (start.empty()) {
    return start;
  }

  RNG rng;
  std::vector<int> current = start;
  int current_len = length(current);

  for (int iter = 0; iter < budget; ++iter) {
    // Create neighbor by mutating a single mark
    std::vector<int> neighbor = mutate_single_mark(current, ub, rng);

    // Check if valid and better
    if (is_valid_rule(neighbor)) {
      int neighbor_len = length(neighbor);
      if (neighbor_len < current_len) {
        current = neighbor;
        current_len = neighbor_len;
      }
    }
  }

  return current;
}

std::vector<int> simulated_annealing(const std::vector<int>& start, int ub, int budget,
                                     double initial_temp, double cooling_rate) {
  if (start.empty()) {
    return start;
  }

  RNG rng;
  std::vector<int> current = start;
  int current_len = length(current);

  std::vector<int> best = current;
  int best_len = current_len;

  double temperature = initial_temp;

  for (int iter = 0; iter < budget; ++iter) {
    // Create neighbor
    std::vector<int> neighbor = mutate_single_mark(current, ub, rng);

    if (!is_valid_rule(neighbor)) {
      continue; // Skip invalid solutions
    }

    int neighbor_len = length(neighbor);
    int delta = neighbor_len - current_len; // Change in energy

    // Acceptance probability
    bool accept = false;
    if (delta < 0) {
      // Better solution: always accept
      accept = true;
    } else {
      // Worse solution: accept with probability exp(-ΔE/T)
      double prob = std::exp(-static_cast<double>(delta) / temperature);
      accept = (rng.uniform_double(0.0, 1.0) < prob);
    }

    if (accept) {
      current = neighbor;
      current_len = neighbor_len;

      // Track best solution found
      if (current_len < best_len) {
        best = current;
        best_len = current_len;
      }
    }

    // Cool down temperature (geometric cooling)
    temperature *= cooling_rate;

    // Prevent temperature from getting too low
    if (temperature < 0.01) {
      temperature = 0.01;
    }
  }

  return best;
}

std::vector<int> tabu_search(const std::vector<int>& start, int ub, int budget, int tabu_tenure) {
  if (start.empty()) {
    return start;
  }

  RNG rng;
  std::vector<int> current = start;
  int current_len = length(current);

  std::vector<int> best = current;
  int best_len = current_len;

  // Tabu list: stores hash of recently modified mark positions
  // Format: (mark_index, new_position)
  std::deque<std::pair<int, int>> tabu_list;

  // Helper: compute hash for a move
  auto move_hash = [](int mark_idx, int new_pos) -> long long {
    return (static_cast<long long>(mark_idx) << 32) | static_cast<long long>(new_pos);
  };

  // Helper: check if move is tabu
  auto is_tabu = [&](int mark_idx, int new_pos) -> bool {
    long long hash = move_hash(mark_idx, new_pos);
    for (const auto& [idx, pos] : tabu_list) {
      if (move_hash(idx, pos) == hash) {
        return true;
      }
    }
    return false;
  };

  int stagnation_count = 0;
  const int max_stagnation = budget / 10; // Restart if no improvement for 10% of budget

  for (int iter = 0; iter < budget; ++iter) {
    // Generate neighborhood (explore multiple neighbors)
    std::vector<std::pair<std::vector<int>, std::pair<int, int>>> neighbors;

    // Try modifying each mark (except first mark which is fixed at 0)
    for (size_t i = 1; i < current.size(); ++i) {
      // Try several random positions for this mark
      for (int attempt = 0; attempt < 5; ++attempt) {
        std::vector<int> neighbor = current;
        int old_pos = neighbor[i];
        int new_pos = rng.uniform_int(1, ub - 1);

        neighbor[i] = new_pos;
        std::sort(neighbor.begin(), neighbor.end());

        if (is_valid_rule(neighbor)) {
          neighbors.emplace_back(neighbor, std::make_pair(static_cast<int>(i), new_pos));
        }
      }
    }

    if (neighbors.empty()) {
      // No valid neighbors, diversification: random restart
      current = greedy_seed(static_cast<int>(start.size()), ub);
      current_len = length(current);
      tabu_list.clear();
      stagnation_count = 0;
      continue;
    }

    // Select best non-tabu neighbor
    std::vector<int> best_neighbor;
    int best_neighbor_len = INT_MAX;
    std::pair<int, int> best_move = {-1, -1};
    bool found_non_tabu = false;

    for (const auto& [neighbor, move] : neighbors) {
      int neighbor_len = length(neighbor);
      bool move_is_tabu = is_tabu(move.first, move.second);

      // Aspiration criterion: accept tabu move if it improves best-so-far
      bool aspiration = (neighbor_len < best_len);

      if (!move_is_tabu || aspiration) {
        if (neighbor_len < best_neighbor_len) {
          best_neighbor = neighbor;
          best_neighbor_len = neighbor_len;
          best_move = move;
          found_non_tabu = true;
        }
      }
    }

    if (!found_non_tabu) {
      // All neighbors are tabu and don't meet aspiration criterion
      // Accept least-bad tabu neighbor
      for (const auto& [neighbor, move] : neighbors) {
        int neighbor_len = length(neighbor);
        if (neighbor_len < best_neighbor_len) {
          best_neighbor = neighbor;
          best_neighbor_len = neighbor_len;
          best_move = move;
        }
      }
    }

    // Move to best neighbor
    current = best_neighbor;
    current_len = best_neighbor_len;

    // Add move to tabu list
    if (best_move.first != -1) {
      tabu_list.push_back(best_move);
      if (static_cast<int>(tabu_list.size()) > tabu_tenure) {
        tabu_list.pop_front(); // Remove oldest tabu
      }
    }

    // Update best solution
    if (current_len < best_len) {
      best = current;
      best_len = current_len;
      stagnation_count = 0;
    } else {
      stagnation_count++;
    }

    // Diversification: restart if stagnant
    if (stagnation_count > max_stagnation) {
      current = greedy_seed(static_cast<int>(start.size()), ub);
      current_len = length(current);
      tabu_list.clear();
      stagnation_count = 0;
    }
  }

  return best;
}

} // namespace golomb
