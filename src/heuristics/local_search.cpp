#include "heuristics/local_search.hpp"
#include "core/golomb.hpp"
#include "utils/random.hpp"
#include <algorithm>

namespace golomb {

std::vector<int> hill_climb(const std::vector<int>& start, int ub, int budget) {
  if (start.empty()) {
    return start;
  }

  RNG rng;
  std::vector<int> current = start;
  int current_len = length(current);

  // TODO: implement neighborhood exploration, acceptance criteria, restart strategies
  for (int iter = 0; iter < budget; ++iter) {
    // Try moving a random mark (excluding 0) to a random position
    if (current.size() < 2) {
      break;
    }

    int idx = rng.uniform_int(1, static_cast<int>(current.size()) - 1);
    int old_pos = current[idx];
    int new_pos = rng.uniform_int(1, ub - 1);

    if (new_pos == old_pos) {
      continue;
    }

    // Create neighbor
    std::vector<int> neighbor = current;
    neighbor[idx] = new_pos;
    std::sort(neighbor.begin(), neighbor.end());
    neighbor.erase(std::unique(neighbor.begin(), neighbor.end()), neighbor.end());

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

}  // namespace golomb
