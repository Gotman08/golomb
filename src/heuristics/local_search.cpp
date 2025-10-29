#include "heuristics/local_search.hpp"
#include "core/golomb.hpp"
#include "utils/mutations.hpp"
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

} // namespace golomb
