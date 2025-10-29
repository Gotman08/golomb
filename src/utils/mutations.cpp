#include "utils/mutations.hpp"
#include "utils/random.hpp"
#include <algorithm>

namespace golomb {

std::vector<int> mutate_single_mark(const std::vector<int>& marks, int ub, RNG& rng) {
  if (marks.size() < 2) {
    return marks;
  }

  std::vector<int> mutated = marks;
  // Select random mark to move (excluding position 0)
  int idx = rng.uniform_int(1, static_cast<int>(marks.size()) - 1);
  // Move to new random position
  int new_pos = rng.uniform_int(1, ub - 1);

  mutated[idx] = new_pos;
  // Sort and remove duplicates
  std::sort(mutated.begin(), mutated.end());
  mutated.erase(std::unique(mutated.begin(), mutated.end()), mutated.end());

  return mutated;
}

} // namespace golomb
