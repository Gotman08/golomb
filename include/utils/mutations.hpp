#ifndef GOLOMB_UTILS_MUTATIONS_HPP
#define GOLOMB_UTILS_MUTATIONS_HPP

#include <vector>

namespace golomb {

class RNG;

// Mutate a ruler by moving one random mark (excluding position 0) to a new random position
// Returns the mutated ruler with marks sorted and deduplicated
std::vector<int> mutate_single_mark(const std::vector<int>& marks, int ub, RNG& rng);

} // namespace golomb

#endif // GOLOMB_UTILS_MUTATIONS_HPP
