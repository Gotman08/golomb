#pragma once

#include <vector>

namespace golomb {

/**
 * @brief Local search (hill climbing) for Golomb ruler improvement.
 *
 * Performs iterative improvement by moving marks to better positions.
 *
 * @param start Initial ruler configuration.
 * @param ub Upper bound for mark positions.
 * @param budget Maximum number of iterations.
 * @return Improved ruler (sorted marks).
 *
 * TODO: Implement simulated annealing, tabu search variants.
 */
std::vector<int> hill_climb(const std::vector<int>& start, int ub, int budget);

}  // namespace golomb
