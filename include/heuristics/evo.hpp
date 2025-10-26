#pragma once

#include <vector>

namespace golomb {

/**
 * @brief Evolutionary search for Golomb ruler optimization.
 *
 * Uses a population-based approach with mutation, crossover, and local improvement.
 *
 * @param n Number of marks to place.
 * @param ub Upper bound for mark positions.
 * @param pop Population size (default 64).
 * @param iters Number of generations (default 1000).
 * @return Best ruler found (sorted marks).
 *
 * TODO: Implement adaptive mutation rates, tournament selection variants.
 */
std::vector<int> evolutionary_search(int n, int ub, int pop = 64, int iters = 1000);

}  // namespace golomb
