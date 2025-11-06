#pragma once

#include <vector>

namespace golomb {

/**
 * @brief Evolutionary search for Golomb ruler optimization.
 *
 * Uses a population-based approach with mutation, crossover, and local improvement.
 * Basic version with elitism and simple selection.
 *
 * @param n Number of marks to place.
 * @param ub Upper bound for mark positions.
 * @param pop Population size (default 64).
 * @param iters Number of generations (default 1000).
 * @return Best ruler found (sorted marks).
 */
std::vector<int> evolutionary_search(int n, int ub, int pop = 64, int iters = 1000);

/**
 * @brief Evolutionary search with adaptive mutation rate.
 *
 * Mutation rate adapts based on population diversity and progress:
 * - Increases when population converges (low diversity)
 * - Decreases when making good progress
 * - Formula: rate(t) = base_rate * (1 + diversity_factor)
 *
 * @param n Number of marks to place.
 * @param ub Upper bound for mark positions.
 * @param pop Population size (default 64).
 * @param iters Number of generations (default 1000).
 * @param base_mutation_rate Base mutation probability (default 0.1).
 * @return Best ruler found (sorted marks).
 */
std::vector<int> evolutionary_search_adaptive(int n, int ub, int pop = 64, int iters = 1000,
                                              double base_mutation_rate = 0.1);

/**
 * @brief Evolutionary search with tournament selection.
 *
 * Uses tournament selection instead of elitism:
 * - Select k random individuals
 * - Choose the best from this tournament
 * - More diverse than pure elitism
 *
 * Tournament size controls selection pressure:
 * - Small tournament (k=2): low pressure, more diversity
 * - Large tournament (k=7): high pressure, faster convergence
 *
 * @param n Number of marks to place.
 * @param ub Upper bound for mark positions.
 * @param pop Population size (default 64).
 * @param iters Number of generations (default 1000).
 * @param tournament_size Number of individuals per tournament (default 5).
 * @return Best ruler found (sorted marks).
 */
std::vector<int> evolutionary_search_tournament(int n, int ub, int pop = 64, int iters = 1000,
                                                int tournament_size = 5);

} // namespace golomb
