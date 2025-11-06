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
 */
std::vector<int> hill_climb(const std::vector<int>& start, int ub, int budget);

/**
 * @brief Simulated annealing for Golomb ruler optimization.
 *
 * Uses temperature-based probabilistic acceptance to escape local optima.
 * Accepts worse solutions with probability exp(-ΔE/T) where:
 * - ΔE = increase in energy (ruler length)
 * - T = temperature (decreases over time)
 *
 * Temperature schedule:
 * - Geometric cooling: T(t) = T0 * α^t
 * - α = 0.95 (cooling rate)
 * - T0 = initial temperature
 *
 * @param start Initial ruler configuration.
 * @param ub Upper bound for mark positions.
 * @param budget Maximum number of iterations.
 * @param initial_temp Initial temperature (default: 100.0).
 * @param cooling_rate Cooling rate α (default: 0.95).
 * @return Best ruler found (sorted marks).
 */
std::vector<int> simulated_annealing(const std::vector<int>& start, int ub, int budget,
                                     double initial_temp = 100.0, double cooling_rate = 0.95);

/**
 * @brief Tabu search for Golomb ruler optimization.
 *
 * Maintains a tabu list of recently visited solutions to avoid cycling.
 * Accepts best non-tabu neighbor even if worse than current solution.
 *
 * Features:
 * - Tabu list with recency-based memory
 * - Aspiration criterion: accept tabu move if it improves best-so-far
 * - Diversification: restart if stuck
 *
 * @param start Initial ruler configuration.
 * @param ub Upper bound for mark positions.
 * @param budget Maximum number of iterations.
 * @param tabu_tenure Number of iterations a move remains tabu (default: 10).
 * @return Best ruler found (sorted marks).
 */
std::vector<int> tabu_search(const std::vector<int>& start, int ub, int budget,
                             int tabu_tenure = 10);

} // namespace golomb
