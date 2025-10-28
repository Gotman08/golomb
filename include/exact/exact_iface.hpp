#pragma once

#include <string>
#include <vector>

namespace golomb {

/**
 * @brief Options for exact solver.
 */
struct ExactOptions {
  int n;          ///< Number of marks to place.
  int ub;         ///< Upper bound for mark positions.
  int timeout_ms; ///< Maximum solving time in milliseconds.
};

/**
 * @brief Result from exact solver.
 */
struct ExactResult {
  bool optimal;          ///< Whether optimal solution was found.
  std::vector<int> rule; ///< Best ruler found.
  int lb;                ///< Lower bound on optimal length.
  int ub;                ///< Upper bound on optimal length.
  std::string message;   ///< Solver status message.
};

/**
 * @brief Stub for exact solver interface.
 *
 * Placeholder that returns a greedy solution. Use solve_exact_cpsat() for exact
 * solving.
 *
 * @param opts Solver options.
 * @return Solver result.
 */
ExactResult solve_exact_stub(const ExactOptions& opts);

/**
 * @brief Solve Golomb ruler exactly using OR-Tools CP-SAT.
 *
 * Uses constraint programming with AllDifferent constraints on pairwise
 * distances. Minimizes ruler length (last mark position).
 *
 * Model formulation:
 * - Variables: marks[i] ∈ [0, ub] for i ∈ [0, n-1]
 * - Constraints:
 *   1. marks[0] = 0 (fixed first mark)
 *   2. marks[i] < marks[i+1] (strictly increasing)
 *   3. AllDifferent({marks[j] - marks[i] | 0 ≤ i < j < n})
 * - Objective: Minimize marks[n-1]
 * - Optimizations: Symmetry breaking, implied bounds
 *
 * Performance expectations:
 * - n=3: < 0.1s (trivial)
 * - n=4: < 0.5s
 * - n=5: < 2s
 * - n=6: < 10s
 * - n=7: < 1min
 * - n=8: 1-5min
 * - n=9: 5-30min
 * - n≥10: May timeout (requires hours)
 *
 * Recommended: Use for n ≤ 8. For larger instances, use heuristics.
 *
 * @param opts Solver options (n, ub, timeout_ms).
 * @return Optimal or best feasible solution found.
 */
ExactResult solve_exact_cpsat(const ExactOptions& opts);

} // namespace golomb
