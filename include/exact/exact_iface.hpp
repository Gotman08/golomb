#pragma once

#include <string>
#include <vector>

namespace golomb {

/**
 * @brief Options for exact solver.
 */
struct ExactOptions {
  int n;               ///< Number of marks to place.
  int ub;              ///< Upper bound for mark positions.
  int timeout_ms;      ///< Maximum solving time in milliseconds.
};

/**
 * @brief Result from exact solver.
 */
struct ExactResult {
  bool optimal;             ///< Whether optimal solution was found.
  std::vector<int> rule;    ///< Best ruler found.
  int lb;                   ///< Lower bound on optimal length.
  int ub;                   ///< Upper bound on optimal length.
  std::string message;      ///< Solver status message.
};

/**
 * @brief Stub for exact solver interface.
 *
 * Placeholder for future integration with CP-SAT, ILP, or Benders decomposition.
 *
 * TODO: Integrate OR-Tools CP-SAT solver.
 * TODO: Implement ILP formulation with distance variables.
 * TODO: Add Benders decomposition for large instances.
 * TODO: Support incremental solving with learned clauses.
 *
 * @param opts Solver options.
 * @return Solver result.
 */
ExactResult solve_exact_stub(const ExactOptions& opts);

}  // namespace golomb
