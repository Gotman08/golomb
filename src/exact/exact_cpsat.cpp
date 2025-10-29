#include "core/golomb.hpp"
#include "exact/exact_iface.hpp"
#include "ortools/sat/cp_model.h"
#include <algorithm>
#include <cmath>

namespace golomb {

using operations_research::Domain;
using operations_research::sat::CpModelBuilder;
using operations_research::sat::CpModelProto;
using operations_research::sat::CpSolverResponse;
using operations_research::sat::CpSolverStatus;
using operations_research::sat::IntVar;
using operations_research::sat::LinearExpr;
using operations_research::sat::SatParameters;
using operations_research::sat::SolutionIntegerValue;
using operations_research::sat::SolveWithParameters;

/**
 * @brief Solve Golomb ruler exactly using OR-Tools CP-SAT.
 *
 * Uses constraint programming with AllDifferent constraints on pairwise
 * distances. Minimizes ruler length (last mark position).
 *
 * Model:
 * - Variables: marks[i] ∈ [0, ub] for i ∈ [0, n-1]
 * - Constraints:
 *   1. marks[0] = 0 (fixed first mark)
 *   2. marks[i] < marks[i+1] (strictly increasing)
 *   3. AllDifferent({marks[j] - marks[i] | 0 ≤ i < j < n})
 * - Objective: Minimize marks[n-1]
 *
 * Optimizations:
 * - Symmetry breaking: first_distance < last_distance
 * - Implied lower bound: marks[n-1] ≥ n*(n-1)/2
 *
 * @param opts Solver options (n, ub, timeout_ms).
 * @return Optimal or best feasible solution found.
 */
ExactResult solve_exact_cpsat(const ExactOptions& opts) {
  CpModelBuilder model;
  ExactResult result;

  // 1. Create decision variables for mark positions
  std::vector<IntVar> marks;
  marks.reserve(opts.n);

  for (int i = 0; i < opts.n; ++i) {
    marks.push_back(model.NewIntVar(Domain(0, opts.ub)).WithName("mark_" + std::to_string(i)));
  }

  // 2. Fix first mark at position 0
  model.AddEquality(marks[0], 0);

  // 3. Add ordering constraints (strictly increasing)
  for (int i = 0; i < opts.n - 1; ++i) {
    model.AddLessThan(marks[i], marks[i + 1]);
  }

  // 4. Compute all pairwise distances
  // Number of distances: n*(n-1)/2
  std::vector<IntVar> distances;
  int num_distances = opts.n * (opts.n - 1) / 2;
  distances.reserve(num_distances);

  for (int i = 0; i < opts.n; ++i) {
    for (int j = i + 1; j < opts.n; ++j) {
      // Distance between mark i and mark j
      IntVar dist = model.NewIntVar(Domain(1, opts.ub))
                        .WithName("dist_" + std::to_string(i) + "_" + std::to_string(j));

      // dist = marks[j] - marks[i]
      model.AddEquality(dist, LinearExpr(marks[j]) - LinearExpr(marks[i]));

      distances.push_back(dist);
    }
  }

  // 5. All distances must be different (main constraint)
  model.AddAllDifferent(distances);

  // 6. Symmetry breaking: first distance < last distance
  // This breaks reflection symmetry (ruler can be read left-to-right or
  // right-to-left)
  if (opts.n >= 3) {
    // first_distance = marks[1] - marks[0] = marks[1]
    // last_distance = marks[n-1] - marks[n-2]
    IntVar first_dist = marks[1]; // Since marks[0] = 0
    IntVar last_dist = model.NewIntVar(Domain(1, opts.ub));
    model.AddEquality(last_dist, LinearExpr(marks[opts.n - 1]) - LinearExpr(marks[opts.n - 2]));
    model.AddLessThan(first_dist, last_dist);
  }

  // 7. Implied lower bound constraint
  // The minimum possible length is the triangular number: n*(n-1)/2
  int min_length = opts.n * (opts.n - 1) / 2;
  model.AddGreaterOrEqual(marks[opts.n - 1], min_length);

  // 8. Set objective: minimize ruler length (last mark position)
  model.Minimize(marks[opts.n - 1]);

  // 9. Configure solver parameters
  SatParameters params;

  // Set timeout
  double timeout_seconds = opts.timeout_ms / 1000.0;
  params.set_max_time_in_seconds(timeout_seconds);

  // Disable search progress logging (keep output clean)
  params.set_log_search_progress(false);

  // Use default search (automatic)
  params.set_search_branching(SatParameters::AUTOMATIC_SEARCH);

  // Enable all optimizations
  params.set_cp_model_presolve(true);
  params.set_cp_model_probing_level(2);
  params.set_symmetry_level(2);

  // 10. Build and solve the model
  CpModelProto model_proto = model.Build();
  CpSolverResponse response = SolveWithParameters(model_proto, params);

  // 11. Extract results based on solver status
  CpSolverStatus status = response.status();

  if (status == CpSolverStatus::OPTIMAL) {
    // Found provably optimal solution
    result.optimal = true;
    result.rule.clear();
    result.rule.reserve(opts.n);

    for (const auto& mark : marks) {
      result.rule.push_back(static_cast<int>(SolutionIntegerValue(response, mark)));
    }

    result.lb = static_cast<int>(response.objective_value());
    result.ub = result.lb; // lb = ub when optimal
    result.message = "optimal";

  } else if (status == CpSolverStatus::FEASIBLE) {
    // Found feasible solution but not proven optimal (timeout)
    result.optimal = false;
    result.rule.clear();
    result.rule.reserve(opts.n);

    for (const auto& mark : marks) {
      result.rule.push_back(static_cast<int>(SolutionIntegerValue(response, mark)));
    }

    result.lb = static_cast<int>(response.best_objective_bound());
    result.ub = static_cast<int>(response.objective_value());
    result.message = "feasible (timeout)";

  } else if (status == CpSolverStatus::INFEASIBLE) {
    // Problem is infeasible (upper bound too tight)
    result.optimal = false;
    result.rule.clear();
    result.lb = opts.ub + 1; // Indicates infeasibility
    result.ub = opts.ub;
    result.message = "infeasible (ub too small)";

  } else {
    // Unknown or other error (no solution found before timeout)
    result.optimal = false;

    // Return greedy fallback solution
    result.rule = greedy_seed(opts.n, opts.ub);
    result.lb = min_length;
    result.ub = length(result.rule);
    result.message = "unknown/timeout (no solution)";
  }

  return result;
}

} // namespace golomb
