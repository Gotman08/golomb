#include "exact/exact_iface.hpp"
#include "core/golomb.hpp"

namespace golomb {

ExactResult solve_exact_stub(const ExactOptions& opts) {
  ExactResult result;
  result.optimal = false;
  result.lb = 0;
  result.ub = opts.ub;
  result.message = "stub: exact not implemented";

  // TODO: integrate real solver (CP-SAT, MILP, Benders)
  // FIXME: return greedy solution as placeholder
  result.rule = greedy_seed(opts.n, opts.ub);

  return result;
}

}  // namespace golomb
