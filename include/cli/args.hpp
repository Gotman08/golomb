#pragma once

#include <string>

namespace golomb {

/**
 * @brief Command-line arguments for golomb_cli.
 */
struct Args {
  int order = 8;                  ///< Number of marks (n).
  int ub = 120;                   ///< Upper bound for positions.
  std::string mode = "heur";      ///< Mode: "heur", "mcts", or "exact".
  int iters = 1000;               ///< Iterations for heur/mcts.
  double c_puct = 1.4;            ///< PUCT exploration constant.
  int timeout_ms = 10000;         ///< Timeout for exact solver.

  /**
   * @brief Parse command-line arguments.
   * @param argc Argument count.
   * @param argv Argument vector.
   * @return Parsed arguments.
   */
  static Args parse(int argc, char** argv);

  /**
   * @brief Print usage information.
   */
  static void print_usage();
};

}  // namespace golomb
