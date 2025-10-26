#include "cli/args.hpp"
#include "core/golomb.hpp"
#include "exact/exact_iface.hpp"
#include "heuristics/evo.hpp"
#include "mcts/mcts.hpp"
#include <chrono>
#include <cstring>
#include <iostream>

namespace golomb {

Args Args::parse(int argc, char** argv) {
  Args args;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--order") == 0 && i + 1 < argc) {
      args.order = std::stoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--ub") == 0 && i + 1 < argc) {
      args.ub = std::stoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
      args.mode = argv[++i];
    } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      args.iters = std::stoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--c-puct") == 0 && i + 1 < argc) {
      args.c_puct = std::stod(argv[++i]);
    } else if (std::strcmp(argv[i], "--timeout") == 0 && i + 1 < argc) {
      args.timeout_ms = std::stoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      print_usage();
      std::exit(0);
    }
  }

  return args;
}

void Args::print_usage() {
  std::cout << "usage: golomb_cli [options]\n"
            << "options:\n"
            << "  --order <n>       marks count (default: 8)\n"
            << "  --ub <value>      upper bound (default: 120)\n"
            << "  --mode <mode>     heur|mcts|exact (default: heur)\n"
            << "  --iters <n>       iterations (default: 1000)\n"
            << "  --c-puct <value>  puct const (default: 1.4)\n"
            << "  --timeout <ms>    exact timeout (default: 10000)\n"
            << "  --help, -h        show help\n";
}

}  // namespace golomb

using namespace golomb;

int main(int argc, char** argv) {
  Args args = Args::parse(argc, argv);

  std::cout << "golomb ruler search\n";
  std::cout << "order: " << args.order << ", ub: " << args.ub << ", mode: " << args.mode << "\n";

  auto start_time = std::chrono::high_resolution_clock::now();

  std::vector<int> result;

  if (args.mode == "heur") {
    std::cout << "run evolutionary (pop=64, iters=" << args.iters << ")\n";
    result = evolutionary_search(args.order, args.ub, 64, args.iters);
  } else if (args.mode == "mcts") {
    std::cout << "run mcts (iters=" << args.iters << ", c_puct=" << args.c_puct << ")\n";
    result = mcts_build(args.order, args.ub, args.iters, args.c_puct);
  } else if (args.mode == "exact") {
    std::cout << "run exact (timeout=" << args.timeout_ms << "ms)\n";
    ExactOptions opts{args.order, args.ub, args.timeout_ms};
    ExactResult res = solve_exact_stub(opts);
    result = res.rule;
    std::cout << "exact result: " << res.message << "\n";
  } else {
    std::cerr << "error: unknown mode '" << args.mode << "'\n";
    Args::print_usage();
    return 1;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  std::cout << "rule: ";
  for (int m : result) {
    std::cout << m << " ";
  }
  std::cout << "\nlength: " << length(result);
  std::cout << "\nvalid: " << (is_valid_rule(result) ? "yes" : "no");
  std::cout << "\ntime: " << elapsed << " ms\n";

  return 0;
}
