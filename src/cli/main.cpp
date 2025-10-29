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
    try {
      if (std::strcmp(argv[i], "--order") == 0 && i + 1 < argc) {
        args.order = std::stoi(argv[++i]);
        if (args.order < 2 || args.order > 1000) {
          throw std::out_of_range("order must be between 2 and 1000");
        }
      } else if (std::strcmp(argv[i], "--ub") == 0 && i + 1 < argc) {
        args.ub = std::stoi(argv[++i]);
        if (args.ub < 1 || args.ub > 1000000) {
          throw std::out_of_range("ub must be between 1 and 1000000");
        }
      } else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
        args.mode = argv[++i];
      } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
        args.iters = std::stoi(argv[++i]);
        if (args.iters < 1 || args.iters > 10000000) {
          throw std::out_of_range("iters must be between 1 and 10000000");
        }
      } else if (std::strcmp(argv[i], "--c-puct") == 0 && i + 1 < argc) {
        args.c_puct = std::stod(argv[++i]);
        if (args.c_puct < 0.0 || args.c_puct > 100.0) {
          throw std::out_of_range("c-puct must be between 0.0 and 100.0");
        }
      } else if (std::strcmp(argv[i], "--timeout") == 0 && i + 1 < argc) {
        args.timeout_ms = std::stoi(argv[++i]);
        if (args.timeout_ms < 1 || args.timeout_ms > 3600000) {
          throw std::out_of_range("timeout must be between 1ms and 3600000ms (1 hour)");
        }
      } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
        print_usage();
        std::exit(0);
      }
    } catch (const std::invalid_argument& e) {
      std::cerr << "Error: invalid argument for " << argv[i - 1] << ": '" << argv[i]
                << "' (not a valid number)\n";
      print_usage();
      std::exit(1);
    } catch (const std::out_of_range& e) {
      std::cerr << "Error: " << e.what() << "\n";
      print_usage();
      std::exit(1);
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
    ExactResult res = solve_exact_cpsat(opts);
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
