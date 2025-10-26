#include <benchmark/benchmark.h>
#include "core/golomb.hpp"
#include "heuristics/evo.hpp"
#include "mcts/mcts.hpp"

using namespace golomb;

// ============================================================================
// Core operations benchmarks
// ============================================================================

static void BM_BuildPartialRule(benchmark::State& state) {
  int n = state.range(0);
  int ub = n * n;

  for (auto _ : state) {
    RuleState st(ub);
    st.marks.push_back(0);

    int pos = 1;
    while (static_cast<int>(st.marks.size()) < n && pos < ub) {
      if (st.used.can_add_mark(st.marks, pos)) {
        st.used.add_mark(st.marks, pos);
      }
      ++pos;
    }

    benchmark::DoNotOptimize(st.marks);
  }
}

BENCHMARK(BM_BuildPartialRule)->Range(4, 32);

static void BM_ValidateRule(benchmark::State& state) {
  int n = state.range(0);
  auto rule = greedy_seed(n, n * n);

  for (auto _ : state) {
    bool valid = is_valid_rule(rule);
    benchmark::DoNotOptimize(valid);
  }
}

BENCHMARK(BM_ValidateRule)->Range(4, 32);

// ============================================================================
// Greedy seed generation benchmarks
// ============================================================================

static void BM_GreedySeed_Small(benchmark::State& state) {
  int n = state.range(0);
  int ub = n * n;

  for (auto _ : state) {
    auto rule = greedy_seed(n, ub);
    benchmark::DoNotOptimize(rule);
  }
}

BENCHMARK(BM_GreedySeed_Small)->DenseRange(4, 12, 2);

static void BM_GreedySeed_Medium(benchmark::State& state) {
  int n = state.range(0);
  int ub = n * n;

  for (auto _ : state) {
    auto rule = greedy_seed(n, ub);
    benchmark::DoNotOptimize(rule);
  }
}

BENCHMARK(BM_GreedySeed_Medium)->DenseRange(16, 32, 4);

// ============================================================================
// Evolutionary algorithm benchmarks
// ============================================================================

static void BM_Evolutionary_SmallPopulation(benchmark::State& state) {
  int n = 8;
  int ub = 120;
  int pop = 32;
  int iters = state.range(0);

  for (auto _ : state) {
    auto rule = evolutionary_search(n, ub, pop, iters);
    benchmark::DoNotOptimize(rule);
    benchmark::DoNotOptimize(length(rule));
  }
}

BENCHMARK(BM_Evolutionary_SmallPopulation)->Arg(100)->Arg(500)->Arg(1000);

static void BM_Evolutionary_LargePopulation(benchmark::State& state) {
  int n = 8;
  int ub = 120;
  int pop = 128;
  int iters = state.range(0);

  for (auto _ : state) {
    auto rule = evolutionary_search(n, ub, pop, iters);
    benchmark::DoNotOptimize(rule);
    benchmark::DoNotOptimize(length(rule));
  }
}

BENCHMARK(BM_Evolutionary_LargePopulation)->Arg(100)->Arg(500)->Arg(1000);

// ============================================================================
// MCTS benchmarks
// ============================================================================

static void BM_MCTS_ShortRun(benchmark::State& state) {
  int n = 7;
  int ub = 80;
  int iters = state.range(0);
  double c_puct = 1.4;

  for (auto _ : state) {
    auto rule = mcts_build(n, ub, iters, c_puct);
    benchmark::DoNotOptimize(rule);
    benchmark::DoNotOptimize(length(rule));
  }
}

BENCHMARK(BM_MCTS_ShortRun)->Arg(100)->Arg(500)->Arg(1000);

static void BM_MCTS_ExplorationVariance(benchmark::State& state) {
  int n = 7;
  int ub = 80;
  int iters = 500;
  double c_puct = static_cast<double>(state.range(0)) / 10.0;  // 1.0, 1.4, 2.0

  for (auto _ : state) {
    auto rule = mcts_build(n, ub, iters, c_puct);
    benchmark::DoNotOptimize(rule);
    benchmark::DoNotOptimize(length(rule));
  }
}

BENCHMARK(BM_MCTS_ExplorationVariance)->Arg(10)->Arg(14)->Arg(20);

// ============================================================================
// Comparison benchmarks: Greedy vs Evolutionary vs MCTS
// ============================================================================

static void BM_Compare_Greedy(benchmark::State& state) {
  int n = 8;
  int ub = 120;

  for (auto _ : state) {
    auto rule = greedy_seed(n, ub);
    benchmark::DoNotOptimize(rule);
    benchmark::DoNotOptimize(length(rule));
  }
}

BENCHMARK(BM_Compare_Greedy);

static void BM_Compare_Evolutionary(benchmark::State& state) {
  int n = 8;
  int ub = 120;
  int pop = 64;
  int iters = 1000;

  for (auto _ : state) {
    auto rule = evolutionary_search(n, ub, pop, iters);
    benchmark::DoNotOptimize(rule);
    benchmark::DoNotOptimize(length(rule));
  }
}

BENCHMARK(BM_Compare_Evolutionary);

static void BM_Compare_MCTS(benchmark::State& state) {
  int n = 8;
  int ub = 120;
  int iters = 1000;
  double c_puct = 1.4;

  for (auto _ : state) {
    auto rule = mcts_build(n, ub, iters, c_puct);
    benchmark::DoNotOptimize(rule);
    benchmark::DoNotOptimize(length(rule));
  }
}

BENCHMARK(BM_Compare_MCTS);

BENCHMARK_MAIN();
