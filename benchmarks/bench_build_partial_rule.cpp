#include <benchmark/benchmark.h>
#include "core/golomb.hpp"

using namespace golomb;

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

BENCHMARK(BM_BuildPartialRule)->Range(4, 64);

static void BM_ValidateRule(benchmark::State& state) {
  int n = state.range(0);
  auto rule = greedy_seed(n, n * n);

  for (auto _ : state) {
    bool valid = is_valid_rule(rule);
    benchmark::DoNotOptimize(valid);
  }
}

BENCHMARK(BM_ValidateRule)->Range(4, 64);

BENCHMARK_MAIN();
