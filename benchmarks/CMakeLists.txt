# Fetch Google Benchmark via CPM
CPMAddPackage(
  NAME benchmark
  GITHUB_REPOSITORY google/benchmark
  VERSION 1.8.3
  OPTIONS "BENCHMARK_ENABLE_TESTING OFF"
)

add_executable(golomb_benchmarks
  bench_build_partial_rule.cpp
)

target_link_libraries(golomb_benchmarks
  PRIVATE
    golomb_core
    benchmark::benchmark
)
