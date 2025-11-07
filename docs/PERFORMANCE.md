# Performance Benchmarks

This document tracks the performance impact of CSAPP-based optimizations.

**Last Updated:** November 6, 2025

---

## Executive Summary

| Metric | Baseline | Optimized | Speedup | Status |
|--------|----------|-----------|---------|--------|
| **Overall Speedup** | - | - | **Target: 15-25×** | ⏳ Pending |
| Matrix Multiplication | - | - | Target: 10-15× | ⏳ Pending |
| MCTS Simulations | - | - | Target: 5-10× | ⏳ Pending |
| Neural Network Forward | - | - | Target: 3-5× | ⏳ Pending |
| Adam Optimizer Step | - | - | Target: 1.5-2× | ⏳ Pending |
| Parallel MCTS Efficiency | - | - | Target: 2-3× | ⏳ Pending |

---

## Benchmark Setup

### Hardware

**Test Machine:**
- CPU: [TODO: e.g., Intel Core i7-12700K @ 3.6GHz]
- Cores: [TODO: e.g., 12 cores / 20 threads]
- RAM: [TODO: e.g., 32GB DDR4-3200]
- L1 Cache: [TODO: e.g., 32KB per core]
- L2 Cache: [TODO: e.g., 256KB per core]
- L3 Cache: [TODO: e.g., 12MB shared]

**CPU Features:**
```bash
# Check AVX2 support
grep -o 'avx2' /proc/cpuinfo | wc -l
# Should show number of cores with AVX2

# Check cache sizes
lscpu | grep cache
```

### Software

- **OS:** [TODO: e.g., Ubuntu 22.04 LTS]
- **Compiler:** [TODO: e.g., GCC 11.3.0]
- **Build Type:** Release with `-O3 -march=native -flto`
- **CMake Version:** [TODO]

### Build Commands

```bash
# Baseline (before optimizations)
git checkout [baseline-commit]
cmake -B build-baseline -DCMAKE_BUILD_TYPE=Release
cmake --build build-baseline -j

# Optimized (after optimizations)
git checkout master
cmake -B build-optimized -DCMAKE_BUILD_TYPE=Release
cmake --build build-optimized -j
```

---

## Microbenchmarks

### 1. Matrix Multiplication (OPT-1A)

**Test:** Multiply two 512×512 matrices

**Baseline Implementation:** Naive i-j-k loop order

**Optimized Implementation:** Blocked i-k-j with 64×64 tiles

**Command:**
```bash
./build/golomb_benchmarks --benchmark_filter=BM_MatrixMultiply
```

**Results:**

| Version | Time (ms) | Speedup | Notes |
|---------|-----------|---------|-------|
| Baseline | [TODO] | 1.0× | Naive i-j-k |
| Optimized | [TODO] | **[TODO]** | 64×64 blocking + i-k-j |

**Expected:** 10-15× speedup

**Analysis:**
- [ ] Cache miss rate (use `perf stat -e cache-misses,cache-references`)
- [ ] L1/L2/L3 hit rates
- [ ] Memory bandwidth utilization

---

### 2. AVX2 Activations (OPT-1C)

**Test:** Apply activation functions to 100k elements

**Command:**
```bash
./build/golomb_benchmarks --benchmark_filter=BM_Activation
```

**Results:**

| Function | Baseline (μs) | Optimized (μs) | Speedup | Notes |
|----------|---------------|----------------|---------|-------|
| ReLU | [TODO] | [TODO] | **Target: 3-4×** | AVX2 `_mm256_max_pd` |
| Tanh | [TODO] | [TODO] | **Target: 5-6×** | Padé approximation |
| Sigmoid | [TODO] | [TODO] | **Target: 4-5×** | Fast exp |

**Analysis:**
- [ ] Verify AVX2 instructions used (check assembly: `objdump -d`)
- [ ] Scalar fallback performance (compile without `-march=native`)

---

### 3. MCTS Legal Actions (OPT-2A, OPT-2B)

**Test:** Generate legal actions for n=10, ub=100 (1000 iterations)

**Baseline:** O(n×ub) with `std::find`

**Optimized:** O(ub) with `std::unordered_set` + caching

**Command:**
```bash
./build/golomb_benchmarks --benchmark_filter=BM_LegalActions
```

**Results:**

| Version | Time (ms) | Speedup | Notes |
|---------|-----------|---------|-------|
| Baseline (std::find) | [TODO] | 1.0× | O(n×ub) |
| Hash table (OPT-2A) | [TODO] | **[TODO]** | O(ub) |
| + Caching (OPT-2B) | [TODO] | **[TODO]** | Cached on revisits |

**Expected:** 5-10× speedup (combined)

---

### 4. Adam Optimizer (OPT-2C)

**Test:** Adam step with 1M parameters

**Baseline:** 3 separate loops (m, v, param update)

**Optimized:** Single fused loop

**Command:**
```bash
./build/golomb_benchmarks --benchmark_filter=BM_AdamOptimizer
```

**Results:**

| Version | Time (ms) | Speedup | Notes |
|---------|-----------|---------|-------|
| Baseline (3 loops) | [TODO] | 1.0× | Poor cache locality |
| Fused loop | [TODO] | **Target: 1.5-2×** | Single pass |

**Analysis:**
- [ ] Cache miss rate before/after
- [ ] Memory bandwidth utilization

---

### 5. Tensor Operations (OPT-1B, OPT-4A, OPT-4B)

**Test:** Tensor arithmetic (10k iterations of 100×100 tensors)

**Command:**
```bash
./build/golomb_benchmarks --benchmark_filter=BM_TensorOps
```

**Results:**

| Operation | Baseline (ms) | Optimized (ms) | Speedup | Optimization |
|-----------|---------------|----------------|---------|--------------|
| Addition | [TODO] | [TODO] | **[TODO]** | OPT-1B (no copy) |
| Subtraction | [TODO] | [TODO] | **[TODO]** | OPT-1B (no copy) |
| Move assignment | [TODO] | [TODO] | **[TODO]** | OPT-4A (move semantics) |

**Expected:** 1.5-2× speedup

---

## System Benchmarks

### 6. MCTS Builder (End-to-End)

**Test:** Build Golomb ruler (n=8, ub=120, 5000 iterations)

**Command:**
```bash
# Baseline
./build-baseline/golomb_cli --order 8 --ub 120 --mode mcts --iters 5000

# Optimized
./build-optimized/golomb_cli --order 8 --ub 120 --mode mcts --iters 5000
```

**Results:**

| Version | Time (s) | Ruler Length | Speedup | Notes |
|---------|----------|--------------|---------|-------|
| Baseline | [TODO] | [TODO] | 1.0× | Before optimizations |
| Optimized | [TODO] | [TODO] | **Target: 5-10×** | All MCTS optimizations |

**Optimizations Applied:**
- OPT-2A: Hash table for legal actions
- OPT-2B: Cached legal actions
- OPT-3B: False sharing fix (parallel MCTS)
- OPT-4C: Object pooling (when integrated)

---

### 7. Parallel MCTS (OPT-3B)

**Test:** Parallel MCTS with 8 threads (n=8, ub=120, 10k iterations)

**Command:**
```bash
./build/golomb_cli --order 8 --ub 120 --mode mcts --iters 10000 --threads 8
```

**Results:**

| Version | Time (s) | Speedup (vs 1 thread) | Parallel Efficiency | Notes |
|---------|----------|----------------------|---------------------|-------|
| Baseline | [TODO] | [TODO] | [TODO]% | False sharing |
| Optimized | [TODO] | **Target: 6-7×** | **~80-90%** | alignas(64) + atomics |

**Parallel Efficiency Formula:**
```
efficiency = (speedup / num_threads) × 100%
```

**Expected:** 2-3× improvement over baseline parallel version

**Analysis:**
- [ ] Cache coherence traffic (use `perf stat -e bus-cycles`)
- [ ] Thread scaling (1, 2, 4, 8, 16 threads)

---

### 8. Neural Network Training

**Test:** Train on 100 episodes of self-play

**Command:**
```bash
# TODO: Add training benchmark once NN integration complete
```

**Results:**

| Component | Baseline (ms) | Optimized (ms) | Speedup | Optimizations |
|-----------|---------------|----------------|---------|---------------|
| Forward pass | [TODO] | [TODO] | **Target: 3-5×** | AVX2 + blocked matmul |
| Backward pass | [TODO] | [TODO] | **Target: 3-5×** | AVX2 + blocked matmul |
| Optimizer step | [TODO] | [TODO] | **Target: 1.5-2×** | Loop fusion |
| **Total epoch** | [TODO] | [TODO] | **Target: 3-4×** | Combined |

---

## Profiling

### CPU Profiling with perf

**Baseline Hotspots:**
```bash
perf record -g ./build-baseline/golomb_cli --order 8 --ub 120 --mode mcts --iters 1000
perf report
```

**Expected Hotspots (Baseline):**
- [ ] `get_legal_actions()` - Fixed by OPT-2A/2B
- [ ] Matrix multiplication - Fixed by OPT-1A
- [ ] Activation functions - Fixed by OPT-1C
- [ ] Adam optimizer - Fixed by OPT-2C

**Optimized Profile:**
```bash
perf record -g ./build-optimized/golomb_cli --order 8 --ub 120 --mode mcts --iters 1000
perf report
```

**Analysis:**
- [ ] Verify hotspots eliminated
- [ ] Identify new bottlenecks

---

### Cache Analysis

**Baseline Cache Misses:**
```bash
perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses \
  ./build-baseline/golomb_cli --order 8 --ub 120 --mode mcts --iters 1000
```

**Expected Metrics:**

| Metric | Baseline | Optimized | Improvement | Optimization |
|--------|----------|-----------|-------------|--------------|
| L1 miss rate | [TODO]% | **Target: <5%** | [TODO] | OPT-1A (blocking) |
| L2 miss rate | [TODO]% | **Target: <10%** | [TODO] | OPT-1A (blocking) |
| Cache references | [TODO]M | [TODO]M | **[TODO]** | OPT-2C (fusion) |
| Cache misses | [TODO]M | [TODO]M | **[TODO]** | OPT-1A + OPT-2C |

---

### Memory Bandwidth

**Test memory bandwidth utilization:**

```bash
perf stat -e cycles,instructions,mem_load_retired.l1_miss,mem_load_retired.l2_miss \
  ./build/golomb_cli --order 10 --ub 200 --mode mcts --iters 5000
```

**Analysis:**
- [ ] Bytes per cycle (should increase with blocking)
- [ ] L1/L2 miss rates (should decrease)

---

## Compilation Flags Impact

**Test:** Compare different optimization levels

| Flags | Time (s) | Speedup | Notes |
|-------|----------|---------|-------|
| `-O0` | [TODO] | 1.0× | No optimization |
| `-O2` | [TODO] | **~5-7×** | Standard optimization |
| `-O3` | [TODO] | **~6-8×** | Aggressive optimization |
| `-O3 -march=native` | [TODO] | **~8-12×** | + AVX2/SSE |
| `-O3 -march=native -flto` | [TODO] | **~10-15×** | + LTO |
| **Full (OPT-5A)** | [TODO] | **~12-20×** | All flags |

**Full flags:** `-O3 -march=native -mtune=native -funroll-loops -ffast-math -flto`

---

## Compiler Comparison

| Compiler | Version | Time (s) | Speedup | Notes |
|----------|---------|----------|---------|-------|
| GCC | [TODO] | [TODO] | 1.0× | Baseline |
| Clang | [TODO] | [TODO] | **[TODO]** | Often better vectorization |
| ICC (Intel) | [TODO] | [TODO] | **[TODO]** | Best for AVX2 |

---

## Expected vs Actual Performance

### Component-Level Speedups

| Optimization | Component | Expected | Actual | Status |
|--------------|-----------|----------|--------|--------|
| OPT-1A | Matrix Multiply | 10-15× | [TODO] | ⏳ |
| OPT-1B | Tensor Ops | 2× | [TODO] | ⏳ |
| OPT-1C | ReLU | 3-4× | [TODO] | ⏳ |
| OPT-1C | Tanh | 5-6× | [TODO] | ⏳ |
| OPT-1C | Sigmoid | 4-5× | [TODO] | ⏳ |
| OPT-2A | Legal Actions | 5-10× | [TODO] | ⏳ |
| OPT-2B | Cached Actions | 2-3× | [TODO] | ⏳ |
| OPT-2C | Adam Optimizer | 1.5-2× | [TODO] | ⏳ |
| OPT-3B | Parallel MCTS | 2-3× | [TODO] | ⏳ |
| OPT-4A | Move Semantics | 1.5-2× | [TODO] | ⏳ |
| OPT-4B | Pointer Cache | 1.1-1.15× | [TODO] | ⏳ |
| OPT-4C | Object Pool | 2-3× | [TODO] | ⏳ |
| OPT-5A | Compiler Flags | 1.2-1.3× | [TODO] | ⏳ |
| OPT-5B | No Bounds Check | 1.05-1.1× | [TODO] | ⏳ |

### Overall Speedup Calculation

**Formula:**
```
Overall Speedup ≈ ∏ (component_speedup)^(component_weight)
```

**Example:**
- If MCTS is 50% of runtime and speeds up 7×
- If NN is 30% of runtime and speeds up 4×
- If other is 20% of runtime and speeds up 2×

**Expected:** 7^0.5 × 4^0.3 × 2^0.2 ≈ **4.8×** total

**Target:** **15-25× overall** (based on CSAPP principles)

---

## Regression Tests

Ensure optimizations don't break correctness:

```bash
# Run all tests
./scripts/run_tests.sh

# Verify ruler correctness
./build/golomb_cli --order 8 --ub 120 --mode mcts --iters 10000 --verify
```

**Checklist:**
- [ ] All unit tests pass
- [ ] MCTS produces valid rulers
- [ ] NN convergence unchanged
- [ ] Deterministic results (with fixed seed)

---

## Next Steps

### 1. Run Baseline Benchmarks
```bash
# Build baseline version
git checkout [pre-optimization-commit]
cmake -B build-baseline -DCMAKE_BUILD_TYPE=Release
cmake --build build-baseline -j

# Run benchmarks
./build-baseline/golomb_benchmarks --benchmark_format=json > baseline.json
```

### 2. Run Optimized Benchmarks
```bash
# Build optimized version
git checkout master
cmake -B build-optimized -DCMAKE_BUILD_TYPE=Release
cmake --build build-optimized -j

# Run benchmarks
./build-optimized/golomb_benchmarks --benchmark_format=json > optimized.json
```

### 3. Compare Results
```bash
# Use Google Benchmark compare script
./scripts/compare_benchmarks.py baseline.json optimized.json
```

### 4. Profile with perf
```bash
# CPU hotspots
perf record -g ./build-optimized/golomb_cli --order 8 --ub 120 --mode mcts --iters 5000
perf report

# Cache analysis
perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses \
  ./build-optimized/golomb_cli --order 8 --ub 120 --mode mcts --iters 5000
```

### 5. Integrate Object Pool (OPT-4C)
- [ ] Update `mcts_build()` to use `NodePool`
- [ ] Add `thread_local` pool for parallel version
- [ ] Benchmark allocation overhead reduction

### 6. Validate on Different Hardware
- [ ] Test on AMD CPUs (different cache hierarchy)
- [ ] Test on Intel CPUs with different AVX support
- [ ] Test on ARM (Apple M1/M2) - may need NEON instead of AVX2

---

## Performance Dashboard

Track performance over time:

```bash
# Run benchmark suite
./build/golomb_benchmarks --benchmark_format=json | tee results/$(git rev-parse --short HEAD).json

# Generate HTML report
./scripts/generate_perf_report.py results/*.json > docs/perf_report.html
```

**Example Dashboard:**

![Performance Dashboard](perf_dashboard.png)

---

## References

- [Google Benchmark User Guide](https://github.com/google/benchmark/blob/main/docs/user_guide.md)
- [Linux perf Tutorial](https://perf.wiki.kernel.org/index.php/Tutorial)
- [Intel VTune Profiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html)
- [CSAPP Performance Lab](http://csapp.cs.cmu.edu/3e/labs.html)

---

**Status:** ⏳ **Awaiting baseline and optimized benchmark runs**

**Next Action:** Compile both versions and run benchmark suite

**Contact:** Nicolas Martinez (Gotman08)
