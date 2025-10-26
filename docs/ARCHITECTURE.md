# Architecture Documentation

## Module Overview

The Golomb ruler optimization project is structured into four main layers:

### 1. Core Layer (`core/`, `utils/`)

**Purpose**: Fundamental data structures and validation.

**Components**:
- `DistBitset`: Dynamic bitset (vector of uint64_t) for O(1) distance checking
  - `set(d)`: Mark distance d as used
  - `test(d)`: Check if distance d is already used
  - `can_add_mark()`: Test if mark can be added without conflicts
  - `add_mark()`: Add mark and update all new distances atomically

- `RuleState`: Combines sorted marks vector with DistBitset
  - Enables efficient incremental construction
  - Guarantees consistency between marks and distance set

- `golomb.hpp`: High-level utilities
  - `is_valid_rule()`: O(n²) validation for final verification
  - `greedy_seed()`: Simple constructive heuristic
  - `try_add()`: Safe mark addition with rollback on failure

**Dependencies**: None (self-contained)

### 2. Heuristics Layer (`heuristics/`)

**Purpose**: Fast approximate solutions using metaheuristics.

**Components**:
- `evolutionary_search()`:
  - Population-based optimization (default pop=64)
  - Mutation: random mark repositioning
  - Crossover: blend parents' mark positions
  - Elitism: keep top 25% each generation
  - Fitness: negative length with heavy penalty for invalid rules
  - Post-processing: local search on best solution

- `hill_climb()`:
  - Iterative improvement by mark repositioning
  - Acceptance: only strictly better (shorter) valid rulers
  - Neighborhood: single-mark moves to random positions

**Dependencies**: `core/`, `utils/random`

**Extension Points**:
- TODO: Adaptive mutation rates based on population diversity
- TODO: Simulated annealing with temperature scheduling
- TODO: Tabu search with recency-based memory

### 3. MCTS Layer (`mcts/`)

**Purpose**: Balanced exploration using Monte Carlo Tree Search.

**Components**:
- `MCTSNode`:
  - `state`: Partial ruler state (RuleState)
  - `children`: Map from action (position) to child node
  - `N`, `W`: Visit count and total value
  - `P`: Prior probabilities per action (uniform by default)

- `mcts_build()`:
  - Selection: PUCT formula `Q + c_puct * P * sqrt(parent_N) / (1 + N)`
  - Expansion: initialize children for all legal actions
  - Simulation: evaluate leaf with heuristic value
  - Backpropagation: update N and W along path
  - Result extraction: follow most-visited path

**Dependencies**: `core/`, `utils/random`

**Extension Points**:
- TODO: Neural network for `policy_priors()` and `evaluate_leaf()`
- TODO: Virtual loss for parallel tree search
- TODO: Progressive widening to focus on promising actions
- TODO: Graphviz export for tree visualization

### 4. Exact Layer (`exact/`)

**Purpose**: Optimal solutions via constraint programming or integer linear programming.

**Current Status**: Stub implementation returning greedy seed.

**Interface**:
- `ExactOptions`: n, ub, timeout_ms
- `ExactResult`: optimal flag, rule, bounds, status message

**Future Implementations**:
- TODO: OR-Tools CP-SAT model
  - Variables: positions[i] ∈ [0, ub] with ordering constraints
  - Constraints: AllDifferent on pairwise distances
  - Objective: minimize positions[n-1]

- TODO: ILP formulation
  - Binary variables: x[i,p] = 1 if mark i at position p
  - Distance variables: d[i,j] = |pos[i] - pos[j]|
  - AllDifferent on distances via big-M constraints

- TODO: Benders decomposition
  - Master: select mark positions
  - Subproblem: verify distance uniqueness
  - Cuts: no-good cuts for invalid configurations

**Dependencies**: `core/` (will require OR-Tools or similar)

### 5. CLI Layer (`cli/`)

**Purpose**: Command-line interface for running solvers.

**Components**:
- `Args`: argument parsing with defaults
- `main.cpp`: dispatch to selected mode (heur/mcts/exact)
- Output: ruler, length, validity, elapsed time

**Dependencies**: All layers (core, heuristics, mcts, exact)

## Data Flow

```
User Input (CLI)
    ↓
Mode Selection
    ↓
┌───────────┬──────────┬─────────────┐
│  Heuristic│   MCTS   │    Exact    │
└───────────┴──────────┴─────────────┘
    ↓            ↓            ↓
    └────────────┴────────────┘
              ↓
        Core Validation
              ↓
          Output Result
```

## Build System

- **CMake 3.22+**: modular library structure
- **CPM.cmake**: FetchContent wrapper for dependencies
  - Catch2 3.5.0 (tests)
  - Google Benchmark 1.8.3 (benchmarks)
- **Compiler flags**:
  - C++20 standard
  - `-Wall -Wextra -Wpedantic` (strict warnings)
  - Debug: AddressSanitizer + UndefinedBehaviorSanitizer

## Testing Strategy

### Unit Tests (`tests/test_golomb_core.cpp`)
- DistBitset: set/test/clear, can_add_mark, add_mark
- Validation: known valid/invalid rulers
- Greedy seed: correctness, uniqueness

### Benchmarks (`benchmarks/bench_build_partial_rule.cpp`)
- Incremental mark addition performance
- Validation overhead for varying ruler sizes

## Performance Considerations

1. **Bitset representation**: O(1) distance checks vs O(n) set lookup
2. **Sorted marks**: binary search for insertion (O(log n))
3. **Mutation strategy**: avoid full validation per candidate
4. **MCTS expansion**: lazy child creation on first visit
5. **Memory**: tree pruning for long searches (not implemented)

## Code Style

- **Javadoc comments**: all public APIs
- **Better Comments style**:
  - `// TODO:` - future work
  - `// FIXME:` - known issues
  - `// NOTE:` - important implementation details
  - `// WARNING:` - potential pitfalls
- **Naming**: lowerCamelCase (functions/vars), PascalCase (types)
- **Formatting**: clang-format (LLVM, 100 column limit)
- **Linting**: clang-tidy (modernize, readability, performance, bugprone)

## Extension Guidelines

### Adding a New Heuristic
1. Add function signature to `include/heuristics/<name>.hpp`
2. Implement in `src/heuristics/<name>.cpp`
3. Use `RuleState` for efficient incremental construction
4. Add tests in `tests/test_<name>.cpp`
5. Link new source in `CMakeLists.txt` (golomb_heur target)

### Integrating Neural Networks for MCTS
1. Add dependency: PyTorch C++ (libtorch) or ONNX Runtime
2. Replace `compute_policy_priors_uniform()` with network forward pass
3. Replace `evaluate_leaf()` with value network output
4. Add model loading in `mcts_build()` initialization
5. Consider GPU acceleration for batch inference

### Adding Exact Solver
1. Add OR-Tools to CMakeLists via CPM or find_package
2. Implement CP-SAT model in `src/exact/exact_cpsat.cpp`
3. Update `solve_exact_stub()` to call real solver
4. Add timeout handling and incremental bounds
5. Test on small instances (n=4-6) with known optima

## Maintenance Notes

- **Sanitizers**: Run tests with ASAN/UBSAN enabled (default in Debug)
- **Valgrind**: For leak detection on production builds
- **Profiling**: Use `perf` or Instruments for hotspot analysis
- **Versioning**: Semantic versioning (MAJOR.MINOR.PATCH)
- **CI**: GitHub Actions on Ubuntu (Debug + Release builds)
