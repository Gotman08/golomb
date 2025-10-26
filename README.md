# Golomb Ruler Optimization

C++20 project for finding optimal Golomb rulers using multiple approaches: heuristics, MCTS, and exact methods.

## Overview

A **Golomb ruler** is a set of marks at integer positions such that all pairwise distances are distinct. This project implements three optimization layers:

1. **Core** - Bitset-based distance tracking and validation (O(1) distance checks)
2. **Heuristics** - Evolutionary algorithm with local search for quick solutions
3. **MCTS** - Monte Carlo Tree Search with PUCT selection for balanced exploration
4. **Exact** - Interface for CP-SAT/ILP solvers (stub implementation)

## Features

- **Efficient distance tracking**: Bitset representation for O(1) conflict detection
- **Multiple solvers**: Evolutionary, MCTS, exact (stub)
- **Modular architecture**: Clean separation between core, heuristics, MCTS, and exact layers
- **Comprehensive testing**: Catch2 unit tests and Google Benchmark performance tests
- **Modern C++**: C++20 standard with strict warnings and sanitizers in debug builds

## Building

### Prerequisites

- CMake 3.22+
- C++20 compiler (GCC 10+, Clang 13+, MSVC 2019+)
- Git (for CPM package manager)

### Quick Start

```bash
# Unix-like systems
./scripts/build.sh
./scripts/run_tests.sh

# Or manually
mkdir build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

### Windows

```powershell
mkdir build
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release
```

## Usage

```bash
./build/golomb_cli --order 8 --ub 120 --mode mcts --iters 5000

# Options:
#   --order <n>       Number of marks (default: 8)
#   --ub <value>      Upper bound for positions (default: 120)
#   --mode <mode>     heur|mcts|exact (default: heur)
#   --iters <n>       Iterations for heur/mcts (default: 1000)
#   --c-puct <value>  PUCT exploration constant (default: 1.4)
#   --timeout <ms>    Exact solver timeout (default: 10000)
```

## Architecture

```
core/              - Distance bitset, validation, state management
utils/             - RNG utilities
heuristics/        - Evolutionary algorithm, local search
mcts/              - PUCT-based MCTS with hooks for policy/value networks
exact/             - Stub interface for CP-SAT/ILP/Benders
cli/               - Command-line interface
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for detailed design documentation.

## Testing

```bash
# Run all tests
./scripts/run_tests.sh

# Run benchmarks
./build/golomb_benchmarks
```

## Code Formatting

```bash
./scripts/format.sh
```

## Future Work

- [ ] Integrate OR-Tools CP-SAT solver for exact solving
- [ ] Add neural network policy/value for MCTS
- [ ] Implement parallel MCTS with virtual loss
- [ ] Export MCTS tree to Graphviz for visualization
- [ ] Add ILP formulation with distance variables
- [ ] Implement Benders decomposition for large instances

## License

MIT License - See [LICENSE](LICENSE)

## References

- [Golomb Ruler on Wikipedia](https://en.wikipedia.org/wiki/Golomb_ruler)
- [Known Optimal Golomb Rulers](http://www.research.ibm.com/people/s/shearer/grule.html)
