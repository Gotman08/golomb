# Golomb Ruler Optimization

[![CI](https://github.com/Gotman08/golomb/actions/workflows/ci.yml/badge.svg)](https://github.com/Gotman08/golomb/actions/workflows/ci.yml)
[![Documentation](https://github.com/Gotman08/golomb/actions/workflows/docs.yml/badge.svg)](https://github.com/Gotman08/golomb/actions/workflows/docs.yml)
[![Benchmarks](https://github.com/Gotman08/golomb/actions/workflows/benchmark.yml/badge.svg)](https://github.com/Gotman08/golomb/actions/workflows/benchmark.yml)
[![Release](https://github.com/Gotman08/golomb/actions/workflows/release.yml/badge.svg)](https://github.com/Gotman08/golomb/actions/workflows/release.yml)
[![GitHub release](https://img.shields.io/github/v/release/Gotman08/golomb)](https://github.com/Gotman08/golomb/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://conventionalcommits.org)

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
- C++20 compiler (GCC 10+, Clang 13+)
- Git (for CPM package manager)

### Quick Start

```bash
./scripts/build.sh
./scripts/run_tests.sh

# Or manually
mkdir build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

## Pre-built Binaries

Download pre-compiled binaries from the [latest successful CI run](https://github.com/Gotman08/golomb/actions/workflows/ci.yml):
- **Linux**: `golomb_cli-Linux` artifact

Extract the executable and run directly.

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

## Documentation

Full API documentation is automatically generated with Doxygen and deployed to GitHub Pages:
- **[View Online Documentation](https://gotman08.github.io/golomb/)**

Documentation is automatically updated on every push to the `master` branch.

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

## Performance Benchmarks

This project uses Google Benchmark for performance testing. Benchmarks run automatically on every push to master, comparing performance with previous commits to detect regressions.

**View benchmark results:**
- [Latest benchmark run](https://github.com/Gotman08/golomb/actions/workflows/benchmark.yml)
- Historical benchmark data is stored in the `gh-pages` branch

**Run benchmarks locally:**
```bash
./build/golomb_benchmarks
```

**Benchmark categories:**
- Core operations (rule building, validation)
- Greedy seed generation (various sizes)
- Evolutionary algorithm (different population sizes and iterations)
- MCTS (various iteration counts and exploration parameters)
- Algorithm comparisons (Greedy vs Evolutionary vs MCTS)

Performance regressions >10% will trigger warnings in pull requests.

## Claude Code Integration (MCP)

This project is configured with [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers for enhanced development with Claude Code in VS Code.

**Configured MCP servers:**
- **GitHub**: Manage issues, PRs, and workflows
- **Git**: Advanced git operations (log, diff, blame)
- **Filesystem**: Extended file system access
- **Memory**: Persistent notes between Claude sessions

**Setup instructions:** See [docs/MCP_GUIDE.md](docs/MCP_GUIDE.md)

**Prerequisites:** Node.js 18+

## Contributing

We welcome contributions! This project uses [Conventional Commits](https://conventionalcommits.org/) for automated versioning and changelog generation.

**Quick start:**
1. Fork and clone the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make your changes following our coding standards
4. Use conventional commit messages (see below)
5. Run tests and ensure they pass
6. Submit a pull request

**Commit message format:**
```bash
# Feature (minor version bump)
git commit -m "feat(core): add parallel distance calculation"

# Bug fix (patch version bump)
git commit -m "fix(mcts): correct PUCT formula"

# Breaking change (major version bump)
git commit -m "feat!: redesign API

BREAKING CHANGE: RuleState constructor signature changed"
```

**For detailed guidelines:** See [CONTRIBUTING.md](CONTRIBUTING.md)

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
