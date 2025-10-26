# Contributing to Golomb Ruler Optimization

Thank you for your interest in contributing to the Golomb Ruler Optimization project! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Commit Message Convention](#commit-message-convention)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Performance Considerations](#performance-considerations)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Assume good intentions

## Getting Started

### Prerequisites

- CMake 3.22+
- C++20 compatible compiler (GCC 10+, Clang 13+, MSVC 2019+)
- Git for version control
- (Optional) Node.js 18+ for MCP servers

### Setting Up Development Environment

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/golomb.git
   cd golomb
   ```

2. **Build the project:**
   ```bash
   mkdir build
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
   cmake --build build -j
   ```

3. **Run tests:**
   ```bash
   ctest --test-dir build --output-on-failure
   ```

4. **Run benchmarks (optional):**
   ```bash
   ./build/golomb_benchmarks
   ```

### MCP Setup (Optional)

If you're using Claude Code with VS Code, configure MCP servers for enhanced development experience. See [docs/MCP_GUIDE.md](docs/MCP_GUIDE.md) for details.

## Development Workflow

1. **Create a feature branch:**
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes:**
   - Write code following our [coding standards](#coding-standards)
   - Add tests for new functionality
   - Update documentation if needed

3. **Run local checks:**
   ```bash
   # Format code
   ./scripts/format.sh

   # Run tests
   ./scripts/run_tests.sh

   # Run benchmarks (if performance-related)
   ./build/golomb_benchmarks
   ```

4. **Commit your changes** using [conventional commits](#commit-message-convention)

5. **Push and create a pull request**

## Commit Message Convention

This project uses [Conventional Commits](https://www.conventionalcommits.org/) for automated versioning and changelog generation.

### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

- **feat:** A new feature (triggers minor version bump)
- **fix:** A bug fix (triggers patch version bump)
- **perf:** Performance improvements (triggers patch version bump)
- **refactor:** Code refactoring without changing behavior (triggers patch version bump)
- **docs:** Documentation changes only (no version bump)
- **style:** Code style changes (formatting, etc.) (no version bump)
- **test:** Adding or updating tests (no version bump)
- **build:** Changes to build system or dependencies (triggers patch version bump)
- **ci:** CI/CD configuration changes (no version bump)
- **chore:** Other changes (no version bump)

### Breaking Changes

For breaking changes that trigger a major version bump:

```
feat!: remove support for old API

BREAKING CHANGE: The old API has been completely removed.
Users must migrate to the new API.
```

### Examples

```bash
# Feature addition (version 1.0.0 → 1.1.0)
git commit -m "feat(mcts): add parallel tree search with virtual loss"

# Bug fix (version 1.0.0 → 1.0.1)
git commit -m "fix(core): correct distance calculation in bitset"

# Performance improvement (version 1.0.0 → 1.0.1)
git commit -m "perf(evo): optimize mutation operator using cache"

# Documentation (no version change)
git commit -m "docs: add examples for MCTS usage"

# Breaking change (version 1.0.0 → 2.0.0)
git commit -m "feat!: redesign core API for better performance

BREAKING CHANGE: RuleState constructor now requires max_distance parameter"
```

### Scopes

Common scopes in this project:
- `core` - Core algorithms and data structures
- `mcts` - Monte Carlo Tree Search implementation
- `evo` - Evolutionary algorithms
- `exact` - Exact solver interfaces
- `cli` - Command-line interface
- `benchmarks` - Performance benchmarks
- `docs` - Documentation
- `ci` - CI/CD workflows

## Pull Request Process

1. **Ensure all checks pass:**
   - All tests pass on both Linux and Windows
   - Code follows formatting guidelines
   - No compiler warnings
   - Benchmarks show no significant regressions (>10%)

2. **Update documentation:**
   - Add/update docstrings for new functions
   - Update README if adding user-facing features
   - Update ARCHITECTURE.md if changing design

3. **Describe your changes:**
   - Write a clear PR description
   - Reference related issues (e.g., "Fixes #123")
   - Explain the motivation and approach

4. **Review process:**
   - Address review comments
   - Keep the PR focused (one feature/fix per PR)
   - Rebase on master if needed

5. **Automated checks:**
   - CI builds on Ubuntu and Windows
   - Unit tests execution
   - Static analysis (clang-tidy, cppcheck)
   - Benchmark comparison (for performance-related changes)

## Coding Standards

### C++ Style

- **Standard:** C++20
- **Formatting:** Follow `.clang-format` configuration
- **Naming:**
  - Classes/Structs: `PascalCase`
  - Functions: `snake_case`
  - Variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Namespaces: `snake_case`

### Code Organization

```cpp
// Example header file structure
#pragma once

#include <standard_library>  // Standard library first
#include "project/headers.hpp"  // Then project headers

namespace golomb {

/**
 * @brief Brief description
 *
 * Detailed description if needed.
 *
 * @param n Description of parameter
 * @return Description of return value
 */
std::vector<int> function_name(int n);

}  // namespace golomb
```

### Best Practices

- Use RAII for resource management
- Prefer `const` correctness
- Use `std::vector` and standard containers
- Avoid raw pointers (use smart pointers if needed)
- Write self-documenting code with good variable names
- Add comments for complex algorithms
- Keep functions focused and small

## Testing Guidelines

### Writing Tests

Use Catch2 for unit tests:

```cpp
#include <catch2/catch_test_macros.hpp>
#include "your/header.hpp"

TEST_CASE("Descriptive test name", "[tag]") {
  SECTION("specific scenario") {
    // Arrange
    auto input = create_test_data();

    // Act
    auto result = function_under_test(input);

    // Assert
    REQUIRE(result == expected_value);
  }
}
```

### Test Coverage

- Write tests for all public APIs
- Test edge cases (empty inputs, large inputs, invalid inputs)
- Test error handling
- For algorithms, test correctness and properties

### Running Tests

```bash
# All tests
ctest --test-dir build --output-on-failure

# Specific test
./build/golomb_tests "test name"

# With verbose output
./build/golomb_tests --success
```

## Performance Considerations

### Benchmarking

When making performance-related changes:

1. **Measure before and after:**
   ```bash
   ./build/golomb_benchmarks --benchmark_filter=YourBenchmark
   ```

2. **Check for regressions:**
   - CI will automatically compare benchmarks
   - Regressions >10% will trigger warnings

3. **Profile if needed:**
   ```bash
   # Linux with perf
   perf record ./build/golomb_benchmarks
   perf report

   # Or use valgrind/cachegrind
   valgrind --tool=cachegrind ./build/golomb_cli
   ```

### Writing Benchmarks

Add benchmarks for performance-critical code:

```cpp
static void BM_YourFunction(benchmark::State& state) {
  int n = state.range(0);

  for (auto _ : state) {
    auto result = your_function(n);
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK(BM_YourFunction)->Range(8, 64);
```

## Questions or Need Help?

- Open an issue for bugs or feature requests
- Use discussions for questions and ideas
- Check existing issues and PRs before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Golomb Ruler Optimization! 🎉
