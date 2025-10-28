cmake_minimum_required(VERSION 3.22)
project(golomb-opt LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_TESTING "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# Include CPM for package management
include(cmake/CPM.cmake)

# Fetch OR-Tools for exact solving with CP-SAT
CPMAddPackage(
  NAME ortools
  VERSION 9.10
  GITHUB_REPOSITORY google/or-tools
  OPTIONS
    "BUILD_DEPS ON"
    "BUILD_SAMPLES OFF"
    "BUILD_EXAMPLES OFF"
    "BUILD_PYTHON OFF"
    "BUILD_JAVA OFF"
    "BUILD_DOTNET OFF"
    "BUILD_TESTING OFF"
)

# Strict warnings
if(MSVC)
  add_compile_options(/W4 /WX)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Debug flags with sanitizers (Unix-like systems)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT MSVC)
  add_compile_options(-fsanitize=address -fsanitize=undefined)
  add_link_options(-fsanitize=address -fsanitize=undefined)
endif()

# Export compile commands for clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Core library
add_library(golomb_core
  src/core/golomb.cpp
  src/core/bitset_dist.cpp
  src/utils/random.cpp
)
target_include_directories(golomb_core PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Heuristics library
add_library(golomb_heur
  src/heuristics/evo.cpp
  src/heuristics/local_search.cpp
)
target_link_libraries(golomb_heur PUBLIC golomb_core)

# MCTS library
add_library(golomb_mcts
  src/mcts/mcts.cpp
)
target_link_libraries(golomb_mcts PUBLIC golomb_core golomb_nn)

# Exact solver library with CP-SAT
add_library(golomb_exact
  src/exact/exact_stub.cpp
  src/exact/exact_cpsat.cpp
)
target_link_libraries(golomb_exact
  PUBLIC
    golomb_core
  PRIVATE
    ortools::ortools
)

# Neural network library
add_library(golomb_nn
  src/nn/tensor.cpp
  src/nn/activations.cpp
  src/nn/linear.cpp
  src/nn/state_encoder.cpp
  src/nn/golomb_net.cpp
)
target_link_libraries(golomb_nn PUBLIC golomb_core)

# CLI executable
add_executable(golomb_cli
  src/cli/main.cpp
)
target_link_libraries(golomb_cli 
  PRIVATE 
    golomb_core 
    golomb_heur 
    golomb_mcts 
    golomb_exact
)

# Tests
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# Examples
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
