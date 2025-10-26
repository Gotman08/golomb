cmake_minimum_required(VERSION 3.22)
project(golomb-opt LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_TESTING "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)

# Include CPM for package management
include(cmake/CPM.cmake)

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
target_link_libraries(golomb_mcts PUBLIC golomb_core)

# Exact solver library (stub)
add_library(golomb_exact
  src/exact/exact_stub.cpp
)
target_link_libraries(golomb_exact PUBLIC golomb_core)

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
