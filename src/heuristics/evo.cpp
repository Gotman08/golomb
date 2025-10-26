#include "heuristics/evo.hpp"
#include "core/golomb.hpp"
#include "heuristics/local_search.hpp"
#include "utils/random.hpp"
#include <algorithm>

namespace golomb {

namespace {

// NOTE: fitness = negative length + penalty for conflicts
int evaluate_fitness(const std::vector<int>& marks) {
  if (!is_valid_rule(marks)) {
    return -10000;  // Heavy penalty for invalid rulers
  }
  return -length(marks);
}

// NOTE: mutate by moving one random mark to a new position
std::vector<int> mutate(const std::vector<int>& marks, int ub, RNG& rng) {
  if (marks.size() < 2) {
    return marks;
  }

  std::vector<int> mutated = marks;
  int idx = rng.uniform_int(1, static_cast<int>(marks.size()) - 1);  // Don't move 0
  int new_pos = rng.uniform_int(1, ub - 1);

  mutated[idx] = new_pos;
  std::sort(mutated.begin(), mutated.end());
  mutated.erase(std::unique(mutated.begin(), mutated.end()), mutated.end());

  return mutated;
}

// NOTE: simple crossover - take first half from parent1, second from parent2
std::vector<int> crossover(const std::vector<int>& p1, const std::vector<int>& p2, RNG& rng) {
  std::vector<int> child;
  child.reserve(p1.size());

  for (size_t i = 0; i < p1.size(); ++i) {
    child.push_back(rng.uniform_real() < 0.5 ? p1[i] : p2[i]);
  }

  std::sort(child.begin(), child.end());
  child.erase(std::unique(child.begin(), child.end()), child.end());

  return child;
}

}  // namespace

std::vector<int> evolutionary_search(int n, int ub, int pop, int iters) {
  RNG rng;

  // Initialize population with greedy seeds and random variations
  std::vector<std::vector<int>> population;
  for (int i = 0; i < pop; ++i) {
    auto seed = greedy_seed(n, ub);
    if (i > 0) {
      seed = mutate(seed, ub, rng);
    }
    population.push_back(seed);
  }

  std::vector<int> best = population[0];
  int best_fitness = evaluate_fitness(best);

  // TODO: implement proper selection, elitism, adaptive parameters
  for (int iter = 0; iter < iters; ++iter) {
    // Evaluate population
    std::vector<std::pair<int, int>> fitness_idx;
    for (size_t i = 0; i < population.size(); ++i) {
      int fit = evaluate_fitness(population[i]);
      fitness_idx.push_back({fit, static_cast<int>(i)});
      if (fit > best_fitness) {
        best = population[i];
        best_fitness = fit;
      }
    }

    // Sort by fitness (descending)
    std::sort(fitness_idx.begin(), fitness_idx.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Create new population: keep top 25%, generate rest via mutation/crossover
    std::vector<std::vector<int>> new_pop;
    int elite_count = pop / 4;
    for (int i = 0; i < elite_count; ++i) {
      new_pop.push_back(population[fitness_idx[i].second]);
    }

    while (static_cast<int>(new_pop.size()) < pop) {
      int p1_idx = fitness_idx[rng.uniform_int(0, elite_count - 1)].second;
      int p2_idx = fitness_idx[rng.uniform_int(0, elite_count - 1)].second;

      auto child = crossover(population[p1_idx], population[p2_idx], rng);
      child = mutate(child, ub, rng);
      new_pop.push_back(child);
    }

    population = std::move(new_pop);
  }

  // Apply local search to best solution
  return hill_climb(best, ub, 1000);
}

}  // namespace golomb
