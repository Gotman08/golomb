#include "heuristics/evo.hpp"
#include "core/golomb.hpp"
#include "heuristics/local_search.hpp"
#include "utils/mutations.hpp"
#include "utils/random.hpp"
#include <algorithm>

namespace golomb {

namespace {

// Constants for evolutionary algorithm
constexpr int INVALID_RULER_PENALTY = -10000;
// Keep top 1/4 of population
constexpr int ELITE_FRACTION = 4;
// Hill climb iterations
constexpr int LOCAL_SEARCH_BUDGET = 1000;

// NOTE: fitness = negative length + penalty for conflicts
int evaluate_fitness(const std::vector<int>& marks) {
  if (!is_valid_rule(marks)) {
    return INVALID_RULER_PENALTY; // Heavy penalty for invalid rulers
  }
  return -length(marks);
}

// NOTE: Initialize population with greedy seeds and random variations
// Returns pair of (population, best_individual, best_fitness)
std::tuple<std::vector<std::vector<int>>, std::vector<int>, int>
initialize_population(int n, int ub, int pop_size, RNG& rng) {
  std::vector<std::vector<int>> population;
  population.reserve(static_cast<size_t>(pop_size));

  for (int i = 0; i < pop_size; ++i) {
    auto seed = greedy_seed(n, ub);
    if (i > 0) {
      seed = mutate_single_mark(seed, ub, rng);
    }
    population.push_back(seed);
  }

  std::vector<int> best = population[0];
  int best_fitness = evaluate_fitness(best);

  return {population, best, best_fitness};
}

// NOTE: mutate by moving one random mark to a new position
std::vector<int> mutate(const std::vector<int>& marks, int ub, RNG& rng) {
  return mutate_single_mark(marks, ub, rng);
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

} // namespace

std::vector<int> evolutionary_search(int n, int ub, int pop, int iters) {
  RNG rng;

  // Initialize population
  auto [population, best, best_fitness] = initialize_population(n, ub, pop, rng);

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

    // Create new population: keep elite, generate rest via mutation/crossover
    std::vector<std::vector<int>> new_pop;
    int elite_count = pop / ELITE_FRACTION;
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
  return hill_climb(best, ub, LOCAL_SEARCH_BUDGET);
}

std::vector<int> evolutionary_search_adaptive(int n, int ub, int pop, int iters,
                                              double base_mutation_rate) {
  RNG rng;

  // Initialize population
  auto [population, best, best_fitness] = initialize_population(n, ub, pop, rng);

  int stagnation_count = 0;
  double current_mutation_rate = base_mutation_rate;

  for (int iter = 0; iter < iters; ++iter) {
    // Evaluate population
    std::vector<std::pair<int, int>> fitness_idx;
    int max_fit = INT_MIN;
    int min_fit = INT_MAX;

    for (size_t i = 0; i < population.size(); ++i) {
      int fit = evaluate_fitness(population[i]);
      fitness_idx.push_back({fit, static_cast<int>(i)});
      max_fit = std::max(max_fit, fit);
      min_fit = std::min(min_fit, fit);

      if (fit > best_fitness) {
        best = population[i];
        best_fitness = fit;
        stagnation_count = 0;
      } else {
        stagnation_count++;
      }
    }

    // Calculate population diversity (fitness variance)
    double diversity = (max_fit - min_fit + 1) / static_cast<double>(std::abs(min_fit) + 1);

    // Adapt mutation rate based on diversity and stagnation
    // Low diversity or stagnation -> increase mutation rate
    // High diversity and progress -> decrease mutation rate
    double diversity_factor = 1.0 / (diversity + 0.1); // Inverse relationship
    double stagnation_factor = 1.0 + (stagnation_count / 100.0);

    current_mutation_rate = base_mutation_rate * diversity_factor * stagnation_factor;
    current_mutation_rate = std::min(current_mutation_rate, 0.5); // Cap at 50%
    current_mutation_rate = std::max(current_mutation_rate, 0.01); // Floor at 1%

    // Sort by fitness
    std::sort(fitness_idx.begin(), fitness_idx.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Create new population
    std::vector<std::vector<int>> new_pop;
    int elite_count = pop / ELITE_FRACTION;
    for (int i = 0; i < elite_count; ++i) {
      new_pop.push_back(population[fitness_idx[i].second]);
    }

    while (static_cast<int>(new_pop.size()) < pop) {
      int p1_idx = fitness_idx[rng.uniform_int(0, elite_count - 1)].second;
      int p2_idx = fitness_idx[rng.uniform_int(0, elite_count - 1)].second;

      auto child = crossover(population[p1_idx], population[p2_idx], rng);

      // Apply mutation with adaptive rate
      if (rng.uniform_double(0.0, 1.0) < current_mutation_rate) {
        child = mutate(child, ub, rng);
      }

      new_pop.push_back(child);
    }

    population = std::move(new_pop);
  }

  return hill_climb(best, ub, LOCAL_SEARCH_BUDGET);
}

std::vector<int> evolutionary_search_tournament(int n, int ub, int pop, int iters,
                                                int tournament_size) {
  RNG rng;

  // Initialize population
  auto [population, best, best_fitness] = initialize_population(n, ub, pop, rng);

  // Helper: tournament selection
  auto tournament_select = [&](const std::vector<int>& fitnesses) -> int {
    int best_idx = rng.uniform_int(0, static_cast<int>(population.size()) - 1);
    int best_fit = fitnesses[best_idx];

    for (int i = 1; i < tournament_size; ++i) {
      int candidate_idx = rng.uniform_int(0, static_cast<int>(population.size()) - 1);
      int candidate_fit = fitnesses[candidate_idx];

      if (candidate_fit > best_fit) {
        best_idx = candidate_idx;
        best_fit = candidate_fit;
      }
    }

    return best_idx;
  };

  for (int iter = 0; iter < iters; ++iter) {
    // Evaluate population
    std::vector<int> fitnesses;
    fitnesses.reserve(population.size());

    for (const auto& individual : population) {
      int fit = evaluate_fitness(individual);
      fitnesses.push_back(fit);

      if (fit > best_fitness) {
        best = individual;
        best_fitness = fit;
      }
    }

    // Create new population using tournament selection
    std::vector<std::vector<int>> new_pop;

    // Elitism: keep best individual
    int best_idx = 0;
    for (size_t i = 1; i < fitnesses.size(); ++i) {
      if (fitnesses[i] > fitnesses[best_idx]) {
        best_idx = static_cast<int>(i);
      }
    }
    new_pop.push_back(population[best_idx]);

    while (static_cast<int>(new_pop.size()) < pop) {
      // Select parents via tournament
      int p1_idx = tournament_select(fitnesses);
      int p2_idx = tournament_select(fitnesses);

      // Crossover and mutation
      auto child = crossover(population[p1_idx], population[p2_idx], rng);
      child = mutate(child, ub, rng);

      new_pop.push_back(child);
    }

    population = std::move(new_pop);
  }

  return hill_climb(best, ub, LOCAL_SEARCH_BUDGET);
}

} // namespace golomb
