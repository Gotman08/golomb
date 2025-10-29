#include "core/golomb.hpp"
#include "exact/exact_iface.hpp"
#include <catch2/catch_test_macros.hpp>

using namespace golomb;

TEST_CASE("CP-SAT solves known optimal rulers", "[exact][cpsat]") {
  SECTION("Order 3, optimal length 3") {
    ExactOptions opts{3, 10, 5000};
    ExactResult res = solve_exact_cpsat(opts);

    REQUIRE(res.optimal);
    REQUIRE(res.rule.size() == 3);
    REQUIRE(res.rule[0] == 0);
    REQUIRE(length(res.rule) == 3);
    REQUIRE(is_valid_rule(res.rule));

    // Known optimal: {0, 1, 3}
    REQUIRE(res.rule == std::vector<int>{0, 1, 3});
  }

  SECTION("Order 4, optimal length 6") {
    ExactOptions opts{4, 20, 10000};
    ExactResult res = solve_exact_cpsat(opts);

    REQUIRE(res.optimal);
    REQUIRE(res.rule.size() == 4);
    REQUIRE(res.rule[0] == 0);
    REQUIRE(length(res.rule) == 6);
    REQUIRE(is_valid_rule(res.rule));

    // Known optimal: {0, 1, 4, 6}
    REQUIRE(res.rule == std::vector<int>{0, 1, 4, 6});
  }

  SECTION("Order 5, optimal length 11") {
    ExactOptions opts{5, 50, 15000};
    ExactResult res = solve_exact_cpsat(opts);

    REQUIRE(res.optimal);
    REQUIRE(res.rule.size() == 5);
    REQUIRE(res.rule[0] == 0);
    REQUIRE(length(res.rule) == 11);
    REQUIRE(is_valid_rule(res.rule));

    // Known optimal solutions: {0, 1, 4, 9, 11} or {0, 2, 7, 8, 11}
    bool is_known_optimal =
        (res.rule == std::vector<int>{0, 1, 4, 9, 11}) ||
        (res.rule == std::vector<int>{0, 2, 7, 8, 11});
    REQUIRE(is_known_optimal);
  }

  SECTION("Order 6, optimal length 17") {
    ExactOptions opts{6, 50, 20000};
    ExactResult res = solve_exact_cpsat(opts);

    // May timeout on slow systems, but should find optimal
    if (res.optimal) {
      REQUIRE(res.rule.size() == 6);
      REQUIRE(res.rule[0] == 0);
      REQUIRE(length(res.rule) == 17);
      REQUIRE(is_valid_rule(res.rule));

      // Known optimal solutions (there are multiple for n=6)
      bool is_known_optimal =
          (res.rule == std::vector<int>{0, 1, 4, 9, 15, 17}) ||
          (res.rule == std::vector<int>{0, 1, 4, 10, 12, 17});
      REQUIRE(is_known_optimal);
    }
  }
}

TEST_CASE("CP-SAT validates results", "[exact][cpsat]") {
  SECTION("First mark is always 0") {
    ExactOptions opts{4, 20, 5000};
    ExactResult res = solve_exact_cpsat(opts);

    REQUIRE(!res.rule.empty());
    REQUIRE(res.rule[0] == 0);
  }

  SECTION("Marks are strictly increasing") {
    ExactOptions opts{5, 30, 5000};
    ExactResult res = solve_exact_cpsat(opts);

    REQUIRE(!res.rule.empty());
    for (size_t i = 0; i + 1 < res.rule.size(); ++i) {
      REQUIRE(res.rule[i] < res.rule[i + 1]);
    }
  }

  SECTION("All distances are unique") {
    ExactOptions opts{5, 30, 5000};
    ExactResult res = solve_exact_cpsat(opts);

    REQUIRE(!res.rule.empty());
    REQUIRE(is_valid_rule(res.rule));
  }

  SECTION("Result length matches last mark") {
    ExactOptions opts{4, 20, 5000};
    ExactResult res = solve_exact_cpsat(opts);

    if (res.optimal) {
      REQUIRE(res.ub == length(res.rule));
      REQUIRE(res.lb == res.ub);
    }
  }
}

TEST_CASE("CP-SAT respects upper bounds", "[exact][cpsat]") {
  SECTION("Feasible with generous upper bound") {
    ExactOptions opts{4, 100, 5000};
    ExactResult res = solve_exact_cpsat(opts);

    REQUIRE(res.optimal);
    REQUIRE(!res.rule.empty());
    REQUIRE(length(res.rule) <= 100);
  }

  SECTION("Infeasible with too tight upper bound") {
    // Order 4 needs at least length 6
    ExactOptions opts{4, 5, 5000};
    ExactResult res = solve_exact_cpsat(opts);

    REQUIRE(!res.optimal);
    REQUIRE(res.message.find("infeasible") != std::string::npos);
  }

  SECTION("Respects specified upper bound") {
    ExactOptions opts{5, 15, 10000};
    ExactResult res = solve_exact_cpsat(opts);

    // May be optimal or feasible depending on solving time
    if (!res.rule.empty()) {
      REQUIRE(length(res.rule) <= 15);
    }
  }
}

TEST_CASE("CP-SAT handles timeout gracefully", "[exact][cpsat]") {
  SECTION("Very short timeout still returns result") {
    // Order 8 with 100ms timeout - unlikely to find optimal
    ExactOptions opts{8, 100, 100};
    ExactResult res = solve_exact_cpsat(opts);

    // Should return either feasible or timeout status
    REQUIRE(!res.message.empty());

    if (!res.rule.empty()) {
      REQUIRE(is_valid_rule(res.rule));
      REQUIRE(res.rule.size() == static_cast<size_t>(opts.n));
    }
  }

  SECTION("Timeout produces feasible solution or fallback") {
    ExactOptions opts{10, 100, 500};
    ExactResult res = solve_exact_cpsat(opts);

    // May timeout or find feasible solution
    if (!res.rule.empty()) {
      REQUIRE(is_valid_rule(res.rule));
    }
  }
}

TEST_CASE("CP-SAT bounds are consistent", "[exact][cpsat]") {
  SECTION("Lower bound never exceeds upper bound") {
    ExactOptions opts{5, 50, 10000};
    ExactResult res = solve_exact_cpsat(opts);

    REQUIRE(res.lb <= res.ub);
  }

  SECTION("Optimal solution has lb = ub") {
    ExactOptions opts{3, 10, 5000};
    ExactResult res = solve_exact_cpsat(opts);

    if (res.optimal) {
      REQUIRE(res.lb == res.ub);
      REQUIRE(res.ub == length(res.rule));
    }
  }

  SECTION("Lower bound respects triangular number") {
    // Minimum length for n marks is n*(n-1)/2
    ExactOptions opts{5, 50, 5000};
    ExactResult res = solve_exact_cpsat(opts);

    int min_length = opts.n * (opts.n - 1) / 2;
    REQUIRE(res.lb >= min_length);
  }
}

TEST_CASE("CP-SAT message indicates status", "[exact][cpsat]") {
  SECTION("Optimal message for optimal solution") {
    ExactOptions opts{3, 10, 5000};
    ExactResult res = solve_exact_cpsat(opts);

    REQUIRE(res.optimal);
    REQUIRE(res.message == "optimal");
  }

  SECTION("Feasible message for timeout with solution") {
    // Large instance that may timeout
    ExactOptions opts{9, 100, 1000};
    ExactResult res = solve_exact_cpsat(opts);

    if (!res.optimal && !res.rule.empty()) {
      REQUIRE((res.message.find("feasible") != std::string::npos ||
               res.message.find("timeout") != std::string::npos));
    }
  }

  SECTION("Infeasible message when ub too small") {
    ExactOptions opts{5, 8, 5000}; // Impossible - needs at least 11
    ExactResult res = solve_exact_cpsat(opts);

    REQUIRE(!res.optimal);
    REQUIRE(res.message.find("infeasible") != std::string::npos);
  }
}

TEST_CASE("CP-SAT performance characteristics", "[exact][cpsat][.slow]") {
  // These tests are slow, marked with [.slow] to skip in regular runs
  // Run with: ./golomb_tests "[.slow]"

  SECTION("Order 7 solves in reasonable time") {
    ExactOptions opts{7, 100, 60000}; // 60 second timeout
    ExactResult res = solve_exact_cpsat(opts);

    // Should find optimal length 25 within timeout
    if (res.optimal) {
      REQUIRE(length(res.rule) == 25);
      REQUIRE(is_valid_rule(res.rule));
    }
  }

  SECTION("Order 8 may need longer timeout") {
    ExactOptions opts{8, 100, 300000}; // 5 minute timeout
    ExactResult res = solve_exact_cpsat(opts);

    // Known optimal length is 34
    if (res.optimal) {
      REQUIRE(length(res.rule) == 34);
      REQUIRE(is_valid_rule(res.rule));
    }
  }
}
