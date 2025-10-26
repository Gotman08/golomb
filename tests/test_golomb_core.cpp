#include <catch2/catch_test_macros.hpp>
#include "core/bitset_dist.hpp"
#include "core/golomb.hpp"

using namespace golomb;

TEST_CASE("DistBitset basic operations", "[bitset]") {
  DistBitset bs(100);

  SECTION("initial state") {
    REQUIRE_FALSE(bs.test(5));
    REQUIRE_FALSE(bs.test(99));
  }

  SECTION("set and test") {
    bs.set(10);
    bs.set(25);
    REQUIRE(bs.test(10));
    REQUIRE(bs.test(25));
    REQUIRE_FALSE(bs.test(11));
  }

  SECTION("clear") {
    bs.set(10);
    bs.set(20);
    bs.clear();
    REQUIRE_FALSE(bs.test(10));
    REQUIRE_FALSE(bs.test(20));
  }

  SECTION("can_add_mark") {
    std::vector<int> marks = {0, 1, 3};
    // Distances: 1, 3, 2
    bs.set(1);
    bs.set(2);
    bs.set(3);

    REQUIRE_FALSE(bs.can_add_mark(marks, 4));  // Would create distance 1 (4-3)
    REQUIRE(bs.can_add_mark(marks, 7));        // New distances: 7, 6, 4 (all new)
  }

  SECTION("add_mark") {
    std::vector<int> marks = {0, 1};
    bs.set(1);

    bs.add_mark(marks, 4);
    REQUIRE(marks.size() == 3);
    REQUIRE(marks[2] == 4);
    REQUIRE(bs.test(3));  // Distance 4-1
    REQUIRE(bs.test(4));  // Distance 4-0
  }
}

TEST_CASE("Golomb ruler validation", "[golomb]") {
  SECTION("valid rulers") {
    REQUIRE(is_valid_rule({0, 1, 3}));
    REQUIRE(is_valid_rule({0, 1, 4, 6}));
    REQUIRE(is_valid_rule({0, 1, 4, 9, 11}));
  }

  SECTION("invalid rulers") {
    REQUIRE_FALSE(is_valid_rule({0, 1, 2, 3}));  // Multiple distance 1
    REQUIRE_FALSE(is_valid_rule({0, 2, 4, 6}));  // All distance 2
  }

  SECTION("empty ruler") { REQUIRE(is_valid_rule({})); }
}

TEST_CASE("Ruler length", "[golomb]") {
  REQUIRE(length({0, 1, 3}) == 3);
  REQUIRE(length({0, 1, 4, 6}) == 6);
  REQUIRE(length({}) == 0);
}

TEST_CASE("Greedy seed generation", "[golomb]") {
  auto ruler = greedy_seed(4, 20);
  REQUIRE(ruler.size() >= 1);
  REQUIRE(ruler[0] == 0);
  REQUIRE(is_valid_rule(ruler));
}

TEST_CASE("RuleState try_add", "[golomb]") {
  RuleState st(50);
  st.marks.push_back(0);

  REQUIRE(try_add(st, 1));
  REQUIRE(st.marks.size() == 2);

  REQUIRE(try_add(st, 3));
  REQUIRE(st.marks.size() == 3);

  // Distance 1 already used (1-0), so 4 would create 4-3=1
  REQUIRE_FALSE(try_add(st, 4));
}
