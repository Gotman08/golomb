#pragma once

#include "bitset_dist.hpp"
#include <vector>

namespace golomb {

/**
 * @brief State representation for a partial or complete Golomb ruler.
 *
 * Combines the list of marks with a distance bitset for efficient validation.
 */
struct RuleState {
  std::vector<int> marks; ///< Sorted list of mark positions (includes 0).
  DistBitset used;        ///< Bitset tracking used distances.

  explicit RuleState(int max_dist) : used(max_dist) {}
};

/**
 * @brief Check if a set of marks forms a valid Golomb ruler.
 * @param marks Sorted list of mark positions.
 * @return true if all pairwise distances are unique.
 */
[[nodiscard]] bool is_valid_rule(const std::vector<int>& marks);

/**
 * @brief Compute the length (maximum mark) of a ruler.
 * @param marks Sorted list of mark positions.
 * @return Length of the ruler (last mark value).
 */
[[nodiscard]] int length(const std::vector<int>& marks);

/**
 * @brief Generate a simple greedy seed ruler.
 * @param n Number of marks to place.
 * @param ub Upper bound for mark positions.
 * @return A valid (but not necessarily optimal) ruler.
 */
[[nodiscard]] std::vector<int> greedy_seed(int n, int ub);

/**
 * @brief Try to add a mark to a RuleState.
 * @param st State to modify.
 * @param p Position to add.
 * @return true if mark was added successfully, false if it would create duplicate distances.
 */
bool try_add(RuleState& st, int p);

/**
 * @brief Apply a function to all pairwise distances in a ruler.
 * @param marks Sorted list of mark positions.
 * @param func Function to apply to each distance (signature: void(int distance))
 */
template <typename Func>
void for_each_pairwise_distance(const std::vector<int>& marks, Func&& func) {
  for (size_t i = 0; i < marks.size(); ++i) {
    for (size_t j = i + 1; j < marks.size(); ++j) {
      int dist = marks[j] - marks[i];
      func(dist);
    }
  }
}

/**
 * @brief Compute all pairwise distances in a ruler.
 * @param marks Sorted list of mark positions.
 * @return Vector of all pairwise distances.
 */
[[nodiscard]] std::vector<int> compute_all_distances(const std::vector<int>& marks);

} // namespace golomb
