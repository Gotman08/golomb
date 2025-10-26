#pragma once

#include <cstdint>
#include <vector>

namespace golomb {

/**
 * @brief Bitset-based distance tracker for efficient distance uniqueness checking.
 *
 * Uses a dynamic bitset (vector of uint64_t) to track which distances have been used.
 * Provides O(1) distance checking and efficient update when adding marks.
 */
class DistBitset {
public:
  /**
   * @brief Construct bitset with maximum distance capacity.
   * @param max_dist Maximum distance value to support (exclusive).
   */
  explicit DistBitset(int max_dist);

  /**
   * @brief Set a distance bit to true.
   * @param d Distance to mark as used.
   */
  void set(int d);

  /**
   * @brief Test if a distance bit is set.
   * @param d Distance to check.
   * @return true if distance is already used, false otherwise.
   */
  [[nodiscard]] bool test(int d) const;

  /**
   * @brief Clear all distance bits.
   */
  void clear();

  /**
   * @brief Get the maximum distance capacity.
   * @return Maximum distance value supported.
   */
  [[nodiscard]] int size() const;

  /**
   * @brief Check if a new mark can be added without creating duplicate distances.
   * @param marks Current set of marks (sorted).
   * @param p New mark position to test.
   * @return true if p can be added without distance conflicts.
   */
  [[nodiscard]] bool can_add_mark(const std::vector<int>& marks, int p) const;

  /**
   * @brief Add a mark and update the bitset with all new distances.
   * @param marks Current set of marks (will be modified and kept sorted).
   * @param p New mark position to add.
   */
  void add_mark(std::vector<int>& marks, int p);

private:
  std::vector<uint64_t> bits_;
  int max_dist_;
};

}  // namespace golomb
