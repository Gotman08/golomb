#pragma once

#include <random>

namespace golomb {

/**
 * @brief Wrapper for random number generation utilities.
 *
 * Provides a convenient interface to std::mt19937_64 with common distributions.
 */
class RNG {
public:
  /**
   * @brief Construct RNG with optional seed.
   * @param seed Seed value (defaults to random_device).
   */
  explicit RNG(uint64_t seed = 0);

  /**
   * @brief Generate random integer in [min, max] inclusive.
   */
  int uniform_int(int min, int max);

  /**
   * @brief Generate random double in [0.0, 1.0).
   */
  double uniform_real();

  /**
   * @brief Get reference to underlying engine.
   */
  std::mt19937_64& engine();

private:
  std::mt19937_64 engine_;
};

}  // namespace golomb
