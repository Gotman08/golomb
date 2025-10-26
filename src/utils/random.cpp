#include "utils/random.hpp"

namespace golomb {

RNG::RNG(uint64_t seed) {
  if (seed == 0) {
    std::random_device rd;
    seed = rd();
  }
  engine_.seed(seed);
}

int RNG::uniform_int(int min, int max) {
  std::uniform_int_distribution<int> dist(min, max);
  return dist(engine_);
}

double RNG::uniform_real() {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(engine_);
}

std::mt19937_64& RNG::engine() { return engine_; }

}  // namespace golomb
