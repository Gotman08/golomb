#include "core/bitset_dist.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace golomb {

namespace {

// Constants for bitset word operations
constexpr int BITS_PER_WORD = 64;
constexpr int BITS_PER_WORD_MASK = 63;

// Helper: compute word index and bit index for a distance value
inline std::pair<int, int> compute_bit_indices(int d) {
  return {d / BITS_PER_WORD, d % BITS_PER_WORD};
}

} // anonymous namespace

DistBitset::DistBitset(int max_dist) : max_dist_(max_dist) {
  if (max_dist <= 0) {
    throw std::invalid_argument("max_dist must be positive");
  }
  int num_words = (max_dist + BITS_PER_WORD_MASK) / BITS_PER_WORD;
  bits_.resize(num_words, 0);
}

void DistBitset::set(int d) {
  if (d < 0 || d >= max_dist_) {
    return;
  }
  auto [word_idx, bit_idx] = compute_bit_indices(d);
  bits_[word_idx] |= (1ULL << bit_idx);
}

bool DistBitset::test(int d) const {
  if (d < 0 || d >= max_dist_) {
    return false;
  }
  auto [word_idx, bit_idx] = compute_bit_indices(d);
  return (bits_[word_idx] & (1ULL << bit_idx)) != 0;
}

void DistBitset::clear() {
  std::fill(bits_.begin(), bits_.end(), 0);
}

int DistBitset::size() const {
  return max_dist_;
}

bool DistBitset::can_add_mark(const std::vector<int>& marks, int p) const {
  for (int m : marks) {
    int dist = std::abs(p - m);
    if (test(dist)) {
      return false;
    }
  }
  return true;
}

void DistBitset::add_mark(std::vector<int>& marks, int p) {
  // Add all new distances to bitset
  for (int m : marks) {
    int dist = std::abs(p - m);
    set(dist);
  }

  // Insert p into sorted marks
  auto it = std::lower_bound(marks.begin(), marks.end(), p);
  marks.insert(it, p);
}

} // namespace golomb
