#include "core/bitset_dist.hpp"
#include <algorithm>
#include <stdexcept>

namespace golomb {

DistBitset::DistBitset(int max_dist) : max_dist_(max_dist) {
  if (max_dist <= 0) {
    throw std::invalid_argument("max_dist must be positive");
  }
  int num_words = (max_dist + 63) / 64;
  bits_.resize(num_words, 0);
}

void DistBitset::set(int d) {
  if (d < 0 || d >= max_dist_) {
    return;
  }
  int word_idx = d / 64;
  int bit_idx = d % 64;
  bits_[word_idx] |= (1ULL << bit_idx);
}

bool DistBitset::test(int d) const {
  if (d < 0 || d >= max_dist_) {
    return false;
  }
  int word_idx = d / 64;
  int bit_idx = d % 64;
  return (bits_[word_idx] & (1ULL << bit_idx)) != 0;
}

void DistBitset::clear() { std::fill(bits_.begin(), bits_.end(), 0); }

int DistBitset::size() const { return max_dist_; }

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

}  // namespace golomb
