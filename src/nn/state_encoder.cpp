#include "nn/state_encoder.hpp"
#include "core/golomb.hpp"
#include <algorithm>

namespace golomb {
namespace nn {

StateEncoder::StateEncoder(int ub, int target_marks, int encoding)
    : ub_(ub), target_marks_(target_marks), encoding_type_(encoding), encoding_size_(0) {
  encoding_size_ = compute_encoding_size();
}

size_t StateEncoder::compute_encoding_size() const {
  size_t size = 0;

  if (encoding_type_ & POSITIONS) {
    // Binary vector for each possible position [0, ub]
    size += static_cast<size_t>(ub_ + 1);
  }

  if (encoding_type_ & DISTANCES) {
    // Binary vector for each possible distance [0, ub]
    // Maximum distance in a Golomb ruler is ub
    size += static_cast<size_t>(ub_ + 1);
  }

  if (encoding_type_ & METADATA) {
    // Normalized metadata features:
    // 1. Number of marks placed (normalized by target)
    // 2. Current length (normalized by ub)
    // 3. Progress (marks placed / target marks)
    // 4. Density (marks placed / current length, if length > 0)
    size += 4;
  }

  return size;
}

Tensor StateEncoder::encode(const RuleState& state) const {
  Tensor encoding(encoding_size_);
  encoding.zeros();

  size_t offset = 0;

  if (encoding_type_ & POSITIONS) {
    encode_positions(state, offset, encoding);
    offset += static_cast<size_t>(ub_ + 1);
  }

  if (encoding_type_ & DISTANCES) {
    encode_distances(state, offset, encoding);
    offset += static_cast<size_t>(ub_ + 1);
  }

  if (encoding_type_ & METADATA) {
    encode_metadata(state, offset, encoding);
    offset += 4;
  }

  return encoding;
}

void StateEncoder::encode_positions(const RuleState& state, size_t offset, Tensor& output) const {
  // Set bit to 1 for each position that has a mark
  for (int mark : state.marks) {
    if (mark >= 0 && mark <= ub_) {
      output(offset + static_cast<size_t>(mark)) = 1.0;
    }
  }
}

void StateEncoder::encode_distances(const RuleState& state, size_t offset, Tensor& output) const {
  // Set bit to 1 for each distance that is used
  for_each_pairwise_distance(state.marks, [&](int dist) {
    if (dist >= 0 && dist <= ub_) {
      output(offset + static_cast<size_t>(dist)) = 1.0;
    }
  });
}

void StateEncoder::encode_metadata(const RuleState& state, size_t offset, Tensor& output) const {
  int num_marks = static_cast<int>(state.marks.size());
  int current_length = state.marks.empty() ? 0 : state.marks.back();

  // Feature 1: Number of marks placed (normalized by target)
  output(offset + 0) = static_cast<double>(num_marks) / static_cast<double>(target_marks_);

  // Feature 2: Current length (normalized by ub)
  output(offset + 1) = static_cast<double>(current_length) / static_cast<double>(ub_);

  // Feature 3: Progress (ratio of marks placed to target)
  output(offset + 2) = static_cast<double>(num_marks) / static_cast<double>(target_marks_);

  // Feature 4: Density (marks per unit length, if length > 0)
  if (current_length > 0) {
    output(offset + 3) = static_cast<double>(num_marks) / static_cast<double>(current_length);
  } else {
    output(offset + 3) = 0.0;
  }
}

} // namespace nn
} // namespace golomb
