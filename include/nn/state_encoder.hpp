#pragma once

#include "core/golomb.hpp"
#include "nn/tensor.hpp"
#include <vector>

namespace golomb {
namespace nn {

/**
 * @brief Encoder for converting Golomb ruler state to neural network input.
 *
 * Transforms RuleState into a fixed-size tensor representation suitable
 * for neural network processing. Supports multiple encoding strategies
 * that can be combined.
 *
 * The encoder creates a feature vector capturing:
 * 1. Mark positions (binary encoding)
 * 2. Distances used (binary encoding)
 * 3. Metadata (normalized values)
 */
class StateEncoder {
public:
  /**
   * @brief Encoding strategies (can be combined with bitwise OR).
   */
  enum EncodingType {
    POSITIONS = 1 << 0, ///< Binary vector of mark positions [0, ub].
    DISTANCES = 1 << 1, ///< Binary vector of used distances [0, max_dist].
    METADATA = 1 << 2,  ///< Normalized metadata (num_marks, length, progress).
    DEFAULT = POSITIONS | DISTANCES | METADATA ///< All encodings combined.
  };

  /**
   * @brief Construct state encoder.
   * @param ub Upper bound for positions.
   * @param target_marks Target number of marks to place.
   * @param encoding Encoding strategy (default: all features).
   */
  StateEncoder(int ub, int target_marks, int encoding = DEFAULT);

  /**
   * @brief Encode a RuleState into a tensor.
   *
   * Creates a 1D tensor with encoded features.
   *
   * @param state Golomb ruler state to encode.
   * @return 1D tensor with encoded features.
   */
  [[nodiscard]] Tensor encode(const RuleState& state) const;

  /**
   * @brief Get the size of the encoded tensor.
   * @return Number of features in encoded representation.
   */
  [[nodiscard]] size_t encoding_size() const { return encoding_size_; }

  /**
   * @brief Get upper bound.
   * @return Upper bound for positions.
   */
  [[nodiscard]] int ub() const { return ub_; }

  /**
   * @brief Get target number of marks.
   * @return Target marks.
   */
  [[nodiscard]] int target_marks() const { return target_marks_; }

private:
  int ub_;               ///< Upper bound for positions.
  int target_marks_;     ///< Target number of marks.
  int encoding_type_;    ///< Combination of EncodingType flags.
  size_t encoding_size_; ///< Total size of encoding.

  /**
   * @brief Encode mark positions as binary vector.
   * @param state State to encode.
   * @param offset Offset in output vector.
   * @param output Output tensor to write to.
   */
  void encode_positions(const RuleState& state, size_t offset, Tensor& output) const;

  /**
   * @brief Encode used distances as binary vector.
   * @param state State to encode.
   * @param offset Offset in output vector.
   * @param output Output tensor to write to.
   */
  void encode_distances(const RuleState& state, size_t offset, Tensor& output) const;

  /**
   * @brief Encode metadata (normalized scalars).
   * @param state State to encode.
   * @param offset Offset in output vector.
   * @param output Output tensor to write to.
   */
  void encode_metadata(const RuleState& state, size_t offset, Tensor& output) const;

  /**
   * @brief Compute total encoding size based on flags.
   * @return Total number of features.
   */
  size_t compute_encoding_size() const;
};

} // namespace nn
} // namespace golomb
