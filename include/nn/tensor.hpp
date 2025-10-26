#pragma once

#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <cstddef>

namespace golomb {
namespace nn {

/**
 * @brief Multi-dimensional tensor class for neural network operations.
 *
 * Supports 1D (vectors), 2D (matrices), and 3D tensors with efficient
 * storage using std::vector. Implements common operations needed for
 * neural network forward and backward passes.
 *
 * Storage is row-major: for a 2D tensor of shape (rows, cols),
 * element (i, j) is at index i * cols + j.
 */
class Tensor {
public:
  /**
   * @brief Construct a tensor with given shape, initialized to zeros.
   * @param shape Dimensions of the tensor (e.g., {3, 4} for 3x4 matrix).
   */
  explicit Tensor(const std::vector<size_t>& shape);

  /**
   * @brief Construct a 1D tensor (vector) with given size.
   * @param size Number of elements.
   */
  explicit Tensor(size_t size);

  /**
   * @brief Construct a 2D tensor (matrix) with given dimensions.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  Tensor(size_t rows, size_t cols);

  /**
   * @brief Construct a 1D tensor from initializer list.
   * @param values Initial values.
   */
  Tensor(std::initializer_list<double> values);

  /**
   * @brief Get the shape of the tensor.
   * @return Vector containing dimensions.
   */
  [[nodiscard]] const std::vector<size_t>& shape() const { return shape_; }

  /**
   * @brief Get the total number of elements.
   * @return Total size.
   */
  [[nodiscard]] size_t size() const { return data_.size(); }

  /**
   * @brief Get number of dimensions.
   * @return Number of dimensions (1, 2, or 3).
   */
  [[nodiscard]] size_t ndim() const { return shape_.size(); }

  /**
   * @brief Access raw data (const).
   * @return Const reference to internal data vector.
   */
  [[nodiscard]] const std::vector<double>& data() const { return data_; }

  /**
   * @brief Access raw data (mutable).
   * @return Reference to internal data vector.
   */
  std::vector<double>& data() { return data_; }

  /**
   * @brief Access element in 1D tensor.
   * @param i Index.
   * @return Reference to element.
   */
  double& operator()(size_t i);

  /**
   * @brief Access element in 1D tensor (const).
   * @param i Index.
   * @return Const reference to element.
   */
  [[nodiscard]] const double& operator()(size_t i) const;

  /**
   * @brief Access element in 2D tensor.
   * @param i Row index.
   * @param j Column index.
   * @return Reference to element.
   */
  double& operator()(size_t i, size_t j);

  /**
   * @brief Access element in 2D tensor (const).
   * @param i Row index.
   * @param j Column index.
   * @return Const reference to element.
   */
  [[nodiscard]] const double& operator()(size_t i, size_t j) const;

  /**
   * @brief Access element in 3D tensor.
   * @param i First dimension index.
   * @param j Second dimension index.
   * @param k Third dimension index.
   * @return Reference to element.
   */
  double& operator()(size_t i, size_t j, size_t k);

  /**
   * @brief Access element in 3D tensor (const).
   * @param i First dimension index.
   * @param j Second dimension index.
   * @param k Third dimension index.
   * @return Const reference to element.
   */
  [[nodiscard]] const double& operator()(size_t i, size_t j, size_t k) const;

  /**
   * @brief Fill tensor with a constant value.
   * @param value Value to fill with.
   */
  void fill(double value);

  /**
   * @brief Fill tensor with zeros.
   */
  void zeros();

  /**
   * @brief Fill tensor with ones.
   */
  void ones();

  /**
   * @brief Fill tensor with random values from uniform distribution [min, max].
   * @param min Minimum value.
   * @param max Maximum value.
   */
  void random_uniform(double min = 0.0, double max = 1.0);

  /**
   * @brief Fill tensor with random values from normal distribution N(mean, stddev).
   * @param mean Mean of distribution.
   * @param stddev Standard deviation.
   */
  void random_normal(double mean = 0.0, double stddev = 1.0);

  /**
   * @brief Reshape tensor (must preserve total size).
   * @param new_shape New dimensions.
   * @throws std::invalid_argument if total size doesn't match.
   */
  void reshape(const std::vector<size_t>& new_shape);

  /**
   * @brief Transpose a 2D tensor.
   * @return New transposed tensor.
   * @throws std::runtime_error if tensor is not 2D.
   */
  [[nodiscard]] Tensor transpose() const;

  /**
   * @brief Element-wise addition.
   * @param other Tensor to add.
   * @return New tensor with result.
   */
  [[nodiscard]] Tensor operator+(const Tensor& other) const;

  /**
   * @brief Element-wise subtraction.
   * @param other Tensor to subtract.
   * @return New tensor with result.
   */
  [[nodiscard]] Tensor operator-(const Tensor& other) const;

  /**
   * @brief Element-wise multiplication (Hadamard product).
   * @param other Tensor to multiply.
   * @return New tensor with result.
   */
  [[nodiscard]] Tensor operator*(const Tensor& other) const;

  /**
   * @brief Scalar multiplication.
   * @param scalar Scalar value.
   * @return New tensor with result.
   */
  [[nodiscard]] Tensor operator*(double scalar) const;

  /**
   * @brief Scalar division.
   * @param scalar Scalar value.
   * @return New tensor with result.
   */
  [[nodiscard]] Tensor operator/(double scalar) const;

  /**
   * @brief In-place addition.
   * @param other Tensor to add.
   * @return Reference to this tensor.
   */
  Tensor& operator+=(const Tensor& other);

  /**
   * @brief In-place subtraction.
   * @param other Tensor to subtract.
   * @return Reference to this tensor.
   */
  Tensor& operator-=(const Tensor& other);

  /**
   * @brief In-place element-wise multiplication.
   * @param other Tensor to multiply.
   * @return Reference to this tensor.
   */
  Tensor& operator*=(const Tensor& other);

  /**
   * @brief In-place scalar multiplication.
   * @param scalar Scalar value.
   * @return Reference to this tensor.
   */
  Tensor& operator*=(double scalar);

  /**
   * @brief Matrix multiplication (2D tensors only).
   *
   * For tensors A (m x n) and B (n x p), returns C (m x p).
   *
   * @param other Right-hand matrix.
   * @return New tensor with result.
   * @throws std::runtime_error if shapes incompatible or tensors not 2D.
   */
  [[nodiscard]] Tensor matmul(const Tensor& other) const;

  /**
   * @brief Dot product (1D tensors only).
   *
   * Computes sum of element-wise products.
   *
   * @param other Other vector.
   * @return Scalar result.
   * @throws std::runtime_error if tensors not 1D or sizes don't match.
   */
  [[nodiscard]] double dot(const Tensor& other) const;

  /**
   * @brief Sum all elements.
   * @return Scalar sum.
   */
  [[nodiscard]] double sum() const;

  /**
   * @brief Compute mean of all elements.
   * @return Mean value.
   */
  [[nodiscard]] double mean() const;

  /**
   * @brief Apply function element-wise.
   * @param func Function to apply (double -> double).
   */
  void apply(double (*func)(double));

  /**
   * @brief Create a copy of this tensor.
   * @return New tensor with same data.
   */
  [[nodiscard]] Tensor copy() const;

  /**
   * @brief Static factory: create tensor filled with zeros.
   * @param shape Dimensions.
   * @return New zero tensor.
   */
  static Tensor zeros(const std::vector<size_t>& shape);

  /**
   * @brief Static factory: create tensor filled with ones.
   * @param shape Dimensions.
   * @return New ones tensor.
   */
  static Tensor ones(const std::vector<size_t>& shape);

  /**
   * @brief Static factory: create tensor with Xavier/Glorot initialization.
   *
   * Useful for weight initialization: samples from U[-limit, limit]
   * where limit = sqrt(6 / (fan_in + fan_out)).
   *
   * @param rows Number of rows (fan_out).
   * @param cols Number of columns (fan_in).
   * @return New initialized tensor.
   */
  static Tensor xavier(size_t rows, size_t cols);

  /**
   * @brief Static factory: create tensor with He initialization.
   *
   * Useful for ReLU networks: samples from N(0, sqrt(2 / fan_in)).
   *
   * @param rows Number of rows.
   * @param cols Number of columns (fan_in).
   * @return New initialized tensor.
   */
  static Tensor he(size_t rows, size_t cols);

private:
  std::vector<double> data_;    ///< Flattened data storage.
  std::vector<size_t> shape_;   ///< Dimensions.

  /**
   * @brief Compute flat index from multi-dimensional indices.
   * @param indices Vector of indices.
   * @return Flat index.
   */
  [[nodiscard]] size_t compute_index(const std::vector<size_t>& indices) const;

  /**
   * @brief Check if shapes are compatible for element-wise operations.
   * @param other Other tensor.
   * @throws std::invalid_argument if shapes don't match.
   */
  void check_same_shape(const Tensor& other) const;
};

/**
 * @brief Scalar * Tensor (commutative).
 */
[[nodiscard]] Tensor operator*(double scalar, const Tensor& tensor);

}  // namespace nn
}  // namespace golomb
