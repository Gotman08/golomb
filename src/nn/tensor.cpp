#include "nn/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>

namespace golomb {
namespace nn {

namespace {
// Thread-local RNG for random initialization
thread_local std::mt19937 rng{std::random_device{}()};
} // namespace

// ============================================================================
// Constructors
// ============================================================================

Tensor::Tensor() : shape_({1}), data_(1, 0.0) {
  // Default constructor creates a scalar tensor (1 element)
  // This maintains valid state for member variables
}

Tensor::Tensor(const std::vector<size_t>& shape) : shape_(shape) {
  if (shape.empty()) {
    throw std::invalid_argument("Tensor shape cannot be empty");
  }
  size_t total_size =
      std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
  data_.resize(total_size, 0.0);
}

Tensor::Tensor(size_t size) : shape_({size}) {
  data_.resize(size, 0.0);
}

Tensor::Tensor(size_t rows, size_t cols) : shape_({rows, cols}) {
  data_.resize(rows * cols, 0.0);
}

Tensor::Tensor(std::initializer_list<double> values) : shape_({values.size()}) {
  data_.assign(values.begin(), values.end());
}

// OPT-4A: Move constructor - efficient transfer of ownership (CSAPP 9.9)
Tensor::Tensor(Tensor&& other) noexcept
    : data_(std::move(other.data_)), shape_(std::move(other.shape_)) {
  // other is left in valid but unspecified state
}

// OPT-4A: Move assignment - efficient transfer of ownership (CSAPP 9.9)
Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    data_ = std::move(other.data_);
    shape_ = std::move(other.shape_);
  }
  return *this;
}

// ============================================================================
// Element Access
// ============================================================================

double& Tensor::operator()(size_t i) {
#ifndef NDEBUG
  if (ndim() != 1) {
    throw std::runtime_error("1D access requires 1D tensor");
  }
  if (i >= shape_[0]) {
    throw std::out_of_range("Index out of bounds");
  }
#endif
  return data_[i];
}

const double& Tensor::operator()(size_t i) const {
#ifndef NDEBUG
  if (ndim() != 1) {
    throw std::runtime_error("1D access requires 1D tensor");
  }
  if (i >= shape_[0]) {
    throw std::out_of_range("Index out of bounds");
  }
#endif
  return data_[i];
}

double& Tensor::operator()(size_t i, size_t j) {
#ifndef NDEBUG
  if (ndim() != 2) {
    throw std::runtime_error("2D access requires 2D tensor");
  }
  if (i >= shape_[0] || j >= shape_[1]) {
    throw std::out_of_range("Index out of bounds");
  }
#endif
  return data_[i * shape_[1] + j];
}

const double& Tensor::operator()(size_t i, size_t j) const {
#ifndef NDEBUG
  if (ndim() != 2) {
    throw std::runtime_error("2D access requires 2D tensor");
  }
  if (i >= shape_[0] || j >= shape_[1]) {
    throw std::out_of_range("Index out of bounds");
  }
#endif
  return data_[i * shape_[1] + j];
}

double& Tensor::operator()(size_t i, size_t j, size_t k) {
#ifndef NDEBUG
  if (ndim() != 3) {
    throw std::runtime_error("3D access requires 3D tensor");
  }
  if (i >= shape_[0] || j >= shape_[1] || k >= shape_[2]) {
    throw std::out_of_range("Index out of bounds");
  }
#endif
  return data_[(i * shape_[1] + j) * shape_[2] + k];
}

const double& Tensor::operator()(size_t i, size_t j, size_t k) const {
#ifndef NDEBUG
  if (ndim() != 3) {
    throw std::runtime_error("3D access requires 3D tensor");
  }
  if (i >= shape_[0] || j >= shape_[1] || k >= shape_[2]) {
    throw std::out_of_range("Index out of bounds");
  }
#endif
  return data_[(i * shape_[1] + j) * shape_[2] + k];
}

// ============================================================================
// Initialization
// ============================================================================

void Tensor::fill(double value) {
  std::fill(data_.begin(), data_.end(), value);
}

void Tensor::zeros() {
  fill(0.0);
}

void Tensor::ones() {
  fill(1.0);
}

void Tensor::random_uniform(double min, double max) {
  std::uniform_real_distribution<double> dist(min, max);
  for (auto& val : data_) {
    val = dist(rng);
  }
}

void Tensor::random_normal(double mean, double stddev) {
  std::normal_distribution<double> dist(mean, stddev);
  for (auto& val : data_) {
    val = dist(rng);
  }
}

// ============================================================================
// Shape Operations
// ============================================================================

void Tensor::reshape(const std::vector<size_t>& new_shape) {
  size_t new_size =
      std::accumulate(new_shape.begin(), new_shape.end(), size_t{1}, std::multiplies<size_t>());
  if (new_size != size()) {
    throw std::invalid_argument("Reshape: total size must be preserved");
  }
  shape_ = new_shape;
}

Tensor Tensor::transpose() const {
  if (ndim() != 2) {
    throw std::runtime_error("Transpose requires 2D tensor");
  }

  size_t rows = shape_[0];
  size_t cols = shape_[1];
  Tensor result(cols, rows);

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      result(j, i) = (*this)(i, j);
    }
  }

  return result;
}

// ============================================================================
// Element-wise Operations
// ============================================================================

void Tensor::check_same_shape(const Tensor& other) const {
  if (shape_ != other.shape_) {
    std::ostringstream oss;
    oss << "Shape mismatch: (";
    for (size_t i = 0; i < shape_.size(); ++i) {
      if (i > 0)
        oss << ", ";
      oss << shape_[i];
    }
    oss << ") vs (";
    for (size_t i = 0; i < other.shape_.size(); ++i) {
      if (i > 0)
        oss << ", ";
      oss << other.shape_[i];
    }
    oss << ")";
    throw std::invalid_argument(oss.str());
  }
}

// OPT-1B: Eliminate copy() - construct result directly (CSAPP 9.9)
Tensor Tensor::operator+(const Tensor& other) const {
  check_same_shape(other);
  Tensor result(shape_);
  const size_t n = data_.size();
  for (size_t i = 0; i < n; ++i) {
    result.data_[i] = data_[i] + other.data_[i];
  }
  return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
  check_same_shape(other);
  Tensor result(shape_);
  const size_t n = data_.size();
  for (size_t i = 0; i < n; ++i) {
    result.data_[i] = data_[i] - other.data_[i];
  }
  return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
  check_same_shape(other);
  Tensor result(shape_);
  const size_t n = data_.size();
  for (size_t i = 0; i < n; ++i) {
    result.data_[i] = data_[i] * other.data_[i];
  }
  return result;
}

Tensor Tensor::operator*(double scalar) const {
  Tensor result(shape_);
  const size_t n = data_.size();
  for (size_t i = 0; i < n; ++i) {
    result.data_[i] = data_[i] * scalar;
  }
  return result;
}

Tensor Tensor::operator/(double scalar) const {
  if (scalar == 0.0) {
    throw std::invalid_argument("Division by zero");
  }
  Tensor result(shape_);
  const size_t n = data_.size();
  const double inv_scalar = 1.0 / scalar;  // Multiply is faster than divide
  for (size_t i = 0; i < n; ++i) {
    result.data_[i] = data_[i] * inv_scalar;
  }
  return result;
}

Tensor& Tensor::operator+=(const Tensor& other) {
  check_same_shape(other);
  for (size_t i = 0; i < data_.size(); ++i) {
    data_[i] += other.data_[i];
  }
  return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
  check_same_shape(other);
  for (size_t i = 0; i < data_.size(); ++i) {
    data_[i] -= other.data_[i];
  }
  return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
  check_same_shape(other);
  for (size_t i = 0; i < data_.size(); ++i) {
    data_[i] *= other.data_[i];
  }
  return *this;
}

Tensor& Tensor::operator*=(double scalar) {
  for (auto& val : data_) {
    val *= scalar;
  }
  return *this;
}

// ============================================================================
// Matrix Operations
// ============================================================================

Tensor Tensor::matmul(const Tensor& other) const {
  if (ndim() != 2 || other.ndim() != 2) {
    throw std::runtime_error("Matrix multiplication requires 2D tensors");
  }

  size_t m = shape_[0];       // rows of this
  size_t n = shape_[1];       // cols of this / rows of other
  size_t p = other.shape_[1]; // cols of other

  if (n != other.shape_[0]) {
    throw std::invalid_argument("Matrix multiplication: incompatible dimensions");
  }

  Tensor result(m, p);
  result.zeros(); // Initialize to zero for accumulation

  // OPT-1A: Blocked i-k-j matrix multiplication (CSAPP 6.6 - Cache blocking)
  // 64×64 blocks fit in L1 cache (64*64*8 bytes = 32KB < typical 32-64KB L1)
  constexpr size_t BLOCK_SIZE = 64;

  const double* a_data = data_.data();
  const double* b_data = other.data_.data();
  double* c_data = result.data_.data();

  // Blocked outer loops
  for (size_t ii = 0; ii < m; ii += BLOCK_SIZE) {
    size_t i_end = std::min(ii + BLOCK_SIZE, m);

    for (size_t kk = 0; kk < n; kk += BLOCK_SIZE) {
      size_t k_end = std::min(kk + BLOCK_SIZE, n);

      for (size_t jj = 0; jj < p; jj += BLOCK_SIZE) {
        size_t j_end = std::min(jj + BLOCK_SIZE, p);

        // Inner block computation with i-k-j order
        for (size_t i = ii; i < i_end; ++i) {
          for (size_t k = kk; k < k_end; ++k) {
            // Hoist A[i,k] out of inner loop (key optimization!)
            double a_ik = a_data[i * n + k];

            // Vectorizable inner loop over columns of B
            for (size_t j = jj; j < j_end; ++j) {
              c_data[i * p + j] += a_ik * b_data[k * p + j];
            }
          }
        }
      }
    }
  }

  return result;
}

double Tensor::dot(const Tensor& other) const {
  if (ndim() != 1 || other.ndim() != 1) {
    throw std::runtime_error("Dot product requires 1D tensors");
  }
  if (size() != other.size()) {
    throw std::invalid_argument("Dot product: size mismatch");
  }

  double result = 0.0;
  for (size_t i = 0; i < size(); ++i) {
    result += data_[i] * other.data_[i];
  }
  return result;
}

// ============================================================================
// Reduction Operations
// ============================================================================

double Tensor::sum() const {
  return std::accumulate(data_.begin(), data_.end(), 0.0);
}

double Tensor::mean() const {
  if (data_.empty()) {
    return 0.0;
  }
  return sum() / static_cast<double>(size());
}

// ============================================================================
// Utility
// ============================================================================

void Tensor::apply(double (*func)(double)) {
  for (auto& val : data_) {
    val = func(val);
  }
}

Tensor Tensor::copy() const {
  Tensor result(shape_);
  result.data_ = data_;
  return result;
}

// ============================================================================
// Static Factories
// ============================================================================

Tensor Tensor::zeros(const std::vector<size_t>& shape) {
  Tensor result(shape);
  result.zeros();
  return result;
}

Tensor Tensor::ones(const std::vector<size_t>& shape) {
  Tensor result(shape);
  result.ones();
  return result;
}

Tensor Tensor::xavier(size_t rows, size_t cols) {
  // Xavier/Glorot initialization: U[-limit, limit]
  // where limit = sqrt(6 / (fan_in + fan_out))
  double limit = std::sqrt(6.0 / static_cast<double>(rows + cols));
  Tensor result(rows, cols);
  result.random_uniform(-limit, limit);
  return result;
}

Tensor Tensor::he(size_t rows, size_t cols) {
  // He initialization: N(0, sqrt(2 / fan_in))
  // Good for ReLU activations
  double stddev = std::sqrt(2.0 / static_cast<double>(cols));
  Tensor result(rows, cols);
  result.random_normal(0.0, stddev);
  return result;
}

// ============================================================================
// Free Functions
// ============================================================================

Tensor operator*(double scalar, const Tensor& tensor) {
  return tensor * scalar;
}

} // namespace nn
} // namespace golomb
