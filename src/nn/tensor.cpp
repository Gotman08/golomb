#include "nn/tensor.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <sstream>

namespace golomb {
namespace nn {

namespace {
// Thread-local RNG for random initialization
thread_local std::mt19937 rng{std::random_device{}()};
}  // namespace

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
  size_t total_size = std::accumulate(shape.begin(), shape.end(), size_t{1},
                                       std::multiplies<size_t>());
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

// ============================================================================
// Element Access
// ============================================================================

double& Tensor::operator()(size_t i) {
  if (ndim() != 1) {
    throw std::runtime_error("1D access requires 1D tensor");
  }
  if (i >= shape_[0]) {
    throw std::out_of_range("Index out of bounds");
  }
  return data_[i];
}

const double& Tensor::operator()(size_t i) const {
  if (ndim() != 1) {
    throw std::runtime_error("1D access requires 1D tensor");
  }
  if (i >= shape_[0]) {
    throw std::out_of_range("Index out of bounds");
  }
  return data_[i];
}

double& Tensor::operator()(size_t i, size_t j) {
  if (ndim() != 2) {
    throw std::runtime_error("2D access requires 2D tensor");
  }
  if (i >= shape_[0] || j >= shape_[1]) {
    throw std::out_of_range("Index out of bounds");
  }
  return data_[i * shape_[1] + j];
}

const double& Tensor::operator()(size_t i, size_t j) const {
  if (ndim() != 2) {
    throw std::runtime_error("2D access requires 2D tensor");
  }
  if (i >= shape_[0] || j >= shape_[1]) {
    throw std::out_of_range("Index out of bounds");
  }
  return data_[i * shape_[1] + j];
}

double& Tensor::operator()(size_t i, size_t j, size_t k) {
  if (ndim() != 3) {
    throw std::runtime_error("3D access requires 3D tensor");
  }
  if (i >= shape_[0] || j >= shape_[1] || k >= shape_[2]) {
    throw std::out_of_range("Index out of bounds");
  }
  return data_[(i * shape_[1] + j) * shape_[2] + k];
}

const double& Tensor::operator()(size_t i, size_t j, size_t k) const {
  if (ndim() != 3) {
    throw std::runtime_error("3D access requires 3D tensor");
  }
  if (i >= shape_[0] || j >= shape_[1] || k >= shape_[2]) {
    throw std::out_of_range("Index out of bounds");
  }
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
  size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), size_t{1},
                                     std::multiplies<size_t>());
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
      if (i > 0) oss << ", ";
      oss << shape_[i];
    }
    oss << ") vs (";
    for (size_t i = 0; i < other.shape_.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << other.shape_[i];
    }
    oss << ")";
    throw std::invalid_argument(oss.str());
  }
}

Tensor Tensor::operator+(const Tensor& other) const {
  check_same_shape(other);
  Tensor result = copy();
  result += other;
  return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
  check_same_shape(other);
  Tensor result = copy();
  result -= other;
  return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
  check_same_shape(other);
  Tensor result = copy();
  result *= other;
  return result;
}

Tensor Tensor::operator*(double scalar) const {
  Tensor result = copy();
  result *= scalar;
  return result;
}

Tensor Tensor::operator/(double scalar) const {
  if (scalar == 0.0) {
    throw std::invalid_argument("Division by zero");
  }
  Tensor result = copy();
  for (auto& val : result.data_) {
    val /= scalar;
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

  size_t m = shape_[0];  // rows of this
  size_t n = shape_[1];  // cols of this / rows of other
  size_t p = other.shape_[1];  // cols of other

  if (n != other.shape_[0]) {
    throw std::invalid_argument("Matrix multiplication: incompatible dimensions");
  }

  Tensor result(m, p);

  // NOTE: naive implementation - could be optimized with blocking/BLAS
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < p; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < n; ++k) {
        sum += (*this)(i, k) * other(k, j);
      }
      result(i, j) = sum;
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

}  // namespace nn
}  // namespace golomb
