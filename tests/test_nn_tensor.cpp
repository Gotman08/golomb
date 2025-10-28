#include "nn/tensor.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace golomb::nn;

TEST_CASE("Tensor construction", "[nn][tensor]") {
  SECTION("1D tensor from size") {
    Tensor t(5);
    REQUIRE(t.ndim() == 1);
    REQUIRE(t.shape()[0] == 5);
    REQUIRE(t.size() == 5);
  }

  SECTION("2D tensor from dimensions") {
    Tensor t(3, 4);
    REQUIRE(t.ndim() == 2);
    REQUIRE(t.shape()[0] == 3);
    REQUIRE(t.shape()[1] == 4);
    REQUIRE(t.size() == 12);
  }

  SECTION("Tensor from shape vector") {
    Tensor t(std::vector<size_t>{2, 3, 4});
    REQUIRE(t.ndim() == 3);
    REQUIRE(t.shape()[0] == 2);
    REQUIRE(t.shape()[1] == 3);
    REQUIRE(t.shape()[2] == 4);
    REQUIRE(t.size() == 24);
  }

  SECTION("1D tensor from initializer list") {
    Tensor t{1.0, 2.0, 3.0};
    REQUIRE(t.ndim() == 1);
    REQUIRE(t.size() == 3);
    REQUIRE(t(0) == 1.0);
    REQUIRE(t(1) == 2.0);
    REQUIRE(t(2) == 3.0);
  }
}

TEST_CASE("Tensor element access", "[nn][tensor]") {
  SECTION("1D access") {
    Tensor t(3);
    t(0) = 1.0;
    t(1) = 2.0;
    t(2) = 3.0;
    REQUIRE(t(0) == 1.0);
    REQUIRE(t(1) == 2.0);
    REQUIRE(t(2) == 3.0);
  }

  SECTION("2D access") {
    Tensor t(2, 3);
    t(0, 0) = 1.0;
    t(0, 1) = 2.0;
    t(1, 2) = 6.0;
    REQUIRE(t(0, 0) == 1.0);
    REQUIRE(t(0, 1) == 2.0);
    REQUIRE(t(1, 2) == 6.0);
  }

  SECTION("3D access") {
    Tensor t(std::vector<size_t>{2, 2, 2});
    t(0, 0, 0) = 1.0;
    t(1, 1, 1) = 8.0;
    REQUIRE(t(0, 0, 0) == 1.0);
    REQUIRE(t(1, 1, 1) == 8.0);
  }
}

TEST_CASE("Tensor initialization", "[nn][tensor]") {
  SECTION("Fill with value") {
    Tensor t(3, 3);
    t.fill(5.0);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        REQUIRE(t(i, j) == 5.0);
      }
    }
  }

  SECTION("Zeros") {
    Tensor t(2, 2);
    t.fill(1.0);
    t.zeros();
    REQUIRE(t(0, 0) == 0.0);
    REQUIRE(t(1, 1) == 0.0);
  }

  SECTION("Ones") {
    Tensor t(2, 2);
    t.ones();
    REQUIRE(t(0, 0) == 1.0);
    REQUIRE(t(1, 1) == 1.0);
  }

  SECTION("Static zeros") {
    auto t = Tensor::zeros({2, 3});
    REQUIRE(t.size() == 6);
    REQUIRE(t.sum() == 0.0);
  }

  SECTION("Static ones") {
    auto t = Tensor::ones({2, 3});
    REQUIRE(t.size() == 6);
    REQUIRE(t.sum() == 6.0);
  }

  SECTION("Xavier initialization") {
    auto t = Tensor::xavier(100, 100);
    REQUIRE(t.ndim() == 2);
    REQUIRE(t.shape()[0] == 100);
    REQUIRE(t.shape()[1] == 100);
    // Check that values are in reasonable range
    double max_val = 0.0;
    for (const auto& val : t.data()) {
      max_val = std::max(max_val, std::abs(val));
    }
    REQUIRE(max_val > 0.0); // Not all zeros
    REQUIRE(max_val < 1.0); // Reasonable scale
  }

  SECTION("He initialization") {
    auto t = Tensor::he(100, 100);
    REQUIRE(t.ndim() == 2);
    // Check that values have reasonable variance
    double mean = t.mean();
    REQUIRE_THAT(mean, Catch::Matchers::WithinAbs(0.0, 0.1));
  }
}

TEST_CASE("Tensor reshape and transpose", "[nn][tensor]") {
  SECTION("Reshape") {
    Tensor t(2, 3);
    t(0, 0) = 1.0;
    t(0, 1) = 2.0;
    t(0, 2) = 3.0;
    t(1, 0) = 4.0;
    t(1, 1) = 5.0;
    t(1, 2) = 6.0;

    t.reshape({3, 2});
    REQUIRE(t.shape()[0] == 3);
    REQUIRE(t.shape()[1] == 2);
    // Data is preserved in row-major order
    REQUIRE(t(0, 0) == 1.0);
    REQUIRE(t(0, 1) == 2.0);
    REQUIRE(t(1, 0) == 3.0);
    REQUIRE(t(1, 1) == 4.0);
  }

  SECTION("Transpose") {
    Tensor t(2, 3);
    t(0, 0) = 1.0;
    t(0, 1) = 2.0;
    t(0, 2) = 3.0;
    t(1, 0) = 4.0;
    t(1, 1) = 5.0;
    t(1, 2) = 6.0;

    auto t_T = t.transpose();
    REQUIRE(t_T.shape()[0] == 3);
    REQUIRE(t_T.shape()[1] == 2);
    REQUIRE(t_T(0, 0) == 1.0);
    REQUIRE(t_T(1, 0) == 2.0);
    REQUIRE(t_T(2, 0) == 3.0);
    REQUIRE(t_T(0, 1) == 4.0);
    REQUIRE(t_T(1, 1) == 5.0);
    REQUIRE(t_T(2, 1) == 6.0);
  }
}

TEST_CASE("Tensor element-wise operations", "[nn][tensor]") {
  SECTION("Addition") {
    Tensor a{1.0, 2.0, 3.0};
    Tensor b{4.0, 5.0, 6.0};
    auto c = a + b;
    REQUIRE(c(0) == 5.0);
    REQUIRE(c(1) == 7.0);
    REQUIRE(c(2) == 9.0);
  }

  SECTION("Subtraction") {
    Tensor a{5.0, 7.0, 9.0};
    Tensor b{4.0, 5.0, 6.0};
    auto c = a - b;
    REQUIRE(c(0) == 1.0);
    REQUIRE(c(1) == 2.0);
    REQUIRE(c(2) == 3.0);
  }

  SECTION("Element-wise multiplication") {
    Tensor a{2.0, 3.0, 4.0};
    Tensor b{5.0, 6.0, 7.0};
    auto c = a * b;
    REQUIRE(c(0) == 10.0);
    REQUIRE(c(1) == 18.0);
    REQUIRE(c(2) == 28.0);
  }

  SECTION("Scalar multiplication") {
    Tensor a{1.0, 2.0, 3.0};
    auto b = a * 2.0;
    REQUIRE(b(0) == 2.0);
    REQUIRE(b(1) == 4.0);
    REQUIRE(b(2) == 6.0);

    auto c = 3.0 * a;
    REQUIRE(c(0) == 3.0);
    REQUIRE(c(1) == 6.0);
    REQUIRE(c(2) == 9.0);
  }

  SECTION("Scalar division") {
    Tensor a{2.0, 4.0, 6.0};
    auto b = a / 2.0;
    REQUIRE(b(0) == 1.0);
    REQUIRE(b(1) == 2.0);
    REQUIRE(b(2) == 3.0);
  }

  SECTION("In-place operations") {
    Tensor a{1.0, 2.0, 3.0};
    Tensor b{1.0, 1.0, 1.0};

    a += b;
    REQUIRE(a(0) == 2.0);
    REQUIRE(a(1) == 3.0);
    REQUIRE(a(2) == 4.0);

    a -= b;
    REQUIRE(a(0) == 1.0);
    REQUIRE(a(1) == 2.0);
    REQUIRE(a(2) == 3.0);

    a *= 2.0;
    REQUIRE(a(0) == 2.0);
    REQUIRE(a(1) == 4.0);
    REQUIRE(a(2) == 6.0);
  }
}

TEST_CASE("Tensor matrix operations", "[nn][tensor]") {
  SECTION("Matrix multiplication") {
    // 2x3 matrix
    Tensor A(2, 3);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(0, 2) = 3.0;
    A(1, 0) = 4.0;
    A(1, 1) = 5.0;
    A(1, 2) = 6.0;

    // 3x2 matrix
    Tensor B(3, 2);
    B(0, 0) = 7.0;
    B(0, 1) = 8.0;
    B(1, 0) = 9.0;
    B(1, 1) = 10.0;
    B(2, 0) = 11.0;
    B(2, 1) = 12.0;

    // Result: 2x2
    auto C = A.matmul(B);
    REQUIRE(C.shape()[0] == 2);
    REQUIRE(C.shape()[1] == 2);

    // Manual calculation:
    // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    REQUIRE(C(0, 0) == 58.0);
    REQUIRE(C(0, 1) == 64.0);
    REQUIRE(C(1, 0) == 139.0);
    REQUIRE(C(1, 1) == 154.0);
  }

  SECTION("Dot product") {
    Tensor a{1.0, 2.0, 3.0};
    Tensor b{4.0, 5.0, 6.0};
    double result = a.dot(b);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    REQUIRE(result == 32.0);
  }
}

TEST_CASE("Tensor reduction operations", "[nn][tensor]") {
  SECTION("Sum") {
    Tensor t{1.0, 2.0, 3.0, 4.0};
    REQUIRE(t.sum() == 10.0);
  }

  SECTION("Mean") {
    Tensor t{1.0, 2.0, 3.0, 4.0};
    REQUIRE(t.mean() == 2.5);
  }
}

TEST_CASE("Tensor utility functions", "[nn][tensor]") {
  SECTION("Apply function") {
    Tensor t{1.0, 2.0, 3.0, 4.0};
    t.apply([](double x) { return x * x; });
    REQUIRE(t(0) == 1.0);
    REQUIRE(t(1) == 4.0);
    REQUIRE(t(2) == 9.0);
    REQUIRE(t(3) == 16.0);
  }

  SECTION("Copy") {
    Tensor a{1.0, 2.0, 3.0};
    auto b = a.copy();
    b(0) = 99.0;
    REQUIRE(a(0) == 1.0);  // Original unchanged
    REQUIRE(b(0) == 99.0); // Copy modified
  }
}
