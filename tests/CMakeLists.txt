# Fetch Catch2 via CPM
CPMAddPackage(
  NAME Catch2
  GITHUB_REPOSITORY catchorg/Catch2
  VERSION 3.5.0
)

add_executable(golomb_tests
  test_golomb_core.cpp
  test_nn_tensor.cpp
  test_nn_activations.cpp
)

target_link_libraries(golomb_tests
  PRIVATE
    golomb_core
    golomb_nn
    Catch2::Catch2WithMain
)

include(CTest)
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(golomb_tests)
