# Fetch Catch2 via CPM
CPMAddPackage(
  NAME Catch2
  GITHUB_REPOSITORY catchorg/Catch2
  VERSION 3.5.0
)

add_executable(golomb_tests
  test_golomb_core.cpp
)

target_link_libraries(golomb_tests
  PRIVATE
    golomb_core
    Catch2::Catch2WithMain
)

include(CTest)
include(Catch)
catch_discover_tests(golomb_tests)
