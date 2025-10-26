#!/usr/bin/env bash
set -e

echo "format code"
find include src tests benchmarks -name "*.hpp" -o -name "*.cpp" | xargs clang-format -i
echo "format ok"
