#!/usr/bin/env bash
set -e

echo "build golomb-opt"
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j
echo "build ok"
