#!/usr/bin/env bash
set -e

echo "run tests"
ctest --test-dir build --output-on-failure
echo "tests ok"
