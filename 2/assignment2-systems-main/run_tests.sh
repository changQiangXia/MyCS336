#!/bin/bash
set -e

echo "========================================="
echo "Running CS336 Assignment 2 Tests"
echo "========================================="
echo ""

# Fix for Windows PyTorch distributed (if on Windows)
export USE_LIBUV=0

echo "Running tests... (this may take a few minutes)"
uv run pytest -v ./tests --junitxml=test_results.xml 2>&1 | tee test_output.txt

echo ""
echo "========================================="
echo "Tests completed!"
echo "========================================="
echo ""
echo "Results saved to: test_output.txt"
echo "JUnit XML saved to: test_results.xml"
echo ""

# Display summary
echo "Test Summary:"
grep -E "passed|failed|error|FAILED|SKIPPED" test_output.txt | tail -20 || true

echo ""
