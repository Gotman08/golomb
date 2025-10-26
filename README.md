# Golomb Benchmark History

This branch stores historical benchmark results for performance tracking.

## Structure

- `benchmarks/` - Contains benchmark result JSON files
  - `latest.json` - Most recent benchmark results
  - `YYYYMMDD_HHMMSS_<commit>.json` - Historical benchmark results

## Usage

This branch is automatically updated by the Benchmarks workflow. Do not manually edit files here.

Benchmark results are compared against `latest.json` to detect performance regressions.
