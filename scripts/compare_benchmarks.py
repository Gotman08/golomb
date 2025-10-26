#!/usr/bin/env python3
"""
Benchmark comparison script for Golomb project.
Compares current benchmark results with previous runs to detect performance regressions.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BenchmarkResult:
    """Represents a single benchmark result."""
    name: str
    cpu_time: float  # in nanoseconds
    iterations: int
    time_unit: str


@dataclass
class Comparison:
    """Comparison between two benchmark results."""
    name: str
    old_time: float
    new_time: float
    change_percent: float

    @property
    def is_regression(self) -> bool:
        """Check if this is a performance regression (>10% slower)."""
        return self.change_percent > 10.0

    @property
    def is_improvement(self) -> bool:
        """Check if this is a performance improvement (>10% faster)."""
        return self.change_percent < -10.0


def load_benchmark_results(filepath: Path) -> List[BenchmarkResult]:
    """Load benchmark results from JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        results = []
        for benchmark in data.get('benchmarks', []):
            # Skip aggregate entries
            if benchmark.get('aggregate_name'):
                continue

            results.append(BenchmarkResult(
                name=benchmark['name'],
                cpu_time=benchmark['cpu_time'],
                iterations=benchmark['iterations'],
                time_unit=benchmark['time_unit']
            ))

        return results
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error loading benchmark results from {filepath}: {e}", file=sys.stderr)
        return []


def compare_results(old_results: List[BenchmarkResult],
                   new_results: List[BenchmarkResult]) -> List[Comparison]:
    """Compare two sets of benchmark results."""
    old_dict = {r.name: r for r in old_results}
    comparisons = []

    for new_result in new_results:
        old_result = old_dict.get(new_result.name)
        if old_result:
            # Calculate percentage change
            change = ((new_result.cpu_time - old_result.cpu_time) / old_result.cpu_time) * 100

            comparisons.append(Comparison(
                name=new_result.name,
                old_time=old_result.cpu_time,
                new_time=new_result.cpu_time,
                change_percent=change
            ))

    return comparisons


def format_time(nanoseconds: float) -> str:
    """Format time in human-readable format."""
    if nanoseconds < 1000:
        return f"{nanoseconds:.2f} ns"
    elif nanoseconds < 1_000_000:
        return f"{nanoseconds / 1000:.2f} µs"
    elif nanoseconds < 1_000_000_000:
        return f"{nanoseconds / 1_000_000:.2f} ms"
    else:
        return f"{nanoseconds / 1_000_000_000:.2f} s"


def generate_markdown_report(comparisons: List[Comparison]) -> str:
    """Generate a markdown report of the comparisons."""
    if not comparisons:
        return "No benchmark comparisons available (first run or missing baseline)."

    report = ["# Benchmark Comparison Report\n"]
    report.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # Summary statistics
    regressions = [c for c in comparisons if c.is_regression]
    improvements = [c for c in comparisons if c.is_improvement]

    report.append("## Summary\n")
    report.append(f"- Total benchmarks: {len(comparisons)}")
    report.append(f"- Performance regressions (>10% slower): **{len(regressions)}**")
    report.append(f"- Performance improvements (>10% faster): {len(improvements)}")
    report.append("")

    # Detailed results table
    report.append("## Detailed Results\n")
    report.append("| Benchmark | Old Time | New Time | Change | Status |")
    report.append("|-----------|----------|----------|--------|--------|")

    for comp in sorted(comparisons, key=lambda c: c.change_percent, reverse=True):
        old_time_str = format_time(comp.old_time)
        new_time_str = format_time(comp.new_time)
        change_str = f"{comp.change_percent:+.2f}%"

        if comp.is_regression:
            status = "⚠️ REGRESSION"
        elif comp.is_improvement:
            status = "✅ IMPROVEMENT"
        else:
            status = "✓ OK"

        report.append(f"| `{comp.name}` | {old_time_str} | {new_time_str} | {change_str} | {status} |")

    report.append("")

    # Regressions section
    if regressions:
        report.append("## ⚠️ Performance Regressions Detected\n")
        report.append("The following benchmarks are significantly slower than the baseline:\n")
        for comp in sorted(regressions, key=lambda c: c.change_percent, reverse=True):
            report.append(f"- **{comp.name}**: {comp.change_percent:+.2f}% slower "
                         f"({format_time(comp.old_time)} → {format_time(comp.new_time)})")
        report.append("")

    # Improvements section
    if improvements:
        report.append("## ✅ Performance Improvements\n")
        for comp in sorted(improvements, key=lambda c: c.change_percent):
            report.append(f"- **{comp.name}**: {abs(comp.change_percent):.2f}% faster "
                         f"({format_time(comp.old_time)} → {format_time(comp.new_time)})")
        report.append("")

    return "\n".join(report)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: compare_benchmarks.py <new_results.json> [old_results.json]")
        print("  If old_results.json is not provided, only displays new results.")
        sys.exit(1)

    new_results_path = Path(sys.argv[1])
    old_results_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    # Load new results
    new_results = load_benchmark_results(new_results_path)
    if not new_results:
        print("Error: No benchmark results found in new results file.", file=sys.stderr)
        sys.exit(1)

    # If no old results provided, just display new results
    if not old_results_path or not old_results_path.exists():
        print("# Benchmark Results (Baseline)\n")
        print(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        print("| Benchmark | Time | Iterations |")
        print("|-----------|------|------------|")
        for result in new_results:
            print(f"| `{result.name}` | {format_time(result.cpu_time)} | {result.iterations} |")
        sys.exit(0)

    # Load old results and compare
    old_results = load_benchmark_results(old_results_path)
    if not old_results:
        print("Warning: Could not load old results, treating new results as baseline.",
              file=sys.stderr)
        sys.exit(0)

    # Compare and generate report
    comparisons = compare_results(old_results, new_results)
    report = generate_markdown_report(comparisons)

    print(report)

    # Exit with error if regressions detected
    regressions = [c for c in comparisons if c.is_regression]
    if regressions:
        sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()
