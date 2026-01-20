#!/usr/bin/env python3
"""
Flakiness Detection Script

Runs tests multiple times to detect non-deterministic (flaky) tests.
Calculates the flakiness rate as: number of flaky tests / total tests

Usage:
    python tests/scripts/flakiness_detection.py [--runs N] [--output-dir DIR] [--threshold FLOAT]

Examples:
    python tests/scripts/flakiness_detection.py
    python tests/scripts/flakiness_detection.py --runs 20
    python tests/scripts/flakiness_detection.py --runs 10 --threshold 0.05
"""

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple


class FlakyTest(NamedTuple):
    name: str
    pass_count: int
    fail_count: int
    flakiness_rate: float


def run_tests(run_number: int, output_dir: Path, pytest_args: list[str] | None = None) -> Path:
    """Run pytest once and save results to JUnit XML."""
    output_file = output_dir / f"run_{run_number}.xml"
    
    cmd = [
        "uv", "run", "pytest",
        "--tb=no",
        "--no-header",
        "-q",
        f"--junit-xml={output_file}",
    ]
    
    if pytest_args:
        cmd.extend(pytest_args)
    
    print(f"  Run {run_number}...", end=" ", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Count passed/failed from output
    if result.returncode == 0:
        print("✓ passed")
    else:
        print("✗ some failures")
    
    return output_file


def parse_results(results_dir: Path) -> dict[str, list[bool]]:
    """Parse all JUnit XML files and return test results."""
    test_results: dict[str, list[bool]] = defaultdict(list)
    
    for xml_file in sorted(results_dir.glob("run_*.xml")):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            for testcase in root.iter("testcase"):
                classname = testcase.get("classname", "")
                name = testcase.get("name", "")
                test_name = f"{classname}.{name}"
                
                # Check if test failed or errored
                failed = (
                    testcase.find("failure") is not None
                    or testcase.find("error") is not None
                )
                skipped = testcase.find("skipped") is not None
                
                if not skipped:
                    test_results[test_name].append(not failed)
                    
        except ET.ParseError as e:
            print(f"Warning: Could not parse {xml_file.name}: {e}")
    
    return dict(test_results)


def calculate_flakiness(test_results: dict[str, list[bool]]) -> tuple[list[FlakyTest], int, int]:
    """
    Analyze test results and identify flaky tests.
    
    Returns:
        tuple of (flaky_tests, consistent_pass_count, consistent_fail_count)
    """
    flaky_tests: list[FlakyTest] = []
    consistent_pass = 0
    consistent_fail = 0
    
    for test_name, results in test_results.items():
        if len(results) == 0:
            continue
            
        all_passed = all(results)
        all_failed = not any(results)
        
        if all_passed:
            consistent_pass += 1
        elif all_failed:
            consistent_fail += 1
        else:
            # Inconsistent - this is a flaky test
            pass_count = sum(results)
            fail_count = len(results) - pass_count
            # Flakiness rate: how "balanced" the flakiness is (0.5 = maximally flaky)
            flakiness_rate = min(pass_count, fail_count) / len(results)
            flaky_tests.append(FlakyTest(test_name, pass_count, fail_count, flakiness_rate))
    
    # Sort by flakiness rate (most flaky first)
    flaky_tests.sort(key=lambda x: -x.flakiness_rate)
    
    return flaky_tests, consistent_pass, consistent_fail


def print_report(
    num_runs: int,
    total_tests: int,
    consistent_pass: int,
    consistent_fail: int,
    flaky_tests: list[FlakyTest],
) -> float:
    """Print flakiness report and return overall flakiness rate."""
    num_flaky = len(flaky_tests)
    overall_flakiness_rate = num_flaky / total_tests if total_tests > 0 else 0.0
    
    print("\n" + "=" * 60)
    print("FLAKINESS REPORT")
    print("=" * 60)
    print(f"Number of test runs: {num_runs}")
    print(f"Total unique tests: {total_tests}")
    print(f"Consistently passing: {consistent_pass}")
    print(f"Consistently failing: {consistent_fail}")
    print(f"Flaky tests: {num_flaky}")
    print(f"\nOverall Flakiness Rate: {overall_flakiness_rate:.2%}")
    print(f"  (flaky tests / total tests = {num_flaky}/{total_tests})")
    
    if flaky_tests:
        print("\n" + "-" * 60)
        print("FLAKY TESTS (inconsistent results across runs):")
        print("-" * 60)
        for test in flaky_tests:
            print(f"\n  {test.name}")
            print(f"    Passed: {test.pass_count}, Failed: {test.fail_count}, Flakiness: {test.flakiness_rate:.2%}")
    
    return overall_flakiness_rate


def save_summary(output_dir: Path, num_runs: int, total_tests: int, flaky_tests: list[FlakyTest], flakiness_rate: float) -> None:
    """Save summary to a file."""
    summary_file = output_dir / "flakiness_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"flakiness_rate={flakiness_rate:.4f}\n")
        f.write(f"flaky_tests={len(flaky_tests)}\n")
        f.write(f"total_tests={total_tests}\n")
        f.write(f"test_runs={num_runs}\n")
        
        if flaky_tests:
            f.write("\nflaky_test_names:\n")
            for test in flaky_tests:
                f.write(f"  - {test.name} (pass: {test.pass_count}, fail: {test.fail_count})\n")
    
    print(f"\nSummary saved to: {summary_file}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect flaky tests by running the test suite multiple times."
    )
    parser.add_argument(
        "--runs", "-n",
        type=int,
        default=50,
        help="Number of times to run the test suite (default: 50)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("tests/flakiness_results"),
        help="Directory to store test results (default: tests/flakiness_results)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.1,
        help="Flakiness rate threshold for exit code (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--fail-on-flaky",
        action="store_true",
        help="Exit with error code if flakiness exceeds threshold"
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="Additional arguments to pass to pytest"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear any existing results
    for old_file in args.output_dir.glob("run_*.xml"):
        old_file.unlink()
    
    print(f"Running tests {args.runs} times to detect flakiness...")
    print("=" * 60)
    
    # Run tests multiple times
    for i in range(1, args.runs + 1):
        run_tests(i, args.output_dir, args.pytest_args or None)
    
    # Parse and analyze results
    test_results = parse_results(args.output_dir)
    total_tests = len(test_results)
    
    if total_tests == 0:
        print("\nError: No test results found. Check if pytest ran correctly.")
        return 1
    
    flaky_tests, consistent_pass, consistent_fail = calculate_flakiness(test_results)
    
    # Print report
    flakiness_rate = print_report(
        args.runs, total_tests, consistent_pass, consistent_fail, flaky_tests
    )
    
    # Save summary
    save_summary(args.output_dir, args.runs, total_tests, flaky_tests, flakiness_rate)
    
    # Check threshold
    if args.fail_on_flaky and flakiness_rate > args.threshold:
        print(f"\n⚠️  Flakiness rate ({flakiness_rate:.2%}) exceeds threshold ({args.threshold:.2%})")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())