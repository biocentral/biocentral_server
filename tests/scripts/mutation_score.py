#!/usr/bin/env python3
"""Calculate and display mutation score from mutmut results."""

import subprocess
import sys
import re


def get_mutation_score() -> tuple[int, int, int, int, float]:
    """
    Parse mutmut results and calculate mutation score.

    Returns:
        Tuple of (killed, survived, suspicious, total, score_percentage)
    """
    try:
        result = subprocess.run(
            ["mutmut", "results"],
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout + result.stderr
    except FileNotFoundError:
        print("Error: mutmut is not installed. Run 'uv add --dev mutmut' first.")
        sys.exit(1)

    # Parse the summary line from mutmut results
    # Format: "Killed X out of Y mutants"
    # Or parse the status counts

    killed = 0
    survived = 0
    suspicious = 0
    timeout = 0
    skipped = 0

    # Look for status counts in output
    # mutmut results output format varies by version
    lines = output.strip().split("\n")

    for line in lines:
        line_lower = line.lower()
        if "killed" in line_lower:
            match = re.search(r"(\d+)", line)
            if match:
                killed = int(match.group(1))
        elif "survived" in line_lower:
            match = re.search(r"(\d+)", line)
            if match:
                survived = int(match.group(1))
        elif "suspicious" in line_lower:
            match = re.search(r"(\d+)", line)
            if match:
                suspicious = int(match.group(1))
        elif "timeout" in line_lower:
            match = re.search(r"(\d+)", line)
            if match:
                timeout = int(match.group(1))
        elif "skipped" in line_lower:
            match = re.search(r"(\d+)", line)
            if match:
                skipped = int(match.group(1))

    # Alternative: parse "Legend:" section or summary
    # Try to extract from "X/Y" format
    summary_match = re.search(r"(\d+)/(\d+)", output)
    if summary_match and killed == 0:
        killed = int(summary_match.group(1))
        total_from_summary = int(summary_match.group(2))
        survived = total_from_summary - killed

    total = killed + survived + suspicious + timeout
    score = (killed / total * 100) if total > 0 else 0.0

    return killed, survived, suspicious, total, score


def main() -> None:
    """Main entry point."""
    print("=" * 50)
    print("MUTATION TESTING RESULTS")
    print("=" * 50)

    killed, survived, suspicious, total, score = get_mutation_score()

    print(f"\n{'Metric':<20} {'Count':>10}")
    print("-" * 32)
    print(f"{'Killed (caught)':<20} {killed:>10}")
    print(f"{'Survived (missed)':<20} {survived:>10}")
    print(f"{'Suspicious':<20} {suspicious:>10}")
    print(f"{'Total mutants':<20} {total:>10}")
    print("-" * 32)
    print(f"\n🎯 MUTATION SCORE: {killed}/{total} = {score:.1f}%\n")

    if score >= 80:
        print("✅ Excellent mutation score!")
    elif score >= 60:
        print("⚠️  Good mutation score, but room for improvement.")
    elif score >= 40:
        print("⚠️  Moderate mutation score. Consider adding more tests.")
    else:
        print("❌ Low mutation score. Tests may not be catching bugs effectively.")

    # Exit with non-zero code if score is below threshold (useful for CI)
    if score < 60:
        sys.exit(1)


if __name__ == "__main__":
    main()
