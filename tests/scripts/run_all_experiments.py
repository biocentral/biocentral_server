#!/usr/bin/env python3
# Run every experiment in tests/scripts/ and collect CSV reports in tests/reports/.

import sys
import subprocess
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent


def main():
    args = sys.argv[1:]

    pytest_args = [
        sys.executable,
        "-m",
        "pytest",
        str(SCRIPTS_DIR),
        "-v",
        "-s",
        "--tb=short",
    ]

    # Forward --slow → --run-slow
    if "--slow" in args:
        args.remove("--slow")
        pytest_args.append("--run-slow")

    # Forward any remaining args to pytest
    pytest_args.extend(args)

    print(f"Running: {' '.join(pytest_args)}")
    print(f"Reports will be written to: {SCRIPTS_DIR.parent / 'reports'}/")
    print()

    result = subprocess.run(pytest_args, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
