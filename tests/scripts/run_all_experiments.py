#!/usr/bin/env python3

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
        "-W",
        "ignore::DeprecationWarning",
        "-W",
        "ignore::UserWarning",
        "-W",
        "ignore::FutureWarning",
    ]

    pytest_args.extend(args)

    print(f"Running: {' '.join(pytest_args)}")
    print(f"Reports will be written to: {SCRIPTS_DIR.parent / 'reports'}/")
    print()

    result = subprocess.run(pytest_args, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
