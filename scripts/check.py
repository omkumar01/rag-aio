#!/usr/bin/env python3
"""rag-aio developer and CI validation entry point.

Runs repository checks in a deterministic order and returns a non-zero exit
code if any check fails. CI must call this script instead of duplicating
command lists.

Usage:
    python scripts/check.py            # full check (format, lint, types, unit tests)
    python scripts/check.py --fix      # apply formatting/lint autofixes first
    python scripts/check.py --fast     # skip tests marked integration/e2e
    python scripts/check.py --unit     # unit tests only
    python scripts/check.py --integration
    python scripts/check.py --e2e
    python scripts/check.py --coverage
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAILED = []
DURATIONS: list[tuple[str, float]] = []


@dataclass
class Check:
    name: str
    argv: list[str]
    cwd: Path = ROOT
    env: dict[str, str] | None = None
    critical: bool = True  # non-critical failures are reported but don't fail the run


def _uv_run_args(argv: list[str]) -> list[str]:
    uv = shutil.which("uv")
    if uv:
        return [uv, "run", *argv]
    return list(argv)


def run_check(check: Check) -> bool:
    print(f"\n=== {check.name} ===", flush=True)
    env = {**os.environ, **(check.env or {})}
    start = time.monotonic()
    try:
        proc = subprocess.run(
            check.argv,
            cwd=check.cwd,
            env=env,
            check=False,
        )
    except FileNotFoundError as exc:
        print(f"[FAIL] {check.name}: executable not found: {exc}")
        if check.critical:
            FAILED.append(check.name)
        return False
    elapsed = time.monotonic() - start
    DURATIONS.append((check.name, elapsed))
    if proc.returncode == 0:
        print(f"[OK]   {check.name} ({elapsed:.1f}s)")
        return True
    print(f"[FAIL] {check.name} (exit {proc.returncode}, {elapsed:.1f}s)")
    if check.critical:
        FAILED.append(check.name)
    return False


def build_checks(args: argparse.Namespace) -> list[Check]:
    checks: list[Check] = []

    if args.fix:
        checks.append(
            Check(
                "ruff format (fix)",
                _uv_run_args(["ruff", "format", "."]),
            )
        )
        checks.append(
            Check(
                "ruff lint (fix)",
                _uv_run_args(["ruff", "check", "--fix", "."]),
            )
        )

    if not args.skip_format:
        checks.append(
            Check("ruff format --check", _uv_run_args(["ruff", "format", "--check", "."]))
        )
        checks.append(Check("ruff lint", _uv_run_args(["ruff", "check", "."])))

    if not args.fast and not args.skip_types:
        src_dirs = [str(p) for p in sorted(ROOT.glob("packages/*/src"))]
        checks.append(Check("mypy (strict)", _uv_run_args(["mypy", *src_dirs])))

    # Tests
    pytest_argv = ["pytest"]
    if args.coverage:
        pytest_argv += ["--cov", "--cov-report=term-missing"]
    if args.unit:
        pytest_argv += ["-m", "unit"]
    elif args.integration:
        pytest_argv += ["-m", "integration"]
    elif args.e2e:
        pytest_argv += ["-m", "e2e"]
    elif args.fast:
        pytest_argv += ["-m", "not integration and not e2e"]
    checks.append(Check("pytest", _uv_run_args(pytest_argv)))
    # benchmarks/ is outside testpaths, so the harness unit tests need an
    # explicit pass; skipped in the marker-specific integration/e2e modes.
    if not (args.integration or args.e2e):
        checks.append(
            Check("pytest benchmarks (unit)", _uv_run_args(["pytest", "benchmarks", "-m", "unit"]))
        )

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="apply autofixes before checking")
    parser.add_argument(
        "--fast", action="store_true", help="skip type checks and integration/e2e tests"
    )
    parser.add_argument("--unit", action="store_true", help="run unit tests only")
    parser.add_argument("--integration", action="store_true", help="run integration tests only")
    parser.add_argument("--e2e", action="store_true", help="run e2e tests only")
    parser.add_argument("--coverage", action="store_true", help="collect test coverage")
    parser.add_argument("--skip-format", action="store_true")
    parser.add_argument("--skip-types", action="store_true")
    args = parser.parse_args()

    print(f"rag-aio check: root={ROOT}")
    checks = build_checks(args)
    for check in checks:
        run_check(check)

    print("\n=== SUMMARY ===")
    for name, elapsed in DURATIONS:
        print(f"  {name:<24} {elapsed:6.1f}s")
    if FAILED:
        print(f"\nFAILED checks: {', '.join(FAILED)}")
        print("Fix the root cause; do not weaken checks to go green.")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
