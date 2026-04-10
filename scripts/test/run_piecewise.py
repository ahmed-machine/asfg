#!/usr/bin/env python3
"""Run fast pipeline verification subsets with pytest.

Examples:
  python scripts/test/run_piecewise.py
  python scripts/test/run_piecewise.py process manifest
  python scripts/test/run_piecewise.py qa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test.piecewise_harness import (
    DEFAULT_SUMMARY_DIR,
    STAGES,
    build_pytest_command,
    build_summary,
    prepare_run_dir,
    print_summary,
    run_pytest,
    snapshot_git_state,
    write_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fast piecewise pipeline tests")
    parser.add_argument(
        "stages",
        nargs="*",
        choices=sorted(STAGES),
        help="Subset of stages to run. Defaults to all fast unit tests.",
    )
    parser.add_argument(
        "--run-id",
        default="piecewise_latest",
        help="Run directory name under the summary directory.",
    )
    parser.add_argument(
        "--summary-dir",
        default=str(DEFAULT_SUMMARY_DIR),
        help="Directory where run logs and summaries are written.",
    )
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional args passed through to pytest.",
    )
    args = parser.parse_args()

    selected = args.stages or list(STAGES)
    summary_dir = Path(args.summary_dir)
    run_dir = prepare_run_dir(summary_dir, args.run_id)
    snapshot_git_state(run_dir / "code_state.txt")

    cmd = build_pytest_command(selected, run_dir, pytest_args=args.pytest_args)

    print(f"=== run_piecewise.py: starting {args.run_id} ===")
    print(f"  Output: {run_dir}")
    print(f"  Stages: {', '.join(selected)}")
    print(f"  Command: {' '.join(cmd)}")
    print()

    exit_code, wall_clock_s, _ = run_pytest(cmd, run_dir, selected_stages=selected)
    summary = build_summary(args.run_id, run_dir, selected, cmd, exit_code, wall_clock_s)
    write_summary(run_dir, summary)
    print_summary(summary, run_dir)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
