#!/usr/bin/env python3
"""Shared harness utilities for fast piecewise pipeline tests."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SUMMARY_DIR = PROJECT_ROOT / "diagnostics" / "test_runs"
STAGES = {
    "process": "process",
    "manifest": "manifest",
    "qa": "qa",
    "selection": "selection",
}


def get_git_commit() -> str:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            check=False,
        )
        return result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def snapshot_git_state(out_path: Path) -> None:
    """Save git commit hash and diff stat to a file."""
    lines = []
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            check=False,
        )
        lines.append(f"commit: {head.stdout.strip()}")
    except Exception:
        lines.append("commit: unknown")
    try:
        diff = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            check=False,
        )
        lines.append(f"\ngit diff --stat:\n{diff.stdout}")
    except Exception:
        pass
    out_path.write_text("\n".join(lines), encoding="utf-8")


def prepare_run_dir(summary_dir: Path, run_id: str) -> Path:
    """Create a clean-ish run directory for the piecewise harness."""
    run_dir = summary_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = run_dir / "artifacts"
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    for name in ("run.log", "summary.json", "junit.xml", "piecewise_results.json", "code_state.txt"):
        path = run_dir / name
        if path.exists():
            path.unlink()
    return run_dir


def build_pytest_command(selected_stages: Iterable[str], run_dir: Path,
                         pytest_args: list[str] | None = None) -> list[str]:
    """Construct the pytest command for a piecewise run."""
    ordered = [stage for stage in STAGES if stage in set(selected_stages)]
    if not ordered:
        ordered = list(STAGES)
    marker_expr = " or ".join(STAGES[name] for name in ordered)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        f"fast and ({marker_expr})",
        "--junitxml",
        str(run_dir / "junit.xml"),
        "tests",
    ]
    if pytest_args:
        cmd.extend(pytest_args)
    return cmd


def run_pytest(cmd: list[str], run_dir: Path, *, selected_stages: list[str]) -> tuple[int, float, str]:
    """Run pytest, streaming output and capturing a persistent log."""
    log_path = run_dir / "run.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PIECEWISE_RUN_DIR"] = str(run_dir)
    env["PIECEWISE_SELECTED_STAGES"] = ",".join(selected_stages)

    start = time.time()
    log_lines: list[str] = []
    exit_code = -1

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(PROJECT_ROOT),
        text=True,
        bufsize=1,
    )

    try:
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_lines.append(line)
        exit_code = proc.returncode if proc.returncode is not None else -1
    finally:
        wall_clock_s = time.time() - start
        log_path.write_text("".join(log_lines), encoding="utf-8")

    return exit_code, wall_clock_s, "".join(log_lines)


def load_piecewise_results(run_dir: Path) -> list[dict]:
    """Load per-test result records emitted by tests/conftest.py."""
    results_path = run_dir / "piecewise_results.json"
    if not results_path.exists():
        return []
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, dict):
        tests = payload.get("tests", [])
        return tests if isinstance(tests, list) else []
    return payload if isinstance(payload, list) else []


def _parse_junit_fallback(run_dir: Path) -> list[dict]:
    """Fallback result parsing from JUnit XML when piecewise JSON is absent."""
    junit_path = run_dir / "junit.xml"
    if not junit_path.exists():
        return []
    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(junit_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    records = []
    for testcase in root.iter("testcase"):
        nodeid = testcase.attrib.get("classname", "")
        name = testcase.attrib.get("name", "")
        combined = f"{nodeid}::{name}" if nodeid else name
        status = "passed"
        if testcase.find("failure") is not None or testcase.find("error") is not None:
            status = "failed"
        elif testcase.find("skipped") is not None:
            status = "skipped"
        stage = "unassigned"
        for candidate in STAGES:
            if re.search(rf"\b{candidate}\b", combined):
                stage = candidate
                break
        records.append(
            {
                "nodeid": combined,
                "status": status,
                "stage": stage,
                "duration_s": float(testcase.attrib.get("time", 0.0)),
            }
        )
    return records


def build_summary(run_id: str, run_dir: Path, selected_stages: list[str],
                  cmd: list[str], exit_code: int, wall_clock_s: float) -> dict:
    """Build a structured summary for a piecewise run."""
    records = load_piecewise_results(run_dir) or _parse_junit_fallback(run_dir)
    results = {}
    for stage in selected_stages:
        stage_records = [r for r in records if r.get("stage") == stage]
        results[stage] = {
            "passed": sum(1 for r in stage_records if r.get("status") == "passed"),
            "failed": sum(1 for r in stage_records if r.get("status") == "failed"),
            "skipped": sum(1 for r in stage_records if r.get("status") == "skipped"),
            "tests": [
                {
                    "nodeid": r.get("nodeid"),
                    "status": r.get("status"),
                    "duration_s": round(float(r.get("duration_s", 0.0)), 4),
                    "artifact_dir": r.get("artifact_dir"),
                }
                for r in stage_records
            ],
        }

    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "exit_code": exit_code,
        "wall_clock_s": round(wall_clock_s, 3),
        "selected_stages": selected_stages,
        "pytest_cmd": cmd,
        "artifacts_dir": str((run_dir / "artifacts").resolve()),
        "results": results,
    }


def write_summary(run_dir: Path, summary: dict) -> Path:
    """Persist summary.json and return its path."""
    path = run_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def print_summary(summary: dict, run_dir: Path) -> None:
    """Print a compact run summary to stdout."""
    print("\n=== Piecewise Summary ===")
    print(f"  Run: {summary['run_id']}")
    print(f"  Exit: {summary['exit_code']}")
    print(f"  Wall clock: {summary['wall_clock_s']:.2f}s")
    for stage in summary["selected_stages"]:
        stage_data = summary["results"].get(stage, {})
        print(
            f"  {stage}: "
            f"{stage_data.get('passed', 0)} passed, "
            f"{stage_data.get('failed', 0)} failed, "
            f"{stage_data.get('skipped', 0)} skipped"
        )
    print(f"  Summary: {run_dir / 'summary.json'}")
    print(f"  Log: {run_dir / 'run.log'}")
