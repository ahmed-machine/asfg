#!/usr/bin/env python3
"""
End-to-end test runner for multi-frame satellite alignment pipeline.

Downloads multiple frames from USGS, preprocesses, aligns, mosaics, and
runs QA on the final composite image.

Usage:
    python3 scripts/test/run_e2e_test.py --config bahrain_kh4_1968
    python3 scripts/test/run_e2e_test.py --config bahrain_kh4_1968 --skip-download --timeout 14400
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIGS_DIR = Path(__file__).resolve().parent / "e2e_configs"
CACHE_DIR = PROJECT_ROOT / "output"  # Shared preprocessing (downloads, extracted, georef)


def load_config(config_name: str) -> dict:
    """Load a test configuration YAML file."""
    # Try with and without extension
    for ext in ["", ".yaml", ".yml"]:
        path = CONFIGS_DIR / f"{config_name}{ext}"
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)

    print(f"ERROR: Config '{config_name}' not found in {CONFIGS_DIR}")
    available = [p.stem for p in CONFIGS_DIR.glob("*.yaml")]
    if available:
        print(f"Available configs: {', '.join(available)}")
    sys.exit(1)


def resolve_paths(config: dict) -> dict:
    """Resolve relative paths in config to absolute paths."""
    from scripts.paths_config import get_reference

    # Reference
    config["reference_path"] = get_reference(config["reference"])

    # Boundary
    boundary = config["boundary"]
    if not os.path.isabs(boundary):
        boundary = str(PROJECT_ROOT / boundary)
    config["boundary_path"] = boundary

    # Catalogs
    resolved_catalogs = []
    for cat in config.get("catalogs", []):
        if not os.path.isabs(cat):
            cat = str(PROJECT_ROOT / cat)
        resolved_catalogs.append(cat)
    config["catalog_paths"] = resolved_catalogs

    return config


def detect_next_version() -> int:
    """Scan diagnostics/e2e_v*/ and return the next version number."""
    diag_dir = PROJECT_ROOT / "diagnostics"
    existing = glob(str(diag_dir / "e2e_v*"))
    versions = []
    for d in existing:
        m = re.search(r"e2e_v(\d+)$", d)
        if m:
            versions.append(int(m.group(1)))
    return max(versions) + 1 if versions else 1


def snapshot_git_state(out_path: Path):
    """Save git commit hash and diff stat to a file."""
    lines = []
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        lines.append(f"commit: {head.stdout.strip()}")
    except Exception:
        lines.append("commit: unknown")
    try:
        diff = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        lines.append(f"\ngit diff --stat:\n{diff.stdout}")
    except Exception:
        pass
    out_path.write_text("\n".join(lines))


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


def collect_per_scene_results(pipeline_output_dir: Path) -> list:
    """Collect per-scene alignment results from diagnostics directories."""
    diag_dir = pipeline_output_dir / "diagnostics"
    results = []

    if not diag_dir.exists():
        return results

    for entity_dir in sorted(diag_dir.iterdir()):
        if not entity_dir.is_dir():
            continue

        entity_id = entity_dir.name
        entry = {"entity_id": entity_id, "aligned": False}

        # Check for alignment output
        aligned_path = pipeline_output_dir / "aligned" / f"{entity_id}_aligned.tif"
        entry["aligned"] = aligned_path.exists()

        # Check for QA results
        qa_path = entity_dir / "qa.json"
        if qa_path.exists():
            try:
                qa = json.loads(qa_path.read_text())
                # Extract key metrics from QA
                if isinstance(qa, dict):
                    if "selected" in qa:
                        selected = qa["selected"]
                        if isinstance(selected, dict):
                            entry["score"] = selected.get("total_score")
                            img = selected.get("image_metrics", {})
                            entry["patch_med"] = img.get("patch_med")
                            entry["grid_score"] = img.get("grid_score")
                            entry["grade"] = selected.get("grade")
                            entry["accepted"] = selected.get("accepted")
            except Exception:
                pass

        results.append(entry)

    return results


def find_mosaic_qa(pipeline_output_dir: Path) -> dict:
    """Find and load mosaic QA results."""
    mosaic_dir = pipeline_output_dir / "mosaic"
    if not mosaic_dir.exists():
        return {}

    for qa_file in mosaic_dir.glob("*_qa.json"):
        try:
            return json.loads(qa_file.read_text())
        except Exception:
            continue

    return {}


def parse_selection_from_log(log_text: str) -> dict:
    """Extract scene selection metadata from pipeline log."""
    result = {}

    m = re.search(r"Selected (\d+) scenes from (\S+) mission (\S+) \((\S+)\) cam=(\S+) — ([\d.]+)% coverage", log_text)
    if m:
        result["scenes_selected"] = int(m.group(1))
        result["camera_system"] = m.group(2)
        result["mission"] = m.group(3)
        result["date"] = m.group(4)
        result["camera_designation"] = m.group(5)
        result["predicted_coverage"] = float(m.group(6)) / 100.0

    # Count entity IDs
    entity_ids = re.findall(r"Processing: (\S+) \(", log_text)
    result["scenes_processed"] = len(entity_ids)

    # Count successful georefs
    georef_m = re.search(r"Georef complete: (\d+) succeeded, (\d+) failed", log_text)
    if georef_m:
        result["georef_succeeded"] = int(georef_m.group(1))
        result["georef_failed"] = int(georef_m.group(2))

    return result


def build_summary(version: int, config: dict, run_dir: Path,
                  log_text: str, exit_code: int, wall_clock_s: float) -> dict:
    """Build structured summary from pipeline run."""
    pipeline_output_dir = run_dir / "pipeline_output"

    summary = {
        "version": version,
        "test_name": config.get("name", "unknown"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "exit_code": exit_code,
        "wall_clock_s": round(wall_clock_s, 1),
        "config": {
            "boundary": config.get("boundary"),
            "reference": config.get("reference"),
            "prefer_camera": config.get("prefer_camera"),
        },
    }

    # Scene selection
    summary["scene_selection"] = parse_selection_from_log(log_text)

    # Per-scene results
    summary["per_scene"] = collect_per_scene_results(pipeline_output_dir)

    # Mosaic QA
    mosaic_qa = find_mosaic_qa(pipeline_output_dir)
    if mosaic_qa:
        summary["mosaic"] = mosaic_qa

    # Errors from log
    errors = []
    for line in log_text.split("\n"):
        if "ERROR" in line and "WARNING" not in line:
            errors.append(line.strip())
    if errors:
        summary["errors"] = errors[-10:]  # Last 10 errors

    return summary


def run_e2e_test(config: dict, version: int, timeout: int,
                 skip_download: bool = False, cleanup: bool = False):
    """Run the end-to-end pipeline test."""
    run_dir = PROJECT_ROOT / "diagnostics" / f"e2e_v{version}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pipeline_output_dir = run_dir / "pipeline_output"
    log_path = run_dir / "run.log"
    summary_path = run_dir / "summary.json"

    # Snapshot git state
    snapshot_git_state(run_dir / "code_state.txt")

    # Build process.py command.
    # Preprocessing (downloads/extracted/georef) goes to shared CACHE_DIR;
    # per-run outputs (aligned/diagnostics/mosaic) go to pipeline_output_dir.
    cmd = [
        sys.executable, str(PROJECT_ROOT / "process.py"),
        "--csv", *config["catalog_paths"],
        "--reference", config["reference_path"],
        "--boundary", config["boundary_path"],
        "--output-dir", str(pipeline_output_dir),
        "--cache-dir", str(CACHE_DIR),
    ]

    if config.get("prefer_camera"):
        cmd.extend(["--prefer-camera", config["prefer_camera"]])
    if config.get("device"):
        cmd.extend(["--device", str(config["device"])])

    if skip_download:
        cmd.append("--skip-download")

    if cleanup:
        cmd.append("--cleanup")

    print(f"=== run_e2e_test.py: Starting e2e_v{version} ===")
    print(f"  Test: {config.get('name', 'unknown')}")
    print(f"  Output: {run_dir}")
    print(f"  Timeout: {timeout}s ({timeout/3600:.1f}h)")
    print(f"  Command: {' '.join(cmd[:6])}...")
    print()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    start = time.time()
    log_lines = []
    exit_code = -1

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(PROJECT_ROOT),
            bufsize=1,
        )

        try:
            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                if line:
                    decoded = line.decode("utf-8", errors="replace")
                    sys.stdout.write(decoded)
                    sys.stdout.flush()
                    log_lines.append(decoded)

                elapsed = time.time() - start
                if elapsed > timeout:
                    print(f"\n=== TIMEOUT after {elapsed:.0f}s, sending SIGTERM ===")
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        print("=== SIGKILL ===")
                        proc.kill()
                        proc.wait()
                    break

            exit_code = proc.returncode

        except KeyboardInterrupt:
            print("\n=== KeyboardInterrupt, sending SIGTERM ===")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            exit_code = proc.returncode if proc.returncode is not None else -2

    except Exception as e:
        print(f"\n=== Failed to start pipeline: {e} ===")
        log_lines.append(f"\nFailed to start: {e}\n")
        exit_code = -1

    wall_clock_s = time.time() - start

    # Write log
    log_text = "".join(log_lines)
    log_path.write_text(log_text)

    print(f"\n=== Pipeline finished: exit_code={exit_code}, "
          f"wall_clock={wall_clock_s:.1f}s ({wall_clock_s/60:.1f}m) ===\n")

    # Build summary
    summary = build_summary(version, config, run_dir, log_text, exit_code, wall_clock_s)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    # Print summary
    print("=== SUMMARY ===")
    sel = summary.get("scene_selection", {})
    print(f"  Mission: {sel.get('camera_system', '?')} {sel.get('mission', '?')} "
          f"({sel.get('date', '?')}) cam={sel.get('camera_designation', '?')}")
    print(f"  Scenes: {sel.get('scenes_selected', '?')} selected, "
          f"{sel.get('georef_succeeded', '?')} georef'd")

    per_scene = summary.get("per_scene", [])
    aligned_count = sum(1 for s in per_scene if s.get("aligned"))
    print(f"  Aligned: {aligned_count}/{len(per_scene)}")
    for s in per_scene:
        score_str = f"score={s['score']:.1f}" if s.get("score") is not None else "no_score"
        grade_str = f"grade={s.get('grade', '?')}"
        status = "aligned" if s.get("aligned") else "FAILED"
        print(f"    {s['entity_id']}: {status} {score_str} {grade_str}")

    mosaic = summary.get("mosaic", {})
    if mosaic:
        qa = mosaic.get("qa", {})
        print(f"\n  Mosaic QA:")
        print(f"    score={qa.get('score', '?')}, "
              f"grid_score={qa.get('grid_score', '?')}, "
              f"patch_med={qa.get('patch_med', '?')}")
        if "actual_coverage" in mosaic:
            print(f"    coverage={mosaic['actual_coverage']*100:.1f}%")

    if summary.get("errors"):
        print(f"\n  Errors ({len(summary['errors'])}):")
        for err in summary["errors"][:5]:
            print(f"    {err[:120]}")

    print(f"\n  Wall clock: {wall_clock_s/60:.1f} min")
    print(f"  Summary: {summary_path}")

    return summary


def cleanup_old_runs(keep_version: int):
    """Remove large files from old e2e runs, keeping latest + best."""
    diag_dir = PROJECT_ROOT / "diagnostics"
    run_dirs = {}
    for d in sorted(diag_dir.glob("e2e_v*")):
        if d.is_dir():
            m = re.search(r"e2e_v(\d+)$", str(d))
            if m:
                run_dirs[int(m.group(1))] = d

    if len(run_dirs) <= 2:
        return

    # Find best-scoring run
    best_version = None
    best_score = float("inf")
    for ver, d in run_dirs.items():
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text())
            mosaic = summary.get("mosaic", {})
            qa = mosaic.get("qa", {})
            score = qa.get("score")
            if score is not None and score < best_score:
                best_score = score
                best_version = ver
        except Exception:
            continue

    keep = {keep_version}
    if best_version is not None:
        keep.add(best_version)

    preserve = {"summary.json", "run.log", "code_state.txt"}

    freed = 0
    for ver, d in run_dirs.items():
        if ver in keep:
            continue
        # Remove the pipeline_output dir (bulk of disk usage)
        output_dir = d / "pipeline_output"
        if output_dir.exists():
            import shutil
            size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
            shutil.rmtree(output_dir)
            freed += size

    if freed > 0:
        print(f"\n=== Cleanup: freed {freed / 1024 / 1024 / 1024:.1f} GB from old e2e runs ===")


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end multi-frame alignment test")
    parser.add_argument(
        "--config", "-c", required=True,
        help="Test config name (from scripts/test/e2e_configs/)")
    parser.add_argument(
        "--version", "-v", type=int, default=None,
        help="Version number (default: auto-detect next)")
    parser.add_argument(
        "--timeout", "-t", type=int, default=14400,
        help="Timeout in seconds (default: 14400 = 4 hours)")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip USGS download (use cached files)")
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Delete intermediate files after mosaic")
    args = parser.parse_args()

    config = load_config(args.config)
    config = resolve_paths(config)

    version = args.version if args.version is not None else detect_next_version()

    summary = run_e2e_test(
        config, version, args.timeout,
        skip_download=args.skip_download,
        cleanup=args.cleanup,
    )

    cleanup_old_runs(version)

    # Exit with pipeline exit code
    sys.exit(summary.get("exit_code", 1))


if __name__ == "__main__":
    main()
