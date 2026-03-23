#!/usr/bin/env python3
"""
Test runner for the Bahrain alignment pipeline.

Runs the pipeline, captures stdout/stderr, parses logs into a structured
summary.json for automated analysis.

Usage:
    python3 scripts/test/run_test.py [--version N] [--timeout 2400]
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paths_config import get_target, get_reference

TARGET = get_target("bahrain_1977")
REFERENCE = get_reference("kh9_dzb1212")
ANCHORS = str(PROJECT_ROOT / "data" / "bahrain_anchor_gcps.json")


def detect_next_version():
    """Scan diagnostics/run_v*/ and return the next version number."""
    diag_dir = PROJECT_ROOT / "diagnostics"
    existing = glob(str(diag_dir / "run_v*"))
    versions = []
    for d in existing:
        m = re.search(r"run_v(\d+)$", d)
        if m:
            versions.append(int(m.group(1)))
    return max(versions) + 1 if versions else 1


def snapshot_git_state(out_path):
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
    try:
        diff_staged = subprocess.run(
            ["git", "diff", "--staged", "--stat"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        if diff_staged.stdout.strip():
            lines.append(f"\ngit diff --staged --stat:\n{diff_staged.stdout}")
    except Exception:
        pass
    Path(out_path).write_text("\n".join(lines))


def parse_log(log_text):
    """Parse pipeline log into structured data. Returns dict of extracted fields."""
    result = {
        "step_timings": {},
        "coarse_offset": {},
        "anchors": {"located": 0, "total": 0, "rejected": 0},
        "anchor_details": [],
        "auto_gcps": {"ransac_survivors": 0, "roma_coverage": 0},
        "gcp_selection": {"count": 0, "coverage": 0.0},
        "grid_optimizer": {},
        "flow": {"reliable_pct": None, "mean_correction_m": None, "bias_dx_m": None, "bias_dy_m": None, "applied": None},
        "reclamation": {},
        "icp": {},
        "grid_qa": {},
        "tps_qa": {},
        "cv": {"fit_m": None, "cv_m": None},
        "grid_independent": {},
        "tps_independent": {},
        "selected": None,
        "warnings": [],
        "errors": [],
    }

    # Step timings: [step_name] Ns
    for m in re.finditer(r"\[(\w+)\]\s+([\d.]+)s", log_text):
        step, secs = m.group(1), float(m.group(2))
        # Skip grid optimizer iteration lines (they contain extra fields)
        if re.match(r"L\d+", step):
            continue
        result["step_timings"][step] = secs

    # TOTAL timing
    m = re.search(r"\[TOTAL\]\s+([\d.]+)s", log_text)
    if m:
        result["step_timings"]["TOTAL"] = float(m.group(1))

    # Coarse offset (second pass after translation, or only pass)
    # Find the last occurrence of coarse estimate
    for m in re.finditer(r"Coarse estimate:\s*dx=([+\-]?\d+)m,\s*dy=([+\-]?\d+)m", log_text):
        result["coarse_offset"] = {"dx_m": int(m.group(1)), "dy_m": int(m.group(2))}

    # Anchor located
    anchor_names_seen = set()
    for m in re.finditer(r"Located:\s+(.+?)\s+\((.+?)\)\s*\n\s*displacement:\s*dE=([+\-]?[\d.]+)m,\s*dN=([+\-]?[\d.]+)m", log_text):
        name = m.group(1)
        if name not in anchor_names_seen:
            anchor_names_seen.add(name)
            result["anchor_details"].append({
                "name": name,
                "status": "located",
                "method": m.group(2),
                "dE": float(m.group(3)),
                "dN": float(m.group(4)),
            })

    # Anchor recovered via Pass 1.5
    for m in re.finditer(r"(\S.*?):\s+RECOVERED via Pass 1\.5.*?disp:\s*dE=([+\-]?[\d.]+)m,\s*dN=([+\-]?[\d.]+)m", log_text):
        name = m.group(1).strip()
        if name not in anchor_names_seen:
            anchor_names_seen.add(name)
            result["anchor_details"].append({
                "name": name,
                "status": "located",
                "method": "Pass 1.5 recovery",
                "dE": float(m.group(2)),
                "dN": float(m.group(3)),
            })

    # Anchor skipped
    for m in re.finditer(r"Skipped:\s+(.+?)\s+\((.+?)\)", log_text):
        name = m.group(1)
        if name not in anchor_names_seen:
            anchor_names_seen.add(name)
            result["anchor_details"].append({
                "name": name,
                "status": "skipped",
                "reason": m.group(2),
            })

    # Anchor rejection
    m = re.search(r"Anchor QA: rejected (\d+) displacement outlier", log_text)
    if m:
        result["anchors"]["rejected"] = int(m.group(1))

    # Anchor count
    m = re.search(r"(\d+) anchor GCPs located", log_text)
    if m:
        result["anchors"]["located"] = int(m.group(1))

    # Total anchors loaded
    m = re.search(r"Loaded (\d+) anchors", log_text)
    if m:
        result["anchors"]["total"] = int(m.group(1))

    # RoMa coverage
    m = re.search(r"RoMa coverage:\s+(\d+)\s+matches", log_text)
    if m:
        result["auto_gcps"]["roma_coverage"] = int(m.group(1))

    # RANSAC
    m = re.search(r"Geographic RANSAC.*?:\s+(\d+)\s*->\s*(\d+)", log_text)
    if m:
        result["auto_gcps"]["ransac_survivors"] = int(m.group(2))

    # GCP selection
    m = re.search(r"Selected (\d+) GCPs.*?coverage:\s*([\d.]+)%?\)?", log_text)
    if m:
        result["gcp_selection"]["count"] = int(m.group(1))
        cov = float(m.group(2))
        result["gcp_selection"]["coverage"] = cov / 100.0 if cov > 1 else cov

    # Grid optimizer iterations (capture last iteration per level)
    for m in re.finditer(r"\[L(\d+).*?\]\s+Iter\s+(\d+)/(\d+)\s*\|\s*total=([\d.]+)m\s+data=([\d.]+)\s+cham=([\d.]+)", log_text):
        level = int(m.group(1))
        iter_num = int(m.group(2))
        max_iters = int(m.group(3))
        result["grid_optimizer"][f"L{level}_final_total"] = float(m.group(4))
        result["grid_optimizer"][f"L{level}_final_data"] = float(m.group(5))
        result["grid_optimizer"][f"L{level}_final_chamfer"] = float(m.group(6))
        result["grid_optimizer"][f"L{level}_final_iter"] = iter_num
        result["grid_optimizer"][f"L{level}_max_iters"] = max_iters
        result["grid_optimizer"][f"L{level}_converged"] = iter_num < max_iters

    # Grid optimizer iter 1 losses (hierarchical)
    for m in re.finditer(r"\[L(\d+).*?\]\s+Iter 1:\s+data=([\d.]+)\s+cham=([\d.]+)", log_text):
        level = int(m.group(1))
        result["grid_optimizer"][f"L{level}_init_data"] = float(m.group(2))
        result["grid_optimizer"][f"L{level}_init_chamfer"] = float(m.group(3))

    # Grid optimizer iter 1 losses (single-level)
    m = re.search(r"Iter 1 weighted contributions \(m\):\s*\n\s*data=([\d.]+)\s+chamfer=([\d.]+)", log_text)
    if m:
        result["grid_optimizer"]["init_data"] = float(m.group(1))
        result["grid_optimizer"]["init_chamfer"] = float(m.group(2))

    # Flow refinement
    m = re.search(r"\[FlowRefine\]\s+(\d+)%\s+reliable.*?mean correction\s+([\d.]+)m.*?max\s+([\d.]+)m", log_text)
    if m:
        result["flow"]["reliable_pct"] = int(m.group(1))
        result["flow"]["mean_correction_m"] = float(m.group(2))

    # Flow bias
    m = re.search(r"\[FlowRefine\].*?median bias:\s*dx=([+\-]?[\d.]+)m,\s*dy=([+\-]?[\d.]+)m", log_text)
    if m:
        result["flow"]["bias_dx_m"] = float(m.group(1))
        result["flow"]["bias_dy_m"] = float(m.group(2))

    # Flow applied/skipped
    if re.search(r"\[FlowRefine\]\s+Only \d+% reliable.*skipping", log_text):
        result["flow"]["applied"] = False
    elif re.search(r"\[FlowRefine\]\s+Post-refinement complete", log_text):
        result["flow"]["applied"] = True

    # Fold check (final)
    m = re.search(r"WARNING: Final warp has ([\d.]+)% folds.*Falling back to pure affine", log_text)
    if m:
        result["grid_optimizer"]["fold_frac"] = float(m.group(1)) / 100.0
        result["grid_optimizer"]["fold_fallback"] = True
    elif re.search(r"WARNING:.*Falling back to pure affine warp", log_text):
        result["grid_optimizer"]["fold_fallback"] = True
    else:
        result["grid_optimizer"]["fold_fallback"] = False
        # Capture fold fraction from per-level or final check
        fold_fracs = []
        for fm in re.finditer(r"Fold check:\s*([\d.]+)%\s*(?:folded|folds|\(ok\))", log_text):
            fold_fracs.append(float(fm.group(1)) / 100.0)
        if re.search(r"Fold check:\s*clean", log_text):
            fold_fracs.append(0.0)
        if fold_fracs:
            result["grid_optimizer"]["fold_frac"] = fold_fracs[-1]

    # Reclamation
    m = re.search(r"\[Reclamation\]\s*Raw XOR:\s*([\d.]+)%.*?cleaning:\s*([\d.]+)%\s*\((\d+)\s*blobs?,\s*(\d+)\s*large\)", log_text)
    if m:
        result["reclamation"] = {
            "raw_pct": float(m.group(1)),
            "cleaned_pct": float(m.group(2)),
            "n_blobs": int(m.group(3)),
            "n_large": int(m.group(4)),
        }

    # ICP
    m = re.search(r"\[ICP\]\s*Coastline correction:\s*dx=([+\-]?[\d.]+)m,\s*dy=([+\-]?[\d.]+)m", log_text)
    if m:
        result["icp"] = {"applied": True, "dx_m": float(m.group(1)), "dy_m": float(m.group(2))}
    elif re.search(r"\[ICP\]\s*Too few", log_text):
        result["icp"] = {"applied": False, "reason": "too_few_points"}
    elif re.search(r"\[ICP\]\s*Correction too large", log_text):
        result["icp"] = {"applied": False, "reason": "correction_too_large"}
    elif re.search(r"\[ICP\]\s*Negligible", log_text):
        result["icp"] = {"applied": False, "reason": "negligible"}

    # Grid QA — handles both old format (west=42m) and new format (west=n/a, grid=18/24)
    def _parse_qa_line(prefix, text):
        m = re.search(prefix + r"\s*west=(\d+|n/a)m?\s+center=(\d+|n/a)m?\s+east=(\d+|n/a)m?\s+"
                       r"(?:north=([+\-]?\d+)m?\s+)?patch=(\d+)m\s+"
                       r"(?:grid=(\d+)/(\d+)\s+)?"
                       r"stable_iou=([\d.]+)\s+score=(\d+)", text)
        if not m:
            return None
        def _int_or_none(s):
            return int(s) if s and s != "n/a" else None
        qa = {
            "west": _int_or_none(m.group(1)),
            "center": _int_or_none(m.group(2)),
            "east": _int_or_none(m.group(3)),
            "north": _int_or_none(m.group(4)),
            "patch_med": int(m.group(5)),
            "stable_iou": float(m.group(8)),
            "score": int(m.group(9)),
        }
        if m.group(6) is not None:
            qa["grid_valid"] = int(m.group(6))
            qa["grid_total"] = int(m.group(7))
        return qa

    grid_qa = _parse_qa_line(r"Grid warp QA:", log_text)
    if grid_qa:
        result["grid_qa"] = grid_qa

    tps_qa = _parse_qa_line(r"TPS fallback QA:", log_text)
    if tps_qa:
        result["tps_qa"] = tps_qa

    # Cross-validation
    m = re.search(r"Cross-validation:\s*fit=([\d.]+)m,\s*CV=([\d.]+)m", log_text)
    if m:
        result["cv"] = {"fit_m": float(m.group(1)), "cv_m": float(m.group(2))}

    # Grid independent QA
    m = re.search(r"Grid independent QA:\s*total=([\d.]+).*?confidence=([\d.]+).*?accepted=(True|False)", log_text)
    if m:
        result["grid_independent"] = {
            "total": float(m.group(1)),
            "confidence": float(m.group(2)),
            "accepted": m.group(3) == "True",
        }

    # TPS independent QA
    m = re.search(r"TPS independent QA:\s*total=([\d.]+).*?confidence=([\d.]+).*?accepted=(True|False)", log_text)
    if m:
        result["tps_independent"] = {
            "total": float(m.group(1)),
            "confidence": float(m.group(2)),
            "accepted": m.group(3) == "True",
        }

    # Selected candidate
    m = re.search(r"Grid optimizer wins|Selected candidate.*?accepted=(True|False)", log_text)
    if m:
        # Determine from the QA lines which was selected
        pass
    m = re.search(r"(Grid optimizer|TPS fallback) wins", log_text)
    if m:
        result["selected"] = "grid" if "Grid" in m.group(1) else "tps"

    # Warnings
    if re.search(r"cross_validation_high", log_text):
        result["warnings"].append("cross_validation_high")
    for m in re.finditer(r"WARNING:\s+(.+)", log_text):
        result["warnings"].append(m.group(1).strip())

    return result


def build_summary(version, run_dir, log_text, exit_code, wall_clock_s, qa_path):
    """Build the summary.json structure from parsed log and qa.json."""
    parsed = parse_log(log_text)

    # Get git info
    git_commit = "unknown"
    git_dirty = False
    code_state_path = run_dir / "code_state.txt"
    if code_state_path.exists():
        cs = code_state_path.read_text()
        m = re.search(r"commit:\s*(\w+)", cs)
        if m:
            git_commit = m.group(1)[:7]
        git_dirty = "changed" in cs or "modified" in cs or bool(re.search(r"\d+ file", cs))

    # Load qa.json if it exists
    qa_data = {}
    if qa_path.exists():
        try:
            qa_data = json.loads(qa_path.read_text())
        except Exception:
            parsed["errors"].append(f"Failed to parse {qa_path}")

    # Extract grid and tps reports from qa.json
    grid_report = {}
    tps_report = {}
    selected = parsed.get("selected")
    for report in qa_data.get("reports", []):
        if report.get("candidate") == "grid":
            grid_report = report
        elif report.get("candidate") == "tps":
            tps_report = report
    if not selected:
        selected = qa_data.get("selected_candidate", "grid")

    def report_to_summary(report):
        if not report:
            return None
        im = report.get("image_metrics", {})
        s = {
            "score": round(im.get("score", 0), 1),
            "total": round(report.get("total_score", 0), 1),
            "accepted": report.get("accepted", False),
            "west": round(im["west"]) if im.get("west") is not None else None,
            "center": round(im["center"]) if im.get("center") is not None else None,
            "east": round(im["east"]) if im.get("east") is not None else None,
            "north": round(im["north_shift"]) if im.get("north_shift") is not None else None,
            "patch_med": round(im.get("patch_med", 0)),
            "patch_p90": round(im.get("patch_p90", 0), 1) if im.get("patch_p90") is not None else None,
            "patch_count": im.get("patch_count"),
            "stable_iou": round(im.get("stable_iou", 0), 3),
            "shore_iou": round(im.get("shore_iou", 0), 3),
            "grid_score": round(im.get("grid_score", 0), 1) if im.get("grid_score") is not None else None,
        }
        if im.get("grid"):
            s["grid"] = {
                "valid_count": im["grid"].get("valid_count"),
                "total_count": im["grid"].get("total_count"),
            }
        if im.get("score_breakdown"):
            s["score_breakdown"] = im["score_breakdown"]
        return s

    grid_summary = report_to_summary(grid_report)
    tps_summary = report_to_summary(tps_report)

    # Fallback: use parsed log scores when qa.json is missing/incomplete
    if grid_summary is None and parsed.get("grid_qa"):
        gq = parsed["grid_qa"]
        grid_summary = {
            "score": gq["score"], "total": gq["score"],
            "accepted": True, "west": gq.get("west"), "center": gq.get("center"),
            "east": gq.get("east"), "north": gq.get("north"),
            "patch_med": gq["patch_med"], "patch_p90": None,
            "patch_count": None, "stable_iou": gq["stable_iou"],
            "shore_iou": None,
        }
    if tps_summary is None and parsed.get("tps_qa"):
        tq = parsed["tps_qa"]
        tps_summary = {
            "score": tq["score"], "total": tq["score"],
            "accepted": True, "west": tq.get("west"), "center": tq.get("center"),
            "east": tq.get("east"), "north": tq.get("north"),
            "patch_med": tq["patch_med"], "patch_p90": None,
            "patch_count": None, "stable_iou": tq["stable_iou"],
            "shore_iou": None,
        }

    # Rejection reasons from qa.json
    for report in qa_data.get("reports", []):
        for reason in report.get("reasons", []):
            if reason not in parsed["warnings"]:
                parsed["warnings"].append(reason)

    # CV and GCP info from qa.json
    cv_mean_m = None
    gcp_count = None
    coverage = None
    if grid_report:
        cv_mean_m = grid_report.get("cv_mean_m")
        coverage = grid_report.get("coverage")
    gcp_count = qa_data.get("metadata", {}).get("gcp_count")

    # Override from parsed log if available
    if parsed["gcp_selection"]["count"]:
        gcp_count = parsed["gcp_selection"]["count"]
    if parsed["gcp_selection"]["coverage"]:
        coverage = parsed["gcp_selection"]["coverage"]

    summary = {
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "exit_code": exit_code,
        "wall_clock_s": round(wall_clock_s, 1),
        "grid": grid_summary,
        "tps": tps_summary,
        "selected": selected,
        "cv_mean_m": round(cv_mean_m, 1) if cv_mean_m is not None else None,
        "gcp_count": gcp_count,
        "coverage": round(coverage, 3) if coverage is not None else None,
        "anchors": parsed["anchors"],
        "auto_gcps": parsed["auto_gcps"],
        "flow": parsed["flow"],
        "grid_optimizer": parsed["grid_optimizer"],
        "reclamation": parsed["reclamation"],
        "icp": parsed["icp"],
        "step_timings": parsed["step_timings"],
        "anchor_details": parsed["anchor_details"],
        "warnings": parsed["warnings"],
        "errors": parsed["errors"],
    }

    # Load hierarchical profile if available
    profile_path = run_dir / "profile.json"
    if profile_path.exists():
        try:
            summary["profile"] = json.loads(profile_path.read_text())
        except Exception:
            pass

    return summary


def run_pipeline(version, timeout):
    """Run the alignment pipeline and capture everything."""
    run_dir = PROJECT_ROOT / "diagnostics" / f"run_v{version}"
    run_dir.mkdir(parents=True, exist_ok=True)

    qa_path = run_dir / "qa.json"
    log_path = run_dir / "run.log"
    stderr_path = run_dir / "stderr.log"
    summary_path = run_dir / "summary.json"
    output_path = run_dir / "output.tif"

    # Snapshot git state
    snapshot_git_state(run_dir / "code_state.txt")

    cmd = [
        sys.executable, str(PROJECT_ROOT / "auto-align.py"),
        TARGET,
        "-r", REFERENCE,
        "--anchors", ANCHORS,
        "-y", "--best",
        "--diagnostics-dir", str(run_dir) + "/",
        "--qa-json", str(qa_path),
        "-o", str(output_path),
    ]

    print(f"=== run_test.py: Starting v{version} ===")
    print(f"Output dir: {run_dir}")
    print(f"Command: {' '.join(cmd[:5])}...")
    print(f"Timeout: {timeout}s")
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

        # Read stdout line by line for live streaming (stderr merged in)
        stderr_data = b""
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

                # Check timeout
                elapsed = time.time() - start
                if elapsed > timeout:
                    print(f"\n=== TIMEOUT after {elapsed:.0f}s, sending SIGTERM ===")
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
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
        stderr_data = str(e).encode()

    wall_clock_s = time.time() - start

    # Write logs
    log_text = "".join(log_lines)
    log_path.write_text(log_text)

    stderr_text = stderr_data.decode("utf-8", errors="replace") if isinstance(stderr_data, bytes) else ""
    if stderr_text.strip():
        stderr_path.write_text(stderr_text)

    print(f"\n=== Pipeline finished: exit_code={exit_code}, wall_clock={wall_clock_s:.1f}s ===\n")

    # Build and write summary
    if exit_code != 0:
        # Include stderr snippets in errors
        pass

    summary = build_summary(version, run_dir, log_text, exit_code, wall_clock_s, qa_path)

    if exit_code != 0 and stderr_text.strip():
        # Add last 500 chars of stderr as error context
        summary["errors"].append(stderr_text.strip()[-500:])

    summary_path.write_text(json.dumps(summary, indent=2))

    # Print summary
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    return summary


def cleanup_old_runs(keep_version):
    """Remove large files from old runs, keeping only the most recent and the best-scoring.

    Preserves summary.json, qa.json, run.log, code_state.txt, and stderr.log
    in all runs (lightweight). Removes output.tif and other large files from
    runs that are neither the most recent nor the best-scoring.
    """
    diag_dir = PROJECT_ROOT / "diagnostics"
    run_dirs = {}
    for d in sorted(diag_dir.glob("run_v*")):
        if d.is_dir():
            m = re.search(r"run_v(\d+)$", str(d))
            if m:
                run_dirs[int(m.group(1))] = d

    if len(run_dirs) <= 2:
        return

    # Find best-scoring run (lowest grid score = best)
    best_version = None
    best_score = float("inf")
    for ver, d in run_dirs.items():
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text())
            grid = summary.get("grid")
            if grid and grid.get("score") is not None:
                score = grid["score"]
                if score < best_score:
                    best_score = score
                    best_version = ver
        except Exception:
            continue

    most_recent = keep_version

    keep = {most_recent}
    if best_version is not None:
        keep.add(best_version)

    # Lightweight files to always preserve
    preserve = {"summary.json", "qa.json", "run.log", "code_state.txt", "stderr.log"}

    freed = 0
    for ver, d in run_dirs.items():
        if ver in keep:
            continue
        for item in d.iterdir():
            if item.name in preserve:
                continue
            size = item.stat().st_size if item.is_file() else 0
            if item.is_file():
                size = item.stat().st_size
                item.unlink()
                freed += size
            elif item.is_dir():
                import shutil
                size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                shutil.rmtree(item)
                freed += size

    if freed > 0:
        kept_str = ", ".join(f"v{v}" for v in sorted(keep))
        print(f"\n=== Cleanup: freed {freed / 1024 / 1024 / 1024:.1f} GB from old runs "
              f"(kept {kept_str}) ===")
        if best_version is not None:
            print(f"    Best: v{best_version} (score={best_score:.1f}), "
                  f"Most recent: v{most_recent}")


def main():
    parser = argparse.ArgumentParser(description="Run Bahrain alignment test")
    parser.add_argument("--version", "-v", type=int, default=None,
                        help="Version number (default: auto-detect next)")
    parser.add_argument("--timeout", "-t", type=int, default=9000,
                        help="Timeout in seconds (default: 9000 = 150 min)")
    args = parser.parse_args()

    version = args.version if args.version is not None else detect_next_version()
    run_pipeline(version, args.timeout)
    cleanup_old_runs(version)


if __name__ == "__main__":
    main()
