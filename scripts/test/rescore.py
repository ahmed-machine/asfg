#!/usr/bin/env python3
"""Re-run QA score formula on existing pipeline outputs without re-running alignment.

Primary tool for fast iteration on scoring changes: edit align/qa.py, then
run this to see how all historical runs would score under the new formula.

Usage:
    # Single run
    python3 scripts/test/rescore.py --run diagnostics/run_v133

    # Batch across all runs with output.tif
    python3 scripts/test/rescore.py --batch

    # Compare current vs custom weights
    python3 scripts/test/rescore.py --run diagnostics/run_v133 \
        --weights '{"grid": 0.55, "patch": 0.25, "stable": 18, "shore": 12}'

    # Include ground-truth evaluation
    python3 scripts/test/rescore.py --run diagnostics/run_v133 \
        --ground-truth /path/to/gt.warped.tif
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from align.geo import read_overlap_region, compute_overlap_or_none, clear_overlap_cache
from align.qa import evaluate_alignment_quality_arrays
from align.image import to_u8
from rasterio.crs import CRS


def _find_runs_with_output(diag_dir: str) -> list[str]:
    """Find all run directories that have output.tif."""
    runs = []
    for d in sorted(glob(os.path.join(diag_dir, "run_v*"))):
        if os.path.isfile(os.path.join(d, "output.tif")):
            runs.append(d)
    return runs


def _load_run_metadata(run_dir: str) -> dict | None:
    """Load qa.json to get reference path, overlap, and CRS."""
    qa_path = os.path.join(run_dir, "qa.json")
    if not os.path.exists(qa_path):
        return None
    with open(qa_path) as f:
        return json.load(f)


def _extract_overlap_and_crs(qa_data: dict, output_path: str):
    """Extract overlap bounds and work CRS from qa.json metadata."""
    meta = qa_data.get("metadata", {})
    reference_path = meta.get("reference_path")
    if not reference_path or not os.path.exists(reference_path):
        return None, None, None

    # Get work CRS from hypothesis
    hypotheses = meta.get("global_hypotheses", [])
    work_crs_str = None
    for h in hypotheses:
        if "work_crs" in h:
            work_crs_str = h["work_crs"]
            break

    if not work_crs_str:
        return reference_path, None, None

    work_crs = CRS.from_string(work_crs_str)

    # Compute overlap from the actual files
    try:
        with rasterio.open(output_path) as src_out, \
             rasterio.open(reference_path) as src_ref:
            overlap = compute_overlap_or_none(src_out, src_ref, work_crs)
    except Exception as e:
        print(f"  WARNING: Could not compute overlap: {e}")
        return reference_path, work_crs, None

    return reference_path, work_crs, overlap


def rescore_run(run_dir: str, eval_res: float = 4.0,
                weights: dict | None = None,
                ground_truth: str | None = None) -> dict:
    """Re-score a single pipeline run.

    Returns dict with original score, new score, and component breakdown.
    """
    output_path = os.path.join(run_dir, "output.tif")
    if not os.path.exists(output_path):
        return {"error": "output.tif not found", "run": run_dir}

    qa_data = _load_run_metadata(run_dir)
    if not qa_data:
        return {"error": "qa.json not found", "run": run_dir}

    reference_path, work_crs, overlap = _extract_overlap_and_crs(qa_data, output_path)
    if not reference_path:
        return {"error": "reference path not found or missing", "run": run_dir}
    if overlap is None:
        return {"error": "could not compute overlap", "run": run_dir}

    # Read original score from qa.json
    original = {}
    reports = qa_data.get("reports", [])
    selected = qa_data.get("selected_candidate", "grid")
    for report in reports:
        if report.get("candidate") == selected:
            original = report.get("image_metrics", {})
            break

    # Re-compute score from the actual TIFs
    clear_overlap_cache()
    t0 = time.time()

    try:
        with rasterio.open(output_path) as src_out, \
             rasterio.open(reference_path) as src_ref:
            arr_ref, _ = read_overlap_region(src_ref, overlap, work_crs, eval_res)
            arr_out, _ = read_overlap_region(src_out, overlap, work_crs, eval_res)
    except Exception as e:
        return {"error": f"Failed to read images: {e}", "run": run_dir}

    valid = (arr_ref > 0) & (arr_out > 0)
    if np.mean(valid) < 0.05:
        return {"error": "Insufficient overlap", "run": run_dir}

    new_metrics = evaluate_alignment_quality_arrays(
        arr_ref, arr_out, valid, eval_res=eval_res)

    elapsed = time.time() - t0

    if new_metrics is None:
        return {"error": "QA returned None (no valid cells or patches)", "run": run_dir}

    # If custom weights provided, recompute score with those weights
    custom_score = None
    if weights:
        grid_w = weights.get("grid", 0.55)
        patch_w = weights.get("patch", 0.25)
        stable_w = weights.get("stable", 18.0)
        shore_w = weights.get("shore", 12.0)

        grid_score = new_metrics.get("grid_score", 0.0)
        patch_med = new_metrics.get("patch_med", 0.0)
        stable_iou = new_metrics.get("stable_iou", 0.0)
        shore_iou = new_metrics.get("shore_iou", 0.0)

        valid_count = new_metrics.get("grid", {}).get("valid_count", 0)
        grid_contrib = grid_w * grid_score if valid_count > 0 else 0.0
        patch_contrib = patch_w * patch_med

        # Use union pixel counts from the breakdown to determine if penalty applies
        # (approximate: if penalty was non-zero in current formula, apply it)
        stable_penalty = stable_w * (1.0 - stable_iou)
        shore_penalty = shore_w * (1.0 - shore_iou)

        custom_score = {
            "score": grid_contrib + patch_contrib + stable_penalty + shore_penalty,
            "grid_contrib": round(grid_contrib, 2),
            "patch_contrib": round(patch_contrib, 2),
            "stable_penalty": round(stable_penalty, 2),
            "shore_penalty": round(shore_penalty, 2),
            "weights": weights,
        }

    # Ground truth evaluation
    gt_result = None
    if ground_truth and os.path.exists(ground_truth):
        from scripts.test.eval_ground_truth import evaluate_ground_truth
        gt_result = evaluate_ground_truth(
            output_path, ground_truth, eval_res=eval_res)

    run_name = os.path.basename(run_dir)
    return {
        "run": run_name,
        "original_score": original.get("score"),
        "new_score": new_metrics.get("score"),
        "delta": (new_metrics.get("score", 0) - original.get("score", 0))
                 if original.get("score") is not None else None,
        "new_metrics": {
            "score": new_metrics.get("score"),
            "grid_score": new_metrics.get("grid_score"),
            "patch_med": new_metrics.get("patch_med"),
            "patch_p90": new_metrics.get("patch_p90"),
            "stable_iou": new_metrics.get("stable_iou"),
            "shore_iou": new_metrics.get("shore_iou"),
            "grid_valid": new_metrics.get("grid", {}).get("valid_count"),
            "grid_total": new_metrics.get("grid", {}).get("total_count"),
            "score_breakdown": new_metrics.get("score_breakdown"),
        },
        "custom_score": custom_score,
        "ground_truth": gt_result,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Re-score pipeline outputs with current or custom QA formula")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", help="Path to a single run directory")
    group.add_argument("--batch", action="store_true",
                       help="Score all runs with output.tif in diagnostics/")
    parser.add_argument("--eval-res", type=float, default=4.0,
                        help="Evaluation resolution in meters (default: 4.0)")
    parser.add_argument("--weights", type=str, default=None,
                        help='Custom weights as JSON: \'{"grid": 0.55, "patch": 0.25, "stable": 18, "shore": 12}\'')
    parser.add_argument("--ground-truth", default=None,
                        help="Path to ground-truth TIF for oracle comparison")
    parser.add_argument("--output-json", default=None,
                        help="Path to write results JSON")
    args = parser.parse_args()

    weights = None
    if args.weights:
        weights = json.loads(args.weights)

    if args.batch:
        diag_dir = str(PROJECT_ROOT / "diagnostics")
        runs = _find_runs_with_output(diag_dir)
        if not runs:
            print("No runs with output.tif found in diagnostics/")
            sys.exit(1)
        print(f"Found {len(runs)} runs with output.tif")
    else:
        if not os.path.isdir(args.run):
            print(f"ERROR: Run directory not found: {args.run}")
            sys.exit(1)
        runs = [args.run]

    results = []
    for run_dir in runs:
        run_name = os.path.basename(run_dir)
        print(f"\n{'='*60}")
        print(f"  Rescoring {run_name}...")
        result = rescore_run(
            run_dir,
            eval_res=args.eval_res,
            weights=weights,
            ground_truth=args.ground_truth,
        )
        results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        # Print summary line
        orig = result.get("original_score")
        new = result.get("new_score")
        delta = result.get("delta")
        orig_str = f"{orig:.1f}" if orig is not None else "N/A"
        new_str = f"{new:.1f}" if new is not None else "N/A"
        delta_str = f"{delta:+.1f}" if delta is not None else ""
        print(f"  Original: {orig_str}  ->  New: {new_str}  ({delta_str})")

        bd = result.get("new_metrics", {}).get("score_breakdown", {})
        if bd:
            # Support both old (stable_iou_penalty) and new (stable_boundary_penalty) keys
            stable_p = bd.get('stable_boundary_penalty', bd.get('stable_iou_penalty', '?'))
            shore_p = bd.get('shore_boundary_penalty', bd.get('shore_iou_penalty', '?'))
            print(f"    grid={bd.get('grid_contrib', '?')}"
                  f"  patch={bd.get('patch_contrib', '?')}"
                  f"  stable={stable_p}"
                  f"  shore={shore_p}")

        if result.get("custom_score"):
            cs = result["custom_score"]
            print(f"  Custom weights score: {cs['score']:.1f}")

        gt = result.get("ground_truth")
        if gt and "oracle_median_m" in gt:
            print(f"  GT oracle median: {gt['oracle_median_m']:.1f}m")

    # Batch summary table
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("Summary:")
        print(f"  {'Run':<15} {'Original':>10} {'New':>10} {'Delta':>10} {'Elapsed':>8}")
        print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
        for r in results:
            if "error" in r:
                print(f"  {r['run']:<15} ERROR: {r['error']}")
                continue
            orig = r.get("original_score")
            new = r.get("new_score")
            delta = r.get("delta")
            orig_s = f"{orig:>10.1f}" if orig is not None else f"{'N/A':>10}"
            new_s = f"{new:>10.1f}" if new is not None else f"{'N/A':>10}"
            delta_s = f"{delta:>+10.1f}" if delta is not None else f"{'':>10}"
            elapsed_s = f"{r.get('elapsed_s', 0):>7.1f}s"
            print(f"  {r['run']:<15}{orig_s}{new_s}{delta_s} {elapsed_s}")

    # Write JSON
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results if len(results) > 1 else results[0], f, indent=2)
        print(f"\nResults written to {args.output_json}")


if __name__ == "__main__":
    main()
