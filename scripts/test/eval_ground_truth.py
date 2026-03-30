#!/usr/bin/env python3
"""Compare pipeline output against a manually-georeferenced ground-truth image.

Produces direct pixel-level error measurements (oracle metrics) by computing
dense phase correlation between the pipeline output and the ground truth.

Usage:
    python3 scripts/test/eval_ground_truth.py \
        --pipeline-output diagnostics/run_v133/output.tif \
        --ground-truth /path/to/manually_georeferenced.warped.tif \
        [--gcps data/kh4_1968_gcps.json] \
        [--eval-res 8.0] \
        [--output-json diagnostics/run_v133/gt_eval.json] \
        [--heatmap diagnostics/run_v133/gt_heatmap.jpg]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject, transform_bounds

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from align.image import build_semantic_masks, to_u8
from align.models import get_torch_device
from align.qa import _batch_phase_correlate, _compute_grid_metrics

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_DIR = os.path.join("diagnostics", "gt_cache")


def _cache_key(gt_path: str, eval_res: float, crs_epsg: int) -> str:
    """Deterministic hash for cache invalidation."""
    raw = f"{os.path.abspath(gt_path)}|{eval_res}|{crs_epsg}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_is_valid(cache_dir: str, gt_path: str) -> bool:
    """Check if cached data matches the current ground-truth file."""
    sidecar = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(sidecar):
        return False
    with open(sidecar) as f:
        meta = json.load(f)
    try:
        st = os.stat(gt_path)
        return (meta.get("mtime") == st.st_mtime and
                meta.get("size") == st.st_size)
    except OSError:
        return False


def _write_cache(cache_dir: str, gt_path: str, gt_arr: np.ndarray,
                 gt_transform, bounds: tuple, eval_res: float, crs_epsg: int):
    """Persist ground-truth arrays and metadata."""
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(os.path.join(cache_dir, "gt_data.npz"),
                        gt_arr=gt_arr)
    st = os.stat(gt_path)
    meta = {
        "gt_path": os.path.abspath(gt_path),
        "mtime": st.st_mtime,
        "size": st.st_size,
        "eval_res": eval_res,
        "crs_epsg": crs_epsg,
        "bounds": list(bounds),
        "gt_transform": list(gt_transform)[:6],
        "shape": list(gt_arr.shape),
    }
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _load_cache(cache_dir: str):
    """Load cached ground-truth arrays and metadata."""
    with open(os.path.join(cache_dir, "meta.json")) as f:
        meta = json.load(f)
    data = np.load(os.path.join(cache_dir, "gt_data.npz"))
    gt_arr = data["gt_arr"]
    gt_transform = rasterio.transform.from_bounds(
        *meta["bounds"], meta["shape"][1], meta["shape"][0])
    return gt_arr, gt_transform, meta


# ---------------------------------------------------------------------------
# Image reading
# ---------------------------------------------------------------------------

def _read_and_reproject(src_path: str, bounds: tuple, target_crs, eval_res: float):
    """Read a raster, reproject to target CRS at eval_res within bounds."""
    left, bottom, right, top = bounds
    width = int(round((right - left) / eval_res))
    height = int(round((top - bottom) / eval_res))
    dst_transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)
    dst_array = np.zeros((height, width), dtype=np.float32)

    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )
    return dst_array, dst_transform


def _compute_overlap_bounds(path_a: str, path_b: str, work_crs):
    """Compute overlap bounding box between two rasters in work_crs."""
    with rasterio.open(path_a) as src_a, rasterio.open(path_b) as src_b:
        bounds_a = transform_bounds(src_a.crs, work_crs, *src_a.bounds)
        bounds_b = transform_bounds(src_b.crs, work_crs, *src_b.bounds)

    overlap = (
        max(bounds_a[0], bounds_b[0]),
        max(bounds_a[1], bounds_b[1]),
        min(bounds_a[2], bounds_b[2]),
        min(bounds_a[3], bounds_b[3]),
    )
    if overlap[0] >= overlap[2] or overlap[1] >= overlap[3]:
        return None
    return overlap


def _auto_work_crs(path_a: str, path_b: str):
    """Auto-detect a UTM CRS from the center of the overlap."""
    wgs84 = CRS.from_epsg(4326)
    with rasterio.open(path_a) as src_a, rasterio.open(path_b) as src_b:
        bounds_a = transform_bounds(src_a.crs, wgs84, *src_a.bounds)
        bounds_b = transform_bounds(src_b.crs, wgs84, *src_b.bounds)

    left = max(bounds_a[0], bounds_b[0])
    bottom = max(bounds_a[1], bounds_b[1])
    right = min(bounds_a[2], bounds_b[2])
    top = min(bounds_a[3], bounds_b[3])

    if left >= right or bottom >= top:
        raise ValueError("No overlap between the two images")

    center_lon = (left + right) / 2
    center_lat = (bottom + top) / 2
    zone = int((center_lon + 180) / 6) + 1
    epsg = 32600 + zone if center_lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


# ---------------------------------------------------------------------------
# Multi-scale phase correlation
# ---------------------------------------------------------------------------

def _extract_patches(arr: np.ndarray, valid: np.ndarray,
                     patch_size: int, stride: int, min_valid: float = 0.6):
    """Extract patches with position tracking."""
    h, w = arr.shape
    win = np.outer(np.hanning(patch_size), np.hanning(patch_size)).astype(np.float32)
    patches = []
    positions = []  # (row_center, col_center)

    for r0 in range(0, max(1, h - patch_size + 1), stride):
        r1 = min(r0 + patch_size, h)
        if r1 - r0 < patch_size:
            r0 = max(0, r1 - patch_size)
        for c0 in range(0, max(1, w - patch_size + 1), stride):
            c1 = min(c0 + patch_size, w)
            if c1 - c0 < patch_size:
                c0 = max(0, c1 - patch_size)

            if np.mean(valid[r0:r1, c0:c1]) < min_valid:
                continue

            tile = arr[r0:r1, c0:c1].astype(np.float32)
            if tile.shape == win.shape:
                patches.append(tile * win)
            else:
                local_win = np.outer(
                    np.hanning(tile.shape[0]), np.hanning(tile.shape[1])
                ).astype(np.float32)
                patches.append(tile * local_win)
            positions.append((r0 + patch_size // 2, c0 + patch_size // 2))

    return patches, positions


def compute_displacement_field(gt_arr: np.ndarray, out_arr: np.ndarray,
                               valid: np.ndarray, eval_res: float,
                               scales=(64, 128, 256)):
    """Compute dense displacement field via multi-scale phase correlation.

    Returns dict with per-scale results and a merged displacement map.
    """
    device = get_torch_device()
    results_by_scale = {}

    for patch_size in scales:
        stride = patch_size // 2
        gt_patches, positions = _extract_patches(
            to_u8(gt_arr), valid, patch_size, stride)
        out_patches, _ = _extract_patches(
            to_u8(out_arr), valid, patch_size, stride)

        # Both must have same count (same valid mask)
        n = min(len(gt_patches), len(out_patches))
        if n == 0:
            results_by_scale[patch_size] = {
                "count": 0, "median_m": float("inf"),
                "mean_m": float("inf"), "p90_m": float("inf"),
            }
            continue

        gt_t = torch.from_numpy(np.stack(gt_patches[:n])).to(device)
        out_t = torch.from_numpy(np.stack(out_patches[:n])).to(device)

        with torch.no_grad():
            shifts, responses, valid_mask = _batch_phase_correlate(
                gt_t, out_t, min_resp=0.03)

        mags = np.hypot(shifts[valid_mask, 0], shifts[valid_mask, 1]) * eval_res
        positions_valid = [positions[i] for i in range(n) if valid_mask[i]]

        if len(mags) == 0:
            results_by_scale[patch_size] = {
                "count": 0, "median_m": float("inf"),
                "mean_m": float("inf"), "p90_m": float("inf"),
            }
            continue

        results_by_scale[patch_size] = {
            "count": int(len(mags)),
            "median_m": float(np.median(mags)),
            "mean_m": float(np.mean(mags)),
            "p90_m": float(np.percentile(mags, 90)),
            "max_m": float(np.max(mags)),
            "positions": positions_valid,
            "magnitudes": mags.tolist(),
        }

    return results_by_scale


def compute_grid_displacement(gt_arr: np.ndarray, out_arr: np.ndarray,
                              valid: np.ndarray, eval_res: float,
                              n_rows: int = 4, n_cols: int = 6,
                              patch_size: int = 128):
    """Compute per-grid-cell displacement statistics."""
    device = get_torch_device()
    h, w = gt_arr.shape
    row_h = h / n_rows
    col_w = w / n_cols
    stride = patch_size // 2
    cells = []

    for r in range(n_rows):
        r0 = int(round(r * row_h))
        r1 = int(round((r + 1) * row_h))
        for c in range(n_cols):
            c0 = int(round(c * col_w))
            c1 = int(round((c + 1) * col_w))

            cell_gt = to_u8(gt_arr[r0:r1, c0:c1])
            cell_out = to_u8(out_arr[r0:r1, c0:c1])
            cell_valid = valid[r0:r1, c0:c1]

            gt_patches, _ = _extract_patches(cell_gt, cell_valid, patch_size, stride)
            out_patches, _ = _extract_patches(cell_out, cell_valid, patch_size, stride)

            n = min(len(gt_patches), len(out_patches))
            if n == 0:
                cells.append({
                    "row": r, "col": c, "valid": False,
                    "patch_count": 0, "median_m": None,
                })
                continue

            gt_t = torch.from_numpy(np.stack(gt_patches[:n])).to(device)
            out_t = torch.from_numpy(np.stack(out_patches[:n])).to(device)

            with torch.no_grad():
                shifts, responses, vmask = _batch_phase_correlate(gt_t, out_t, 0.03)

            mags = np.hypot(shifts[vmask, 0], shifts[vmask, 1]) * eval_res

            if len(mags) < 2:
                cells.append({
                    "row": r, "col": c, "valid": False,
                    "patch_count": int(len(mags)), "median_m": None,
                })
                continue

            cells.append({
                "row": r, "col": c, "valid": True,
                "patch_count": int(len(mags)),
                "median_m": float(np.median(mags)),
                "mean_m": float(np.mean(mags)),
                "p90_m": float(np.percentile(mags, 90)),
            })

    valid_cells = [c for c in cells if c["valid"]]
    valid_count = len(valid_cells)

    # Regional breakdown
    left_cols = set(range(n_cols // 3))
    mid_cols = set(range(n_cols // 3, 2 * n_cols // 3))
    right_cols = set(range(2 * n_cols // 3, n_cols))

    def _med(col_set):
        vals = [c["median_m"] for c in valid_cells if c["col"] in col_set]
        return float(np.median(vals)) if vals else None

    # Coastal vs inland breakdown (using semantic masks)
    return {
        "cells": cells,
        "valid_count": valid_count,
        "total_count": len(cells),
        "grid_median_m": float(np.mean([c["median_m"] for c in valid_cells])) if valid_cells else None,
        "west_m": _med(left_cols),
        "center_m": _med(mid_cols),
        "east_m": _med(right_cols),
    }


# ---------------------------------------------------------------------------
# GCP evaluation
# ---------------------------------------------------------------------------

def evaluate_gcps(out_path: str, gcps: list[dict], work_crs) -> dict:
    """Measure displacement at known ground-control points.

    Each GCP should have: {"name": str, "lon": float, "lat": float}
    or {"name": str, "x": float, "y": float, "crs": str}.

    Reads the pipeline output at each GCP location and computes the offset
    between expected and actual positions using phase correlation on a local
    patch around each point.
    """
    if not gcps:
        return {"count": 0}

    # For now, return a placeholder — GCP evaluation requires
    # reading the output at specific coordinates and comparing to GT.
    # This will be implemented when GCP format is finalized.
    return {"count": len(gcps), "note": "GCP evaluation pending format specification"}


# ---------------------------------------------------------------------------
# Heatmap generation
# ---------------------------------------------------------------------------

def generate_heatmap(gt_arr: np.ndarray, out_arr: np.ndarray,
                     valid: np.ndarray, eval_res: float,
                     output_path: str, patch_size: int = 128):
    """Generate a displacement heatmap overlay image."""
    h, w = gt_arr.shape
    stride = patch_size // 2
    device = get_torch_device()

    # Build displacement map
    disp_map = np.full((h, w), np.nan, dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    gt_patches, positions = _extract_patches(
        to_u8(gt_arr), valid, patch_size, stride)
    out_patches, _ = _extract_patches(
        to_u8(out_arr), valid, patch_size, stride)

    n = min(len(gt_patches), len(out_patches))
    if n == 0:
        print("  WARNING: No valid patches for heatmap generation")
        return

    gt_t = torch.from_numpy(np.stack(gt_patches[:n])).to(device)
    out_t = torch.from_numpy(np.stack(out_patches[:n])).to(device)

    with torch.no_grad():
        shifts, responses, vmask = _batch_phase_correlate(gt_t, out_t, 0.03)

    for i in range(n):
        if not vmask[i]:
            continue
        row_c, col_c = positions[i]
        mag = float(np.hypot(shifts[i, 0], shifts[i, 1])) * eval_res
        r0 = max(0, row_c - patch_size // 2)
        r1 = min(h, row_c + patch_size // 2)
        c0 = max(0, col_c - patch_size // 2)
        c1 = min(w, col_c + patch_size // 2)
        # Weighted splat
        existing = disp_map[r0:r1, c0:c1]
        mask = np.isnan(existing)
        existing[mask] = 0.0
        disp_map[r0:r1, c0:c1] = existing + mag
        count_map[r0:r1, c0:c1] += 1.0

    # Average overlapping patches
    has_data = count_map > 0
    disp_map[has_data] /= count_map[has_data]
    disp_map[~has_data] = 0.0

    # Colorize: 0m = blue, 50m = green, 100m+ = red
    max_disp = max(100.0, float(np.percentile(disp_map[has_data], 95))) if np.any(has_data) else 100.0
    norm = np.clip(disp_map / max_disp, 0.0, 1.0)
    heatmap_u8 = (norm * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

    # Blend with grayscale base image
    base = cv2.cvtColor(to_u8(gt_arr), cv2.COLOR_GRAY2BGR)
    alpha = 0.5
    blend = cv2.addWeighted(base, 1 - alpha, heatmap_color, alpha, 0)

    # Add legend
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend, f"Displacement (0-{max_disp:.0f}m)",
                (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(blend, "Blue=0m  Green=mid  Red=high",
                (10, 60), font, 0.6, (255, 255, 255), 1)

    cv2.imwrite(output_path, blend, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  Heatmap saved: {output_path}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_ground_truth(pipeline_output: str, ground_truth: str,
                          eval_res: float = 8.0,
                          gcps_path: str | None = None,
                          cache_dir: str | None = None) -> dict:
    """Run full ground-truth evaluation. Returns oracle metrics dict."""

    t0 = time.time()

    # Determine work CRS
    work_crs = _auto_work_crs(pipeline_output, ground_truth)
    crs_epsg = work_crs.to_epsg()
    print(f"  Work CRS: EPSG:{crs_epsg}")

    # Compute overlap
    overlap = _compute_overlap_bounds(pipeline_output, ground_truth, work_crs)
    if overlap is None:
        return {"error": "No overlap between pipeline output and ground truth"}

    area_km2 = ((overlap[2] - overlap[0]) * (overlap[3] - overlap[1])) / 1e6
    print(f"  Overlap area: {area_km2:.1f} km²")

    # Try cache for GT
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    key = _cache_key(ground_truth, eval_res, crs_epsg)
    gt_cache_dir = os.path.join(cache_dir, key)

    if _cache_is_valid(gt_cache_dir, ground_truth):
        print("  Loading cached GT data...")
        gt_arr, gt_transform, meta = _load_cache(gt_cache_dir)
        # Verify bounds match (overlap may differ if pipeline output changed)
        cached_bounds = tuple(meta["bounds"])
        if cached_bounds != overlap:
            print("  Cache bounds mismatch — re-reading GT")
            gt_arr, gt_transform = _read_and_reproject(
                ground_truth, overlap, work_crs, eval_res)
            _write_cache(gt_cache_dir, ground_truth, gt_arr,
                         gt_transform, overlap, eval_res, crs_epsg)
    else:
        print("  Reading and reprojecting GT...")
        gt_arr, gt_transform = _read_and_reproject(
            ground_truth, overlap, work_crs, eval_res)
        _write_cache(gt_cache_dir, ground_truth, gt_arr,
                     gt_transform, overlap, eval_res, crs_epsg)
        print("  GT cached for future runs")

    # Read pipeline output (always fresh — this is what changes between runs)
    print("  Reading pipeline output...")
    out_arr, out_transform = _read_and_reproject(
        pipeline_output, overlap, work_crs, eval_res)

    # Valid mask: both images have data
    valid = (gt_arr > 0) & (out_arr > 0)
    valid_frac = float(np.mean(valid))
    print(f"  Valid overlap: {valid_frac:.1%}")

    if valid_frac < 0.05:
        return {"error": f"Insufficient overlap: {valid_frac:.1%}"}

    # Multi-scale displacement
    print("  Computing multi-scale displacement...")
    displacement = compute_displacement_field(
        gt_arr, out_arr, valid, eval_res, scales=(64, 128, 256))

    # Grid displacement
    print("  Computing grid displacement...")
    grid = compute_grid_displacement(gt_arr, out_arr, valid, eval_res)

    # Coastal vs inland breakdown using semantic masks
    print("  Computing coastal/inland breakdown...")
    try:
        bundle_gt = build_semantic_masks(gt_arr, mode="coastal_obia")
        coastal_mask = (bundle_gt.shoreline > 0) & valid
        inland_mask = (bundle_gt.stable > 0) & valid & ~coastal_mask

        # Re-run displacement on coastal-only and inland-only subsets
        coastal_result = _displacement_summary(
            gt_arr, out_arr, coastal_mask, eval_res, patch_size=128)
        inland_result = _displacement_summary(
            gt_arr, out_arr, inland_mask, eval_res, patch_size=128)
    except Exception as e:
        print(f"  WARNING: Coastal/inland breakdown failed: {e}")
        coastal_result = {"error": str(e)}
        inland_result = {"error": str(e)}

    # GCP evaluation
    gcps = []
    if gcps_path and os.path.exists(gcps_path):
        with open(gcps_path) as f:
            gcps = json.load(f)
    gcp_result = evaluate_gcps(pipeline_output, gcps, work_crs)

    elapsed = time.time() - t0

    # Primary oracle metric: 128px scale median (best balance of precision and coverage)
    primary_scale = 128
    primary = displacement.get(primary_scale, {})

    result = {
        "oracle_median_m": primary.get("median_m"),
        "oracle_mean_m": primary.get("mean_m"),
        "oracle_p90_m": primary.get("p90_m"),
        "oracle_patch_count": primary.get("count", 0),
        "displacement_by_scale": {
            str(k): {kk: vv for kk, vv in v.items()
                     if kk not in ("positions", "magnitudes")}
            for k, v in displacement.items()
        },
        "grid": grid,
        "coastal": coastal_result,
        "inland": inland_result,
        "gcps": gcp_result,
        "metadata": {
            "pipeline_output": os.path.abspath(pipeline_output),
            "ground_truth": os.path.abspath(ground_truth),
            "work_crs": f"EPSG:{crs_epsg}",
            "eval_res_m": eval_res,
            "overlap_bounds": list(overlap),
            "overlap_area_km2": round(area_km2, 2),
            "valid_fraction": round(valid_frac, 4),
            "elapsed_s": round(elapsed, 1),
        },
    }

    return result


def _displacement_summary(gt_arr, out_arr, mask, eval_res, patch_size=128):
    """Quick displacement summary for a masked region."""
    if np.sum(mask) < patch_size * patch_size:
        return {"count": 0, "note": "insufficient pixels"}

    device = get_torch_device()
    stride = patch_size // 2
    gt_patches, _ = _extract_patches(to_u8(gt_arr), mask, patch_size, stride)
    out_patches, _ = _extract_patches(to_u8(out_arr), mask, patch_size, stride)

    n = min(len(gt_patches), len(out_patches))
    if n == 0:
        return {"count": 0}

    gt_t = torch.from_numpy(np.stack(gt_patches[:n])).to(device)
    out_t = torch.from_numpy(np.stack(out_patches[:n])).to(device)

    with torch.no_grad():
        shifts, responses, vmask = _batch_phase_correlate(gt_t, out_t, 0.03)

    mags = np.hypot(shifts[vmask, 0], shifts[vmask, 1]) * eval_res
    if len(mags) == 0:
        return {"count": 0}

    return {
        "count": int(len(mags)),
        "median_m": float(np.median(mags)),
        "mean_m": float(np.mean(mags)),
        "p90_m": float(np.percentile(mags, 90)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline output against ground-truth georeferenced image")
    parser.add_argument("--pipeline-output", required=True,
                        help="Path to pipeline output TIF")
    parser.add_argument("--ground-truth", required=True,
                        help="Path to manually-georeferenced ground-truth TIF")
    parser.add_argument("--gcps", default=None,
                        help="Path to ground-truth GCPs JSON")
    parser.add_argument("--eval-res", type=float, default=8.0,
                        help="Evaluation resolution in meters (default: 8.0)")
    parser.add_argument("--output-json", default=None,
                        help="Path to write JSON results")
    parser.add_argument("--heatmap", default=None,
                        help="Path to write displacement heatmap JPEG")
    parser.add_argument("--cache-dir", default=None,
                        help="Directory for GT preprocessing cache")
    args = parser.parse_args()

    # Validate inputs
    for path, label in [(args.pipeline_output, "Pipeline output"),
                        (args.ground_truth, "Ground truth")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    print(f"Ground-truth evaluation")
    print(f"  Pipeline: {args.pipeline_output}")
    print(f"  GT:       {args.ground_truth}")
    print(f"  Res:      {args.eval_res}m")

    result = evaluate_ground_truth(
        args.pipeline_output,
        args.ground_truth,
        eval_res=args.eval_res,
        gcps_path=args.gcps,
        cache_dir=args.cache_dir,
    )

    # Print summary
    print()
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        sys.exit(1)

    print(f"  Oracle median:  {result['oracle_median_m']:.1f}m")
    print(f"  Oracle mean:    {result['oracle_mean_m']:.1f}m")
    print(f"  Oracle p90:     {result['oracle_p90_m']:.1f}m")
    print(f"  Patch count:    {result['oracle_patch_count']}")

    if result.get("grid", {}).get("grid_median_m") is not None:
        print(f"  Grid median:    {result['grid']['grid_median_m']:.1f}m")
        for key in ("west_m", "center_m", "east_m"):
            v = result["grid"].get(key)
            if v is not None:
                print(f"    {key}: {v:.1f}m")

    coastal = result.get("coastal", {})
    if coastal.get("count", 0) > 0:
        print(f"  Coastal median: {coastal['median_m']:.1f}m ({coastal['count']} patches)")
    inland = result.get("inland", {})
    if inland.get("count", 0) > 0:
        print(f"  Inland median:  {inland['median_m']:.1f}m ({inland['count']} patches)")

    print(f"  Elapsed:        {result['metadata']['elapsed_s']}s")

    # Write JSON
    output_json = args.output_json
    if output_json is None:
        # Default: next to pipeline output
        base = os.path.dirname(args.pipeline_output)
        output_json = os.path.join(base, "gt_eval.json") if base else "gt_eval.json"

    # Strip non-serializable items
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results: {output_json}")

    # Generate heatmap if requested
    if args.heatmap:
        work_crs = _auto_work_crs(args.pipeline_output, args.ground_truth)
        overlap = _compute_overlap_bounds(
            args.pipeline_output, args.ground_truth, work_crs)
        gt_arr, _ = _read_and_reproject(
            args.ground_truth, overlap, work_crs, args.eval_res)
        out_arr, _ = _read_and_reproject(
            args.pipeline_output, overlap, work_crs, args.eval_res)
        valid = (gt_arr > 0) & (out_arr > 0)
        generate_heatmap(gt_arr, out_arr, valid, args.eval_res, args.heatmap)


if __name__ == "__main__":
    main()
