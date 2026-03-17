#!/usr/bin/env python3
"""
Benchmark test harness for end-to-end pipeline validation.

Runs the alignment pipeline against ground-truth reference images (which ARE
the alignment targets), so scores should converge toward near-perfect overlap.

Two test cases:
  1. KH-9 DZB1212 (1976): half-frames → stitch → réseau → georef → align
  2. KH-4 DS1022 007+008 (1965): USGS download → extract → stitch → georef → align

Usage:
    python3 scripts/run_benchmark.py [--version N] [--timeout 3600]
        [--rebuild] [--case kh9_dzb1212|kh4_ds1022] [--compare-only]
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_test import parse_log, build_summary, snapshot_git_state

PYTHON = "/opt/homebrew/bin/python3"
BENCHMARK_DIR = PROJECT_ROOT / "output" / "benchmark"
CORNERS_CACHE = PROJECT_ROOT / "data" / "benchmark_corners.json"

# "Near perfect" targets for self-alignment
NEAR_PERFECT = {
    "score": 20,
    "patch_med": 5,
    "stable_iou": 0.90,
    "shore_iou": 0.80,
    "west": 10,
    "center": 10,
    "east": 10,
    "north": 5,
}

# ─── Case definitions ────────────────────────────────────────────────────────

CASES = {
    "kh9_dzb1212": {
        "name": "KH-9 DZB1212 (1976)",
        "entity_id": "DZB1212-500236L002001",
        "dataset": "declassii",
        "raw_frames": [
            PROJECT_ROOT / "data" / "raw" / "extracted" / "DZB1212-500236L002001_a.tif",
            PROJECT_ROOT / "data" / "raw" / "extracted" / "DZB1212-500236L002001_b.tif",
        ],
        "reference": "/Users/mish/Code/openmaps/public/maps/1976-KH9-DZB1212.warped.tif",
        "anchors": str(PROJECT_ROOT / "data" / "bahrain_anchor_gcps.json"),
        "needs_download": False,
        "needs_reseau": True,
        "force_global": False,  # SIFT pre-alignment positions georef accurately
        "steps": ["stitch", "reseau", "orient", "georef", "align"],
    },
    "kh4_ds1022": {
        "name": "KH-4 DS1022-1024DA007 (1965)",
        "entity_ids": ["DS1022-1024DA007"],
        "dataset": "corona2",
        "catalog_csv": PROJECT_ROOT / "data" / "available" / "corona2_69b17d89ee62ff28.csv",
        "reference": "/Users/mish/Code/openmaps/public/maps/1965-DS1022-1024DA.warped.tif",
        "needs_download": True,
        "needs_reseau": False,
        "force_global": False,
        "steps": ["download", "extract", "stitch_subframes",
                  "orient", "georef", "align"],
    },
}


# ─── Corner caching ──────────────────────────────────────────────────────────

def load_corners_cache() -> dict:
    if CORNERS_CACHE.exists():
        return json.loads(CORNERS_CACHE.read_text())
    return {}


def save_corners_cache(cache: dict):
    CORNERS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CORNERS_CACHE.write_text(json.dumps(cache, indent=2))


def fetch_kh9_corners(entity_id: str, dataset: str, cache: dict) -> dict:
    """Fetch corners for a single entity from USGS API, with caching."""
    if entity_id in cache:
        print(f"  Corners for {entity_id}: cached")
        return cache[entity_id]

    from declass.usgs import fetch_corners_batch
    results = fetch_corners_batch(dataset, [entity_id])
    corners = results[entity_id]
    cache[entity_id] = {
        "NW": list(corners["NW"]), "NE": list(corners["NE"]),
        "SE": list(corners["SE"]), "SW": list(corners["SW"]),
        "source": "usgs_api",
        "fetched": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    save_corners_cache(cache)
    return cache[entity_id]


def parse_csv_corners(csv_path: str, entity_ids: list, cache: dict) -> dict:
    """Parse corners from catalog CSV for given entity IDs, with caching."""
    missing = [eid for eid in entity_ids if eid not in cache]
    if not missing:
        print(f"  Corners for {', '.join(entity_ids)}: cached")
        return {eid: cache[eid] for eid in entity_ids}

    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = row.get("Entity ID", "").strip()
            if eid in missing:
                corners = {
                    "NW": [float(row["NW Cormer Lat dec"]), float(row["NW Corner Long dec"])],
                    "NE": [float(row["NE Corner Lat dec"]), float(row["NE Corner Long dec"])],
                    "SE": [float(row["SE Corner Lat dec"]), float(row["SE Corner Long dec"])],
                    "SW": [float(row["SW Corner Lat dec"]), float(row["SW Corner Long dec"])],
                    "source": "catalog_csv",
                    "fetched": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                }
                cache[eid] = corners
                print(f"  Corners for {eid}: parsed from CSV")

    save_corners_cache(cache)
    return {eid: cache[eid] for eid in entity_ids}


def compute_combined_corners(corners_007: dict, corners_008: dict, cache: dict) -> dict:
    """Compute combined corners for stitched 007+008 frames.

    Takes NW/SW from the westernmost frame's west side, NE/SE from the
    easternmost frame's east side.
    """
    key = "DS1022-1024DA_combined"
    if key in cache:
        print(f"  Combined corners: cached")
        return cache[key]

    # Determine which frame is westernmost by comparing NW longitudes
    nw_007_lon = corners_007["NW"][1]
    nw_008_lon = corners_008["NW"][1]

    if nw_007_lon <= nw_008_lon:
        # 007 is west, 008 is east
        west_corners = corners_007
        east_corners = corners_008
    else:
        west_corners = corners_008
        east_corners = corners_007

    combined = {
        "NW": west_corners["NW"],
        "SW": west_corners["SW"],
        "NE": east_corners["NE"],
        "SE": east_corners["SE"],
        "source": "computed",
        "fetched": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    cache[key] = combined
    save_corners_cache(cache)
    print(f"  Combined corners: NW={combined['NW']}, SE={combined['SE']}")
    return combined


def corners_to_dict(cached_corners: dict) -> dict:
    """Convert cached corners (with metadata) to simple {NW: (lat, lon)} format."""
    return {
        k: tuple(v) for k, v in cached_corners.items()
        if k in ("NW", "NE", "SE", "SW")
    }


# ─── SIFT position refinement ─────────────────────────────────────────────────

def _sift_position_refinement(georef_path: str, reference_path: str,
                               target_res: float = 50.0) -> tuple:
    """Use SIFT+RANSAC to find residual (dx, dy) translation between georef and reference.

    Works at ~50m/px. Returns (shift_x, shift_y) in CRS units (metres),
    or (0, 0) if SIFT matching fails.
    """
    import cv2
    import numpy as np
    import rasterio
    from rasterio.warp import reproject, Resampling

    with rasterio.open(georef_path) as tgt_ds:
        tgt_crs = tgt_ds.crs
        tgt_bounds = tgt_ds.bounds

    with rasterio.open(reference_path) as ref_ds:
        ref_crs = ref_ds.crs
        ref_bounds = ref_ds.bounds

    # Compute overlap in target CRS
    if str(ref_crs) != str(tgt_crs):
        from rasterio.warp import transform_bounds
        rb = transform_bounds(ref_crs, tgt_crs,
                              ref_bounds.left, ref_bounds.bottom,
                              ref_bounds.right, ref_bounds.top)
    else:
        rb = (ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top)

    ovl_left = max(tgt_bounds.left, rb[0])
    ovl_bottom = max(tgt_bounds.bottom, rb[1])
    ovl_right = min(tgt_bounds.right, rb[2])
    ovl_top = min(tgt_bounds.top, rb[3])

    if ovl_left >= ovl_right or ovl_bottom >= ovl_top:
        print("    SIFT: no overlap between georef and reference")
        return 0.0, 0.0

    # Read both images at target_res into overlapping extent
    width = int((ovl_right - ovl_left) / target_res)
    height = int((ovl_top - ovl_bottom) / target_res)
    if width < 50 or height < 50:
        print(f"    SIFT: overlap too small ({width}x{height} px at {target_res}m)")
        return 0.0, 0.0

    dst_transform = rasterio.transform.from_bounds(
        ovl_left, ovl_bottom, ovl_right, ovl_top, width, height)

    tgt_arr = np.zeros((height, width), dtype=np.float32)
    ref_arr = np.zeros((height, width), dtype=np.float32)

    with rasterio.open(georef_path) as tgt_ds:
        reproject(
            rasterio.band(tgt_ds, 1), tgt_arr,
            src_transform=tgt_ds.transform, src_crs=tgt_ds.crs,
            dst_transform=dst_transform, dst_crs=tgt_crs,
            resampling=Resampling.bilinear,
        )

    with rasterio.open(reference_path) as ref_ds:
        reproject(
            rasterio.band(ref_ds, 1), ref_arr,
            src_transform=ref_ds.transform, src_crs=ref_ds.crs,
            dst_transform=dst_transform, dst_crs=tgt_crs,
            resampling=Resampling.bilinear,
        )

    # Normalize to uint8
    def to_u8(arr):
        valid = arr[arr > 0]
        if len(valid) == 0:
            return np.zeros_like(arr, dtype=np.uint8)
        lo, hi = np.percentile(valid, [2, 98])
        if hi <= lo:
            hi = lo + 1
        clipped = np.clip((arr - lo) / (hi - lo) * 255, 0, 255)
        return clipped.astype(np.uint8)

    tgt_u8 = to_u8(tgt_arr)
    ref_u8 = to_u8(ref_arr)

    # CLAHE + SIFT
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    tgt_eq = clahe.apply(tgt_u8)
    ref_eq = clahe.apply(ref_u8)

    # Mask out nodata regions
    tgt_mask = (tgt_arr > 0).astype(np.uint8) * 255
    ref_mask = (ref_arr > 0).astype(np.uint8) * 255

    sift = cv2.SIFT_create(nfeatures=5000)
    kp_tgt, desc_tgt = sift.detectAndCompute(tgt_eq, tgt_mask)
    kp_ref, desc_ref = sift.detectAndCompute(ref_eq, ref_mask)

    if desc_tgt is None or desc_ref is None or len(kp_tgt) < 10 or len(kp_ref) < 10:
        print(f"    SIFT: too few keypoints (tgt={len(kp_tgt) if kp_tgt else 0}, "
              f"ref={len(kp_ref) if kp_ref else 0})")
        return 0.0, 0.0

    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
        dict(checks=50),
    )

    try:
        raw = flann.knnMatch(desc_tgt, desc_ref, k=2)
    except cv2.error:
        print("    SIFT: FLANN matching failed")
        return 0.0, 0.0

    good = []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.7 * n.distance:
                good.append(m)

    if len(good) < 8:
        print(f"    SIFT: too few good matches ({len(good)})")
        return 0.0, 0.0

    pts_tgt = np.float32([kp_tgt[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, inliers = cv2.estimateAffinePartial2D(
        pts_tgt, pts_ref, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    if M is None or inliers is None:
        print("    SIFT: RANSAC failed")
        return 0.0, 0.0

    n_inliers = int(inliers.sum())
    if n_inliers < 6:
        print(f"    SIFT: too few RANSAC inliers ({n_inliers})")
        return 0.0, 0.0

    # Extract translation at image CENTER (not origin) to account for scale.
    # M maps tgt pixel coords to ref pixel coords:
    #   ref_pt = M @ [tgt_x, tgt_y, 1]^T
    # Translation at origin: (tx, ty) = M[:, 2]
    # Translation at center: map center of overlap, measure displacement.
    cx, cy = width / 2.0, height / 2.0
    ref_cx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
    ref_cy = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
    tx_px = ref_cx - cx
    ty_px = ref_cy - cy
    scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)

    # Convert pixel shift to CRS units
    shift_x = tx_px * target_res
    shift_y = -ty_px * target_res  # image y is inverted

    # Sanity checks
    max_shift = min(ovl_right - ovl_left, ovl_top - ovl_bottom) / 2
    shift_dist = np.sqrt(shift_x**2 + shift_y**2)
    if shift_dist > max_shift:
        print(f"    SIFT: shift {shift_dist:.0f}m exceeds half-overlap ({max_shift:.0f}m) — rejected")
        return 0.0, 0.0

    if abs(scale - 1.0) > 0.15:
        print(f"    SIFT: scale {scale:.3f} deviates >15% from 1.0 — rejected")
        return 0.0, 0.0

    print(f"    SIFT: {n_inliers} inliers, shift=({shift_x:+.0f}, {shift_y:+.0f})m, "
          f"scale={scale:.3f}, dist={shift_dist:.0f}m")

    return shift_x, shift_y


# ─── Georef cropping ─────────────────────────────────────────────────────────

def crop_georef_to_reference(georef_path: Path, reference_path: str,
                              cropped_path: Path, margin_m: float = 5000,
                              use_sift: bool = True) -> Path:
    """Crop georef to the reference extent and center it on the reference.

    Two problems this solves:
    1. When the target strip is much larger than the reference, coarse offset
       detection fails (nearly-uniform land mask template).
    2. USGS corner-based georef can be off by several km. We compute the
       approximate offset from the overlap centroids and shift the target
       so the pipeline starts from a small residual offset.

    When use_sift=False, skip SIFT refinement (e.g. when the pipeline will
    run its own global search which handles positioning).
    """
    if cropped_path.exists():
        print(f"  [skip] Cropped georef: {cropped_path}")
        return cropped_path

    import rasterio
    from rasterio.warp import transform_bounds

    # Get reference bounds in target CRS
    with rasterio.open(reference_path) as ref_ds:
        ref_crs = ref_ds.crs
        ref_bounds = ref_ds.bounds

    with rasterio.open(str(georef_path)) as tgt_ds:
        tgt_crs = tgt_ds.crs
        tgt_bounds = tgt_ds.bounds

    # Transform reference bounds to target CRS
    if str(ref_crs) != str(tgt_crs):
        tb = transform_bounds(ref_crs, tgt_crs,
                              ref_bounds.left, ref_bounds.bottom,
                              ref_bounds.right, ref_bounds.top)
    else:
        tb = (ref_bounds.left, ref_bounds.bottom,
              ref_bounds.right, ref_bounds.top)

    # Compute overlap between target and reference (in target CRS)
    ovl_left = max(tgt_bounds.left, tb[0])
    ovl_bottom = max(tgt_bounds.bottom, tb[1])
    ovl_right = min(tgt_bounds.right, tb[2])
    ovl_top = min(tgt_bounds.top, tb[3])

    if ovl_left >= ovl_right or ovl_bottom >= ovl_top:
        raise RuntimeError("No overlap between georef and reference")

    # Compute approximate translation: shift target so its overlap center
    # matches the reference center. This compensates for USGS corner inaccuracy.
    ref_cx = (tb[0] + tb[2]) / 2
    ref_cy = (tb[1] + tb[3]) / 2
    tgt_ovl_cx = (ovl_left + ovl_right) / 2
    tgt_ovl_cy = (ovl_bottom + ovl_top) / 2
    shift_x = ref_cx - tgt_ovl_cx
    shift_y = ref_cy - tgt_ovl_cy

    # Apply shift to crop window (centered on reference + margin)
    west = tb[0] - margin_m - shift_x
    south = tb[1] - margin_m - shift_y
    east = tb[2] + margin_m - shift_x
    north = tb[3] + margin_m - shift_y

    # First crop, then shift the output bounds to align with reference
    # Step 1: crop to the computed window
    tmp_crop = str(cropped_path) + ".tmp.tif"
    cmd_crop = [
        "gdalwarp",
        "-te", str(west), str(south), str(east), str(north),
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=2",
        "-co", "TILED=YES",
        "-co", "BIGTIFF=IF_SAFER",
        str(georef_path),
        tmp_crop,
    ]
    result = subprocess.run(cmd_crop, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Crop failed: {result.stderr}")

    # Step 2: shift the cropped image bounds by the computed offset
    # This translates the image so its content aligns with the reference
    with rasterio.open(tmp_crop) as ds:
        old_bounds = ds.bounds

    new_left = old_bounds.left + shift_x
    new_top = old_bounds.top + shift_y
    new_right = old_bounds.right + shift_x
    new_bottom = old_bounds.bottom + shift_y

    cmd_shift = [
        "gdal_translate",
        "-a_ullr", str(new_left), str(new_top), str(new_right), str(new_bottom),
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=2",
        "-co", "TILED=YES",
        "-co", "BIGTIFF=IF_SAFER",
        tmp_crop,
        str(cropped_path),
    ]
    result = subprocess.run(cmd_shift, capture_output=True, text=True)
    os.remove(tmp_crop)
    if result.returncode != 0:
        raise RuntimeError(f"Shift failed: {result.stderr}")

    shift_dist = (shift_x**2 + shift_y**2) ** 0.5
    print(f"  Cropped georef to reference extent (+{margin_m/1000:.0f}km margin, "
          f"pre-shifted {shift_dist:.0f}m to center on reference)")

    # Step 3: SIFT-based position refinement
    # The centroid shift above is approximate. SIFT matches actual image features
    # for a more precise alignment, especially when reference is derived from the
    # same source (self-alignment benchmark).
    # Skip when force_global=True — the pipeline's global search handles positioning
    # and SIFT would double-shift the image.
    if not use_sift:
        print("  SIFT: skipped (pipeline will run global search)")
        return cropped_path

    print("  SIFT position refinement...")
    sift_dx, sift_dy = _sift_position_refinement(
        str(cropped_path), reference_path)

    if abs(sift_dx) > 1 or abs(sift_dy) > 1:
        # Apply SIFT shift via gdal_translate -a_ullr
        with rasterio.open(str(cropped_path)) as ds:
            cur_bounds = ds.bounds

        sift_left = cur_bounds.left + sift_dx
        sift_top = cur_bounds.top + sift_dy
        sift_right = cur_bounds.right + sift_dx
        sift_bottom = cur_bounds.bottom + sift_dy

        tmp_sift = str(cropped_path) + ".sift.tif"
        cmd_sift = [
            "gdal_translate",
            "-a_ullr", str(sift_left), str(sift_top),
                      str(sift_right), str(sift_bottom),
            "-co", "COMPRESS=LZW",
            "-co", "PREDICTOR=2",
            "-co", "TILED=YES",
            "-co", "BIGTIFF=IF_SAFER",
            str(cropped_path),
            tmp_sift,
        ]
        result = subprocess.run(cmd_sift, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  WARNING: SIFT shift failed: {result.stderr}")
        else:
            os.replace(tmp_sift, str(cropped_path))
            sift_dist = (sift_dx**2 + sift_dy**2) ** 0.5
            print(f"  Applied SIFT refinement: ({sift_dx:+.0f}, {sift_dy:+.0f})m "
                  f"({sift_dist:.0f}m total)")
    else:
        print("  SIFT: no significant shift detected")

    return cropped_path


# ─── USGS credentials check ─────────────────────────────────────────────────

def check_usgs_credentials():
    """Verify USGS credentials exist before attempting downloads."""
    creds_path = os.path.expanduser("~/.usgs/credentials.json")
    if not os.path.exists(creds_path):
        print(f"\nERROR: USGS credentials not found at {creds_path}")
        print(f"Required for downloading KH-4 imagery.")
        print(f"Create the file with: {{\"username\": \"...\", \"token\": \"...\"}}")
        print(f"Register at https://ers.cr.usgs.gov/register")
        sys.exit(1)


# ─── Intermediate preparation ────────────────────────────────────────────────

def prepare_kh9(case: dict, case_dir: Path, rebuild: bool) -> Path:
    """Prepare KH-9 DZB1212: stitch → réseau → orient → georef."""
    import shutil
    inter_dir = case_dir / "intermediates"

    if rebuild and inter_dir.exists():
        print(f"  Rebuilding intermediates (deleting {inter_dir})")
        shutil.rmtree(inter_dir)

    inter_dir.mkdir(parents=True, exist_ok=True)
    georef_path = inter_dir / "georef.tif"
    cropped_path = inter_dir / "georef_cropped.tif"

    # Return cropped version if it already exists
    if cropped_path.exists() and not rebuild:
        print(f"  [skip] Cropped georef already exists: {cropped_path}")
        return cropped_path

    # Check raw frames exist
    for f in case["raw_frames"]:
        if not Path(f).exists():
            print(f"ERROR: Raw frame not found: {f}")
            sys.exit(1)

    frames = [str(f) for f in case["raw_frames"]]

    # Step 1: Stitch half-frames
    stitched_path = inter_dir / "stitched.tif"
    if not stitched_path.exists():
        print(f"\n  --- Stitch: 2 half-frames ---")
        from declass.stitch import stitch_frames
        stitch_frames(frames, str(stitched_path), str(inter_dir),
                      preserve_order=True)
    else:
        print(f"  [skip] Stitched: {stitched_path}")

    # Step 2: Réseau correction
    flattened_path = inter_dir / "flattened.tif"
    if not flattened_path.exists():
        print(f"\n  --- Réseau correction ---")
        from declass.reseau import process_kh9_reseau
        process_kh9_reseau(str(stitched_path), str(flattened_path), joined=True)
    else:
        print(f"  [skip] Flattened: {flattened_path}")

    # Step 3: Fetch corners
    cache = load_corners_cache()
    entity_id = case["entity_id"]
    cached = fetch_kh9_corners(entity_id, case["dataset"], cache)
    corners = corners_to_dict(cached)

    # Step 4: Orientation detection
    # DZB prefix maps to KH-7 in camera.py, but this is actually KH-9 mapping camera.
    # Import KH9 config directly to avoid identify_camera() returning wrong type.
    from declass.camera import KH9
    from declass.orientation import detect_orientation

    # Use the STITCHED image (not flattened) for orientation + georef.
    # USGS corners describe the raw frame geometry. Réseau correction changes
    # the image dimensions (68603×35556 → 67498×33031, losing ~4km×9km),
    # so corners applied to the flattened image produce a systematic offset.
    # The alignment pipeline handles film distortion via feature matching.
    georef_input = stitched_path

    print(f"\n  --- Orientation detection ---")
    # Don't use detect_orientation() — the metadata heuristic can't distinguish
    # 90° from 270° (both produce square pixels), and the reference only covers
    # ~19% of the strip. Instead, try both candidate rotations and pick the one
    # that produces better overlap with the reference via a quick SIFT check.
    from declass.orientation import (
        rotate_corners_cw90, rotate_corners_ccw90,
        detect_orientation_against_reference,
    )

    # Both 90° and 270° produce near-square pixels for this image.
    # Test which one aligns better with the reference.
    candidates = [
        (90, rotate_corners_ccw90(corners)),
        (270, rotate_corners_cw90(corners)),
    ]

    best_rot, best_corners, best_conf = 0, corners, 0
    for rot_deg, rot_corners in candidates:
        _, _, conf = detect_orientation_against_reference(
            str(georef_input), rot_corners, case["reference"])
        print(f"    Rotation {rot_deg}°: confidence={conf}")
        if conf > best_conf:
            best_rot = rot_deg
            best_corners = rot_corners
            best_conf = conf

    rotation = best_rot
    gcp_corners = best_corners
    print(f"  Selected orientation: {rotation}° (confidence={best_conf})")

    # Step 5: Georef from stitched image
    if georef_path.exists():
        georef_path.unlink()

    print(f"\n  --- Georef ---")
    from declass.georef import georef_with_corners
    georef_with_corners(str(georef_input), str(georef_path), gcp_corners)

    # Step 6: Crop georef to reference extent
    # The full DZB1212 strip is ~250km but the reference only covers Bahrain
    # (~95km). Without cropping, coarse offset detection fails because the
    # land mask template is nearly uniform in the large strip.
    cropped_path = inter_dir / "georef_cropped.tif"
    print(f"\n  --- Crop to reference extent ---")
    crop_georef_to_reference(georef_path, case["reference"], cropped_path,
                              use_sift=not case.get("force_global", False))

    return cropped_path


def prepare_kh4(case: dict, case_dir: Path, rebuild: bool) -> Path:
    """Prepare KH-4 DS1022: download → extract → stitch → orient → georef."""
    import shutil

    check_usgs_credentials()

    inter_dir = case_dir / "intermediates"
    downloads_dir = case_dir / "downloads"
    extracted_dir = case_dir / "extracted"

    if rebuild and inter_dir.exists():
        print(f"  Rebuilding intermediates (deleting {inter_dir})")
        shutil.rmtree(inter_dir)

    inter_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    georef_path = inter_dir / "georef.tif"
    cropped_path = inter_dir / "georef_cropped.tif"
    if cropped_path.exists() and not rebuild:
        print(f"  [skip] Cropped georef already exists: {cropped_path}")
        return cropped_path

    entity_ids = case["entity_ids"]

    # Step 1: Download from USGS
    downloaded = {}
    need_download = []
    for eid in entity_ids:
        # Check for existing downloads (try .tif and .tgz)
        for ext in (".tif", ".tgz"):
            path = downloads_dir / f"{eid}{ext}"
            if path.exists() and path.stat().st_size > 0:
                downloaded[eid] = str(path)
                print(f"  [skip] Already downloaded: {eid}")
                break
        if eid not in downloaded:
            need_download.append(eid)

    if need_download:
        print(f"\n  --- Downloading {len(need_download)} scenes ---")
        from declass.usgs import USGSClient
        client = USGSClient()
        client.login()
        try:
            available = client.request_downloads(case["dataset"], need_download)
            for item in available:
                eid = item.get("entityId", "")
                url = item.get("url", "")
                filesize = item.get("filesize", 0)
                if not url:
                    print(f"  WARNING: No download URL for {eid}")
                    continue
                ext = ".tgz" if url.endswith(".tgz") else ".tif"
                path = downloads_dir / f"{eid}{ext}"
                client.download_file(url, str(path), filesize)
                downloaded[eid] = str(path)
        finally:
            client.logout()

    if len(downloaded) < len(entity_ids):
        missing = [eid for eid in entity_ids if eid not in downloaded]
        print(f"ERROR: Missing downloads: {missing}")
        sys.exit(1)

    # Step 2: Extract archives
    from declass.extract import extract_archive, list_frames
    entity_dirs = {}
    for eid in entity_ids:
        print(f"\n  --- Extract: {eid} ---")
        entity_dir = extract_archive(downloaded[eid], str(case_dir), eid)
        entity_dirs[eid] = entity_dir

    # Step 3: Sub-frame detection and stitching per entity
    from declass.stitch import stitch_frames, detect_subframe_seams, split_at_seams
    stitched_per_entity = {}

    for eid in entity_ids:
        print(f"\n  --- Sub-frame stitch: {eid} ---")
        entity_dir = entity_dirs[eid]
        frames = list_frames(entity_dir)
        stitched_path = inter_dir / f"{eid}_stitched.tif"

        if stitched_path.exists():
            print(f"  [skip] Already stitched: {eid}")
            stitched_per_entity[eid] = stitched_path
            continue

        if len(frames) == 1:
            # Single TIF — check for sub-frame seams
            seams = detect_subframe_seams(frames[0])
            if seams:
                info_result = subprocess.run(
                    ["gdalinfo", "-json", frames[0]],
                    capture_output=True, text=True,
                )
                info = json.loads(info_result.stdout)
                img_w, img_h = info["size"]
                is_portrait = img_h > img_w

                sub_frames = split_at_seams(frames[0], seams, entity_dir,
                                            is_portrait=is_portrait)
                if len(sub_frames) > 1:
                    stitch_frames(sub_frames, str(stitched_path), str(inter_dir),
                                  preserve_order=True)
                else:
                    shutil.copy2(frames[0], str(stitched_path))
            else:
                shutil.copy2(frames[0], str(stitched_path))
        else:
            stitch_frames(frames, str(stitched_path), str(inter_dir),
                          preserve_order=True)

        stitched_per_entity[eid] = stitched_path

    # Step 4: Use single entity's stitched output
    # KH-4 frames 007/008 are N-S adjacent (not E-W), so we use one frame.
    eid = entity_ids[0]
    georef_input = stitched_per_entity[eid]

    # Step 5: Corners from catalog CSV
    cache = load_corners_cache()
    all_corners = parse_csv_corners(str(case["catalog_csv"]), entity_ids, cache)
    gcp_corners = corners_to_dict(all_corners[eid])

    # Step 6: Orientation detection
    from declass.camera import KH4
    from declass.orientation import detect_orientation

    print(f"\n  --- Orientation detection ---")
    rotation, gcp_corners = detect_orientation(
        str(georef_input), gcp_corners, KH4,
        reference_path=None,
    )
    if rotation != 0:
        print(f"  Orientation: {rotation}° (via GCP assignment)")

    # Step 7: Georef
    if georef_path.exists():
        georef_path.unlink()

    print(f"\n  --- Georef ---")
    from declass.georef import georef_with_corners
    georef_with_corners(str(georef_input), str(georef_path), gcp_corners)

    # Step 8: Crop georef to reference extent
    # KH-4 strips span ~250km but reference may be smaller.
    cropped_path = inter_dir / "georef_cropped.tif"
    print(f"\n  --- Crop to reference extent ---")
    crop_georef_to_reference(georef_path, case["reference"], cropped_path,
                              use_sift=not case.get("force_global", False))

    return cropped_path


def prepare_intermediates(case_key: str, case: dict, case_dir: Path,
                          rebuild: bool) -> Path:
    """Dispatch to the appropriate preparation function."""
    if case_key == "kh9_dzb1212":
        return prepare_kh9(case, case_dir, rebuild)
    elif case_key == "kh4_ds1022":
        return prepare_kh4(case, case_dir, rebuild)
    else:
        raise ValueError(f"Unknown case: {case_key}")


# ─── Alignment runner ────────────────────────────────────────────────────────

def run_alignment(case: dict, case_key: str, version: int, georef_path: Path,
                  timeout: int) -> dict:
    """Run auto-align.py against the case reference. Returns summary dict."""
    case_dir = BENCHMARK_DIR / case_key
    run_dir = case_dir / f"run_v{version}"
    run_dir.mkdir(parents=True, exist_ok=True)

    qa_path = run_dir / "qa.json"
    log_path = run_dir / "run.log"
    output_path = run_dir / "aligned.tif"
    summary_path = run_dir / "summary.json"

    # Snapshot git state
    snapshot_git_state(run_dir / "code_state.txt")

    cmd = [
        PYTHON, str(PROJECT_ROOT / "auto-align.py"),
        str(georef_path),
        "-r", case["reference"],
        "-y", "--best",
        "--diagnostics-dir", str(run_dir) + "/",
        "--qa-json", str(qa_path),
        "-o", str(output_path),
    ]
    if case.get("force_global"):
        cmd.insert(cmd.index("--best") + 1, "--force-global")
    if case.get("anchors"):
        cmd.extend(["--anchors", case["anchors"]])

    print(f"\n{'=' * 60}")
    print(f"  Aligning: {case['name']}")
    print(f"  Version: v{version}")
    print(f"  Output: {run_dir}")
    print(f"  Timeout: {timeout}s")
    print(f"{'=' * 60}\n")

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
                    print(f"\n=== TIMEOUT after {elapsed:.0f}s ===")
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    break

            exit_code = proc.returncode

        except KeyboardInterrupt:
            print("\n=== KeyboardInterrupt ===")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            exit_code = proc.returncode if proc.returncode is not None else -2

    except Exception as e:
        print(f"\n=== Failed to start: {e} ===")
        log_lines.append(f"\nFailed to start: {e}\n")
        exit_code = -1

    wall_clock_s = time.time() - start

    # Write log
    log_text = "".join(log_lines)
    log_path.write_text(log_text)

    print(f"\n=== Finished: exit_code={exit_code}, wall_clock={wall_clock_s:.1f}s ===\n")

    # If pipeline exited early ("already aligned"), copy the georef as the
    # output. The benchmark still records the result — the QA will measure
    # the raw georef quality, which is exactly the pre-alignment baseline.
    if exit_code == 0 and not output_path.exists():
        import shutil
        print("  Pipeline reports already aligned — using georef as output")
        shutil.copy2(str(georef_path), str(output_path))

    # Build summary (reuse run_test.py logic)
    summary = build_summary(version, run_dir, log_text, exit_code, wall_clock_s, qa_path)
    summary["case"] = case_key
    summary["case_name"] = case["name"]
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary


# ─── Version tracking ────────────────────────────────────────────────────────

def detect_next_version() -> int:
    """Scan output/benchmark/*/run_v*/ for highest version, return N+1."""
    versions = []
    for case_key in CASES:
        case_dir = BENCHMARK_DIR / case_key
        if case_dir.exists():
            for d in case_dir.iterdir():
                if d.is_dir():
                    m = re.search(r"run_v(\d+)$", str(d))
                    if m:
                        versions.append(int(m.group(1)))
    return max(versions) + 1 if versions else 1


# ─── Near-perfect check ─────────────────────────────────────────────────────

def is_near_perfect(summary: dict) -> bool:
    """Check if a run meets all near-perfect targets."""
    grid = summary.get("grid")
    if not grid:
        return False
    checks = [
        grid.get("score", 999) < NEAR_PERFECT["score"],
        grid.get("patch_med", 999) < NEAR_PERFECT["patch_med"],
        grid.get("stable_iou", 0) > NEAR_PERFECT["stable_iou"],
        grid.get("shore_iou", 0) > NEAR_PERFECT["shore_iou"],
        grid.get("west", 999) < NEAR_PERFECT["west"],
        grid.get("center", 999) < NEAR_PERFECT["center"],
        grid.get("east", 999) < NEAR_PERFECT["east"],
        abs(grid.get("north", 999)) < NEAR_PERFECT["north"],
    ]
    return all(checks)


# ─── Aggregate summary ──────────────────────────────────────────────────────

def build_benchmark_summary(version: int, results: dict) -> Path:
    """Build aggregate benchmark_v{N}.json from per-case results.

    Preserves existing case results from prior single-case runs.
    """
    out_path = BENCHMARK_DIR / f"benchmark_v{version}.json"

    # Load existing aggregate to preserve other cases' results
    existing_cases = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            existing_cases = existing.get("cases", {})
        except Exception:
            pass

    aggregate = {
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": existing_cases,
    }

    scores = []
    for case_key, summary in results.items():
        grid = summary.get("grid") or {}
        entry = {
            "name": CASES[case_key]["name"],
            "exit_code": summary.get("exit_code"),
            "wall_clock_s": summary.get("wall_clock_s"),
            "score": grid.get("score"),
            "patch_med": grid.get("patch_med"),
            "stable_iou": grid.get("stable_iou"),
            "shore_iou": grid.get("shore_iou"),
            "west": grid.get("west"),
            "center": grid.get("center"),
            "east": grid.get("east"),
            "north": grid.get("north"),
            "near_perfect": is_near_perfect(summary),
            "accepted": grid.get("accepted"),
        }
        aggregate["cases"][case_key] = entry
        if grid.get("score") is not None:
            scores.append(grid["score"])

    aggregate["mean_score"] = round(sum(scores) / len(scores), 1) if scores else None

    out_path = BENCHMARK_DIR / f"benchmark_v{version}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregate, indent=2))
    print(f"\nAggregate summary: {out_path}")
    return out_path


# ─── Comparison table ────────────────────────────────────────────────────────

def load_all_benchmarks() -> list:
    """Load all benchmark_v*.json files, sorted by version."""
    benchmarks = []
    for f in sorted(BENCHMARK_DIR.glob("benchmark_v*.json")):
        m = re.search(r"benchmark_v(\d+)\.json$", f.name)
        if m:
            data = json.loads(f.read_text())
            data["_version"] = int(m.group(1))
            benchmarks.append(data)
    benchmarks.sort(key=lambda x: x["_version"])
    return benchmarks


def print_comparison(highlight_version: int = None):
    """Print cross-version comparison table."""
    benchmarks = load_all_benchmarks()
    if not benchmarks:
        print("No benchmark results found.")
        return

    # Also load individual summaries for versions with only partial runs
    # (comparison only uses aggregate files)

    case_keys = list(CASES.keys())
    case_labels = {
        "kh9_dzb1212": "KH-9 DZB1212",
        "kh4_ds1022": "KH-4 DS1022",
    }

    # Header
    header_parts = [f"{'Ver':>5}"]
    for ck in case_keys:
        label = case_labels.get(ck, ck)
        header_parts.append(f"  | {label:^30}")
    header_parts.append(f"  | {'Aggregate':^10}")
    header = "".join(header_parts)

    sub_header_parts = [f"{'':>5}"]
    for ck in case_keys:
        sub_header_parts.append(f"  | {'Score':>6} {'Patch':>5} {'StIoU':>6} {'NP?':>4}")
    sub_header_parts.append(f"  | {'Mean':>6}")
    sub_header = "".join(sub_header_parts)

    print(f"\n{'=' * 60}")
    print("=== Benchmark Comparison ===")
    print(f"{'=' * 60}")
    print(header)
    print(sub_header)
    print("-" * len(header))

    for bm in benchmarks:
        ver = bm["_version"]
        marker = "*" if ver == highlight_version else " "
        parts = [f"{marker}v{ver:>3}"]

        for ck in case_keys:
            entry = bm.get("cases", {}).get(ck, {})
            score = entry.get("score")
            patch = entry.get("patch_med")
            siou = entry.get("stable_iou")
            np_flag = "YES" if entry.get("near_perfect") else "no"

            if score is not None:
                parts.append(f"  | {score:>6.1f} {patch or 0:>5} "
                             f"{siou or 0:>6.3f} {np_flag:>4}")
            else:
                parts.append(f"  | {'—':>6} {'—':>5} {'—':>6} {'—':>4}")

        mean = bm.get("mean_score")
        if mean is not None:
            parts.append(f"  | {mean:>6.1f}")
        else:
            parts.append(f"  | {'—':>6}")

        print("".join(parts))

    print()

    # Show near-perfect targets for reference
    print("Near-perfect targets: score < 20, patch_med < 5m, "
          f"stable_iou > 0.90, shore_iou > 0.80")
    print(f"                      west/center/east < 10m, |north| < 5m")


# ─── Cleanup ─────────────────────────────────────────────────────────────────

def cleanup_old_runs(case_key: str, keep_version: int):
    """Remove large files from old runs, keeping best + most recent."""
    case_dir = BENCHMARK_DIR / case_key
    if not case_dir.exists():
        return

    run_dirs = {}
    for d in sorted(case_dir.glob("run_v*")):
        if d.is_dir():
            m = re.search(r"run_v(\d+)$", str(d))
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
            grid = summary.get("grid")
            if grid and grid.get("score") is not None:
                if grid["score"] < best_score:
                    best_score = grid["score"]
                    best_version = ver
        except Exception:
            continue

    keep = {keep_version}
    if best_version is not None:
        keep.add(best_version)

    preserve = {"summary.json", "qa.json", "run.log", "code_state.txt", "stderr.log"}

    freed = 0
    for ver, d in run_dirs.items():
        if ver in keep:
            continue
        for item in d.iterdir():
            if item.name in preserve:
                continue
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
        print(f"  Cleanup ({case_key}): freed {freed / 1024 / 1024 / 1024:.1f} GB "
              f"(kept {kept_str})")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark test harness for alignment pipeline validation.\n"
                    "Disk space: ~20GB total for both cases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", "-v", type=int, default=None,
                        help="Force version number (default: auto-increment)")
    parser.add_argument("--timeout", "-t", type=int, default=3600,
                        help="Per-case alignment timeout in seconds (default: 3600)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Delete and regenerate intermediates (stitch/réseau/georef)")
    parser.add_argument("--case", "-c", type=str, default=None,
                        choices=list(CASES.keys()),
                        help="Run only one case")
    parser.add_argument("--compare-only", action="store_true",
                        help="Print comparison table without running")

    args = parser.parse_args()

    if args.compare_only:
        print_comparison()
        return

    version = args.version if args.version is not None else detect_next_version()
    cases_to_run = [args.case] if args.case else list(CASES.keys())

    print(f"{'=' * 60}")
    print(f"  Benchmark v{version}")
    print(f"  Cases: {', '.join(cases_to_run)}")
    print(f"  Timeout: {args.timeout}s per case")
    if args.rebuild:
        print(f"  Rebuild: yes")
    print(f"{'=' * 60}")

    results = {}
    for case_key in cases_to_run:
        case = CASES[case_key]
        case_dir = BENCHMARK_DIR / case_key

        print(f"\n{'#' * 60}")
        print(f"  Case: {case['name']}")
        print(f"{'#' * 60}")

        # Step 1: Prepare intermediates
        try:
            georef_path = prepare_intermediates(case_key, case, case_dir,
                                                args.rebuild)
        except Exception as e:
            print(f"\nERROR preparing intermediates for {case_key}: {e}")
            import traceback
            traceback.print_exc()
            results[case_key] = {
                "exit_code": -1,
                "grid": None,
                "errors": [str(e)],
            }
            continue

        # Step 2: Run alignment
        try:
            summary = run_alignment(case, case_key, version, georef_path,
                                    args.timeout)
            results[case_key] = summary
        except Exception as e:
            print(f"\nERROR running alignment for {case_key}: {e}")
            import traceback
            traceback.print_exc()
            results[case_key] = {
                "exit_code": -1,
                "grid": None,
                "errors": [str(e)],
            }

        # Step 3: Cleanup old runs
        cleanup_old_runs(case_key, version)

    # Build aggregate summary
    if results:
        build_benchmark_summary(version, results)

    # Print comparison
    print_comparison(highlight_version=version)

    # Print individual summaries
    for case_key, summary in results.items():
        grid = summary.get("grid") or {}
        np_flag = "NEAR PERFECT" if is_near_perfect(summary) else ""
        print(f"\n  {CASES[case_key]['name']}:")
        print(f"    Score: {grid.get('score', '—')}, "
              f"Patch: {grid.get('patch_med', '—')}m, "
              f"StIoU: {grid.get('stable_iou', '—')}, "
              f"ShIoU: {grid.get('shore_iou', '—')}")
        if np_flag:
            print(f"    *** {np_flag} ***")


if __name__ == "__main__":
    main()
