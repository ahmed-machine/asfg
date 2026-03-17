"""Réseau mark detection and film distortion correction for KH-9 mapping camera imagery.

The KH-9 mapping camera exposed a grid of tiny crosses (réseau marks) onto the
film at the moment of capture.  These marks form a mathematically perfect grid
(23 rows × 47 columns, 10 mm spacing).  Because the film physically deforms
during decades of storage, the scanned crosses are no longer in a perfect grid.

This module:
  1. Detects the réseau cross locations via template matching.
  2. Fits the detected locations to the known ideal grid (RANSAC).
  3. Computes a Thin-Plate-Spline (TPS) warp to "flatten" the film distortion.
  4. Applies the warp to produce a geometrically corrected image.

Adapted from the approach in sPyMicMac (Donovan et al.), reimplemented as a
standalone module with no MicMac dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np
from osgeo import gdal
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.transform import AffineTransform
from skimage.measure import ransac


# ---------------------------------------------------------------------------
# KH-9 Mapping Camera constants
# ---------------------------------------------------------------------------
GRID_ROWS = 23
GRID_COLS_JOINED = 47
GRID_COLS_HALF = 24
GRID_SPACING_MM = 10.0  # mm between adjacent marks


def _ideal_grid(joined: bool = True) -> np.ndarray:
    """Return the ideal réseau grid coordinates in mm (N×2, columns=[col_mm, row_mm]).

    Origin at the top-left mark, increasing right (col) and down (row).
    """
    n_cols = GRID_COLS_JOINED if joined else GRID_COLS_HALF
    cols = np.arange(n_cols)
    rows = np.arange(GRID_ROWS)
    cc, rr = np.meshgrid(cols, rows)
    return np.column_stack([cc.ravel() * GRID_SPACING_MM,
                            rr.ravel() * GRID_SPACING_MM])


# ---------------------------------------------------------------------------
# Cross template
# ---------------------------------------------------------------------------

def _cross_template(size: int = 51, width: int = 5) -> np.ndarray:
    """Create a synthetic réseau cross template (white cross on black).

    Matches the actual KH-9 crosses which are ~10-12px wide at ~143 px/mm.
    """
    half = size // 2
    hw = width // 2
    tmpl = np.zeros((size, size), dtype=np.float32)
    tmpl[half - hw:half + hw + 1, :] = 255.0
    tmpl[:, half - hw:half + hw + 1] = 255.0
    return tmpl


# ---------------------------------------------------------------------------
# Cross detection — tiled template matching
# ---------------------------------------------------------------------------

def _detect_crosses_in_tile(tile: np.ndarray, cross: np.ndarray,
                            min_dist: int, threshold_quantile: float) -> np.ndarray:
    """Run template matching on a single image tile, return peak coordinates."""
    tile_inv = (tile.max() - tile).astype(np.uint8)
    res = cv2.matchTemplate(tile_inv, cross.astype(np.uint8), cv2.TM_CCORR_NORMED)
    if res.size == 0:
        return np.empty((0, 2))
    coords = peak_local_max(res, min_distance=min_dist,
                            threshold_abs=np.quantile(res, threshold_quantile))
    coords = coords.astype(np.float64) + cross.shape[0] / 2 - 0.5
    return coords


def detect_crosses(img: np.ndarray, cross_size: int = 51,
                   cross_width: int = 5, grid_spacing_px: int = 1400) -> np.ndarray:
    """Detect all réseau crosses in a full-resolution image.

    Returns
    -------
    coords : ndarray, shape (N, 2)
        Detected cross centres in (row, col) pixel coordinates.
    """
    cross = _cross_template(cross_size, cross_width)
    min_dist = grid_spacing_px // 3  # crosses are ~grid_spacing apart

    overlap = 2 * grid_spacing_px
    n_row_tiles, n_col_tiles = 4, 8
    tile_h = img.shape[0] // n_row_tiles
    tile_w = img.shape[1] // n_col_tiles

    all_coords = []
    for ri in range(n_row_tiles):
        for ci in range(n_col_tiles):
            r0 = max(0, ri * tile_h - overlap)
            r1 = min(img.shape[0], (ri + 1) * tile_h + overlap)
            c0 = max(0, ci * tile_w - overlap)
            c1 = min(img.shape[1], (ci + 1) * tile_w + overlap)
            tile = img[r0:r1, c0:c1]

            local = _detect_crosses_in_tile(tile, cross, min_dist, 0.60)
            if local.size == 0:
                continue
            local[:, 0] += r0
            local[:, 1] += c0
            all_coords.append(local)

    if not all_coords:
        raise RuntimeError("No réseau crosses detected in image")

    coords = np.vstack(all_coords)

    # De-duplicate from overlapping tiles
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=min_dist * 0.5)
    remove = set()
    for i, j in pairs:
        remove.add(j)
    mask = np.ones(len(coords), dtype=bool)
    mask[list(remove)] = False
    coords = coords[mask]

    print(f"  [reseau] Detected {len(coords)} candidate crosses")
    return coords


# ---------------------------------------------------------------------------
# Grid matching (assign detected crosses to the ideal grid)
# ---------------------------------------------------------------------------

def _estimate_grid_from_detections(detected: np.ndarray, n_cols: int
                                    ) -> tuple[float, float, float]:
    """Estimate grid origin and spacing from detected cross positions.

    Uses pairwise distance histogram to find the grid spacing, then
    determines the origin from clustering.

    Returns (origin_col, origin_row, spacing_px).
    """
    det_col = detected[:, 1]
    det_row = detected[:, 0]

    # Build KD-tree and find nearest-neighbour distances
    det_xy = np.column_stack([det_col, det_row])
    tree = cKDTree(det_xy)
    # Query k=5 nearest neighbours for each point
    dists, _ = tree.query(det_xy, k=6)  # k=6 includes self
    nn_dists = dists[:, 1]  # nearest non-self neighbour

    # The grid spacing should be the dominant nearest-neighbour distance
    # Use a histogram to find the peak
    valid_dists = nn_dists[nn_dists > 100]  # filter out very close spurious
    if len(valid_dists) < 20:
        raise RuntimeError("Too few detections to estimate grid spacing")

    hist, edges = np.histogram(valid_dists, bins=200)
    peak_idx = hist.argmax()
    spacing_px = (edges[peak_idx] + edges[peak_idx + 1]) / 2
    print(f"  [reseau] Estimated spacing from NN distances: {spacing_px:.0f}px")

    # Filter detections to only those with NN distance close to the grid spacing
    # These are the "on-grid" candidates
    on_grid = np.abs(nn_dists - spacing_px) < spacing_px * 0.15
    grid_candidates = detected[on_grid]
    print(f"  [reseau] On-grid candidates: {on_grid.sum()}/{len(detected)}")

    if on_grid.sum() < 20:
        # Fallback: use all detections
        grid_candidates = detected

    # Estimate origin: the crosses form a regular grid, so the col/row
    # positions modulo spacing should cluster tightly.
    gc_col = grid_candidates[:, 1]
    gc_row = grid_candidates[:, 0]

    # Find the origin by taking the minimum of the on-grid candidates
    # and rounding to the nearest grid point
    col_mod = gc_col % spacing_px
    row_mod = gc_row % spacing_px

    # The mode of the modular residual gives the grid offset
    def _circular_mean(vals, period):
        """Circular mean for periodic data."""
        angles = 2 * np.pi * vals / period
        return period * np.arctan2(np.mean(np.sin(angles)),
                                   np.mean(np.cos(angles))) / (2 * np.pi) % period

    col_offset = _circular_mean(col_mod, spacing_px)
    row_offset = _circular_mean(row_mod, spacing_px)

    # Origin = the grid point closest to (0, 0)
    # Find the smallest col/row value that matches the offset
    origin_col = col_offset
    origin_row = row_offset

    # Make sure origin is reasonable (within the first spacing from edge)
    while origin_col > spacing_px * 1.5:
        origin_col -= spacing_px
    while origin_row > spacing_px * 1.5:
        origin_row -= spacing_px

    print(f"  [reseau] Grid origin: col={origin_col:.0f}, row={origin_row:.0f}, "
          f"spacing={spacing_px:.0f}px")
    return origin_col, origin_row, spacing_px


def match_to_grid(detected: np.ndarray, joined: bool = True,
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match detected cross positions to the ideal 23×47 réseau grid.

    Uses nearest-neighbour distance analysis to find grid spacing and origin,
    then RANSAC with an affine model for robust assignment.

    Returns
    -------
    ideal_pts : ndarray (M, 2)  — ideal grid coords in mm (col_mm, row_mm)
    detected_pts : ndarray (M, 2) — matched pixel coords (col_px, row_px)
    grid_indices : ndarray (M, 2) — (row_idx, col_idx) into the grid
    """
    n_cols = GRID_COLS_JOINED if joined else GRID_COLS_HALF
    ideal = _ideal_grid(joined)

    # Robust estimation of grid parameters from detections
    origin_col, origin_row, spacing_px = _estimate_grid_from_detections(
        detected, n_cols
    )
    approx_scale = spacing_px / GRID_SPACING_MM
    print(f"  [reseau] Approximate scale: {approx_scale:.1f} px/mm")

    # Predict pixel positions using estimated origin and spacing
    predicted_px = np.column_stack([
        origin_col + ideal[:, 0] * approx_scale,
        origin_row + ideal[:, 1] * approx_scale,
    ])

    # Nearest-neighbour assignment
    det_xy = np.column_stack([detected[:, 1], detected[:, 0]])  # (col, row)
    tree = cKDTree(det_xy)
    dists, indices = tree.query(predicted_px, k=1)

    # Accept matches within 30% of a grid spacing
    threshold = 0.3 * spacing_px
    valid = dists < threshold
    print(f"  [reseau] Initial nearest-neighbour: {valid.sum()}/{len(ideal)} "
          f"within threshold ({threshold:.0f}px)")

    if valid.sum() < 20:
        raise RuntimeError(
            f"Only {valid.sum()} grid points matched (need >= 20). "
            f"Threshold={threshold:.0f}px."
        )

    matched_ideal = ideal[valid]
    matched_det = det_xy[indices[valid]]

    all_grid_rc = np.column_stack([
        np.repeat(np.arange(GRID_ROWS), n_cols),
        np.tile(np.arange(n_cols), GRID_ROWS),
    ])
    matched_grid_rc = all_grid_rc[valid]

    # Refine with RANSAC (affine: ideal_mm -> pixel)
    model, inliers = ransac(
        (matched_ideal, matched_det),
        AffineTransform,
        min_samples=10,
        residual_threshold=spacing_px * 0.1,
        max_trials=5000,
    )

    n_inliers = inliers.sum()
    n_expected = GRID_ROWS * n_cols
    print(f"  [reseau] RANSAC inliers: {n_inliers}/{valid.sum()} "
          f"(of {n_expected} expected)")

    # Re-assign ALL grid points using the refined affine model
    predicted_all = model(ideal)
    dists2, indices2 = tree.query(predicted_all, k=1)
    refined_threshold = spacing_px * 0.15  # tight threshold after RANSAC
    valid2 = dists2 < refined_threshold

    final_ideal = ideal[valid2]
    final_det = det_xy[indices2[valid2]]
    final_grid_rc = all_grid_rc[valid2]

    print(f"  [reseau] After RANSAC re-assignment: {valid2.sum()}/{n_expected} matched")

    # Compute residuals
    predicted_final = model(final_ideal)
    residuals = np.sqrt(((predicted_final - final_det) ** 2).sum(axis=1))
    print(f"  [reseau] Residuals: median={np.median(residuals):.1f}px, "
          f"max={residuals.max():.1f}px, mean={residuals.mean():.1f}px")

    return final_ideal, final_det, final_grid_rc


# ---------------------------------------------------------------------------
# Sub-pixel refinement
# ---------------------------------------------------------------------------

def _subpixel_refine_gdal(ds, cross_size: int, cross_width: int,
                          det_pts: np.ndarray, half_win: int = 100) -> np.ndarray:
    """Refine cross positions by reading small windows from GDAL dataset."""
    cross = _cross_template(cross_size, cross_width)
    refined = det_pts.copy()
    band = ds.GetRasterBand(1)
    h, w = ds.RasterYSize, ds.RasterXSize
    ch = cross.shape[0]

    for i in range(len(det_pts)):
        cx, cy = int(round(det_pts[i, 0])), int(round(det_pts[i, 1]))
        c0 = max(0, cx - half_win)
        r0 = max(0, cy - half_win)
        c1 = min(w, cx + half_win)
        r1 = min(h, cy + half_win)
        pw, ph = c1 - c0, r1 - r0

        if pw < ch or ph < ch:
            continue

        patch = band.ReadAsArray(xoff=c0, yoff=r0, win_xsize=pw, win_ysize=ph)
        if patch is None:
            continue

        patch_inv = (patch.max() - patch).astype(np.uint8)
        res = cv2.matchTemplate(patch_inv, cross.astype(np.uint8), cv2.TM_CCORR_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        refined[i, 0] = c0 + max_loc[0] + ch / 2 - 0.5
        refined[i, 1] = r0 + max_loc[1] + ch / 2 - 0.5

    return refined


# ---------------------------------------------------------------------------
# GDAL-based TPS warp (memory-safe for large images)
# ---------------------------------------------------------------------------

def flatten_with_gdal(input_path: str, output_path: str,
                      detected_pts: np.ndarray, ideal_pts_mm: np.ndarray,
                      scale_px_per_mm: float) -> str:
    """Use GDAL TPS warp to correct film distortion on disk."""
    dst_pts = ideal_pts_mm * scale_px_per_mm

    out_w = int(np.ceil(dst_pts[:, 0].max() + scale_px_per_mm * GRID_SPACING_MM))
    out_h = int(np.ceil(dst_pts[:, 1].max() + scale_px_per_mm * GRID_SPACING_MM))

    ds = gdal.Open(input_path)
    if ds is None:
        raise RuntimeError(f"Cannot open {input_path}")

    # GCPs map from output pixel space (ideal) to input pixel space (detected)
    gcp_list = []
    for i in range(len(detected_pts)):
        gcp = gdal.GCP(
            float(dst_pts[i, 0]),       # output col ("X")
            float(dst_pts[i, 1]),       # output row ("Y")
            0,
            float(detected_pts[i, 0]),  # source col
            float(detected_pts[i, 1]),  # source row
        )
        gcp_list.append(gcp)

    ds.SetGCPs(gcp_list, '')
    ds.FlushCache()

    print(f"  [reseau] GDAL TPS warp: {ds.RasterXSize}x{ds.RasterYSize} -> "
          f"{out_w}x{out_h} ({len(gcp_list)} GCPs)")

    warp_options = gdal.WarpOptions(
        format='GTiff',
        width=out_w,
        height=out_h,
        tps=True,
        resampleAlg=gdal.GRA_Lanczos,
        creationOptions=[
            'COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES', 'BIGTIFF=IF_SAFER',
        ],
    )

    out_ds = gdal.Warp(output_path, ds, options=warp_options)
    if out_ds is None:
        raise RuntimeError("GDAL Warp failed")

    out_ds.FlushCache()
    out_ds = None
    ds = None

    print(f"  [reseau] Flattened image saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def process_kh9_reseau(input_path: str, output_path: str,
                       joined: bool = True,
                       cross_size: int = 51, cross_width: int = 5,
                       save_diagnostics: bool = True) -> dict:
    """Full réseau correction pipeline for a KH-9 mapping camera image.

    Parameters
    ----------
    input_path : str
        Path to the raw scanned KH-9 image (TIFF).
    output_path : str
        Path for the corrected output image.
    joined : bool
        True if the image is a joined full-frame (both halves stitched).
    cross_size : int
        Template size in pixels. Default 51 works for ~143 px/mm scans.
    cross_width : int
        Width of cross arms in pixels. Default 5.
    save_diagnostics : bool
        If True, save a JSON file with detected grid points and residuals.

    Returns
    -------
    dict with keys: n_detected, n_matched, median_residual_px, output_path
    """
    t0 = time.time()
    print(f"[reseau] Processing: {input_path}")

    gdal.UseExceptions()
    ds = gdal.Open(input_path)
    if ds is None:
        raise RuntimeError(f"Cannot open {input_path}")
    width, height = ds.RasterXSize, ds.RasterYSize
    n_bands = ds.RasterCount
    print(f"  [reseau] Image: {width}x{height}, {n_bands} band(s)")

    # Estimate scale from image dimensions
    n_cols = GRID_COLS_JOINED if joined else GRID_COLS_HALF
    expected_width_mm = (n_cols + 1) * GRID_SPACING_MM  # with margins
    scale_est = width / expected_width_mm
    grid_spacing_px = int(round(scale_est * GRID_SPACING_MM))
    print(f"  [reseau] Estimated grid spacing: {grid_spacing_px}px "
          f"(~{scale_est:.0f} px/mm)")

    # Step 1: Read image and detect crosses
    # For images under ~4 GB, read the whole band 1 into memory
    print(f"  [reseau] Step 1: Reading image and detecting crosses...")
    band = ds.GetRasterBand(1)
    img = band.ReadAsArray().astype(np.uint8)
    ds = None

    coords = detect_crosses(img, cross_size, cross_width, grid_spacing_px)
    del img  # free memory

    # Step 2: Match to ideal grid
    print(f"  [reseau] Step 2: Matching detections to ideal grid...")
    ideal_pts, det_pts, grid_indices = match_to_grid(coords, joined=joined)

    # Step 3: Sub-pixel refinement
    print(f"  [reseau] Step 3: Sub-pixel refinement...")
    ds = gdal.Open(input_path)
    refined_pts = _subpixel_refine_gdal(ds, cross_size, cross_width, det_pts)
    ds = None

    # Compute final scale from matched points
    col_range_px = refined_pts[:, 0].max() - refined_pts[:, 0].min()
    col_range_mm = ideal_pts[:, 0].max() - ideal_pts[:, 0].min()
    scale_px_per_mm = col_range_px / col_range_mm if col_range_mm > 0 else scale_est
    print(f"  [reseau] Output scale: {scale_px_per_mm:.1f} px/mm")

    # Step 4: Flatten with GDAL TPS
    print(f"  [reseau] Step 4: Applying TPS correction...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    flatten_with_gdal(input_path, output_path, refined_pts, ideal_pts,
                      scale_px_per_mm)

    elapsed = time.time() - t0

    result = {
        'n_detected': len(coords),
        'n_matched': len(refined_pts),
        'n_expected': GRID_ROWS * n_cols,
        'scale_px_per_mm': round(float(scale_px_per_mm), 2),
        'output_path': output_path,
        'elapsed_s': round(elapsed, 1),
    }

    if save_diagnostics:
        diag_path = output_path.replace('.tif', '_reseau_diag.json')
        diag = {
            **result,
            'ideal_pts_mm': ideal_pts.tolist(),
            'detected_pts_px': refined_pts.tolist(),
            'grid_indices': grid_indices.tolist(),
        }
        with open(diag_path, 'w') as f:
            json.dump(diag, f, indent=2)
        print(f"  [reseau] Diagnostics saved: {diag_path}")

    print(f"[reseau] Done in {elapsed:.0f}s. "
          f"Matched {len(refined_pts)}/{result['n_expected']} grid points.")
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Correct KH-9 mapping camera film distortion using réseau marks."
    )
    parser.add_argument("input", help="Path to raw scanned KH-9 image (TIFF)")
    parser.add_argument("output", help="Path for corrected output image")
    parser.add_argument("--half", action="store_true",
                        help="Image is a single half-frame (23x24 grid)")
    parser.add_argument("--cross-size", type=int, default=51,
                        help="Cross template size in pixels (default: 51)")
    parser.add_argument("--cross-width", type=int, default=5,
                        help="Cross arm width in pixels (default: 5)")
    parser.add_argument("--no-diag", action="store_true",
                        help="Skip saving diagnostics JSON")
    args = parser.parse_args()

    process_kh9_reseau(
        input_path=args.input,
        output_path=args.output,
        joined=not args.half,
        cross_size=args.cross_size,
        cross_width=args.cross_width,
        save_diagnostics=not args.no_diag,
    )


if __name__ == "__main__":
    main()
