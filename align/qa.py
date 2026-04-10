"""Alignment quality metrics, QA orchestration, and diagnostic visualization."""

import json
from typing import Iterable

import cv2
import numpy as np
import rasterio
import rasterio.transform
import torch

from .geo import read_overlap_region
from .image import build_semantic_masks, shoreline_mask, to_u8
from .models import get_torch_device
from .types import MatchPair, QaReport

_QA_GRID_ROWS = 4
_QA_GRID_COLS = 6
_MIN_CELL_EDGE_PX = 20
_MIN_CELL_PATCHES = 2
_IOU_RAMP_LO = 50       # union pixels where IoU penalty starts ramping in
_IOU_RAMP_HI = 200      # union pixels where IoU penalty reaches full weight
_MAX_METRIC_M = 500.0   # cap for inf grid_score / patch_med (meters)


def compute_shoreline_iou_and_median(ref_arr, test_arr, valid_mask, res_m):
    """Return shoreline IoU and median symmetric shoreline mismatch (meters)."""
    m_ref = (shoreline_mask(ref_arr) > 0) & valid_mask
    m_test = (shoreline_mask(test_arr) > 0) & valid_mask

    inter = np.sum(m_ref & m_test)
    union = np.sum(m_ref | m_test)
    iou = float(inter / union) if union > 0 else 0.0

    k = np.ones((3, 3), np.uint8)
    e_ref = cv2.morphologyEx(m_ref.astype(np.uint8), cv2.MORPH_GRADIENT, k) > 0
    e_test = cv2.morphologyEx(m_test.astype(np.uint8), cv2.MORPH_GRADIENT, k) > 0
    common = e_ref | e_test
    if not np.any(common):
        return iou, _MAX_METRIC_M

    d_ref = cv2.distanceTransform((~e_ref).astype(np.uint8), cv2.DIST_L2, 3)
    d_test = cv2.distanceTransform((~e_test).astype(np.uint8), cv2.DIST_L2, 3)
    med_m = float(np.median(0.5 * (d_ref[common] + d_test[common])) * res_m)
    return iou, med_m


def _batch_phase_correlate(ref_patches: torch.Tensor, test_patches: torch.Tensor,
                           min_resp: float = 0.03):
    """Batched phase correlation using torch.fft on GPU/MPS.

    Args:
        ref_patches: (N, H, W) float32 tensor (already windowed)
        test_patches: (N, H, W) float32 tensor (already windowed)
        min_resp: minimum response threshold

    Returns:
        shifts: (N, 2) array of (dx, dy) shifts
        responses: (N,) array of response values
        valid: (N,) bool array
    """
    N, H, W = ref_patches.shape
    A = torch.fft.fft2(ref_patches)
    B = torch.fft.fft2(test_patches)
    cross = A * B.conj()
    mag = cross.abs().clamp(min=1e-8)
    cross_norm = cross / mag
    corr = torch.fft.ifft2(cross_norm).real

    # Find peak in each correlation surface
    corr_flat = corr.view(N, -1)
    peak_vals, peak_idxs = corr_flat.max(dim=1)
    peak_rows = peak_idxs // W
    peak_cols = peak_idxs % W

    # Wrap around for negative shifts
    dy = peak_rows.float()
    dx = peak_cols.float()
    dy = torch.where(dy > H / 2, dy - H, dy)
    dx = torch.where(dx > W / 2, dx - W, dx)

    # Response: peak value relative to mean
    mean_vals = corr_flat.mean(dim=1)
    responses = (peak_vals - mean_vals).clamp(min=0)

    shifts = torch.stack([dx, dy], dim=1).cpu().numpy()
    responses = responses.cpu().numpy()
    valid = responses >= min_resp

    return shifts, responses, valid


def compute_patch_residual_median(ref_arr, test_arr, valid_mask, res_m,
                                  patch=160, stride=160,
                                  min_valid=0.6, min_resp=0.03):
    """Return median local residual shift (meters) via batched phase correlation."""
    h, w = ref_arr.shape
    win = np.outer(np.hanning(patch), np.hanning(patch)).astype(np.float32)

    # Collect all valid patches
    ref_patches = []
    test_patches = []

    for r0 in range(0, max(1, h - patch + 1), stride):
        r1 = min(r0 + patch, h)
        if r1 - r0 < patch:
            r0 = max(0, r1 - patch)
        for c0 in range(0, max(1, w - patch + 1), stride):
            c1 = min(c0 + patch, w)
            if c1 - c0 < patch:
                c0 = max(0, c1 - patch)

            if np.mean(valid_mask[r0:r1, c0:c1]) < min_valid:
                continue

            a = ref_arr[r0:r1, c0:c1].astype(np.float32)
            b = test_arr[r0:r1, c0:c1].astype(np.float32)
            if a.shape == win.shape:
                ref_patches.append(a * win)
                test_patches.append(b * win)
            else:
                local_win = np.outer(
                    np.hanning(a.shape[0]), np.hanning(a.shape[1])
                ).astype(np.float32)
                ref_patches.append(a * local_win)
                test_patches.append(b * local_win)

    if not ref_patches:
        return {"median": _MAX_METRIC_M, "p90": _MAX_METRIC_M, "max": _MAX_METRIC_M, "count": 0}

    # Batch phase correlation on GPU/MPS
    device = get_torch_device()
    ref_t = torch.from_numpy(np.stack(ref_patches)).to(device)
    test_t = torch.from_numpy(np.stack(test_patches)).to(device)

    with torch.no_grad():
        shifts, responses, valid = _batch_phase_correlate(ref_t, test_t, min_resp)

    mags = np.hypot(shifts[valid, 0], shifts[valid, 1]) * res_m

    if len(mags) == 0:
        return {"median": _MAX_METRIC_M, "p90": _MAX_METRIC_M, "max": _MAX_METRIC_M, "count": 0}
    return {
        "median": float(np.median(mags)),
        "p90": float(np.percentile(mags, 90)),
        "max": float(np.max(mags)),
        "count": int(len(mags)),
    }


def _compute_grid_metrics(sym, common, h, w, n_rows, n_cols):
    """Subdivide image into n_rows x n_cols cells and compute per-cell shoreline median."""
    cells = []
    row_h = h / n_rows
    col_w = w / n_cols
    for r in range(n_rows):
        r0 = int(round(r * row_h))
        r1 = int(round((r + 1) * row_h))
        for c in range(n_cols):
            c0 = int(round(c * col_w))
            c1 = int(round((c + 1) * col_w))
            cell_common = common[r0:r1, c0:c1]
            edge_px = int(np.sum(cell_common))
            if edge_px >= _MIN_CELL_EDGE_PX:
                cell_vals = sym[r0:r1, c0:c1][cell_common]
                med = float(np.median(cell_vals))
                cells.append({"row": r, "col": c, "shoreline_med": med,
                              "edge_px": edge_px, "valid": True})
            else:
                cells.append({"row": r, "col": c, "shoreline_med": None,
                              "edge_px": edge_px, "valid": False})
    return cells


def _derive_legacy_from_grid(cells, n_cols):
    """Derive legacy west/center/east/north_shift from grid cells."""
    left_cols = set(range(n_cols // 3))                          # cols 0-1
    mid_cols = set(range(n_cols // 3, 2 * n_cols // 3))          # cols 2-3
    right_cols = set(range(2 * n_cols // 3, n_cols))             # cols 4-5

    def _median_for_cols(col_set):
        vals = [c["shoreline_med"] for c in cells
                if c["valid"] and c["col"] in col_set]
        return float(np.median(vals)) if vals else None

    west = _median_for_cols(left_cols)
    center = _median_for_cols(mid_cols)
    east = _median_for_cols(right_cols)

    # north_shift: median of top-row cells minus bottom-row cells
    top_vals = [c["shoreline_med"] for c in cells if c["valid"] and c["row"] == 0]
    bot_vals = [c["shoreline_med"] for c in cells
                if c["valid"] and c["row"] == _QA_GRID_ROWS - 1]
    if top_vals and bot_vals:
        north_shift = float(np.median(top_vals) - np.median(bot_vals))
    else:
        north_shift = None

    return west, center, east, north_shift


def _compute_patch_grid_metrics(ref_arr, out_arr, valid_mask, eval_res,
                                n_rows, n_cols, patch=128, stride=64,
                                min_valid=0.5, min_resp=0.03):
    """Per-grid-cell patch displacement via batched phase correlation.

    Returns list of cell dicts with patch_med (meters) for each cell.
    """
    device = get_torch_device()
    h, w = ref_arr.shape
    row_h = h / n_rows
    col_w = w / n_cols
    win = np.outer(np.hanning(patch), np.hanning(patch)).astype(np.float32)
    cells = []

    for r in range(n_rows):
        r0 = int(round(r * row_h))
        r1 = int(round((r + 1) * row_h))
        for c in range(n_cols):
            c0 = int(round(c * col_w))
            c1 = int(round((c + 1) * col_w))

            cell_ref = ref_arr[r0:r1, c0:c1]
            cell_out = out_arr[r0:r1, c0:c1]
            cell_valid = valid_mask[r0:r1, c0:c1]
            ch, cw = cell_ref.shape

            ref_patches = []
            out_patches = []
            for pr0 in range(0, max(1, ch - patch + 1), stride):
                pr1 = min(pr0 + patch, ch)
                if pr1 - pr0 < patch:
                    pr0 = max(0, pr1 - patch)
                for pc0 in range(0, max(1, cw - patch + 1), stride):
                    pc1 = min(pc0 + patch, cw)
                    if pc1 - pc0 < patch:
                        pc0 = max(0, pc1 - patch)
                    if np.mean(cell_valid[pr0:pr1, pc0:pc1]) < min_valid:
                        continue
                    a = cell_ref[pr0:pr1, pc0:pc1].astype(np.float32)
                    b = cell_out[pr0:pr1, pc0:pc1].astype(np.float32)
                    if a.shape == win.shape:
                        ref_patches.append(a * win)
                        out_patches.append(b * win)

            if len(ref_patches) < 2:
                cells.append({"row": r, "col": c, "patch_med": None,
                              "patch_count": len(ref_patches), "valid": False})
                continue

            ref_t = torch.from_numpy(np.stack(ref_patches)).to(device)
            out_t = torch.from_numpy(np.stack(out_patches)).to(device)
            with torch.no_grad():
                shifts, _, vmask = _batch_phase_correlate(ref_t, out_t, min_resp)
            mags = np.hypot(shifts[vmask, 0], shifts[vmask, 1]) * eval_res

            if len(mags) < 2:
                cells.append({"row": r, "col": c, "patch_med": None,
                              "patch_count": int(len(mags)), "valid": False})
                continue

            cells.append({
                "row": r, "col": c,
                "patch_med": float(np.median(mags)),
                "patch_count": int(len(mags)),
                "valid": True,
            })

    return cells


def _compute_boundary_distance(mask_ref, mask_out, valid_mask, eval_res):
    """Compute median boundary distance (meters) between two binary masks.

    Uses distance transforms on the mask boundaries. Returns the median
    symmetric distance in meters, or None if no boundary pixels exist.
    """
    k = np.ones((3, 3), np.uint8)
    edge_ref = cv2.morphologyEx(mask_ref.astype(np.uint8), cv2.MORPH_GRADIENT, k) > 0
    edge_out = cv2.morphologyEx(mask_out.astype(np.uint8), cv2.MORPH_GRADIENT, k) > 0
    edge_ref = edge_ref & valid_mask
    edge_out = edge_out & valid_mask
    common = edge_ref | edge_out

    if not np.any(common):
        return None

    d_ref = cv2.distanceTransform((~edge_ref).astype(np.uint8), cv2.DIST_L2, 3)
    d_out = cv2.distanceTransform((~edge_out).astype(np.uint8), cv2.DIST_L2, 3)
    sym = 0.5 * (d_ref[common] + d_out[common]) * eval_res
    return float(np.median(sym))


def evaluate_alignment_quality_arrays(ref_arr, out_arr, valid_mask, eval_res=8.0, mask_mode="coastal_obia"):
    """Compute grid-based shoreline drift and patch residual metrics."""
    bundle_ref = build_semantic_masks(ref_arr, mode=mask_mode)
    bundle_out = build_semantic_masks(out_arr, mode=mask_mode)
    m_ref = (bundle_ref.shoreline > 0) & valid_mask
    m_out = (bundle_out.shoreline > 0) & valid_mask
    k = np.ones((3, 3), np.uint8)
    e_ref = cv2.morphologyEx(m_ref.astype(np.uint8), cv2.MORPH_GRADIENT, k) > 0
    e_out = cv2.morphologyEx(m_out.astype(np.uint8), cv2.MORPH_GRADIENT, k) > 0
    common = e_ref | e_out

    # Compute distance transforms even if common is empty (needed for grid)
    has_edges = np.any(common)
    if has_edges:
        d_ref = cv2.distanceTransform((~e_ref).astype(np.uint8), cv2.DIST_L2, 3)
        d_out = cv2.distanceTransform((~e_out).astype(np.uint8), cv2.DIST_L2, 3)
        h, w = d_ref.shape
        sym = np.zeros((h, w), dtype=np.float32)
        sym[common] = 0.5 * (d_ref[common] + d_out[common]) * eval_res
    else:
        h, w = ref_arr.shape
        sym = np.zeros((h, w), dtype=np.float32)

    # Grid-based regional scoring
    n_rows, n_cols = _QA_GRID_ROWS, _QA_GRID_COLS
    cells = _compute_grid_metrics(sym, common, h, w, n_rows, n_cols)
    valid_cells = [c for c in cells if c["valid"]]
    valid_count = len(valid_cells)
    total_count = len(cells)

    if valid_count > 0:
        grid_score = float(np.mean([c["shoreline_med"] for c in valid_cells]))
    else:
        grid_score = _MAX_METRIC_M
    grid_coverage = valid_count / total_count if total_count > 0 else 0.0

    # Derive legacy directional keys from grid
    west, center, east, north_shift = _derive_legacy_from_grid(cells, n_cols)

    # IoU metrics
    stable_ref = (bundle_ref.stable > 0) & valid_mask
    stable_out = (bundle_out.stable > 0) & valid_mask
    stable_union = int(np.sum(stable_ref | stable_out))
    stable_iou = float(np.sum(stable_ref & stable_out) / stable_union) if stable_union > 0 else 0.0
    shore_union = int(np.sum(m_ref | m_out))
    shore_iou = float(np.sum(m_ref & m_out) / shore_union) if shore_union > 0 else 0.0

    # Patch residual (global)
    patch_result = compute_patch_residual_median(
        to_u8(ref_arr),
        to_u8(out_arr),
        valid_mask & (stable_ref | stable_out),
        eval_res,
        patch=192,
        stride=96,
    )
    patch_med = patch_result["median"]

    # Per-cell patch displacement (Phase 3a: inland coverage)
    patch_grid_cells = _compute_patch_grid_metrics(
        to_u8(ref_arr), to_u8(out_arr),
        valid_mask & (stable_ref | stable_out),
        eval_res, n_rows, n_cols,
    )
    patch_grid_valid = [c for c in patch_grid_cells if c["valid"]]
    patch_grid_coverage = len(patch_grid_valid) / len(patch_grid_cells) if patch_grid_cells else 0.0

    # Boundary distance metrics (Phase 3b: replaces IoU as shift-sensitive)
    stable_boundary_m = _compute_boundary_distance(stable_ref, stable_out, valid_mask, eval_res)
    shore_boundary_m = _compute_boundary_distance(m_ref, m_out, valid_mask, eval_res)

    # No valid grid cells and no patches — nothing to score
    if valid_count == 0 and patch_result["count"] == 0:
        return None

    # Score formula — weights shift toward patch_med when grid has low coverage
    if grid_coverage >= 0.25:
        grid_w, patch_w = 0.55, 0.25
    elif grid_coverage > 0:
        blend = grid_coverage / 0.25
        grid_w = 0.55 * blend
        patch_w = 0.25 + 0.55 * (1.0 - blend)
    else:
        grid_w, patch_w = 0.0, 0.80

    grid_contrib = grid_w * grid_score
    patch_contrib = patch_w * patch_med

    # Boundary distance penalties (smooth ramp by union size)
    def _boundary_penalty(weight, boundary_m, union_px):
        if boundary_m is None:
            return 0.0
        ramp = float(np.clip((union_px - _IOU_RAMP_LO) / (_IOU_RAMP_HI - _IOU_RAMP_LO), 0.0, 1.0))
        # Scale: 0m = 0 penalty, boundary_m feeds directly (capped at weight)
        return min(weight, boundary_m * (weight / 50.0)) * ramp

    stable_boundary_penalty = _boundary_penalty(18.0, stable_boundary_m, stable_union)
    shore_boundary_penalty = _boundary_penalty(12.0, shore_boundary_m, shore_union)

    score = grid_contrib + patch_contrib + stable_boundary_penalty + shore_boundary_penalty

    return {
        # Legacy keys (derived from grid cells for compat)
        "west": west,
        "center": center,
        "east": east,
        "north_shift": north_shift,
        # Grid detail
        "grid": {
            "rows": n_rows,
            "cols": n_cols,
            "cells": cells,
            "valid_count": valid_count,
            "total_count": total_count,
        },
        "grid_score": grid_score,
        "grid_coverage": round(grid_coverage, 3),
        # Patch metrics
        "patch_med": patch_med,
        "patch_p90": patch_result["p90"],
        "patch_max": patch_result["max"],
        "patch_count": patch_result["count"],
        # Per-cell patch grid (inland coverage)
        "patch_grid": {
            "cells": patch_grid_cells,
            "valid_count": len(patch_grid_valid),
            "total_count": len(patch_grid_cells),
            "coverage": round(patch_grid_coverage, 3),
            "median_m": float(np.mean([c["patch_med"] for c in patch_grid_valid]))
                        if patch_grid_valid else None,
        },
        # IoU (kept for backward compat)
        "stable_iou": stable_iou,
        "shore_iou": shore_iou,
        # Boundary distance (meters — shift-sensitive replacement for IoU)
        "stable_boundary_m": stable_boundary_m,
        "shore_boundary_m": shore_boundary_m,
        # Score
        "score": float(score),
        "score_breakdown": {
            "grid_contrib": round(float(grid_contrib), 2),
            "patch_contrib": round(float(patch_contrib), 2),
            "stable_boundary_penalty": round(float(stable_boundary_penalty), 2),
            "shore_boundary_penalty": round(float(shore_boundary_penalty), 2),
            "grid_weight": round(grid_w, 3),
            "patch_weight": round(patch_w, 3),
        },
    }


def evaluate_alignment_quality_paths(output_path, reference_path, overlap, work_crs, eval_res=8.0,
                                     mask_mode="coastal_obia"):
    """Compute alignment QA metrics directly from raster paths."""
    with rasterio.open(reference_path) as src_ref, rasterio.open(output_path) as src_out:
        arr_ref, _ = read_overlap_region(src_ref, overlap, work_crs, eval_res)
        arr_out, _ = read_overlap_region(src_out, overlap, work_crs, eval_res)

    valid = (arr_ref > 0) & (arr_out > 0)
    if np.mean(valid) < 0.05:
        return None
    return evaluate_alignment_quality_arrays(arr_ref, arr_out, valid, eval_res=eval_res, mask_mode=mask_mode)


# ---------------------------------------------------------------------------
# Debug visualization
# ---------------------------------------------------------------------------

def generate_debug_image(src_ref, src_offset, overlap, work_crs,
                         matched_pairs, geo_residuals, mean_residual,
                         output_path):
    """Generate a diagnostic side-by-side JPEG showing GCP positions.

    Anchors are drawn as yellow crosses, good matches as green dots,
    higher-residual matches as orange dots.
    """
    try:
        vis_res = 15.0
        arr_ref_vis, ref_vis_transform = read_overlap_region(
            src_ref, overlap, work_crs, vis_res)
        arr_off_vis, off_vis_transform = read_overlap_region(
            src_offset, overlap, work_crs, vis_res)

        ref_rgb = cv2.cvtColor(to_u8(arr_ref_vis), cv2.COLOR_GRAY2BGR)
        off_rgb = cv2.cvtColor(to_u8(arr_off_vis), cv2.COLOR_GRAY2BGR)

        match_coords = []

        for i, pair in enumerate(matched_pairs):
            rgx, rgy = pair.ref_x, pair.ref_y
            ogx, ogy = pair.off_x, pair.off_y
            name = pair.name
            residual = geo_residuals[i]

            ref_row, ref_col = rasterio.transform.rowcol(ref_vis_transform, rgx, rgy)
            off_row, off_col = rasterio.transform.rowcol(off_vis_transform, ogx, ogy)
            ref_row, ref_col = int(ref_row), int(ref_col)
            off_row, off_col = int(off_row), int(off_col)

            if not (0 <= ref_row < ref_rgb.shape[0] and
                    0 <= ref_col < ref_rgb.shape[1] and
                    0 <= off_row < off_rgb.shape[0] and
                    0 <= off_col < off_rgb.shape[1]):
                continue

            match_coords.append((ref_col, ref_row, off_col, off_row))

            if name.startswith("anchor:"):
                color = (0, 255, 255)
                radius = 5
            elif residual < mean_residual:
                color = (0, 255, 0)
                radius = 3
            else:
                color = (0, 165, 255)
                radius = 3

            cv2.circle(ref_rgb, (ref_col, ref_row), radius, color, -1)
            cv2.circle(ref_rgb, (ref_col, ref_row), radius + 2, (255, 255, 255), 1)
            cv2.circle(off_rgb, (off_col, off_row), radius, color, -1)
            cv2.circle(off_rgb, (off_col, off_row), radius + 2, (255, 255, 255), 1)

            if name.startswith("anchor:"):
                cv2.line(ref_rgb, (ref_col - 8, ref_row), (ref_col + 8, ref_row), (0, 255, 255), 2)
                cv2.line(ref_rgb, (ref_col, ref_row - 8), (ref_col, ref_row + 8), (0, 255, 255), 2)
                cv2.line(off_rgb, (off_col - 8, off_row), (off_col + 8, off_row), (0, 255, 255), 2)
                cv2.line(off_rgb, (off_col, off_row - 8), (off_col, off_row + 8), (0, 255, 255), 2)

        h_ref, w_ref = ref_rgb.shape[:2]
        h_off, w_off = off_rgb.shape[:2]
        h_max = max(h_ref, h_off)

        ref_scale = 1.0
        off_scale = 1.0
        if h_ref != h_max:
            ref_scale = h_max / h_ref
            ref_rgb = cv2.resize(ref_rgb, (int(w_ref * ref_scale), h_max))
        if h_off != h_max:
            off_scale = h_max / h_off
            off_rgb = cv2.resize(off_rgb, (int(w_off * off_scale), h_max))

        combined = np.hstack([ref_rgb, off_rgb])

        ref_w_final = ref_rgb.shape[1]
        for rc, rr, oc, orr in match_coords:
            pt1 = (int(rc * ref_scale), int(rr * ref_scale))
            pt2 = (int(oc * off_scale) + ref_w_final, int(orr * off_scale))
            cv2.line(combined, pt1, pt2, (0, 255, 0), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Reference", (10, 30), font, 1.0, (255, 255, 255), 2)
        cv2.putText(combined, "Target (offset)", (ref_rgb.shape[1] + 10, 30),
                    font, 1.0, (255, 255, 255), 2)
        cv2.putText(combined,
                    f"Matches: {len(matched_pairs)} | Mean residual: {mean_residual:.1f}m",
                    (10, combined.shape[0] - 20), font, 0.7, (255, 255, 255), 2)

        legend_y = 60
        cv2.circle(combined, (15, legend_y), 5, (0, 255, 255), -1)
        cv2.putText(combined, "Anchor GCP", (25, legend_y + 5), font, 0.5, (255, 255, 255), 1)
        cv2.circle(combined, (15, legend_y + 25), 3, (0, 255, 0), -1)
        cv2.putText(combined, "Good match", (25, legend_y + 30), font, 0.5, (255, 255, 255), 1)
        cv2.circle(combined, (15, legend_y + 50), 3, (0, 165, 255), -1)
        cv2.putText(combined, "Higher residual", (25, legend_y + 55), font, 0.5, (255, 255, 255), 1)
        cv2.line(combined, (5, legend_y + 75), (25, legend_y + 75), (0, 255, 0), 1)
        cv2.putText(combined, "Match connection", (30, legend_y + 80), font, 0.5, (255, 255, 255), 1)

        cv2.imwrite(output_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"  Diagnostic image saved: {output_path}")
    except Exception as e:
        print(f"  WARNING: Could not create diagnostic visualization: {e}")


# ---------------------------------------------------------------------------
# Holdout validation and QA report building (merged from qa_runner.py)
# ---------------------------------------------------------------------------

def split_holdout_pairs(matched_pairs: Iterable[MatchPair], holdout_fraction: float = 0.2,
                        min_holdout: int = 4, seed: int = 42):
    """Split match pairs into training and holdout sets."""
    pairs = list(matched_pairs)
    auto_idx = [idx for idx, pair in enumerate(pairs) if not pair.is_anchor]
    if len(auto_idx) < max(min_holdout + 6, 12):
        return pairs, []

    holdout_n = max(min_holdout, int(round(len(auto_idx) * holdout_fraction)))
    holdout_n = min(holdout_n, max(0, len(auto_idx) - 6))
    if holdout_n <= 0:
        return pairs, []

    rng = np.random.default_rng(seed)
    shuffled = np.array(auto_idx, dtype=np.int32)
    rng.shuffle(shuffled)
    holdout_idx = set(int(idx) for idx in shuffled[:holdout_n])
    train = [pair for idx, pair in enumerate(pairs) if idx not in holdout_idx]
    holdout = [pair for idx, pair in enumerate(pairs) if idx in holdout_idx]
    return train, holdout


def compute_holdout_affine_metrics(M_geo, holdout_pairs: Iterable[MatchPair]):
    """Return independent residual stats for held-out matches."""
    if M_geo is None:
        return {}

    holdout = list(holdout_pairs)
    if not holdout:
        return {}

    errors = []
    for pair in holdout:
        pred_x = M_geo[0, 0] * pair.off_x + M_geo[0, 1] * pair.off_y + M_geo[0, 2]
        pred_y = M_geo[1, 0] * pair.off_x + M_geo[1, 1] * pair.off_y + M_geo[1, 2]
        errors.append(float(np.hypot(pred_x - pair.ref_x, pred_y - pair.ref_y)))

    err_arr = np.array(errors, dtype=np.float32)
    return {
        "count": int(len(errors)),
        "mean_m": float(np.mean(err_arr)),
        "median_m": float(np.median(err_arr)),
        "p90_m": float(np.percentile(err_arr, 90)),
        "max_m": float(np.max(err_arr)),
    }


def compute_holdout_warp_metrics(output_path: str, reference_path: str,
                                 holdout_pairs: Iterable[MatchPair],
                                 work_crs, overlap, eval_res: float = 4.0):
    """Validate holdout pairs against the final warped output (not just affine)."""
    holdout = list(holdout_pairs)
    if not holdout or len(holdout) < 4:
        return {}

    try:
        with rasterio.open(reference_path) as src_ref, rasterio.open(output_path) as src_out:
            arr_ref, ref_transform = read_overlap_region(src_ref, overlap, work_crs, eval_res)
            arr_out, _ = read_overlap_region(src_out, overlap, work_crs, eval_res)
    except Exception:
        return {}

    valid = (arr_ref > 0) & (arr_out > 0)
    h, w = arr_ref.shape
    patch = 64
    half = patch // 2
    device = get_torch_device()

    ref_patches = []
    out_patches = []
    win = np.outer(np.hanning(patch), np.hanning(patch)).astype(np.float32)

    for pair in holdout:
        col, row = ~ref_transform * (pair.ref_x, pair.ref_y)
        row, col = int(round(row)), int(round(col))
        r0, r1 = row - half, row + half
        c0, c1 = col - half, col + half
        if r0 < 0 or r1 > h or c0 < 0 or c1 > w:
            continue
        if np.mean(valid[r0:r1, c0:c1]) < 0.5:
            continue
        ref_patches.append(arr_ref[r0:r1, c0:c1].astype(np.float32) * win)
        out_patches.append(arr_out[r0:r1, c0:c1].astype(np.float32) * win)

    if len(ref_patches) < 4:
        return {}

    ref_t = torch.from_numpy(np.stack(ref_patches)).to(device)
    out_t = torch.from_numpy(np.stack(out_patches)).to(device)

    with torch.no_grad():
        shifts, _, vmask = _batch_phase_correlate(ref_t, out_t, 0.02)

    mags = np.hypot(shifts[vmask, 0], shifts[vmask, 1]) * eval_res

    if len(mags) < 2:
        return {}

    return {
        "count": int(len(mags)),
        "mean_m": float(np.mean(mags)),
        "median_m": float(np.median(mags)),
        "p90_m": float(np.percentile(mags, 90)),
        "max_m": float(np.max(mags)),
        "method": "post_warp_phase_correlation",
    }


def _effective_holdout_metrics(holdout_metrics):
    if not holdout_metrics:
        return {}
    post_warp = holdout_metrics.get("post_warp")
    if isinstance(post_warp, dict) and post_warp:
        return post_warp
    return holdout_metrics


def _confidence_from_metrics(image_metrics, holdout_metrics, coverage, cv_mean_m):
    components = []
    if image_metrics and "score" in image_metrics:
        image_score = float(image_metrics["score"])
        components.append(np.clip(1.0 - (image_score / 200.0), 0.0, 1.0))
    effective_holdout = _effective_holdout_metrics(holdout_metrics)
    if effective_holdout:
        components.append(np.clip(1.0 - (float(effective_holdout.get("mean_m", 200.0)) / 120.0), 0.0, 1.0))
        components.append(np.clip(1.0 - (float(effective_holdout.get("p90_m", 240.0)) / 180.0), 0.0, 1.0))
    if cv_mean_m is not None:
        components.append(np.clip(1.0 - (float(cv_mean_m) / 120.0), 0.0, 1.0))
    components.append(np.clip(float(coverage) / 0.35, 0.0, 1.0))
    return float(np.mean(components)) if components else 0.0


def _total_score(image_metrics, holdout_metrics, coverage, cv_mean_m):
    score = float(image_metrics["score"]) if image_metrics and "score" in image_metrics else 0.0
    effective_holdout = _effective_holdout_metrics(holdout_metrics)
    if effective_holdout:
        score += 0.60 * float(effective_holdout.get("mean_m", 0.0))
        score += 0.25 * float(effective_holdout.get("p90_m", 0.0))
    if cv_mean_m is not None:
        score += 0.35 * float(cv_mean_m)
    if coverage < 0.12:
        score += (0.12 - coverage) * 400.0
    return float(score)


def _compute_quality_grade(image_score, holdout_median, cv_mean, qa_params=None):
    """Compute quality grade A/B/C/D from metrics and thresholds."""
    from .params import get_params
    qp = qa_params or get_params().qa
    score = None if image_score is None else float(image_score)
    holdout = None if holdout_median is None else float(holdout_median)
    cv = None if cv_mean is None else float(cv_mean)
    if holdout is None:
        return "D"
    if score is None:
        if holdout <= qp.grade_a_holdout and (cv is None or cv <= qp.grade_a_holdout):
            return "A"
        if holdout <= qp.grade_b_holdout and (cv is None or cv <= qp.grade_b_holdout):
            return "B"
        if holdout <= qp.accept_holdout_median_max and (cv is None or cv <= qp.accept_cv_mean_max):
            return "C"
        return "D"
    if score <= qp.grade_a_score and holdout <= qp.grade_a_holdout and (cv is None or cv <= qp.grade_a_holdout):
        return "A"
    if score <= qp.grade_b_score and holdout <= qp.grade_b_holdout and (cv is None or cv <= qp.grade_b_holdout):
        return "B"
    if score <= qp.accept_image_score_max and holdout <= qp.accept_holdout_median_max and (cv is None or cv <= qp.accept_cv_mean_max):
        return "C"
    return "D"


def build_candidate_report(candidate_name: str, output_path: str, reference_path: str,
                           overlap, work_crs, *, holdout_pairs=None, M_geo=None,
                           coverage: float = 0.0, cv_mean_m: float | None = None,
                           hypothesis_id: str = "", eval_res: float = 4.0,
                           image_metrics=None, qa_params=None) -> QaReport:
    """Compute an independent QA report for a candidate output."""
    from .params import get_params
    qp = qa_params or get_params().qa

    if image_metrics is None:
        image_metrics = evaluate_alignment_quality_paths(
            output_path, reference_path, overlap, work_crs, eval_res=eval_res,
        ) or {}
    holdout_metrics = compute_holdout_affine_metrics(M_geo, holdout_pairs or [])
    holdout_warp_metrics = compute_holdout_warp_metrics(
        output_path, reference_path, holdout_pairs or [],
        work_crs, overlap, eval_res=eval_res,
    )
    if holdout_warp_metrics:
        holdout_metrics["post_warp"] = holdout_warp_metrics
    effective_holdout = _effective_holdout_metrics(holdout_metrics)
    total = _total_score(image_metrics, holdout_metrics, coverage, cv_mean_m)
    confidence = _confidence_from_metrics(image_metrics, holdout_metrics, coverage, cv_mean_m)

    image_score = float(image_metrics["score"]) if image_metrics and "score" in image_metrics else None
    holdout_median = float(effective_holdout.get("median_m")) if effective_holdout else None
    geometry_only = not image_metrics or "score" not in image_metrics

    reasons = []
    accepted = True
    if not geometry_only and image_score is not None and image_score > qp.accept_image_score_max:
        accepted = False
        reasons.append("image_alignment_score_high")
    if holdout_median is not None and holdout_median > qp.accept_holdout_median_max:
        accepted = False
        reasons.append("holdout_residual_high")
    if cv_mean_m is not None and float(cv_mean_m) > qp.accept_cv_mean_max:
        accepted = False
        reasons.append("cross_validation_high")
    if not geometry_only and coverage < qp.accept_coverage_min:
        accepted = False
        reasons.append("gcp_coverage_low")
    if holdout_warp_metrics and holdout_metrics:
        affine_mean = float(holdout_metrics.get("mean_m", 0.0))
        warp_mean = float(holdout_warp_metrics.get("mean_m", 0.0))
        affine_p90 = float(holdout_metrics.get("p90_m", 0.0))
        warp_p90 = float(holdout_warp_metrics.get("p90_m", 0.0))
        if (
            affine_mean > 0.0 and warp_mean > affine_mean * 1.5
        ) or (
            affine_p90 > 0.0 and warp_p90 > affine_p90 * 1.5
        ):
            accepted = False
            reasons.append("post_warp_holdout_regression")

    grade = _compute_quality_grade(image_score, holdout_median, cv_mean_m, qp)

    return QaReport(
        candidate=candidate_name,
        output_path=output_path,
        total_score=total,
        confidence=confidence,
        accepted=accepted,
        image_metrics=image_metrics,
        holdout_metrics=holdout_metrics,
        coverage=float(coverage),
        cv_mean_m=None if cv_mean_m is None else float(cv_mean_m),
        hypothesis_id=hypothesis_id,
        reasons=reasons,
        quality_grade=grade,
    )


def write_qa_report(path: str, reports: list[QaReport], *, selected_candidate: str | None = None,
                    metadata: dict | None = None):
    """Persist QA reports to JSON."""
    payload = {
        "selected_candidate": selected_candidate,
        "metadata": metadata or {},
        "reports": [report.to_dict() for report in reports],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
