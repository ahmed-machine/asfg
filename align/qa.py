"""Alignment quality metrics and diagnostic visualization."""

import cv2
import numpy as np
import rasterio
import rasterio.transform
import torch

from .geo import read_overlap_region
from .image import build_semantic_masks, shoreline_mask, to_u8
from .models import get_torch_device

_QA_GRID_ROWS = 4
_QA_GRID_COLS = 6
_MIN_CELL_EDGE_PX = 20
_MIN_CELL_PATCHES = 2
_MIN_IOU_UNION_PX = 100


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
        return iou, float("inf")

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
        return {"median": float("inf"), "p90": float("inf"), "max": float("inf"), "count": 0}

    # Batch phase correlation on GPU/MPS
    device = get_torch_device()
    ref_t = torch.from_numpy(np.stack(ref_patches)).to(device)
    test_t = torch.from_numpy(np.stack(test_patches)).to(device)

    with torch.no_grad():
        shifts, responses, valid = _batch_phase_correlate(ref_t, test_t, min_resp)

    mags = np.hypot(shifts[valid, 0], shifts[valid, 1]) * res_m

    if len(mags) == 0:
        return {"median": float("inf"), "p90": float("inf"), "max": float("inf"), "count": 0}
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
        grid_score = float("inf")

    # Derive legacy directional keys from grid
    west, center, east, north_shift = _derive_legacy_from_grid(cells, n_cols)

    # IoU metrics
    stable_ref = (bundle_ref.stable > 0) & valid_mask
    stable_out = (bundle_out.stable > 0) & valid_mask
    stable_union = int(np.sum(stable_ref | stable_out))
    stable_iou = float(np.sum(stable_ref & stable_out) / stable_union) if stable_union > 0 else 0.0
    shore_union = int(np.sum(m_ref | m_out))
    shore_iou = float(np.sum(m_ref & m_out) / shore_union) if shore_union > 0 else 0.0

    # Patch residual
    patch_result = compute_patch_residual_median(
        to_u8(ref_arr),
        to_u8(out_arr),
        valid_mask & (stable_ref | stable_out),
        eval_res,
        patch=192,
        stride=96,
    )
    patch_med = patch_result["median"]

    # No valid grid cells and no patches — nothing to score
    if valid_count == 0 and patch_result["count"] == 0:
        return None

    # Score formula
    grid_contrib = 0.55 * grid_score if valid_count > 0 else 0.0
    patch_contrib = 0.25 * patch_med

    # IoU penalties — drop when union is too small for meaningful comparison
    stable_iou_penalty = 18.0 * (1.0 - stable_iou) if stable_union >= _MIN_IOU_UNION_PX else 0.0
    shore_iou_penalty = 12.0 * (1.0 - shore_iou) if shore_union >= _MIN_IOU_UNION_PX else 0.0

    score = grid_contrib + patch_contrib + stable_iou_penalty + shore_iou_penalty

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
        # Patch metrics (unchanged)
        "patch_med": patch_med,
        "patch_p90": patch_result["p90"],
        "patch_max": patch_result["max"],
        "patch_count": patch_result["count"],
        # IoU (unchanged)
        "stable_iou": stable_iou,
        "shore_iou": shore_iou,
        # Score
        "score": float(score),
        "score_breakdown": {
            "grid_contrib": round(float(grid_contrib), 2),
            "patch_contrib": round(float(patch_contrib), 2),
            "stable_iou_penalty": round(float(stable_iou_penalty), 2),
            "shore_iou_penalty": round(float(shore_iou_penalty), 2),
        },
    }


def evaluate_alignment_quality_paths(output_path, reference_path, overlap, work_crs, eval_res=8.0,
                                     mask_mode="coastal_obia"):
    """Compute alignment QA metrics directly from raster paths."""
    src_ref = rasterio.open(reference_path)
    src_out = rasterio.open(output_path)
    try:
        arr_ref, _ = read_overlap_region(src_ref, overlap, work_crs, eval_res)
        arr_out, _ = read_overlap_region(src_out, overlap, work_crs, eval_res)
    finally:
        src_ref.close()
        src_out.close()

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
            rgx, rgy, ogx, ogy = pair[0], pair[1], pair[2], pair[3]
            name = pair[5]
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
