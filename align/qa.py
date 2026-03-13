"""Shared alignment quality metrics used across pipeline stages."""

import cv2
import numpy as np
import rasterio

from .geo import read_overlap_region
from .image import build_semantic_masks, shoreline_mask, to_u8


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


def compute_patch_residual_median(ref_arr, test_arr, valid_mask, res_m,
                                  patch=160, stride=160,
                                  min_valid=0.6, min_resp=0.03):
    """Return median local residual shift (meters) via phase correlation."""
    h, w = ref_arr.shape
    win = np.outer(np.hanning(patch), np.hanning(patch)).astype(np.float32)
    mags = []

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
            local_win = win
            if a.shape != win.shape:
                local_win = np.outer(
                    np.hanning(a.shape[0]), np.hanning(a.shape[1])
                ).astype(np.float32)

            try:
                shift, resp = cv2.phaseCorrelate(a * local_win, b * local_win)
            except Exception:
                continue
            if resp < min_resp:
                continue
            mags.append(float(np.hypot(shift[0], shift[1]) * res_m))

    if not mags:
        return {"median": float("inf"), "p90": float("inf"), "max": float("inf"), "count": 0}
    arr = np.array(mags, dtype=np.float32)
    return {
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "count": int(len(arr)),
    }


def evaluate_alignment_quality_arrays(ref_arr, out_arr, valid_mask, eval_res=8.0, mask_mode="coastal_obia"):
    """Compute regional shoreline drift and patch residual metrics."""
    bundle_ref = build_semantic_masks(ref_arr, mode=mask_mode)
    bundle_out = build_semantic_masks(out_arr, mode=mask_mode)
    m_ref = (bundle_ref.shoreline > 0) & valid_mask
    m_out = (bundle_out.shoreline > 0) & valid_mask
    k = np.ones((3, 3), np.uint8)
    e_ref = cv2.morphologyEx(m_ref.astype(np.uint8), cv2.MORPH_GRADIENT, k) > 0
    e_out = cv2.morphologyEx(m_out.astype(np.uint8), cv2.MORPH_GRADIENT, k) > 0
    common = e_ref | e_out
    if not np.any(common):
        return None

    d_ref = cv2.distanceTransform((~e_ref).astype(np.uint8), cv2.DIST_L2, 3)
    d_out = cv2.distanceTransform((~e_out).astype(np.uint8), cv2.DIST_L2, 3)
    sym = np.zeros_like(d_ref, dtype=np.float32)
    sym[common] = 0.5 * (d_ref[common] + d_out[common]) * eval_res

    h, w = sym.shape
    west_vals = sym[:, :int(w * 0.33)][common[:, :int(w * 0.33)]]
    center_vals = sym[:, int(w * 0.33):int(w * 0.66)][common[:, int(w * 0.33):int(w * 0.66)]]
    east_vals = sym[:, int(w * 0.66):][common[:, int(w * 0.66):]]

    def _north_profile(mask):
        prof = np.full(mask.shape[1], np.nan)
        for c in range(mask.shape[1]):
            rows = np.where(mask[:, c])[0]
            if rows.size:
                prof[c] = rows.min()
        return prof

    pr = _north_profile((bundle_ref.land > 0) & valid_mask)
    po = _north_profile((bundle_out.land > 0) & valid_mask)
    valid_cols = ~np.isnan(pr) & ~np.isnan(po)
    north_shift = np.median((po[valid_cols] - pr[valid_cols]) * eval_res) if np.any(valid_cols) else np.inf

    stable_ref = (bundle_ref.stable > 0) & valid_mask
    stable_out = (bundle_out.stable > 0) & valid_mask
    stable_union = np.sum(stable_ref | stable_out)
    stable_iou = float(np.sum(stable_ref & stable_out) / stable_union) if stable_union > 0 else 0.0
    shore_union = np.sum(m_ref | m_out)
    shore_iou = float(np.sum(m_ref & m_out) / shore_union) if shore_union > 0 else 0.0

    patch_result = compute_patch_residual_median(
        to_u8(ref_arr),
        to_u8(out_arr),
        valid_mask & (stable_ref | stable_out),
        eval_res,
        patch=192,
        stride=96,
    )
    patch_med = patch_result["median"]

    west = float(np.median(west_vals)) if west_vals.size else np.inf
    center = float(np.median(center_vals)) if center_vals.size else np.inf
    east = float(np.median(east_vals)) if east_vals.size else np.inf

    capped_north = min(abs(float(north_shift)), 150.0)

    west_contrib = 0.22 * west
    center_contrib = 0.28 * center
    east_contrib = 0.15 * east
    north_contrib = 0.15 * capped_north
    patch_contrib = 0.20 * patch_med
    stable_iou_penalty = 18.0 * (1.0 - stable_iou)
    shore_iou_penalty = 12.0 * (1.0 - shore_iou)

    score = (
        west_contrib + center_contrib + east_contrib
        + north_contrib + patch_contrib
        + stable_iou_penalty + shore_iou_penalty
    )

    return {
        "west": west,
        "center": center,
        "east": east,
        "north_shift": float(north_shift),
        "patch_med": patch_med,
        "patch_p90": patch_result["p90"],
        "patch_max": patch_result["max"],
        "patch_count": patch_result["count"],
        "stable_iou": stable_iou,
        "shore_iou": shore_iou,
        "score": float(score),
        "score_breakdown": {
            "west_contrib": round(float(west_contrib), 2),
            "center_contrib": round(float(center_contrib), 2),
            "east_contrib": round(float(east_contrib), 2),
            "north_contrib": round(float(north_contrib), 2),
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
