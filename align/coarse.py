"""Coarse offset detection via semantic-weighted template matching."""

import os

import cv2
import numpy as np

from .geo import read_overlap_region
from .image import class_weight_map, make_land_mask, stable_feature_mask, to_u8


def _save_coarse_diagnostic(template, search, result_map, max_loc, max_val,
                            base_r, base_c, tr0, tc0, res, dx_m, dy_m,
                            diagnostics_dir, label):
    """Save coarse offset diagnostic: template, search region, correlation heatmap."""
    # Normalize images to uint8 for display
    def _to_vis(arr):
        if arr.max() == 0:
            return np.zeros_like(arr, dtype=np.uint8)
        return (arr / max(arr.max(), 1e-6) * 255).clip(0, 255).astype(np.uint8)

    tmpl_vis = _to_vis(template.astype(np.float32))
    search_vis = _to_vis(search.astype(np.float32))

    # Resize template to match search height for side-by-side
    th, tw = tmpl_vis.shape
    sh, sw = search_vis.shape
    scale = sh / th if th > 0 else 1.0
    tmpl_resized = cv2.resize(tmpl_vis, (max(1, int(tw * scale)), sh))
    trw = tmpl_resized.shape[1]

    # Correlation heatmap
    corr_norm = ((result_map - result_map.min()) /
                 max(result_map.max() - result_map.min(), 1e-6) * 255).astype(np.uint8)
    corr_color = cv2.applyColorMap(corr_norm, cv2.COLORMAP_JET)
    corr_resized = cv2.resize(corr_color, (sw, sh))

    # Canvas: template | search | correlation heatmap
    canvas_w = trw + sw + sw
    canvas = np.zeros((sh, canvas_w, 3), dtype=np.uint8)
    canvas[:, :trw] = cv2.cvtColor(tmpl_resized, cv2.COLOR_GRAY2BGR)
    canvas[:, trw:trw + sw] = cv2.cvtColor(search_vis, cv2.COLOR_GRAY2BGR)
    canvas[:, trw + sw:] = corr_resized

    # Draw crosshair at best match on search region
    match_x = max_loc[0]
    match_y = max_loc[1]
    cx = trw + match_x + tw // 2
    cy = match_y + th // 2
    cv2.drawMarker(canvas, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

    # Draw match rectangle on search
    cv2.rectangle(canvas,
                  (trw + match_x, match_y),
                  (trw + match_x + tw, match_y + th),
                  (0, 255, 0), 1)

    # Labels
    cv2.putText(canvas, "template", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "search", (trw + 4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "correlation", (trw + sw + 4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    # Offset text
    text = f"dx={dx_m:.1f}m dy={dy_m:.1f}m corr={max_val:.3f} res={res:.0f}m"
    cv2.putText(canvas, text, (4, sh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    os.makedirs(diagnostics_dir, exist_ok=True)
    out_path = os.path.join(diagnostics_dir, f"coarse_{label}.jpg")
    cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"  Saved diagnostic: {os.path.basename(out_path)}")


def detect_offset_at_resolution(src_offset, src_ref, overlap, work_crs, res,
                                template_radius_m=6000, coarse_offset=None,
                                search_margin_m=None, mask_mode="coastal_obia",
                                diagnostics_dir=None):
    """Detect the offset between two images at a given resolution using
    template matching on binary land masks.

    Returns (dx_m, dy_m, correlation) where dx/dy are the offset of the
    misaligned image in metres (east, south).
    """
    arr_offset, _ = read_overlap_region(src_offset, overlap, work_crs, res)
    arr_ref, _ = read_overlap_region(src_ref, overlap, work_crs, res)

    land_offset = make_land_mask(arr_offset, mode=mask_mode)
    land_ref = make_land_mask(arr_ref, mode=mask_mode)
    stable_ref = stable_feature_mask(arr_ref, mode=mask_mode)
    weight_offset = class_weight_map(arr_offset, mode=mask_mode)
    weight_ref = class_weight_map(arr_ref, mode=mask_mode)

    search_img = weight_offset if np.mean(weight_offset > 0) > 0.02 else land_offset
    template_img = weight_ref if np.mean(weight_ref > 0) > 0.02 else land_ref
    ref_seed = stable_ref if np.mean(stable_ref > 0) > 0.01 else land_ref

    # Find a robust semantic centroid for template extraction.
    ref_rows = np.any(ref_seed > 0, axis=1)
    ref_cols = np.any(ref_seed > 0, axis=0)
    if not ref_rows.any() or not ref_cols.any():
        return None, None, 0
    rr, cc = np.where(ref_seed > 0)
    ww = weight_ref[rr, cc] if len(rr) else np.array([])
    if len(rr) and np.sum(ww) > 0:
        r_center = int(np.average(rr, weights=ww))
        c_center = int(np.average(cc, weights=ww))
    else:
        r_center = int(np.mean(np.where(ref_rows)[0]))
        c_center = int(np.mean(np.where(ref_cols)[0]))

    # Extract template from the semantic center.
    th = int(template_radius_m / res)
    tr0 = max(0, r_center - th)
    tr1 = min(land_ref.shape[0], r_center + th)
    tc0 = max(0, c_center - th)
    tc1 = min(land_ref.shape[1], c_center + th)
    template = template_img[tr0:tr1, tc0:tc1]

    if template.shape[0] < 10 or template.shape[1] < 10:
        return None, None, 0

    restricted = False
    if coarse_offset and search_margin_m:
        coarse_dx_px = int(round(coarse_offset[0] / res))
        coarse_dy_px = int(round(coarse_offset[1] / res))
        margin_px = int(search_margin_m / res)
        s_c0 = max(0, tc0 + coarse_dx_px - margin_px)
        s_c1 = min(land_offset.shape[1], tc0 + template.shape[1] + coarse_dx_px + margin_px)
        s_r0 = max(0, tr0 + coarse_dy_px - margin_px)
        s_r1 = min(land_offset.shape[0], tr0 + template.shape[0] + coarse_dy_px + margin_px)
        search = search_img[s_r0:s_r1, s_c0:s_c1]
        base_c, base_r = s_c0, s_r0
    else:
        search = search_img
        base_c, base_r = 0, 0

        # Restrict search to valid-data region — prevents spurious matches
        # when a long strip (e.g. 270km KH-4) overlaps a small reference
        valid_rows = np.any(search > 0, axis=1)
        valid_cols = np.any(search > 0, axis=0)
        restricted = False
        if valid_rows.any() and valid_cols.any():
            r_idxs = np.where(valid_rows)[0]
            c_idxs = np.where(valid_cols)[0]
            pad = max(template.shape[0], template.shape[1])
            r0 = max(0, r_idxs[0] - pad)
            r1 = min(search.shape[0], r_idxs[-1] + pad + 1)
            c0 = max(0, c_idxs[0] - pad)
            c1 = min(search.shape[1], c_idxs[-1] + pad + 1)
            # Only restrict if the crop is meaningfully smaller
            if (r1 - r0) < search.shape[0] * 0.95 or (c1 - c0) < search.shape[1] * 0.95:
                search = search[r0:r1, c0:c1]
                base_c, base_r = c0, r0
                restricted = True

    if search.shape[0] <= template.shape[0] or search.shape[1] <= template.shape[1]:
        return None, None, 0

    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # If restricted search gave a poor result, retry with full search area
    if max_val < 0.3 and restricted:
        search = search_img
        base_c, base_r = 0, 0
        if search.shape[0] > template.shape[0] and search.shape[1] > template.shape[1]:
            result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < 0.3:
        print(f"  [coarse] NCC too low: max_val={max_val:.4f} (threshold=0.3)")
        return None, None, 0

    # Sub-pixel refinement via parabolic interpolation
    pr, pc = max_loc[1], max_loc[0]
    sub_x, sub_y = float(pc), float(pr)
    if 0 < pr < result.shape[0] - 1 and 0 < pc < result.shape[1] - 1:
        vc = result[pr, pc]
        vl = result[pr, pc - 1]
        vr = result[pr, pc + 1]
        denom_x = vl - 2 * vc + vr
        if abs(denom_x) > 1e-10:
            sub_x = pc + 0.5 * (vl - vr) / denom_x

        vt = result[pr - 1, pc]
        vb = result[pr + 1, pc]
        denom_y = vt - 2 * vc + vb
        if abs(denom_y) > 1e-10:
            sub_y = pr + 0.5 * (vt - vb) / denom_y

    # Convert match position to geographic offset
    actual_c = base_c + sub_x
    actual_r = base_r + sub_y
    dx_m = (actual_c - tc0) * res
    dy_m = (actual_r - tr0) * res

    if diagnostics_dir is not None:
        label = f"{res:.0f}m" + ("_refined" if coarse_offset else "_coarse")
        _save_coarse_diagnostic(template, search, result, max_loc, max_val,
                                base_r, base_c, tr0, tc0, res, dx_m, dy_m,
                                diagnostics_dir, label)

    return dx_m, dy_m, max_val


def _nmi_offset(arr_offset, arr_ref, template_img, search_img,
                tr0, tr1, tc0, tc1, base_r, base_c, res,
                coarse_offset=None, search_margin_m=None):
    """Compute offset using Normalized Mutual Information template matching.

    NMI captures non-linear intensity relationships between cross-temporal
    satellite imagery, unlike NCC which assumes linear correlation.
    """
    # Use grayscale images (not land masks) for MI — MI handles radiometric diffs
    ref_u8 = to_u8(arr_ref).astype(np.float32)
    off_u8 = to_u8(arr_offset).astype(np.float32)

    template = ref_u8[tr0:tr1, tc0:tc1]
    if template.shape[0] < 10 or template.shape[1] < 10:
        return None, None, 0

    if coarse_offset and search_margin_m:
        coarse_dx_px = int(round(coarse_offset[0] / res))
        coarse_dy_px = int(round(coarse_offset[1] / res))
        margin_px = int(search_margin_m / res)
        s_c0 = max(0, tc0 + coarse_dx_px - margin_px)
        s_c1 = min(off_u8.shape[1], tc0 + template.shape[1] + coarse_dx_px + margin_px)
        s_r0 = max(0, tr0 + coarse_dy_px - margin_px)
        s_r1 = min(off_u8.shape[0], tr0 + template.shape[0] + coarse_dy_px + margin_px)
        search = off_u8[s_r0:s_r1, s_c0:s_c1]
        b_c, b_r = s_c0, s_r0
    else:
        search = off_u8
        b_c, b_r = 0, 0

    if search.shape[0] <= template.shape[0] or search.shape[1] <= template.shape[1]:
        return None, None, 0

    # Compute NMI via sliding window (using joint histogram approach)
    # For efficiency, downsample both to ~32m resolution
    ds_factor = max(1, int(res / 32))
    if ds_factor < 1:
        ds_factor = 1

    n_bins = 32
    th, tw = template.shape

    # Use NCC as a fast proxy for MI search region, then compute MI around the peak
    # This avoids the O(n^4) full-search MI computation
    result_ncc = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc_ncc = cv2.minMaxLoc(result_ncc)

    # Search a ±20px window around NCC peak for MI maximum
    mi_search_rad = 20
    pc_ncc, pr_ncc = max_loc_ncc
    best_mi = -1.0
    best_pos = (pc_ncc, pr_ncc)

    for dy in range(-mi_search_rad, mi_search_rad + 1, 2):
        for dx in range(-mi_search_rad, mi_search_rad + 1, 2):
            sy = pr_ncc + dy
            sx = pc_ncc + dx
            if sy < 0 or sy + th > search.shape[0] or sx < 0 or sx + tw > search.shape[1]:
                continue
            patch = search[sy:sy+th, sx:sx+tw]

            # Compute NMI
            t_valid = template > 0
            p_valid = patch > 0
            valid = t_valid & p_valid
            if np.sum(valid) < 100:
                continue

            t_vals = template[valid]
            p_vals = patch[valid]

            # Joint histogram with Parzen (Gaussian) smoothing
            hist_joint, _, _ = np.histogram2d(
                t_vals, p_vals, bins=n_bins, range=[[0, 255], [0, 255]])
            hist_joint = cv2.GaussianBlur(hist_joint.astype(np.float32), (3, 3), 0.5)
            hist_joint = hist_joint / (hist_joint.sum() + 1e-10)

            hist_t = hist_joint.sum(axis=1)
            hist_p = hist_joint.sum(axis=0)

            # H(T) + H(P) - H(T,P) = MI;  NMI = 2*MI / (H(T)+H(P))
            eps = 1e-10
            h_t = -np.sum(hist_t * np.log(hist_t + eps))
            h_p = -np.sum(hist_p * np.log(hist_p + eps))
            h_tp = -np.sum(hist_joint * np.log(hist_joint + eps))
            mi = h_t + h_p - h_tp
            nmi = 2.0 * mi / (h_t + h_p + eps) if (h_t + h_p) > eps else 0.0

            if nmi > best_mi:
                best_mi = nmi
                best_pos = (sx, sy)

    # Convert to geographic offset
    actual_c = b_c + best_pos[0]
    actual_r = b_r + best_pos[1]
    nmi_dx = (actual_c - tc0) * res
    nmi_dy = (actual_r - tr0) * res

    return nmi_dx, nmi_dy, best_mi
