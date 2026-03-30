"""Coarse offset detection via semantic-weighted template matching.

Fallback chain (tried in order until one succeeds):
1. Semantic-weighted NCC on binary land masks (fast, works for coastal areas).
2. CLAHE-normalized grayscale NCC (works on texture-rich areas like deserts).
3. NMI (Normalized Mutual Information) — handles non-linear radiometric
   differences between historical film and modern satellite imagery.
"""

import os
from dataclasses import dataclass

import cv2
import numpy as np

from .geo import read_overlap_region
from .image import to_u8
from .semantic_masking import build_semantic_masks


@dataclass
class CoarseResult:
    """Result from a single coarse offset detection method."""
    dx_m: float
    dy_m: float
    confidence: float
    method: str

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


def _bundle_weights(bundle):
    """Compute class weight map from a MaskBundle."""
    w = np.zeros_like(bundle.land, dtype=np.float32)
    w += 1.25 * bundle.stable
    w += 0.75 * bundle.shoreline
    w += 0.35 * bundle.dark_farmland
    w -= 0.80 * bundle.shallow_water
    return np.clip(w, 0.0, 1.5)


def _build_search_region(off_img, template_shape, tc0, tr0, res,
                         coarse_offset=None, search_margin_m=None):
    """Extract the search region from an offset image, optionally restricted by a coarse offset.

    Returns (search_array, base_col, base_row).
    """
    if coarse_offset and search_margin_m:
        coarse_dx_px = int(round(coarse_offset[0] / res))
        coarse_dy_px = int(round(coarse_offset[1] / res))
        margin_px = int(search_margin_m / res)
        s_c0 = max(0, tc0 + coarse_dx_px - margin_px)
        s_c1 = min(off_img.shape[1], tc0 + template_shape[1] + coarse_dx_px + margin_px)
        s_r0 = max(0, tr0 + coarse_dy_px - margin_px)
        s_r1 = min(off_img.shape[0], tr0 + template_shape[0] + coarse_dy_px + margin_px)
        search = off_img[s_r0:s_r1, s_c0:s_c1]
        return search, s_c0, s_r0
    else:
        return off_img, 0, 0


def _subpixel_refine(result, max_loc):
    """Sub-pixel refinement via parabolic interpolation on a correlation map.

    Returns (sub_x, sub_y) as float coordinates.
    """
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
    return sub_x, sub_y


def _match_to_offset(base_c, base_r, sub_x, sub_y, tc0, tr0, res):
    """Convert a match position to geographic offset in metres."""
    actual_c = base_c + sub_x
    actual_r = base_r + sub_y
    dx_m = (actual_c - tc0) * res
    dy_m = (actual_r - tr0) * res
    return dx_m, dy_m


def detect_offset_at_resolution(src_offset, src_ref, overlap, work_crs, res,
                                template_radius_m=6000, coarse_offset=None,
                                search_margin_m=None, mask_mode="coastal_obia",
                                diagnostics_dir=None, min_ncc=0.3):
    """Detect the offset between two images at a given resolution using
    template matching on binary land masks.

    Returns (dx_m, dy_m, correlation) where dx/dy are the offset of the
    misaligned image in metres (east, south).
    """
    arr_offset, _ = read_overlap_region(src_offset, overlap, work_crs, res)
    arr_ref, _ = read_overlap_region(src_ref, overlap, work_crs, res)

    bundle_offset = build_semantic_masks(arr_offset, mode=mask_mode)
    bundle_ref = build_semantic_masks(arr_ref, mode=mask_mode)

    land_offset = bundle_offset.land
    land_ref = bundle_ref.land
    stable_ref = bundle_ref.stable
    weight_offset = _bundle_weights(bundle_offset)
    weight_ref = _bundle_weights(bundle_ref)

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

    # Shared context passed to each fallback method
    ctx = dict(
        arr_offset=arr_offset, arr_ref=arr_ref,
        search_img=search_img, template=template,
        tr0=tr0, tr1=tr1, tc0=tc0, tc1=tc1, res=res,
        coarse_offset=coarse_offset, search_margin_m=search_margin_m,
    )

    # Try each method in order until one succeeds
    methods = [
        ("land_mask_ncc", _land_mask_ncc_offset),
        ("clahe_ncc",     _clahe_ncc_offset),
        ("nmi",           _nmi_offset),
    ]

    for name, method in methods:
        try:
            result = method(ctx, min_confidence=min_ncc)
        except Exception as e:
            print(f"  [coarse/{name}] failed: {e}", flush=True)
            continue

        if result is not None and result.confidence >= min_ncc:
            if diagnostics_dir is not None:
                suffix = "_refined" if coarse_offset else "_coarse"
                label = f"{res:.0f}m_{name}{suffix}"
                # Build a minimal diagnostic from the land-mask search for context
                search_diag, base_c_d, base_r_d = _build_search_region(
                    search_img, template.shape, tc0, tr0, res,
                    coarse_offset, search_margin_m)
                if search_diag.shape[0] > template.shape[0] and search_diag.shape[1] > template.shape[1]:
                    result_map = cv2.matchTemplate(search_diag, template, cv2.TM_CCOEFF_NORMED)
                    _, dv, _, dl = cv2.minMaxLoc(result_map)
                    _save_coarse_diagnostic(
                        template, search_diag, result_map, dl, dv,
                        base_r_d, base_c_d, tr0, tc0, res,
                        result.dx_m, result.dy_m, diagnostics_dir, label)
            return result.dx_m, result.dy_m, result.confidence

        if result is not None:
            print(f"  [coarse/{name}] confidence too low: {result.confidence:.4f} "
                  f"(threshold={min_ncc})", flush=True)
        else:
            print(f"  [coarse/{name}] returned no result", flush=True)

    print(f"  [coarse] All methods failed at {res:.0f}m/px", flush=True)
    return None, None, 0


def _land_mask_ncc_offset(ctx, min_confidence=0.3):
    """Primary method: NCC on semantic-weighted land masks.

    Uses the pre-built search_img (weight map or land mask) and template
    from the orchestrator. Handles restricted search regions for long
    strip imagery (e.g. KH-4).

    Returns CoarseResult or None.
    """
    search_img = ctx['search_img']
    template = ctx['template']
    tc0, tr0, res = ctx['tc0'], ctx['tr0'], ctx['res']
    coarse_offset = ctx['coarse_offset']
    search_margin_m = ctx['search_margin_m']

    search, base_c, base_r = _build_search_region(
        search_img, template.shape, tc0, tr0, res,
        coarse_offset, search_margin_m)

    # When no coarse offset is given, restrict search to valid-data region
    # to prevent spurious matches on long strips (e.g. 270km KH-4)
    restricted = False
    if not (coarse_offset and search_margin_m):
        valid_rows = np.any(search > 0, axis=1)
        valid_cols = np.any(search > 0, axis=0)
        if valid_rows.any() and valid_cols.any():
            r_idxs = np.where(valid_rows)[0]
            c_idxs = np.where(valid_cols)[0]
            pad = max(template.shape[0], template.shape[1])
            r0 = max(0, r_idxs[0] - pad)
            r1 = min(search.shape[0], r_idxs[-1] + pad + 1)
            c0 = max(0, c_idxs[0] - pad)
            c1 = min(search.shape[1], c_idxs[-1] + pad + 1)
            if (r1 - r0) < search.shape[0] * 0.95 or (c1 - c0) < search.shape[1] * 0.95:
                search = search[r0:r1, c0:c1]
                base_c, base_r = c0, r0
                restricted = True

    if search.shape[0] <= template.shape[0] or search.shape[1] <= template.shape[1]:
        return None

    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # If restricted search gave a poor result, retry with full search area
    if max_val < min_confidence and restricted:
        search = search_img
        base_c, base_r = 0, 0
        if search.shape[0] > template.shape[0] and search.shape[1] > template.shape[1]:
            result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < min_confidence:
        return CoarseResult(dx_m=0, dy_m=0, confidence=max_val, method="land_mask_ncc")

    sub_x, sub_y = _subpixel_refine(result, max_loc)
    dx_m, dy_m = _match_to_offset(base_c, base_r, sub_x, sub_y, tc0, tr0, res)
    return CoarseResult(dx_m=dx_m, dy_m=dy_m, confidence=max_val, method="land_mask_ncc")


def _clahe_ncc_offset(ctx, min_confidence=0.3):
    """Fallback: CLAHE-normalized grayscale NCC.

    When land-mask NCC fails (featureless desert, ice, etc.), direct
    grayscale correlation on CLAHE-normalized imagery can succeed because
    it captures texture differences invisible in binary masks.

    Returns CoarseResult or None.
    """
    from .image import clahe_normalize

    arr_offset, arr_ref = ctx['arr_offset'], ctx['arr_ref']
    tr0, tr1 = ctx['tr0'], ctx['tr1']
    tc0, tc1 = ctx['tc0'], ctx['tc1']
    res = ctx['res']

    ref_u8 = clahe_normalize(arr_ref).astype(np.float32)
    off_u8 = clahe_normalize(arr_offset).astype(np.float32)

    template = ref_u8[tr0:tr1, tc0:tc1]
    if template.shape[0] < 10 or template.shape[1] < 10:
        return None

    search, b_c, b_r = _build_search_region(
        off_u8, template.shape, tc0, tr0, res,
        ctx['coarse_offset'], ctx['search_margin_m'])

    if search.shape[0] <= template.shape[0] or search.shape[1] <= template.shape[1]:
        return None

    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < min_confidence:
        return CoarseResult(dx_m=0, dy_m=0, confidence=max_val, method="clahe_ncc")

    sub_x, sub_y = _subpixel_refine(result, max_loc)
    dx_m, dy_m = _match_to_offset(b_c, b_r, sub_x, sub_y, tc0, tr0, res)
    return CoarseResult(dx_m=dx_m, dy_m=dy_m, confidence=max_val, method="clahe_ncc")


def _compute_nmi(template, patch, n_bins=32):
    """Compute Normalized Mutual Information between two grayscale patches.

    Returns NMI in [0, 2] where higher = more similar.  Handles zero-data
    regions (nodata = 0) by masking them out before histogram computation.
    """
    t_valid = template > 0
    p_valid = patch > 0
    valid = t_valid & p_valid
    if np.sum(valid) < 100:
        return 0.0

    t_vals = template[valid]
    p_vals = patch[valid]

    # Joint histogram with Parzen (Gaussian) smoothing
    hist_joint, _, _ = np.histogram2d(
        t_vals, p_vals, bins=n_bins, range=[[0, 255], [0, 255]])
    hist_joint = cv2.GaussianBlur(hist_joint.astype(np.float32), (3, 3), 0.5)
    hist_joint = hist_joint / (hist_joint.sum() + 1e-10)

    hist_t = hist_joint.sum(axis=1)
    hist_p = hist_joint.sum(axis=0)

    eps = 1e-10
    h_t = -np.sum(hist_t * np.log(hist_t + eps))
    h_p = -np.sum(hist_p * np.log(hist_p + eps))
    h_tp = -np.sum(hist_joint * np.log(hist_joint + eps))
    mi = h_t + h_p - h_tp
    nmi = 2.0 * mi / (h_t + h_p + eps) if (h_t + h_p) > eps else 0.0
    return float(nmi)


def _nmi_offset(ctx, min_confidence=0.3, min_nmi=0.7):
    """Fallback: Normalized Mutual Information template matching.

    NMI captures non-linear intensity relationships between cross-temporal
    satellite imagery, unlike NCC which assumes linear correlation.  This
    makes it suitable as a fallback when land-mask NCC fails due to
    radiometric differences between eras (e.g. 1968 film vs modern satellite).

    Strategy: coarse-to-fine approach.
    1. Downsample to ~32m/px and exhaustively search (fast).
    2. Refine at full resolution around the coarse peak (accurate).

    Returns CoarseResult or None.
    """
    arr_offset, arr_ref = ctx['arr_offset'], ctx['arr_ref']
    tr0, tr1 = ctx['tr0'], ctx['tr1']
    tc0, tc1 = ctx['tc0'], ctx['tc1']
    res = ctx['res']

    ref_u8 = to_u8(arr_ref).astype(np.float32)
    off_u8 = to_u8(arr_offset).astype(np.float32)

    template = ref_u8[tr0:tr1, tc0:tc1]
    if template.shape[0] < 10 or template.shape[1] < 10:
        return None

    search, b_c, b_r = _build_search_region(
        off_u8, template.shape, tc0, tr0, res,
        ctx['coarse_offset'], ctx['search_margin_m'])

    if search.shape[0] <= template.shape[0] or search.shape[1] <= template.shape[1]:
        return None

    print(f"  [coarse/nmi] Starting NMI search at res {res}. "
          f"Search shape: {search.shape}, template shape: {template.shape}", flush=True)

    th, tw = template.shape

    # --- Coarse pass: downsample for speed ---
    # Target ~32m/px for the coarse NMI scan
    ds_factor = max(1, int(round(32.0 / res))) if res < 32 else 1
    if ds_factor > 1:
        tmpl_ds = cv2.resize(template, (tw // ds_factor, th // ds_factor),
                             interpolation=cv2.INTER_AREA)
        search_ds = cv2.resize(search,
                               (search.shape[1] // ds_factor, search.shape[0] // ds_factor),
                               interpolation=cv2.INTER_AREA)
    else:
        tmpl_ds = template
        search_ds = search

    th_ds, tw_ds = tmpl_ds.shape
    sh_ds, sw_ds = search_ds.shape

    if sh_ds <= th_ds or sw_ds <= tw_ds:
        return None

    # Exhaustive NMI search on downsampled images (stride 2 for speed)
    stride = 2
    best_mi = -1.0
    best_pos_ds = (0, 0)

    for sy in range(0, sh_ds - th_ds + 1, stride):
        for sx in range(0, sw_ds - tw_ds + 1, stride):
            patch = search_ds[sy:sy + th_ds, sx:sx + tw_ds]
            nmi = _compute_nmi(tmpl_ds, patch)
            if nmi > best_mi:
                best_mi = nmi
                best_pos_ds = (sx, sy)

    # --- Fine pass: hierarchical local search around coarse peak ---
    cx = best_pos_ds[0] * ds_factor
    cy = best_pos_ds[1] * ds_factor
    best_mi_fine = -1.0
    best_pos = (cx, cy)

    if 0 <= cy and cy + th <= search.shape[0] and 0 <= cx and cx + tw <= search.shape[1]:
        patch = search[cy:cy + th, cx:cx + tw]
        best_mi_fine = _compute_nmi(template, patch)

    step = max(1, ds_factor)
    while step >= 1:
        improved = True
        while improved:
            improved = False
            for dy in [-step, 0, step]:
                for dx in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    sy = best_pos[1] + dy
                    sx = best_pos[0] + dx
                    if sy < 0 or sy + th > search.shape[0] or sx < 0 or sx + tw > search.shape[1]:
                        continue
                    patch = search[sy:sy + th, sx:sx + tw]
                    nmi = _compute_nmi(template, patch)
                    if nmi > best_mi_fine:
                        best_mi_fine = nmi
                        best_pos = (sx, sy)
                        improved = True
        if step == 1:
            break
        step = max(1, step // 2)

    if best_mi_fine < min_nmi:
        return CoarseResult(dx_m=0, dy_m=0, confidence=best_mi_fine, method="nmi")

    # Convert to geographic offset
    actual_c = b_c + best_pos[0]
    actual_r = b_r + best_pos[1]
    dx_m = (actual_c - tc0) * res
    dy_m = (actual_r - tr0) * res

    return CoarseResult(dx_m=dx_m, dy_m=dy_m, confidence=best_mi_fine, method="nmi")
