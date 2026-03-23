"""RANSAC, phase refinement, GCP selection, and outlier removal."""

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import rasterio
import rasterio.transform
from scipy.spatial import cKDTree  # type: ignore

from .geo import fit_affine_from_gcps
from .image import make_land_mask
from .types import GCP, MatchPair


def _fit_initial_affine(anchors, auto, correction_outliers=None, top_n_auto=None):
    """Fit initial affine from anchors (preferred) or mixed pool.

    Returns (M_init, med_res, raw_med_res, fit_anchors, anchor_fit_unreliable).
    M_init may be None if insufficient data.
    """
    outlier_set = set(correction_outliers) if correction_outliers else set()
    fit_anchors = ([a for a in anchors
                    if a.name.replace("anchor:", "") not in outlier_set]
                   if outlier_set else anchors)

    if top_n_auto is None:
        top_n_auto = min(max(10, len(auto) // 3), len(auto))

    anchor_fit_unreliable = False
    M_init = None
    med_res = None
    raw_med_res = None

    if len(fit_anchors) >= 4:
        src_pts = np.array([p.off_coords() for p in fit_anchors])
        dst_pts = np.array([p.ref_coords() for p in fit_anchors])
        M_init, residuals = fit_affine_from_gcps(src_pts, dst_pts)
        raw_med_res = float(np.median(residuals))
        n_excluded = len(anchors) - len(fit_anchors)
        extra = f" ({n_excluded} outlier(s) excluded)" if n_excluded else ""
        if raw_med_res > 30:
            print(f"  WARNING: Anchors are internally inconsistent "
                  f"(median error: {raw_med_res:.1f}m) -- reference may have "
                  f"different distortion than target")
        if raw_med_res > 120:
            anchor_fit_unreliable = True
        print(f"  Initial fit driven by {len(fit_anchors)} anchors "
              f"(median error: {raw_med_res:.1f}m){extra}")
        med_res = min(raw_med_res, 30.0)

        if anchor_fit_unreliable:
            print("  Anchor fit is too inconsistent for truth-filtering; "
                  "falling back to auto-match consensus")
            M_init = None
            med_res = None
            raw_med_res = None
    else:
        fit_pool = fit_anchors + auto[:top_n_auto]
        if len(fit_pool) >= 3:
            src_pts = np.array([p.off_coords() for p in fit_pool])
            dst_pts = np.array([p.ref_coords() for p in fit_pool])
            fit_weights = np.array([50.0 if p.is_anchor else 1.0
                                    for p in fit_pool])
            M_init, residuals = fit_affine_from_gcps(src_pts, dst_pts, weights=fit_weights)
            med_res = float(np.median(residuals))
            raw_med_res = med_res

    return M_init, med_res, raw_med_res, fit_anchors, anchor_fit_unreliable


def _filter_by_anchor_truth(auto, anchors, M_init, med_res):
    """Filter neural matches by distance from anchor-driven affine truth.

    Returns (filtered_auto, rejected_auto).
    """
    if len(auto) < 6:
        return auto, []

    all_src = np.array([p.off_coords() for p in auto])
    all_dst = np.array([p.ref_coords() for p in auto])
    pred_all = cv2.transform(
        all_src.astype(np.float32).reshape(-1, 1, 2), M_init).reshape(-1, 2)
    all_residuals = np.sqrt(np.sum((pred_all - all_dst)**2, axis=1))

    global_threshold = max(5.0 * float(med_res), 100.0)

    auto_filtered = []
    auto_rejected = []
    n_rejected_by_anchor = 0
    for p, r in zip(auto, all_residuals):
        if r < global_threshold:
            auto_filtered.append(p)
        else:
            n_rejected_by_anchor += 1
            auto_rejected.append(p)

    if n_rejected_by_anchor > 0:
        print(f"  Anchor-truth filter: rejected {n_rejected_by_anchor} neural matches "
              f"not matching anchor geometry (> {global_threshold:.0f}m error)")

    # Local consistency rescue
    if auto_rejected and auto_filtered:
        rescued = local_consistency_filter(
            auto_rejected, anchors + auto_filtered,
            threshold_m=50.0, k_neighbors=5, search_radius=5000.0)
        if rescued:
            print(f"  Local consistency rescue: recovered {len(rescued)} "
                  f"of {len(auto_rejected)} rejected matches")
            auto_filtered.extend(rescued)

    return auto_filtered, auto_rejected


def _validate_anchors(anchors, M_init, med_res, raw_med_res, overlap, target_count):
    """Validate anchors against the initial affine and force-include for diversity.

    Returns validated anchor list.
    """
    a, bv, tx = M_init[0]
    c, d, ty = M_init[1]
    anchor_threshold = max(4.0 * float(med_res), 100.0)
    valid_anchors = []
    rejected_anchors = []
    for anc in anchors:
        ogx, ogy = anc.off_x, anc.off_y
        rgx, rgy = anc.ref_x, anc.ref_y
        pred_x = a * ogx + bv * ogy + tx
        pred_y = c * ogx + d * ogy + ty
        dx = rgx - pred_x
        dy = rgy - pred_y
        res = np.sqrt(dx ** 2 + dy ** 2)
        aname = anc.name.replace("anchor:", "")
        if res <= anchor_threshold:
            valid_anchors.append(anc)
        else:
            rejected_anchors.append((anc, res))
            print(f"    Rejected anchor {aname} (res={res:.0f}m "
                  f"[dx={dx:.1f}, dy={dy:.1f}] > {anchor_threshold:.0f}m)")

    # Force-include rejected anchors for spatial diversity when anchors are consistent
    n_rejected = len(rejected_anchors)
    raw_med_res_val = raw_med_res if raw_med_res is not None else 1e9
    if n_rejected >= 2 and raw_med_res_val <= 30:
        rejected_anchors.sort(key=lambda x: x[1])
        o_left, o_bottom, o_right, o_top = overlap
        gc = 7 if target_count > 25 else 5
        gr = 7 if target_count > 25 else 5
        cw = (o_right - o_left) / gc
        ch = (o_top - o_bottom) / gr
        occupied = set()
        for va in valid_anchors:
            col = min(gc - 1, max(0, int((va.ref_x - o_left) / cw)))
            row = min(gr - 1, max(0, int((va.ref_y - o_bottom) / ch)))
            occupied.add((row, col))
        forced = 0
        max_force = min(3, n_rejected)
        for anc, res in rejected_anchors:
            col = min(gc - 1, max(0, int((anc.ref_x - o_left) / cw)))
            row = min(gr - 1, max(0, int((anc.ref_y - o_bottom) / ch)))
            if (row, col) not in occupied:
                valid_anchors.append(anc)
                occupied.add((row, col))
                aname = anc.name.replace("anchor:", "")
                print(f"    Force-included anchor {aname} (residual={res:.0f}m, spatial diversity)")
                forced += 1
                if forced >= max_force:
                    break
        if forced > 0:
            print(f"    Neural model appears spatially biased "
                  f"({n_rejected}/{len(anchors)} anchors rejected), "
                  f"force-included {forced} for coverage")

    if len(valid_anchors) < len(anchors):
        print(f"    Kept {len(valid_anchors)}/{len(anchors)} anchors")

    return valid_anchors


def local_consistency_filter(candidates: list[MatchPair], reference_pairs: list[MatchPair], threshold_m=50.0,
                             k_neighbors=5, search_radius=5000.0,
                             min_confidence=0.65):
    """Keep *candidates* that are locally consistent with *reference_pairs*.

    For each candidate, find its ``k_neighbors`` nearest surviving neighbours
    (by ref position) and compare displacement vectors.  A candidate is kept if
    its offset agrees within ``threshold_m`` of the mean neighbour offset.
    Candidates with no nearby neighbours are kept if their confidence is at
    least ``min_confidence``.

    Both *candidates* and *reference_pairs* are lists of ``MatchPair``
    objects with attributes ``ref_x, ref_y, off_x, off_y, confidence, name``.

    Returns the filtered list of candidates.
    """
    if not reference_pairs or not candidates:
        return candidates

    fp_ref = np.array([m.ref_coords() for m in reference_pairs])
    fp_off = np.array([m.off_coords() for m in reference_pairs])
    fp_dx = fp_off[:, 0] - fp_ref[:, 0]
    fp_dy = fp_off[:, 1] - fp_ref[:, 1]
    tree = cKDTree(fp_ref)

    kept = []
    for cp in candidates:
        r_xy = np.array([cp.ref_x, cp.ref_y])
        dists, idxs = tree.query(r_xy, k=min(k_neighbors, len(fp_ref)),
                                 distance_upper_bound=search_radius)
        valid = [i for d, i in zip(dists, idxs) if d < search_radius]
        if not valid:
            if cp.confidence >= min_confidence:
                kept.append(cp)
        else:
            cp_dx = cp.off_x - cp.ref_x
            cp_dy = cp.off_y - cp.ref_y
            neighbor_dx = np.mean(fp_dx[valid])
            neighbor_dy = np.mean(fp_dy[valid])
            offset_diff = np.sqrt(
                (cp_dx - neighbor_dx) ** 2 + (cp_dy - neighbor_dy) ** 2)
            if offset_diff < threshold_m:
                kept.append(cp)
    return kept


def matched_pairs_sufficient(matched_pairs: list[MatchPair], target=25):
    """Check if matched pairs have sufficient count AND spatial distribution."""
    if len(matched_pairs) < target:
        return False
    if len(matched_pairs) < 4:
        return False

    xs = [p.ref_x for p in matched_pairs]
    ys = [p.ref_y for p in matched_pairs]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range < 100 or y_range < 100:
        return False

    grid = set()
    for p in matched_pairs:
        gc = min(2, int((p.ref_x - x_min) / x_range * 3))
        gr = min(2, int((p.ref_y - y_min) / y_range * 3))
        grid.add((gr, gc))

    return len(grid) >= 4


def _multiscale_phase_correlate(ref_patch, off_patch, fine_res):
    """Multi-scale phase correlation with iterative refinement.

    Tries 3 scales (full, 3/4, 1/2 of patch) and picks the one with
    highest response, then does one iterative refinement pass.
    Returns (dx_m, dy_m, response).
    """
    best_shift = (0.0, 0.0)
    best_resp = 0.0

    ph, pw = ref_patch.shape
    scales = [1.0, 0.75, 0.5]

    for s in scales:
        sh = max(32, int(ph * s))
        sw = max(32, int(pw * s))
        # Center crop
        r0 = (ph - sh) // 2
        c0 = (pw - sw) // 2
        rp = ref_patch[r0:r0+sh, c0:c0+sw]
        op = off_patch[r0:r0+sh, c0:c0+sw]

        window = np.outer(np.hanning(sh), np.hanning(sw))
        shift_val, response = cv2.phaseCorrelate(
            rp * window, op * window)

        if response > best_resp:
            best_resp = response
            best_shift = shift_val

    if best_resp < 0.02:
        return 0.0, 0.0, best_resp

    # Iterative refinement: shift the off patch by the detected shift,
    # then re-correlate for sub-pixel accuracy
    dx_px, dy_px = best_shift
    if abs(dx_px) < pw * 0.3 and abs(dy_px) < ph * 0.3:
        M = np.float32([[1, 0, -dx_px], [0, 1, -dy_px]])
        shifted_off = cv2.warpAffine(
            off_patch.astype(np.float32), M, (pw, ph),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        window = np.outer(np.hanning(ph), np.hanning(pw))
        refine_shift, refine_resp = cv2.phaseCorrelate(
            ref_patch * window, shifted_off.astype(np.float64) * window)
        if refine_resp > 0.02:
            dx_px += refine_shift[0]
            dy_px += refine_shift[1]
            best_resp = max(best_resp, refine_resp)

    return dx_px * fine_res, dy_px * fine_res, best_resp


def _refine_one_match(pair, arr_ref, arr_off_shifted, ref_transform,
                      off_transform, shift_px_x, shift_py_y, fine_res,
                      phase_half, h_img, w_img):
    """Refine a single match via multi-scale phase correlation (thread-safe).

    Returns (refined_pair, was_refined).
    """
    ref_gx, ref_gy = pair.ref_x, pair.ref_y
    off_gx, off_gy = pair.off_x, pair.off_y
    quality, name = pair.confidence, pair.name

    ref_row, ref_col = rasterio.transform.rowcol(ref_transform, ref_gx, ref_gy)
    ref_row, ref_col = int(ref_row), int(ref_col)

    off_row_raw, off_col_raw = rasterio.transform.rowcol(off_transform, off_gx, off_gy)
    off_row_shifted = int(off_row_raw) - shift_py_y
    off_col_shifted = int(off_col_raw) - shift_px_x

    if (ref_row - phase_half < 0 or ref_row + phase_half > h_img or
            ref_col - phase_half < 0 or ref_col + phase_half > w_img or
            off_row_shifted - phase_half < 0 or
            off_row_shifted + phase_half > h_img or
            off_col_shifted - phase_half < 0 or
            off_col_shifted + phase_half > w_img):
        return pair, False

    ref_patch = arr_ref[
        ref_row - phase_half:ref_row + phase_half,
        ref_col - phase_half:ref_col + phase_half
    ].astype(np.float64)
    off_patch = arr_off_shifted[
        off_row_shifted - phase_half:off_row_shifted + phase_half,
        off_col_shifted - phase_half:off_col_shifted + phase_half
    ].astype(np.float64)

    if ref_patch.shape[0] < 32 or ref_patch.shape[1] < 32:
        return pair, False

    sub_dx_m, sub_dy_m, response = _multiscale_phase_correlate(
        ref_patch, off_patch, fine_res)

    if response > 0.03:
        new_off_gx = off_gx + sub_dx_m
        new_off_gy = off_gy + sub_dy_m
        return MatchPair(ref_x=ref_gx, ref_y=ref_gy, off_x=new_off_gx,
                         off_y=new_off_gy, confidence=quality, name=name,
                         precision=pair.precision, source=pair.source,
                         hypothesis_id=pair.hypothesis_id), True
    else:
        return MatchPair(ref_x=ref_gx, ref_y=ref_gy, off_x=off_gx,
                         off_y=off_gy, confidence=quality * 0.7, name=name,
                         precision=pair.precision, source=pair.source,
                         hypothesis_id=pair.hypothesis_id), False


def refine_matches_phase_correlation(matched_pairs: list[MatchPair], arr_ref, arr_off_shifted,
                                     ref_transform, off_transform,
                                     shift_px_x, shift_py_y, fine_res):
    """Refine neural matches using multi-scale phase correlation.

    Uses multiple patch sizes and iterative refinement for improved
    sub-pixel accuracy. Always attempts refinement on every match.
    Runs in parallel using threads (OpenCV/NumPy release the GIL).
    """
    if not matched_pairs:
        return []

    h_img, w_img = arr_ref.shape
    phase_half = int(128 / fine_res)

    n_workers = min(os.cpu_count() or 4, len(matched_pairs), 8)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(_refine_one_match, pair, arr_ref, arr_off_shifted,
                        ref_transform, off_transform, shift_px_x, shift_py_y,
                        fine_res, phase_half, h_img, w_img)
            for pair in matched_pairs
        ]
        results = [f.result() for f in futures]

    refined = [r[0] for r in results]
    n_refined = sum(1 for r in results if r[1])

    print(f"    Phase correlation refined {n_refined}/{len(matched_pairs)} matches"
          f" ({n_workers} threads)")

    return refined


def correct_reference_offset(matched_pairs: list[MatchPair]) -> tuple[list[MatchPair], bool, list[str]]:
    """Correct systematic reference image offset using anchor ground truth.

    Fits a spatially varying (affine) correction from anchor displacements,
    handling reference images where the geographic offset varies across the
    image extent (e.g. due to different map projection distortions).

    Should be called BEFORE select_best_gcps so that the anchor-truth filter
    does not reject correctly-matched features.

    Returns (corrected_pairs, was_corrected, outlier_names).
    ``outlier_names`` is a list of anchor name strings excluded from the
    correction fit (useful for downstream filtering).
    """
    anchors = [p for p in matched_pairs if p.is_anchor]
    auto = [p for p in matched_pairs if not p.is_anchor]

    if len(anchors) < 1 or len(auto) < 6:
        return matched_pairs, False, []

    # Fit affine from auto matches only
    auto_src = np.array([p.off_coords() for p in auto])
    auto_dst = np.array([p.ref_coords() for p in auto])
    M_auto, _ = fit_affine_from_gcps(auto_src, auto_dst)

    # For each anchor, compute displacement between ground truth ref
    # and auto-affine prediction
    anchor_positions = []
    shifts = []
    for anchor in anchors:
        anchor_ref_gx, anchor_ref_gy = anchor.ref_x, anchor.ref_y
        anchor_off_gx, anchor_off_gy = anchor.off_x, anchor.off_y

        pred_ref_gx = (M_auto[0, 0] * anchor_off_gx
                       + M_auto[0, 1] * anchor_off_gy + M_auto[0, 2])
        pred_ref_gy = (M_auto[1, 0] * anchor_off_gx
                       + M_auto[1, 1] * anchor_off_gy + M_auto[1, 2])

        anchor_positions.append((anchor_off_gx, anchor_off_gy))
        shifts.append((anchor_ref_gx - pred_ref_gx, anchor_ref_gy - pred_ref_gy))

    shifts_e = np.array([s[0] for s in shifts])
    shifts_n = np.array([s[1] for s in shifts])
    median_e = float(np.median(shifts_e))
    median_n = float(np.median(shifts_n))
    total = np.sqrt(median_e ** 2 + median_n ** 2)

    if total < 20:
        return matched_pairs, False, []

    # If we only have 1-2 anchors, we cannot fit a stable spatially varying
    # correction, but we can still use them as a translation sanity check.
    # This is especially useful for cropped historical references where only
    # a few anchors remain in overlap.
    if len(anchors) < 3:
        std_e = float(np.std(shifts_e)) if len(shifts_e) > 1 else 0.0
        std_n = float(np.std(shifts_n)) if len(shifts_n) > 1 else 0.0
        if total <= 1500.0 and max(std_e, std_n) <= 150.0:
            print("  Reference offset detected (translation-only from limited anchors):")
            print(f"    Median: dE={median_e:+.1f}m, dN={median_n:+.1f}m "
                  f"(total: {total:.1f}m), std: E={std_e:.1f}m, N={std_n:.1f}m")
            corrected = []
            for p in matched_pairs:
                if p.is_anchor:
                    corrected.append(p)
                else:
                    corrected.append(MatchPair(
                        ref_x=p.ref_x + median_e,
                        ref_y=p.ref_y + median_n,
                        off_x=p.off_x, off_y=p.off_y,
                        confidence=p.confidence, name=p.name,
                        precision=p.precision, source=p.source,
                        hypothesis_id=p.hypothesis_id,
                    ))
            return corrected, True, []
        print(f"  Reference offset check: only {len(anchors)} anchor(s) with "
              f"inconsistent translation (std_E={std_e:.1f}m, std_N={std_n:.1f}m), skipping")
        return matched_pairs, False, []

    # Reject anchor outliers using affine-residual rejection instead of MAD.
    # MAD on raw shifts rejects anchors whose displacement differs from the
    # median, but with spatially-varying reference offsets (e.g. increasing
    # error westward) a distant anchor can have a genuinely different shift
    # that follows a smooth spatial gradient.  An affine fit naturally
    # captures such gradients, so we fit from ALL anchors first and reject
    # only those with high residuals against the spatial model.
    anchor_pos_arr = np.array(anchor_positions)
    shift_arr = np.array(shifts)
    M_test, test_residuals = fit_affine_from_gcps(anchor_pos_arr, shift_arr)
    med_test_res = float(np.median(test_residuals))
    residual_threshold = max(3.0 * med_test_res, 50.0)
    inlier_mask = np.array(test_residuals) < residual_threshold
    n_rejected_anchors = int(np.sum(~inlier_mask))
    outlier_names = [anchors[i].name.replace("anchor:", "")
                     for i in range(len(anchors)) if not inlier_mask[i]]
    if n_rejected_anchors > 0:
        print(f"  Reference offset: rejected {n_rejected_anchors} anchor outlier(s) "
              f"({', '.join(outlier_names)}) before fitting correction "
              f"(affine residual > {residual_threshold:.0f}m)")
    inlier_positions = [anchor_positions[i] for i in range(len(anchors)) if inlier_mask[i]]
    inlier_shifts = [shifts[i] for i in range(len(anchors)) if inlier_mask[i]]

    if len(inlier_positions) < 3:
        print(f"  Reference offset check: only {len(inlier_positions)} consistent anchors "
              f"(need 3+), skipping")
        return matched_pairs, False, outlier_names

    # Fit affine correction from inliers: offset_position -> correction_shift
    # This captures translation + rotation + scale variations in the
    # reference offset across the image.
    anchor_pos_arr = np.array(inlier_positions)
    shift_arr = np.array(inlier_shifts)
    M_corr, corr_residuals = fit_affine_from_gcps(anchor_pos_arr, shift_arr)

    max_res = float(max(corr_residuals))
    mean_res = float(np.mean(corr_residuals))

    # With exactly 3 anchors the affine is exactly determined (residuals ≈ 0).
    # With 4+ anchors, check that the affine explains the shifts well.
    if len(inlier_positions) > 3 and mean_res > 50:
        print(f"  Reference offset check: median dE={median_e:+.1f}m, dN={median_n:+.1f}m "
              f"but affine correction residuals too high "
              f"(max={max_res:.1f}m, mean={mean_res:.1f}m), skipping")
        return matched_pairs, False, outlier_names

    std_e = float(np.std(shifts_e[inlier_mask]))
    std_n = float(np.std(shifts_n[inlier_mask]))
    print(f"  Reference offset detected (spatially varying):")
    print(f"    Median: dE={median_e:+.1f}m, dN={median_n:+.1f}m "
          f"(total: {total:.1f}m), std: E={std_e:.1f}m, N={std_n:.1f}m")
    if len(inlier_positions) > 3:
        print(f"    Affine correction fit: mean residual={mean_res:.1f}m, "
              f"max={max_res:.1f}m")
    print(f"    Correcting {len(auto)} auto-match ref positions using "
          f"{len(inlier_positions)} anchor ground truth"
          f" ({n_rejected_anchors} outliers excluded)" if n_rejected_anchors > 0
          else f"    Correcting {len(auto)} auto-match ref positions using "
          f"{len(inlier_positions)} anchor ground truth")

    # Apply spatially varying correction to each auto match
    corrected = []
    for p in matched_pairs:
        if p.is_anchor:
            corrected.append(p)
        else:
            off_gx, off_gy = p.off_x, p.off_y
            local_shift_e = (M_corr[0, 0] * off_gx + M_corr[0, 1] * off_gy
                             + M_corr[0, 2])
            local_shift_n = (M_corr[1, 0] * off_gx + M_corr[1, 1] * off_gy
                             + M_corr[1, 2])
            corrected.append(MatchPair(
                ref_x=p.ref_x + local_shift_e,
                ref_y=p.ref_y + local_shift_n,
                off_x=p.off_x, off_y=p.off_y,
                confidence=p.confidence, name=p.name,
                precision=p.precision, source=p.source,
                hypothesis_id=p.hypothesis_id,
            ))

    return corrected, True, outlier_names


def select_best_gcps(matched_pairs: list[MatchPair], overlap, target_count=25,
                     correction_outliers=None,
                     arr_ref=None, ref_transform=None,
                     shoreline_quota=0.40) -> tuple[list[MatchPair], float]:
    """Select the best GCPs with good spatial distribution.

    *correction_outliers* is an optional list of anchor name strings
    (without the ``anchor:`` prefix) that were excluded from the
    correction fit.  When provided, those anchors are excluded from the
    anchor-only initial affine fit so the resulting M_init and threshold
    are not polluted by outlier anchors.

    Returns (selected_gcps, coverage_fraction).
    """
    def _compute_coverage(points, overlap_extent):
        o_left, o_bottom, o_right, o_top = overlap_extent
        gc = 7 if target_count > 25 else 5
        gr = 7 if target_count > 25 else 5
        cw = (o_right - o_left) / gc
        ch = (o_top - o_bottom) / gr
        occupied = set()
        for p in points:
            col = min(gc - 1, max(0, int((p.ref_x - o_left) / cw)))
            row = min(gr - 1, max(0, int((p.ref_y - o_bottom) / ch)))
            occupied.add((row, col))
        coverage = len(occupied) / (gc * gr)

        if arr_ref is not None and ref_transform is not None:
            try:
                land = (make_land_mask(arr_ref) > 0)
                h, w = land.shape
                land_cells = set()
                for r in range(0, h, 32):
                    cols = np.where(land[r])[0]
                    if cols.size == 0:
                        continue
                    for c in cols[::16]:
                        gx, gy = rasterio.transform.xy(ref_transform, float(r), float(c))
                        if not (o_left <= gx <= o_right and o_bottom <= gy <= o_top):
                            continue
                        col = min(gc - 1, max(0, int((gx - o_left) / cw)))
                        row = min(gr - 1, max(0, int((gy - o_bottom) / ch)))
                        land_cells.add((row, col))
                if land_cells:
                    land_occ = len(occupied & land_cells)
                    land_cov = land_occ / len(land_cells)
                    if land_cov < 0.55:
                        print(f"  WARNING: GCPs occupy only {land_occ}/{len(land_cells)} "
                              f"land grid cells ({land_cov:.0%} land coverage)")
                    else:
                        print(f"  Land-aware coverage: {land_occ}/{len(land_cells)} "
                              f"cells ({land_cov:.0%})")
            except Exception:
                pass

        if coverage < 0.5:
            print(f"  WARNING: GCPs occupy only {len(occupied)}/{gc * gr} grid cells "
                  f"({coverage:.0%} coverage)")
        return coverage

    def _classify_match_points(points):
        if arr_ref is None or ref_transform is None or not points:
            return ["unknown"] * len(points)
        land = (make_land_mask(arr_ref) > 0)
        shore = cv2.morphologyEx(land.astype(np.uint8), cv2.MORPH_GRADIENT,
                                 np.ones((3, 3), np.uint8)) > 0
        shore = cv2.dilate(shore.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
        h, w = land.shape
        labels = []
        for p in points:
            try:
                rr, cc = rasterio.transform.rowcol(ref_transform, p.ref_x, p.ref_y)
                rr = int(rr)
                cc = int(cc)
            except Exception:
                labels.append("unknown")
                continue
            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                labels.append("unknown")
            elif shore[rr, cc]:
                labels.append("shore")
            elif land[rr, cc]:
                labels.append("inland")
            else:
                labels.append("water")
        return labels

    def _grid_key(p, o_left, o_bottom, cell_w, cell_h, grid_cols, grid_rows):
        gc = min(grid_cols - 1, max(0, int((p.ref_x - o_left) / cell_w)))
        gr = min(grid_rows - 1, max(0, int((p.ref_y - o_bottom) / cell_h)))
        return gr, gc

    def _spatial_pick(pool, target, min_conf, occupied_keys):
        if target <= 0 or not pool:
            return []
        by_cell = {}
        for p in pool:
            if p.confidence < min_conf:
                continue
            key = _grid_key(p, o_left, o_bottom, cell_w, cell_h, grid_cols, grid_rows)
            by_cell.setdefault(key, []).append(p)

        selected_local = []
        # pass 1: one per currently empty cell
        for key in sorted(by_cell.keys()):
            if key in occupied_keys:
                continue
            cand = max(by_cell[key], key=lambda x: x.confidence)
            selected_local.append(cand)
            occupied_keys.add(key)
            if len(selected_local) >= target:
                return selected_local

        # pass 2: strongest remaining globally
        rem = []
        used = set(id(p) for p in selected_local)
        for v in by_cell.values():
            rem.extend(v)
        rem.sort(key=lambda x: -x.confidence)
        for p in rem:
            if id(p) in used:
                continue
            selected_local.append(p)
            used.add(id(p))
            if len(selected_local) >= target:
                break
        return selected_local

    if len(matched_pairs) <= target_count:
        coverage = _compute_coverage(matched_pairs, overlap)
        return matched_pairs, coverage

    anchors = [p for p in matched_pairs if p.is_anchor]
    auto = [p for p in matched_pairs if not p.is_anchor]
    auto.sort(key=lambda x: -x.confidence)

    remaining_target = target_count - len(anchors)
    rejected_auto = []

    # Fit initial affine (anchors-first, then mixed pool fallback)
    M_init, med_res, raw_med_res, _, anchor_fit_unreliable = _fit_initial_affine(
        anchors, auto, correction_outliers=correction_outliers)

    if anchor_fit_unreliable:
        anchors = []

    # Filter neural matches by anchor-driven truth
    if M_init is not None and med_res is not None:
        auto, rejected_auto = _filter_by_anchor_truth(auto, anchors, M_init, med_res)

    # Validate anchors against the fit
    if M_init is not None and med_res is not None and anchors:
        anchors = _validate_anchors(
            anchors, M_init, med_res, raw_med_res, overlap, target_count)

    # Spatial diversity selection
    if len(auto) <= remaining_target:
        result = anchors + auto
        coverage = _compute_coverage(result, overlap)
        return result, coverage

    o_left, o_bottom, o_right, o_top = overlap
    grid_cols = 7 if target_count > 25 else 5
    grid_rows = 7 if target_count > 25 else 5
    cell_w = (o_right - o_left) / grid_cols
    cell_h = (o_top - o_bottom) / grid_rows

    labels = _classify_match_points(auto)
    shoreline_auto = [p for p, l in zip(auto, labels) if l == "shore"]
    inland_auto = [p for p, l in zip(auto, labels) if l == "inland"]
    other_auto = [p for p, l in zip(auto, labels) if l not in ("shore", "inland")]

    target_shore = int(round(remaining_target * shoreline_quota))
    target_inland = remaining_target - target_shore
    if len(shoreline_auto) < target_shore:
        target_inland += target_shore - len(shoreline_auto)
        target_shore = len(shoreline_auto)
    if len(inland_auto) < target_inland:
        target_shore += target_inland - len(inland_auto)
        target_inland = len(inland_auto)

    anchor_occ = set()
    for p in anchors:
        anchor_occ.add(_grid_key(p, o_left, o_bottom, cell_w, cell_h, grid_cols, grid_rows))

    selected = []
    selected.extend(_spatial_pick(shoreline_auto, target_shore, 0.60, anchor_occ))
    selected.extend(_spatial_pick(inland_auto, target_inland, 0.65, anchor_occ))

    # Backfill from strongest remaining points (land first, then others)
    if len(selected) < remaining_target:
        selected_set = set(id(p) for p in selected)
        backfill_pool = shoreline_auto + inland_auto + other_auto
        backfill_pool.sort(key=lambda x: -x.confidence)
        for p in backfill_pool:
            min_conf = 0.58 if p in shoreline_auto else 0.67 if p in inland_auto else 0.82
            if p.confidence < min_conf:
                continue
            if id(p) in selected_set:
                continue
            if p in other_auto:
                # Water/unknown points are only useful if they are locally
                # consistent with existing trusted controls.
                keep = local_consistency_filter(
                    [p], anchors + selected,
                    threshold_m=60.0,
                    k_neighbors=5,
                    search_radius=6000.0,
                    min_confidence=0.85,
                )
                if not keep:
                    continue
            selected.append(p)
            selected_set.add(id(p))
            if len(selected) >= remaining_target:
                break

    # Coverage-first rescue: fill empty grid cells with locally consistent
    # candidates even below the global confidence cutoff. This improves
    # stability near sparse coastlines without hardcoding any direction.
    def _occupied(points):
        occ = set()
        for pp in points:
            occ.add(_grid_key(pp, o_left, o_bottom, cell_w, cell_h, grid_cols, grid_rows))
        return occ

    selected_set = set(id(p) for p in selected)
    occ = _occupied(anchors + selected)
    all_cells = {(r, c) for r in range(grid_rows) for c in range(grid_cols)}
    empty_cells = sorted(all_cells - occ)
    if empty_cells:
        candidates_relaxed = [p for p in auto if id(p) not in selected_set and p.confidence >= 0.50]
        if rejected_auto:
            candidates_relaxed.extend([p for p in rejected_auto if p.confidence >= 0.55])

        rescue_added = 0
        for key in empty_cells:
            cell_pool = [p for p in candidates_relaxed
                         if _grid_key(p, o_left, o_bottom, cell_w, cell_h, grid_cols, grid_rows) == key]
            if not cell_pool:
                continue
            cell_pool.sort(key=lambda x: -x.confidence)
            # Try strongest first; require local-consistency agreement with
            # already trusted points.
            for cand in cell_pool[:5]:
                keep = local_consistency_filter(
                    [cand], anchors + selected,
                    threshold_m=80.0,
                    k_neighbors=5,
                    search_radius=8000.0,
                    min_confidence=0.75,
                )
                if keep:
                    selected.append(cand)
                    selected_set.add(id(cand))
                    rescue_added += 1
                    break

        if rescue_added > 0:
            print(f"    Coverage rescue: added {rescue_added} locally-consistent "
                  f"GCP(s) in previously empty cells")

    # Edge-cell backfill
    if rejected_auto and len(selected) < remaining_target + 4:
        selected_set = set(id(p) for p in selected)
        edge_keys = set()
        for r in range(grid_rows):
            for c in range(grid_cols):
                if r == 0 or r == grid_rows - 1 or c == 0 or c == grid_cols - 1:
                    edge_keys.add((r, c))
        occupied_edge = set()
        for p in anchors + selected:
            gc = min(grid_cols - 1, max(0, int((p.ref_x - o_left) / cell_w)))
            gr = min(grid_rows - 1, max(0, int((p.ref_y - o_bottom) / cell_h)))
            if (gr, gc) in edge_keys:
                occupied_edge.add((gr, gc))
        empty_edge = edge_keys - occupied_edge
        if empty_edge:
            edge_candidates = {}
            for p in rejected_auto:
                gc = min(grid_cols - 1, max(0, int((p.ref_x - o_left) / cell_w)))
                gr = min(grid_rows - 1, max(0, int((p.ref_y - o_bottom) / cell_h)))
                key = (gr, gc)
                if key in empty_edge:
                    if key not in edge_candidates:
                        edge_candidates[key] = []
                    edge_candidates[key].append(p)
            backfilled = 0
            for key in sorted(edge_candidates.keys()):
                best = max(edge_candidates[key], key=lambda x: x.confidence)
                if id(best) not in selected_set:
                    selected.append(best)
                    selected_set.add(id(best))
                    backfilled += 1
            if backfilled > 0:
                print(f"    Edge backfill: added {backfilled} GCPs to empty boundary cells")

    # Final local smoothness cleanup for non-anchor picks to reduce jitter.
    if selected:
        selected_before = len(selected)
        selected = local_consistency_filter(
            selected,
            anchors + selected,
            threshold_m=65.0,
            k_neighbors=5,
            search_radius=6500.0,
            min_confidence=0.80,
        )
        if len(selected) < selected_before:
            print(f"    Smoothness filter: {selected_before} -> {len(selected)} selected GCPs")

    # Ensure we still hit target count if smoothness pruning was aggressive.
    if len(selected) < remaining_target:
        selected_set = set(id(p) for p in selected)
        refill = [p for p in (shoreline_auto + inland_auto) if id(p) not in selected_set and p.confidence >= 0.70]
        refill.sort(key=lambda x: -x.confidence)
        for p in refill:
            selected.append(p)
            selected_set.add(id(p))
            if len(selected) >= remaining_target:
                break

    result = anchors + selected
    coverage = _compute_coverage(result, overlap)
    return result, coverage


def iterative_outlier_removal(matched_pairs: list[MatchPair], neural_res, use_sift_refinement,
                              used_neural) -> tuple[list[MatchPair], np.ndarray, list[float]]:
    """Iteratively remove GCP outliers using spatial-aware thresholds.

    Returns (cleaned_matched_pairs, M_geo, geo_residuals).
    """
    offset_geo_pts = np.array([p.off_coords() for p in matched_pairs])
    ref_geo_pts = np.array([p.ref_coords() for p in matched_pairs])
    corr_weights = np.array([
        p.confidence * (5.0 if p.is_anchor else 1.0)
        for p in matched_pairs
    ])

    n_anchors_in_fit = sum(1 for p in matched_pairs if p.is_anchor)
    if n_anchors_in_fit > 0:
        print(f"  Anchor weight boost: {n_anchors_in_fit} anchors at 5x weight")

    M_geo, geo_residuals = fit_affine_from_gcps(
        offset_geo_pts, ref_geo_pts, weights=corr_weights)

    active_mask = np.ones(len(matched_pairs), dtype=bool)
    max_iter = min(20, len(matched_pairs) // 3)
    residual_floor = (max(5.0 * neural_res, 10.0)
                      if (use_sift_refinement or used_neural) else 500.0)
    all_eastings = np.array([matched_pairs[i].off_x for i in range(len(matched_pairs))])

    for iteration in range(max_iter):
        active_idx = np.where(active_mask)[0]
        if len(active_idx) <= 6:
            break
        residuals_arr = np.array(geo_residuals)
        active_residuals = residuals_arr[active_mask]
        active_eastings = all_eastings[active_mask]
        median_res = np.median(active_residuals)

        mid_x = (active_eastings.min() + active_eastings.max()) / 2
        west_region = active_eastings < mid_x
        east_region = ~west_region
        if west_region.sum() >= 3 and east_region.sum() >= 3:
            median_west = np.median(active_residuals[west_region])
            median_east = np.median(active_residuals[east_region])
            regional_medians = np.where(
                all_eastings < mid_x, median_west, median_east)
        else:
            regional_medians = np.full(len(matched_pairs), median_res)

        worst_idx = None
        worst_res = 0
        for ai in active_idx:
            if residuals_arr[ai] > worst_res:
                worst_res = residuals_arr[ai]
                worst_idx = ai
        if worst_idx is None:
            break
        # Anchors are ground truth: use a higher removal threshold because
        # high residuals in the affine model usually indicate model
        # inadequacy (e.g., non-linear distortion to islands) rather than
        # a wrong match.  The TPS can handle them.
        worst_pair = matched_pairs[worst_idx]
        is_anchor = worst_pair.is_anchor
        anchor_floor = max(5.0 * residual_floor, 75.0) if is_anchor else 0
        point_threshold = max(3.0 * regional_medians[worst_idx],
                              residual_floor, anchor_floor)
        if worst_res > point_threshold:
            active_mask[worst_idx] = False
            active_pairs = [matched_pairs[i] for i in np.where(active_mask)[0]]
            active_offset = np.array([p.off_coords() for p in active_pairs])
            active_ref = np.array([p.ref_coords() for p in active_pairs])
            active_weights = np.array([p.confidence for p in active_pairs])
            M_geo, _ = fit_affine_from_gcps(active_offset, active_ref,
                                            weights=active_weights)
            geo_residuals = []
            for i in range(len(matched_pairs)):
                ogx, ogy = matched_pairs[i].off_x, matched_pairs[i].off_y
                rgx, rgy = matched_pairs[i].ref_x, matched_pairs[i].ref_y
                pred_x = M_geo[0, 0] * ogx + M_geo[0, 1] * ogy + M_geo[0, 2]
                pred_y = M_geo[1, 0] * ogx + M_geo[1, 1] * ogy + M_geo[1, 2]
                geo_residuals.append(np.sqrt((pred_x - rgx) ** 2 + (pred_y - rgy) ** 2))
            gcp_label = matched_pairs[worst_idx].name
            print(f"  Removed GCP {gcp_label} with {worst_res:.0f}m residual")
        else:
            break

    active_idx = np.where(active_mask)[0]
    if len(active_idx) < len(matched_pairs):
        matched_pairs = [matched_pairs[i] for i in active_idx]
        geo_residuals = [geo_residuals[i] for i in active_idx]

        offset_geo_pts = np.array([p.off_coords() for p in matched_pairs])
        ref_geo_pts = np.array([p.ref_coords() for p in matched_pairs])
        corr_weights = np.array([
            p.confidence * (5.0 if p.is_anchor else 1.0)
            for p in matched_pairs
        ])
        M_geo, geo_residuals = fit_affine_from_gcps(
            offset_geo_pts, ref_geo_pts, weights=corr_weights)

    return matched_pairs, M_geo, geo_residuals


def detect_and_correct_reference_offset(original_pairs: list[MatchPair],
                                         filtered_pairs: list[MatchPair], M_geo,
                                         neural_res, use_sift_refinement,
                                         used_neural):
    """Detect and correct systematic reference image offset using anchor ground truth.

    When a reference image has a systematic geographic offset from ground truth,
    Neural matches are self-consistent but shifted.  Anchors (ground truth ref
    coordinates) get rejected as outliers because they disagree with the
    match-dominated affine.  This function detects that pattern and shifts all
    neural match ref positions to align with anchor ground truth, then re-runs
    outlier removal.

    Returns (matched_pairs, M_geo, geo_residuals, was_corrected).
    """
    original_anchors = [p for p in original_pairs if p.is_anchor]
    if len(original_anchors) < 3:
        return filtered_pairs, M_geo, None, False

    filtered_names = set(p.name for p in filtered_pairs)
    removed_anchors = [p for p in original_anchors if p.name not in filtered_names]

    removal_fraction = len(removed_anchors) / len(original_anchors)
    if removal_fraction < 0.6:
        return filtered_pairs, M_geo, None, False

    print(f"\n  Reference offset detection: {len(removed_anchors)}/{len(original_anchors)} "
          f"anchors rejected ({removal_fraction:.0%})")

    # For each anchor, compute discrepancy between ground truth ref position
    # and where the neural-match affine predicts the anchor's offset position maps.
    anchor_positions = []
    shifts = []
    for anchor in original_anchors:
        anchor_ref_gx, anchor_ref_gy = anchor.ref_x, anchor.ref_y   # ground truth
        anchor_off_gx, anchor_off_gy = anchor.off_x, anchor.off_y   # offset position

        pred_ref_gx = (M_geo[0, 0] * anchor_off_gx
                       + M_geo[0, 1] * anchor_off_gy + M_geo[0, 2])
        pred_ref_gy = (M_geo[1, 0] * anchor_off_gx
                       + M_geo[1, 1] * anchor_off_gy + M_geo[1, 2])

        shift_e = anchor_ref_gx - pred_ref_gx
        shift_n = anchor_ref_gy - pred_ref_gy
        anchor_positions.append((anchor_off_gx, anchor_off_gy))
        shifts.append((shift_e, shift_n))

        aname = anchor.name.replace("anchor:", "")
        print(f"    {aname}: dE={shift_e:+.1f}m, dN={shift_n:+.1f}m")

    shifts_e = np.array([s[0] for s in shifts])
    shifts_n = np.array([s[1] for s in shifts])
    median_shift_e = float(np.median(shifts_e))
    median_shift_n = float(np.median(shifts_n))
    total_shift = np.sqrt(median_shift_e ** 2 + median_shift_n ** 2)
    std_e = float(np.std(shifts_e))
    std_n = float(np.std(shifts_n))

    print(f"    Median shift: dE={median_shift_e:+.1f}m, dN={median_shift_n:+.1f}m "
          f"(total: {total_shift:.1f}m)")
    print(f"    Consistency: std_E={std_e:.1f}m, std_N={std_n:.1f}m")

    if total_shift < 10:
        print(f"    Reference offset < 10m, no correction needed")
        return filtered_pairs, M_geo, None, False

    # Skip reference offset correction if every anchor was rejected by the
    # spatial consensus. In that case the anchor set is likely unreliable and
    # using it to steer the reference would be worse than doing nothing.
    if removal_fraction == 1.0:
        print(f"    WARNING: All {len(original_anchors)} anchors were rejected by the spatial consensus.")
        print("    Skipping reference offset correction because anchors are likely unreliable.")
        return filtered_pairs, M_geo, None, False

    # Reject anchor outliers using affine-residual rejection (same approach
    # as correct_reference_offset).  Fit from ALL anchors first, then reject
    # those with high residuals against the spatial model.
    anchor_pos_arr = np.array(anchor_positions)
    shift_arr = np.array(shifts)
    M_test, test_residuals = fit_affine_from_gcps(anchor_pos_arr, shift_arr)
    med_test_res = float(np.median(test_residuals))
    residual_threshold = max(3.0 * med_test_res, 50.0)
    inlier_mask = np.array(test_residuals) < residual_threshold
    n_rejected_anchors = int(np.sum(~inlier_mask))
    if n_rejected_anchors > 0:
        rejected_names = [original_anchors[i].name.replace("anchor:", "")
                          for i in range(len(original_anchors)) if not inlier_mask[i]]
        print(f"    Rejected {n_rejected_anchors} anchor outlier(s) "
              f"({', '.join(rejected_names)}) before fitting correction "
              f"(affine residual > {residual_threshold:.0f}m)")
    inlier_positions = [anchor_positions[i] for i in range(len(original_anchors))
                        if inlier_mask[i]]
    inlier_shifts = [shifts[i] for i in range(len(original_anchors)) if inlier_mask[i]]

    if len(inlier_positions) < 3:
        print(f"    Only {len(inlier_positions)} consistent anchors (need 3+), skipping")
        return filtered_pairs, M_geo, None, False

    # Fit affine correction from inliers: offset_position -> correction_shift
    # This handles spatially varying offsets (translation + rotation + scale).
    anchor_pos_arr = np.array(inlier_positions)
    shift_arr = np.array(inlier_shifts)
    M_corr, corr_residuals = fit_affine_from_gcps(anchor_pos_arr, shift_arr)

    max_res = float(max(corr_residuals))
    mean_res = float(np.mean(corr_residuals))

    if len(inlier_positions) > 3:
        print(f"    Affine correction fit: mean residual={mean_res:.1f}m, "
              f"max={max_res:.1f}m")

    if len(inlier_positions) > 3 and mean_res > 50:
        print(f"    Affine correction residuals too high, not a systematic offset")
        return filtered_pairs, M_geo, None, False

    # Apply spatially varying correction to all non-anchor ref positions.
    # Start from original_pairs so removed anchors are re-included.
    corrected_pairs = []
    for p in original_pairs:
        if p.is_anchor:
            corrected_pairs.append(p)
        else:
            off_gx, off_gy = p.off_x, p.off_y
            local_shift_e = (M_corr[0, 0] * off_gx + M_corr[0, 1] * off_gy
                             + M_corr[0, 2])
            local_shift_n = (M_corr[1, 0] * off_gx + M_corr[1, 1] * off_gy
                             + M_corr[1, 2])
            corrected_pairs.append(MatchPair(
                ref_x=p.ref_x + local_shift_e,
                ref_y=p.ref_y + local_shift_n,
                off_x=p.off_x, off_y=p.off_y,
                confidence=p.confidence, name=p.name,
                precision=p.precision, source=p.source,
                hypothesis_id=p.hypothesis_id,
            ))

    print(f"    Applied spatially varying correction to {len(corrected_pairs)} GCPs, "
          f"re-running outlier removal...")

    matched_pairs, M_geo_new, geo_residuals = iterative_outlier_removal(
        corrected_pairs, neural_res, use_sift_refinement, used_neural)

    return matched_pairs, M_geo_new, geo_residuals, True
