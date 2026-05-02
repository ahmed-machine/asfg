"""Coarse offset detection and global localization.

Coarse offset uses tiled ELoFTR matching on a shared canvas (see
:func:`align.scale.eloftr_translation_estimate`). The historic NCC chain
(land-mask NCC, CLAHE-grayscale NCC, NMI) was removed in 2026-04 because
it was repeatedly fooled by similar-shape coastlines (Bahrain north
coast vs Saudi mainland) — high correlation, wrong geography, no
internal way to detect the ambiguity. ELoFTR's per-match MAD-derived
agreement score abstains in those cases instead.

Global localization continues to search the full reference for the
target footprint via weighted-NCC; that's a different problem (no prior
on position at all) and the current matcher is acceptable there.
"""

import os
import tempfile

import cv2
import numpy as np
import rasterio
import rasterio.transform
from affine import Affine
from rasterio.warp import Resampling, reproject, transform_bounds

from .geo import dataset_bounds_in_crs, read_overlap_region, transform_point, work_shift_to_dataset_shift
from .image import clahe_normalize
from .types import GlobalHypothesis, MetadataPrior


def detect_offset_at_resolution(src_offset, src_ref, overlap, work_crs, res,
                                template_radius_m=6000, coarse_offset=None,
                                search_margin_m=None, mask_mode="coastal_obia",
                                diagnostics_dir=None, min_ncc=0.3,
                                model_cache=None):
    """Detect the offset between two images at a given resolution via
    tiled ELoFTR matching.

    The previous NCC-based implementation (semantic-weighted land-mask
    NCC, CLAHE grayscale NCC, NMI) was repeatedly fooled by similar-shape
    coastlines: high correlation, wrong geography, no internal way to
    detect the ambiguity. ELoFTR's per-match MAD-derived agreement score
    abstains in those cases instead.

    The signature is preserved for caller compatibility:
      * ``template_radius_m``, ``coarse_offset``, ``search_margin_m`` and
        ``mask_mode`` are accepted but ignored — ELoFTR matches on the
        full overlap canvas and doesn't need a search-window prior.
      * ``min_ncc`` is reused as the agreement floor (typically 0.3).
      * ``diagnostics_dir`` is accepted; ELoFTR currently writes no
        per-call jpg, but pipeline-level diagnostics still land there.

    Returns ``(dx_m, dy_m, agreement)`` where ``dx_m`` / ``dy_m`` are the
    metric offset to apply (positive = shift target east / south,
    matching the historic NCC sign convention) and ``agreement`` is in
    [0, 1] with 1 = matches in perfect agreement. Returns
    ``(None, None, 0)`` when ELoFTR abstains.

    ``model_cache`` may be passed in for batch processing; otherwise a
    fresh ``ModelCache`` is created and disposed inside the call.
    """
    from .scale import (eloftr_translation_estimate,
                        _ELOFTR_MIN_AGREEMENT)

    arr_offset, _ = read_overlap_region(src_offset, overlap, work_crs, res)
    arr_ref, _ = read_overlap_region(src_ref, overlap, work_crs, res)

    result = eloftr_translation_estimate(
        arr_ref, arr_offset, res, model_cache=model_cache,
    )
    if result is None:
        return None, None, 0

    dx_eloftr, dy_eloftr, n_matches, agreement = result
    floor = max(_ELOFTR_MIN_AGREEMENT, float(min_ncc))
    if agreement < floor:
        print(f"  [coarse:eloftr] agreement {agreement:.2f} < {floor:.2f} "
              f"(matches={n_matches}); abstaining", flush=True)
        return None, None, 0

    # Sign convention reconciliation. The shared helper returns dx/dy in
    # the *shift-to-apply* convention (positive dx_m = shift target east
    # to align with reference). The historic NCC contract here is the
    # *amount-of-misalignment* convention (positive dx_m = target is
    # currently east of reference). pipeline.py's downstream translator
    # then converts via ``work_shift_to_dataset_shift`` to a westward
    # bounds shift (see step_coarse_offset's coarse_dx comment). Negate
    # so callers see the historic convention.
    dx_m = -dx_eloftr
    dy_m = -dy_eloftr
    print(f"  [coarse:eloftr] dx={dx_m:+.0f}m, dy={dy_m:+.0f}m "
          f"(matches={n_matches}, agreement={agreement:.2f})", flush=True)
    return dx_m, dy_m, agreement


# ---------------------------------------------------------------------------
# Global localization (merged from global_localization.py)
# ---------------------------------------------------------------------------

def _adaptive_res(bounds, requested_res, max_pixels=4096):
    width_m = max(1.0, bounds[2] - bounds[0])
    height_m = max(1.0, bounds[3] - bounds[1])
    return max(float(requested_res), width_m / max_pixels, height_m / max_pixels)


def _read_raw_resized(src, bounds, target_res):
    """Read a non-georeferenced image and resize to match expected bounds/res."""
    left, bottom, right, top = bounds
    width = max(1, int(round((right - left) / target_res)))
    height = max(1, int(round((top - bottom) / target_res)))
    raw = src.read(1).astype(np.float32)
    if raw.shape[0] != height or raw.shape[1] != width:
        raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_AREA)
    return raw


def _read_bounds(src, bounds, target_crs, target_res):
    left, bottom, right, top = bounds
    width = max(1, int(round((right - left) / target_res)))
    height = max(1, int(round((top - bottom) / target_res)))
    dst_transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)
    dst_array = np.zeros((height, width), dtype=np.float32)
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


def _prior_bounds_in_work_crs(prior: MetadataPrior, work_crs):
    bounds = prior.bounds()
    if bounds is None:
        return None
    west, south, east, north = bounds
    return transform_bounds(prior.crs, work_crs, west, south, east, north)


def _expand_bounds(bounds, margin_m, clamp_bounds=None):
    left, bottom, right, top = bounds
    expanded = (left - margin_m, bottom - margin_m, right + margin_m, top + margin_m)
    if clamp_bounds is None:
        return expanded
    return (
        max(clamp_bounds[0], expanded[0]),
        max(clamp_bounds[1], expanded[1]),
        min(clamp_bounds[2], expanded[2]),
        min(clamp_bounds[3], expanded[3]),
    )


def _prepare_search_bounds(ref_bounds, priors, work_crs, explicit_bounds=None):
    if explicit_bounds is not None:
        return explicit_bounds
    candidate = None
    for prior in priors or []:
        bounds = _prior_bounds_in_work_crs(prior, work_crs)
        if bounds is None:
            continue
        margin = max((bounds[2] - bounds[0]) * 0.5, (bounds[3] - bounds[1]) * 0.5, 5000.0)
        bounds = _expand_bounds(bounds, margin, clamp_bounds=ref_bounds)
        if candidate is None:
            candidate = bounds
        else:
            candidate = (
                max(ref_bounds[0], min(candidate[0], bounds[0])),
                max(ref_bounds[1], min(candidate[1], bounds[1])),
                min(ref_bounds[2], max(candidate[2], bounds[2])),
                min(ref_bounds[3], max(candidate[3], bounds[3])),
            )
    return candidate or ref_bounds


def _resize_to_eloftr_tile(arr_u8, max_dim=800):
    """Resize a CLAHE-normalized uint8 array to fit ELoFTR's input
    expectations (max ``max_dim`` on either axis, dimensions divisible
    by 14). Returns ``(resized_arr, scale_factor_used)``."""
    h, w = arr_u8.shape[:2]
    longest = max(h, w)
    scale = max_dim / longest if longest > max_dim else 1.0
    new_h = max(56, int(round(h * scale)))
    new_w = max(56, int(round(w * scale)))
    new_h = (new_h // 14) * 14 or 56
    new_w = (new_w // 14) * 14 or 56
    # Recompute the actual scale we got after rounding
    actual_scale_x = new_w / w
    actual_scale_y = new_h / h
    if (new_h, new_w) == (h, w):
        return arr_u8, 1.0, 1.0
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(arr_u8, (new_w, new_h), interpolation=interp)
    return resized, actual_scale_x, actual_scale_y


def _eloftr_global_search(target_arr, ref_arr, ref_res, target_res,
                          search_bounds, top_k=3,
                          tile_max_px=800,
                          conf_min=0.20, min_matches=30,
                          min_agreement=0.30,
                          model_cache=None):
    """Tile the reference and run ELoFTR(target, each tile) to find
    where the target's content lives in the reference frame.

    Replaces the multi-scale NCC grid search that the historic
    ``localize_to_reference`` used. Returns a list of
    ``(score, ref_pixel_origin, target_pixel_size, agreement, n_matches)``
    tuples sorted by ``score = agreement * sqrt(n_matches)`` descending.
    ``ref_pixel_origin`` is the (col, row) in ``ref_arr`` where the
    target's top-left would land if the median match holds; combined
    with ``ref_res`` and ``search_bounds`` the caller can convert that
    back to world coords.
    """
    from .scale import _run_eloftr_batch

    if min(target_arr.shape[:2]) < 32 or min(ref_arr.shape[:2]) < 32:
        return []

    target_u8 = clahe_normalize(target_arr)
    ref_u8 = clahe_normalize(ref_arr)

    # Resize target to fit within an ELoFTR tile (max tile_max_px).
    target_resized, tx_scale, ty_scale = _resize_to_eloftr_tile(target_u8, tile_max_px)
    th_r, tw_r = target_resized.shape

    # Tile the reference at the same shape as the resized target. ELoFTR
    # batches need same-shape pairs, so each tile is exactly (th_r, tw_r);
    # tiles that fall off the right/bottom edge get clamped.
    rh, rw = ref_u8.shape
    if rh < th_r or rw < tw_r:
        return []
    stride_r = max(8, th_r // 2)
    stride_c = max(8, tw_r // 2)
    row_starts = list(range(0, max(1, rh - th_r) + 1, stride_r))
    col_starts = list(range(0, max(1, rw - tw_r) + 1, stride_c))
    if row_starts and row_starts[-1] + th_r < rh:
        row_starts.append(rh - th_r)
    if col_starts and col_starts[-1] + tw_r < rw:
        col_starts.append(rw - tw_r)

    tiles = []
    for r0 in row_starts:
        for c0 in col_starts:
            ref_tile = ref_u8[r0:r0 + th_r, c0:c0 + tw_r]
            if ref_tile.shape != target_resized.shape:
                continue
            if np.mean(ref_tile > 0) < 0.05:
                continue
            tiles.append({"ref": ref_tile, "off": target_resized,
                          "r0": int(r0), "c0": int(c0)})
    if not tiles:
        return []

    cache = model_cache
    created_cache = False
    if cache is None:
        try:
            from .models import ModelCache
            from .models import get_torch_device as _get_dev
            cache = ModelCache(_get_dev())
            created_cache = True
        except Exception as exc:
            print(f"  [global:eloftr] ModelCache unavailable ({exc})")
            return []

    try:
        try:
            batch_results = _run_eloftr_batch(
                tiles, cache.eloftr, cache.device, batch_size=4)
        except Exception as exc:
            print(f"  [global:eloftr] inference failed ({exc})")
            return []
    finally:
        if created_cache:
            cache.close()

    candidates = []
    for tile, result in zip(tiles, batch_results):
        kp_ref, kp_tgt, conf = result
        if kp_ref is None or len(kp_ref) == 0:
            continue
        mask = conf > conf_min
        if mask.sum() < min_matches:
            continue
        kp_ref_f = kp_ref[mask].astype(np.float32)
        kp_tgt_f = kp_tgt[mask].astype(np.float32)
        # Per-match pixel deltas inside the (resized target, ref tile)
        # frame. We'll convert to ref-canvas coords below.
        dx_px = kp_ref_f[:, 0] - kp_tgt_f[:, 0]
        dy_px = kp_tgt_f[:, 1] - kp_ref_f[:, 1]
        med_dx = float(np.median(dx_px))
        med_dy = float(np.median(dy_px))
        mad_x = float(np.median(np.abs(dx_px - med_dx)))
        mad_y = float(np.median(np.abs(dy_px - med_dy)))
        agreement = 1.0 / (1.0 + mad_x + mad_y)
        if agreement < min_agreement:
            continue

        # Where does the target's top-left land in ref pixel coords?
        # Match: target pixel (x_t, y_t) → ref-tile pixel (x_r, y_r).
        # x_r - x_t = med_dx, y_t - y_r = med_dy → y_r = y_t - med_dy.
        # Target top-left (0, 0) → ref-tile pixel (med_dx, -med_dy).
        # Then add tile origin in the full ref array.
        target_origin_col_in_ref = tile["c0"] + med_dx
        target_origin_row_in_ref = tile["r0"] + (-med_dy)
        # Convert resized-target coords back to original target res.
        # The target was resized by (tx_scale, ty_scale); the resulting
        # bbox covers the original target's footprint at ``ref_res`` pixel
        # spacing. In the ref array (also at ref_res), the bbox width is
        # the original target width scaled by the *ratio of resolutions*
        # — i.e. the world-bbox of the target spans target_w_orig *
        # target_res metres, which at ref_res is target_w_orig *
        # target_res / ref_res pixels.
        # We compute that explicitly below from target_arr.shape.
        candidates.append({
            "score": float(agreement * np.sqrt(int(mask.sum()))),
            "ref_col": float(target_origin_col_in_ref),
            "ref_row": float(target_origin_row_in_ref),
            "agreement": float(agreement),
            "n_matches": int(mask.sum()),
            "target_resized_shape": (th_r, tw_r),
            "tx_scale": float(tx_scale),
            "ty_scale": float(ty_scale),
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:top_k]


def localize_to_reference(src_offset, src_ref, work_crs, priors=None, coarse_res=40.0,
                          top_k=3, mask_mode="coastal_obia", search_bounds=None,
                          model_cache=None):
    """Search the reference image for the target footprint via tiled
    ELoFTR matching.

    The historic implementation slid a rotated/rescaled target template
    across the reference's land-mask + stable + gradient channels via
    cv2.matchTemplate and combined the three score maps. That fell over
    when the reference contained multiple plausible coastlines (Bahrain
    vs Saudi mainland): the highest peak was often on the wrong one.
    The ELoFTR scan here detects that inconsistency through per-match
    MAD (the agreement floor); tiles with scattered matches are dropped
    entirely instead of contributing a high-score wrong hypothesis.
    Scale / rotation grid was retired alongside the NCC chain because
    ``detect_scale_rotation`` (also ELoFTR-based) is the dedicated stage
    for that correction.

    ``mask_mode`` is accepted for caller compatibility but no longer
    used; ELoFTR works on CLAHE-normalised grayscale.
    """
    priors = priors or []
    ref_bounds = dataset_bounds_in_crs(src_ref, work_crs)
    search_bounds = _prepare_search_bounds(ref_bounds, priors, work_crs, explicit_bounds=search_bounds)
    ref_res = _adaptive_res(search_bounds, coarse_res, max_pixels=4096)
    ref_arr, _ = _read_bounds(src_ref, search_bounds, work_crs, ref_res)

    target_bounds = dataset_bounds_in_crs(src_offset, work_crs)
    if target_bounds is None:
        for prior in priors:
            target_bounds = _prior_bounds_in_work_crs(prior, work_crs)
            if target_bounds is not None:
                break
    if target_bounds is None:
        return []
    target_res = max(ref_res, _adaptive_res(target_bounds, coarse_res, max_pixels=1024))
    if src_offset.crs is not None:
        target_arr, _ = _read_bounds(src_offset, target_bounds, work_crs, target_res)
    else:
        target_arr = _read_raw_resized(src_offset, target_bounds, target_res)

    candidates = _eloftr_global_search(
        target_arr, ref_arr, ref_res=ref_res, target_res=target_res,
        search_bounds=search_bounds, top_k=top_k,
        model_cache=model_cache,
    )

    # Convert each candidate to a GlobalHypothesis. Target's top-left in
    # ref pixel coords (``ref_col``, ``ref_row``) is at ELoFTR's
    # resized-target scale; the bounding box width in ref pixels is the
    # original target's bbox width scaled to ref's resolution.
    target_w_world = target_bounds[2] - target_bounds[0]
    target_h_world = target_bounds[3] - target_bounds[1]
    target_w_ref_px = target_w_world / ref_res
    target_h_ref_px = target_h_world / ref_res
    target_center_x = (target_bounds[0] + target_bounds[2]) / 2.0
    target_center_y = (target_bounds[1] + target_bounds[3]) / 2.0
    ref_full_bounds = dataset_bounds_in_crs(src_ref, work_crs)

    hypotheses = []
    for idx, cand in enumerate(candidates):
        # Target top-left in ref pixel coords, but at the resized scale.
        # Convert to ref-canvas coords at ref_res by accounting for the
        # resized-target → original-target ratio. Within a tile the
        # match was computed in pixels of the resized target (which is
        # itself in ref-res pixels of the tile); the top-left we
        # reported is already in ref-canvas pixel coords AT THE
        # RESIZED-TARGET SCALE. Multiply the *footprint* by the
        # ``original_target_size / resized_target_size`` ratio so the
        # hypothesis bbox covers the full target footprint at ref_res.
        tx_scale = cand["tx_scale"] or 1.0
        ty_scale = cand["ty_scale"] or 1.0
        ref_col = cand["ref_col"] / tx_scale
        ref_row = cand["ref_row"] / ty_scale
        left = search_bounds[0] + (ref_col * ref_res)
        top = search_bounds[3] - (ref_row * ref_res)
        right = left + (target_w_ref_px * ref_res)
        bottom = top - (target_h_ref_px * ref_res)
        margin = max((right - left) * 0.20, (top - bottom) * 0.20, 2000.0)
        left, bottom, right, top = _expand_bounds(
            (left, bottom, right, top), margin, clamp_bounds=ref_full_bounds)
        center_x = (left + right) / 2.0
        center_y = (bottom + top) / 2.0
        hypotheses.append(
            GlobalHypothesis(
                hypothesis_id=f"global_{idx}",
                score=cand["score"],
                source="global_template_search",
                left=left,
                bottom=bottom,
                right=right,
                top=top,
                dx_m=target_center_x - center_x,
                dy_m=center_y - target_center_y,
                scale_hint=1.0,
                rotation_hint_deg=0.0,
                work_crs=str(work_crs),
                diagnostics={
                    "search_res": ref_res,
                    "target_res": target_res,
                    "agreement": cand["agreement"],
                    "n_matches": cand["n_matches"],
                },
            )
        )
    return hypotheses


def translate_input_to_hypothesis(input_path, hypothesis: GlobalHypothesis, work_crs, *, suffix=".global_shift.tif"):
    """Create a temporary copy of *input_path* with a translated georeference."""
    with rasterio.open(input_path) as src:
        dx_src, dy_src = work_shift_to_dataset_shift(src, work_crs, hypothesis.dx_m, hypothesis.dy_m)
        profile = src.profile.copy()
        transform = src.transform
        profile["transform"] = Affine(
            transform.a, transform.b, transform.c + dx_src,
            transform.d, transform.e, transform.f + dy_src,
        )
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=os.path.dirname(input_path) or ".")
        os.close(tmp_fd)

        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.update_tags(**src.tags())
            for band_idx in range(1, src.count + 1):
                for _, window in src.block_windows(band_idx):
                    data = src.read(band_idx, window=window)
                    dst.write(data, band_idx, window=window)

    return tmp_path
