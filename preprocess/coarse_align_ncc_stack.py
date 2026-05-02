"""Stacked NCC -> ELoFTR -> phase-correlate coarse-align fallback.

Engaged by ``preprocess/georef.py::coarse_align_and_crop`` only when the
single-shot ELoFTR call abstains AND the camera profile flags USGS
corners as unreliable (KH-4A, KH-4B, KH-7, KH-8). KH-9's tighter pose
keeps the fast path; this fallback only fires for frames whose USGS
corner positioning may be 10-30 km off from truth.

Pipeline
--------
1. Stage A — bounded NCC top-K. Build binary land masks for ref/tgt on
   the shared coarse canvas, slide the target template across a sub-
   window of the reference (radius = ``coarse_ncc_search_radius_m``)
   via ``cv2.matchTemplate``, extract top-K NCC peaks under NMS.
2. Stage B — per-peak ELoFTR validation. For each NCC candidate, crop
   the implied overlap and rerun ``eloftr_translation_estimate`` on the
   shifted window. Score by ``agreement * sqrt(n_matches)``; pick the
   best clearing the same gates as the single-shot call.
3. Stage C — sub-pixel phase correlation. On the canvas-resolution
   arrays, crop a small window around the Stage-B peak and run
   ``_batch_phase_correlate`` for a sub-pixel residual. Floor on
   response; otherwise the Stage-B shift stands as-is.

The historic NCC chain (removed 2026-04 from ``align/coarse.py``) failed
because it picked the global NCC peak unconditionally, locking onto
Saudi-vs-Bahrain wrong-coast at high score. Two safeguards prevent
recurrence:
- the search is **bounded** to ±radius around the USGS-implied position
  so the wrong coast 60-80 km away is excluded from contention;
- the **ELoFTR per-peak validation** uses pixel-content keypoints to
  break ties between similar-shape candidates within the radius.

Returns ``(dx_m, dy_m, n_matches, agreement)`` matching the contract of
``align/scale.py::eloftr_translation_estimate``, or ``None`` on abstain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class _NCCPeak:
    """One candidate from Stage A. ``shift_dx_m`` / ``shift_dy_m`` are
    the metric offsets implied by the NCC peak (positive east / north,
    matching ``eloftr_translation_estimate``'s sign convention).

    Stage A uses **ref's content as the NCC template** and slides it
    across the (much larger) target canvas. Field meanings:

    - ``peak_r`` / ``peak_c``: top-left of the best ref-template match
      position in *target-canvas* pixel coords. This is "where in
      target the ref-content currently appears."
    - ``anchor_r`` / ``anchor_c``: top-left of ref's actual content
      bbox in canvas coords ("where the ref content really sits").
    - ``template_h`` / ``template_w``: size of the ref content bbox.

    Stage B's per-peak windows: ``arr_ref[anchor:anchor+template]`` for
    the reference; ``arr_tgt[peak:peak+template]`` for the target —
    the two share the same physical content if the NCC peak is
    correct.
    """

    shift_dx_m: float
    shift_dy_m: float
    ncc_score: float
    peak_r: int
    peak_c: int
    anchor_r: int
    anchor_c: int
    template_h: int
    template_w: int


def _land_mask_u8(arr: np.ndarray, mask_mode: str) -> np.ndarray:
    """Build a 0/255 uint8 land mask gated by actual content presence.

    Mirrors the boolean-mask conversion pattern from
    ``align/anchors.py::_island_presearch`` but adds a critical
    intersection with ``arr > 0``: the upstream OBIA / heuristic
    providers run a morphological CLOSE (lines ~64-68 of
    ``align/semantic_masking.py``) that fills nodata gaps inside the
    land mask. On sparse-coverage KH-4 panoramic strip orthos, the
    strip's data has small interior nodata holes (sparse-tiled
    GeoTIFF blocks), and CLOSE fills those holes — so the mask grows
    "phantom land" into nodata regions at the strip-coverage envelope
    rather than tracking actual coastline shape.

    Without this content gate, NCC's matchTemplate matches the
    *coverage envelope shape* (where the camera swath has data)
    against the reference's coastline — which is meaningless for
    alignment because the swath outline is set by the camera's path,
    not the ground. The gate restores the "shape of actual land" that
    Stage A is supposed to be correlating.
    """
    from align.image import make_land_mask
    land_bool = make_land_mask(arr, mode=mask_mode) > 0
    valid = arr > 0
    land_bool = land_bool & valid
    return (land_bool.astype(np.uint8)) * 255


def _ncc_top_k_peaks(
    ref_land_u8: np.ndarray,
    tgt_land_u8: np.ndarray,
    *,
    radius_px: int,
    top_k: int,
    nms_px: int,
    min_ncc: float,
) -> list[_NCCPeak]:
    """Slide the ref-content template across the tgt canvas, extract
    top-K NCC peaks under NMS within a bounded search around ref's
    actual canvas position.

    Both arrays share the canvas (union of ref+tgt at coarse_res). The
    KH-9 reference is small (~50 km × 100 km) and its land content
    occupies a tight bbox; the KH-4 target ortho is wide (~250 km ×
    60 km) and its land content can span almost the full canvas. So
    we use **ref as the template** and **tgt as the search image** —
    cv2.matchTemplate then has many valid positions to slide across.

    The output peak ``(peak_r, peak_c)`` is the top-left position in
    tgt-canvas where the ref's land shape best matches. To align tgt
    onto ref, we shift tgt by ``(anchor - peak)`` in pixel space:
    that brings tgt's "ref-shaped" content onto ref's actual canvas
    position.

    ``radius_px`` bounds the NCC peak search to ±radius_px (in canvas
    pixels) around ref's anchor — i.e. we only accept candidate peak
    positions that imply a shift of less than ``radius_m``. This is
    the safeguard that prevents the historic wrong-coast failure.
    """
    H, W = ref_land_u8.shape
    if tgt_land_u8.shape != (H, W):
        return []

    if float(np.mean(ref_land_u8 > 0)) < 0.01:
        print(f"  [coarse:ncc] ref land mask <1% coverage — abstain")
        return []
    if float(np.mean(tgt_land_u8 > 0)) < 0.01:
        print(f"  [coarse:ncc] tgt land mask <1% coverage — abstain")
        return []

    # Template = ref's land content bbox.
    ref_rows, ref_cols = np.nonzero(ref_land_u8 > 0)
    if ref_rows.size == 0:
        return []
    rr0, rr1 = int(ref_rows.min()), int(ref_rows.max() + 1)
    rc0, rc1 = int(ref_cols.min()), int(ref_cols.max() + 1)
    template = ref_land_u8[rr0:rr1, rc0:rc1]
    th, tw = template.shape
    if th < 16 or tw < 16:
        print(f"  [coarse:ncc] ref land bbox too small ({th}x{tw}) — abstain")
        return []
    if th > H or tw > W:
        print(f"  [coarse:ncc] ref template ({th}x{tw}) larger than "
              f"canvas ({H}x{W}) — abstain")
        return []

    # Slide ref-template across tgt-canvas. Output shape (H-th+1, W-tw+1).
    nc_map = cv2.matchTemplate(tgt_land_u8, template, cv2.TM_CCOEFF_NORMED)

    # Bounded peak search: only accept matches within ±radius_px of
    # ref's actual anchor (rr0, rc0) on the canvas. A peak at the
    # anchor itself implies "zero shift"; peaks within radius imply
    # shifts of at most ``radius_m``.
    rmin = max(0, rr0 - radius_px)
    rmax = min(nc_map.shape[0] - 1, rr0 + radius_px)
    cmin = max(0, rc0 - radius_px)
    cmax = min(nc_map.shape[1] - 1, rc0 + radius_px)
    if rmin > rmax or cmin > cmax:
        print(f"  [coarse:ncc] bounded search window empty around "
              f"anchor=({rr0},{rc0}) (radius={radius_px}px, "
              f"map_shape={nc_map.shape})")
        return []

    bounded = np.full_like(nc_map, -np.inf, dtype=np.float32)
    bounded[rmin:rmax + 1, cmin:cmax + 1] = nc_map[rmin:rmax + 1, cmin:cmax + 1]

    peaks: list[_NCCPeak] = []
    # Track the unconditional best peak across iterations so we can fall
    # back to it when nothing clears ``min_ncc``. Cross-temporal land
    # masks (1968 KH-4B vs modern reference) often produce weak NCC
    # signals on shallow shorelines / mudflats; the threshold can throw
    # out a real peak that downstream ELoFTR validation would still
    # confirm. The threshold continues to gate which peaks ``top_k``
    # retains, but we never abstain solely because the strongest peak
    # sat just below ``min_ncc``.
    best_below_floor: Optional[_NCCPeak] = None
    best_below_score: float = -math.inf
    for _ in range(top_k):
        peak_val = float(bounded.max())
        if not math.isfinite(peak_val):
            break
        below_floor = peak_val < min_ncc
        if below_floor and peaks:
            # Already have at least one above-floor peak — stop expanding.
            break
        flat_idx = int(np.argmax(bounded))
        peak_r, peak_c = divmod(flat_idx, bounded.shape[1])
        # Sub-pixel parabolic refinement on the 3×3 neighbourhood of
        # the integer peak. cv2.matchTemplate returns NCC at integer
        # offsets; the true peak lies between cells. Fitting a 1-D
        # parabola along each axis and taking its vertex gives ~10×
        # better precision (50 m → 5 m at coarse_res=50 m/px) and is
        # critical for KH-4: a 0.5-px snap at +44 km offset costs
        # ~25 m of westward bias on the final mosaic.
        sub_dr = 0.0
        sub_dc = 0.0
        if (1 <= peak_r <= nc_map.shape[0] - 2
                and 1 <= peak_c <= nc_map.shape[1] - 2):
            # Use the original (unbounded) NCC map for the 3×3 window
            # so the bounded -inf sentinel doesn't leak into the fit.
            n = nc_map[peak_r - 1: peak_r + 2, peak_c - 1: peak_c + 2]
            denom_c = 2.0 * n[1, 1] - n[1, 0] - n[1, 2]
            denom_r = 2.0 * n[1, 1] - n[0, 1] - n[2, 1]
            if abs(denom_c) > 1e-6:
                sub_dc = 0.5 * (n[1, 2] - n[1, 0]) / denom_c
                if abs(sub_dc) > 0.5:    # parabola breakdown — discard
                    sub_dc = 0.0
            if abs(denom_r) > 1e-6:
                sub_dr = 0.5 * (n[2, 1] - n[0, 1]) / denom_r
                if abs(sub_dr) > 0.5:
                    sub_dr = 0.0
        # Implied tgt shift to align with ref. Tgt's matching content
        # currently sits at (peak_r + sub_dr, peak_c + sub_dc) in canvas;
        # ref's actual content sits at (anchor_r, anchor_c). To move
        # tgt onto ref, shift its content from peak → anchor.
        #
        # Sign convention (matching `eloftr_translation_estimate`):
        # positive dx_m = east (canvas cols increase); positive dy_m =
        # north (canvas rows decrease, so the row delta is negated).
        dc_px = float(rc0) - (float(peak_c) + sub_dc)   # east
        dr_px = (float(peak_r) + sub_dr) - float(rr0)   # north
        candidate = _NCCPeak(
            shift_dx_m=float(dc_px),     # caller multiplies by coarse_res
            shift_dy_m=float(dr_px),
            ncc_score=peak_val,
            peak_r=int(peak_r), peak_c=int(peak_c),
            anchor_r=int(rr0),  anchor_c=int(rc0),
            template_h=int(th), template_w=int(tw),
        )
        if below_floor:
            if peak_val > best_below_score:
                best_below_floor = candidate
                best_below_score = peak_val
            break
        peaks.append(candidate)
        if peak_val > best_below_score:
            best_below_floor = candidate
            best_below_score = peak_val
        rr, cc = np.ogrid[:bounded.shape[0], :bounded.shape[1]]
        disk = (rr - peak_r) ** 2 + (cc - peak_c) ** 2 <= nms_px ** 2
        bounded[disk] = -np.inf
    if not peaks and best_below_floor is not None:
        # Floor produced nothing; emit the single best peak as a weak
        # candidate so downstream ELoFTR validation + strip-prior tie-
        # breaker can still gate it on real correspondence quality.
        print(f"  [coarse:ncc] no peak above floor {min_ncc:.2f}; "
              f"forwarding best below-floor peak ncc={best_below_floor.ncc_score:.2f} "
              f"for ELoFTR validation")
        peaks.append(best_below_floor)
    return peaks


def _validate_peak_with_eloftr(
    arr_ref: np.ndarray,
    arr_tgt: np.ndarray,
    peak: _NCCPeak,
    coarse_res: float,
    *,
    model_cache,
    window_px: int,
    target_path: Optional[str] = None,
    reference_path: Optional[str] = None,
    work_crs=None,
    union_bounds: Optional[tuple[float, float, float, float]] = None,
    validation_window_m: float = 4000.0,
    fine_res_m: float = 10.0,
) -> Optional[tuple[float, float, int, float]]:
    """Validate one Stage A NCC candidate by running single-window
    ELoFTR on a small bounded sub-window of both images.

    Two modes:
    - **Fine-res re-warp** (preferred). When ``target_path``,
      ``reference_path``, ``work_crs`` and ``union_bounds`` are all
      supplied, re-warp both source files to ``fine_res_m`` (default
      10 m/px) on a ``validation_window_m``-side window centred on the
      candidate's implied overlap. Reuses
      ``align/geo.py::read_overlap_region`` (in-memory rasterio
      reproject + LRU cache + scratch-cleaned-sidecar handling).
      ELoFTR sees a 400×400 px window with real coastline texture and
      can find ≥30 keypoints — the canvas-array crop at 50 m/px could
      not.
    - **Coarse-array crop fallback**. When any path is missing (legacy
      callers, tests), crop fixed-size windows from the existing
      50 m/px canvas arrays. Same behaviour as before this fix.

    Returns the *combined* shift (NCC + ELoFTR residual), match count,
    and agreement — or ``None`` on abstain.
    """
    from align.scale import eloftr_translation_estimate

    use_fine_rewarp = (
        target_path and reference_path
        and work_crs is not None and union_bounds is not None
    )

    result = None
    if use_fine_rewarp:
        result = _eloftr_at_fine_res(
            target_path, reference_path, work_crs, union_bounds,
            peak, coarse_res, fine_res_m, validation_window_m,
            model_cache,
        )
    if result is None:
        # Fallback: canvas-array crop centred on the candidate's peak
        # canvas position. Used when paths aren't supplied OR when the
        # fine-res re-warp returned empty windows (sparse-strip KH-4
        # ortho — the strip's actual content doesn't fully fill the
        # canvas-warped land mask area).
        result = _eloftr_on_canvas_arrays(
            arr_ref, arr_tgt, peak, coarse_res, model_cache, window_px,
        )
    if result is None:
        return None
    dx_resid_m, dy_resid_m, n_matches, agreement = result

    dx_m = peak.shift_dx_m * coarse_res + dx_resid_m
    dy_m = peak.shift_dy_m * coarse_res + dy_resid_m
    return dx_m, dy_m, int(n_matches), float(agreement)


def _eloftr_on_canvas_arrays(arr_ref, arr_tgt, peak, coarse_res,
                             model_cache, window_px):
    """Original 50 m/px canvas-array crop path. Used when source paths
    aren't threaded through (tests, legacy callers)."""
    from align.scale import eloftr_translation_estimate
    H, W = arr_ref.shape
    ref_cr = peak.anchor_r + peak.template_h // 2
    ref_cc = peak.anchor_c + peak.template_w // 2
    tgt_cr = peak.peak_r + peak.template_h // 2
    tgt_cc = peak.peak_c + peak.template_w // 2
    half = window_px // 2
    rr0 = max(0, ref_cr - half)
    rr1 = min(H, rr0 + window_px)
    cc0 = max(0, ref_cc - half)
    cc1 = min(W, cc0 + window_px)
    rh = rr1 - rr0
    cw = cc1 - cc0
    if rh < 64 or cw < 64:
        return None
    tr0 = max(0, tgt_cr - half)
    tr1 = min(H, tr0 + rh)
    tc0 = max(0, tgt_cc - half)
    tc1 = min(W, tc0 + cw)
    if (tr1 - tr0) < rh or (tc1 - tc0) < cw:
        return None
    ref_window = arr_ref[rr0:rr1, cc0:cc1]
    tgt_window = arr_tgt[tr0:tr1, tc0:tc1]
    return eloftr_translation_estimate(
        ref_window, tgt_window, coarse_res, model_cache=model_cache,
    )


def _eloftr_at_fine_res(target_path, reference_path, work_crs,
                        union_bounds, peak, coarse_res, fine_res_m,
                        validation_window_m, model_cache):
    """Re-warp both source files to ``fine_res_m`` on a small bounded
    window centred on the candidate's implied overlap, then run
    single-window ELoFTR. ``work_crs`` is the metric CRS the canvas
    was built in (UTM); ``union_bounds`` lets us convert canvas pixel
    coords back to metric coords.

    ``read_overlap_region`` handles the rasterio reproject + LRU cache
    + scratch-cleaned-sidecar swap-in. We only need to compute the
    metric bounds for both ref and tgt windows.
    """
    from align.scale import eloftr_translation_estimate
    from align.geo import read_overlap_region
    import rasterio

    union_left, union_bottom, union_right, union_top = union_bounds

    # Centre of the matched template in canvas pixel coords:
    # - ref content lives at (anchor_r, anchor_c) top-left
    # - tgt content (matching shape) lives at (peak_r, peak_c) top-left
    ref_cr_px = float(peak.anchor_r) + peak.template_h / 2.0
    ref_cc_px = float(peak.anchor_c) + peak.template_w / 2.0
    tgt_cr_px = float(peak.peak_r) + peak.template_h / 2.0
    tgt_cc_px = float(peak.peak_c) + peak.template_w / 2.0

    # Convert canvas pixel to metric coords. Canvas row 0 = union_top
    # (rasterio convention: row increases downward → y decreases).
    def canvas_px_to_metric(row_px, col_px):
        x = union_left + col_px * coarse_res
        y = union_top - row_px * coarse_res
        return x, y

    ref_cx, ref_cy = canvas_px_to_metric(ref_cr_px, ref_cc_px)
    tgt_cx, tgt_cy = canvas_px_to_metric(tgt_cr_px, tgt_cc_px)
    half = validation_window_m / 2.0

    ref_bounds = (ref_cx - half, ref_cy - half, ref_cx + half, ref_cy + half)
    tgt_bounds = (tgt_cx - half, tgt_cy - half, tgt_cx + half, tgt_cy + half)

    try:
        with rasterio.open(reference_path) as src_ref:
            ref_arr, _ = read_overlap_region(
                src_ref, ref_bounds, work_crs, fine_res_m,
            )
        with rasterio.open(target_path) as src_tgt:
            tgt_arr, _ = read_overlap_region(
                src_tgt, tgt_bounds, work_crs, fine_res_m,
            )
    except Exception as exc:
        print(f"  [coarse:ncc-eloftr] fine-res re-warp failed ({exc}); "
              f"abstaining this candidate")
        return None

    # Validity guard: if either window has too few non-zero pixels at
    # fine_res, ELoFTR will abstain anyway — short-circuit.
    if (np.count_nonzero(ref_arr > 0) < 100
            or np.count_nonzero(tgt_arr > 0) < 100):
        return None

    return eloftr_translation_estimate(
        ref_arr, tgt_arr, fine_res_m, model_cache=model_cache,
    )


def _phase_correlate_at_fine_res(
    target_path, reference_path, work_crs, union_bounds,
    refined_shift_m, fine_res_m, fine_window_m, min_resp,
    torch, batch_phase_correlate,
    *, ref_centre_xy=None,
):
    """Stage C at fine res. Re-warp both source files to ``fine_res_m``
    on a small window centred on ``ref_centre_xy`` in ``work_crs``
    metric coords (where ref content actually lives in the canvas).
    The target is read at metric bounds shifted by Stage-B's
    ``refined_shift_m``, so the two windows show the same physical
    land. Phase correlation then measures the residual sub-pixel
    disagreement at fine resolution.

    ``ref_centre_xy`` defaults to the canvas centre when not supplied
    — but for KH-4 strips the canvas centre is OUTSIDE the smaller
    KH-9 reference's footprint, so callers must pass the actual ref
    centroid for the windows to contain content.
    """
    from align.geo import read_overlap_region
    import rasterio

    if ref_centre_xy is None:
        union_left, union_bottom, union_right, union_top = union_bounds
        ref_cx = 0.5 * (union_left + union_right)
        ref_cy = 0.5 * (union_bottom + union_top)
    else:
        ref_cx, ref_cy = ref_centre_xy
    half = fine_window_m / 2.0
    ref_bounds = (ref_cx - half, ref_cy - half,
                  ref_cx + half, ref_cy + half)
    # Target window: shifted by the inverse of the Stage-B alignment so
    # the same physical land lands in both windows. Stage B says
    # "shift target by (dx_m, dy_m) east/north to align"; equivalently,
    # in *metric* coords, the target's content currently sits at
    # (ref_cx - dx_m, ref_cy - dy_m) when we want it at (ref_cx, ref_cy).
    dx_m, dy_m = refined_shift_m
    tgt_bounds = (ref_cx - half - dx_m, ref_cy - half - dy_m,
                  ref_cx + half - dx_m, ref_cy + half - dy_m)
    try:
        with rasterio.open(reference_path) as src_ref:
            ref_arr, _ = read_overlap_region(
                src_ref, ref_bounds, work_crs, fine_res_m,
            )
        with rasterio.open(target_path) as src_tgt:
            tgt_arr, _ = read_overlap_region(
                src_tgt, tgt_bounds, work_crs, fine_res_m,
            )
    except Exception as exc:
        print(f"  [coarse:phase] fine-res re-warp failed ({exc}); skipping")
        return 0.0, 0.0
    if (np.count_nonzero(ref_arr > 0) < 100
            or np.count_nonzero(tgt_arr > 0) < 100):
        print(f"  [coarse:phase] fine windows have <100 valid pixels; skipping")
        return 0.0, 0.0

    h, w = min(ref_arr.shape[0], tgt_arr.shape[0]), min(ref_arr.shape[1], tgt_arr.shape[1])
    if h < 64 or w < 64:
        return 0.0, 0.0
    ref_win = ref_arr[:h, :w].astype(np.float32)
    tgt_win = tgt_arr[:h, :w].astype(np.float32)
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    ref_t = torch.as_tensor((ref_win * win)[None, ...])
    tgt_t = torch.as_tensor((tgt_win * win)[None, ...])
    shifts, responses, valid = batch_phase_correlate(
        ref_t, tgt_t, min_resp=min_resp,
    )
    if not bool(valid[0]):
        print(f"  [coarse:phase] fine-res response {float(responses[0]):.3f} "
              f"below floor {min_resp:.3f}; using Stage-B shift unchanged")
        return 0.0, 0.0
    dx_resid_m = float(shifts[0, 0]) * fine_res_m
    dy_resid_m = -float(shifts[0, 1]) * fine_res_m
    print(f"  [coarse:phase] fine-res sub-pixel residual: "
          f"dx={dx_resid_m:+.1f}m, dy={dy_resid_m:+.1f}m, "
          f"response={float(responses[0]):.3f}")
    return dx_resid_m, dy_resid_m


def _phase_correlate_finalize(
    arr_ref: np.ndarray,
    arr_tgt: np.ndarray,
    coarse_res: float,
    refined_shift_m: tuple[float, float],
    *,
    fine_window_m: float,
    min_resp: float,
    target_path: Optional[str] = None,
    reference_path: Optional[str] = None,
    work_crs=None,
    union_bounds: Optional[tuple[float, float, float, float]] = None,
    fine_res_m: float = 10.0,
    ref_centre_xy: Optional[tuple[float, float]] = None,
    anchor_canvas_rc: Optional[tuple[float, float]] = None,
) -> tuple[float, float]:
    """Sub-pixel residual via phase correlation on a small window
    around the Stage-B peak.

    Two modes (mirrors :func:`_validate_peak_with_eloftr`):
    - **Fine-res re-warp**: when source paths + work_crs + union_bounds
      are supplied, re-warp both files to ``fine_res_m`` (default
      10 m/px) on a ``fine_window_m``-side window centred on the
      Stage-B-aligned position. Phase correlation at fine res gives
      sub-pixel-of-fine-res precision (10 m × 0.1 px ≈ 1 m).
    - **Canvas-array fallback**: Hann-window the existing 50 m/px
      canvas crop centred on the canvas mid-point. Behavior pre-fix.

    Returns ``(dx_residual_m, dy_residual_m)``. Returns ``(0.0, 0.0)``
    on low response — Stage B is already a confident result, so a
    failed Stage C is not an abstain.
    """
    try:
        import torch
        from align.qa import _batch_phase_correlate
    except Exception as exc:
        print(f"  [coarse:phase] torch unavailable ({exc}); skipping refinement")
        return 0.0, 0.0

    use_fine_rewarp = (
        target_path and reference_path
        and work_crs is not None and union_bounds is not None
    )
    if use_fine_rewarp:
        out = _phase_correlate_at_fine_res(
            target_path, reference_path, work_crs, union_bounds,
            refined_shift_m, fine_res_m, fine_window_m, min_resp,
            torch, _batch_phase_correlate,
            ref_centre_xy=ref_centre_xy,
        )
        # Fall through to the canvas-array path when fine-res returned
        # zero residual due to sparse-strip empty windows. The canvas
        # arrays at coarse_res were already validated by Stage A as
        # having content at the matched position.
        if out != (0.0, 0.0):
            return out
    H, W = arr_ref.shape
    side = max(64, int(round(fine_window_m / coarse_res)))
    side = min(side, H, W)
    # Centre the window on the anchor canvas position (where ref's
    # content sits) when supplied — fall back to canvas centre when not.
    # The KH-9 reference covers a small portion of the union canvas
    # (~50 km × 100 km in a 250 km × 100 km canvas for KH-4 strips), so
    # canvas centre is often outside ref's footprint.
    if anchor_canvas_rc is not None:
        cr = int(round(anchor_canvas_rc[0]))
        cc = int(round(anchor_canvas_rc[1]))
    else:
        cr = H // 2
        cc = W // 2
    half = side // 2
    rr0 = max(0, cr - half)
    rr1 = min(H, rr0 + side)
    cc0 = max(0, cc - half)
    cc1 = min(W, cc0 + side)
    side_h = rr1 - rr0
    side_w = cc1 - cc0
    if side_h < 64 or side_w < 64:
        return 0.0, 0.0

    # Apply Stage-B's metric shift to the target window before phase
    # correlation: shift in canvas-pixel space via cv2.warpAffine so
    # the residual we measure is sub-pixel only.
    dx_px = refined_shift_m[0] / coarse_res
    dy_px = -refined_shift_m[1] / coarse_res     # row sign flip
    M = np.float32([[1, 0, dx_px], [0, 1, dy_px]])
    tgt_shifted = cv2.warpAffine(
        arr_tgt, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    ref_window = arr_ref[rr0:rr1, cc0:cc1].astype(np.float32)
    tgt_window = tgt_shifted[rr0:rr1, cc0:cc1].astype(np.float32)
    ref_valid_pct = float((ref_window > 0).mean()) * 100
    tgt_valid_pct = float((tgt_window > 0).mean()) * 100
    if ref_valid_pct < 5.0 or tgt_valid_pct < 5.0:
        print(f"  [coarse:phase] canvas-array windows insufficient "
              f"(ref={ref_valid_pct:.1f}%, tgt={tgt_valid_pct:.1f}%); "
              f"skipping")
        return 0.0, 0.0
    print(f"  [coarse:phase] canvas-array windows: ref={ref_valid_pct:.1f}% "
          f"tgt={tgt_valid_pct:.1f}% at anchor=({cr},{cc}) side={side_h}x{side_w}")
    win = np.outer(np.hanning(side_h), np.hanning(side_w)).astype(np.float32)

    ref_t = torch.as_tensor((ref_window * win)[None, ...])
    tgt_t = torch.as_tensor((tgt_window * win)[None, ...])
    shifts, responses, valid = _batch_phase_correlate(
        ref_t, tgt_t, min_resp=min_resp,
    )
    if not bool(valid[0]):
        print(f"  [coarse:phase] response {float(responses[0]):.3f} below "
              f"floor {min_resp:.3f}; using Stage-B shift unchanged")
        return 0.0, 0.0
    dx_resid_m = float(shifts[0, 0]) * coarse_res
    dy_resid_m = -float(shifts[0, 1]) * coarse_res     # row sign flip
    print(f"  [coarse:phase] sub-pixel residual: dx={dx_resid_m:+.1f}m, "
          f"dy={dy_resid_m:+.1f}m, response={float(responses[0]):.3f}")
    return dx_resid_m, dy_resid_m


def run_stacked_coarse_align(
    arr_ref: np.ndarray,
    arr_tgt: np.ndarray,
    coarse_res: float,
    *,
    work_crs,
    union_bounds: tuple[float, float, float, float],
    target_bbox_wgs: Optional[tuple[float, float, float, float]],
    params,
    model_cache=None,
    target_path: Optional[str] = None,
    reference_path: Optional[str] = None,
    neighbour_shifts_m: Optional[list[tuple[float, float]]] = None,
    single_shot_dxdy_m: Optional[tuple[float, float]] = None,
) -> Optional[tuple[float, float, int, float]]:
    """Stacked NCC -> ELoFTR -> phase-correlate fallback.

    Args:
        arr_ref / arr_tgt: float32 arrays on a shared canvas at
            ``coarse_res`` m/px (output of ``coarse_align_and_crop``'s
            gdalwarp step).
        coarse_res: canvas resolution, metres per pixel.
        work_crs: pyproj CRS of the canvas (UTM or whatever the caller
            chose).
        union_bounds: (left, bottom, right, top) of the canvas in
            ``work_crs`` metric coords. Used for log diagnostics; the
            bounded NCC search itself works in canvas-pixel space.
        target_bbox_wgs: USGS corner bbox of the target in EPSG:4326,
            ``(west, south, east, north)``. May be ``None``; in that
            case the function uses the target's existing canvas
            footprint as the search anchor (already implicitly the
            USGS-corner-derived position because ``coarse_align_and_
            crop`` warped using the target's geotransform).
        params: ``align.params.AlignParams`` — provides camera knobs.
        model_cache: optional shared ELoFTR model cache; lazy-created
            if ``None``.

    Returns:
        ``(dx_m, dy_m, n_matches, agreement)`` matching the single-shot
        ``eloftr_translation_estimate`` contract, or ``None`` on
        abstain.
    """
    radius_m = float(params.camera.coarse_ncc_search_radius_m)
    if radius_m <= 0:
        return None
    radius_px = int(round(radius_m / coarse_res))
    nms_px = max(1, int(round(params.camera.coarse_ncc_nms_distance_m / coarse_res)))
    top_k = int(params.camera.coarse_ncc_top_k)
    min_ncc = float(params.camera.coarse_ncc_min_ncc)
    mask_mode = str(params.camera.coarse_ncc_mask_mode)
    validation_window_m = float(params.camera.coarse_ncc_validation_window_m)
    fine_window_m = float(params.camera.coarse_ncc_fine_window_m)
    fine_resp_min = float(params.camera.coarse_ncc_fine_resp_min)
    fine_res_m = float(getattr(params.camera, "coarse_ncc_fine_res_m", 10.0))
    strip_coherence_max_m = float(
        getattr(params.camera, "coarse_ncc_strip_coherence_max_m", 5000.0))
    # Per-axis strip-coherence gate (KH satellites fly near-polar orbits, so
    # ``dx`` is cross-strip and ``dy`` is along-strip). Wrong-coast confusion
    # shows up as cross-strip drift; legitimate USGS-corner errors along the
    # orbit track show up as along-strip drift. Splitting the gate lets the
    # along-strip threshold loosen for KH-4 / KH-7 strips without weakening
    # cross-strip wrong-coast safety. ``None`` means inherit the legacy
    # isotropic threshold (so older profiles keep current behaviour).
    _cross_max_cfg = getattr(params.camera, "coarse_ncc_strip_coherence_cross_max_m", None)
    _along_max_cfg = getattr(params.camera, "coarse_ncc_strip_coherence_along_max_m", None)
    strip_cross_max_m = float(_cross_max_cfg) if _cross_max_cfg is not None else strip_coherence_max_m
    strip_along_max_m = float(_along_max_cfg) if _along_max_cfg is not None else strip_coherence_max_m

    H, W = arr_ref.shape
    if arr_tgt.shape != (H, W):
        print(f"  [coarse:ncc] shape mismatch ref={arr_ref.shape} "
              f"tgt={arr_tgt.shape}; abstain")
        return None

    print(f"  [coarse:ncc] stacked fallback engaging "
          f"(radius={radius_m/1000:.0f}km, top_k={top_k}, "
          f"min_ncc={min_ncc:.2f}, mask_mode={mask_mode})")

    # Stage A — bounded NCC top-K. The USGS-implied origin is implicit
    # in arr_tgt's canvas position (coarse_align_and_crop already warped
    # the target using its USGS-corner geotransform, so any offset in
    # the canvas is the offset we're recovering).
    #
    # Mask-mode robustness: try the configured mode first. If it returns
    # no peaks above floor, retry with the alternate mode — coastal_obia
    # can over-suppress mudflats / shoals on southern-Bahrain KH-4
    # frames, while heuristic (multi-Otsu) preserves them. Falling back
    # to heuristic gives the search a second chance to find a shape
    # peak rather than abstain.
    fallback_modes = [mask_mode]
    if mask_mode == "coastal_obia":
        fallback_modes.append("heuristic")
    elif mask_mode == "heuristic":
        fallback_modes.append("coastal_obia")
    peaks: list[_NCCPeak] = []
    chosen_mode = mask_mode
    for mode in fallback_modes:
        ref_land_u8 = _land_mask_u8(arr_ref, mode)
        tgt_land_u8 = _land_mask_u8(arr_tgt, mode)
        peaks = _ncc_top_k_peaks(
            ref_land_u8, tgt_land_u8,
            radius_px=radius_px, top_k=top_k, nms_px=nms_px,
            min_ncc=min_ncc,
        )
        if peaks:
            chosen_mode = mode
            if mode != mask_mode:
                print(f"  [coarse:ncc] retried with mask_mode={mode} "
                      f"after {mask_mode} produced no peaks")
            break
    if not peaks:
        # Reverted 2026-04-27 (validation v26): a "no peaks → trust the
        # validated strip prior with N=1" fallback shipped a +30km shift
        # from DA023 to its strip neighbours and made DA025 score regress
        # 791 → 2282 because intra-strip USGS-corner errors can vary
        # 30+ km between adjacent KH-4B sub-frames (well above the strip-
        # coherence threshold). Without per-frame corroboration (a second
        # validated neighbour or a weak ELoFTR result that agrees with
        # the prior), N=1 priors are unsafe to apply blindly. Re-enable
        # only behind a corroboration gate.
        print(f"  [coarse:ncc] no peak above floor {min_ncc:.2f} "
              f"within ±{radius_m/1000:.0f}km (tried modes: "
              f"{fallback_modes}) — abstain")
        return None

    for i, p in enumerate(peaks):
        dx_m_a = p.shift_dx_m * coarse_res
        dy_m_a = p.shift_dy_m * coarse_res
        print(f"  [coarse:ncc] candidate {i}: "
              f"dx={dx_m_a:+.0f}m, dy={dy_m_a:+.0f}m, ncc={p.ncc_score:.2f}")

    # Stage B — ELoFTR per-peak validation; pick the highest
    # `agreement * sqrt(n_matches)` score that clears ELoFTR's gates.
    # When source paths are supplied, validation re-warps both files to
    # ``fine_res_m`` (default 10 m/px) on a small bounded window so
    # ELoFTR has enough pixels to find ≥30 keypoints — the canvas
    # 50 m/px crop wasn't producing enough matches on cross-temporal
    # KH-4 vs KH-9 windows.
    validation_window_px = max(64, int(round(validation_window_m / coarse_res)))
    best = None
    best_score = -1.0
    best_idx = -1
    for i, p in enumerate(peaks):
        result = _validate_peak_with_eloftr(
            arr_ref, arr_tgt, p, coarse_res,
            model_cache=model_cache, window_px=validation_window_px,
            target_path=target_path, reference_path=reference_path,
            work_crs=work_crs, union_bounds=union_bounds,
            validation_window_m=validation_window_m,
            fine_res_m=fine_res_m,
        )
        if result is None:
            print(f"  [coarse:ncc-eloftr] candidate {i}: ELoFTR abstained")
            continue
        dx_m, dy_m, n_matches, agreement = result
        score = agreement * math.sqrt(n_matches)
        print(f"  [coarse:ncc-eloftr] candidate {i} validated: "
              f"matches={n_matches}, agreement={agreement:.2f}, "
              f"score={score:.2f} (combined dx={dx_m:+.0f}m, dy={dy_m:+.0f}m)")
        if score > best_score:
            best_score = score
            best = (dx_m, dy_m, n_matches, agreement)
            best_idx = i
    if best is None and len(peaks) >= 3:
        # NCC self-consistency: when ≥3 NCC peaks cluster tightly within
        # the per-axis strip-coherence bounds, the cluster centroid is
        # itself evidence of a real shift even when per-peak ELoFTR
        # can't refine. Cross-temporal KH-4B vs KH-9 sometimes produces
        # high coarse-NCC at 50 m/px but dispersed fine matches at the
        # ELoFTR resolution, leaving multi-peak agreement as the only
        # available signal. When a strip prior is supplied the cluster
        # must also agree with it within the per-axis bounds — without
        # that gate, wrong-coast NCC peaks (consistent among themselves
        # because they're matching a similar shoreline shape elsewhere)
        # could be accepted. Validate at the peak closest to the cluster
        # median so phase refinement still anchors on a real candidate.
        peak_dx_arr = np.array([p.shift_dx_m * coarse_res for p in peaks])
        peak_dy_arr = np.array([p.shift_dy_m * coarse_res for p in peaks])
        cross_spread = float(peak_dx_arr.max() - peak_dx_arr.min())
        along_spread = float(peak_dy_arr.max() - peak_dy_arr.min())
        if cross_spread <= strip_cross_max_m and along_spread <= strip_along_max_m:
            med_dx = float(np.median(peak_dx_arr))
            med_dy = float(np.median(peak_dy_arr))
            med_ncc = float(np.median([p.ncc_score for p in peaks]))
            prior_self_consistency_ok = True
            prior_for_gate = list(neighbour_shifts_m or [])
            if prior_for_gate and (strip_cross_max_m > 0 or strip_along_max_m > 0):
                prior_dx = float(np.median([s[0] for s in prior_for_gate]))
                prior_dy = float(np.median([s[1] for s in prior_for_gate]))
                cross_disagree = abs(med_dx - prior_dx)
                along_disagree = abs(med_dy - prior_dy)
                if (cross_disagree > strip_cross_max_m
                        or along_disagree > strip_along_max_m):
                    prior_self_consistency_ok = False
                    print(f"  [coarse:ncc-eloftr] NCC cluster at "
                          f"dx={med_dx:+.0f}m, dy={med_dy:+.0f}m disagrees "
                          f"with strip prior dx={prior_dx:+.0f}m, dy={prior_dy:+.0f}m "
                          f"(cross={cross_disagree:.0f}m, along={along_disagree:.0f}m); "
                          f"falling through to prior tie-breaker")
            if prior_self_consistency_ok:
                dists = np.hypot(peak_dx_arr - med_dx, peak_dy_arr - med_dy)
                anchor_idx = int(np.argmin(dists))
                print(f"  [coarse:ncc-eloftr] all {len(peaks)} ELoFTR validations "
                      f"abstained; NCC self-consistency tight "
                      f"(cross={cross_spread:.0f}m ≤ {strip_cross_max_m:.0f}m, "
                      f"along={along_spread:.0f}m ≤ {strip_along_max_m:.0f}m); "
                      f"using cluster-median peak dx={med_dx:+.0f}m, dy={med_dy:+.0f}m "
                      f"(med_ncc={med_ncc:.2f})")
                # Agreement reported as max(0.30, median NCC) so the
                # downstream agreement floor (0.30) is just cleared
                # while the value still reflects NCC strength when high.
                best = (med_dx, med_dy, 0, max(0.30, med_ncc))
                best_idx = anchor_idx
    if best is None:
        prior = list(neighbour_shifts_m or [])
        if prior and (strip_cross_max_m > 0 or strip_along_max_m > 0):
            prior_dx = float(np.median([s[0] for s in prior]))
            prior_dy = float(np.median([s[1] for s in prior]))
            # Prior-corroboration: when the parent caller forwarded a
            # rejected single-shot ELoFTR (dx, dy), require the prior to
            # agree with that local measurement within the per-axis
            # strip-coherence bounds before believing it. This protects
            # against the v26 failure mode where DA023's +30 km east
            # shift was propagated to siblings whose own (weak) ELoFTR
            # reported a small or opposite shift; intra-strip USGS
            # errors can vary that much, so the prior is unsafe to apply
            # without local corroboration.
            if single_shot_dxdy_m is not None and len(prior) < 2:
                ss_dx, ss_dy = single_shot_dxdy_m
                cross_disagree = abs(prior_dx - ss_dx)
                along_disagree = abs(prior_dy - ss_dy)
                if (cross_disagree > strip_cross_max_m
                        or along_disagree > strip_along_max_m):
                    print(f"  [coarse:ncc-eloftr] strip prior dx={prior_dx:+.0f}m, "
                          f"dy={prior_dy:+.0f}m disagrees with single-shot ELoFTR "
                          f"dx={ss_dx:+.0f}m, dy={ss_dy:+.0f}m "
                          f"(cross={cross_disagree:.0f}m > {strip_cross_max_m:.0f}m "
                          f"or along={along_disagree:.0f}m > {strip_along_max_m:.0f}m); "
                          f"prior not corroborated, abstaining")
                    return None
            # Pick the candidate closest to the strip prior by Euclidean
            # distance, then enforce per-axis gates separately. KH
            # satellites fly near-polar orbits so dx is cross-strip and
            # dy along-strip; cross-strip drift is the wrong-coast
            # signal (tight gate) while along-strip drift is legitimate
            # USGS-corner error along the orbit track (loose gate).
            ranked = []
            for i, p in enumerate(peaks):
                dx_i = p.shift_dx_m * coarse_res
                dy_i = p.shift_dy_m * coarse_res
                cross_i = abs(dx_i - prior_dx)
                along_i = abs(dy_i - prior_dy)
                ranked.append((cross_i, along_i, i, p, dx_i, dy_i))
            cross, along, i, p, dx_m_only, dy_m_only = min(
                ranked, key=lambda item: math.hypot(item[0], item[1])
            )
            print(f"  [coarse:ncc-eloftr] all {len(peaks)} ELoFTR validations "
                  f"abstained; strip prior dx={prior_dx:+.0f}m, "
                  f"dy={prior_dy:+.0f}m selected candidate {i} "
                  f"(cross={cross:.0f}m, along={along:.0f}m)")
            if cross > strip_cross_max_m:
                print(f"  [coarse:ncc-eloftr] nearest NCC candidate has "
                      f"cross-strip residual {cross:.0f}m > "
                      f"{strip_cross_max_m:.0f}m gate — abstain "
                      f"(wrong-coast risk)")
                return None
            if along > strip_along_max_m:
                print(f"  [coarse:ncc-eloftr] nearest NCC candidate has "
                      f"along-strip residual {along:.0f}m > "
                      f"{strip_along_max_m:.0f}m gate — abstain")
                return None
            best = (dx_m_only, dy_m_only, 0, float(p.ncc_score))
            best_idx = i
        else:
            print(f"  [coarse:ncc-eloftr] all {len(peaks)} ELoFTR "
                  f"validations abstained and no validated strip prior "
                  f"is available — abstain")
            return None

    dx_m, dy_m, n_matches, agreement = best
    total_offset = math.hypot(dx_m, dy_m)

    # Sanity ceiling — a reported shift > 1.5x the bounded search radius
    # would mean we ran outside the bound (impossible by construction)
    # or hit a numerical bug.
    ceiling = radius_m * 1.5
    if total_offset > ceiling:
        print(f"  [coarse:ncc-eloftr] shift {total_offset/1000:.0f}km exceeds "
              f"ceiling {ceiling/1000:.0f}km — abstain")
        return None

    print(f"  [coarse:ncc-eloftr] selected candidate {best_idx} "
          f"(total_offset={total_offset/1000:.1f}km)")

    # Stage C — sub-pixel phase-correlation refinement (fine-res when
    # paths supplied, canvas-array fallback otherwise). Centre Stage
    # C's window on ref's actual canvas centroid (in metric coords).
    # The KH-9 reference covers a much smaller area than the full
    # canvas — the canvas-centre default would put the window outside
    # ref's footprint and trigger "fine windows have <100 valid
    # pixels".
    union_left, _ub, _ur, union_top = union_bounds
    chosen_peak = peaks[best_idx]
    ref_centroid_x = float(union_left + (chosen_peak.anchor_c
                                          + chosen_peak.template_w / 2.0)
                            * coarse_res)
    ref_centroid_y = float(union_top - (chosen_peak.anchor_r
                                         + chosen_peak.template_h / 2.0)
                            * coarse_res)
    anchor_canvas_rc = (
        chosen_peak.anchor_r + chosen_peak.template_h / 2.0,
        chosen_peak.anchor_c + chosen_peak.template_w / 2.0,
    )
    dx_phase, dy_phase = _phase_correlate_finalize(
        arr_ref, arr_tgt, coarse_res, (dx_m, dy_m),
        fine_window_m=fine_window_m, min_resp=fine_resp_min,
        target_path=target_path, reference_path=reference_path,
        work_crs=work_crs, union_bounds=union_bounds,
        fine_res_m=fine_res_m,
        ref_centre_xy=(ref_centroid_x, ref_centroid_y),
        anchor_canvas_rc=anchor_canvas_rc,
    )
    dx_m += dx_phase
    dy_m += dy_phase
    return dx_m, dy_m, int(n_matches), float(agreement)
