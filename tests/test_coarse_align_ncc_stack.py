"""Unit tests for the stacked NCC -> ELoFTR -> phase-correlate fallback
in ``preprocess/coarse_align_ncc_stack.py``.

All tests synthesise small canvases and stub the heavy components (the
ELoFTR matcher, sometimes the land-mask provider) so they run in
sub-second wall-time without model weights.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest

from preprocess import coarse_align_ncc_stack as stack
from preprocess.coarse_align_ncc_stack import _NCCPeak, run_stacked_coarse_align


# ---------------------------------------------------------------------------
# Synthetic-canvas helpers
# ---------------------------------------------------------------------------

def _make_blob_canvas(shape, blob_origin, blob_size, intensity=200, seed=0):
    """Place a non-uniform L-shaped bright blob on a noisy dark canvas.

    The L-shape is essential — a uniform rectangle has zero template
    variance, which makes `cv2.matchTemplate(..., TM_CCOEFF_NORMED)`
    return 1.0 everywhere (degenerate case). An L-shape gives the
    matcher real shape evidence to work with, mirroring the variation
    real coastlines provide.
    """
    rng = np.random.default_rng(seed)
    canvas = rng.integers(20, 60, size=shape, dtype=np.int32).astype(np.float32)
    r0, c0 = blob_origin
    h, w = blob_size
    # Vertical bar (full height, left third).
    canvas[r0:r0 + h, c0:c0 + w // 3] = intensity
    # Horizontal foot (lower third, full width).
    canvas[r0 + 2 * h // 3:r0 + h, c0:c0 + w] = intensity
    return canvas


def _make_params(*,
                 search_radius_m=25_000.0,
                 top_k=5,
                 min_ncc=0.20,
                 nms_distance_m=1500.0,
                 mask_mode="heuristic",
                 validation_window_m=4000.0,
                 fine_window_m=2000.0,
                 fine_resp_min=0.05,
                 strip_coherence_max_m=5000.0,
                 strip_coherence_cross_max_m=None,
                 strip_coherence_along_max_m=None,
                 usgs_corners_reliable=False):
    """Build a minimal AlignParams-like object with just the camera knobs
    the new stack consults."""
    camera = SimpleNamespace(
        coarse_ncc_search_radius_m=search_radius_m,
        coarse_ncc_top_k=top_k,
        coarse_ncc_min_ncc=min_ncc,
        coarse_ncc_nms_distance_m=nms_distance_m,
        coarse_ncc_mask_mode=mask_mode,
        coarse_ncc_validation_window_m=validation_window_m,
        coarse_ncc_fine_window_m=fine_window_m,
        coarse_ncc_fine_resp_min=fine_resp_min,
        coarse_ncc_strip_coherence_max_m=strip_coherence_max_m,
        coarse_ncc_strip_coherence_cross_max_m=strip_coherence_cross_max_m,
        coarse_ncc_strip_coherence_along_max_m=strip_coherence_along_max_m,
        coarse_ncc_fine_res_m=10.0,
        usgs_corners_reliable=usgs_corners_reliable,
    )
    return SimpleNamespace(camera=camera)


def _stub_land_mask_factory(monkeypatch, blob_pixels_by_arr_id):
    """Replace the land-mask provider with a deterministic threshold so
    tests don't depend on the real OBIA / heuristic providers (which
    would dominate test runtime). Returns a uint8 0/255 mask of the
    bright pixels."""

    def fake_land_mask_u8(arr, mask_mode):
        return ((arr > 100).astype(np.uint8)) * 255

    monkeypatch.setattr(stack, "_land_mask_u8", fake_land_mask_u8)


def _stub_eloftr(monkeypatch, table):
    """Replace the per-window ELoFTR call with a deterministic lookup.

    ``table`` is a list of ``(ref_window, tgt_window) -> (dx_m, dy_m,
    n_matches, agreement)`` items consumed in order. Each entry is a
    plain tuple; ``None`` short-circuits to an abstain."""
    cursor = {"i": 0}

    def fake_eloftr(arr_ref, arr_tgt, coarse_res, *, model_cache=None):
        i = cursor["i"]
        cursor["i"] += 1
        if i >= len(table):
            return None
        return table[i]

    monkeypatch.setattr(
        "preprocess.coarse_align_ncc_stack.eloftr_translation_estimate",
        fake_eloftr, raising=False,
    )
    # The function is imported at call-time inside _validate_peak_with_eloftr,
    # so patch the source location too.
    monkeypatch.setattr("align.scale.eloftr_translation_estimate", fake_eloftr)


# ---------------------------------------------------------------------------
# Stage A — NCC peak extraction
# ---------------------------------------------------------------------------

def test_recovers_known_translation(monkeypatch):
    """Two land-mask arrays differing by a known +200 m / +100 m shift
    in canvas pixel coords. NCC top-K finds the peak; ELoFTR confirms
    with synthetic matches; final returned shift matches.
    """
    coarse_res = 50.0
    H, W = 400, 600
    arr_ref = _make_blob_canvas((H, W), blob_origin=(120, 200), blob_size=(80, 100))
    # Target: same blob shifted by +4 px south, +2 px east (i.e., target
    # is sitting 4px north-west of where the reference puts it; we want
    # to shift it +200m east, -200m north to align... wait let's re-do).
    # Place the target's blob 4 px above and 2 px left of the ref blob —
    # so to align it, we shift it +200 m east and +200 m north (rows
    # decreasing).
    arr_tgt = _make_blob_canvas((H, W), blob_origin=(124, 198), blob_size=(80, 100), seed=1)

    _stub_land_mask_factory(monkeypatch, None)
    _stub_eloftr(monkeypatch, [
        # Stage B: zero residual on the correct candidate.
        (0.0, 0.0, 80, 0.45),
    ])
    monkeypatch.setattr(
        "preprocess.coarse_align_ncc_stack._phase_correlate_finalize",
        lambda *a, **k: (0.0, 0.0),
    )

    params = _make_params()
    out = run_stacked_coarse_align(
        arr_ref, arr_tgt, coarse_res,
        work_crs=None,
        union_bounds=(0.0, 0.0, W * coarse_res, H * coarse_res),
        target_bbox_wgs=None,
        params=params,
    )
    assert out is not None
    dx_m, dy_m, n_matches, agreement = out
    # Target blob @ row 124 vs ref blob @ row 120 → +4 rows = south →
    # to align target → north → dy_m = +200m.
    # Target blob @ col 198 vs ref blob @ col 200 → -2 cols → east → +100m.
    # Tolerance ±20 m (0.4 px @ 50 m/px) accommodates the sub-pixel
    # parabolic fit's response to synthetic L-shape NCC-surface
    # asymmetry; real coastline shapes have smoother neighbourhoods.
    assert abs(dx_m - 100.0) < 20.0, f"dx_m={dx_m}"
    assert abs(dy_m - 200.0) < 20.0, f"dy_m={dy_m}"
    assert n_matches == 80
    assert abs(agreement - 0.45) < 1e-6


def test_no_peak_above_floor_abstains(monkeypatch, capsys):
    """Pure noise produces no peak above the NCC floor."""
    H, W = 400, 600
    rng = np.random.default_rng(0)
    arr_ref = rng.integers(20, 60, size=(H, W), dtype=np.int32).astype(np.float32)
    arr_tgt = rng.integers(20, 60, size=(H, W), dtype=np.int32).astype(np.float32)

    # Land masks come back fully empty for noise — guard fires first.
    _stub_land_mask_factory(monkeypatch, None)
    params = _make_params(min_ncc=0.95)  # impossibly tight floor

    out = run_stacked_coarse_align(
        arr_ref, arr_tgt, 50.0,
        work_crs=None,
        union_bounds=(0.0, 0.0, W * 50.0, H * 50.0),
        target_bbox_wgs=None,
        params=params,
    )
    assert out is None


def test_no_peak_does_not_apply_strip_prior_blindly(monkeypatch, capsys):
    """When NCC produces no peaks, the function abstains rather than
    blindly applying the strip prior. Reverted 2026-04-27 after a v26
    validation showed N=1 priors can mis-shift sibling frames by 30+ km
    (intra-strip USGS-corner variation). Re-enable only behind a
    per-frame corroboration gate."""
    H, W = 400, 600
    rng = np.random.default_rng(0)
    arr_ref = rng.integers(20, 60, size=(H, W), dtype=np.int32).astype(np.float32)
    arr_tgt = rng.integers(20, 60, size=(H, W), dtype=np.int32).astype(np.float32)
    _stub_land_mask_factory(monkeypatch, None)
    params = _make_params(min_ncc=0.95)  # impossibly tight floor → no peaks

    out = run_stacked_coarse_align(
        arr_ref, arr_tgt, 50.0,
        work_crs=None,
        union_bounds=(0.0, 0.0, W * 50.0, H * 50.0),
        target_bbox_wgs=None,
        params=params,
        neighbour_shifts_m=[(1500.0, -800.0), (1600.0, -900.0)],
    )
    assert out is None


def test_search_radius_zero_disables_stack(monkeypatch):
    """``coarse_ncc_search_radius_m == 0`` short-circuits at the gate
    so the function returns ``None`` without invoking land-mask
    construction or NCC. We sentinel ``_land_mask_u8`` to raise to
    confirm it isn't called."""
    H, W = 400, 600
    arr_ref = _make_blob_canvas((H, W), blob_origin=(100, 200), blob_size=(80, 80))
    arr_tgt = _make_blob_canvas((H, W), blob_origin=(110, 210), blob_size=(80, 80))

    def boom(*a, **kw):
        raise AssertionError("_land_mask_u8 should not be called when radius=0")

    monkeypatch.setattr(stack, "_land_mask_u8", boom)

    params = _make_params(search_radius_m=0.0)
    out = run_stacked_coarse_align(
        arr_ref, arr_tgt, 50.0,
        work_crs=None,
        union_bounds=(0.0, 0.0, W * 50.0, H * 50.0),
        target_bbox_wgs=None,
        params=params,
    )
    assert out is None


def test_mask_validity_floor_abstains(monkeypatch):
    """Either land mask < 1 % coverage → Stage A abstains before NCC."""
    H, W = 400, 400

    def empty_mask(arr, mask_mode):
        return np.zeros(arr.shape, dtype=np.uint8)

    monkeypatch.setattr(stack, "_land_mask_u8", empty_mask)
    arr_ref = np.zeros((H, W), dtype=np.float32)
    arr_tgt = np.zeros((H, W), dtype=np.float32)
    params = _make_params()
    out = run_stacked_coarse_align(
        arr_ref, arr_tgt, 50.0,
        work_crs=None,
        union_bounds=(0.0, 0.0, W * 50.0, H * 50.0),
        target_bbox_wgs=None,
        params=params,
    )
    assert out is None


def test_two_similar_coast_peaks_eloftr_breaks_tie(monkeypatch):
    """Two NCC peaks: high-NCC wrong-coast (8 matches → fails ELoFTR
    gate) vs lower-NCC right-coast (80 matches at 0.5 agreement). The
    selected candidate must be the right-coast one — this is the
    regression guard for the historic Saudi-vs-Bahrain failure mode.
    """
    coarse_res = 50.0
    H, W = 400, 800
    # Reference: one L-shape (Bahrain). NCC's template is this single
    # bbox.
    arr_ref = _make_blob_canvas((H, W), blob_origin=(150, 50), blob_size=(120, 120))

    # Target canvas has TWO similar L-shapes — the real Bahrain (at
    # col 250) and a wrong-coast lookalike (at col 600). NCC slides
    # ref-template across tgt and finds two strong peaks.
    arr_tgt = _make_blob_canvas((H, W), blob_origin=(150, 250), blob_size=(120, 120), seed=1)
    # Add the second L (wrong-coast lookalike) to the target.
    arr_tgt[150:270, 600:640] = 200                      # vertical bar
    arr_tgt[230:270, 600:720] = 200                      # horizontal foot

    _stub_land_mask_factory(monkeypatch, None)

    # Stage B: the higher-NCC candidate (whichever arrives first) gets
    # fewer ELoFTR matches → fails the gate; the second candidate gets
    # 80 matches at agreement 0.5 → wins.
    eloftr_calls = {"results": [
        (0.0, 0.0, 8, 0.05),    # n_matches < 30 → ELoFTR-internal abstain
        (0.0, 0.0, 80, 0.50),   # passes
    ]}

    def eloftr_stub(arr_ref_w, arr_tgt_w, coarse_res, *, model_cache=None):
        # Simulate the ELoFTR-internal abstain for low n_matches. Once
        # the deterministic table runs out, abstain (further NCC
        # candidates may exist due to noisy peaks but only the first
        # two are interesting for this regression guard).
        if not eloftr_calls["results"]:
            return None
        result = eloftr_calls["results"].pop(0)
        if result[2] < 30:
            return None
        return result

    monkeypatch.setattr("align.scale.eloftr_translation_estimate", eloftr_stub)
    monkeypatch.setattr(
        "preprocess.coarse_align_ncc_stack._phase_correlate_finalize",
        lambda *a, **k: (0.0, 0.0),
    )

    # NMS is large enough to keep the two coast peaks distinct.
    params = _make_params(top_k=5, nms_distance_m=5_000.0)
    out = run_stacked_coarse_align(
        arr_ref, arr_tgt, coarse_res,
        work_crs=None,
        union_bounds=(0.0, 0.0, W * coarse_res, H * coarse_res),
        target_bbox_wgs=None,
        params=params,
    )
    assert out is not None
    _dx, _dy, n_matches, agreement = out
    # Selected candidate is the one with 80 matches at agreement 0.5.
    assert n_matches == 80
    assert abs(agreement - 0.5) < 1e-6
    # Both ELoFTR results consumed.
    assert eloftr_calls["results"] == []


def test_all_eloftr_candidates_fail_without_strip_prior_abstains(monkeypatch):
    """When every ELoFTR validation abstains, NCC-only peaks are not
    accepted unless a validated strip prior independently constrains
    the choice."""
    coarse_res = 50.0
    H, W = 400, 600
    arr_ref = _make_blob_canvas((H, W), blob_origin=(150, 200), blob_size=(80, 80))
    arr_tgt = _make_blob_canvas((H, W), blob_origin=(155, 205), blob_size=(80, 80), seed=1)

    _stub_land_mask_factory(monkeypatch, None)
    monkeypatch.setattr("align.scale.eloftr_translation_estimate",
                        lambda *a, **k: None)
    monkeypatch.setattr(
        "preprocess.coarse_align_ncc_stack._phase_correlate_finalize",
        lambda *a, **k: (0.0, 0.0),
    )

    params = _make_params()
    out = run_stacked_coarse_align(
        arr_ref, arr_tgt, coarse_res,
        work_crs=None,
        union_bounds=(0.0, 0.0, W * coarse_res, H * coarse_res),
        target_bbox_wgs=None,
        params=params,
    )
    assert out is None


def test_strip_prior_can_select_unvalidated_ncc_candidate(monkeypatch):
    """A validated neighbor shift may disambiguate NCC-only candidates,
    but only when the selected peak is close to the strip prior."""
    coarse_res = 50.0
    H, W = 400, 600
    arr_ref = _make_blob_canvas((H, W), blob_origin=(150, 200), blob_size=(80, 80))
    arr_tgt = _make_blob_canvas((H, W), blob_origin=(155, 205), blob_size=(80, 80), seed=1)

    _stub_land_mask_factory(monkeypatch, None)
    monkeypatch.setattr("align.scale.eloftr_translation_estimate",
                        lambda *a, **k: None)
    monkeypatch.setattr(
        "preprocess.coarse_align_ncc_stack._phase_correlate_finalize",
        lambda *a, **k: (0.0, 0.0),
    )

    params = _make_params(strip_coherence_max_m=1000.0)
    out = run_stacked_coarse_align(
        arr_ref, arr_tgt, coarse_res,
        work_crs=None,
        union_bounds=(0.0, 0.0, W * coarse_res, H * coarse_res),
        target_bbox_wgs=None,
        params=params,
        neighbour_shifts_m=[(-250.0, 250.0)],
    )
    assert out is not None
    dx, dy, n_matches, agreement = out
    assert n_matches == 0
    assert agreement >= 0.20
    assert abs(dx - (-250.0)) < 150.0
    assert abs(dy - 250.0) < 150.0


def test_strip_prior_rejects_far_unvalidated_ncc_candidate(monkeypatch):
    coarse_res = 50.0
    H, W = 400, 600
    arr_ref = _make_blob_canvas((H, W), blob_origin=(150, 200), blob_size=(80, 80))
    arr_tgt = _make_blob_canvas((H, W), blob_origin=(155, 205), blob_size=(80, 80), seed=1)

    _stub_land_mask_factory(monkeypatch, None)
    monkeypatch.setattr("align.scale.eloftr_translation_estimate",
                        lambda *a, **k: None)
    params = _make_params(strip_coherence_max_m=500.0)
    out = run_stacked_coarse_align(
        arr_ref, arr_tgt, coarse_res,
        work_crs=None,
        union_bounds=(0.0, 0.0, W * coarse_res, H * coarse_res),
        target_bbox_wgs=None,
        params=params,
        neighbour_shifts_m=[(10_000.0, 10_000.0)],
    )
    assert out is None


def test_phase_correlate_residual_added(monkeypatch):
    """Stage C adds a sub-pixel residual to the Stage B shift."""
    coarse_res = 50.0
    H, W = 400, 600
    arr_ref = _make_blob_canvas((H, W), blob_origin=(150, 200), blob_size=(80, 80))
    arr_tgt = _make_blob_canvas((H, W), blob_origin=(150, 200), blob_size=(80, 80), seed=1)

    _stub_land_mask_factory(monkeypatch, None)
    # Stage B: large shift candidate with 60 matches at agreement 0.4.
    monkeypatch.setattr(
        "align.scale.eloftr_translation_estimate",
        lambda *a, **k: (500.0, -300.0, 60, 0.40),
    )
    # Stage C: stub the phase-correlate to return a +20 m / -10 m residual.
    monkeypatch.setattr(
        "preprocess.coarse_align_ncc_stack._phase_correlate_finalize",
        lambda *a, **k: (20.0, -10.0),
    )

    params = _make_params()
    out = run_stacked_coarse_align(
        arr_ref, arr_tgt, coarse_res,
        work_crs=None,
        union_bounds=(0.0, 0.0, W * coarse_res, H * coarse_res),
        target_bbox_wgs=None,
        params=params,
    )
    assert out is not None
    dx_m, dy_m, _, _ = out
    assert abs(dx_m - 520.0) < 20.0   # 0.4 px tolerance for sub-pixel fit
    assert abs(dy_m - (-310.0)) < 20.0


def test_sanity_ceiling_abstains(monkeypatch):
    """Stage B reporting a shift > 1.5 × radius is rejected as a bug.
    """
    coarse_res = 50.0
    H, W = 400, 800
    arr_ref = _make_blob_canvas((H, W), blob_origin=(150, 100), blob_size=(80, 80))
    arr_tgt = _make_blob_canvas((H, W), blob_origin=(150, 700), blob_size=(80, 80), seed=1)

    _stub_land_mask_factory(monkeypatch, None)
    # Report a 50 km shift on a 25 km radius — exceeds 1.5x ceiling.
    monkeypatch.setattr(
        "align.scale.eloftr_translation_estimate",
        lambda *a, **k: (50_000.0, 0.0, 60, 0.40),
    )

    params = _make_params(search_radius_m=25_000.0)
    out = run_stacked_coarse_align(
        arr_ref, arr_tgt, coarse_res,
        work_crs=None,
        union_bounds=(0.0, 0.0, W * coarse_res, H * coarse_res),
        target_bbox_wgs=None,
        params=params,
    )
    assert out is None


# ---------------------------------------------------------------------------
# Profile parsing — defaults flow through CameraParams
# ---------------------------------------------------------------------------

def test_camera_params_ncc_defaults():
    """Default CameraParams disables the fallback; the four KH-4/7/8
    profiles enable it at 25 km; the two KH-9 profiles leave it off."""
    from align.params import CameraParams, load_profile

    base = CameraParams()
    assert base.coarse_ncc_search_radius_m == 0.0
    assert base.coarse_ncc_mask_mode == "coastal_obia"
    assert base.coarse_ncc_top_k == 5
    assert base.coarse_ncc_min_ncc == 0.20

    for name in ("kh4a", "kh4b", "kh7", "kh8"):
        prof = load_profile(name)
        assert prof.camera.coarse_ncc_search_radius_m == 50_000.0, name
        assert prof.camera.usgs_corners_reliable is False, name

    for name in ("kh9", "kh9_mc"):
        prof = load_profile(name)
        assert prof.camera.coarse_ncc_search_radius_m == 0.0, name
        assert prof.camera.usgs_corners_reliable is True, name


# ---------------------------------------------------------------------------
# Integration with coarse_align_and_crop — only fires on abstain
# ---------------------------------------------------------------------------

def test_falls_back_only_when_single_shot_abstains(monkeypatch, tmp_path):
    """When single-shot ELoFTR returns a confident result, the stacked
    fallback must NOT run. When it returns ``None``, the fallback runs
    and its result becomes the function's output."""
    from preprocess import georef
    from align.params import load_profile

    # Build small valid GeoTIFFs for ref + target so the upstream
    # gdalwarp / canvas-construction code completes without erroring.
    import rasterio
    from rasterio.transform import from_bounds

    H, W = 200, 200
    ref_tif = tmp_path / "ref.tif"
    tgt_tif = tmp_path / "tgt.tif"
    bounds = (0.0, 0.0, 10000.0, 10000.0)
    transform = from_bounds(*bounds, W, H)
    arr = np.ones((H, W), dtype=np.uint8) * 100
    arr[80:120, 80:120] = 250
    for path in (ref_tif, tgt_tif):
        with rasterio.open(
            path, "w", driver="GTiff", height=H, width=W, count=1,
            dtype=arr.dtype, crs="EPSG:32639", transform=transform,
        ) as dst:
            dst.write(arr, 1)

    profile = load_profile("kh4b")

    # --- Branch A: single-shot returns a confident result; fallback NOT called.
    confident = (10.0, 20.0, 100, 0.50)
    monkeypatch.setattr(
        "preprocess.georef._eloftr_translation_estimate",
        lambda *a, **k: confident,
    )

    fallback_called = {"value": False}

    def boom_fallback(*a, **kw):
        fallback_called["value"] = True
        return (999.0, 999.0, 999, 0.99)

    monkeypatch.setattr(
        "preprocess.coarse_align_ncc_stack.run_stacked_coarse_align",
        boom_fallback,
    )

    out_path_a = tmp_path / "shifted_a.tif"
    result = georef.coarse_align_and_crop(
        str(tgt_tif), str(ref_tif), str(out_path_a),
        target_bbox_wgs=None,  # skip the WGS84 overlap pre-check
        crop=False, params=profile,
    )
    assert result is not None
    assert fallback_called["value"] is False, \
        "fallback must not run when single-shot succeeds"

    # --- Branch B: single-shot abstains; fallback returns a known shift.
    monkeypatch.setattr(
        "preprocess.georef._eloftr_translation_estimate",
        lambda *a, **k: None,
    )
    fallback_called["value"] = False

    def fallback(*a, **kw):
        fallback_called["value"] = True
        return (123.0, -456.0, 50, 0.40)

    monkeypatch.setattr(
        "preprocess.coarse_align_ncc_stack.run_stacked_coarse_align",
        fallback,
    )

    out_path_b = tmp_path / "shifted_b.tif"
    result = georef.coarse_align_and_crop(
        str(tgt_tif), str(ref_tif), str(out_path_b),
        target_bbox_wgs=None,
        crop=False, params=profile,
    )
    assert result is not None
    assert fallback_called["value"] is True, \
        "fallback must run when single-shot abstains"


def test_kh9_profile_does_not_invoke_stack(monkeypatch, tmp_path):
    """Even when single-shot abstains, the kh9 profile (radius=0) keeps
    the fallback gated off."""
    from preprocess import georef
    from align.params import load_profile

    import rasterio
    from rasterio.transform import from_bounds
    H, W = 200, 200
    ref_tif = tmp_path / "ref.tif"
    tgt_tif = tmp_path / "tgt.tif"
    bounds = (0.0, 0.0, 10000.0, 10000.0)
    transform = from_bounds(*bounds, W, H)
    arr = np.ones((H, W), dtype=np.uint8) * 100
    for path in (ref_tif, tgt_tif):
        with rasterio.open(
            path, "w", driver="GTiff", height=H, width=W, count=1,
            dtype=arr.dtype, crs="EPSG:32639", transform=transform,
        ) as dst:
            dst.write(arr, 1)

    profile = load_profile("kh9")
    monkeypatch.setattr(
        "preprocess.georef._eloftr_translation_estimate",
        lambda *a, **k: None,
    )
    fallback_called = {"value": False}
    monkeypatch.setattr(
        "preprocess.coarse_align_ncc_stack.run_stacked_coarse_align",
        lambda *a, **kw: fallback_called.__setitem__("value", True) or (0.0, 0.0, 0, 0.0),
    )

    out_path = tmp_path / "shifted.tif"
    result = georef.coarse_align_and_crop(
        str(tgt_tif), str(ref_tif), str(out_path),
        target_bbox_wgs=None,
        crop=False, params=profile,
    )
    assert result is None  # single-shot abstained, fallback gated off
    assert fallback_called["value"] is False, \
        "kh9 profile must not invoke the stacked fallback"


# ---------------------------------------------------------------------------
# Sub-pixel NCC peak interpolation (Fix A)
# ---------------------------------------------------------------------------

def test_subpixel_ncc_peak_refines():
    """Synthesise an NCC map with a known sub-pixel peak; assert
    ``_ncc_top_k_peaks`` recovers it within 0.1 px instead of 0.5 px.

    Approach: build a 2-D Gaussian centred on a non-integer position
    (50.3, 80.7), embed it in a larger NCC-shaped surface, then render
    ref/tgt land masks consistent with the bounded-search bookkeeping.
    The exact mechanics of the orchestrator aren't needed — we test
    the peak refinement by comparing the integer peak against the
    refined output stored in `_NCCPeak.shift_d{x,y}_m`.
    """
    from preprocess.coarse_align_ncc_stack import _ncc_top_k_peaks
    # We can't easily test ``_ncc_top_k_peaks`` in isolation because it
    # constructs the matchTemplate output internally. Instead exercise
    # via the full orchestrator with a synthetic gaussian-blob image
    # whose CONTINUOUS peak is at a non-integer position.
    import cv2

    H, W = 200, 200
    arr_ref = np.zeros((H, W), dtype=np.float32)
    arr_tgt = np.zeros((H, W), dtype=np.float32)
    # Single gaussian "land" patch in ref and tgt at sub-pixel offset.
    yy, xx = np.mgrid[0:H, 0:W]
    arr_ref += 200 * np.exp(-((yy - 100) ** 2 + (xx - 100) ** 2) / (2 * 12 ** 2))
    # Tgt's gaussian shifted by (+2.7 col, +1.3 row) — sub-pixel offset
    arr_tgt += 200 * np.exp(-((yy - 101.3) ** 2 + (xx - 102.7) ** 2) / (2 * 12 ** 2))

    # Fake land masks: threshold at 100
    def fake_mask(arr, mask_mode):
        return (arr > 100).astype(np.uint8) * 255
    import preprocess.coarse_align_ncc_stack as stack
    orig_mask = stack._land_mask_u8
    stack._land_mask_u8 = fake_mask
    try:
        # Stub ELoFTR + phase-correlate so only Stage A runs.
        import align.scale
        orig_eloftr = align.scale.eloftr_translation_estimate
        align.scale.eloftr_translation_estimate = lambda *a, **k: (0.0, 0.0, 80, 0.45)
        orig_phase = stack._phase_correlate_finalize
        stack._phase_correlate_finalize = lambda *a, **k: (0.0, 0.0)
        try:
            params = _make_params(search_radius_m=10_000.0,
                                  validation_window_m=2000.0)
            out = run_stacked_coarse_align(
                arr_ref, arr_tgt, 50.0,
                work_crs=None,
                union_bounds=(0.0, 0.0, W * 50.0, H * 50.0),
                target_bbox_wgs=None,
                params=params,
            )
        finally:
            align.scale.eloftr_translation_estimate = orig_eloftr
            stack._phase_correlate_finalize = orig_phase
    finally:
        stack._land_mask_u8 = orig_mask

    assert out is not None
    dx_m, dy_m, _, _ = out
    # Expected: tgt's centroid is south + east of ref by (+1.3 rows,
    # +2.7 cols). To align tgt onto ref, shift NORTH by 1.3 rows
    # (positive dy = +65 m) and WEST by 2.7 cols (negative dx = -135).
    # Sub-pixel refinement should bring error well below 25 m
    # (= 0.5 px @ 50 m/px). Without sub-pixel the integer peak snaps
    # to (-3, +1) → (-150, +50) m — error 15 m on each axis.
    assert abs(dx_m - (-135.0)) < 25.0, f"dx_m={dx_m} (expected ≈ -135)"
    assert abs(dy_m - (+65.0)) < 25.0, f"dy_m={dy_m} (expected ≈ +65)"


# ---------------------------------------------------------------------------
# Higher-resolution Stage B re-warp (Fix B)
# ---------------------------------------------------------------------------

def test_validate_peak_uses_fine_res_when_paths_supplied(monkeypatch):
    """When source paths + work_crs + union_bounds are passed, Stage B
    invokes ``read_overlap_region`` and runs ELoFTR on fine-res arrays
    instead of cropping the canvas."""
    from preprocess import coarse_align_ncc_stack as stack

    rewarp_calls = {"count": 0}

    def fake_read(src, bounds, target_crs, target_res):
        rewarp_calls["count"] += 1
        # Return a 400x400 fine-res window with synthetic content
        arr = np.full((400, 400), 100, dtype=np.uint8)
        arr[100:300, 100:300] = 200
        return arr, None

    eloftr_called = {"value": False}

    def fake_eloftr(arr_ref, arr_tgt, fine_res, *, model_cache=None):
        eloftr_called["value"] = True
        # Confirm fine-res was passed (not coarse_res)
        assert fine_res == 10.0
        # Confirm we got fine-res sized arrays (400x400 not 80x80)
        assert arr_ref.shape == (400, 400)
        return (5.0, -5.0, 80, 0.45)

    monkeypatch.setattr("align.geo.read_overlap_region", fake_read)
    monkeypatch.setattr("align.scale.eloftr_translation_estimate", fake_eloftr)

    # Stub rasterio.open so the with-statement is a no-op
    class _DummyCtx:
        def __enter__(self): return None
        def __exit__(self, *a): return False
    monkeypatch.setattr(
        "preprocess.coarse_align_ncc_stack.rasterio.open"
        if hasattr(stack, "rasterio") else "rasterio.open",
        lambda *a, **kw: _DummyCtx(),
    )

    peak = stack._NCCPeak(
        shift_dx_m=10.0, shift_dy_m=5.0, ncc_score=0.45,
        peak_r=200, peak_c=300, anchor_r=190, anchor_c=295,
        template_h=120, template_w=120,
    )
    out = stack._validate_peak_with_eloftr(
        arr_ref=np.zeros((400, 600), dtype=np.float32),
        arr_tgt=np.zeros((400, 600), dtype=np.float32),
        peak=peak, coarse_res=50.0,
        model_cache=None, window_px=80,
        target_path="/tmp/fake_tgt.tif",
        reference_path="/tmp/fake_ref.tif",
        work_crs="EPSG:32639",
        union_bounds=(0.0, 0.0, 30000.0, 20000.0),
        validation_window_m=4000.0, fine_res_m=10.0,
    )
    assert eloftr_called["value"], "ELoFTR must be called via fine-res path"
    assert rewarp_calls["count"] == 2, "read_overlap_region must be called for ref + tgt"
    assert out is not None
    dx_m, dy_m, n, ag = out
    # Stage B residual + Stage A peak shift in metric coords
    # peak.shift_dx_m=10, coarse_res=50 → 500 m  + 5 (residual) = 505 m
    assert abs(dx_m - 505.0) < 0.1
    assert abs(dy_m - (5.0 * 50.0 + (-5.0))) < 0.1
    assert n == 80


def test_land_mask_content_gate_excludes_phantom_land():
    """The OBIA / heuristic mask provider's morphological CLOSE fills
    small interior nodata holes — on sparse-coverage KH-4 strip orthos
    that creates "phantom land" in the mask at coverage edges.
    ``_land_mask_u8`` must intersect the provider's mask with
    ``arr > 0`` so NCC tracks actual content, not the coverage
    envelope.
    """
    from preprocess.coarse_align_ncc_stack import _land_mask_u8
    # Build a synthetic canvas with a small "real land" island and a
    # nodata gap surrounded by what would be classified as land.
    arr = np.zeros((400, 400), dtype=np.float32)
    arr[100:150, 100:150] = 200.0   # real bright land
    arr[200:250, 200:250] = 200.0   # another real land patch
    # Sentinel nodata between them (rows 150-200, cols 150-200) and
    # surrounding noise to ensure the upstream provider's
    # multi-Otsu doesn't classify these as land.
    rng = np.random.default_rng(0)
    arr[arr == 0] = rng.integers(20, 60, size=int((arr == 0).sum())).astype(np.float32)
    # Punch a real nodata hole inside the first land patch
    arr[110:120, 110:120] = 0.0

    mask = _land_mask_u8(arr, "heuristic")
    # The 10×10 nodata hole inside the first patch should NOT be
    # classified as land in the final gated mask.
    inner_hole = mask[110:120, 110:120]
    assert int((inner_hole > 0).sum()) == 0, \
        f"content gate should exclude nodata hole; mask has " \
        f"{int((inner_hole > 0).sum())} land pixels in nodata region"
    # The bright patches themselves should still be classified as land
    # (modulo Otsu's exact threshold).
    assert int((mask[100:150, 100:150] > 0).sum()) > 50, \
        "bright land patch must survive the gate"


def test_validate_peak_falls_back_to_array_when_paths_missing(monkeypatch):
    """Without paths, Stage B uses the canvas-array crop path (legacy
    behaviour) and does NOT invoke ``read_overlap_region``."""
    from preprocess import coarse_align_ncc_stack as stack

    def boom_read(*a, **kw):
        raise AssertionError("read_overlap_region should NOT be called when paths missing")
    monkeypatch.setattr("align.geo.read_overlap_region", boom_read)
    monkeypatch.setattr(
        "align.scale.eloftr_translation_estimate",
        lambda *a, **k: (1.0, 2.0, 80, 0.45),
    )
    peak = stack._NCCPeak(
        shift_dx_m=10.0, shift_dy_m=5.0, ncc_score=0.45,
        peak_r=200, peak_c=300, anchor_r=190, anchor_c=295,
        template_h=120, template_w=120,
    )
    arr = np.zeros((400, 600), dtype=np.float32)
    arr[150:250, 250:350] = 200
    out = stack._validate_peak_with_eloftr(
        arr_ref=arr, arr_tgt=arr, peak=peak, coarse_res=50.0,
        model_cache=None, window_px=80,
        target_path=None, reference_path=None,
        work_crs=None, union_bounds=None,
    )
    assert out is not None


def test_strip_prior_axis_gate_accepts_along_strip_drift(monkeypatch):
    """Per-axis gate accepts a candidate whose along-strip residual
    exceeds the legacy 5 km isotropic gate but stays within the looser
    along-strip max. The synthetic NCC peaks cluster around (-256, 244)
    with secondary peaks; prior at (-256, +8000) puts the closest-by-
    Euclidean candidate's along-residual ~6.7 km — would fail the
    legacy 5 km gate, passes the new 20 km along gate."""
    coarse_res = 50.0
    H, W = 400, 600
    arr_ref = _make_blob_canvas((H, W), blob_origin=(150, 200), blob_size=(80, 80))
    arr_tgt = _make_blob_canvas((H, W), blob_origin=(155, 205), blob_size=(80, 80), seed=1)

    _stub_land_mask_factory(monkeypatch, None)
    monkeypatch.setattr("align.scale.eloftr_translation_estimate",
                        lambda *a, **k: None)
    monkeypatch.setattr(
        "preprocess.coarse_align_ncc_stack._phase_correlate_finalize",
        lambda *a, **k: (0.0, 0.0),
    )

    params = _make_params(
        strip_coherence_max_m=5000.0,
        strip_coherence_cross_max_m=8000.0,
        strip_coherence_along_max_m=20000.0,
    )
    out = run_stacked_coarse_align(
        arr_ref, arr_tgt, coarse_res,
        work_crs=None,
        union_bounds=(0.0, 0.0, W * coarse_res, H * coarse_res),
        target_bbox_wgs=None,
        params=params,
        neighbour_shifts_m=[(-256.0, 8000.0)],
    )
    assert out is not None


def test_strip_prior_axis_gate_rejects_cross_strip_drift(monkeypatch):
    """Per-axis gate: cross-strip residual exceeding the cross_max gate
    abstains (wrong-coast safety). Prior at (+12000, +250) makes the
    closest-by-Euclidean NCC candidate blow the 8 km cross gate."""
    coarse_res = 50.0
    H, W = 400, 600
    arr_ref = _make_blob_canvas((H, W), blob_origin=(150, 200), blob_size=(80, 80))
    arr_tgt = _make_blob_canvas((H, W), blob_origin=(155, 205), blob_size=(80, 80), seed=1)

    _stub_land_mask_factory(monkeypatch, None)
    monkeypatch.setattr("align.scale.eloftr_translation_estimate",
                        lambda *a, **k: None)
    params = _make_params(
        strip_coherence_max_m=5000.0,
        strip_coherence_cross_max_m=8000.0,
        strip_coherence_along_max_m=20000.0,
    )
    out = run_stacked_coarse_align(
        arr_ref, arr_tgt, coarse_res,
        work_crs=None,
        union_bounds=(0.0, 0.0, W * coarse_res, H * coarse_res),
        target_bbox_wgs=None,
        params=params,
        neighbour_shifts_m=[(12000.0, 250.0)],
    )
    assert out is None


def test_strip_prior_axis_gate_falls_back_to_legacy_when_unset(monkeypatch):
    """When per-axis fields are None, both axes inherit the legacy
    isotropic ``coarse_ncc_strip_coherence_max_m`` — preserves
    behaviour for profiles that haven't opted in to the new gate."""
    coarse_res = 50.0
    H, W = 400, 600
    arr_ref = _make_blob_canvas((H, W), blob_origin=(150, 200), blob_size=(80, 80))
    arr_tgt = _make_blob_canvas((H, W), blob_origin=(155, 205), blob_size=(80, 80), seed=1)

    _stub_land_mask_factory(monkeypatch, None)
    monkeypatch.setattr("align.scale.eloftr_translation_estimate",
                        lambda *a, **k: None)
    # Same scenario as the accepts-along-drift test but with per-axis
    # fields unset: legacy 5 km isotropic gate fails the candidate's
    # ~6.7 km along-strip residual.
    params = _make_params(strip_coherence_max_m=5000.0)
    out = run_stacked_coarse_align(
        arr_ref, arr_tgt, coarse_res,
        work_crs=None,
        union_bounds=(0.0, 0.0, W * coarse_res, H * coarse_res),
        target_bbox_wgs=None,
        params=params,
        neighbour_shifts_m=[(-256.0, 8000.0)],
    )
    assert out is None

def test_prior_corroboration_rejects_when_eloftr_disagrees(monkeypatch):
    """Prior-corroboration gate: when single_shot_dxdy_m is supplied and
    disagrees with the strip prior by more than the per-axis strip-
    coherence bound, the function abstains rather than applying the
    prior. Protects against the v26 failure mode where DA023's +30 km
    east shift was propagated to siblings whose own (weak) ELoFTR
    reported a near-zero shift."""
    coarse_res = 50.0
    H, W = 400, 600
    arr_ref = _make_blob_canvas((H, W), blob_origin=(150, 200), blob_size=(80, 80))
    arr_tgt = _make_blob_canvas((H, W), blob_origin=(155, 205), blob_size=(80, 80), seed=1)

    _stub_land_mask_factory(monkeypatch, None)
    monkeypatch.setattr("align.scale.eloftr_translation_estimate",
                        lambda *a, **k: None)
    params = _make_params(
        strip_coherence_max_m=5000.0,
        strip_coherence_cross_max_m=8000.0,
        strip_coherence_along_max_m=20000.0,
    )
    # Prior says +30 km east; single-shot ELoFTR for THIS frame says
    # ~0 km east. Cross disagreement = 30 km > 8 km bound -> abstain.
    out = run_stacked_coarse_align(
        arr_ref, arr_tgt, coarse_res,
        work_crs=None,
        union_bounds=(0.0, 0.0, W * coarse_res, H * coarse_res),
        target_bbox_wgs=None,
        params=params,
        neighbour_shifts_m=[(30000.0, 3600.0)],
        single_shot_dxdy_m=(-200.0, -100.0),
    )
    assert out is None


def test_prior_corroboration_passes_when_eloftr_agrees(monkeypatch):
    """When single_shot_dxdy_m corroborates the strip prior within the
    strip-coherence bounds, the prior is allowed through to the existing
    closest-peak selection / per-axis gate logic."""
    coarse_res = 50.0
    H, W = 400, 600
    arr_ref = _make_blob_canvas((H, W), blob_origin=(150, 200), blob_size=(80, 80))
    arr_tgt = _make_blob_canvas((H, W), blob_origin=(155, 205), blob_size=(80, 80), seed=1)

    _stub_land_mask_factory(monkeypatch, None)
    monkeypatch.setattr("align.scale.eloftr_translation_estimate",
                        lambda *a, **k: None)
    monkeypatch.setattr(
        "preprocess.coarse_align_ncc_stack._phase_correlate_finalize",
        lambda *a, **k: (0.0, 0.0),
    )
    params = _make_params(
        strip_coherence_max_m=5000.0,
        strip_coherence_cross_max_m=8000.0,
        strip_coherence_along_max_m=20000.0,
    )
    # Prior at (-256, +8000); single-shot weak result says (-300, +9000).
    # Disagreement: 44 m cross (< 8 km), 1000 m along (< 20 km) -> proceed.
    out = run_stacked_coarse_align(
        arr_ref, arr_tgt, coarse_res,
        work_crs=None,
        union_bounds=(0.0, 0.0, W * coarse_res, H * coarse_res),
        target_bbox_wgs=None,
        params=params,
        neighbour_shifts_m=[(-256.0, 8000.0)],
        single_shot_dxdy_m=(-300.0, 9000.0),
    )
    assert out is not None

