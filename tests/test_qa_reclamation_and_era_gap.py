"""Tests for the cross-temporal corrections in QA scoring.

The reclamation mask excludes pixels where the source/reference land
masks disagree at scale (≥150 m × 150 m, after morphological open) so
that physical land-water change does not inflate shoreline residuals.
The era-gap grid_weight downweight handles the residual drift the
mask doesn't catch.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from align.qa import (
    _compute_qa_reclamation_mask,
    _era_gap_grid_factor,
    _ERA_GAP_DOWNWEIGHT_THRESHOLD_YEARS,
    _ERA_GAP_DECAY_PER_YEAR,
    _ERA_GAP_MIN_FACTOR,
    evaluate_alignment_quality_arrays,
)


# ---------------------------------------------------------------------------
# _era_gap_grid_factor
# ---------------------------------------------------------------------------

@pytest.mark.fast
@pytest.mark.qa
def test_era_gap_factor_below_threshold_is_one():
    assert _era_gap_grid_factor(0.0) == pytest.approx(1.0)
    assert _era_gap_grid_factor(_ERA_GAP_DOWNWEIGHT_THRESHOLD_YEARS) == pytest.approx(1.0)
    assert _era_gap_grid_factor(_ERA_GAP_DOWNWEIGHT_THRESHOLD_YEARS - 0.1) == pytest.approx(1.0)


@pytest.mark.fast
@pytest.mark.qa
def test_era_gap_factor_decays_past_threshold():
    over_5 = _ERA_GAP_DOWNWEIGHT_THRESHOLD_YEARS + 5.0
    expected = max(_ERA_GAP_MIN_FACTOR, 1.0 - _ERA_GAP_DECAY_PER_YEAR * 5.0)
    assert _era_gap_grid_factor(over_5) == pytest.approx(expected)


@pytest.mark.fast
@pytest.mark.qa
def test_era_gap_factor_clamped_at_min():
    assert _era_gap_grid_factor(50.0) == pytest.approx(_ERA_GAP_MIN_FACTOR)
    assert _era_gap_grid_factor(1000.0) == pytest.approx(_ERA_GAP_MIN_FACTOR)


@pytest.mark.fast
@pytest.mark.qa
def test_era_gap_factor_handles_none_and_inf():
    assert _era_gap_grid_factor(None) == pytest.approx(1.0)
    assert _era_gap_grid_factor(float("nan")) == pytest.approx(1.0)
    assert _era_gap_grid_factor(float("inf")) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _compute_qa_reclamation_mask
# ---------------------------------------------------------------------------

@pytest.mark.fast
@pytest.mark.qa
def test_reclamation_mask_empty_when_land_masks_agree():
    h, w = 80, 80
    land = np.zeros((h, w), dtype=bool)
    land[10:30, 10:30] = True
    valid = np.ones((h, w), dtype=bool)
    mask = _compute_qa_reclamation_mask(land, land, valid, eval_res=4.0)
    assert mask.shape == (h, w)
    assert not mask.any()


@pytest.mark.fast
@pytest.mark.qa
def test_reclamation_mask_flags_large_disagreement():
    """A 200 m × 200 m land disagreement (50 px at 4 m/px) should survive
    the morphological open + min-area filter and produce a non-empty mask.
    """
    h, w = 200, 200
    eval_res = 4.0
    land_ref = np.zeros((h, w), dtype=bool)
    land_out = np.zeros((h, w), dtype=bool)
    # 60 px ≈ 240 m square — well above the 150 m × 150 m min area
    land_out[60:120, 60:120] = True
    valid = np.ones((h, w), dtype=bool)
    mask = _compute_qa_reclamation_mask(land_ref, land_out, valid, eval_res=eval_res)
    assert mask.any()
    # The mask should cover the disagreement (with some dilation)
    assert mask[80:100, 80:100].all()


@pytest.mark.fast
@pytest.mark.qa
def test_reclamation_mask_drops_small_disagreement():
    """A handful of scattered single-pixel disagreements should be
    dropped by the morphological open + min-area filter (likely
    alignment noise, not physical change).
    """
    h, w = 200, 200
    land_ref = np.zeros((h, w), dtype=bool)
    land_out = np.zeros((h, w), dtype=bool)
    # Sparse 1-px disagreements, far apart
    for (r, c) in [(20, 20), (50, 80), (130, 40), (170, 170)]:
        land_out[r, c] = True
    valid = np.ones((h, w), dtype=bool)
    mask = _compute_qa_reclamation_mask(land_ref, land_out, valid, eval_res=4.0)
    assert not mask.any()


@pytest.mark.fast
@pytest.mark.qa
def test_reclamation_mask_handles_none_inputs():
    valid = np.ones((20, 20), dtype=bool)
    out_none_ref = _compute_qa_reclamation_mask(None, np.ones_like(valid), valid, 4.0)
    out_none_out = _compute_qa_reclamation_mask(np.ones_like(valid), None, valid, 4.0)
    assert not out_none_ref.any()
    assert not out_none_out.any()


# ---------------------------------------------------------------------------
# evaluate_alignment_quality_arrays — era_gap and reclamation integration
# ---------------------------------------------------------------------------

def _synthetic_bundle(land_mask, shoreline_mask, stable_mask):
    """Build a MaskBundle stand-in for testing the scorer in isolation."""
    return SimpleNamespace(
        land=land_mask.astype(np.float32),
        water=(~land_mask).astype(np.float32),
        shoreline=shoreline_mask.astype(np.float32),
        stable=stable_mask.astype(np.float32),
        shallow_water=np.zeros_like(land_mask, dtype=np.float32),
        dark_farmland=np.zeros_like(land_mask, dtype=np.float32),
        texture=np.zeros_like(land_mask, dtype=np.float32),
        brightness=np.zeros_like(land_mask, dtype=np.float32),
    )


@pytest.mark.fast
@pytest.mark.qa
def test_era_gap_downweights_grid_in_score_breakdown(monkeypatch):
    """With era_gap=15 yr (10 yr past threshold), grid_weight should
    be downweighted by 0.06×10=0.40 (factor 0.40 → expect 0.6 of original)
    and the difference shifted into patch_weight.
    """
    h, w = 200, 200
    ref_arr = (np.random.RandomState(0).rand(h, w) * 255).astype(np.uint8)
    out_arr = ref_arr.copy()
    valid = np.ones((h, w), dtype=bool)

    # Build a synthetic shoreline pattern so grid_score has data to compute.
    shoreline = np.zeros((h, w), dtype=bool)
    shoreline[80:120, :] = True  # horizontal band
    land = np.zeros((h, w), dtype=bool)
    land[100:, :] = True
    stable = np.zeros((h, w), dtype=bool)
    stable[20:60, 20:60] = True

    bundle = _synthetic_bundle(land, shoreline, stable)
    monkeypatch.setattr("align.qa.build_semantic_masks", lambda arr, mode: bundle)

    no_gap = evaluate_alignment_quality_arrays(
        ref_arr, out_arr, valid, eval_res=4.0, era_gap_years=0.0,
    )
    big_gap = evaluate_alignment_quality_arrays(
        ref_arr, out_arr, valid, eval_res=4.0, era_gap_years=15.0,
    )
    assert no_gap is not None and big_gap is not None
    no_gw = no_gap["score_breakdown"]["grid_weight"]
    big_gw = big_gap["score_breakdown"]["grid_weight"]
    no_pw = no_gap["score_breakdown"]["patch_weight"]
    big_pw = big_gap["score_breakdown"]["patch_weight"]

    expected_factor = _era_gap_grid_factor(15.0)
    assert big_gw == pytest.approx(no_gw * expected_factor, rel=1e-3)
    # Total weight (grid + patch) should be conserved
    assert (big_gw + big_pw) == pytest.approx(no_gw + no_pw, rel=1e-3)
    # Score breakdown should record the era-gap factor for traceability
    assert big_gap["score_breakdown"]["era_gap_factor"] == pytest.approx(expected_factor, rel=1e-3)
    assert big_gap["score_breakdown"]["era_gap_years"] == pytest.approx(15.0)


@pytest.mark.fast
@pytest.mark.qa
def test_default_era_gap_preserves_original_scoring(monkeypatch):
    """When era_gap_years is None or not supplied, the scoring matches
    the no-gap baseline — backward compatibility for callers that
    haven't been wired to supply the new kwarg.
    """
    h, w = 200, 200
    ref_arr = (np.random.RandomState(1).rand(h, w) * 255).astype(np.uint8)
    out_arr = ref_arr.copy()
    valid = np.ones((h, w), dtype=bool)
    shoreline = np.zeros((h, w), dtype=bool)
    shoreline[80:120, :] = True
    land = np.zeros((h, w), dtype=bool)
    land[100:, :] = True
    stable = np.zeros((h, w), dtype=bool)
    stable[20:60, 20:60] = True

    bundle = _synthetic_bundle(land, shoreline, stable)
    monkeypatch.setattr("align.qa.build_semantic_masks", lambda arr, mode: bundle)

    default = evaluate_alignment_quality_arrays(
        ref_arr, out_arr, valid, eval_res=4.0,
    )
    explicit = evaluate_alignment_quality_arrays(
        ref_arr, out_arr, valid, eval_res=4.0,
        era_gap_years=0.0,
    )
    assert default["score"] == pytest.approx(explicit["score"], rel=1e-6)
    assert default["score_breakdown"]["grid_weight"] == pytest.approx(
        explicit["score_breakdown"]["grid_weight"], rel=1e-6,
    )


@pytest.mark.fast
@pytest.mark.qa
def test_patch_supported_filter_subsets_grid_score(monkeypatch):
    """When a subset of grid cells is patch-corroborated, grid_score
    must be the mean over that subset only — not the mean over all
    valid shoreline cells. Otherwise reclamation regions whose
    shorelines look "valid" but have no inland features to ground
    the alignment will dominate the score.
    """
    h, w = 200, 200
    ref_arr = (np.random.RandomState(7).rand(h, w) * 255).astype(np.uint8)
    out_arr = ref_arr.copy()
    valid = np.ones((h, w), dtype=bool)
    shoreline = np.zeros((h, w), dtype=bool)
    shoreline[80:120, :] = True
    land = np.zeros((h, w), dtype=bool)
    land[100:, :] = True
    stable = np.zeros((h, w), dtype=bool)
    stable[20:60, 20:60] = True

    bundle = _synthetic_bundle(land, shoreline, stable)
    monkeypatch.setattr("align.qa.build_semantic_masks", lambda arr, mode: bundle)

    # Two shoreline cells with very different residuals; only one is
    # corroborated by patch support.
    base_cells = [
        {"row": 0, "col": 0, "shoreline_med": 50.0, "edge_px": 100, "valid": True},
        {"row": 1, "col": 0, "shoreline_med": 5000.0, "edge_px": 100, "valid": True},
    ]
    patch_cells = [
        {"row": 0, "col": 0, "patch_med": 60.0, "patch_count": 4, "valid": True},
        {"row": 1, "col": 0, "patch_med": None, "patch_count": 0, "valid": False},
    ]
    monkeypatch.setattr("align.qa._compute_grid_metrics",
                        lambda *a, **kw: list(base_cells))
    monkeypatch.setattr("align.qa._compute_patch_grid_metrics",
                        lambda *a, **kw: list(patch_cells))

    out = evaluate_alignment_quality_arrays(
        ref_arr, out_arr, valid, eval_res=4.0,
    )
    assert out is not None
    # Only the patch-supported cell (50 m) should drive grid_score —
    # not the average of [50, 5000].
    assert out["grid"]["reliability"] == "patch_supported"
    assert out["grid"]["scored_valid_count"] == 1
    assert out["grid_score"] == pytest.approx(50.0)


@pytest.mark.fast
@pytest.mark.qa
def test_patch_supported_falls_back_when_no_patches(monkeypatch):
    """When no patches are valid, fall back to the shoreline_only path
    so the score still has a value to report."""
    h, w = 200, 200
    ref_arr = (np.random.RandomState(8).rand(h, w) * 255).astype(np.uint8)
    out_arr = ref_arr.copy()
    valid = np.ones((h, w), dtype=bool)
    shoreline = np.zeros((h, w), dtype=bool)
    shoreline[80:120, :] = True
    land = np.zeros((h, w), dtype=bool)
    land[100:, :] = True
    stable = np.zeros((h, w), dtype=bool)
    stable[20:60, 20:60] = True

    bundle = _synthetic_bundle(land, shoreline, stable)
    monkeypatch.setattr("align.qa.build_semantic_masks", lambda arr, mode: bundle)

    base_cells = [
        {"row": 0, "col": 0, "shoreline_med": 50.0, "edge_px": 100, "valid": True},
        {"row": 1, "col": 0, "shoreline_med": 100.0, "edge_px": 100, "valid": True},
    ]
    patch_cells = [
        {"row": r, "col": c, "patch_med": None, "patch_count": 0, "valid": False}
        for r in range(4) for c in range(6)
    ]
    monkeypatch.setattr("align.qa._compute_grid_metrics",
                        lambda *a, **kw: list(base_cells))
    monkeypatch.setattr("align.qa._compute_patch_grid_metrics",
                        lambda *a, **kw: list(patch_cells))

    out = evaluate_alignment_quality_arrays(
        ref_arr, out_arr, valid, eval_res=4.0,
    )
    assert out is not None
    assert out["grid"]["reliability"] == "shoreline_only"
    # Falls back to the mean over both valid shoreline cells.
    assert out["grid"]["scored_valid_count"] == 2
    assert out["grid_score"] == pytest.approx(75.0)


@pytest.mark.fast
@pytest.mark.qa
def test_reclamation_mask_excludes_disagreement_from_shoreline(monkeypatch):
    """When ref and out land masks disagree over a large region, the
    shoreline edges in that region should be excluded from grid scoring
    (reflected in `reclamation` block + lower edge counts in those cells).
    """
    h, w = 320, 320
    eval_res = 4.0
    ref_arr = (np.random.RandomState(2).rand(h, w) * 255).astype(np.uint8)
    out_arr = ref_arr.copy()
    valid = np.ones((h, w), dtype=bool)

    # Both share a horizontal shoreline at row ~100. Ref also has a
    # reclaimed "peninsula" north of the shoreline at rows 10:90,
    # cols 10:90 — 80 px × 80 px (320 m × 320 m at 4 m/px), well
    # above the 150 m × 150 m min-area threshold and large enough to
    # survive the morphological open (~13 px kernel at this res).
    shoreline = np.zeros((h, w), dtype=bool)
    shoreline[95:105, :] = True
    land_ref = np.zeros((h, w), dtype=bool)
    land_ref[100:, :] = True
    land_ref[10:90, 10:90] = True  # ref-only land patch (reclamation)
    land_out = np.zeros((h, w), dtype=bool)
    land_out[100:, :] = True
    stable = np.zeros((h, w), dtype=bool)
    stable[200:260, 20:60] = True

    bundle_ref = _synthetic_bundle(land_ref, shoreline, stable)
    bundle_out = _synthetic_bundle(land_out, shoreline, stable)

    # Alternate per-call: first invocation gets the ref bundle, second
    # gets the out bundle (matches the ordering inside
    # evaluate_alignment_quality_arrays).
    bundles = iter([bundle_ref, bundle_out])

    def _next_bundle(arr, mode):
        return next(bundles)

    monkeypatch.setattr("align.qa.build_semantic_masks", _next_bundle)

    out = evaluate_alignment_quality_arrays(
        ref_arr, out_arr, valid, eval_res=eval_res,
    )
    assert out is not None
    assert "reclamation" in out
    assert out["reclamation"]["pixels"] > 0
    assert out["reclamation"]["fraction_of_valid"] > 0.0
