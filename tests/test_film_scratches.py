"""Tests for the film-scratch detection / inpainting primitives and
the cleaned-reference sidecar plumbing.

White diagonal scratches on declassified film scans (KH-9, KH-4) confuse
RoMa / ELoFTR by acting as high-contrast "features" with no
geographic correspondence. ``align/film_scratches.py`` detects and
inpaints them; the cleaned sidecar avoids paying the detection cost on
every match call.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from align import film_scratches
import paths as _paths


def _scene_with_scratch(size=(400, 400),
                       scratch_endpoints=((200, 40), (200, 380)),
                       seed=0) -> np.ndarray:
    """Build a synthetic image: dim textured background + one bright
    near-vertical scratch. The detector default catches near-vertical
    scratches (±75–90°); see ``align.film_scratches``."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(40, 120, size=size, dtype=np.int32).astype(np.uint8)
    cv2.line(arr, scratch_endpoints[0], scratch_endpoints[1],
             color=255, thickness=3)
    return arr


def _scene_no_scratch(size=(400, 400), seed=0) -> np.ndarray:
    """Background only, no scratch."""
    rng = np.random.default_rng(seed)
    return rng.integers(40, 120, size=size, dtype=np.int32).astype(np.uint8)


def test_detector_finds_a_vertical_scratch():
    arr = _scene_with_scratch()
    mask = film_scratches.detect_scratches(arr)
    assert mask.dtype == np.uint8
    assert mask.shape == arr.shape
    # The scratch should produce a non-trivial mask
    assert int((mask > 0).sum()) > 100, \
        "near-vertical scratch should be picked up by the detector"


def test_detector_with_diagonal_opt_in():
    """Diagonal scratches are excluded from the default angle range
    (Bahrain shorelines trend diagonally and would be caught) but
    callers can opt in via ``angle_ranges_deg``."""
    arr = _scene_with_scratch(scratch_endpoints=((40, 60), (380, 200)))
    # Default config — diagonal not in default angles
    default_mask = film_scratches.detect_scratches(arr)
    assert int((default_mask > 0).sum()) < 50, \
        "diagonal scratch should NOT be caught by the vertical-only default"
    # Opt in to diagonals
    diag_params = film_scratches.ScratchDetectorParams(
        angle_ranges_deg=((-45.0, -10.0), (10.0, 45.0)),
    )
    diag_mask = film_scratches.detect_scratches(arr, diag_params)
    assert int((diag_mask > 0).sum()) > 100, \
        "diagonal scratch should be picked up when diagonals are in angle range"


def test_detector_no_false_positive_on_blank_image():
    arr = _scene_no_scratch()
    mask = film_scratches.detect_scratches(arr)
    # Some noise pixels may slip through, but the detector should not
    # produce a large false-positive mask on a blank scene.
    fraction = float((mask > 0).sum()) / mask.size
    assert fraction < 0.05, f"too many false positives: {fraction:.2%}"


def test_inpainter_lowers_brightness_in_scratch_region():
    arr = _scene_with_scratch()
    mask = film_scratches.detect_scratches(arr)
    cleaned = film_scratches.inpaint_scratches(arr, mask=mask)
    assert cleaned.shape == arr.shape
    # Inside the masked region, the cleaned image should no longer be
    # uniformly bright — inpaint replaces with surrounding texture.
    if int((mask > 0).sum()) > 0:
        before = float(arr[mask > 0].mean())
        after = float(cleaned[mask > 0].mean())
        assert after < before, \
            f"inpainted scratch region should be dimmer than the bright streak ({before} → {after})"


def test_inpainter_returns_copy_when_mask_empty():
    arr = _scene_no_scratch()
    empty_mask = np.zeros_like(arr, dtype=np.uint8)
    cleaned = film_scratches.inpaint_scratches(arr, mask=empty_mask)
    assert cleaned.shape == arr.shape
    assert np.array_equal(cleaned, arr)


def test_params_hash_is_stable():
    p1 = film_scratches.ScratchDetectorParams()
    p2 = film_scratches.ScratchDetectorParams()
    assert p1.hash() == p2.hash()
    p3 = film_scratches.ScratchDetectorParams(min_length_px=999)
    assert p1.hash() != p3.hash(), "param change must invalidate hash"


def test_provenance_round_trip(tmp_path):
    ref = tmp_path / "ref.tif"
    ref.write_bytes(b"x" * 16)
    prov = _paths.reference_scratch_cleaned_provenance_path(str(ref))
    p = film_scratches.ScratchDetectorParams()
    film_scratches.write_provenance(prov, str(ref), p)
    assert film_scratches.provenance_matches(prov, str(ref), p)


def test_provenance_invalidates_on_param_change(tmp_path):
    ref = tmp_path / "ref.tif"
    ref.write_bytes(b"x" * 16)
    prov = _paths.reference_scratch_cleaned_provenance_path(str(ref))
    p = film_scratches.ScratchDetectorParams()
    film_scratches.write_provenance(prov, str(ref), p)
    p2 = film_scratches.ScratchDetectorParams(min_length_px=999)
    assert not film_scratches.provenance_matches(prov, str(ref), p2)


def test_provenance_invalidates_on_reference_change(tmp_path):
    ref = tmp_path / "ref.tif"
    ref.write_bytes(b"x" * 16)
    prov = _paths.reference_scratch_cleaned_provenance_path(str(ref))
    p = film_scratches.ScratchDetectorParams()
    film_scratches.write_provenance(prov, str(ref), p)
    ref.write_bytes(b"y" * 32)  # mtime + size change
    assert not film_scratches.provenance_matches(prov, str(ref), p)


def test_path_helpers_match_naming_convention(tmp_path):
    ref = tmp_path / "kh9_dzb1212.warped.tif"
    cleaned = _paths.reference_scratch_cleaned_path(str(ref))
    prov = _paths.reference_scratch_cleaned_provenance_path(str(ref))
    assert cleaned.endswith("kh9_dzb1212.warped.scratch_cleaned.tif")
    assert prov.endswith("kh9_dzb1212.warped.scratch_cleaned.json")


def _write_geotiff(path: Path, arr: np.ndarray, bounds=(0.0, 0.0, 1000.0, 1000.0)):
    h, w = arr.shape
    transform = from_bounds(*bounds, w, h)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1,
        dtype=arr.dtype, crs="EPSG:3857", transform=transform,
    ) as dst:
        dst.write(arr, 1)


def test_generate_cleaned_reference_writes_sidecar_and_provenance(tmp_path):
    ref = tmp_path / "ref.tif"
    arr = _scene_with_scratch(size=(200, 200))
    _write_geotiff(ref, arr)
    cleaned = _paths.reference_scratch_cleaned_path(str(ref))
    prov = _paths.reference_scratch_cleaned_provenance_path(str(ref))
    out = film_scratches.generate_scratch_cleaned_reference(
        str(ref), cleaned, prov,
    )
    assert out == cleaned
    assert os.path.exists(cleaned)
    assert os.path.exists(prov)
    # Cleaned image should still be a valid GeoTIFF with the same shape
    with rasterio.open(cleaned) as src:
        assert src.height == arr.shape[0]
        assert src.width == arr.shape[1]


def test_generate_cleaned_reference_skips_when_provenance_matches(tmp_path):
    ref = tmp_path / "ref.tif"
    arr = _scene_with_scratch(size=(200, 200))
    _write_geotiff(ref, arr)
    cleaned = _paths.reference_scratch_cleaned_path(str(ref))
    prov = _paths.reference_scratch_cleaned_provenance_path(str(ref))
    film_scratches.generate_scratch_cleaned_reference(str(ref), cleaned, prov)
    mtime_before = os.path.getmtime(cleaned)
    # Re-run; should short-circuit without rewriting
    film_scratches.generate_scratch_cleaned_reference(str(ref), cleaned, prov)
    assert os.path.getmtime(cleaned) == mtime_before


def test_read_overlap_region_picks_up_cleaned_sidecar(tmp_path, monkeypatch):
    """When a cleaned sidecar exists with valid provenance,
    read_overlap_region must read from it instead of the original."""
    from align import geo
    geo.clear_overlap_cache()

    ref = tmp_path / "ref.tif"
    bounds = (0.0, 0.0, 1000.0, 1000.0)
    raw = np.full((200, 200), 50, dtype=np.uint8)
    _write_geotiff(ref, raw, bounds=bounds)

    cleaned_path = _paths.reference_scratch_cleaned_path(str(ref))
    prov_path = _paths.reference_scratch_cleaned_provenance_path(str(ref))
    distinct = np.full((200, 200), 200, dtype=np.uint8)
    _write_geotiff(Path(cleaned_path), distinct, bounds=bounds)
    film_scratches.write_provenance(prov_path, str(ref),
                                    film_scratches._DEFAULT_PARAMS)

    with rasterio.open(ref) as src:
        arr, _ = geo.read_overlap_region(src, bounds, "EPSG:3857", 5.0)
    # If the cleaned sidecar was honoured, we read the value 200, not 50
    assert int(arr.mean()) > 150, \
        f"expected cleaned sidecar (mean ≈ 200), got {arr.mean()}"


def test_read_overlap_region_ignores_cleaned_sidecar_when_provenance_stale(tmp_path, monkeypatch):
    from align import geo
    geo.clear_overlap_cache()

    ref = tmp_path / "ref.tif"
    bounds = (0.0, 0.0, 1000.0, 1000.0)
    raw = np.full((200, 200), 50, dtype=np.uint8)
    _write_geotiff(ref, raw, bounds=bounds)

    cleaned_path = _paths.reference_scratch_cleaned_path(str(ref))
    prov_path = _paths.reference_scratch_cleaned_provenance_path(str(ref))
    distinct = np.full((200, 200), 200, dtype=np.uint8)
    _write_geotiff(Path(cleaned_path), distinct, bounds=bounds)
    # Stale provenance: written for a different size
    Path(prov_path).write_text(json.dumps({
        "reference_basename": ref.name,
        "reference_mtime_ns": os.stat(ref).st_mtime_ns,
        "reference_size": 99999999,  # mismatch
        "params_hash": film_scratches._DEFAULT_PARAMS.hash(),
    }))

    with rasterio.open(ref) as src:
        arr, _ = geo.read_overlap_region(src, bounds, "EPSG:3857", 5.0)
    # Stale provenance → cleaned sidecar ignored → we read raw (50)
    assert int(arr.mean()) < 100, \
        f"expected raw reference (mean ≈ 50), got {arr.mean()}"


def test_match_time_mask_zeroes_scratch_pixels_when_env_var_set(tmp_path, monkeypatch):
    """With DECLASS_MATCH_TIME_SCRATCH_MASK=1 and no cleaned sidecar,
    read_overlap_region should detect+zero scratch pixels in the
    returned array."""
    from align import geo
    geo.clear_overlap_cache()
    monkeypatch.setenv("DECLASS_MATCH_TIME_SCRATCH_MASK", "1")

    ref = tmp_path / "ref.tif"
    bounds = (0.0, 0.0, 4000.0, 4000.0)
    arr = _scene_with_scratch(size=(400, 400))
    _write_geotiff(ref, arr, bounds=bounds)

    with rasterio.open(ref) as src:
        out, _ = geo.read_overlap_region(src, bounds, "EPSG:3857", 10.0)
    # The bright scratch (255) should no longer dominate; some pixels
    # should be zeroed. We check that the brightest tail shrank.
    bright_frac_before = float((arr >= 250).sum()) / arr.size
    bright_frac_after = float((out >= 250).sum()) / out.size
    assert bright_frac_after < bright_frac_before, \
        ("match-time masking should reduce the bright scratch tail "
         f"({bright_frac_before:.4f} → {bright_frac_after:.4f})")


def _scene_with_lines(size, line_specs, *, seed=0,
                      bg_lo=40, bg_hi=120, color=255,
                      thickness=3) -> np.ndarray:
    """Build a synthetic dim-textured image with multiple bright lines.
    ``line_specs`` is a list of ``((x1, y1), (x2, y2))`` endpoint pairs."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(bg_lo, bg_hi, size=size, dtype=np.int32).astype(np.uint8)
    for p1, p2 in line_specs:
        cv2.line(arr, p1, p2, color=color, thickness=thickness)
    return arr


def test_cluster_filter_rejects_off_axis_outlier():
    """A field of vertical scratches plus one off-axis line should leave
    the verticals intact and drop the outlier — the dominant-cluster
    filter is the whole point of this test.

    Layout: 5 verticals fill the left half (x∈[20,180]); the diagonal
    runs through the right half (x∈[220,380]) so each region exercises
    only one population.
    """
    verticals = [((20 + 40 * i, 30), (20 + 40 * i, 370)) for i in range(5)]
    diagonal = [((220, 30), (380, 370))]
    arr = _scene_with_lines((400, 400), verticals + diagonal)
    # Open angle range so the prefilter doesn't drop the diagonal —
    # clustering is what must reject it.
    params = film_scratches.ScratchDetectorParams(
        angle_ranges_deg=((-90.0, 90.0),),
    )
    mask = film_scratches.detect_scratches(arr, params)
    right = mask[:, 220:]   # diagonal lives entirely on the right
    left = mask[:, :200]    # verticals live entirely on the left
    diag_px = int((right > 0).sum())
    vert_px = int((left > 0).sum())
    # Diagonal occupies ~3500 px with no clustering; with clustering the
    # right half should be near-empty (only top-hat speckle).
    assert diag_px < 200, \
        f"diagonal outlier should be filtered by cluster step; got {diag_px} px"
    assert vert_px > 1000, \
        f"verticals must survive clustering; got {vert_px} px"


def test_cluster_filter_disabled_when_k_zero():
    """Setting cluster_mad_k=0 turns clustering off — both verticals and
    the off-axis line are kept (subject to the normal angle prefilter)."""
    verticals = [((20 + 40 * i, 30), (20 + 40 * i, 370)) for i in range(5)]
    diagonal = [((220, 30), (380, 370))]
    arr = _scene_with_lines((400, 400), verticals + diagonal)
    params = film_scratches.ScratchDetectorParams(
        angle_ranges_deg=((-90.0, 90.0),),
        cluster_mad_k=0.0,  # disable clustering
    )
    mask = film_scratches.detect_scratches(arr, params)
    diag_px = int((mask[:, 220:] > 0).sum())
    # With clustering off, the diagonal leaves a clear trace
    assert diag_px > 500, \
        f"with clustering disabled the diagonal must remain; got {diag_px} px"


def test_cluster_filter_skipped_below_min_lines():
    """Below cluster_min_lines we don't have enough samples to fit a
    cluster; the prefilter alone gates."""
    arr = _scene_with_scratch(size=(400, 400),
                              scratch_endpoints=((200, 40), (200, 380)))
    # min_lines high enough that we never have that many candidates
    params = film_scratches.ScratchDetectorParams(cluster_min_lines=999)
    mask = film_scratches.detect_scratches(arr, params)
    # Single vertical should still be caught (clustering bypassed)
    assert int((mask > 0).sum()) > 100


def test_cluster_filter_handles_wraparound_at_pm_90():
    """Near-vertical Hough segments often land at both +89° and -89° due
    to discretisation. The doubled-angle representation must merge them
    into the same cluster — neither should be treated as an outlier."""
    # Tilted +1° and -1° from vertical (both very close to ±90)
    h, w = 400, 400
    line_specs = [
        ((200 + dx, 30), (200 + dx + dxi, 370))
        for dx, dxi in [(-30, 7), (-10, -7), (10, 7), (30, -7), (50, 7)]
    ]
    arr = _scene_with_lines((h, w), line_specs)
    params = film_scratches.ScratchDetectorParams(
        angle_ranges_deg=((-90.0, 90.0),),
    )
    mask = film_scratches.detect_scratches(arr, params)
    # All five tilt-mixed near-vertical lines should survive (cluster
    # centre near ±90° captures both signs).
    assert int((mask > 0).sum()) > 500, \
        "doubled-angle clustering must merge ±89° detections"


def test_density_filter_rejects_heavy_overlap_blob():
    """Five vertical lines packed within 5 px of each other fuse into a
    single fat blob whose minor axis exceeds the threshold — that blob
    must be rejected. A separate isolated vertical scratch on the same
    image must survive.
    """
    h, w = 400, 400
    # Tight stack: five verticals 5 px apart at x ∈ [200, 220]. With
    # rendered thickness max_width_px=8 they fully merge.
    tight_stack = [((200 + 5 * i, 30), (200 + 5 * i, 370)) for i in range(5)]
    isolated = [((80, 30), (80, 370))]
    arr = _scene_with_lines((h, w), tight_stack + isolated)
    mask = film_scratches.detect_scratches(arr)
    # Stack region near x=200-225: width ~ 25-30 px (above cap of 25)
    stack_band = mask[:, 195:235]
    # Isolated region near x=80
    iso_band = mask[:, 60:100]
    assert int((stack_band > 0).sum()) < 200, \
        f"heavy-overlap blob should be rejected; got {int((stack_band > 0).sum())} px"
    assert int((iso_band > 0).sum()) > 1500, \
        f"isolated scratch must survive density filter; got {int((iso_band > 0).sum())} px"


def test_density_filter_keeps_modestly_paired_scratches():
    """Two parallel scratches 30 px apart do NOT fuse into a single
    component (the 8-px render width leaves a gap). Both should
    survive."""
    h, w = 400, 400
    pair = [((150, 30), (150, 370)), ((180, 30), (180, 370))]
    arr = _scene_with_lines((h, w), pair)
    mask = film_scratches.detect_scratches(arr)
    band = mask[:, 130:200]
    # Both scratches present → roughly 2 * (h-60) * (8 + dilation) px
    assert int((band > 0).sum()) > 2500, \
        f"well-spaced parallel pair must both survive; got {int((band > 0).sum())} px"


def test_density_filter_disabled_when_zero():
    """Setting cluster_max_short_axis_px=0 turns the density filter off
    — heavy-overlap blobs survive."""
    h, w = 400, 400
    tight_stack = [((200 + 5 * i, 30), (200 + 5 * i, 370)) for i in range(5)]
    arr = _scene_with_lines((h, w), tight_stack)
    params = film_scratches.ScratchDetectorParams(
        cluster_max_short_axis_px=0,
    )
    mask = film_scratches.detect_scratches(arr, params)
    stack_band = mask[:, 195:235]
    assert int((stack_band > 0).sum()) > 5000, \
        f"with density filter off, heavy stack must remain; got {int((stack_band > 0).sum())} px"


def test_density_filter_info_reports_kept_and_rejected():
    """detect_scratches_with_info must surface n_components_in/out so
    the cleaning CLI can report rejection counts."""
    h, w = 400, 400
    tight_stack = [((200 + 5 * i, 30), (200 + 5 * i, 370)) for i in range(5)]
    isolated = [((80, 30), (80, 370))]
    arr = _scene_with_lines((h, w), tight_stack + isolated)
    _mask, info = film_scratches.detect_scratches_with_info(arr)
    # At least 2 components went in (stack + isolated, possibly more
    # from texture noise); at least 1 (isolated) survived.
    assert info["n_components_in"] >= 2
    assert info["n_components_out"] >= 1
    assert info["n_components_out"] <= info["n_components_in"]
    assert info["n_components_in"] > info["n_components_out"], \
        "density filter should reject at least one component on this scene"


def test_match_time_mask_disabled_by_default(tmp_path, monkeypatch):
    """Without the env var, read_overlap_region returns the raw
    (unmasked) reproject of the source."""
    from align import geo
    geo.clear_overlap_cache()
    monkeypatch.delenv("DECLASS_MATCH_TIME_SCRATCH_MASK", raising=False)

    ref = tmp_path / "ref.tif"
    bounds = (0.0, 0.0, 4000.0, 4000.0)
    arr = _scene_with_scratch(size=(400, 400))
    _write_geotiff(ref, arr, bounds=bounds)

    with rasterio.open(ref) as src:
        out, _ = geo.read_overlap_region(src, bounds, "EPSG:3857", 10.0)
    # Bright pixels should still be present — feature unchanged from
    # the historic behaviour when the opt-in flag is absent.
    bright_frac_before = float((arr >= 250).sum()) / arr.size
    bright_frac_after = float((out >= 250).sum()) / out.size
    assert bright_frac_after >= bright_frac_before * 0.5, \
        "without the env var, masking must NOT alter the array"
