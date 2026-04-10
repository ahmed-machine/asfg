from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

import align.scale as scale
from align.filtering import refine_matches_phase_correlation
from align.types import MatchPair

from .helpers import make_synthetic_feature_image, warp_synthetic_image, write_array_raster


class _SequentialProcessPool:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def map(self, fn, iterable, chunksize=1):
        for item in iterable:
            yield fn(item)


@pytest.mark.fast
@pytest.mark.selection
def test_detect_multiscale_ncc_recovers_scale_and_rotation(monkeypatch, piecewise_case):
    monkeypatch.setattr(scale, "ProcessPoolExecutor", _SequentialProcessPool)

    ref_image = make_synthetic_feature_image(size=320)
    offset_image = warp_synthetic_image(ref_image, scale=1.12, rotation_deg=3.0)

    result = scale.detect_multiscale_ncc(ref_image, offset_image, expected_scale=1.12)

    assert result is not None
    detected_scale, detected_rotation, correlation = result
    assert detected_scale == pytest.approx(1.12, abs=0.03)
    assert detected_rotation == pytest.approx(3.0, abs=1.5)
    assert correlation >= 0.75
    piecewise_case.record_case_summary(
        {
            "detected_scale": detected_scale,
            "detected_rotation_deg": detected_rotation,
            "correlation": correlation,
        }
    )


@pytest.mark.fast
@pytest.mark.selection
def test_detect_scale_rotation_recovers_transform_from_synthetic_rasters(
    tmp_path,
    monkeypatch,
    piecewise_case,
):
    monkeypatch.setattr(scale, "ProcessPoolExecutor", _SequentialProcessPool)
    monkeypatch.setattr(
        scale,
        "make_land_mask",
        lambda arr, mode="coastal_obia": ((arr > 0).astype(np.uint8) * 255),
    )

    ref_image = make_synthetic_feature_image(size=320)
    offset_image = warp_synthetic_image(ref_image, scale=1.12, rotation_deg=3.0)

    ref_path = write_array_raster(
        tmp_path / "reference.tif",
        ref_image,
        bounds=(0.0, 0.0, 1600.0, 1600.0),
        crs="EPSG:3857",
    )
    offset_path = write_array_raster(
        tmp_path / "offset.tif",
        offset_image,
        bounds=(0.0, 0.0, 1600.0, 1600.0),
        crs="EPSG:3857",
    )

    with rasterio.open(offset_path) as src_offset, rasterio.open(ref_path) as src_ref:
        result = scale.detect_scale_rotation(
            src_offset,
            src_ref,
            (0.0, 0.0, 1600.0, 1600.0),
            CRS.from_epsg(3857),
            0.0,
            0.0,
            1.12,
            model_cache=None,
        )

    assert result.method == "multiscale-ncc"
    assert result.scale_x == pytest.approx(1.12, abs=0.03)
    assert result.scale_y == pytest.approx(1.12, abs=0.03)
    assert result.rotation == pytest.approx(3.0, abs=2.0)
    piecewise_case.record_case_summary(
        {
            "method": result.method,
            "scale_x": result.scale_x,
            "scale_y": result.scale_y,
            "rotation_deg": result.rotation,
        }
    )


@pytest.mark.fast
@pytest.mark.selection
def test_refine_matches_phase_correlation_corrects_synthetic_shift(piecewise_case):
    ref_image = make_synthetic_feature_image(size=512)
    offset_image = warp_synthetic_image(ref_image, shift_x_px=6.0)

    transform = from_bounds(0.0, 0.0, 512.0, 512.0, 512, 512)
    true_off_x = 262.0
    true_off_y = 256.0
    initial_pair = MatchPair(
        ref_x=256.0,
        ref_y=256.0,
        off_x=true_off_x - 3.0,
        off_y=true_off_y,
        confidence=0.9,
        name="synthetic",
    )

    refined_pair = refine_matches_phase_correlation(
        [initial_pair],
        ref_image,
        offset_image,
        transform,
        transform,
        0,
        0,
        1.0,
    )[0]

    initial_error = float(np.hypot(initial_pair.off_x - true_off_x, initial_pair.off_y - true_off_y))
    refined_error = float(np.hypot(refined_pair.off_x - true_off_x, refined_pair.off_y - true_off_y))

    assert refined_error < initial_error
    assert refined_error <= 0.1
    piecewise_case.record_case_summary(
        {
            "initial_error_px": initial_error,
            "refined_error_px": refined_error,
            "refined_off_x": refined_pair.off_x,
            "refined_off_y": refined_pair.off_y,
        }
    )
