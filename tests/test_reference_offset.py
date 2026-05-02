from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from rasterio.crs import CRS

from align.filtering import correct_reference_offset
from align.pipeline import step_validate_and_filter
from align.qa import estimate_residual_translation_vector
from align.state import AlignState
from align.types import BBox, MatchPair
import align.pipeline as pipeline
from tests.helpers import write_test_raster


def _make_sparse_anchor_bias_case(bias_e: float = 220.0,
                                  bias_n: float = -130.0) -> tuple[list[MatchPair], list[MatchPair]]:
    auto = []
    for idx in range(6):
        off_x = 1_000.0 + idx * 350.0
        off_y = 2_000.0 + (idx % 3) * 250.0
        auto.append(MatchPair(
            ref_x=off_x - bias_e,
            ref_y=off_y - bias_n,
            off_x=off_x,
            off_y=off_y,
            confidence=0.95,
            name=f"roma_{idx}",
        ))
    anchor = MatchPair(
        ref_x=4_200.0,
        ref_y=5_100.0,
        off_x=4_200.0,
        off_y=5_100.0,
        confidence=0.55,
        name="anchor:auto_r1c2",
    )
    return [anchor] + auto, auto


def _make_textured_image(size: int = 768, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.normal(128.0, 42.0, size=(size, size)).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 1.4)
    cv2.rectangle(img, (80, 90), (size - 120, size - 180), 170.0, 4)
    cv2.circle(img, (size // 3, size // 2), 55, 220.0, -1)
    cv2.putText(img, "KH4B", (size // 2 - 80, size // 3),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, 210.0, 6, cv2.LINE_AA)
    return np.clip(img, 0, 255)


@pytest.mark.fast
@pytest.mark.qa
def test_correct_reference_offset_applies_corroborated_single_anchor_translation():
    pairs, auto = _make_sparse_anchor_bias_case()

    corrected, was_corrected, outliers = correct_reference_offset(
        pairs,
        anchor_presearch_offset_m=(210.0, -120.0),
    )

    assert was_corrected is True
    assert outliers == []
    corrected_auto = [p for p in corrected if not p.is_anchor]
    assert len(corrected_auto) == len(auto)
    for pair in corrected_auto:
        assert pair.ref_x == pytest.approx(pair.off_x, abs=1e-6)
        assert pair.ref_y == pytest.approx(pair.off_y, abs=1e-6)


@pytest.mark.fast
@pytest.mark.qa
def test_correct_reference_offset_skips_uncorroborated_single_anchor_translation():
    pairs, auto = _make_sparse_anchor_bias_case()

    corrected, was_corrected, outliers = correct_reference_offset(
        pairs,
        anchor_presearch_offset_m=None,
    )

    assert was_corrected is False
    assert outliers == []
    corrected_auto = [p for p in corrected if not p.is_anchor]
    for before, after in zip(auto, corrected_auto, strict=True):
        assert after.ref_x == before.ref_x
        assert after.ref_y == before.ref_y


@pytest.mark.fast
@pytest.mark.qa
def test_estimate_residual_translation_vector_recovers_signed_shift():
    base = _make_textured_image()
    shift_x_px = -16.0
    shift_y_px = 9.0
    matrix = np.float32([[1.0, 0.0, shift_x_px], [0.0, 1.0, shift_y_px]])
    shifted = cv2.warpAffine(
        base,
        matrix,
        (base.shape[1], base.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    result = estimate_residual_translation_vector(
        base,
        shifted,
        np.ones_like(base, dtype=bool),
        res_m=5.0,
        patch=192,
        stride=96,
    )

    assert result["accepted"] is True
    assert result["count"] >= 24
    assert result["dx_m"] == pytest.approx(-shift_x_px * 5.0, abs=10.0)
    assert result["dy_m"] == pytest.approx(-shift_y_px * 5.0, abs=10.0)


@pytest.mark.fast
@pytest.mark.qa
def test_estimate_residual_translation_vector_rejects_mixed_direction_field():
    base = _make_textured_image()
    height, width = base.shape
    half = height // 2

    top = cv2.warpAffine(
        base[:half],
        np.float32([[1.0, 0.0, 14.0], [0.0, 1.0, 0.0]]),
        (width, half),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    bottom = cv2.warpAffine(
        base[half:],
        np.float32([[1.0, 0.0, -14.0], [0.0, 1.0, 0.0]]),
        (width, height - half),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    mixed = np.vstack([top, bottom])

    result = estimate_residual_translation_vector(
        base,
        mixed,
        np.ones_like(base, dtype=bool),
        res_m=5.0,
        patch=192,
        stride=96,
    )

    assert result["accepted"] is False
    assert result["count"] >= 24
    assert (
        result["mad_x_m"] > 60.0
        or result["sign_coherence_x"] < 0.70
    )


@pytest.mark.fast
@pytest.mark.qa
def test_step_validate_and_filter_applies_residual_translation_before_boundaries(tmp_path, monkeypatch):
    input_path = write_test_raster(
        tmp_path / "input.tif",
        bounds=(0.0, 0.0, 100.0, 100.0),
        crs="EPSG:3857",
        width=32,
        height=32,
        fill=1,
    )
    reference_path = write_test_raster(
        tmp_path / "reference.tif",
        bounds=(0.0, 0.0, 100.0, 100.0),
        crs="EPSG:3857",
        width=32,
        height=32,
        fill=1,
    )

    def fake_iterative_outlier_removal(pairs, *_args, **_kwargs):
        dE = float(np.median([p.ref_x - p.off_x for p in pairs]))
        dN = float(np.median([p.ref_y - p.off_y for p in pairs]))
        M = np.array([[1.0, 0.0, dE], [0.0, 1.0, dN]], dtype=np.float64)
        return list(pairs), M, [0.0 for _ in pairs]

    boundary_capture = {}

    def fake_generate_boundary_gcps(gcps, M_geo, *_args, **_kwargs):
        boundary_capture["M_geo"] = np.array(M_geo, copy=True)
        boundary_capture["gcp_gx"] = [g.gx for g in gcps]
        return []

    monkeypatch.setattr(pipeline, "iterative_outlier_removal", fake_iterative_outlier_removal)
    monkeypatch.setattr(
        pipeline,
        "detect_and_correct_reference_offset",
        lambda original_pairs, filtered_pairs, M_geo, *_args, **_kwargs: (
            filtered_pairs,
            M_geo,
            [0.0 for _ in filtered_pairs],
            False,
        ),
    )
    monkeypatch.setattr(pipeline, "_affine_warp_gcps", lambda *_args, **_kwargs: Path(_args[1]).write_text("preview", encoding="utf-8") or True)
    monkeypatch.setattr(
        pipeline,
        "estimate_residual_translation_vector_paths",
        lambda *args, **kwargs: {
            "accepted": True,
            "dx_m": 120.0,
            "dy_m": -40.0,
            "magnitude_m": float(np.hypot(120.0, -40.0)),
            "count": 30,
            "mad_x_m": 12.0,
            "mad_y_m": 9.0,
            "sign_coherence_x": 0.95,
            "sign_coherence_y": 0.92,
        },
    )
    monkeypatch.setattr(pipeline, "generate_boundary_gcps", fake_generate_boundary_gcps)
    monkeypatch.setattr(pipeline, "generate_debug_image", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_cross_validate_and_robust_refit", lambda state: None)

    matched_pairs = [
        MatchPair(ref_x=10.0 + i * 5.0, ref_y=20.0 + i * 3.0,
                  off_x=0.0 + i * 5.0, off_y=25.0 + i * 3.0,
                  confidence=0.9, name=f"roma_{i}")
        for i in range(6)
    ]
    holdout_pairs = [
        MatchPair(ref_x=42.0, ref_y=18.0, off_x=30.0, off_y=24.0,
                  confidence=0.9, name="holdout_0")
    ]

    state = AlignState(
        input_path=str(input_path),
        current_input=str(input_path),
        reference_path=str(reference_path),
        output_path=str(tmp_path / "aligned.tif"),
        work_crs=CRS.from_epsg(3857),
        overlap=BBox(0.0, 0.0, 100.0, 100.0),
        offset_res_m=2.0,
        ref_res_m=2.0,
        matched_pairs=matched_pairs,
        qa_holdout_pairs=holdout_pairs,
        used_neural=True,
        skip_fpp=True,
    )

    result = step_validate_and_filter(state)

    assert result.residual_translation_calibration_m == pytest.approx((120.0, -40.0))
    assert result.residual_translation_patch_count == 30
    assert result.matched_pairs[0].ref_x == pytest.approx(130.0)
    assert result.matched_pairs[0].ref_y == pytest.approx(-20.0)
    assert result.qa_holdout_pairs[0].ref_x == pytest.approx(162.0)
    assert result.qa_holdout_pairs[0].ref_y == pytest.approx(-22.0)
    assert boundary_capture["M_geo"][0, 2] == pytest.approx(130.0)
    assert boundary_capture["M_geo"][1, 2] == pytest.approx(-45.0)
