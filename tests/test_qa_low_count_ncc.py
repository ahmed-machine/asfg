"""Tests for the QA low-count-NCC abstain rule.

When fewer than 10 match pairs reach the M_geo fit and the majority came
from the NCC fallback workers, the holdout CV cannot reliably run
(``split_holdout_pairs`` needs ≥12 auto pairs). QA must reject that
combination rather than silently accept.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio

from align.qa import build_candidate_report

from .helpers import write_test_raster


@pytest.fixture
def paired_rasters(tmp_path: Path):
    ref = write_test_raster(tmp_path / "ref.tif",
                            bounds=(50.0, 25.0, 51.0, 26.0),
                            crs="EPSG:4326", width=32, height=32, fill=200)
    out = write_test_raster(tmp_path / "out.tif",
                            bounds=(50.0, 25.0, 51.0, 26.0),
                            crs="EPSG:4326", width=32, height=32, fill=200)
    with rasterio.open(ref) as src:
        bounds = src.bounds
    overlap = (bounds.left, bounds.bottom, bounds.right, bounds.top)
    return str(out), str(ref), overlap


def test_low_count_ncc_majority_rejects(paired_rasters):
    out, ref, overlap = paired_rasters
    report = build_candidate_report(
        "primary", out, ref, overlap, work_crs="EPSG:4326",
        holdout_pairs=[],
        M_geo=None,
        coverage=0.0,
        cv_mean_m=None,
        hypothesis_id="h1",
        eval_res=8.0,
        image_metrics={},  # geometry_only path → no image-score gate
        match_evidence={"total_pairs": 6, "ncc_fallback_pairs": 5},
    )
    assert report.accepted is False
    assert "low_count_ncc_only" in report.reasons


def test_low_count_dense_majority_not_flagged(paired_rasters):
    out, ref, overlap = paired_rasters
    report = build_candidate_report(
        "primary", out, ref, overlap, work_crs="EPSG:4326",
        holdout_pairs=[],
        M_geo=None,
        coverage=0.0,
        cv_mean_m=None,
        hypothesis_id="h1",
        eval_res=8.0,
        image_metrics={},
        match_evidence={"total_pairs": 6, "ncc_fallback_pairs": 1},
    )
    assert "low_count_ncc_only" not in report.reasons


def test_high_count_ncc_majority_not_flagged(paired_rasters):
    out, ref, overlap = paired_rasters
    report = build_candidate_report(
        "primary", out, ref, overlap, work_crs="EPSG:4326",
        holdout_pairs=[],
        M_geo=None,
        coverage=0.0,
        cv_mean_m=None,
        hypothesis_id="h1",
        eval_res=8.0,
        image_metrics={},
        match_evidence={"total_pairs": 40, "ncc_fallback_pairs": 30},
    )
    assert "low_count_ncc_only" not in report.reasons


def test_missing_match_evidence_is_benign(paired_rasters):
    out, ref, overlap = paired_rasters
    report = build_candidate_report(
        "primary", out, ref, overlap, work_crs="EPSG:4326",
        holdout_pairs=[],
        M_geo=None,
        coverage=0.0,
        cv_mean_m=None,
        hypothesis_id="h1",
        eval_res=8.0,
        image_metrics={},
        match_evidence=None,
    )
    assert "low_count_ncc_only" not in report.reasons
