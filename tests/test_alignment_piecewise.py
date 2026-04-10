from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rasterio.crs import CRS

from align.pipeline import step_coarse_offset, step_global_localization, step_select_warp_and_apply
from align.qa import build_candidate_report
from align.state import AlignState
from align.types import BBox, GlobalHypothesis, QaReport
import align.pipeline as pipeline
import align.qa as qa


@pytest.mark.fast
@pytest.mark.qa
def test_build_candidate_report_rejects_post_warp_regression(monkeypatch, piecewise_case):
    monkeypatch.setattr(qa, "evaluate_alignment_quality_paths", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        qa,
        "compute_holdout_affine_metrics",
        lambda *args, **kwargs: {
            "count": 6,
            "mean_m": 10.0,
            "median_m": 10.0,
            "p90_m": 14.0,
            "max_m": 18.0,
        },
    )
    monkeypatch.setattr(
        qa,
        "compute_holdout_warp_metrics",
        lambda *args, **kwargs: {
            "count": 6,
            "mean_m": 30.0,
            "median_m": 25.0,
            "p90_m": 45.0,
            "max_m": 60.0,
        },
    )

    report = build_candidate_report(
        "grid",
        "grid.tif",
        "reference.tif",
        (0.0, 0.0, 1.0, 1.0),
        "EPSG:3857",
        holdout_pairs=[object()],
        M_geo=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        coverage=0.25,
    )

    assert report.accepted is False
    assert "post_warp_holdout_regression" in report.reasons
    assert report.total_score == pytest.approx((0.60 * 30.0) + (0.25 * 45.0))
    assert report.holdout_metrics["post_warp"]["mean_m"] == 30.0
    piecewise_case.record_case_summary(
        {
            "candidate": report.candidate,
            "accepted": report.accepted,
            "reasons": report.reasons,
            "total_score": report.total_score,
        }
    )


@pytest.mark.fast
@pytest.mark.qa
def test_build_candidate_report_geometry_only_ignores_low_coverage(monkeypatch, piecewise_case):
    monkeypatch.setattr(qa, "evaluate_alignment_quality_paths", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        qa,
        "compute_holdout_affine_metrics",
        lambda *args, **kwargs: {
            "count": 6,
            "mean_m": 8.0,
            "median_m": 8.0,
            "p90_m": 10.0,
            "max_m": 12.0,
        },
    )
    monkeypatch.setattr(qa, "compute_holdout_warp_metrics", lambda *args, **kwargs: {})

    report = build_candidate_report(
        "affine",
        "affine.tif",
        "reference.tif",
        (0.0, 0.0, 1.0, 1.0),
        "EPSG:3857",
        holdout_pairs=[object()],
        M_geo=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        coverage=0.04,
    )

    assert report.accepted is True
    assert "gcp_coverage_low" not in report.reasons
    piecewise_case.record_case_summary(
        {
            "candidate": report.candidate,
            "accepted": report.accepted,
            "reasons": report.reasons,
            "coverage": report.coverage,
        }
    )


@pytest.mark.fast
@pytest.mark.selection
def test_step_global_localization_marks_preprocessed_overlap(piecewise_case):
    state = AlignState(
        input_path="input.tif",
        reference_path="reference.tif",
        output_path="output.tif",
        work_crs=CRS.from_epsg(3857),
        overlap=BBox(0.0, 0.0, 100.0, 100.0),
        anchors_path="anchors.json",
        reference_window=(0.0, 0.0, 1.0, 1.0),
    )

    result = step_global_localization(state, SimpleNamespace(force_global=False))

    assert result.chosen_hypothesis is not None
    assert result.chosen_hypothesis.source == "preprocessed_crop_overlap"
    assert result.chosen_hypothesis.hypothesis_id == "preprocessed_overlap"
    piecewise_case.record_case_summary(
        {
            "hypothesis_id": result.chosen_hypothesis.hypothesis_id,
            "source": result.chosen_hypothesis.source,
        }
    )


@pytest.mark.fast
@pytest.mark.selection
def test_step_select_warp_and_apply_prefers_affine_candidate(tmp_path, monkeypatch, piecewise_case):
    output_path = tmp_path / "aligned.tif"
    qa_selections = {}

    def fake_apply_warp(*args, **kwargs):
        Path(args[1]).write_text("grid", encoding="utf-8")
        return args[1]

    def fake_affine_warp(*args, **kwargs):
        Path(args[1]).write_text("affine", encoding="utf-8")
        return True

    def fake_build_candidate_report(candidate_name, output_path, *args, **kwargs):
        if candidate_name == "affine":
            return QaReport(
                candidate="affine",
                output_path=output_path,
                total_score=12.0,
                confidence=0.92,
                accepted=True,
            )
        return QaReport(
            candidate="grid",
            output_path=output_path,
            total_score=55.0,
            confidence=0.30,
            accepted=False,
            reasons=["post_warp_holdout_regression"],
        )

    monkeypatch.setattr(pipeline, "apply_warp", fake_apply_warp)
    monkeypatch.setattr(pipeline, "_affine_warp_gcps", fake_affine_warp)
    monkeypatch.setattr(pipeline, "evaluate_alignment_quality_paths", lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline, "build_candidate_report", fake_build_candidate_report)
    monkeypatch.setattr(
        pipeline,
        "write_qa_report",
        lambda path, reports, *, selected_candidate=None, metadata=None: qa_selections.setdefault("selected", selected_candidate),
    )

    state = AlignState(
        input_path="input.tif",
        current_input="input.tif",
        reference_path="reference.tif",
        output_path=str(output_path),
        work_crs=CRS.from_epsg(3857),
        overlap=BBox(0.0, 0.0, 100.0, 100.0),
        yes=True,
        offset_res_m=2.0,
        ref_res_m=2.0,
        M_geo=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        gcp_coverage=0.30,
        qa_holdout_pairs=[],
        diagnostics_dir=str(tmp_path / "diagnostics"),
    )

    result = step_select_warp_and_apply(state)

    assert result.abstained is False
    assert output_path.read_text(encoding="utf-8") == "affine"
    assert qa_selections["selected"] == "affine"
    piecewise_case.record_case_summary(
        {
            "selected_candidate": qa_selections["selected"],
            "output_contents": output_path.read_text(encoding="utf-8"),
        }
    )


@pytest.mark.fast
@pytest.mark.qa
def test_build_candidate_report_rejects_high_image_score(monkeypatch, piecewise_case):
    monkeypatch.setattr(
        qa,
        "evaluate_alignment_quality_paths",
        lambda *args, **kwargs: {"score": 220.0, "patch_med": 150.0},
    )
    monkeypatch.setattr(
        qa,
        "compute_holdout_affine_metrics",
        lambda *args, **kwargs: {
            "count": 6,
            "mean_m": 6.0,
            "median_m": 6.0,
            "p90_m": 9.0,
            "max_m": 10.0,
        },
    )
    monkeypatch.setattr(qa, "compute_holdout_warp_metrics", lambda *args, **kwargs: {})

    report = build_candidate_report(
        "grid",
        "grid.tif",
        "reference.tif",
        (0.0, 0.0, 1.0, 1.0),
        "EPSG:3857",
        holdout_pairs=[object()],
        M_geo=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        coverage=0.25,
    )

    assert report.accepted is False
    assert "image_alignment_score_high" in report.reasons
    piecewise_case.record_case_summary(
        {
            "candidate": report.candidate,
            "accepted": report.accepted,
            "reasons": report.reasons,
            "image_score": report.image_metrics["score"],
        }
    )


@pytest.mark.fast
@pytest.mark.selection
def test_step_coarse_offset_skips_for_preprocessed_overlap_with_anchors(tmp_path, piecewise_case):
    state = AlignState(
        input_path="input.tif",
        current_input="input.tif",
        reference_path="reference.tif",
        output_path=str(tmp_path / "output.tif"),
        work_crs=CRS.from_epsg(3857),
        overlap=BBox(0.0, 0.0, 100.0, 100.0),
        anchors_path="anchors.json",
        chosen_hypothesis=GlobalHypothesis(
            hypothesis_id="preprocessed_overlap",
            score=1.0,
            source="preprocessed_crop_overlap",
            left=0.0,
            bottom=0.0,
            right=100.0,
            top=100.0,
            work_crs="EPSG:3857",
        ),
        diagnostics_dir=str(tmp_path / "diagnostics"),
    )

    result = step_coarse_offset(state)

    assert result.coarse_total == 0.0
    assert result.coarse_dx == 0.0
    assert result.coarse_dy == 0.0
    piecewise_case.record_case_summary(
        {
            "coarse_dx": result.coarse_dx,
            "coarse_dy": result.coarse_dy,
            "coarse_total": result.coarse_total,
        }
    )


@pytest.mark.fast
@pytest.mark.selection
def test_step_select_warp_and_apply_abstains_on_rejected_candidate(tmp_path, monkeypatch, piecewise_case):
    output_path = tmp_path / "aligned.tif"
    selected = {}

    monkeypatch.setattr(pipeline, "apply_warp", lambda *args, **kwargs: Path(args[1]).write_text("grid", encoding="utf-8"))
    monkeypatch.setattr(pipeline, "_affine_warp_gcps", lambda *args, **kwargs: False)
    monkeypatch.setattr(pipeline, "evaluate_alignment_quality_paths", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        pipeline,
        "build_candidate_report",
        lambda candidate_name, output_path, *args, **kwargs: QaReport(
            candidate=candidate_name,
            output_path=output_path,
            total_score=120.0,
            confidence=0.20,
            accepted=False,
            reasons=["post_warp_holdout_regression"],
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "write_qa_report",
        lambda path, reports, *, selected_candidate=None, metadata=None: selected.setdefault("candidate", selected_candidate),
    )

    state = AlignState(
        input_path="input.tif",
        current_input="input.tif",
        reference_path="reference.tif",
        output_path=str(output_path),
        work_crs=CRS.from_epsg(3857),
        overlap=BBox(0.0, 0.0, 100.0, 100.0),
        yes=True,
        allow_abstain=True,
        offset_res_m=2.0,
        ref_res_m=2.0,
        M_geo=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        gcp_coverage=0.30,
        qa_holdout_pairs=[],
        diagnostics_dir=str(tmp_path / "diagnostics"),
    )

    result = step_select_warp_and_apply(state)

    assert result.abstained is True
    assert not output_path.exists()
    assert selected["candidate"] == "grid"
    piecewise_case.record_case_summary(
        {
            "selected_candidate": selected["candidate"],
            "abstained": result.abstained,
            "output_exists": output_path.exists(),
        }
    )
