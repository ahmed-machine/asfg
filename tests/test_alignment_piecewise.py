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
def test_affine_candidate_skips_post_warp_regression(monkeypatch, piecewise_case):
    """The affine candidate's "warp" is just M_geo applied to pixels, so any
    gap between coord-space (holdout_metrics) and image-space
    (holdout_warp_metrics) is content drift + resampling, not warp quality.
    The regression rule must skip the affine candidate even when the gap
    is large enough that it would fire for a non-affine candidate.
    """
    monkeypatch.setattr(qa, "evaluate_alignment_quality_paths", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        qa,
        "compute_holdout_affine_metrics",
        lambda *args, **kwargs: {
            "count": 6, "mean_m": 10.0, "median_m": 10.0, "p90_m": 14.0, "max_m": 18.0,
        },
    )
    monkeypatch.setattr(
        qa,
        "compute_holdout_warp_metrics",
        lambda *args, **kwargs: {
            "count": 6, "mean_m": 30.0, "median_m": 25.0, "p90_m": 45.0, "max_m": 60.0,
        },
    )

    report = build_candidate_report(
        "affine",
        "affine.tif",
        "reference.tif",
        (0.0, 0.0, 1.0, 1.0),
        "EPSG:3857",
        holdout_pairs=[object()],
        M_geo=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        coverage=0.25,
    )

    assert "post_warp_holdout_regression" not in report.reasons
    piecewise_case.record_case_summary(
        {
            "candidate": report.candidate,
            "reasons": report.reasons,
            "post_warp_mean": report.holdout_metrics["post_warp"]["mean_m"],
        }
    )


@pytest.mark.fast
@pytest.mark.qa
def test_grid_candidate_uses_affine_post_warp_baseline(monkeypatch, piecewise_case):
    """When the affine candidate's image-space post-warp metrics are
    supplied, the rule compares grid against that baseline (apples-to-apples)
    instead of against the M_geo coord-space prediction. With a tight
    coord-space prediction (~10 m) but an affine post-warp around 28 m
    (cross-temporal content-drift floor), grid post-warp at 30 m is NOT a
    regression: comparing 30 against 10 would (incorrectly) trip the rule,
    but comparing 30 against 28 correctly does not.
    """
    monkeypatch.setattr(qa, "evaluate_alignment_quality_paths", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        qa,
        "compute_holdout_affine_metrics",
        lambda *args, **kwargs: {
            "count": 6, "mean_m": 10.0, "median_m": 10.0, "p90_m": 14.0, "max_m": 18.0,
        },
    )
    monkeypatch.setattr(
        qa,
        "compute_holdout_warp_metrics",
        lambda *args, **kwargs: {
            "count": 6, "mean_m": 30.0, "median_m": 25.0, "p90_m": 45.0, "max_m": 60.0,
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
        affine_post_warp_metrics={
            "count": 6, "mean_m": 28.0, "median_m": 24.0, "p90_m": 42.0, "max_m": 55.0,
        },
    )

    assert "post_warp_holdout_regression" not in report.reasons
    piecewise_case.record_case_summary(
        {"candidate": report.candidate, "reasons": report.reasons}
    )


@pytest.mark.fast
@pytest.mark.qa
def test_grid_candidate_trips_when_regresses_vs_affine_post_warp(monkeypatch, piecewise_case):
    """When grid post-warp regresses against the affine post-warp baseline
    by ratio >1.5, the rule fires correctly.
    """
    monkeypatch.setattr(qa, "evaluate_alignment_quality_paths", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        qa,
        "compute_holdout_affine_metrics",
        lambda *args, **kwargs: {
            "count": 6, "mean_m": 10.0, "median_m": 10.0, "p90_m": 14.0, "max_m": 18.0,
        },
    )
    monkeypatch.setattr(
        qa,
        "compute_holdout_warp_metrics",
        lambda *args, **kwargs: {
            "count": 6, "mean_m": 60.0, "median_m": 55.0, "p90_m": 90.0, "max_m": 110.0,
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
        affine_post_warp_metrics={
            "count": 6, "mean_m": 28.0, "median_m": 24.0, "p90_m": 42.0, "max_m": 55.0,
        },
    )

    assert report.accepted is False
    assert "post_warp_holdout_regression" in report.reasons
    piecewise_case.record_case_summary(
        {"candidate": report.candidate, "reasons": report.reasons}
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
    # ``qa_writes`` collects every selected_candidate passed to
    # write_qa_report. step_select_warp_and_apply now writes a
    # provisional qa.json after the grid candidate (so a C-level crash
    # during the affine baseline doesn't lose the alignment artifact)
    # and re-writes after the full ranking, so we expect two writes:
    # ["grid", "affine"] for this test where affine wins.
    qa_writes: list = []

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
        lambda path, reports, *, selected_candidate=None, metadata=None: qa_writes.append(selected_candidate),
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
    # Final write must reflect the affine winner; an earlier provisional
    # write from the grid candidate is also expected so a downstream
    # crash can't lose the alignment artifact.
    assert qa_writes, "expected at least one write_qa_report call"
    assert qa_writes[-1] == "affine"
    assert qa_writes[0] == "grid"
    piecewise_case.record_case_summary(
        {
            "selected_candidate": qa_writes[-1],
            "provisional_writes": qa_writes,
            "output_contents": output_path.read_text(encoding="utf-8"),
        }
    )


@pytest.mark.fast
@pytest.mark.selection
def test_step_select_warp_writes_provisional_qa_before_affine(tmp_path, monkeypatch, piecewise_case):
    """Even if the affine baseline aborts the step (raised exception
    standing in for the C-level segfault / OOM-kill that gdal.Warp has
    been observed to trigger on huge KH-4B panoramic overlap regions),
    the grid candidate's qa.json must already be on disk so the
    strip-manifest's idempotency + extrapolation-retry chain can reach
    DA026 on a re-run.
    """
    output_path = tmp_path / "aligned.tif"
    qa_writes: list = []

    def fake_apply_warp(*args, **kwargs):
        Path(args[1]).write_text("grid", encoding="utf-8")
        return args[1]

    def fake_affine_warp_crash(*args, **kwargs):
        # Stand-in for the C-level crash: a Python exception that
        # propagates out of the affine warp candidate before any QA is
        # built. With the provisional-write fix in place, the grid
        # candidate's qa.json is already on disk by this point.
        raise RuntimeError("simulated gdal.Warp segfault")

    def fake_build_candidate_report(candidate_name, output_path, *args, **kwargs):
        return QaReport(
            candidate=candidate_name,
            output_path=output_path,
            total_score=80.0 if candidate_name == "grid" else 25.0,
            confidence=0.50 if candidate_name == "grid" else 0.90,
            accepted=candidate_name == "grid",
        )

    monkeypatch.setattr(pipeline, "apply_warp", fake_apply_warp)
    monkeypatch.setattr(pipeline, "_apply_affine_warp", fake_affine_warp_crash)
    monkeypatch.setattr(pipeline, "evaluate_alignment_quality_paths", lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline, "build_candidate_report", fake_build_candidate_report)
    monkeypatch.setattr(
        pipeline,
        "write_qa_report",
        lambda path, reports, *, selected_candidate=None, metadata=None: qa_writes.append(selected_candidate),
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

    # The fake affine raises after the provisional grid write. We let
    # the exception propagate; what matters is that qa.json was already
    # written for grid before the raise.
    with pytest.raises(RuntimeError, match="simulated gdal.Warp segfault"):
        step_select_warp_and_apply(state)

    # The grid candidate must have produced a provisional qa.json write
    # before the affine attempt blew up.
    assert qa_writes == ["grid"], (
        f"expected exactly one provisional write of grid before the "
        f"affine crash; got {qa_writes!r}"
    )
    piecewise_case.record_case_summary(
        {
            "provisional_writes_before_crash": qa_writes,
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
    output_path = tmp_path / "frame_aligned.tif"
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

    # Abstained run preserves the warp under a ``_aligned_rejected.tif``
    # name (mosaic glob skips it; user can still inspect manually).
    rejected_path = output_path.parent / "frame_aligned_rejected.tif"
    assert result.abstained is True
    assert not output_path.exists(), (
        "abstained run should rename the output, not leave it under the "
        "canonical _aligned.tif name"
    )
    assert rejected_path.exists(), (
        f"abstained run should preserve the warp under {rejected_path.name}"
    )
    assert selected["candidate"] == "grid"
    piecewise_case.record_case_summary(
        {
            "selected_candidate": selected["candidate"],
            "abstained": result.abstained,
            "output_exists": output_path.exists(),
            "rejected_path_exists": rejected_path.exists(),
        }
    )
