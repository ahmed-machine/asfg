"""Tests for the anchor-derived rigid precorrection branch in
``step_feature_matching``.

The anchor pre-search rigid fit captures translation, rotation and scale
from the located anchors. Translation has always been propagated, but
rotation + scale were previously logged and discarded — leaving the
post-coarse target rotated several degrees off from the reference on
KH-4B Bahrain (run_v9: anchor pre-search detected rot=−5.67°, only the
translation was applied). The new branch in pipeline.py applies the
rotation/scale via ``apply_scale_rotation_precorrection`` when they
clear the significance threshold, then re-locates anchors on the
precorrected target before RoMa runs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import align.pipeline as pipeline


def _bare_state(rot_deg: float, scale: float):
    """A SimpleNamespace with just the fields the precorrection branch
    reads + writes. Avoids constructing a full AlignState (which needs
    paths and a work CRS)."""
    return SimpleNamespace(
        anchor_presearch_rotation_deg=rot_deg,
        anchor_presearch_scale=scale,
        overlap=SimpleNamespace(left=0.0, bottom=0.0, right=1000.0, top=1000.0),
        current_input="/tmp/ignored.tif",
        work_crs="EPSG:3857",
        precorrection_applied=False,
        precorrection_tmp=None,
        temp_paths=[],
        matched_pairs=[],
        match_source_counts={},
        anchors_path=None,  # so the re-locate call short-circuits
    )


def test_precorrection_skipped_when_rotation_and_scale_both_small(monkeypatch):
    """Identity (0° rotation, 1.0 scale) → precorrection NOT applied."""
    state = _bare_state(rot_deg=0.0, scale=1.0)

    called = {"value": False}

    def _fake_apply(*args, **kwargs):
        called["value"] = True
        return "/tmp/precorrected.tif"

    monkeypatch.setattr(pipeline, "apply_scale_rotation_precorrection", _fake_apply)

    sentinel_src = SimpleNamespace(close=lambda: None)
    sentinel_ws = SimpleNamespace()
    applied, src, ws = pipeline._maybe_apply_anchor_rigid_precorrection(
        state, sentinel_ws, src_ref=None, src_offset=sentinel_src, profiler=None,
    )

    assert applied is False
    assert called["value"] is False
    # Caller's src/ws should be returned unchanged
    assert src is sentinel_src
    assert ws is sentinel_ws


def test_precorrection_skipped_when_just_below_threshold(monkeypatch):
    """0.5° rotation, 1.5% scale change — both below the trigger floors
    (1°, 2%). Identity transforms should NOT precorrect."""
    state = _bare_state(rot_deg=0.5, scale=1.015)

    called = {"value": False}
    monkeypatch.setattr(pipeline, "apply_scale_rotation_precorrection",
                        lambda *a, **kw: called.__setitem__("value", True))

    sentinel_src = SimpleNamespace(close=lambda: None)
    applied, _, _ = pipeline._maybe_apply_anchor_rigid_precorrection(
        state, ws=SimpleNamespace(), src_ref=None,
        src_offset=sentinel_src, profiler=None,
    )

    assert applied is False
    assert called["value"] is False


def test_precorrection_fires_on_significant_rotation(monkeypatch, tmp_path):
    """5.67° rotation (the v9 KH-4B Bahrain case) → precorrection
    applied, workspace rebuilt, anchors re-located."""
    state = _bare_state(rot_deg=-5.67, scale=0.98)
    state.anchors_path = "/tmp/whatever.json"  # to exercise the re-locate

    precorrected_path = str(tmp_path / "precorrected.tif")
    apply_calls = []

    def _fake_apply(input_path, scale, rotation_deg, work_crs,
                    overlap_center=None, scale_y=None):
        apply_calls.append({
            "input_path": input_path,
            "scale": scale,
            "rotation_deg": rotation_deg,
            "overlap_center": overlap_center,
        })
        return precorrected_path

    monkeypatch.setattr(pipeline, "apply_scale_rotation_precorrection", _fake_apply)

    # Stub rasterio.open so the precorrection branch's reopen succeeds
    new_src_sentinel = SimpleNamespace(close=lambda: None, _rebuilt=True)
    monkeypatch.setattr(pipeline.rasterio, "open",
                        lambda *a, **kw: new_src_sentinel)

    # Stub _prepare_matching_workspace + _locate_anchor_matches
    new_ws_sentinel = SimpleNamespace(_rebuilt=True)
    prepare_calls = []
    monkeypatch.setattr(pipeline, "_prepare_matching_workspace",
                        lambda s, off, ref: (prepare_calls.append(1)
                                             or new_ws_sentinel))
    relocate_calls = []
    monkeypatch.setattr(pipeline, "_locate_anchor_matches",
                        lambda s, ws, sr, so, p: relocate_calls.append(1))

    closed = {"value": False}
    sentinel_old_src = SimpleNamespace(close=lambda: closed.__setitem__("value", True))

    applied, returned_src, returned_ws = pipeline._maybe_apply_anchor_rigid_precorrection(
        state, ws=SimpleNamespace(), src_ref=None,
        src_offset=sentinel_old_src, profiler=None,
    )

    assert applied is True
    assert apply_calls and apply_calls[0]["rotation_deg"] == pytest.approx(-5.67)
    assert apply_calls[0]["scale"] == pytest.approx(0.98)
    assert prepare_calls == [1], "workspace must be rebuilt"
    assert relocate_calls == [1], "anchors must be re-located after precorrection"
    assert state.current_input == precorrected_path
    assert state.precorrection_applied is True
    assert state.precorrection_tmp == precorrected_path
    assert precorrected_path in state.temp_paths
    assert closed["value"] is True, "old src_offset must be closed"
    assert returned_src is new_src_sentinel
    assert returned_ws is new_ws_sentinel


def test_precorrection_fires_on_significant_scale(monkeypatch, tmp_path):
    """0.95 scale (5% off) with no rotation → precorrection fires."""
    state = _bare_state(rot_deg=0.1, scale=0.95)
    monkeypatch.setattr(pipeline, "apply_scale_rotation_precorrection",
                        lambda *a, **kw: str(tmp_path / "precorr.tif"))
    monkeypatch.setattr(pipeline.rasterio, "open",
                        lambda *a, **kw: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(pipeline, "_prepare_matching_workspace",
                        lambda s, off, ref: SimpleNamespace())
    monkeypatch.setattr(pipeline, "_locate_anchor_matches",
                        lambda s, ws, sr, so, p: None)

    applied, _, _ = pipeline._maybe_apply_anchor_rigid_precorrection(
        state, ws=SimpleNamespace(), src_ref=None,
        src_offset=SimpleNamespace(close=lambda: None), profiler=None,
    )
    assert applied is True


def test_precorrection_failure_falls_through_without_crash(monkeypatch):
    """If apply_scale_rotation_precorrection returns None (e.g. a GDAL
    error), the branch must leave state unchanged and return the
    original src_offset / ws so the caller's RoMa stage still runs."""
    state = _bare_state(rot_deg=-5.67, scale=0.98)
    monkeypatch.setattr(pipeline, "apply_scale_rotation_precorrection",
                        lambda *a, **kw: None)

    original_src = SimpleNamespace(close=lambda: None)
    original_ws = SimpleNamespace()

    applied, src, ws = pipeline._maybe_apply_anchor_rigid_precorrection(
        state, ws=original_ws, src_ref=None,
        src_offset=original_src, profiler=None,
    )

    assert applied is False
    assert src is original_src, "src must be unchanged on failure"
    assert ws is original_ws, "ws must be unchanged on failure"
    assert state.precorrection_applied is False


def test_per_patch_threshold_is_3(monkeypatch, tmp_path):
    """3 patches with significant rotations is enough to prefer the
    non-rigid path. LinearNDInterpolator needs ≥ 3 points in 2D for a
    non-degenerate triangulation; outside the convex hull the
    NearestNDInterpolator fallback handles extrapolation."""
    state = _bare_state(rot_deg=-5.67, scale=0.98)
    state.anchor_patch_results = [
        {"cx": 0.0, "cy": 0.0, "scale_x": 0.99, "scale_y": 0.99,
         "rotation": -2.21, "status": "ok"},
        {"cx": 1000.0, "cy": 0.0, "scale_x": 0.98, "scale_y": 0.98,
         "rotation": 3.61, "status": "ok"},
        {"cx": 0.0, "cy": 1000.0, "scale_x": 0.97, "scale_y": 0.97,
         "rotation": 1.05, "status": "ok"},
    ]
    state.anchors_path = "/tmp/whatever.json"

    local_calls = []
    rigid_calls = []
    monkeypatch.setattr(pipeline, "apply_local_scale_precorrection",
                        lambda *a, **kw: local_calls.append((a, kw))
                        or str(tmp_path / "local.tif"))
    monkeypatch.setattr(pipeline, "apply_scale_rotation_precorrection",
                        lambda *a, **kw: rigid_calls.append((a, kw))
                        or str(tmp_path / "rigid.tif"))
    monkeypatch.setattr(pipeline.rasterio, "open",
                        lambda *a, **kw: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(pipeline, "_prepare_matching_workspace",
                        lambda s, off, ref: SimpleNamespace())
    monkeypatch.setattr(pipeline, "_locate_anchor_matches",
                        lambda s, ws, sr, so, p: None)

    applied, _, _ = pipeline._maybe_apply_anchor_rigid_precorrection(
        state, ws=SimpleNamespace(), src_ref=None,
        src_offset=SimpleNamespace(close=lambda: None), profiler=None,
    )

    assert applied is True
    assert len(local_calls) == 1, "non-rigid path must fire at N=3"
    assert len(rigid_calls) == 0


def test_per_patch_path_used_when_patches_available(monkeypatch, tmp_path):
    """≥4 per-anchor patches with significant rotations → use
    apply_local_scale_precorrection, not the rigid global fit."""
    state = _bare_state(rot_deg=-5.67, scale=0.98)
    state.anchor_patch_results = [
        {"cx": 0.0, "cy": 0.0, "scale_x": 0.99, "scale_y": 0.99,
         "rotation": -2.21, "status": "ok"},
        {"cx": 1000.0, "cy": 0.0, "scale_x": 0.98, "scale_y": 0.98,
         "rotation": 3.61, "status": "ok"},
        {"cx": 0.0, "cy": 1000.0, "scale_x": 0.97, "scale_y": 0.97,
         "rotation": 1.05, "status": "ok"},
        {"cx": 1000.0, "cy": 1000.0, "scale_x": 1.01, "scale_y": 1.01,
         "rotation": -1.5, "status": "ok"},
    ]
    state.anchors_path = "/tmp/whatever.json"

    local_calls = []
    rigid_calls = []
    monkeypatch.setattr(pipeline, "apply_local_scale_precorrection",
                        lambda *a, **kw: local_calls.append((a, kw))
                        or str(tmp_path / "local.tif"))
    monkeypatch.setattr(pipeline, "apply_scale_rotation_precorrection",
                        lambda *a, **kw: rigid_calls.append((a, kw))
                        or str(tmp_path / "rigid.tif"))
    monkeypatch.setattr(pipeline.rasterio, "open",
                        lambda *a, **kw: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(pipeline, "_prepare_matching_workspace",
                        lambda s, off, ref: SimpleNamespace())
    monkeypatch.setattr(pipeline, "_locate_anchor_matches",
                        lambda s, ws, sr, so, p: None)

    applied, _, _ = pipeline._maybe_apply_anchor_rigid_precorrection(
        state, ws=SimpleNamespace(), src_ref=None,
        src_offset=SimpleNamespace(close=lambda: None), profiler=None,
    )

    assert applied is True
    assert len(local_calls) == 1, "non-rigid path must be taken"
    assert len(rigid_calls) == 0, "rigid path must NOT fire when patches present"
    # Patches are passed through verbatim
    args, _kw = local_calls[0]
    passed_patches = args[1]
    assert len(passed_patches) == 4


def test_per_patch_path_skipped_when_all_patches_below_threshold(monkeypatch, tmp_path):
    """4 patches with tiny rotations (all < 1°) and tiny scale (all < 2%)
    → fall back to rigid path (which itself may be skipped by the
    significance gate)."""
    state = _bare_state(rot_deg=-5.67, scale=0.98)  # rigid IS significant
    state.anchor_patch_results = [
        {"cx": 0.0, "cy": 0.0, "scale_x": 1.0, "scale_y": 1.0,
         "rotation": 0.1, "status": "ok"},
        {"cx": 1000.0, "cy": 0.0, "scale_x": 1.0, "scale_y": 1.0,
         "rotation": -0.2, "status": "ok"},
        {"cx": 0.0, "cy": 1000.0, "scale_x": 1.0, "scale_y": 1.0,
         "rotation": 0.05, "status": "ok"},
        {"cx": 1000.0, "cy": 1000.0, "scale_x": 1.0, "scale_y": 1.0,
         "rotation": -0.3, "status": "ok"},
    ]
    state.anchors_path = "/tmp/whatever.json"

    local_calls = []
    rigid_calls = []
    monkeypatch.setattr(pipeline, "apply_local_scale_precorrection",
                        lambda *a, **kw: local_calls.append((a, kw))
                        or str(tmp_path / "local.tif"))
    monkeypatch.setattr(pipeline, "apply_scale_rotation_precorrection",
                        lambda *a, **kw: rigid_calls.append((a, kw))
                        or str(tmp_path / "rigid.tif"))
    monkeypatch.setattr(pipeline.rasterio, "open",
                        lambda *a, **kw: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(pipeline, "_prepare_matching_workspace",
                        lambda s, off, ref: SimpleNamespace())
    monkeypatch.setattr(pipeline, "_locate_anchor_matches",
                        lambda s, ws, sr, so, p: None)

    applied, _, _ = pipeline._maybe_apply_anchor_rigid_precorrection(
        state, ws=SimpleNamespace(), src_ref=None,
        src_offset=SimpleNamespace(close=lambda: None), profiler=None,
    )

    assert applied is True
    assert len(local_calls) == 0, "non-rigid path must skip when no patch is significant"
    assert len(rigid_calls) == 1, "rigid path should fire on the global rot=-5.67° fit"
