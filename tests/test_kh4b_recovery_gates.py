from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rasterio.crs import CRS

import paths as paths_mod
import process
from align import pipeline
from align.grid_optim import _residual_fit_observable
from align.state import AlignState
from align.types import BBox, GCP, MatchPair
from preprocess import mosaic

from .helpers import make_scene, write_test_raster


def _state_for_pairs(pairs: list[MatchPair]) -> AlignState:
    state = AlignState(
        input_path="target.tif",
        reference_path="reference.tif",
        output_path="aligned.tif",
        work_crs=CRS.from_epsg(3857),
        overlap=BBox(0.0, 0.0, 100.0, 100.0),
    )
    state.matched_pairs = pairs
    state.gcps = [
        GCP(col=p.off_x, row=p.off_y, gx=p.ref_x, gy=p.ref_y, name=p.name)
        for p in pairs
    ]
    state.M_geo = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    state.geo_residuals = [8.0 for _ in pairs]
    state.gcp_coverage = 1.0
    state.qa_holdout_pairs = []
    state.cv_mean = None
    return state


@pytest.mark.fast
@pytest.mark.process
def test_georef_cache_rejects_unreliable_profile_when_coarse_status_not_ok(tmp_path):
    scene = make_scene(entity_id="DS1104-1057DA023")
    cache_dir = tmp_path / "cache"
    paths_mod.ensure_pipeline_dirs(str(tmp_path / "out"), str(cache_dir))

    reference = write_test_raster(
        tmp_path / "reference.tif",
        bounds=(50.4, 25.7, 50.7, 26.0),
        crs="EPSG:4326",
        width=64,
        height=64,
    )
    georef_path = Path(paths_mod.georef_path(str(cache_dir), scene.entity_id))
    write_test_raster(
        georef_path,
        bounds=(50.4, 25.7, 50.7, 26.0),
        crs="EPSG:4326",
        width=64,
        height=64,
    )

    cache_key = process._build_georef_cache_key(
        scene, str(reference), str(georef_path))
    metadata = process._default_scene_metadata(scene, str(cache_dir))
    metadata.update({
        "georef_path": str(georef_path.resolve()),
        "primary_input_kind": "georef",
        "primary_input_path": str(georef_path.resolve()),
        "coarse_align_status": "abstained",
        "georef_cache_key": cache_key,
    })
    process._save_scene_metadata(str(cache_dir), scene.entity_id, metadata)

    ok = process._reuse_georef_cache(
        scene,
        str(cache_dir),
        str(reference),
        {},
        str(georef_path),
        {"completed": {}, "failed": {}},
        scene.entity_id,
    )

    assert ok is False
    assert not georef_path.exists()
    refreshed = process._load_scene_metadata(str(cache_dir), scene.entity_id)
    assert refreshed["georef_cache_reuse_status"] == "rejected"
    assert refreshed["georef_cache_reuse_reason"].startswith(
        "coarse_align_status_not_ok")


@pytest.mark.fast
@pytest.mark.qa
def test_prewarp_gate_rejects_collinear_gcp_conditioning():
    pairs = [
        MatchPair(float(x), 10.0, float(x) + 1.0, 11.0, 0.9, f"auto:{i}")
        for i, x in enumerate(np.linspace(10.0, 90.0, 8))
    ]
    state = _state_for_pairs(pairs)

    reasons, diagnostics = pipeline._prewarp_gate_reasons(state)

    assert "gcp_leverage_conditioning_failed" in reasons
    assert "gcp_conditioning_failed" in reasons
    assert diagnostics["rank"] < 6


@pytest.mark.fast
@pytest.mark.qa
def test_prewarp_gate_rejects_anchor_collapse():
    coords = [(10, 10), (12, 10), (14, 12), (16, 12), (18, 14), (20, 14)]
    pairs = [
        MatchPair(float(x), float(y), float(x) + 1.0, float(y) + 1.0,
                  0.95, f"anchor:{idx}")
        for idx, (x, y) in enumerate(coords)
    ]
    state = _state_for_pairs(pairs)

    reasons, diagnostics = pipeline._prewarp_gate_reasons(state)

    assert "anchor_collapse_detected" in reasons
    assert diagnostics["anchor_count"] == len(pairs)
    assert diagnostics["auto_count"] == 0


@pytest.mark.fast
@pytest.mark.process
def test_kh4b_metric_ortho_forces_pose_preflight(tmp_path, monkeypatch):
    target = write_test_raster(
        tmp_path / "target_ortho.tif",
        bounds=(0.0, 0.0, 100.0, 100.0),
        crs="EPSG:3857",
        width=32,
        height=32,
    )
    reference = write_test_raster(
        tmp_path / "reference.tif",
        bounds=(0.0, 0.0, 100.0, 100.0),
        crs="EPSG:3857",
        width=32,
        height=32,
    )
    state = AlignState(
        input_path=str(target),
        reference_path=str(reference),
        output_path=str(tmp_path / "aligned.tif"),
        work_crs=CRS.from_epsg(3857),
        overlap=BBox(0.0, 0.0, 100.0, 100.0),
        current_input=str(target),
        expected_scale=1.0,
        force_pose_preflight=True,
    )
    called = {"global": False}

    monkeypatch.setattr(pipeline, "detect_local_scales", lambda *a, **k: None)

    def fake_global_scale_rotation(state_arg, _args):
        called["global"] = True
        state_arg.scale_total_patches = 1
        return state_arg

    monkeypatch.setattr(
        pipeline,
        "_apply_global_scale_rotation_correction",
        fake_global_scale_rotation,
    )

    result = pipeline.step_scale_rotation(state, SimpleNamespace())

    assert result.needs_scale_rotation is True
    assert called["global"] is True


@pytest.mark.fast
@pytest.mark.process
def test_residual_fit_requires_scan_axis_observability():
    clustered = np.array([
        [-0.20, -0.05], [-0.12, -0.04], [-0.08, -0.02], [-0.02, -0.01],
        [0.02, 0.01], [0.08, 0.02], [0.12, 0.04], [0.20, 0.05],
    ])
    spread = np.array([
        [-0.95, -0.55], [-0.70, -0.35], [-0.45, 0.10], [-0.20, 0.35],
        [0.10, -0.45], [0.35, -0.15], [0.65, 0.25], [0.95, 0.55],
    ])

    ok_clustered, diag_clustered = _residual_fit_observable(clustered)
    ok_spread, diag_spread = _residual_fit_observable(spread)

    assert ok_clustered is False
    assert diag_clustered["occupied_scan_bins"] < diag_clustered["min_scan_bins"]
    assert ok_spread is True
    assert diag_spread["scan_span_frac"] >= diag_spread["min_scan_span_frac"]


@pytest.mark.fast
@pytest.mark.process
def test_mosaic_incoherence_writes_status_and_skips_output(tmp_path, monkeypatch):
    pytest.importorskip("osgeo.gdal")
    a = write_test_raster(
        tmp_path / "a.tif",
        bounds=(0.0, 0.0, 100.0, 100.0),
        crs="EPSG:3857",
        width=32,
        height=32,
    )
    b = write_test_raster(
        tmp_path / "b.tif",
        bounds=(0.0, -100.0, 100.0, 0.0),
        crs="EPSG:3857",
        width=32,
        height=32,
    )
    out = tmp_path / "mosaic" / "merged.tif"

    def fake_mosaic_pair(*_args, **_kwargs):
        raise mosaic.MosaicIncoherentError({
            "mosaic_status": "incoherent",
            "message": "pairwise shift exceeds post-alignment bound",
            "pair_idx": 0,
            "strip_a": "a.tif",
            "strip_b": "b.tif",
            "inliers": 12,
            "rejected_pair_shift": {
                "dx_m": 1500.0,
                "dy_m": 0.0,
                "magnitude_m": 1500.0,
            },
            "max_pair_shift_m": 1000.0,
        })

    monkeypatch.setattr(mosaic, "_mosaic_pair", fake_mosaic_pair)

    result = mosaic.build_mosaic([str(a), str(b)], str(out))

    assert result is None
    assert not out.exists()
    status_path = Path(mosaic._mosaic_status_path(str(out)))
    payload = json.loads(status_path.read_text())
    assert payload["mosaic_status"] == "incoherent"
    assert payload["rejected_pair_shift"]["magnitude_m"] == 1500.0
    assert [os.path.basename(p) for p in payload["inputs"]] == ["a.tif", "b.tif"]

    assert mosaic.build_mosaic([str(a), str(b)], str(out)) is None
