from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import align.flow_refine as flow_refine
import align.params as params
import align.pipeline as pipeline
import align.scale as scale
import align.warp as warp
import paths as paths_mod
import process
from align.errors import AlreadyAlignedError
from align.manifest import _job_to_namespace, _qa_accepts_output
from align.types import AlignmentJob
from rasterio.crs import CRS

from .helpers import make_scene, write_test_raster


@pytest.mark.fast
@pytest.mark.manifest
def test_manifest_namespace_preserves_profile_option():
    job = AlignmentJob(
        input_path="input.tif",
        reference_path="reference.tif",
        options={"profile": "kh4b"},
    )

    args = _job_to_namespace(job, {"device": "cpu"})

    assert args.profile == "kh4b"


@pytest.mark.fast
@pytest.mark.manifest
def test_manifest_arap_default_uses_profile():
    job = AlignmentJob(
        input_path="input.tif",
        reference_path="reference.tif",
        options={},
    )

    args = _job_to_namespace(job, {})

    assert args.arap_weight is None


@pytest.mark.fast
@pytest.mark.qa
def test_dense_roma_cpu_guard_skips_large_overlap(monkeypatch):
    state = SimpleNamespace(
        matcher_dense="roma",
        model_cache=SimpleNamespace(device="cpu"),
    )
    ws = SimpleNamespace(arr_ref_neural=np.zeros((1000, 1000), dtype=np.uint8))
    monkeypatch.setenv("DECLASS_CPU_ROMA_MAX_PIXELS", "999999")

    assert pipeline._should_skip_dense_roma_on_cpu(state, ws) is True

    monkeypatch.setenv("DECLASS_FORCE_CPU_ROMA", "1")
    assert pipeline._should_skip_dense_roma_on_cpu(state, ws) is False


@pytest.mark.fast
@pytest.mark.qa
def test_cap_grid_axes_keeps_spatial_stride_under_budget():
    rows = list(range(100))
    cols = list(range(50))

    capped_rows, capped_cols, stride = pipeline._cap_grid_axes(rows, cols, 1000)

    assert stride > 1
    assert len(capped_rows) * len(capped_cols) <= 1000
    assert capped_rows[0] == rows[0]
    assert capped_cols[0] == cols[0]


@pytest.mark.fast
@pytest.mark.qa
def test_process_pool_guard_retries_serial(monkeypatch):
    class FailingPool:
        def __init__(self, *args, **kwargs):
            raise PermissionError("fork blocked")

    def double(value):
        return value * 2

    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", FailingPool)

    result = pipeline._map_with_process_pool_or_serial(
        double, [1, 2, 3], n_workers=2, chunksize=1, label="test")

    assert result == [2, 4, 6]


@pytest.mark.fast
@pytest.mark.selection
def test_failed_coarse_zero_offset_continues_to_matching(monkeypatch):
    state = SimpleNamespace(
        coarse_total=0.0,
        expected_scale=1.0,
        was_corrected=False,
        coarse_status="failed",
    )
    monkeypatch.setattr(
        pipeline, "_already_aligned_passthrough_allowed", lambda _state: False)

    assert pipeline.step_handle_large_offset(state, SimpleNamespace()) is state


@pytest.mark.fast
@pytest.mark.selection
def test_successful_coarse_zero_offset_can_passthrough():
    state = SimpleNamespace(
        coarse_total=0.0,
        expected_scale=1.0,
        was_corrected=False,
        coarse_status="ok",
    )

    with pytest.raises(AlreadyAlignedError):
        pipeline.step_handle_large_offset(state, SimpleNamespace())


@pytest.mark.fast
@pytest.mark.selection
def test_metric_ortho_input_expected_scale_ignores_pixel_ratio():
    dataset = SimpleNamespace(crs=CRS.from_epsg(3857))

    assert pipeline._is_preorthorectified_metric_input(
        "/tmp/ortho/SCENE_ortho.coarse.tif", [], dataset)


@pytest.mark.fast
@pytest.mark.manifest
def test_manifest_progressive_anchor_requires_accepted_qa(tmp_path):
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(
        '{"selected_candidate":"affine","reports":[{"candidate":"affine","accepted":false}]}',
        encoding="utf-8",
    )

    assert _qa_accepts_output(str(qa_path)) is False


@pytest.mark.fast
@pytest.mark.qa
def test_profile_loader_rejects_unknown_grid_key():
    with pytest.raises(ValueError, match="grid_optim"):
        params._dict_to_params({"grid_optim": {"not_a_real_weight": 1.0}})


@pytest.mark.fast
@pytest.mark.qa
def test_runtime_override_rejects_unknown_key():
    with pytest.raises(KeyError, match="grid_optim__not_a_real_weight"):
        with params.override(grid_optim__not_a_real_weight=1.0):
            pass


@pytest.mark.fast
@pytest.mark.process
def test_georef_cache_reuse_rejects_non_ok_coarse_status(tmp_path):
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
    metadata = process._default_scene_metadata(scene, str(cache_dir))
    metadata.update({
        "georef_path": str(georef_path.resolve()),
        "primary_input_kind": "georef",
        "primary_input_path": str(georef_path.resolve()),
        "coarse_align_status": "abstained",
        "georef_cache_key": process._build_georef_cache_key(
            scene, str(reference), str(georef_path)),
    })
    process._save_scene_metadata(str(cache_dir), scene.entity_id, metadata)

    accepted = process._reuse_georef_cache(
        scene,
        str(cache_dir),
        str(reference),
        {},
        str(georef_path),
        {"completed": {}, "failed": {}},
        scene.entity_id,
    )

    assert accepted is False
    assert not georef_path.exists()
    refreshed = process._load_scene_metadata(str(cache_dir), scene.entity_id)
    assert refreshed["georef_cache_reuse_status"] == "rejected"
    assert refreshed["georef_cache_reuse_reason"].startswith(
        "coarse_align_status_not_ok")


@pytest.mark.fast
@pytest.mark.selection
def test_forward_backward_mask_can_skip_uncertainty(monkeypatch):
    ref = np.zeros((8, 8), dtype=np.uint8)
    warped = np.zeros((8, 8), dtype=np.uint8)
    flow = np.zeros((8, 8, 2), dtype=np.float32)

    def fail_uncertainty(*_args, **_kwargs):
        raise AssertionError("SEA-RAFT uncertainty should not be loaded")

    monkeypatch.setattr(flow_refine, "_estimate_uncertainty_raft", fail_uncertainty)
    monkeypatch.setattr(
        flow_refine,
        "_forward_backward_error",
        lambda *_args, **_kwargs: (flow, flow, np.zeros((8, 8), dtype=np.float32)),
    )

    mask = flow_refine._forward_backward_mask(
        ref, warped, flow, threshold_px=3.0, use_uncertainty=False)

    assert mask.all()


@pytest.mark.fast
@pytest.mark.selection
def test_grid_warp_uses_active_profile_params(monkeypatch):
    captured = {}
    coast = np.zeros((0, 2), dtype=np.float32)
    rgb = np.zeros((8, 8, 3), dtype=np.float32)
    u8 = np.zeros((8, 8), dtype=np.uint8)

    monkeypatch.setattr(
        warp, "_load_target_features",
        lambda *_args, **_kwargs: (coast, rgb, u8, 1.0),
    )
    monkeypatch.setattr(
        warp, "_load_source_features",
        lambda *_args, **_kwargs: (u8.astype(np.float32), 8, 8, coast, rgb, u8, 1.0),
    )
    monkeypatch.setattr(
        warp, "_prepare_gcps",
        lambda *_args, **_kwargs: (
            np.array([[1.0, 1.0]], dtype=np.float32),
            np.array([[1.0, 1.0]], dtype=np.float32),
            1,
            None,
        ),
    )
    monkeypatch.setattr(warp, "_compute_reclamation_mask", lambda *_args, **_kwargs: None)

    def fake_optimize_grid_hierarchical(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(displacements=None)

    monkeypatch.setattr(warp, "optimize_grid_hierarchical", fake_optimize_grid_hierarchical)

    with params.override(
        grid_optim__pyramid_levels=[[3, 4]],
        grid_optim__lr=0.123,
        grid_optim__w_data=2.0,
        grid_optim__w_chamfer=0.4,
        grid_optim__w_arap=0.7,
        grid_optim__w_laplacian=0.8,
        grid_optim__w_disp=0.9,
        grid_optim__max_residual_norm=0.01,
    ):
        warp.apply_warp_grid_only(
            "input.tif",
            "reference.tif",
            [],
            work_crs="EPSG:3857",
            output_bounds=(0.0, 0.0, 8.0, 8.0),
            output_res=1.0,
        )

    assert captured["levels"] == [(3, 4)]
    assert captured["lr"] == 0.123
    assert captured["w_data"] == 2.0
    assert captured["w_chamfer"] == 0.4
    assert captured["w_arap"] == 0.7
    assert captured["w_laplacian"] == 0.8
    assert captured["w_disp"] == 0.9
    assert captured["max_residual_norm"] == 0.01
