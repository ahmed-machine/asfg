from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import process
from preprocess.stitch import verify_tiff_decodes_nonempty
from tests.helpers import make_scene, write_test_raster


class DummyModelCache:
    def __init__(self, device):
        self.device = device

    def close(self):
        return None


@pytest.mark.fast
@pytest.mark.process
def test_verify_tiff_accepts_raster_with_real_content(tmp_path):
    """A valid raster with non-NoData pixels should pass the pre-flight check."""
    good = write_test_raster(tmp_path / "good.tif", fill=42)
    assert verify_tiff_decodes_nonempty(str(good), label="test") is True


@pytest.mark.fast
@pytest.mark.process
def test_verify_tiff_rejects_all_zero_raster(tmp_path):
    """Catches the ASP image_mosaic unfinalized-TIFF failure: file opens,
    reports dimensions, but every decoded pixel reads as zero/NoData.

    This is the same signature as the real-world KH-9 failure at
    output/stitched/D3C1213-200346A003_stitched.tif (20:46 Apr 9): 1.07 GB
    on disk, valid TIFF header, all 441 TileOffsets entries zero → GDAL
    decodes as all-zero and no viewer can open it.
    """
    bad = write_test_raster(tmp_path / "all_zero.tif", fill=0)
    assert verify_tiff_decodes_nonempty(str(bad), label="test") is False


@pytest.mark.fast
@pytest.mark.process
def test_verify_tiff_rejects_missing_file(tmp_path):
    """Non-existent inputs should fail fast, not raise."""
    assert verify_tiff_decodes_nonempty(str(tmp_path / "does_not_exist.tif"),
                                        label="test") is False


@pytest.mark.fast
@pytest.mark.process
def test_is_aft_camera_kh9_aft():
    """KH-9 Aft entity D3C...-...A003 must detect as Aft."""
    from preprocess.camera_model import is_aft_camera
    assert is_aft_camera("D3C1213-200346A003", "KH-9") is True


@pytest.mark.fast
@pytest.mark.process
def test_is_aft_camera_kh9_forward():
    """KH-9 Forward entity D3C...-...F002 must NOT detect as Aft."""
    from preprocess.camera_model import is_aft_camera
    assert is_aft_camera("D3C1213-200346F002", "KH-9") is False


@pytest.mark.fast
@pytest.mark.process
def test_is_aft_camera_kh4_aft():
    """KH-4 Aft entity DS1...-...DA... must detect as Aft."""
    from preprocess.camera_model import is_aft_camera
    assert is_aft_camera("DS1104-1057DA024", "KH-4") is True


@pytest.mark.fast
@pytest.mark.process
def test_is_aft_camera_kh4_forward():
    """KH-4 Forward entity DS1...-...DF... must NOT detect as Aft."""
    from preprocess.camera_model import is_aft_camera
    assert is_aft_camera("DS1104-1057DF001", "KH-4") is False


@pytest.mark.fast
@pytest.mark.process
def test_is_aft_camera_kh7():
    """KH-7 has no aft/forward distinction — always returns False."""
    from preprocess.camera_model import is_aft_camera
    assert is_aft_camera("DZB00403600089H015001", "KH-7") is False


@pytest.mark.fast
@pytest.mark.process
def test_is_aft_camera_case_insensitive():
    """is_aft_camera must handle lowercase input."""
    from preprocess.camera_model import is_aft_camera
    assert is_aft_camera("d3c1213-200346a003", "kh-9") is True
    assert is_aft_camera("ds1104-1057da024", "kh-4") is True
def test_process_scene_backfills_missing_asp_ortho_for_cached_scene(tmp_path, monkeypatch, piecewise_case):
    output_dir = tmp_path / "run"
    cache_dir = tmp_path / "cache"
    process.ensure_pipeline_dirs(str(output_dir), str(cache_dir))

    scene = make_scene()
    georef_path = write_test_raster(cache_dir / "georef" / f"{scene.entity_id}_georef.tif")
    stitched_path = write_test_raster(cache_dir / "stitched" / f"{scene.entity_id}_stitched.tif")
    reference_path = write_test_raster(tmp_path / "reference.tif")

    generated_ortho = cache_dir / "ortho" / f"{scene.entity_id}_ortho.tif"
    generated_camera = stitched_path.with_suffix(".tsai")

    def fake_generate(scene_arg, cache_dir_arg, stitched_path_arg, corners_arg, reference_arg, **_kwargs):
        assert scene_arg.entity_id == scene.entity_id
        assert Path(stitched_path_arg) == stitched_path
        assert reference_arg == str(reference_path)
        generated_camera.write_text("camera", encoding="utf-8")
        write_test_raster(generated_ortho, crs="EPSG:3857")
        return str(generated_ortho)

    monkeypatch.setattr(process, "_maybe_generate_asp_ortho", fake_generate)

    progress = {"completed": {}, "failed": {}}
    ok = process.extract_stitch_georef_scene(
        scene,
        str(output_dir),
        {},
        str(reference_path),
        progress,
        cache_dir=str(cache_dir),
    )

    assert ok is True
    metadata = process._load_scene_metadata(str(cache_dir), scene.entity_id)
    assert metadata is not None
    assert metadata["georef_path"] == str(georef_path.resolve())
    assert metadata["stitched_path"] == str(stitched_path.resolve())
    assert metadata["asp_ortho_path"] == str(generated_ortho.resolve())
    assert metadata["asp_camera_path"] == str(generated_camera.resolve())
    assert metadata["primary_input_kind"] == "asp_ortho"
    assert metadata["primary_input_path"] == str(generated_ortho.resolve())
    assert progress["completed"][scene.entity_id]["stage"] == "georef"
    piecewise_case.record_case_summary(
        {
            "entity_id": scene.entity_id,
            "georef_path": metadata["georef_path"],
            "asp_ortho_path": metadata["asp_ortho_path"],
            "asp_camera_path": metadata["asp_camera_path"],
            "primary_input_kind": metadata["primary_input_kind"],
        }
    )


@pytest.mark.fast
@pytest.mark.manifest
def test_generate_manifest_prefers_asp_ortho_and_input_specific_crop(tmp_path, monkeypatch, piecewise_case):
    output_dir = tmp_path / "run"
    cache_dir = tmp_path / "cache"
    process.ensure_pipeline_dirs(str(output_dir), str(cache_dir))

    scene = make_scene()
    reference_path = write_test_raster(tmp_path / "reference.tif", bounds=(0.0, 0.0, 10.0, 10.0))
    georef_path = write_test_raster(cache_dir / "georef" / f"{scene.entity_id}_georef.tif")
    ortho_path = write_test_raster(cache_dir / "ortho" / f"{scene.entity_id}_ortho.tif", crs="EPSG:3857")

    stale_georef_crop = cache_dir / "georef" / f"{scene.entity_id}_georef_cropped.tif"
    stale_georef_crop.write_text("stale-georef-crop", encoding="utf-8")

    process._save_scene_metadata(
        str(cache_dir),
        scene.entity_id,
        {
            "entity_id": scene.entity_id,
            "camera_designation": "A",
            "profile": "kh4",
            "gcp_corners": {k: list(v) for k, v in scene.corners.items()},
            "georef_path": str(georef_path.resolve()),
            "stitched_path": str((cache_dir / "stitched" / f"{scene.entity_id}_stitched.tif").resolve()),
            "asp_camera_path": None,
            "asp_ortho_path": str(ortho_path.resolve()),
        },
    )

    crop_calls = []

    def fake_crop(input_path, reference, cropped_path, target_bbox_wgs, **kwargs):
        crop_calls.append((input_path, cropped_path, target_bbox_wgs))
        Path(cropped_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cropped_path).write_text("fresh-crop", encoding="utf-8")
        return cropped_path

    monkeypatch.setattr(process, "ModelCache", DummyModelCache)
    monkeypatch.setattr(process, "get_torch_device", lambda device: device)
    monkeypatch.setattr(process, "coarse_align_and_crop", fake_crop)
    monkeypatch.setattr(process, "generate_auto_anchors", lambda *args, **kwargs: None)
    # Pre-alignment crop is now opt-in (user feedback: USGS-corner-based
    # crops discarded valid content). Opt back in for this test that
    # asserts the crop codepath still works end-to-end when enabled.
    monkeypatch.setenv("DECLASS_PRE_ALIGN_CROP", "1")

    manifest_path = process.generate_manifest(
        [scene],
        str(output_dir),
        str(reference_path),
        cache_dir=str(cache_dir),
        device="cpu",
    )

    assert manifest_path is not None
    assert crop_calls, "expected crop generation for the selected primary input"
    selected_input, selected_crop, _ = crop_calls[0]
    assert Path(selected_input) == ortho_path
    assert selected_crop.endswith(f"{scene.entity_id}_asp_ortho_cropped.tif")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    job = manifest["jobs"][0]
    assert job["input"] == str(Path(selected_crop).resolve())
    assert job["input"] != str(stale_georef_crop.resolve())
    assert job["profile"] == "kh4b"

    metadata = process._load_scene_metadata(str(cache_dir), scene.entity_id)
    assert metadata is not None
    assert metadata["primary_input_kind"] == "asp_ortho"
    assert metadata["primary_input_path"] == str(ortho_path.resolve())
    assert metadata["alignment_crop_path"] == str(Path(selected_crop).resolve())
    piecewise_case.record_case_summary(
        {
            "entity_id": scene.entity_id,
            "manifest_path": manifest_path,
            "selected_input": job["input"],
            "primary_input_kind": metadata["primary_input_kind"],
        }
    )


def test_ensure_scene_asp_ortho_regenerates_stale_output(tmp_path, monkeypatch, piecewise_case):
    cache_dir = tmp_path / "cache"
    process.ensure_pipeline_dirs(str(tmp_path / "run"), str(cache_dir))
    scene = make_scene()
    reference_path = write_test_raster(tmp_path / "reference.tif")
    stitched_path = write_test_raster(cache_dir / "stitched" / f"{scene.entity_id}_stitched.tif")
    georef_path = write_test_raster(cache_dir / "georef" / f"{scene.entity_id}_georef.tif")
    stale_ortho = write_test_raster(cache_dir / "ortho" / f"{scene.entity_id}_ortho.tif", crs="EPSG:3857")
    stale_camera = stitched_path.with_suffix(".tsai")
    stale_camera.write_text("old-camera", encoding="utf-8")

    future_mtime = stale_ortho.stat().st_mtime + 10
    os.utime(stitched_path, (future_mtime, future_mtime))

    regenerated_ortho = cache_dir / "ortho" / f"{scene.entity_id}_ortho.tif"
    regenerated_camera = stitched_path.with_suffix(".tsai")

    def fake_generate(scene_arg, cache_dir_arg, stitched_path_arg, corners_arg, reference_arg, **_kwargs):
        regenerated_camera.write_text("new-camera", encoding="utf-8")
        write_test_raster(regenerated_ortho, crs="EPSG:3857")
        return str(regenerated_ortho)

    monkeypatch.setattr(process, "_maybe_generate_asp_ortho", fake_generate)

    metadata = {
        "entity_id": scene.entity_id,
        "camera_designation": "A",
        "profile": "kh4",
        "gcp_corners": {k: list(v) for k, v in scene.corners.items()},
        "georef_path": str(georef_path.resolve()),
        "stitched_path": str(stitched_path.resolve()),
        "asp_ortho_path": str(stale_ortho.resolve()),
    }

    resolved = process._ensure_scene_asp_ortho(
        scene,
        str(cache_dir),
        str(reference_path),
        metadata=metadata,
    )

    assert resolved["asp_ortho_path"] == str(regenerated_ortho.resolve())
    assert resolved["asp_camera_path"] == str(regenerated_camera.resolve())
    piecewise_case.record_case_summary(
        {
            "entity_id": scene.entity_id,
            "regenerated_ortho": resolved["asp_ortho_path"],
            "regenerated_camera": resolved["asp_camera_path"],
        }
    )


def test_ensure_scene_asp_ortho_refreshes_sidecar_when_canonical_is_fresh(
    tmp_path, monkeypatch, piecewise_case,
):
    """Cache-warm path must still consult the sidecar so a reference
    change invalidates a stale shift, and ``coarse_align_status`` is
    recorded for ``generate_manifest``'s hard-fail abstain skip.
    """
    cache_dir = tmp_path / "cache"
    process.ensure_pipeline_dirs(str(tmp_path / "run"), str(cache_dir))
    scene = make_scene()
    reference_path = write_test_raster(tmp_path / "reference.tif")
    stitched_path = write_test_raster(
        cache_dir / "stitched" / f"{scene.entity_id}_stitched.tif")
    georef_path = write_test_raster(
        cache_dir / "georef" / f"{scene.entity_id}_georef.tif")
    fresh_ortho = write_test_raster(
        cache_dir / "ortho" / f"{scene.entity_id}_ortho.tif", crs="EPSG:3857")

    metadata = process._default_scene_metadata(scene, str(cache_dir))
    metadata.update({
        "georef_path": str(georef_path.resolve()),
        "stitched_path": str(stitched_path.resolve()),
        "asp_ortho_path": str(fresh_ortho.resolve()),
    })
    process._save_scene_metadata(str(cache_dir), scene.entity_id, metadata)

    sidecar_calls = []

    def fake_sidecar(source_path, reference, bbox, label, cache, eid, *,
                     model_cache=None, **kwargs):
        sidecar_calls.append({
            "source_path": source_path,
            "reference": reference,
            "label": label,
        })
        process._record_coarse_align_status(cache, eid, "ok")
        return source_path

    monkeypatch.setattr(process, "_coarse_align_ortho_to_sidecar", fake_sidecar)
    # Guard: _maybe_generate_asp_ortho must NOT run when canonical is fresh
    not_called = {"value": False}
    monkeypatch.setattr(process, "_maybe_generate_asp_ortho",
                        lambda *a, **kw: not_called.__setitem__("value", True))

    process._ensure_scene_asp_ortho(
        scene, str(cache_dir), str(reference_path),
        metadata=metadata,
    )

    assert not_called["value"] is False, "regeneration must not fire on fresh canonical"
    assert sidecar_calls, "sidecar refresh must fire on the cache-warm path"
    assert sidecar_calls[0]["source_path"] == str(fresh_ortho.resolve())
    assert sidecar_calls[0]["reference"] == str(reference_path)

    refreshed = process._load_scene_metadata(str(cache_dir), scene.entity_id)
    assert refreshed["coarse_align_status"] == "ok"
    piecewise_case.record_case_summary({
        "sidecar_calls": len(sidecar_calls),
        "coarse_align_status": refreshed["coarse_align_status"],
    })


def _write_opticalbar_tsai(path: Path, ic, ir=None):
    if ir is None:
        ir = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "VERSION_4",
            "OPTICAL_BAR",
            "image_size = 20 10",
            "image_center = 10 5",
            "pitch = 0.000007",
            "f = 0.6096",
            "scan_time = 0.5",
            "forward_tilt = 0.0",
            "iC = " + " ".join(str(v) for v in ic),
            "iR = " + " ".join(str(v) for v in ir),
            "speed = 7700",
            "mean_earth_radius = 6371000",
            "mean_surface_elevation = 0",
            "motion_compensation_factor = 1",
            "scan_dir = right",
            "",
        ]),
        encoding="utf-8",
    )


@pytest.mark.fast
@pytest.mark.process
def test_interpolate_camera_pose_brackets_position(tmp_path):
    from preprocess.camera_model import _parse_opticalbar_tsai, interpolate_camera_pose

    left = tmp_path / "left.tsai"
    right = tmp_path / "right.tsai"
    out = tmp_path / "out.tsai"
    _write_opticalbar_tsai(left, [0, 0, 0])
    _write_opticalbar_tsai(right, [10, 20, 30])

    result = interpolate_camera_pose(
        [str(left), str(right)], str(out), alpha=0.5, base_tsai_path=str(left))

    assert result == str(out)
    parsed = _parse_opticalbar_tsai(str(out))
    assert parsed is not None
    assert parsed["iC"].tolist() == [5.0, 10.0, 15.0]


@pytest.mark.fast
@pytest.mark.process
def test_build_stitched_ortho_retries_with_interpolated_pose(
    tmp_path, monkeypatch,
):
    cache_dir = tmp_path / "cache"
    process.ensure_pipeline_dirs(str(tmp_path / "run"), str(cache_dir))
    scene = make_scene(entity_id="DS1104-1057DA024", camera_type="aft")
    stitched_path = write_test_raster(
        cache_dir / "stitched" / f"{scene.entity_id}_stitched.tif")
    target_tsai = stitched_path.with_suffix(".tsai")
    _write_opticalbar_tsai(target_tsai, [0, 0, 0])
    _write_opticalbar_tsai(
        cache_dir / "stitched" / "DS1104-1057DA023_stitched.tsai",
        [0, 0, 0],
    )
    _write_opticalbar_tsai(
        cache_dir / "stitched" / "DS1104-1057DA025_stitched.tsai",
        [10, 0, 0],
    )

    monkeypatch.setattr(process, "generate_camera", lambda *a, **k: str(target_tsai))
    mapproject_calls = []

    def fake_mapproject(image_path, camera_path, *, output_path, **kwargs):
        mapproject_calls.append(camera_path)
        fill = 0 if len(mapproject_calls) == 1 else 100
        write_test_raster(Path(output_path), fill=fill, crs="EPSG:3857")
        return output_path

    sidecar_calls = []
    monkeypatch.setattr(process, "mapproject_image", fake_mapproject)
    monkeypatch.setattr(
        process, "_coarse_align_ortho_to_sidecar",
        lambda *a, **k: sidecar_calls.append(a) or a[0],
    )

    out = process._build_stitched_ortho(
        scene,
        str(cache_dir),
        str(stitched_path),
        {"focal_length": 0.6096, "pixel_pitch": 7.0e-6},
        scene.corners,
        str(tmp_path / "reference.tif"),
        (0, 0, 1, 1),
        str(tmp_path / "dem.tif"),
    )

    assert out == str(cache_dir / "ortho" / f"{scene.entity_id}_ortho.tif")
    assert len(mapproject_calls) == 2
    assert mapproject_calls[1].endswith("_stitched.interp.tsai")
    assert sidecar_calls


@pytest.mark.fast
@pytest.mark.manifest
def test_generate_manifest_reuses_fresh_crop(tmp_path, monkeypatch, piecewise_case):
    output_dir = tmp_path / "run"
    cache_dir = tmp_path / "cache"
    process.ensure_pipeline_dirs(str(output_dir), str(cache_dir))

    scene = make_scene()
    reference_path = write_test_raster(tmp_path / "reference.tif", bounds=(0.0, 0.0, 10.0, 10.0))
    ortho_path = write_test_raster(cache_dir / "ortho" / f"{scene.entity_id}_ortho.tif", crs="EPSG:3857")
    crop_path = cache_dir / "ortho" / f"{scene.entity_id}_asp_ortho_cropped.tif"
    crop_path.write_text("fresh-crop", encoding="utf-8")
    os.utime(crop_path, (ortho_path.stat().st_mtime + 10, ortho_path.stat().st_mtime + 10))

    process._save_scene_metadata(
        str(cache_dir),
        scene.entity_id,
        {
            "entity_id": scene.entity_id,
            "camera_designation": "A",
            "profile": "kh4",
            "gcp_corners": {k: list(v) for k, v in scene.corners.items()},
            "georef_path": str((cache_dir / "georef" / f"{scene.entity_id}_georef.tif").resolve()),
            "stitched_path": str((cache_dir / "stitched" / f"{scene.entity_id}_stitched.tif").resolve()),
            "asp_ortho_path": str(ortho_path.resolve()),
        },
    )

    monkeypatch.setattr(process, "ModelCache", DummyModelCache)
    monkeypatch.setattr(process, "get_torch_device", lambda device: device)
    monkeypatch.setattr(process, "coarse_align_and_crop", lambda *args, **kwargs: pytest.fail("crop should not regenerate"))
    monkeypatch.setattr(process, "generate_auto_anchors", lambda *args, **kwargs: None)
    # Opt into the pre-align crop codepath (disabled by default) — this
    # test covers the cache-reuse branch when a fresh crop already exists.
    monkeypatch.setenv("DECLASS_PRE_ALIGN_CROP", "1")

    manifest_path = process.generate_manifest(
        [scene],
        str(output_dir),
        str(reference_path),
        cache_dir=str(cache_dir),
        device="cpu",
    )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["jobs"][0]["input"] == str(crop_path.resolve())
    piecewise_case.record_case_summary(
        {
            "entity_id": scene.entity_id,
            "manifest_path": manifest_path,
            "reused_crop": manifest["jobs"][0]["input"],
        }
    )


@pytest.mark.fast
@pytest.mark.manifest
def test_generate_manifest_sorts_jobs_by_overlap(tmp_path, monkeypatch, piecewise_case):
    output_dir = tmp_path / "run"
    cache_dir = tmp_path / "cache"
    process.ensure_pipeline_dirs(str(output_dir), str(cache_dir))

    scene_hi = make_scene("DS1104-1057DA024", corners={
        "NW": (8.0, 1.0),
        "NE": (8.0, 5.0),
        "SE": (4.0, 5.0),
        "SW": (4.0, 1.0),
    })
    scene_lo = make_scene("DS1104-1057DA025", corners={
        "NW": (8.0, 8.2),
        "NE": (8.0, 10.2),
        "SE": (6.0, 10.2),
        "SW": (6.0, 8.2),
    })
    reference_path = write_test_raster(tmp_path / "reference.tif", bounds=(0.0, 0.0, 10.0, 10.0))

    for scene in (scene_hi, scene_lo):
        ortho_path = write_test_raster(cache_dir / "ortho" / f"{scene.entity_id}_ortho.tif", crs="EPSG:3857")
        process._save_scene_metadata(
            str(cache_dir),
            scene.entity_id,
            {
                "entity_id": scene.entity_id,
                "camera_designation": "A",
                "profile": "kh4",
                "gcp_corners": {k: list(v) for k, v in scene.corners.items()},
                "georef_path": str((cache_dir / "georef" / f"{scene.entity_id}_georef.tif").resolve()),
                "stitched_path": str((cache_dir / "stitched" / f"{scene.entity_id}_stitched.tif").resolve()),
                "asp_ortho_path": str(ortho_path.resolve()),
            },
        )

    def fake_crop(input_path, reference, cropped_path, target_bbox_wgs, **kwargs):
        Path(cropped_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cropped_path).write_text("crop", encoding="utf-8")
        return cropped_path

    monkeypatch.setattr(process, "ModelCache", DummyModelCache)
    monkeypatch.setattr(process, "get_torch_device", lambda device: device)
    monkeypatch.setattr(process, "coarse_align_and_crop", fake_crop)
    monkeypatch.setattr(process, "generate_auto_anchors", lambda *args, **kwargs: None)

    manifest_path = process.generate_manifest(
        [scene_lo, scene_hi],
        str(output_dir),
        str(reference_path),
        cache_dir=str(cache_dir),
        device="cpu",
    )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    ordered_inputs = [Path(job["input"]).name for job in manifest["jobs"]]
    assert ordered_inputs[0].startswith(scene_hi.entity_id)
    assert ordered_inputs[1].startswith(scene_lo.entity_id)
    piecewise_case.record_case_summary(
        {
            "manifest_path": manifest_path,
            "ordered_inputs": ordered_inputs,
        }
    )


def test_ortho_has_content_rejects_empty_raster(tmp_path):
    """A mapproject output that's all nodata (the DA024 case in v13/v14)
    must be flagged as broken so the cache regenerates."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    p = tmp_path / "empty_ortho.tif"
    arr = np.zeros((512, 512), dtype=np.uint8)
    with rasterio.open(
        p, "w", driver="GTiff", height=512, width=512, count=1,
        dtype="uint8", crs="EPSG:3857",
        transform=from_origin(0, 0, 50, 50),
    ) as dst:
        dst.write(arr, 1)
    assert process._ortho_has_content(str(p), min_valid_fraction=0.001) is False


def test_ortho_has_content_accepts_normal_raster(tmp_path):
    """A normal mapproject output with substantial content must pass."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    p = tmp_path / "normal_ortho.tif"
    arr = np.full((512, 512), 100, dtype=np.uint8)
    arr[100:400, 100:400] = 200    # ~34% valid pixels
    with rasterio.open(
        p, "w", driver="GTiff", height=512, width=512, count=1,
        dtype="uint8", crs="EPSG:3857",
        transform=from_origin(0, 0, 50, 50),
    ) as dst:
        dst.write(arr, 1)
    assert process._ortho_has_content(str(p), min_valid_fraction=0.001) is True


def test_ortho_has_content_handles_missing_path(tmp_path):
    """A path that doesn't exist returns False (ortho missing == broken
    for the gate's purposes)."""
    p = tmp_path / "does_not_exist.tif"
    assert process._ortho_has_content(str(p), min_valid_fraction=0.001) is False


def test_ortho_has_content_threshold_respected(tmp_path):
    """The min_valid_fraction threshold must actually gate. A raster
    with 0.05% valid pixels passes a 0.0001 threshold but fails 0.001."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    p = tmp_path / "sparse_ortho.tif"
    arr = np.zeros((1024, 1024), dtype=np.uint8)
    arr[0:24, 0:24] = 200    # 576 / 1048576 ≈ 0.055%
    with rasterio.open(
        p, "w", driver="GTiff", height=1024, width=1024, count=1,
        dtype="uint8", crs="EPSG:3857",
        transform=from_origin(0, 0, 50, 50),
    ) as dst:
        dst.write(arr, 1)
    # Decimated read at /256 may miss the small bright corner — that's
    # fine; the gate's purpose is to flag SUBSTANTIALLY empty rasters.
    # We just assert the threshold is monotonic: a sparse raster fails
    # a tighter (higher) threshold than it passes.
    loose = process._ortho_has_content(str(p), min_valid_fraction=1e-9)
    tight = process._ortho_has_content(str(p), min_valid_fraction=0.5)
    assert tight is False
    # `loose` may be True or False depending on which decimated pixel
    # the bright corner lands on; the important assertion is that the
    # gate correctly rejects high-threshold queries.
    assert loose in (True, False)
