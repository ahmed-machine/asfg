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

    def fake_generate(scene_arg, cache_dir_arg, stitched_path_arg, corners_arg, reference_arg):
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
    assert progress["completed"][scene.entity_id]["stage"] == "georef"
    piecewise_case.record_case_summary(
        {
            "entity_id": scene.entity_id,
            "georef_path": metadata["georef_path"],
            "asp_ortho_path": metadata["asp_ortho_path"],
            "asp_camera_path": metadata["asp_camera_path"],
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

    def fake_crop(input_path, reference, cropped_path, target_bbox_wgs):
        crop_calls.append((input_path, cropped_path, target_bbox_wgs))
        Path(cropped_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cropped_path).write_text("fresh-crop", encoding="utf-8")
        return cropped_path

    monkeypatch.setattr(process, "ModelCache", DummyModelCache)
    monkeypatch.setattr(process, "get_torch_device", lambda device: device)
    monkeypatch.setattr(process, "coarse_align_and_crop", fake_crop)
    monkeypatch.setattr(process, "generate_auto_anchors", lambda *args, **kwargs: None)

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


@pytest.mark.fast
@pytest.mark.process
def test_ensure_scene_asp_ortho_keeps_bundle_adjust_output(tmp_path, monkeypatch, piecewise_case):
    cache_dir = tmp_path / "cache"
    process.ensure_pipeline_dirs(str(tmp_path / "run"), str(cache_dir))
    scene = make_scene()
    georef_path = write_test_raster(cache_dir / "georef" / f"{scene.entity_id}_georef.tif")
    ba_ortho = write_test_raster(cache_dir / "ortho" / f"{scene.entity_id}_ortho_ba.tif", crs="EPSG:3857")
    metadata = {
        "entity_id": scene.entity_id,
        "camera_designation": "A",
        "profile": "kh4",
        "gcp_corners": {k: list(v) for k, v in scene.corners.items()},
        "georef_path": str(georef_path.resolve()),
        "stitched_path": str((cache_dir / "stitched" / f"{scene.entity_id}_stitched.tif").resolve()),
        "asp_ortho_path": str(ba_ortho.resolve()),
    }

    called = {"value": False}
    monkeypatch.setattr(process, "_maybe_generate_asp_ortho", lambda *args, **kwargs: called.__setitem__("value", True))

    resolved = process._ensure_scene_asp_ortho(
        scene,
        str(cache_dir),
        str(tmp_path / "missing-reference.tif"),
        metadata=metadata,
    )

    assert resolved["asp_ortho_path"] == str(ba_ortho.resolve())
    assert called["value"] is False
    piecewise_case.record_case_summary(
        {
            "entity_id": scene.entity_id,
            "authoritative_ortho": resolved["asp_ortho_path"],
            "regenerated": called["value"],
        }
    )


@pytest.mark.fast
@pytest.mark.process
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

    def fake_generate(scene_arg, cache_dir_arg, stitched_path_arg, corners_arg, reference_arg):
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

    def fake_crop(input_path, reference, cropped_path, target_bbox_wgs):
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
