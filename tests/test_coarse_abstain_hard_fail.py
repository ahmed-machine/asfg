"""Hard-fail when preprocess coarse-align abstained on a profile whose
USGS corners are flagged unreliable.

The tests cover the three layers of the guard:

1. The conservative dataclass default (`CameraParams.usgs_corners_reliable`).
2. The per-profile YAML values (KH-4 / KH-7 / KH-8 false, KH-9 true).
3. The end-to-end skip in :func:`process.generate_manifest`, including the
   ``DECLASS_ALLOW_UNCOARSE_ALIGN=1`` override and the no-skip path on a
   reliable profile.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import paths as paths_mod
import process
from align.params import CameraParams, load_profile

from .helpers import make_scene, write_test_raster


def test_camera_params_default_unreliable():
    """Conservative default: unknown profiles must hard-skip rather than
    silently align against an ortho positioned only by USGS corners."""
    assert CameraParams().usgs_corners_reliable is False


@pytest.mark.parametrize(
    "profile_name,expected",
    [
        ("kh4a", False),
        ("kh4b", False),
        ("kh7", False),
        ("kh8", False),
        ("kh9", True),
        ("kh9_mc", True),
    ],
)
def test_profile_usgs_corners_reliable(profile_name, expected):
    profile = load_profile(profile_name)
    assert profile.camera.usgs_corners_reliable is expected, (
        f"Profile '{profile_name}' usgs_corners_reliable should be {expected}"
    )


# ---------------------------------------------------------------------------
# generate_manifest skip behaviour
# ---------------------------------------------------------------------------

# Bahrain-ish bbox used so frame and reference overlap (otherwise the
# pre-existing <1% overlap skip pre-empts the new check).
_REF_BOUNDS_WGS = (50.4, 25.7, 50.7, 26.0)
_FRAME_CORNERS = {
    "NW": (26.0, 50.4),
    "NE": (26.0, 50.7),
    "SE": (25.7, 50.7),
    "SW": (25.7, 50.4),
}


def _kh9_scene(entity_id: str = "D3C1213-200346A003") -> SimpleNamespace:
    """SimpleNamespace scene with the D3C prefix so _profile_name_for_scene
    routes it to kh9 (usgs_corners_reliable=True)."""
    camera_system = SimpleNamespace(
        entity_prefix="D3C",
        name="KH-9",
        program="HEXAGON",
        needs_stitching=False,
    )
    return SimpleNamespace(
        entity_id=entity_id,
        camera_system=camera_system,
        camera_type="aft",
        corners=_FRAME_CORNERS,
        acquisition_date="1977-08-27",
    )


def _setup_cache(tmp_path: Path, scene, *, coarse_status: str | None) -> tuple[str, str, str]:
    """Lay out a minimal pipeline cache so generate_manifest can iterate
    one scene without hitting filesystem-missing branches.

    Writes:
    - a tiny reference TIFF in WGS84
    - a georef TIFF for the scene at the canonical georef path
    - scene metadata with `coarse_align_status` set as requested

    Returns (cache_dir, output_dir, reference_path).
    """
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    paths_mod.ensure_pipeline_dirs(str(out), str(cache))

    ref_path = tmp_path / "ref.tif"
    write_test_raster(ref_path, bounds=_REF_BOUNDS_WGS,
                      crs="EPSG:4326", width=64, height=64)

    georef = paths_mod.georef_path(str(cache), scene.entity_id)
    write_test_raster(Path(georef), bounds=_REF_BOUNDS_WGS,
                      crs="EPSG:4326", width=64, height=64)

    metadata = process._default_scene_metadata(scene, str(cache))
    metadata["primary_input_kind"] = "georef"
    metadata["primary_input_path"] = os.path.abspath(georef)
    metadata["coarse_align_status"] = coarse_status
    process._save_scene_metadata(str(cache), scene.entity_id, metadata)
    return str(cache), str(out), str(ref_path)


def test_skip_when_kh4b_abstained(tmp_path, capsys, monkeypatch):
    """KH-4B abstain → manifest excludes the entity and prints the
    [skip] line with the override hint."""
    monkeypatch.delenv("DECLASS_ALLOW_UNCOARSE_ALIGN", raising=False)
    scene = make_scene(entity_id="DS1104-1057DA023", corners=_FRAME_CORNERS)
    cache, out, ref = _setup_cache(tmp_path, scene, coarse_status="abstained")

    result = process.generate_manifest([scene], out, ref, cache_dir=cache)

    assert result is None  # no jobs → no manifest written
    captured = capsys.readouterr()
    assert "coarse-align abstained" in captured.out
    assert "DECLASS_ALLOW_UNCOARSE_ALIGN=1" in captured.out


def test_no_skip_when_kh9_abstained(tmp_path, monkeypatch):
    """KH-9 abstain → entity proceeds because the profile is flagged
    reliable. We stub generate_auto_anchors to keep the test fast."""
    monkeypatch.delenv("DECLASS_ALLOW_UNCOARSE_ALIGN", raising=False)
    monkeypatch.setattr(process, "generate_auto_anchors",
                        lambda *args, **kwargs: None)
    scene = _kh9_scene()
    cache, out, ref = _setup_cache(tmp_path, scene, coarse_status="abstained")

    result = process.generate_manifest([scene], out, ref, cache_dir=cache)

    assert result is not None  # manifest written
    with open(result) as f:
        manifest = json.load(f)
    assert len(manifest["jobs"]) == 1


def test_no_skip_when_kh4b_status_ok(tmp_path, monkeypatch):
    """KH-4B with `coarse_align_status="ok"` (sidecar exists) proceeds."""
    monkeypatch.delenv("DECLASS_ALLOW_UNCOARSE_ALIGN", raising=False)
    monkeypatch.setattr(process, "generate_auto_anchors",
                        lambda *args, **kwargs: None)
    scene = make_scene(entity_id="DS1104-1057DA023", corners=_FRAME_CORNERS)
    cache, out, ref = _setup_cache(tmp_path, scene, coarse_status="ok")

    result = process.generate_manifest([scene], out, ref, cache_dir=cache)

    assert result is not None
    with open(result) as f:
        manifest = json.load(f)
    assert len(manifest["jobs"]) == 1


def test_env_override_bypasses_skip(tmp_path, monkeypatch):
    """DECLASS_ALLOW_UNCOARSE_ALIGN=1 lets KH-4B proceed even when
    coarse-align abstained — diagnostic-only escape hatch."""
    monkeypatch.setenv("DECLASS_ALLOW_UNCOARSE_ALIGN", "1")
    monkeypatch.setattr(process, "generate_auto_anchors",
                        lambda *args, **kwargs: None)
    scene = make_scene(entity_id="DS1104-1057DA023", corners=_FRAME_CORNERS)
    cache, out, ref = _setup_cache(tmp_path, scene, coarse_status="abstained")

    result = process.generate_manifest([scene], out, ref, cache_dir=cache)

    assert result is not None
    with open(result) as f:
        manifest = json.load(f)
    assert len(manifest["jobs"]) == 1


def test_anchors_override_bypasses_skip(tmp_path, monkeypatch):
    """--anchors-override supplies hand-curated GCPs that give the
    geographic prior coarse-align couldn't; the hard-fail skip on
    unreliable-corners profiles must yield to that prior so the
    iteration loop on KH-4B keeps working."""
    monkeypatch.delenv("DECLASS_ALLOW_UNCOARSE_ALIGN", raising=False)
    monkeypatch.setattr(process, "generate_auto_anchors",
                        lambda *args, **kwargs: None)
    scene = make_scene(entity_id="DS1104-1057DA023", corners=_FRAME_CORNERS)
    cache, out, ref = _setup_cache(tmp_path, scene, coarse_status="abstained")

    anchors_override = tmp_path / "bahrain_anchors.json"
    anchors_override.write_text(json.dumps({"gcps": [
        {"name": "x", "lon": 50.55, "lat": 25.85}
    ]}))

    result = process.generate_manifest(
        [scene], out, ref, cache_dir=cache,
        anchors_override=str(anchors_override),
    )

    assert result is not None, "anchors-override should bypass the hard-fail skip"
    with open(result) as f:
        manifest = json.load(f)
    assert len(manifest["jobs"]) == 1


def test_anchors_override_missing_does_not_bypass(tmp_path, monkeypatch):
    """A non-existent --anchors-override path is treated as no override:
    the hard-fail skip still fires."""
    monkeypatch.delenv("DECLASS_ALLOW_UNCOARSE_ALIGN", raising=False)
    monkeypatch.setattr(process, "generate_auto_anchors",
                        lambda *args, **kwargs: None)
    scene = make_scene(entity_id="DS1104-1057DA023", corners=_FRAME_CORNERS)
    cache, out, ref = _setup_cache(tmp_path, scene, coarse_status="abstained")

    result = process.generate_manifest(
        [scene], out, ref, cache_dir=cache,
        anchors_override=str(tmp_path / "does_not_exist.json"),
    )
    assert result is None  # skip fires → no manifest


# ---------------------------------------------------------------------------
# coarse_align_status persistence
# ---------------------------------------------------------------------------

def test_record_coarse_align_status_round_trip(tmp_path):
    scene = make_scene(entity_id="DS1104-1057DA023", corners=_FRAME_CORNERS)
    cache = str(tmp_path / "cache")
    paths_mod.ensure_pipeline_dirs(str(tmp_path / "out"), cache)

    # Initial metadata write
    process._merge_scene_metadata(cache, scene)
    initial = process._load_scene_metadata(cache, scene.entity_id)
    assert initial["coarse_align_status"] is None

    process._record_coarse_align_status(cache, scene.entity_id, "abstained")
    after_abstain = process._load_scene_metadata(cache, scene.entity_id)
    assert after_abstain["coarse_align_status"] == "abstained"

    process._record_coarse_align_status(cache, scene.entity_id, "ok")
    after_ok = process._load_scene_metadata(cache, scene.entity_id)
    assert after_ok["coarse_align_status"] == "ok"


def test_record_coarse_align_status_no_metadata_is_noop(tmp_path):
    """Tolerant of pre-metadata calls (e.g. coarse-align attempted before
    georef write completed) — no-op rather than crash."""
    cache = str(tmp_path / "cache")
    paths_mod.ensure_pipeline_dirs(str(tmp_path / "out"), cache)
    process._record_coarse_align_status(cache, "ENTITY_NOT_REGISTERED", "abstained")
    # No exception, no file written.
    assert process._load_scene_metadata(cache, "ENTITY_NOT_REGISTERED") is None
