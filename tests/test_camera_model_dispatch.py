"""Tests for preprocess.camera_model.generate_camera type dispatch.

Verifies the dispatch added when we wired KH-7 (strip/linescan) and KH-9 MC
(frame/pinhole) into the ASP cam_gen pathway. Before this change
`_camera_params_for_scene` returned None for anything non-panoramic, and
`generate_camera` always called the OpticalBar subprocess; KH-7 scenes
skipped ASP orthorectification entirely. See `data/profiles/kh7.yaml`
and `memory/cross_kh_system_audit.md`.
"""

import pytest

from preprocess import camera_model


CORNERS = {
    "nw": (26.45, 50.515),
    "ne": (26.40, 50.700),
    "se": (26.065, 50.582),
    "sw": (26.115, 50.400),
}


def _fake_success_subprocess(monkeypatch, output_tsai, cam_types_recorded):
    """Stub out cam_gen tool discovery + subprocess so we can observe dispatch."""
    monkeypatch.setattr(camera_model, "find_asp_tool", lambda name: f"/fake/{name}")

    orig_opticalbar = camera_model._run_cam_gen_subprocess

    def fake_opticalbar(*args, **kwargs):
        cam_types_recorded.append("opticalbar")
        # Create a stub .tsai so the caller's existence check passes.
        with open(output_tsai, "w") as fh:
            fh.write("VERSION_4\nOPTICAL BAR\n")
        return True

    def fake_pinhole(cam_gen, focal_length, pixel_pitch, ll_str,
                     image_path, out_path, dem_path, timeout_s=120.0):
        cam_types_recorded.append("pinhole")
        with open(out_path, "w") as fh:
            fh.write(f"VERSION_4\nPINHOLE\nfu = {focal_length}\n")
        return True

    monkeypatch.setattr(camera_model, "_run_cam_gen_subprocess", fake_opticalbar)
    monkeypatch.setattr(camera_model, "_run_cam_gen_pinhole_subprocess", fake_pinhole)

    return orig_opticalbar


@pytest.mark.fast
def test_generate_camera_dispatches_opticalbar_when_unspecified(tmp_path, monkeypatch):
    """Default (no 'type' key) must still hit the OpticalBar path for KH-4/KH-9 PC."""
    from osgeo import gdal
    gdal.UseExceptions()
    img = str(tmp_path / "im.tif")
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(img, 64, 32, 1, gdal.GDT_Byte)
    ds.FlushCache()
    ds = None

    recorded = []
    _fake_success_subprocess(monkeypatch, str(tmp_path / "im.tsai"), recorded)

    params = {
        "focal_length": 1.524,
        "pixel_pitch": 7e-6,
        "scan_time": 0.5,
        "speed": 7500.0,
        "forward_tilt": 0.0,
        "scan_dir": "right",
        "motion_compensation_factor": 1.0,
    }
    out = camera_model.generate_camera(img, params, CORNERS)
    assert out is not None and out.endswith(".tsai")
    assert recorded == ["opticalbar"]


@pytest.mark.fast
@pytest.mark.parametrize("cam_type", ["linescan", "pinhole", "frame"])
def test_generate_camera_skips_pinhole_returns_none(tmp_path, monkeypatch, cam_type):
    """Pinhole / linescan cam_gen is currently disabled — the 4-coplanar
    P4P solve has branch ambiguity and produces 180°-flipped orthos.
    The function returns None so callers fall back to the georef-only
    path. See camera_model.py::generate_camera for the rationale.
    """
    img = str(tmp_path / "im.tif")
    with open(img, "wb") as fh:
        fh.write(b"dummy")

    recorded = []
    _fake_success_subprocess(monkeypatch, str(tmp_path / "im.tsai"), recorded)

    params = {"type": cam_type, "focal_length": 1.956, "pixel_pitch": 7e-6}
    out = camera_model.generate_camera(img, params, CORNERS)
    assert out is None
    # Neither opticalbar nor pinhole subprocess should run.
    assert recorded == []


@pytest.mark.fast
def test_camera_params_for_scene_returns_dict_for_linescan(monkeypatch):
    """_camera_params_for_scene used to gate non-panoramic cameras to None,
    which meant KH-7 skipped ASP entirely."""
    from types import SimpleNamespace

    import process
    from align.params import AlignParams, CameraParams, MetaParams

    linescan_profile = AlignParams(
        meta=MetaParams(name="kh7"),
        camera=CameraParams(
            type="linescan",
            focal_length=1.956,
            pixel_pitch=7e-6,
            scan_time=0.0,
            speed=7800.0,
            forward_tilt=0.0,
        ),
    )

    monkeypatch.setattr(process, "load_profile", lambda _name: linescan_profile)
    monkeypatch.setattr(process, "_profile_name_for_scene", lambda _s: "kh7")

    scene = SimpleNamespace(
        entity_id="DZB00403600089H016001",
        camera_system=SimpleNamespace(entity_prefix="DZB"),
        camera_type=None,
    )

    params = process._camera_params_for_scene(scene, "roma")
    assert params is not None, "linescan profile must return camera_params (was None)"
    assert params.get("type") == "linescan"
    assert params.get("focal_length") == pytest.approx(1.956)
