"""Unit tests for ``align/experimental/bundle_adjust.py``.

Phase 4 of the ortho recovery plan: the BA integration is being
rebuilt around absolute GCPs (from RoMa-vs-reference matching) plus
shared intrinsics. These tests lock the GCP file writer's format
contract before we wire it into the per-segment pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from align.experimental.bundle_adjust import (
    _write_absolute_gcp_file,
    _write_cam_gen_controls,
)


def test_write_absolute_gcp_file_format(tmp_path):
    """ASP `bundle_adjust` reads GCP files with the following
    whitespace-separated schema per line:

        <id> <lat> <lon> <height> <sigma_east_m> <sigma_north_m>
            <sigma_up_m> <image_name> <col> <row> <sig_col> <sig_row>

    Check that a small synthetic input produces 11-field lines, the
    IDs are sequential across images, and the lat/lon conversion
    matches pyproj's EPSG:32639 → WGS84 output.
    """
    from pyproj import Transformer

    # EPSG:32639 = UTM zone 39N (Bahrain/Qatar area).
    local_crs = "EPSG:32639"
    tr_forward = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)

    # Two sub-frames, 3 GCPs and 2 GCPs respectively.
    gcps_seg00 = np.array([
        # col, row, X_local, Y_local, Z_local
        [100.5, 200.0,  450_000.0, 2_900_000.0, 10.0],
        [300.0, 400.0,  451_000.0, 2_901_000.0, 15.0],
        [250.0, 350.0,  450_500.0, 2_900_500.0, 12.5],
    ])
    gcps_seg01 = np.array([
        [150.0, 220.0,  452_000.0, 2_902_000.0, 20.0],
        [280.0, 410.0,  452_500.0, 2_902_500.0, 22.0],
    ])

    out_path = tmp_path / "gcps.csv"
    n_written = _write_absolute_gcp_file(
        [("seg00.tif", gcps_seg00), ("seg01.tif", gcps_seg01)],
        local_crs,
        str(out_path),
        pixel_sigma=1.5,
        xy_sigma_m=10.0,
        z_sigma_m=20.0,
    )
    assert n_written == 5

    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 5

    for idx, line in enumerate(lines):
        fields = line.split()
        assert len(fields) == 12, (
            f"expected 12 fields per GCP line (id + 6 position + 5 image), "
            f"got {len(fields)} in: {line!r}"
        )
        # Sequential IDs across segments.
        assert int(fields[0]) == idx
        # Sigma fields match what we passed.
        assert float(fields[4]) == pytest.approx(10.0)
        assert float(fields[5]) == pytest.approx(10.0)
        assert float(fields[6]) == pytest.approx(20.0)
        assert float(fields[10]) == pytest.approx(1.5)
        assert float(fields[11]) == pytest.approx(1.5)

    # Confirm lat/lon for the first GCP (seg00, row 0).
    expected_lon, expected_lat = tr_forward.transform(450_000.0, 2_900_000.0)
    f0 = lines[0].split()
    assert float(f0[1]) == pytest.approx(expected_lat, abs=1e-6)
    assert float(f0[2]) == pytest.approx(expected_lon, abs=1e-6)
    assert float(f0[3]) == pytest.approx(10.0, abs=1e-3)
    assert f0[7] == "seg00.tif"
    assert float(f0[8]) == pytest.approx(100.5, abs=1e-3)
    assert float(f0[9]) == pytest.approx(200.0, abs=1e-3)

    # Confirm seg01 starts at the correct line.
    f3 = lines[3].split()
    assert f3[7] == "seg01.tif"


def test_write_absolute_gcp_file_skips_empty_and_bad_shapes(tmp_path):
    """Empty GCP arrays or wrong-shape entries should be skipped
    silently rather than raising — keeps the per-segment caller
    robust to segments where GCP extraction failed."""
    out_path = tmp_path / "gcps.csv"
    n = _write_absolute_gcp_file(
        [
            ("seg00.tif", np.zeros((0, 5))),
            ("seg01.tif", None),
            ("seg02.tif", np.array([]).reshape(0, 0)),
        ],
        "EPSG:32639",
        str(out_path),
    )
    assert n == 0
    assert out_path.read_text() == ""


def _fake_tsai(path):
    path.write_text(
        "VERSION_4\nOPTICAL_BAR\nimage_size = 100 100\n"
        "image_center = 50 50\npitch = 7e-6\nf = 1.524\n"
        "scan_time = 0.5\nforward_tilt = 0.0\n"
        "iC = 6574000 0 0\niR = 1 0 0 0 1 0 0 0 1\n"
        "speed = 7800\nmean_earth_radius = 6371000\n"
        "mean_surface_elevation = 0.0\nmotion_compensation_factor = 1.0\n"
        "scan_dir = right\n"
    )
    return str(path)


def test_run_strip_bundle_adjustment_phase4_flags(tmp_path, monkeypatch):
    """Phase 4: ``run_strip_bundle_adjustment`` must assemble the right
    ASP command-line flags when the Phase 4 capability parameters are
    passed in. Test the command construction by mocking the subprocess
    and ``find_asp_tool``, then inspect the captured argv."""
    import align.experimental.bundle_adjust as ba_mod

    captured_cmd = {"argv": None}

    def fake_run(cmd, *a, **kw):
        captured_cmd["argv"] = list(cmd)
        class _Res:
            returncode = 0
            stdout = ""
            stderr = ""
        return _Res()

    monkeypatch.setattr(ba_mod, "find_asp_tool", lambda name: "/bin/true")
    monkeypatch.setattr(ba_mod.subprocess, "run", fake_run)

    frames = [str(tmp_path / "f0.tif"), str(tmp_path / "f1.tif"),
              str(tmp_path / "f2.tif")]
    for p in frames:
        open(p, "w").close()

    seeds = [_fake_tsai(tmp_path / f"seed_{i}.tsai") for i in range(3)]
    gcp_file = tmp_path / "gcps.csv"
    gcp_file.write_text("0 26.2 50.4 10.0  10 10 20  f0.tif  100 200 1.5 1.5\n")
    dem_file = tmp_path / "dem.tif"
    dem_file.write_text("")

    # Drop the `ba-*.tsai` file so the function can find a post-run
    # output camera (otherwise the output-collection step returns None).
    for i in range(3):
        out_cam = tmp_path / f"ba-seed_{i}.tsai"
        _fake_tsai(out_cam)

    # First invocation: reference_terrain_weight set but no disparity_list,
    # so ASP path falls back to --heights-from-dem.
    result = ba_mod.run_strip_bundle_adjustment(
        frames=frames,
        camera_params={"focal_length": 1.524, "pixel_pitch": 7e-6,
                       "scan_time": 0.5, "speed": 7800,
                       "forward_tilt": 0.0, "scan_dir": "right",
                       "motion_compensation_factor": 1.0},
        corners_list=[{}] * 3,
        output_dir=str(tmp_path),
        initial_tsai_paths=seeds,
        absolute_gcp_file=str(gcp_file),
        dem_path=str(dem_file),
        solve_intrinsics=True,
        shared_intrinsics=True,
        intrinsics_limits=(0.92, 1.08),
        reference_terrain_weight=1000.0,
        robust_threshold=2.0,
        camera_weight=0,
    )

    argv = captured_cmd["argv"]
    assert argv is not None, "subprocess.run was not called"

    def _has(flag, value=None):
        if flag not in argv:
            return False
        if value is None:
            return True
        idx = argv.index(flag)
        return idx + 1 < len(argv) and argv[idx + 1] == str(value)

    # Phase 4 capability flags must all be present.
    assert _has("--solve-intrinsics")
    assert _has("--intrinsics-to-float", "focal_length")
    assert _has("--intrinsics-to-share", "focal_length")
    assert _has("--intrinsics-limits", "0.92 1.08")
    assert _has("--robust-threshold", "2.0")
    assert _has("--camera-weight", "0")

    # Without disparities, Phase 4 falls back to --heights-from-dem so
    # --reference-terrain must NOT be present (ASP would hard-error).
    assert "--reference-terrain" not in argv
    assert _has("--heights-from-dem", str(dem_file))

    # Absolute GCP file must appear on the command line.
    assert str(gcp_file) in argv

    # The seed tsai files must appear (not freshly generated cameras).
    for seed in seeds:
        assert seed in argv

    # Second invocation: if a disparity_list is supplied, --reference-terrain
    # path is taken and --heights-from-dem is not emitted.
    disp = tmp_path / "disp.tif"
    disp.write_text("")
    result2 = ba_mod.run_strip_bundle_adjustment(
        frames=frames,
        camera_params={"focal_length": 1.524, "pixel_pitch": 7e-6,
                       "scan_time": 0.5, "speed": 7800,
                       "forward_tilt": 0.0, "scan_dir": "right",
                       "motion_compensation_factor": 1.0},
        corners_list=[{}] * 3,
        output_dir=str(tmp_path),
        initial_tsai_paths=seeds,
        absolute_gcp_file=str(gcp_file),
        dem_path=str(dem_file),
        solve_intrinsics=True,
        shared_intrinsics=True,
        intrinsics_limits=(0.92, 1.08),
        reference_terrain_weight=1000.0,
        disparity_list=[str(disp)],
        robust_threshold=2.0,
        camera_weight=0,
    )
    argv2 = captured_cmd["argv"]
    assert "--reference-terrain" in argv2
    assert "--heights-from-dem" not in argv2
    assert "--disparity-list" in argv2


def test_run_strip_bundle_adjustment_rejects_legacy_gcps_format(tmp_path, monkeypatch):
    """The legacy per-frame dict GCP format is no longer supported;
    callers that pass it must get a clear failure, not a silent
    fallback to cam_gen-only BA."""
    import align.experimental.bundle_adjust as ba_mod

    monkeypatch.setattr(ba_mod, "find_asp_tool", lambda name: "/bin/true")
    frames = [str(tmp_path / "f0.tif"), str(tmp_path / "f1.tif")]
    for p in frames:
        open(p, "w").close()
    seeds = [_fake_tsai(tmp_path / f"seed_{i}.tsai") for i in range(2)]

    result = ba_mod.run_strip_bundle_adjustment(
        frames=frames,
        camera_params={"focal_length": 1.524, "pixel_pitch": 7e-6,
                       "scan_time": 0.5, "speed": 7800,
                       "forward_tilt": 0.0, "scan_dir": "right",
                       "motion_compensation_factor": 1.0},
        corners_list=[{}] * 2,
        output_dir=str(tmp_path),
        initial_tsai_paths=seeds,
        gcps_per_frame=[[{"lon": 50.4, "lat": 26.2, "col": 100, "row": 200}]] * 2,
    )
    assert result is None


def test_write_absolute_gcp_file_accepts_wider_arrays(tmp_path):
    """Arrays with >5 columns (e.g., optional confidence score)
    should still write the first 5 columns without error."""
    out_path = tmp_path / "gcps.csv"
    gcps = np.array([
        [100.0, 200.0, 450_000.0, 2_900_000.0, 10.0, 0.95],
        [300.0, 400.0, 451_000.0, 2_901_000.0, 15.0, 0.88],
    ])
    n = _write_absolute_gcp_file(
        [("seg.tif", gcps)],
        "EPSG:32639",
        str(out_path),
    )
    assert n == 2
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 2
    # Each line has 12 fields regardless of the extra confidence column.
    for line in lines:
        assert len(line.split()) == 12


def test_write_cam_gen_controls_pairs_and_lon_lat():
    """Phase 4 seed emitter: the ``--lon-lat-values`` and
    ``--pixel-values`` strings must have 2N tokens each (one lon lat
    pair and one col row pair per GCP), and the lon/lat values must
    match pyproj's local-CRS → WGS84 transform."""
    from pyproj import Transformer

    gcps = np.array([
        [100.5, 200.0, 450_000.0, 2_900_000.0, 10.0],
        [300.0, 400.0, 451_000.0, 2_901_000.0, 15.0],
        [250.0, 350.0, 450_500.0, 2_900_500.0, 12.5],
    ])
    lon_lat_str, pixel_str, n = _write_cam_gen_controls(gcps, "EPSG:32639")
    assert n == 3
    ll_tokens = lon_lat_str.split()
    px_tokens = pixel_str.split()
    assert len(ll_tokens) == 6
    assert len(px_tokens) == 6

    tr = Transformer.from_crs("EPSG:32639", "EPSG:4326", always_xy=True)
    lon0, lat0 = tr.transform(450_000.0, 2_900_000.0)
    assert float(ll_tokens[0]) == pytest.approx(lon0, abs=1e-8)
    assert float(ll_tokens[1]) == pytest.approx(lat0, abs=1e-8)
    # Pixel string preserves insertion order (col row col row …).
    assert float(px_tokens[0]) == pytest.approx(100.5, abs=1e-3)
    assert float(px_tokens[1]) == pytest.approx(200.0, abs=1e-3)


def test_write_cam_gen_controls_max_points_subsamples():
    """When ``max_points`` is smaller than the input, the controls
    string subsamples deterministically and preserves along-scan
    coverage (sorted by col before striding). This matters for cam_gen
    LSQ stability when GCP count reaches the several-hundreds."""
    cols = np.linspace(0, 999, 100)
    rows = np.full(100, 500.0)
    xs = np.linspace(450_000.0, 470_000.0, 100)
    ys = np.full(100, 2_900_000.0)
    zs = np.full(100, 10.0)
    gcps = np.column_stack([cols, rows, xs, ys, zs])
    _, pixel_str, n = _write_cam_gen_controls(
        gcps, "EPSG:32639", max_points=10,
    )
    assert n <= 10
    # First and last pixel-col tokens should span most of the 0..999 range
    # (stratified sampling preserves extent).
    tokens = pixel_str.split()
    first_col = float(tokens[0])
    last_col = float(tokens[-2])
    assert first_col < 100
    assert last_col > 900


def test_write_cam_gen_controls_empty_input():
    """Empty or malformed GCP arrays return empty strings and count 0,
    mirroring ``_write_absolute_gcp_file``'s no-raise contract."""
    ll, px, n = _write_cam_gen_controls(
        np.zeros((0, 5)), "EPSG:32639",
    )
    assert ll == ""
    assert px == ""
    assert n == 0
