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


@pytest.mark.fast
@pytest.mark.process
def test_measure_segment_seams_uses_phase_aligned_correlation(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from preprocess.camera_model import _measure_segment_seams

    rng = np.random.default_rng(7)
    base = rng.normal(loc=120.0, scale=25.0, size=(160, 192)).astype(np.float32)
    shift_px = 12
    shifted = np.zeros_like(base)
    shifted[:, shift_px:] = base[:, :-shift_px]

    profile = {
        "driver": "GTiff",
        "width": base.shape[1],
        "height": base.shape[0],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:3857",
        "transform": from_origin(5_620_000.0, 3_035_000.0, 3.5, 3.5),
        "nodata": -32768.0,
    }
    a_path = tmp_path / "seam_a.tif"
    b_path = tmp_path / "seam_b.tif"
    with rasterio.open(a_path, "w", **profile) as ds:
        ds.write(base, 1)
    with rasterio.open(b_path, "w", **profile) as ds:
        ds.write(shifted, 1)

    report = _measure_segment_seams([str(a_path), str(b_path)])[0]

    assert report["status"] == "ok"
    assert abs(report["phase_shift_px"] - shift_px) < 1.0
    assert report["zncc"] > 0.7
    assert report["raw_zncc"] < report["zncc"] - 0.3


@pytest.mark.fast
@pytest.mark.process
def test_seam_report_passes_on_low_shift_positive_response():
    from preprocess.camera_model import _seam_report_passes

    report = {
        "status": "ok",
        "zncc": -0.16,
        "phase_shift_px": 1.98,
        "response": 0.008,
    }
    assert _seam_report_passes(report, seam_shift_px_max=2.0) is True

    report_bad = dict(report, response=-0.01)
    assert _seam_report_passes(report_bad, seam_shift_px_max=2.0) is False


@pytest.mark.fast
@pytest.mark.process
def test_per_segment_precorrect_reverses_and_rotates_aft_frames(tmp_path, monkeypatch):
    """With is_aft=True, opticalbar_per_segment_precorrect must reverse
    frame order (so last delivery frame becomes seg 0) AND pass rotated,
    cleaned copies into the 14p control-extraction path.  This replicates
    image_mosaic --rotate, which rotates the assembled mosaic 180
    (= reverse + flip each)."""
    import preprocess.camera_model as cm
    import preprocess.kh_panoramic as kp

    # Create 3 small sub-frames with an asymmetric pattern
    import numpy as np
    from osgeo import gdal
    frames = []
    for letter in ("a", "b", "c"):
        p = tmp_path / f"TEST_SEG_{letter}.tif"
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(str(p), 64, 32, 1, gdal.GDT_Byte)
        arr = np.zeros((32, 64), dtype=np.uint8)
        arr[:16, :32] = 200  # top-left quadrant bright
        ds.GetRasterBand(1).WriteArray(arr)
        ds.FlushCache()
        ds = None
        frames.append(str(p))

    # Track what paths are passed into the 14p GCP extractor.
    observed_paths = []

    def _synthetic_gcps():
        cols = np.linspace(4, 59, 6)
        rows = np.linspace(4, 27, 5)
        cc, rr = np.meshgrid(cols, rows)
        x = 5600000.0 + cc.ravel() * 8.0
        y = 3040000.0 - rr.ravel() * 8.0
        z = np.zeros_like(x)
        return np.column_stack([cc.ravel(), rr.ravel(), x, y, z]).astype(np.float64)

    def fake_extract_reference_gcps(sub_frame_path, *a, **kw):
        observed_paths.append(sub_frame_path)
        return _synthetic_gcps()

    def fake_fit_panoramic(sub_frame_gcps, initial, *a, **kw):
        return type(
            "FitResult",
            (),
            {
                "params": initial,
                "reprojection_rms_px": 1.0,
                "success": True,
            },
        )()

    def fake_extract_model_guided_gcps(*a, **kw):
        return None

    def fake_mapproject(*a, **kw):
        out = kw["out_path"]
        seg_idx = int(Path(out).stem.split("_seg", 1)[1][:2])
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(out, 64, 32, 1, gdal.GDT_Float32)
        x_origin = 5600000.0 + seg_idx * (64 - 20) * 3.5
        ds.SetGeoTransform([x_origin, 3.5, 0, 3040000.0, 0, -3.5])
        from osgeo import osr
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        ds.SetProjection(srs.ExportToWkt())
        cols = seg_idx * (64 - 20) + np.arange(64, dtype=np.float32)
        rows = np.arange(32, dtype=np.float32)[:, None]
        arr = rows * 4.0 + cols[None, :] * 2.0
        ds.GetRasterBand(1).WriteArray(arr)
        ds.GetRasterBand(1).SetNoDataValue(-32768)
        ds.FlushCache()
        ds = None
        return out

    monkeypatch.setattr(kp, "extract_reference_gcps", fake_extract_reference_gcps)
    monkeypatch.setattr(kp, "fit_panoramic", fake_fit_panoramic)
    monkeypatch.setattr(kp, "extract_model_guided_gcps", fake_extract_model_guided_gcps)
    monkeypatch.setattr(kp, "extract_raw_subframe_tie_points", lambda *a, **kw: None)
    monkeypatch.setattr(kp, "raw_tie_points_to_gcps", lambda *a, **kw: None)
    monkeypatch.setattr(kp, "mapproject", fake_mapproject)

    corners = {"NW": (26.3, 50.3), "NE": (26.1, 50.6),
               "SE": (25.9, 50.6), "SW": (26.1, 50.3)}
    cam_params = {"focal_length": 1.524, "pixel_pitch": 7e-6,
                  "scan_time": 0.5, "speed": 7500, "forward_tilt": 0.0,
                  "scan_dir": "right", "motion_compensation_factor": 1.0}
    reference = write_test_raster(tmp_path / "reference.tif", crs="EPSG:3857")

    out_dir = str(tmp_path / "segments")
    out = cm.opticalbar_per_segment_precorrect(
        frames, cam_params, corners, out_dir,
        scene_id="TEST_SEG", is_aft=True, reference_path=str(reference),
    )
    assert out is not None

    # The 14p matcher should receive rotated + cleaned paths, not originals.
    # The rotation step produces {scene}_seg{i:02d}_rot180.tif and the
    # clean step produces {scene}_seg{i:02d}_rot180_clean.tif; cam_gen
    # receives the cleaned variant (see _clean_raw_subframe for why).
    assert len(observed_paths) == 3
    for p in observed_paths:
        assert "_rot180" in p, f"Expected rotated path, got {p}"
        assert "_clean" in p, f"Expected cleaned path, got {p}"

    # Order must be reversed: seg00 was built from frame "c", seg02 from "a"
    assert "seg00_rot180" in observed_paths[0]
    assert "seg02_rot180" in observed_paths[2]


@pytest.mark.fast
@pytest.mark.process
def test_phase4_resolve_bbox_policy_defaults_and_unknown():
    """Phase 4: empty defaults to predicted_union_gcp (experimental),
    unknown values warn and fall back to gcp_hull, known values pass
    through unchanged."""
    from preprocess.camera_model import _resolve_bbox_policy

    assert _resolve_bbox_policy(None) == "predicted_union_gcp"
    assert _resolve_bbox_policy("") == "predicted_union_gcp"
    assert _resolve_bbox_policy("gcp_hull") == "gcp_hull"
    assert _resolve_bbox_policy("predicted_union_gcp") == "predicted_union_gcp"
    assert _resolve_bbox_policy("bogus") == "gcp_hull"


@pytest.mark.fast
@pytest.mark.process
def test_phase4_predicted_bbox_unions_with_gcp_hull():
    """Phase 4: when GCPs cluster in a narrow Y-band but the predicted
    footprint spans the full sub-frame, the union bbox is Y-tall enough
    to share real overlap with a neighbouring segment's ortho.

    Constructed scenario: 20 GCPs packed into a 1 km Y-strip near the
    segment centroid, but the predicted sub-frame footprint (from a
    nadir camera at 170 km) spans ~50 km in Y. The legacy gcp_hull
    bbox would be ~1-3 km tall (after padding); the predicted_union_gcp
    bbox must be at least an order of magnitude taller.
    """
    import math
    import numpy as np
    from preprocess.camera_model import (
        _bbox_from_gcps,
        _predicted_segment_bbox,
        _resolve_render_bbox,
    )
    from preprocess.kh_panoramic import PanoramicParams

    cx, cy = 5_620_000.0, 3_035_000.0
    img_w = 20_000
    img_h = 20_000
    pixel_pitch = 7e-6
    params = PanoramicParams(
        Xs0=cx, Ys0=cy, Zs0=170_000.0,
        omega0=0.0, phi0=0.0, kappa0=0.0,
        Xs1=0.0, Ys1=0.0, Zs1=0.0,
        omega1=0.0, phi1=0.0, kappa1=0.0,
        P=0.0, f=1.524,
    )

    # 20 GCPs packed into a narrow along-track band (typical Bahrain).
    rng = np.random.default_rng(0)
    gcps = np.column_stack([
        rng.uniform(0, img_w, size=20),
        rng.uniform(img_h * 0.48, img_h * 0.52, size=20),  # ~2 % of image height
        rng.uniform(cx - 1_000, cx + 1_000, size=20),
        rng.uniform(cy - 500, cy + 500, size=20),          # ~1 km Y-span
        np.zeros(20),
    ]).astype(np.float64)

    legacy_bbox = _bbox_from_gcps(gcps)
    predicted_bbox = _predicted_segment_bbox(
        params, pixel_pitch, img_w, img_h,
    )
    union_bbox, pred_echo, gcp_echo = _resolve_render_bbox(
        bbox_policy="predicted_union_gcp",
        gcps=gcps,
        fit_params=params,
        pixel_pitch=pixel_pitch,
        image_width_px=img_w,
        image_height_px=img_h,
    )

    # Legacy hull Y-span should be small (GCPs in a narrow band).
    legacy_span_y = legacy_bbox[3] - legacy_bbox[1]
    assert legacy_span_y < 5_000, (
        f"legacy GCP-hull Y-span {legacy_span_y:.0f} m should be < 5 km"
    )

    # Predicted footprint is driven by image dimensions × pixel_pitch × f.
    assert predicted_bbox is not None
    pred_span_y = predicted_bbox[3] - predicted_bbox[1]
    assert pred_span_y > 5_000, (
        f"predicted footprint Y-span {pred_span_y:.0f} m should be > 5 km "
        f"for a 170-km-altitude nadir camera"
    )

    # Union bbox must be at least as tall as the predicted footprint.
    union_span_y = union_bbox[3] - union_bbox[1]
    assert union_span_y >= pred_span_y, (
        f"union Y-span {union_span_y:.0f} should include predicted "
        f"{pred_span_y:.0f}"
    )


@pytest.mark.fast
@pytest.mark.process
def test_phase4_resolve_render_bbox_gcp_hull_mode_skips_forward_project():
    """Phase 4 back-compat: explicit 'gcp_hull' mode must not touch the
    fit params. Protects users who opt into the legacy bbox on new
    scenes without a usable Stage A/B fit."""
    import numpy as np
    from preprocess.camera_model import _resolve_render_bbox

    gcps = np.array([
        [0.0, 0.0, 5_620_000.0, 3_035_000.0, 0.0],
        [100.0, 100.0, 5_620_500.0, 3_035_500.0, 0.0],
    ])
    final_bbox, predicted_bbox, gcp_bbox = _resolve_render_bbox(
        bbox_policy="gcp_hull",
        gcps=gcps,
        fit_params=None,           # would crash forward_project
        pixel_pitch=7e-6,
        image_width_px=20_000,
        image_height_px=20_000,
    )
    assert predicted_bbox is None
    assert final_bbox == gcp_bbox


def _phase3_altitude_test_fixture(tmp_path, monkeypatch, cam_gen_altitude_m,
                                  tle_altitude_m, catalog_mean_m=None,
                                  fit_rms_override=None):
    """Shared setup for the Phase 3 altitude-gate tests.

    Builds 3 minimal sub-frames, mocks the heavy helpers, and installs a
    mock ``altitude_m_at`` / ``cam_gen_opticalbar_per_subframe`` pair so
    the test can control both candidates that feed the resolver.

    Returns (out_path, telemetry_dict).
    """
    import json
    import numpy as np
    import preprocess.camera_model as cm
    import preprocess.kh_panoramic as kp
    import preprocess.mission_altitude as ma
    from osgeo import gdal, osr
    from pathlib import Path as _P

    frames = []
    for idx in range(3):
        p = tmp_path / f"TEST_ALT_{idx}.tif"
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(str(p), 64, 32, 1, gdal.GDT_Byte)
        arr = np.zeros((32, 64), dtype=np.uint8)
        arr[:, 12:52] = 180
        ds.GetRasterBand(1).WriteArray(arr)
        ds.FlushCache()
        ds = None
        frames.append(str(p))

    def _synthetic_gcps():
        cols = np.linspace(4, 59, 6)
        rows = np.linspace(4, 27, 5)
        cc, rr = np.meshgrid(cols, rows)
        x = 5600000.0 + cc.ravel() * 8.0
        y = 3040000.0 - rr.ravel() * 8.0
        z = np.zeros_like(x)
        return np.column_stack([cc.ravel(), rr.ravel(), x, y, z]).astype(np.float64)

    def fake_fit_panoramic(sub_frame_gcps, initial, *a, **kw):
        return type(
            "FitResult", (),
            {"params": initial, "reprojection_rms_px": 1.0, "success": True},
        )()

    def fake_mapproject(*a, **kw):
        out = kw["out_path"]
        seg_idx = int(_P(out).stem.split("_seg", 1)[1][:2])
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(out, 64, 32, 1, gdal.GDT_Float32)
        x_origin = 5600000.0 + seg_idx * (64 - 20) * 3.5
        ds.SetGeoTransform([x_origin, 3.5, 0, 3040000.0, 0, -3.5])
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        ds.SetProjection(srs.ExportToWkt())
        cols = seg_idx * (64 - 20) + np.arange(64, dtype=np.float32)
        rows = np.arange(32, dtype=np.float32)[:, None]
        arr = rows * 4.0 + cols[None, :] * 2.0
        ds.GetRasterBand(1).WriteArray(arr)
        ds.GetRasterBand(1).SetNoDataValue(-32768)
        ds.FlushCache()
        ds = None
        return out

    def fake_cam_gen(*a, **kw):
        # cam_gen must write something to output_tsai to not look broken
        # downstream. The callers only consult the returned dict.
        return {
            "altitude_m": float(cam_gen_altitude_m),
            "focal_length": 1.524,
            "lat_rad": 0.458,
            "lon_rad": 0.881,
            "iC": [0.0, 0.0, 0.0],
        }

    class _FakeAltitudeResult:
        def __init__(self, alt_m):
            self.altitude_m = float(alt_m)
            self.source = "from_tle_at_closest_pass"
            self.tle_epoch_utc = "1977-08-27T00:00:00Z"
            self.subpoint_distance_km = 12.3

    def fake_altitude_m_at(*a, **kw):
        return _FakeAltitudeResult(tle_altitude_m)

    monkeypatch.setattr(kp, "extract_reference_gcps",
                        lambda *a, **kw: _synthetic_gcps())
    monkeypatch.setattr(kp, "fit_panoramic", fake_fit_panoramic)
    monkeypatch.setattr(kp, "extract_model_guided_gcps",
                        lambda *a, **kw: None)
    monkeypatch.setattr(kp, "extract_raw_subframe_tie_points",
                        lambda *a, **kw: None)
    monkeypatch.setattr(kp, "raw_tie_points_to_gcps",
                        lambda *a, **kw: None)
    monkeypatch.setattr(kp, "mapproject", fake_mapproject)
    monkeypatch.setattr(cm, "cam_gen_opticalbar_per_subframe", fake_cam_gen)
    monkeypatch.setattr(cm, "altitude_m_at", fake_altitude_m_at)
    # Phase 3c: the catalog-mean altitude is looked up from
    # ``data/kh_missions.yaml`` via ``catalog_mean_altitude_m``. In
    # tests we control it explicitly — None means "no catalog_mean
    # candidate", a float means "return this value in metres".
    monkeypatch.setattr(cm, "catalog_mean_altitude_m",
                        lambda *a, **kw: catalog_mean_m)

    corners = {"NW": (26.3, 50.3), "NE": (26.1, 50.6),
               "SE": (25.9, 50.6), "SW": (26.1, 50.3)}
    cam_params = {"focal_length": 1.524, "pixel_pitch": 7e-6,
                  "scan_time": 0.5, "speed": 7500, "forward_tilt": 0.0,
                  "scan_dir": "right", "motion_compensation_factor": 1.0,
                  "cam_gen_altitude": True}
    reference = write_test_raster(tmp_path / "ref_alt.tif", crs="EPSG:3857")
    seg_dir = tmp_path / "segments_alt"

    out = cm.opticalbar_per_segment_precorrect(
        frames, cam_params, corners, str(seg_dir),
        scene_id="D3C1213-200346A003",  # real scene_id so parse_entity_id works
        is_aft=True,
        reference_path=str(reference),
        acq_date=__import__("datetime").date(1977, 8, 27),
    )

    telem_path = seg_dir / "per_segment_telemetry.json"
    with open(telem_path) as fh:
        telem = json.load(fh)
    return out, telem


@pytest.mark.fast
@pytest.mark.process
def test_phase3_altitude_gate_tiebreak_on_moderate_disagreement(tmp_path, monkeypatch):
    """Phase 3b: a 17 km cam_gen-vs-TLE delta (the observed Bahrain
    case) is in the 5-30 km moderate-disagreement band. The resolver
    defers to a Stage-A/B fit-quality tiebreak rather than rejecting
    outright. With the fake_fit_panoramic returning a constant RMS for
    both candidates, TLE wins the tie via the built-in 0.25 px
    hysteresis, so altitude_source_used ends at 'tle'. The candidates
    list records both Stage A/B RMS values for attribution."""
    out, telem = _phase3_altitude_test_fixture(
        tmp_path, monkeypatch,
        cam_gen_altitude_m=147_800.0,
        tle_altitude_m=165_300.0,
    )
    assert telem["strip_cam_gen_altitude_m"] == pytest.approx(147_800.0)
    assert telem["strip_tle_altitude_m"] == pytest.approx(165_300.0)
    assert telem["cam_gen_altitude_delta_km"] == pytest.approx(-17.5, abs=0.01)
    assert telem["cam_gen_altitude_status"] == "tiebreak_tle_wins", (
        f"17 km delta should trigger tiebreak resolved to TLE; got "
        f"{telem['cam_gen_altitude_status']}"
    )
    assert telem["altitude_source_used"] == "tle"
    assert telem["altitude_used_m"] == pytest.approx(165_300.0)
    # Candidates list recorded for diffability.
    candidates = telem.get("altitude_tiebreak_candidates", [])
    sources = {c["source"] for c in candidates}
    assert sources == {"cam_gen", "tle"}
    for c in candidates:
        assert c["usable"] is True
        assert c["rms_px"] == pytest.approx(1.0, abs=0.01)


@pytest.mark.fast
@pytest.mark.process
def test_phase3_altitude_tiebreak_prefers_cam_gen_when_it_fits_better(tmp_path, monkeypatch):
    """Phase 3b: when cam_gen's altitude produces a materially better
    Stage-A/B fit than TLE's (memory records the 216 km cam_gen /
    165 km TLE scenario where cam_gen was the correct call), the
    tiebreak preserves cam_gen instead of defaulting to TLE."""
    import numpy as np
    import preprocess.camera_model as cm
    import preprocess.kh_panoramic as kp
    from osgeo import gdal, osr
    from pathlib import Path as _P
    import json

    frames = []
    for idx in range(3):
        p = tmp_path / f"TEST_TB_{idx}.tif"
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(str(p), 64, 32, 1, gdal.GDT_Byte)
        arr = np.zeros((32, 64), dtype=np.uint8)
        arr[:, 12:52] = 180
        ds.GetRasterBand(1).WriteArray(arr)
        ds.FlushCache()
        ds = None
        frames.append(str(p))

    def _synthetic_gcps():
        cols = np.linspace(4, 59, 6)
        rows = np.linspace(4, 27, 5)
        cc, rr = np.meshgrid(cols, rows)
        x = 5600000.0 + cc.ravel() * 8.0
        y = 3040000.0 - rr.ravel() * 8.0
        z = np.zeros_like(x)
        return np.column_stack([cc.ravel(), rr.ravel(), x, y, z]).astype(np.float64)

    def fake_fit_panoramic(sub_frame_gcps, initial, *a, **kw):
        # Altitude-dependent RMS: the cam_gen candidate at 195 km yields
        # low RMS; the TLE candidate at 165 km fits badly.
        zs0_km = float(initial.Zs0) / 1000.0
        rms = 1.2 if abs(zs0_km - 195.0) < 5.0 else 25.0
        return type(
            "FitResult", (),
            {"params": initial, "reprojection_rms_px": rms,
             "reprojection_rms_m": rms * 7e-6,
             "success": True, "message": "synthetic"},
        )()

    def fake_mapproject(*a, **kw):
        out = kw["out_path"]
        seg_idx = int(_P(out).stem.split("_seg", 1)[1][:2])
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(out, 64, 32, 1, gdal.GDT_Float32)
        x_origin = 5600000.0 + seg_idx * (64 - 20) * 3.5
        ds.SetGeoTransform([x_origin, 3.5, 0, 3040000.0, 0, -3.5])
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        ds.SetProjection(srs.ExportToWkt())
        cols = seg_idx * (64 - 20) + np.arange(64, dtype=np.float32)
        rows = np.arange(32, dtype=np.float32)[:, None]
        arr = rows * 4.0 + cols[None, :] * 2.0
        ds.GetRasterBand(1).WriteArray(arr)
        ds.GetRasterBand(1).SetNoDataValue(-32768)
        ds.FlushCache()
        ds = None
        return out

    def fake_cam_gen(*a, **kw):
        return {
            "altitude_m": 195_000.0,   # 30 km above TLE but physical
            "focal_length": 1.524,
            "lat_rad": 0.458,
            "lon_rad": 0.881,
            "iC": [0.0, 0.0, 0.0],
        }

    class _FakeAltitudeResult:
        altitude_m = 165_300.0
        source = "from_tle_at_closest_pass"
        tle_epoch_utc = "1977-08-27T00:00:00Z"
        subpoint_distance_km = 12.3

    monkeypatch.setattr(kp, "extract_reference_gcps",
                        lambda *a, **kw: _synthetic_gcps())
    monkeypatch.setattr(kp, "fit_panoramic", fake_fit_panoramic)
    monkeypatch.setattr(kp, "extract_model_guided_gcps",
                        lambda *a, **kw: None)
    monkeypatch.setattr(kp, "extract_raw_subframe_tie_points",
                        lambda *a, **kw: None)
    monkeypatch.setattr(kp, "raw_tie_points_to_gcps",
                        lambda *a, **kw: None)
    monkeypatch.setattr(kp, "mapproject", fake_mapproject)
    monkeypatch.setattr(cm, "cam_gen_opticalbar_per_subframe", fake_cam_gen)
    monkeypatch.setattr(cm, "altitude_m_at",
                        lambda *a, **kw: _FakeAltitudeResult())
    # Phase 3b test: restrict to 2 candidates (no catalog_mean).
    monkeypatch.setattr(cm, "catalog_mean_altitude_m",
                        lambda *a, **kw: None)

    corners = {"NW": (26.3, 50.3), "NE": (26.1, 50.6),
               "SE": (25.9, 50.6), "SW": (26.1, 50.3)}
    cam_params = {"focal_length": 1.524, "pixel_pitch": 7e-6,
                  "scan_time": 0.5, "speed": 7500, "forward_tilt": 0.0,
                  "scan_dir": "right", "motion_compensation_factor": 1.0,
                  "cam_gen_altitude": True}
    reference = write_test_raster(tmp_path / "ref_tb.tif", crs="EPSG:3857")
    seg_dir = tmp_path / "segments_tb"

    cm.opticalbar_per_segment_precorrect(
        frames, cam_params, corners, str(seg_dir),
        scene_id="D3C1213-200346A003",
        is_aft=True, reference_path=str(reference),
        acq_date=__import__("datetime").date(1977, 8, 27),
    )
    telem = json.loads((seg_dir / "per_segment_telemetry.json").read_text())
    assert telem["cam_gen_altitude_status"] == "tiebreak_cam_gen_wins", (
        f"cam_gen at 195 km (RMS 1.2) should beat TLE 165 km (RMS 25); "
        f"got {telem['cam_gen_altitude_status']}"
    )
    assert telem["altitude_source_used"] == "cam_gen"
    assert telem["altitude_used_m"] == pytest.approx(195_000.0)


@pytest.mark.fast
@pytest.mark.process
def test_phase3_altitude_gate_rejects_extreme_disagreement(tmp_path, monkeypatch):
    """Phase 3b: a 50 km cam_gen-vs-TLE delta exceeds the 30 km reject
    gate; cam_gen is rejected outright without running the tiebreak.
    Protects against the degenerate 51 km / unphysical-refinement case
    where running cam_gen through the fit would still be a waste."""
    out, telem = _phase3_altitude_test_fixture(
        tmp_path, monkeypatch,
        cam_gen_altitude_m=215_300.0,   # +50 km from TLE, still physical
        tle_altitude_m=165_300.0,
    )
    assert telem["cam_gen_altitude_status"] == "rejected_extreme_disagreement"
    assert telem["altitude_source_used"] == "tle"
    # Tiebreak should not have run at extreme delta.
    assert telem.get("altitude_tiebreak_candidates") in (None, [])


@pytest.mark.fast
@pytest.mark.process
def test_phase3_altitude_gate_accepts_cam_gen_when_agrees_with_tle(tmp_path, monkeypatch):
    """Phase 3 hygiene: when cam_gen agrees with TLE within the 5 km
    tight band, the resolver keeps cam_gen's refined value without a
    tiebreak. Prevents the gate from becoming a universal 'ignore
    cam_gen' switch."""
    out, telem = _phase3_altitude_test_fixture(
        tmp_path, monkeypatch,
        cam_gen_altitude_m=168_000.0,
        tle_altitude_m=165_300.0,
    )
    assert telem["cam_gen_altitude_status"] == "used"
    assert telem["altitude_source_used"] == "cam_gen"
    assert telem["altitude_used_m"] == pytest.approx(168_000.0)
    assert telem.get("altitude_tiebreak_candidates") in (None, [])


@pytest.mark.fast
@pytest.mark.process
def test_phase3f_subpixel_accept_rule():
    """Phase 3f: a seam with sub-2-pixel shift AND ZNCC ≥ 0.3 passes
    the gate even when ZNCC < 0.4 and response < 0.005. Motivated by
    the observed Bahrain seam 0-1 after Phase 3d (shift 1.97 px,
    ZNCC 0.388, response 0.001) — the geometry is demonstrably good
    but cross-temporal terrain change depresses ZNCC below 0.4.

    Must still reject large-shift seams even with high ZNCC, and
    low-ZNCC seams even at sub-pixel shift (those are low_texture)."""
    from preprocess.camera_model import _seam_report_passes

    # Bahrain seam 0-1 after Phase 3d — the motivating case.
    bahrain_01 = {
        "status": "ok",
        "zncc": 0.388,
        "raw_zncc": 0.379,
        "phase_shift_px": 1.97,
        "response": 0.0009,
    }
    assert _seam_report_passes(bahrain_01, 80.0) is True

    # Sub-pixel but ZNCC too low → still fails (below 0.3 rule).
    subpx_low_zncc = {
        "status": "ok",
        "zncc": 0.22,
        "raw_zncc": 0.22,
        "phase_shift_px": 0.5,
        "response": 0.001,
    }
    assert _seam_report_passes(subpx_low_zncc, 80.0) is False

    # High ZNCC but large shift → fails (geometry bad).
    high_zncc_big_shift = {
        "status": "ok",
        "zncc": 0.55,
        "raw_zncc": 0.52,
        "phase_shift_px": 15.0,
        "response": 0.003,
    }
    assert _seam_report_passes(high_zncc_big_shift, 80.0) is True  # 0.55>=0.4
    high_zncc_huge_shift = {
        "status": "ok",
        "zncc": 0.55,
        "raw_zncc": 0.52,
        "phase_shift_px": 200.0,
        "response": 0.003,
    }
    assert _seam_report_passes(high_zncc_huge_shift, 80.0) is False  # above cap


@pytest.mark.fast
@pytest.mark.process
def test_phase3e_low_texture_seam_treated_as_skipped():
    """Phase 3e: a seam report with low response and low |ZNCC|
    (cloud/ocean overlap — insufficient signal to measure) gets status
    'low_texture'. Both the production QA gate (_seam_report_passes)
    and the ranking helper (_seam_report_score) treat it like a skip
    (pass / rank above the overlap-geometry failures)."""
    from preprocess.camera_model import (
        _seam_report_passes, _seam_report_score,
    )

    low_tex = {
        "index": "1-2",
        "status": "low_texture",
        "zncc": -0.126,
        "raw_zncc": -0.125,
        "phase_shift_px": 0.89,
        "response": 0.002,
    }
    assert _seam_report_passes(low_tex, 80.0) is True
    score = _seam_report_score(low_tex)
    assert score[0] == 2  # equal to small_overlap rank, below "ok" rank (3)

    # Comparison: an ``ok`` report with content match wins on score.
    ok_good = {
        "index": "0-1",
        "status": "ok",
        "zncc": 0.80,
        "raw_zncc": 0.75,
        "phase_shift_px": 1.2,
        "response": 0.04,
    }
    assert _seam_report_score(ok_good)[0] == 3

    # And an ``ok`` report with bad metrics should still rank higher
    # than low_texture in the bucket sense (both get a status bucket,
    # but only low_texture passes the production gate).
    ok_bad = {
        "index": "2-3",
        "status": "ok",
        "zncc": -0.05,
        "raw_zncc": -0.05,
        "phase_shift_px": 120.0,
        "response": 0.0001,
    }
    assert _seam_report_passes(ok_bad, 80.0) is False
    assert _seam_report_passes(low_tex, 80.0) is True


def _phase3d_test_fixture(tmp_path, monkeypatch, seg_f_values,
                          shared_f_refit_enabled=True,
                          min_spread_frac=0.02,
                          refit_f_rms_multiplier=1.0):
    """Fixture for Phase 3d shared-f tests.

    ``seg_f_values`` — dict {seg_idx: (fitted_f_m, fitted_rms_px)} that
    the fake fit returns on each segment's ``initial.Zs0``/GCPs. When
    Phase 3d issues a refit (initial.f = shared_f, fix_f=True in kwargs),
    the fake returns ``(shared_f, original_rms * refit_f_rms_multiplier)``
    so tests can control whether the refit is accepted.
    """
    import numpy as np
    import preprocess.camera_model as cm
    import preprocess.kh_panoramic as kp
    from osgeo import gdal, osr
    from pathlib import Path as _P
    import json

    frames = []
    for idx in range(3):
        p = tmp_path / f"TEST_P3D_{idx}.tif"
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(str(p), 64, 32, 1, gdal.GDT_Byte)
        arr = np.zeros((32, 64), dtype=np.uint8)
        arr[:, 12:52] = 180
        ds.GetRasterBand(1).WriteArray(arr)
        ds.FlushCache()
        ds = None
        frames.append(str(p))

    def _synthetic_gcps():
        cols = np.linspace(4, 59, 6)
        rows = np.linspace(4, 27, 5)
        cc, rr = np.meshgrid(cols, rows)
        x = 5600000.0 + cc.ravel() * 8.0
        y = 3040000.0 - rr.ravel() * 8.0
        z = np.zeros_like(x)
        return np.column_stack([cc.ravel(), rr.ravel(), x, y, z]).astype(np.float64)

    _fit_call_count = {"n": 0}

    def fake_fit_panoramic(sub_frame_gcps, initial, *a, **kw):
        _fit_call_count["n"] += 1
        # Determine which segment this call is for by looking at the
        # ground-point centroid of the GCPs (same for each seg in our
        # fixture — so we thread the segment index through the initial
        # via a small trick: the fake_mapproject encodes seg_idx into
        # the initial's Xs0 offset when building. Simpler: match by
        # initial.f if we already refitted, otherwise use next seg.)
        #
        # Here we key on kw.get("fix_f") + initial.f to decide whether
        # this is a Phase 3d refit or an initial fit. Initial fits have
        # fix_f=False; Phase 3d refits have fix_f=True and initial.f set
        # to the shared value.
        fix_f = bool(kw.get("fix_f", False))
        initial_f = float(initial.f)
        if fix_f:
            # Phase 3d refit: return (shared_f, scaled rms).
            return type(
                "FitResult", (),
                {
                    "params": initial,
                    "reprojection_rms_px": _fit_call_count.get(
                        "last_seg_rms", 10.0
                    ) * refit_f_rms_multiplier,
                    "reprojection_rms_m": 10.0 * 7e-6,
                    "success": True,
                    "message": "synthetic-refit",
                },
            )()
        # Initial fit: return per-seg f + rms based on which seg is
        # next in processing order. Heuristic: count preceding calls.
        # There are 2 calls (Stage A + Stage B) per segment in the
        # real _fit_segment_staged, so seg_idx = (n-1) // 2.
        seg_idx = (_fit_call_count["n"] - 1) // 2
        seg_idx = min(seg_idx, max(seg_f_values.keys()))
        f_val, rms_val = seg_f_values[seg_idx]
        _fit_call_count["last_seg_rms"] = rms_val
        # Build a new params with the target f
        params = type(initial)(
            **{k: getattr(initial, k) for k in (
                "Xs0", "Ys0", "Zs0",
                "omega0", "phi0", "kappa0",
                "Xs1", "Ys1", "Zs1",
                "omega1", "phi1", "kappa1",
                "P", "f",
            )}
        )
        params.f = float(f_val)
        return type(
            "FitResult", (),
            {
                "params": params,
                "reprojection_rms_px": float(rms_val),
                "reprojection_rms_m": float(rms_val) * 7e-6,
                "success": True,
                "message": "synthetic",
            },
        )()

    def fake_mapproject(*a, **kw):
        out = kw["out_path"]
        seg_idx = int(_P(out).stem.split("_seg", 1)[1][:2])
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(out, 64, 32, 1, gdal.GDT_Float32)
        x_origin = 5600000.0 + seg_idx * (64 - 20) * 3.5
        ds.SetGeoTransform([x_origin, 3.5, 0, 3040000.0, 0, -3.5])
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        ds.SetProjection(srs.ExportToWkt())
        cols = seg_idx * (64 - 20) + np.arange(64, dtype=np.float32)
        rows = np.arange(32, dtype=np.float32)[:, None]
        arr = rows * 4.0 + cols[None, :] * 2.0
        ds.GetRasterBand(1).WriteArray(arr)
        ds.GetRasterBand(1).SetNoDataValue(-32768)
        ds.FlushCache()
        ds = None
        return out

    class _FakeAltitudeResult:
        altitude_m = 170_000.0
        source = "from_catalog_nominal"
        tle_epoch_utc = None
        subpoint_distance_km = None

    monkeypatch.setattr(kp, "extract_reference_gcps",
                        lambda *a, **kw: _synthetic_gcps())
    monkeypatch.setattr(kp, "fit_panoramic", fake_fit_panoramic)
    monkeypatch.setattr(kp, "extract_model_guided_gcps",
                        lambda *a, **kw: None)
    monkeypatch.setattr(kp, "extract_raw_subframe_tie_points",
                        lambda *a, **kw: None)
    monkeypatch.setattr(kp, "raw_tie_points_to_gcps",
                        lambda *a, **kw: None)
    monkeypatch.setattr(kp, "mapproject", fake_mapproject)
    monkeypatch.setattr(cm, "altitude_m_at",
                        lambda *a, **kw: _FakeAltitudeResult())
    monkeypatch.setattr(cm, "catalog_mean_altitude_m",
                        lambda *a, **kw: None)

    corners = {"NW": (26.3, 50.3), "NE": (26.1, 50.6),
               "SE": (25.9, 50.6), "SW": (26.1, 50.3)}
    cam_params = {"focal_length": 1.524, "pixel_pitch": 7e-6,
                  "scan_time": 0.5, "speed": 7500, "forward_tilt": 0.0,
                  "scan_dir": "right", "motion_compensation_factor": 1.0,
                  "shared_f_refit_enabled": bool(shared_f_refit_enabled),
                  "shared_f_refit_min_spread_frac": float(min_spread_frac),
                  "f_frac_range": 0.30}
    reference = write_test_raster(tmp_path / "ref_p3d.tif", crs="EPSG:3857")
    seg_dir = tmp_path / "segments_p3d"

    cm.opticalbar_per_segment_precorrect(
        frames, cam_params, corners, str(seg_dir),
        scene_id="TEST_P3D", is_aft=False,
        reference_path=str(reference),
    )
    telem = json.loads((seg_dir / "per_segment_telemetry.json").read_text())
    return telem


@pytest.mark.fast
@pytest.mark.process
def test_phase3d_shares_best_off_bound_f_across_segments(tmp_path, monkeypatch):
    """Phase 3d: three segments with f = 1.0668 (bound) / 1.17 / 1.10
    where 1.17 has the lowest RMS. The off-bound-lowest-RMS wins as the
    shared f; seg00 (at bound) and seg02 get refitted with f=1.17. The
    refit rms multiplier is 1.0 so both accept."""
    telem = _phase3d_test_fixture(
        tmp_path, monkeypatch,
        seg_f_values={
            0: (1.0668, 18.0),   # at ±30% bound
            1: (1.17, 10.0),     # off-bound, best RMS → source
            2: (1.10, 15.0),     # off-bound
        },
        shared_f_refit_enabled=True,
        refit_f_rms_multiplier=1.0,
    )
    assert telem["phase3d_enabled"] is True
    assert telem["phase3d_applied"] is True
    assert telem["phase3d_shared_f_m"] == pytest.approx(1.17, abs=1e-4)
    assert telem["phase3d_shared_f_source_seg"] == 1
    refits = telem["phase3d_refit_per_segment"]
    accepted = {r["seg_idx"]: r for r in refits if r["accepted"]}
    assert set(accepted.keys()) == {0, 1, 2}, (
        f"all three should accept (source=1 is no-op-accept, 0/2 refitted); "
        f"got {sorted(accepted.keys())}"
    )


@pytest.mark.fast
@pytest.mark.process
def test_phase3d_skipped_when_all_agree(tmp_path, monkeypatch):
    """Phase 3d: when per-segment f values agree within
    ``shared_f_refit_min_spread_frac`` (default 2%), no refit runs."""
    telem = _phase3d_test_fixture(
        tmp_path, monkeypatch,
        seg_f_values={
            0: (1.520, 12.0),
            1: (1.525, 12.5),   # spread = (1.525-1.520)/1.522 = 0.33%
            2: (1.523, 13.0),
        },
        shared_f_refit_enabled=True,
    )
    assert telem["phase3d_enabled"] is True
    assert telem["phase3d_applied"] is False
    assert telem["phase3d_skipped_reason"] is not None
    assert "spread" in telem["phase3d_skipped_reason"]


@pytest.mark.fast
@pytest.mark.process
def test_phase3d_skipped_when_all_at_bound(tmp_path, monkeypatch):
    """Phase 3d: when every segment's f sits at a ±30% bound (lower
    or upper), no off-bound source exists to share. The spread check
    passes because some segments are at the lower bound and some at
    the upper bound, but no source can be picked so the phase skips
    with ``phase3d_skipped_reason = 'all_segments_at_f_bound'``."""
    telem = _phase3d_test_fixture(
        tmp_path, monkeypatch,
        seg_f_values={
            0: (1.0668, 18.0),   # lower bound (1.524 * 0.70)
            1: (1.9812, 17.0),   # upper bound (1.524 * 1.30)
            2: (1.0668, 16.0),   # lower bound
        },
        shared_f_refit_enabled=True,
        min_spread_frac=0.02,
    )
    assert telem["phase3d_enabled"] is True
    assert telem["phase3d_applied"] is False
    assert telem["phase3d_skipped_reason"] == "all_segments_at_f_bound"


@pytest.mark.fast
@pytest.mark.process
def test_phase3d_rejects_refit_on_rms_regression(tmp_path, monkeypatch):
    """Phase 3d: when the shared-f refit blows up RMS beyond
    1.5× original, the refit is rejected and the original fit is kept.
    The source segment always accepts (no-op). The other segments see
    the inflated refit RMS and reject."""
    telem = _phase3d_test_fixture(
        tmp_path, monkeypatch,
        seg_f_values={
            0: (1.0668, 18.0),
            1: (1.17, 10.0),
            2: (1.10, 15.0),
        },
        shared_f_refit_enabled=True,
        refit_f_rms_multiplier=3.0,   # each refit comes back 3× original RMS
    )
    assert telem["phase3d_enabled"] is True
    refits = telem["phase3d_refit_per_segment"]
    rejected_non_source = [
        r for r in refits
        if not r["accepted"] and r["seg_idx"] != 1
    ]
    assert len(rejected_non_source) == 2
    for r in rejected_non_source:
        assert "refit_rms" in (r["reject_reason"] or "")


@pytest.mark.fast
@pytest.mark.process
def test_phase3c_catalog_mean_enters_tiebreak_candidate_list(tmp_path, monkeypatch):
    """Phase 3c: when ``catalog_mean_altitude_m`` returns a value that
    differs from the cam_gen/TLE candidates, the tiebreak includes it
    as a third candidate. With a fake_fit_panoramic that returns RMS
    based on altitude, the lowest-RMS candidate wins.

    Setup: cam_gen @ 145 km → RMS 20 (bad); TLE @ 165 km → RMS 18 (ok);
    catalog_mean @ 200 km → RMS 14 (best). Expected winner:
    catalog_mean; altitude_source_used = 'catalog_mean'; status
    'tiebreak_catalog_mean_wins'."""
    import numpy as np
    import preprocess.camera_model as cm
    import preprocess.kh_panoramic as kp
    from osgeo import gdal, osr
    from pathlib import Path as _P
    import json

    frames = []
    for idx in range(3):
        p = tmp_path / f"TEST_PHASE3C_{idx}.tif"
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(str(p), 64, 32, 1, gdal.GDT_Byte)
        arr = np.zeros((32, 64), dtype=np.uint8)
        arr[:, 12:52] = 180
        ds.GetRasterBand(1).WriteArray(arr)
        ds.FlushCache()
        ds = None
        frames.append(str(p))

    def _synthetic_gcps():
        cols = np.linspace(4, 59, 6)
        rows = np.linspace(4, 27, 5)
        cc, rr = np.meshgrid(cols, rows)
        x = 5600000.0 + cc.ravel() * 8.0
        y = 3040000.0 - rr.ravel() * 8.0
        z = np.zeros_like(x)
        return np.column_stack([cc.ravel(), rr.ravel(), x, y, z]).astype(np.float64)

    def fake_fit_panoramic(sub_frame_gcps, initial, *a, **kw):
        # Altitude-dependent RMS: cam_gen @ 145 km → 20; TLE @ 165 → 18;
        # catalog_mean @ 200 → 14.
        zs0_km = float(initial.Zs0) / 1000.0
        if abs(zs0_km - 145.0) < 3.0:
            rms = 20.0
        elif abs(zs0_km - 165.0) < 3.0:
            rms = 18.0
        elif abs(zs0_km - 200.0) < 3.0:
            rms = 14.0
        else:
            rms = 25.0
        return type(
            "FitResult", (),
            {"params": initial, "reprojection_rms_px": rms,
             "reprojection_rms_m": rms * 7e-6,
             "success": True, "message": "synthetic"},
        )()

    def fake_mapproject(*a, **kw):
        out = kw["out_path"]
        seg_idx = int(_P(out).stem.split("_seg", 1)[1][:2])
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(out, 64, 32, 1, gdal.GDT_Float32)
        x_origin = 5600000.0 + seg_idx * (64 - 20) * 3.5
        ds.SetGeoTransform([x_origin, 3.5, 0, 3040000.0, 0, -3.5])
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        ds.SetProjection(srs.ExportToWkt())
        cols = seg_idx * (64 - 20) + np.arange(64, dtype=np.float32)
        rows = np.arange(32, dtype=np.float32)[:, None]
        arr = rows * 4.0 + cols[None, :] * 2.0
        ds.GetRasterBand(1).WriteArray(arr)
        ds.GetRasterBand(1).SetNoDataValue(-32768)
        ds.FlushCache()
        ds = None
        return out

    def fake_cam_gen(*a, **kw):
        return {
            "altitude_m": 145_000.0,
            "focal_length": 1.524,
            "lat_rad": 0.458,
            "lon_rad": 0.881,
            "iC": [0.0, 0.0, 0.0],
        }

    class _FakeAltitudeResult:
        altitude_m = 165_000.0
        source = "from_tle_at_closest_pass"
        tle_epoch_utc = "1977-08-27T00:00:00Z"
        subpoint_distance_km = 12.3

    monkeypatch.setattr(kp, "extract_reference_gcps",
                        lambda *a, **kw: _synthetic_gcps())
    monkeypatch.setattr(kp, "fit_panoramic", fake_fit_panoramic)
    monkeypatch.setattr(kp, "extract_model_guided_gcps",
                        lambda *a, **kw: None)
    monkeypatch.setattr(kp, "extract_raw_subframe_tie_points",
                        lambda *a, **kw: None)
    monkeypatch.setattr(kp, "raw_tie_points_to_gcps",
                        lambda *a, **kw: None)
    monkeypatch.setattr(kp, "mapproject", fake_mapproject)
    monkeypatch.setattr(cm, "cam_gen_opticalbar_per_subframe", fake_cam_gen)
    monkeypatch.setattr(cm, "altitude_m_at",
                        lambda *a, **kw: _FakeAltitudeResult())
    monkeypatch.setattr(cm, "catalog_mean_altitude_m",
                        lambda *a, **kw: 200_000.0)

    corners = {"NW": (26.3, 50.3), "NE": (26.1, 50.6),
               "SE": (25.9, 50.6), "SW": (26.1, 50.3)}
    cam_params = {"focal_length": 1.524, "pixel_pitch": 7e-6,
                  "scan_time": 0.5, "speed": 7500, "forward_tilt": 0.0,
                  "scan_dir": "right", "motion_compensation_factor": 1.0,
                  "cam_gen_altitude": True}
    reference = write_test_raster(tmp_path / "ref_3c.tif", crs="EPSG:3857")
    seg_dir = tmp_path / "segments_3c"

    cm.opticalbar_per_segment_precorrect(
        frames, cam_params, corners, str(seg_dir),
        scene_id="D3C1213-200346A003",
        is_aft=True, reference_path=str(reference),
        acq_date=__import__("datetime").date(1977, 8, 27),
    )
    telem = json.loads((seg_dir / "per_segment_telemetry.json").read_text())

    assert telem["strip_catalog_mean_altitude_m"] == pytest.approx(200_000.0)
    assert telem["cam_gen_altitude_status"] == "tiebreak_catalog_mean_wins", (
        f"catalog_mean @ RMS 14 should beat cam_gen (20) and TLE (18); "
        f"got {telem['cam_gen_altitude_status']}"
    )
    assert telem["altitude_source_used"] == "catalog_mean"
    assert telem["altitude_used_m"] == pytest.approx(200_000.0)
    candidates = telem["altitude_tiebreak_candidates"]
    sources = {c["source"] for c in candidates}
    assert sources == {"cam_gen", "tle", "catalog_mean"}, (
        f"expected 3 candidates, got {sources}"
    )


@pytest.mark.fast
@pytest.mark.process
def test_phase3c_catalog_mean_skipped_when_mission_missing(tmp_path, monkeypatch):
    """Phase 3c: when catalog_mean_altitude_m returns None (non-catalog
    scene or mission missing), the tiebreak runs with 2 candidates only
    and falls back to Phase 3b behaviour. Bahrain's 17 km cam_gen/TLE
    delta still triggers the tiebreak; TLE wins via hysteresis."""
    out, telem = _phase3_altitude_test_fixture(
        tmp_path, monkeypatch,
        cam_gen_altitude_m=147_800.0,
        tle_altitude_m=165_300.0,
        catalog_mean_m=None,
    )
    assert telem.get("strip_catalog_mean_altitude_m") is None
    assert telem["cam_gen_altitude_status"] == "tiebreak_tle_wins"
    candidates = telem["altitude_tiebreak_candidates"]
    sources = {c["source"] for c in candidates}
    assert sources == {"cam_gen", "tle"}, (
        f"catalog_mean absent → only 2 candidates; got {sources}"
    )


@pytest.mark.fast
@pytest.mark.process
def test_phase3_altitude_gate_rejects_cam_gen_out_of_range(tmp_path, monkeypatch):
    """Phase 3: cam_gen returning a physically impossible altitude
    (the 51 km artefact from the pre-fix path) still rejects even
    without TLE comparison. The out-of-range branch takes precedence."""
    out, telem = _phase3_altitude_test_fixture(
        tmp_path, monkeypatch,
        cam_gen_altitude_m=51_000.0,
        tle_altitude_m=165_300.0,
    )
    assert telem["cam_gen_altitude_status"] == "rejected_out_of_range"
    assert telem["altitude_source_used"] == "tle"


@pytest.mark.fast
@pytest.mark.process
def test_per_segment_precorrect_falls_back_when_seam_qa_fails(tmp_path, monkeypatch):
    import numpy as np
    import preprocess.camera_model as cm
    import preprocess.kh_panoramic as kp
    from osgeo import gdal, osr

    frames = []
    for idx in range(3):
        p = tmp_path / f"TEST_FAIL_{idx}.tif"
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(str(p), 64, 32, 1, gdal.GDT_Byte)
        arr = np.zeros((32, 64), dtype=np.uint8)
        arr[:, 12:52] = 180
        ds.GetRasterBand(1).WriteArray(arr)
        ds.FlushCache()
        ds = None
        frames.append(str(p))

    def _synthetic_gcps():
        cols = np.linspace(4, 59, 6)
        rows = np.linspace(4, 27, 5)
        cc, rr = np.meshgrid(cols, rows)
        x = 5600000.0 + cc.ravel() * 8.0
        y = 3040000.0 - rr.ravel() * 8.0
        z = np.zeros_like(x)
        return np.column_stack([cc.ravel(), rr.ravel(), x, y, z]).astype(np.float64)

    def fake_extract_reference_gcps(*a, **kw):
        return _synthetic_gcps()

    def fake_fit_panoramic(sub_frame_gcps, initial, *a, **kw):
        return type(
            "FitResult",
            (),
            {
                "params": initial,
                "reprojection_rms_px": 1.0,
                "success": True,
            },
        )()

    def fake_extract_model_guided_gcps(*a, **kw):
        return None

    def fake_mapproject(*a, **kw):
        out = kw["out_path"]
        seg_idx = int(Path(out).stem.split("_seg", 1)[1][:2])
        drv = gdal.GetDriverByName("GTiff")
        # Use tight spacing so the segments DO overlap geographically and
        # the seam QA actually measures a shift — 'no_overlap' is now a
        # soft skip (not a fallback trigger) since valid data is only
        # compared where both segments reach. Use a crafted content that
        # triggers a large measured phase shift.
        ds = drv.Create(out, 64, 32, 1, gdal.GDT_Float32)
        ds.SetGeoTransform([5600000.0 + seg_idx * 100.0, 3.5, 0, 3040000.0, 0, -3.5])
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        ds.SetProjection(srs.ExportToWkt())
        arr = np.zeros((32, 64), dtype=np.float32)
        # Each segment has a sharp edge at a different column so phase
        # correlation measures a big shift between neighbours.
        edge = 8 + seg_idx * 12
        arr[:, edge:edge + 30] = 100.0
        ds.GetRasterBand(1).WriteArray(arr)
        ds.GetRasterBand(1).SetNoDataValue(-32768)
        ds.FlushCache()
        ds = None
        return out

    # Force _measure_segment_seams to return a failing shift; this directly
    # exercises the QA fallback path without depending on real phase
    # correlation behaviour on synthetic fixtures.
    def fake_measure_segment_seams(seg_orthos):
        reports = []
        for i in range(len(seg_orthos) - 1):
            reports.append({
                "index": f"{i}-{i+1}",
                "status": "ok",
                "zncc": -0.2,
                "phase_shift_px": 200.0,  # way above any sane gate
                "response": 0.01,
                "phase_corr": {"shift_px": 200.0, "response": 0.01},
            })
        return reports

    monkeypatch.setattr(kp, "extract_reference_gcps", fake_extract_reference_gcps)
    monkeypatch.setattr(kp, "fit_panoramic", fake_fit_panoramic)
    monkeypatch.setattr(kp, "extract_model_guided_gcps", fake_extract_model_guided_gcps)
    monkeypatch.setattr(kp, "extract_raw_subframe_tie_points", lambda *a, **kw: None)
    monkeypatch.setattr(kp, "raw_tie_points_to_gcps", lambda *a, **kw: None)
    monkeypatch.setattr(kp, "mapproject", fake_mapproject)
    monkeypatch.setattr(cm, "_measure_segment_seams", fake_measure_segment_seams)

    corners = {"NW": (26.3, 50.3), "NE": (26.1, 50.6),
               "SE": (25.9, 50.6), "SW": (26.1, 50.3)}
    cam_params = {"focal_length": 1.524, "pixel_pitch": 7e-6,
                  "scan_time": 0.5, "speed": 7500, "forward_tilt": 0.0,
                  "scan_dir": "right", "motion_compensation_factor": 1.0}
    reference = write_test_raster(tmp_path / "reference_fail.tif", crs="EPSG:3857")

    out = cm.opticalbar_per_segment_precorrect(
        frames,
        cam_params,
        corners,
        str(tmp_path / "segments_fail"),
        scene_id="TEST_FAIL",
        is_aft=False,
        reference_path=str(reference),
    )
    assert out is None


@pytest.mark.fast
@pytest.mark.process
def test_per_segment_precorrect_can_skip_rms_gate_for_debug(tmp_path, monkeypatch):
    import numpy as np
    import preprocess.camera_model as cm
    import preprocess.kh_panoramic as kp
    from osgeo import gdal, osr

    monkeypatch.setenv("DECLASS_SKIP_PER_SEGMENT_RMS_GATE", "1")

    frames = []
    for idx in range(3):
        p = tmp_path / f"TEST_RMS_{idx}.tif"
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(str(p), 64, 32, 1, gdal.GDT_Byte)
        arr = np.zeros((32, 64), dtype=np.uint8)
        arr[:, 8:56] = 180
        ds.GetRasterBand(1).WriteArray(arr)
        ds.FlushCache()
        ds = None
        frames.append(str(p))

    def _synthetic_gcps():
        cols = np.linspace(4, 59, 6)
        rows = np.linspace(4, 27, 5)
        cc, rr = np.meshgrid(cols, rows)
        x = 5600000.0 + cc.ravel() * 8.0
        y = 3040000.0 - rr.ravel() * 8.0
        z = np.zeros_like(x)
        return np.column_stack([cc.ravel(), rr.ravel(), x, y, z]).astype(np.float64)

    def fake_extract_reference_gcps(*a, **kw):
        return _synthetic_gcps()

    def fake_fit_panoramic(sub_frame_gcps, initial, *a, **kw):
        return type(
            "FitResult",
            (),
            {
                "params": initial,
                "reprojection_rms_px": 12.0,
                "reprojection_rms_m": 12.0 * 7e-6,
                "success": True,
                "message": "synthetic",
            },
        )()

    def fake_extract_model_guided_gcps(*a, **kw):
        return None

    def fake_mapproject(*a, **kw):
        out = kw["out_path"]
        seg_idx = int(Path(out).stem.split("_seg", 1)[1][:2])
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(out, 64, 32, 1, gdal.GDT_Float32)
        ds.SetGeoTransform([5600000.0 + seg_idx * (64 - 20) * 3.5, 3.5, 0, 3040000.0, 0, -3.5])
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        ds.SetProjection(srs.ExportToWkt())
        cols = seg_idx * (64 - 20) + np.arange(64, dtype=np.float32)
        if seg_idx == 1:
            cols = cols + 18.0
        rows = np.arange(32, dtype=np.float32)[:, None]
        arr = rows * 4.0 + cols[None, :] * 2.0
        ds.GetRasterBand(1).WriteArray(arr)
        ds.GetRasterBand(1).SetNoDataValue(-32768)
        ds.FlushCache()
        ds = None
        return out

    monkeypatch.setattr(kp, "extract_reference_gcps", fake_extract_reference_gcps)
    monkeypatch.setattr(kp, "fit_panoramic", fake_fit_panoramic)
    monkeypatch.setattr(kp, "extract_model_guided_gcps", fake_extract_model_guided_gcps)
    monkeypatch.setattr(kp, "extract_raw_subframe_tie_points", lambda *a, **kw: None)
    monkeypatch.setattr(kp, "raw_tie_points_to_gcps", lambda *a, **kw: None)
    monkeypatch.setattr(kp, "mapproject", fake_mapproject)

    corners = {"NW": (26.3, 50.3), "NE": (26.1, 50.6),
               "SE": (25.9, 50.6), "SW": (26.1, 50.3)}
    cam_params = {"focal_length": 1.524, "pixel_pitch": 7e-6,
                  "scan_time": 0.5, "speed": 7500, "forward_tilt": 0.0,
                  "scan_dir": "right", "motion_compensation_factor": 1.0}
    reference = write_test_raster(tmp_path / "reference_rms.tif", crs="EPSG:3857")

    out = cm.opticalbar_per_segment_precorrect(
        frames,
        cam_params,
        corners,
        str(tmp_path / "segments_rms"),
        scene_id="TEST_RMS",
        is_aft=False,
        reference_path=str(reference),
    )
    assert out is not None


@pytest.mark.fast
@pytest.mark.process
def test_per_segment_precorrect_allows_soft_rms_when_seams_pass(tmp_path, monkeypatch):
    import numpy as np
    import preprocess.camera_model as cm
    import preprocess.kh_panoramic as kp
    from osgeo import gdal, osr

    monkeypatch.delenv("DECLASS_SKIP_PER_SEGMENT_RMS_GATE", raising=False)

    frames = []
    for idx in range(3):
        p = tmp_path / f"TEST_SOFT_RMS_{idx}.tif"
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(str(p), 64, 32, 1, gdal.GDT_Byte)
        arr = np.zeros((32, 64), dtype=np.uint8)
        arr[:, 8:56] = 180
        ds.GetRasterBand(1).WriteArray(arr)
        ds.FlushCache()
        ds = None
        frames.append(str(p))

    def _synthetic_gcps():
        cols = np.linspace(4, 59, 6)
        rows = np.linspace(4, 27, 5)
        cc, rr = np.meshgrid(cols, rows)
        x = 5600000.0 + cc.ravel() * 8.0
        y = 3040000.0 - rr.ravel() * 8.0
        z = np.zeros_like(x)
        return np.column_stack([cc.ravel(), rr.ravel(), x, y, z]).astype(np.float64)

    def fake_extract_reference_gcps(*a, **kw):
        return _synthetic_gcps()

    def fake_fit_panoramic(sub_frame_gcps, initial, *a, **kw):
        return type(
            "FitResult",
            (),
            {
                "params": initial,
                "reprojection_rms_px": 12.0,
                "reprojection_rms_m": 12.0 * 7e-6,
                "success": True,
                "message": "synthetic",
            },
        )()

    def fake_extract_model_guided_gcps(*a, **kw):
        return None

    def fake_mapproject(*a, **kw):
        out = kw["out_path"]
        seg_idx = int(Path(out).stem.split("_seg", 1)[1][:2])
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(out, 64, 32, 1, gdal.GDT_Float32)
        ds.SetGeoTransform([5600000.0 + seg_idx * (64 - 20) * 3.5, 3.5, 0, 3040000.0, 0, -3.5])
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        ds.SetProjection(srs.ExportToWkt())
        cols = seg_idx * (64 - 20) + np.arange(64, dtype=np.float32)
        if seg_idx == 1:
            cols = cols + 18.0
        rows = np.arange(32, dtype=np.float32)[:, None]
        arr = rows * 4.0 + cols[None, :] * 2.0
        ds.GetRasterBand(1).WriteArray(arr)
        ds.GetRasterBand(1).SetNoDataValue(-32768)
        ds.FlushCache()
        ds = None
        return out

    monkeypatch.setattr(kp, "extract_reference_gcps", fake_extract_reference_gcps)
    monkeypatch.setattr(kp, "fit_panoramic", fake_fit_panoramic)
    monkeypatch.setattr(kp, "extract_model_guided_gcps", fake_extract_model_guided_gcps)
    monkeypatch.setattr(kp, "extract_raw_subframe_tie_points", lambda *a, **kw: None)
    monkeypatch.setattr(kp, "raw_tie_points_to_gcps", lambda *a, **kw: None)
    monkeypatch.setattr(kp, "mapproject", fake_mapproject)

    corners = {"NW": (26.3, 50.3), "NE": (26.1, 50.6),
               "SE": (25.9, 50.6), "SW": (26.1, 50.3)}
    cam_params = {"focal_length": 1.524, "pixel_pitch": 7e-6,
                  "scan_time": 0.5, "speed": 7500, "forward_tilt": 0.0,
                  "scan_dir": "right", "motion_compensation_factor": 1.0}
    reference = write_test_raster(tmp_path / "reference_soft_rms.tif", crs="EPSG:3857")

    out = cm.opticalbar_per_segment_precorrect(
        frames,
        cam_params,
        corners,
        str(tmp_path / "segments_soft_rms"),
        scene_id="TEST_SOFT_RMS",
        is_aft=False,
        reference_path=str(reference),
    )
    assert out is not None


@pytest.mark.fast
@pytest.mark.process
def test_per_segment_precorrect_iterative_refit_stops_when_unimproving(tmp_path, monkeypatch):
    import numpy as np
    import preprocess.camera_model as cm
    import preprocess.kh_panoramic as kp
    from osgeo import gdal, osr

    frame = tmp_path / "TEST_GUIDED_0.tif"
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(frame), 64, 32, 1, gdal.GDT_Byte)
    arr = np.zeros((32, 64), dtype=np.uint8)
    arr[:, 8:56] = 180
    ds.GetRasterBand(1).WriteArray(arr)
    ds.FlushCache()
    ds = None

    def _coarse_gcps():
        cols = np.linspace(4, 59, 6)
        rows = np.linspace(4, 27, 5)
        cc, rr = np.meshgrid(cols, rows)
        x = 5600000.0 + cc.ravel() * 8.0
        y = 3040000.0 - rr.ravel() * 8.0
        z = np.zeros_like(x)
        return np.column_stack([cc.ravel(), rr.ravel(), x, y, z]).astype(np.float64)

    def _localized_guided_gcps():
        cols = np.linspace(8, 14, 24)
        rows = np.linspace(10, 14, 24)
        x = 5600100.0 + np.linspace(0, 30, 24)
        y = 3039900.0 - np.linspace(0, 20, 24)
        z = np.zeros_like(x)
        return np.column_stack([cols, rows, x, y, z]).astype(np.float64)

    fit_calls = {"count": 0}

    def fake_extract_reference_gcps(*a, **kw):
        return _coarse_gcps()

    def fake_fit_panoramic(sub_frame_gcps, initial, *a, **kw):
        fit_calls["count"] += 1
        return type(
            "FitResult",
            (),
            {
                "params": initial,
                "reprojection_rms_px": 2.0,
                "reprojection_rms_m": 2.0 * 7e-6,
                "success": True,
                "message": "synthetic",
            },
        )()

    def fake_extract_model_guided_gcps(*a, **kw):
        return _localized_guided_gcps()

    def fake_extract_raw_subframe_tie_points(*a, **kw):
        return None

    def fake_raw_tie_points_to_gcps(*a, **kw):
        return None

    def fake_mapproject(*a, **kw):
        out = kw["out_path"]
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(out, 64, 32, 1, gdal.GDT_Float32)
        ds.SetGeoTransform([5600000.0, 3.5, 0, 3040000.0, 0, -3.5])
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        ds.SetProjection(srs.ExportToWkt())
        arr = np.arange(64, dtype=np.float32)[None, :] + np.arange(32, dtype=np.float32)[:, None]
        ds.GetRasterBand(1).WriteArray(arr)
        ds.GetRasterBand(1).SetNoDataValue(-32768)
        ds.FlushCache()
        ds = None
        return out

    monkeypatch.setattr(kp, "extract_reference_gcps", fake_extract_reference_gcps)
    monkeypatch.setattr(kp, "fit_panoramic", fake_fit_panoramic)
    monkeypatch.setattr(kp, "extract_model_guided_gcps", fake_extract_model_guided_gcps)
    monkeypatch.setattr(kp, "extract_raw_subframe_tie_points", fake_extract_raw_subframe_tie_points)
    monkeypatch.setattr(kp, "raw_tie_points_to_gcps", fake_raw_tie_points_to_gcps)
    monkeypatch.setattr(kp, "mapproject", fake_mapproject)

    corners = {"NW": (26.3, 50.3), "NE": (26.1, 50.6),
               "SE": (25.9, 50.6), "SW": (26.1, 50.3)}
    cam_params = {"focal_length": 1.524, "pixel_pitch": 7e-6,
                  "scan_time": 0.5, "speed": 7500, "forward_tilt": 0.0,
                  "scan_dir": "right", "motion_compensation_factor": 1.0}
    reference = write_test_raster(tmp_path / "reference_guided.tif", crs="EPSG:3857")

    out = cm.opticalbar_per_segment_precorrect(
        [str(frame)],
        cam_params,
        corners,
        str(tmp_path / "segments_guided"),
        scene_id="TEST_GUIDED",
        is_aft=False,
        reference_path=str(reference),
    )
    assert out is not None
    # Default profile has ``guided_refit_max_iter: 0`` (legacy), so
    # only the initial ``_fit_segment_staged`` runs — Stage A + Stage B
    # = 2 fit calls. Phase 5 turned the iteration count into a profile
    # knob; this test exercises the legacy path (see
    # ``test_phase5_guided_refit_runs_when_profile_enables`` for the
    # ≥ 2-fit-call case).
    assert fit_calls["count"] == 2


@pytest.mark.fast
@pytest.mark.process
def test_phase5_guided_refit_runs_when_profile_enables(tmp_path, monkeypatch):
    """Phase 5: setting ``guided_refit_max_iter: 1`` in camera_params
    must re-enter ``_iterative_guided_refit`` after Stage A/B. On a
    synthetic scene where each re-fit lowers RMS by 0.5 px, the fit
    call count goes from 2 (Stage A + B) to at least 4 (Stage A + B
    + combined coarse/guided refit's Stage A + B). If the loop were
    ignored we'd still see 2."""
    import numpy as np
    import preprocess.camera_model as cm
    import preprocess.kh_panoramic as kp
    from osgeo import gdal, osr

    frame = tmp_path / "TEST_PHASE5_0.tif"
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(frame), 64, 32, 1, gdal.GDT_Byte)
    arr = np.zeros((32, 64), dtype=np.uint8)
    arr[:, 8:56] = 180
    ds.GetRasterBand(1).WriteArray(arr)
    ds.FlushCache()
    ds = None

    def _coarse_gcps():
        cols = np.linspace(4, 59, 6)
        rows = np.linspace(4, 27, 5)
        cc, rr = np.meshgrid(cols, rows)
        x = 5600000.0 + cc.ravel() * 8.0
        y = 3040000.0 - rr.ravel() * 8.0
        z = np.zeros_like(x)
        return np.column_stack([cc.ravel(), rr.ravel(), x, y, z]).astype(np.float64)

    def _guided_gcps():
        cols = np.linspace(8, 56, 24)
        rows = np.linspace(6, 26, 24)
        x = 5600000.0 + cols * 8.0
        y = 3040000.0 - rows * 8.0
        z = np.zeros_like(x)
        return np.column_stack([cols, rows, x, y, z]).astype(np.float64)

    fit_calls = {"count": 0}
    rms_schedule = [4.0, 4.0, 3.5, 3.5]  # each call returns next value

    def fake_fit_panoramic(sub_frame_gcps, initial, *a, **kw):
        idx = fit_calls["count"]
        fit_calls["count"] += 1
        rms = rms_schedule[min(idx, len(rms_schedule) - 1)]
        return type(
            "FitResult", (),
            {"params": initial, "reprojection_rms_px": rms,
             "reprojection_rms_m": rms * 7e-6,
             "success": True, "message": "synthetic"},
        )()

    def fake_mapproject(*a, **kw):
        out = kw["out_path"]
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(out, 64, 32, 1, gdal.GDT_Float32)
        ds.SetGeoTransform([5600000.0, 3.5, 0, 3040000.0, 0, -3.5])
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        ds.SetProjection(srs.ExportToWkt())
        arr = np.arange(64, dtype=np.float32)[None, :] + np.arange(32, dtype=np.float32)[:, None]
        ds.GetRasterBand(1).WriteArray(arr)
        ds.GetRasterBand(1).SetNoDataValue(-32768)
        ds.FlushCache()
        ds = None
        return out

    monkeypatch.setattr(kp, "extract_reference_gcps",
                        lambda *a, **kw: _coarse_gcps())
    monkeypatch.setattr(kp, "fit_panoramic", fake_fit_panoramic)
    monkeypatch.setattr(kp, "extract_model_guided_gcps",
                        lambda *a, **kw: _guided_gcps())
    monkeypatch.setattr(kp, "extract_raw_subframe_tie_points",
                        lambda *a, **kw: None)
    monkeypatch.setattr(kp, "raw_tie_points_to_gcps",
                        lambda *a, **kw: None)
    monkeypatch.setattr(kp, "mapproject", fake_mapproject)

    corners = {"NW": (26.3, 50.3), "NE": (26.1, 50.6),
               "SE": (25.9, 50.6), "SW": (26.1, 50.3)}
    cam_params = {"focal_length": 1.524, "pixel_pitch": 7e-6,
                  "scan_time": 0.5, "speed": 7500, "forward_tilt": 0.0,
                  "scan_dir": "right", "motion_compensation_factor": 1.0,
                  "guided_refit_max_iter": 1}
    reference = write_test_raster(tmp_path / "reference_phase5.tif", crs="EPSG:3857")

    out = cm.opticalbar_per_segment_precorrect(
        [str(frame)], cam_params, corners,
        str(tmp_path / "segments_phase5"),
        scene_id="TEST_PHASE5", is_aft=False,
        reference_path=str(reference),
    )
    assert out is not None
    # Stage A + Stage B + (at least one guided refit's Stage A + B) = ≥ 4
    assert fit_calls["count"] >= 4, (
        f"expected ≥ 4 fit calls with guided_refit_max_iter=1, "
        f"got {fit_calls['count']}"
    )


@pytest.mark.fast
@pytest.mark.process
def test_per_segment_precorrect_can_recover_bad_provisional_seam_with_raw_ties(tmp_path, monkeypatch):
    import numpy as np
    import preprocess.camera_model as cm
    import preprocess.kh_panoramic as kp
    from osgeo import gdal, osr

    frames = []
    for idx in range(2):
        p = tmp_path / f"TEST_TIE_{idx}.tif"
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(str(p), 64, 32, 1, gdal.GDT_Byte)
        arr = np.zeros((32, 64), dtype=np.uint8)
        arr[:, 8:56] = 180
        ds.GetRasterBand(1).WriteArray(arr)
        ds.FlushCache()
        ds = None
        frames.append(str(p))

    def _synthetic_gcps():
        cols = np.linspace(4, 59, 6)
        rows = np.linspace(4, 27, 5)
        cc, rr = np.meshgrid(cols, rows)
        x = 5600000.0 + cc.ravel() * 8.0
        y = 3040000.0 - rr.ravel() * 8.0
        z = np.zeros_like(x)
        return np.column_stack([cc.ravel(), rr.ravel(), x, y, z]).astype(np.float64)

    tie_calls = {"raw": 0, "gcps": 0}

    def fake_extract_reference_gcps(*a, **kw):
        return _synthetic_gcps()

    def fake_fit_panoramic(sub_frame_gcps, initial, *a, **kw):
        return type(
            "FitResult",
            (),
            {
                "params": initial,
                "reprojection_rms_px": 3.0,
                "reprojection_rms_m": 3.0 * 7e-6,
                "success": True,
                "message": "synthetic",
            },
        )()

    def fake_extract_model_guided_gcps(*a, **kw):
        return None

    def fake_extract_raw_subframe_tie_points(*a, **kw):
        tie_calls["raw"] += 1
        cols = np.linspace(10, 54, 6)
        rows = np.linspace(6, 26, 6)
        cc, rr = np.meshgrid(cols, rows)
        return np.column_stack([
            cc.ravel(),
            rr.ravel(),
            (cc - 6).ravel(),
            rr.ravel(),
        ]).astype(np.float64)

    def fake_raw_tie_points_to_gcps(*a, **kw):
        tie_calls["gcps"] += 1
        return _synthetic_gcps()

    def _world_texture(x_origin: float, width: int = 64, height: int = 32) -> np.ndarray:
        px = 3.5
        y_top = 3040000.0
        xs = x_origin + np.arange(width, dtype=np.float32) * px
        ys = y_top - np.arange(height, dtype=np.float32)[:, None] * px
        return ys * 0.01 + xs[None, :] * 0.01

    def fake_mapproject(*a, **kw):
        out = kw["out_path"]
        seg_idx = int(Path(out).stem.split("_seg", 1)[1][:2])
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(out, 64, 32, 1, gdal.GDT_Float32)
        x_origin = 5600000.0 + seg_idx * (64 - 32) * 3.5
        ds.SetGeoTransform([x_origin, 3.5, 0, 3040000.0, 0, -3.5])
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        ds.SetProjection(srs.ExportToWkt())
        arr = np.full((32, 64), -32768.0, dtype=np.float32)
        tex = _world_texture(x_origin)
        if seg_idx == 0:
            arr[:18, 12:] = tex[:18, 12:]
        elif out.endswith("_ortho_seam.tif"):
            arr[:18, :52] = tex[:18, :52]
        else:
            arr[18:, :52] = tex[18:, :52]
        ds.GetRasterBand(1).WriteArray(arr)
        ds.GetRasterBand(1).SetNoDataValue(-32768)
        ds.FlushCache()
        ds = None
        return out

    monkeypatch.setattr(kp, "extract_reference_gcps", fake_extract_reference_gcps)
    monkeypatch.setattr(kp, "fit_panoramic", fake_fit_panoramic)
    monkeypatch.setattr(kp, "extract_model_guided_gcps", fake_extract_model_guided_gcps)
    monkeypatch.setattr(kp, "extract_raw_subframe_tie_points", fake_extract_raw_subframe_tie_points)
    monkeypatch.setattr(kp, "raw_tie_points_to_gcps", fake_raw_tie_points_to_gcps)
    monkeypatch.setattr(kp, "mapproject", fake_mapproject)

    corners = {"NW": (26.3, 50.3), "NE": (26.1, 50.6),
               "SE": (25.9, 50.6), "SW": (26.1, 50.3)}
    cam_params = {"focal_length": 1.524, "pixel_pitch": 7e-6,
                  "scan_time": 0.5, "speed": 7500, "forward_tilt": 0.0,
                  "scan_dir": "right", "motion_compensation_factor": 1.0}
    reference = write_test_raster(tmp_path / "reference_tie.tif", crs="EPSG:3857")

    out = cm.opticalbar_per_segment_precorrect(
        frames,
        cam_params,
        corners,
        str(tmp_path / "segments_tie"),
        scene_id="TEST_TIE",
        is_aft=False,
        reference_path=str(reference),
    )
    assert out is not None
    assert tie_calls["raw"] >= 1
    assert tie_calls["gcps"] >= 1


@pytest.mark.fast
@pytest.mark.process
def test_per_segment_precorrect_prefers_ortho_ties_when_overlap_exists(tmp_path, monkeypatch):
    import numpy as np
    import preprocess.camera_model as cm
    import preprocess.kh_panoramic as kp
    from osgeo import gdal, osr

    frames = []
    for idx in range(2):
        p = tmp_path / f"TEST_ORTHO_TIE_{idx}.tif"
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(str(p), 64, 32, 1, gdal.GDT_Byte)
        arr = np.zeros((32, 64), dtype=np.uint8)
        arr[:, 8:56] = 180
        ds.GetRasterBand(1).WriteArray(arr)
        ds.FlushCache()
        ds = None
        frames.append(str(p))

    def _synthetic_gcps():
        cols = np.linspace(4, 59, 6)
        rows = np.linspace(4, 27, 5)
        cc, rr = np.meshgrid(cols, rows)
        x = 5600000.0 + cc.ravel() * 8.0
        y = 3040000.0 - rr.ravel() * 8.0
        z = np.zeros_like(x)
        return np.column_stack([cc.ravel(), rr.ravel(), x, y, z]).astype(np.float64)

    calls = {"ortho": 0, "raw": 0}
    seam_calls = {"count": 0}

    def fake_extract_reference_gcps(*a, **kw):
        return _synthetic_gcps()

    def fake_fit_panoramic(sub_frame_gcps, initial, *a, **kw):
        return type(
            "FitResult",
            (),
            {
                "params": initial,
                "reprojection_rms_px": 2.0,
                "reprojection_rms_m": 2.0 * 7e-6,
                "success": True,
                "message": "synthetic",
            },
        )()

    def fake_extract_model_guided_gcps(*a, **kw):
        return None

    def fake_extract_ortho_tie_point_gcps(*a, **kw):
        calls["ortho"] += 1
        return _synthetic_gcps()

    def fake_extract_raw_subframe_tie_points(*a, **kw):
        calls["raw"] += 1
        return None

    def fake_raw_tie_points_to_gcps(*a, **kw):
        return None

    def fake_measure_segment_seams(seg_orthos):
        seam_calls["count"] += 1
        if seam_calls["count"] == 1:
            return [{
                "index": "0-1",
                "status": "ok",
                "zncc": -0.2,
                "phase_shift_px": 12.0,
                "response": 0.05,
            }]
        return [{
            "index": "0-1",
            "status": "ok",
            "zncc": 0.9,
            "phase_shift_px": 0.2,
            "response": 0.9,
        }]

    def fake_mapproject(*a, **kw):
        out = kw["out_path"]
        seg_idx = int(Path(out).stem.split("_seg", 1)[1][:2])
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(out, 64, 32, 1, gdal.GDT_Float32)
        ds.SetGeoTransform([5600000.0 + seg_idx * (64 - 20) * 3.5, 3.5, 0, 3040000.0, 0, -3.5])
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        ds.SetProjection(srs.ExportToWkt())
        cols = seg_idx * (64 - 20) + np.arange(64, dtype=np.float32)
        if seg_idx == 1:
            cols = cols + 18.0
        rows = np.arange(32, dtype=np.float32)[:, None]
        arr = rows * 4.0 + cols[None, :] * 2.0
        ds.GetRasterBand(1).WriteArray(arr)
        ds.GetRasterBand(1).SetNoDataValue(-32768)
        ds.FlushCache()
        ds = None
        return out

    monkeypatch.setattr(kp, "extract_reference_gcps", fake_extract_reference_gcps)
    monkeypatch.setattr(kp, "fit_panoramic", fake_fit_panoramic)
    monkeypatch.setattr(kp, "extract_model_guided_gcps", fake_extract_model_guided_gcps)
    monkeypatch.setattr(kp, "extract_ortho_tie_point_gcps", fake_extract_ortho_tie_point_gcps)
    monkeypatch.setattr(kp, "extract_raw_subframe_tie_points", fake_extract_raw_subframe_tie_points)
    monkeypatch.setattr(kp, "raw_tie_points_to_gcps", fake_raw_tie_points_to_gcps)
    monkeypatch.setattr(kp, "mapproject", fake_mapproject)
    monkeypatch.setattr(cm, "_measure_segment_seams", fake_measure_segment_seams)

    corners = {"NW": (26.3, 50.3), "NE": (26.1, 50.6),
               "SE": (25.9, 50.6), "SW": (26.1, 50.3)}
    cam_params = {"focal_length": 1.524, "pixel_pitch": 7e-6,
                  "scan_time": 0.5, "speed": 7500, "forward_tilt": 0.0,
                  "scan_dir": "right", "motion_compensation_factor": 1.0}
    reference = write_test_raster(tmp_path / "reference_ortho_tie.tif", crs="EPSG:3857")

    out = cm.opticalbar_per_segment_precorrect(
        frames,
        cam_params,
        corners,
        str(tmp_path / "segments_ortho_tie"),
        scene_id="TEST_ORTHO_TIE",
        is_aft=False,
        reference_path=str(reference),
    )
    assert out is not None
    assert calls["ortho"] >= 1
    assert calls["raw"] == 0


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

    def fake_generate(scene_arg, cache_dir_arg, stitched_path_arg, corners_arg, reference_arg, preprocess_matcher=None):
        assert scene_arg.entity_id == scene.entity_id
        assert Path(stitched_path_arg) == stitched_path
        assert reference_arg == str(reference_path)
        assert preprocess_matcher == "roma"
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
    assert metadata["preprocess_matcher"] == "roma"
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

    def fake_generate(scene_arg, cache_dir_arg, stitched_path_arg, corners_arg, reference_arg, preprocess_matcher=None):
        assert preprocess_matcher == "roma"
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
    assert resolved["preprocess_matcher"] == "roma"
    piecewise_case.record_case_summary(
        {
            "entity_id": scene.entity_id,
            "regenerated_ortho": resolved["asp_ortho_path"],
            "regenerated_camera": resolved["asp_camera_path"],
        }
    )


@pytest.mark.fast
@pytest.mark.process
def test_ensure_scene_asp_ortho_regenerates_when_matcher_changes(tmp_path, monkeypatch, piecewise_case):
    cache_dir = tmp_path / "cache"
    process.ensure_pipeline_dirs(str(tmp_path / "run"), str(cache_dir))
    scene = make_scene()
    reference_path = write_test_raster(tmp_path / "reference.tif")
    stitched_path = write_test_raster(cache_dir / "stitched" / f"{scene.entity_id}_stitched.tif")
    georef_path = write_test_raster(cache_dir / "georef" / f"{scene.entity_id}_georef.tif")
    stale_ortho = write_test_raster(cache_dir / "ortho" / f"{scene.entity_id}_ortho.tif", crs="EPSG:3857")
    regenerated_ortho = cache_dir / "ortho" / f"{scene.entity_id}_ortho_regenerated.tif"

    call = {"matcher": None}

    def fake_generate(scene_arg, cache_dir_arg, stitched_path_arg, corners_arg, reference_arg, preprocess_matcher=None):
        call["matcher"] = preprocess_matcher
        write_test_raster(regenerated_ortho, crs="EPSG:3857")
        return str(regenerated_ortho)

    monkeypatch.setattr(process, "_maybe_generate_asp_ortho", fake_generate)

    metadata = {
        "entity_id": scene.entity_id,
        "camera_designation": "A",
        "profile": "kh4",
        "preprocess_matcher": "roma",
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
        preprocess_matcher="nift",
    )

    assert call["matcher"] == "nift"
    assert resolved["asp_ortho_path"] == str(regenerated_ortho.resolve())
    assert resolved["preprocess_matcher"] == "nift"
    piecewise_case.record_case_summary(
        {
            "entity_id": scene.entity_id,
            "matcher_before": "roma",
            "matcher_after": resolved["preprocess_matcher"],
            "regenerated_ortho": resolved["asp_ortho_path"],
        }
    )


@pytest.mark.fast
@pytest.mark.process
def test_maybe_generate_asp_ortho_can_skip_stitched_fallback(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import preprocess.camera_model as cm

    scene = make_scene(entity_id="D3C1213-200346A003", camera_type="aft")
    scene.camera_system.entity_prefix = "D3C"
    scene.camera_system.name = "KH-9"
    stitched_path = tmp_path / "stitched.tif"
    stitched_path.write_text("stitched", encoding="utf-8")
    reference_path = write_test_raster(tmp_path / "reference.tif")

    monkeypatch.setenv("DECLASS_SKIP_STITCHED_FALLBACK", "1")
    monkeypatch.setattr(
        process,
        "load_profile",
        lambda *_: SimpleNamespace(camera=SimpleNamespace(per_segment_ortho=True)),
    )
    monkeypatch.setattr(
        process,
        "_camera_params_for_scene",
        lambda *_: {
            "focal_length": 1.524,
            "pixel_pitch": 7e-6,
            "scan_time": 0.5,
            "speed": 7500,
            "forward_tilt": -0.1745,
            "scan_dir": "right",
            "motion_compensation_factor": 1.0,
        },
    )
    monkeypatch.setattr(process, "fetch_and_prepare_dem", lambda **kwargs: None)
    monkeypatch.setattr(process, "_per_segment_sub_frames", lambda *_: ["a.tif", "b.tif"])
    monkeypatch.setattr(cm, "is_aft_camera", lambda *args, **kwargs: True)
    monkeypatch.setattr(cm, "opticalbar_per_segment_precorrect", lambda **kwargs: None)

    called = {"generate_camera": False}

    def fake_generate_camera(*args, **kwargs):
        called["generate_camera"] = True
        return "should_not_happen.tsai"

    monkeypatch.setattr(process, "generate_camera", fake_generate_camera)

    result = process._maybe_generate_asp_ortho(
        scene,
        str(tmp_path),
        str(stitched_path),
        scene.corners,
        str(reference_path),
    )

    assert result is None
    assert called["generate_camera"] is False


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


@pytest.mark.fast
def test_snap_bbox_to_grid_aligns_adjacent_segments():
    """Two adjacent bboxes that start at non-grid-aligned coordinates should
    both snap to the SAME shared grid after ``_snap_bbox_to_grid``, so the
    downstream blend can place them without sub-pixel drift."""
    from preprocess.camera_model import _snap_bbox_to_grid

    res = 4.777314267158508  # ESRI World Imagery at z=15
    # seg00-like bbox — deliberately off-grid by ~0.28 px.
    a = _snap_bbox_to_grid(
        (425853.7, 2888826.9, 462755.2, 2915481.6), res,
    )
    # seg01-like bbox, offset by 3334 m east (matches Bahrain D3C1213 obs).
    b = _snap_bbox_to_grid(
        (425853.7 + 3334.023, 2888826.9, 462755.2 + 3334.023, 2915481.6), res,
    )
    # Both corners must land on integer multiples of ``res`` from (0, 0).
    for edge in (a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]):
        frac = (edge / res) % 1.0
        assert abs(frac) < 1e-9 or abs(frac - 1.0) < 1e-9, (
            f"edge {edge} fractional px {frac:.6f} — grid-snap failed"
        )
    # Snap is outward: a expands slightly but never loses the original span.
    assert a[0] <= 425853.7 and a[2] >= 462755.2
    assert a[1] <= 2888826.9 and a[3] >= 2915481.6


@pytest.mark.fast
def test_snap_bbox_to_grid_noop_on_none():
    from preprocess.camera_model import _snap_bbox_to_grid
    assert _snap_bbox_to_grid(None, 4.7) is None
    # Zero resolution → passthrough so we never divide by zero.
    assert _snap_bbox_to_grid((0, 0, 1, 1), 0.0) == (0, 0, 1, 1)
