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
    # With Phase 2 iterative refit disabled (_PHASE2_MAX_ITER = 0), only
    # the initial _fit_segment_staged runs — Stage A + Stage B = 2 fit
    # calls. Re-enabling iteration will bump this to 4 (one re-fit with
    # coarse + guided GCPs) then more on further iterations.
    assert fit_calls["count"] == 2


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
