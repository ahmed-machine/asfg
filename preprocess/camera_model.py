"""ASP OpticalBar camera model generation and map-projection.

Uses NASA Ames Stereo Pipeline (ASP) cam_gen + mapproject to produce
physically correct orthorectification of panoramic KH satellite imagery,
removing bow-tie distortion, S-shaped IMC residuals, and (with DEM)
terrain parallax.

Falls back gracefully if ASP is not installed.
"""

import math
import os
import subprocess
import tempfile
import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .asp import find_asp_tool
from .mission_altitude import (
    NOMINAL_ALTITUDE_M,
    altitude_m_at,
    catalog_mean_altitude_m,
    parse_entity_id,
)


# ---------------------------------------------------------------------------
# Per-segment telemetry (Phase 0.3)
# ---------------------------------------------------------------------------
#
# A single JSON file (``{scene_id}_segments/per_segment_telemetry.json``) is
# written at the end of every ``opticalbar_per_segment_precorrect`` call —
# both success and fallback paths. It captures the decisions the per-segment
# code took so that post-hoc diff of run A vs run B becomes mechanical.
#
# Fields the recovery plan's later phases rely on:
#   - altitude authority provenance (Phase 3)
#   - per-segment fitted f + deviation from nominal (Phase 5, Phase 9)
#   - bbox source (Phase 4)
#   - coarse GCP count + image-space coverage (Phase 4, Phase 7)
#   - which seam-regularizer path fired (Phase 6)
#   - whether §4.4 iterated at all (Phase 5)
#   - whether Phase 3 TPS warp fired (Phase 10)
#   - whether the stitched fallback path was taken


@dataclass
class _SegmentTelemetry:
    seg_idx: int
    rotated_path: Optional[str] = None
    cleaned_path: Optional[str] = None
    coarse_gcp_count: Optional[int] = None
    coarse_gcp_coverage_px_w: Optional[int] = None
    coarse_gcp_coverage_px_h: Optional[int] = None
    stage_ab_rms_px: Optional[float] = None
    stage_ab_fitted_f_m: Optional[float] = None
    stage_ab_f_deviation_pct: Optional[float] = None
    guided_iter_max: int = 0
    guided_iter_accepted: int = 0
    final_fit_rms_px: Optional[float] = None
    final_fitted_f_m: Optional[float] = None
    final_f_deviation_pct: Optional[float] = None
    final_gcp_count: Optional[int] = None
    regularizer_fired: Optional[str] = None   # 'none' | 'ortho_tie' | 'raw_tie'
    bbox_source: Optional[str] = None         # 'gcp_hull' | 'predicted_union' (post-Phase-4)
    predicted_bbox: Optional[list] = None
    gcp_bbox: Optional[list] = None
    final_bbox: Optional[list] = None
    ortho_path: Optional[str] = None
    rejected: bool = False
    reject_reason: Optional[str] = None


@dataclass
class _SceneTelemetry:
    scene_id: str
    started_at_utc: str
    ortho_strategy: str = "per_segment_experimental"
    n_subframes_input: int = 0
    active_indices: list = field(default_factory=list)
    is_aft: bool = False
    # Altitude authority provenance:
    strip_tle_altitude_m: Optional[float] = None
    strip_tle_source: Optional[str] = None
    strip_cam_gen_altitude_m: Optional[float] = None
    cam_gen_altitude_delta_km: Optional[float] = None
    strip_catalog_mean_altitude_m: Optional[float] = None  # Phase 3c
    # One of: 'not_attempted' | 'used' (≤ tight gate, no tiebreak)
    #       | 'rejected_out_of_range' (physical bounds)
    #       | 'rejected_extreme_disagreement' (> reject gate vs TLE)
    #       | 'tiebreak_cam_gen_wins' | 'tiebreak_tle_wins'
    #       | 'tiebreak_catalog_mean_wins' (Phase 3c)
    #       | 'cam_gen_failed'
    cam_gen_altitude_status: Optional[str] = None
    altitude_source_used: Optional[str] = None     # 'cam_gen' | 'tle' | 'catalog_mean' | 'nominal'
    altitude_used_m: Optional[float] = None
    # Phase 3b — fit-quality tiebreak candidates (populated only when the
    # tiebreak fired; each entry is a dict of source/alt_m/rms_px).
    altitude_tiebreak_candidates: list = field(default_factory=list)
    # Phase 3d — cross-segment focal-length consistency refit.
    phase3d_enabled: bool = False
    phase3d_applied: bool = False
    phase3d_shared_f_m: Optional[float] = None
    phase3d_shared_f_source_seg: Optional[int] = None
    phase3d_skipped_reason: Optional[str] = None
    # Per-segment refit telemetry (one dict per segment that was
    # considered). Each entry: seg_idx, original_f_m, original_rms_px,
    # refit_f_m, refit_rms_px, accepted (bool), reject_reason.
    phase3d_refit_per_segment: list = field(default_factory=list)
    # Phase 4 — joint BA refinement via ASP bundle_adjust.
    phase4_enabled: bool = False
    phase4_applied: bool = False
    phase4_skipped_reason: Optional[str] = None
    phase4_gcp_count: Optional[int] = None
    phase4_pre_seams: list = field(default_factory=list)
    phase4_post_seams: list = field(default_factory=list)
    phase4_shared_f_m_before: Optional[float] = None
    phase4_shared_f_m_after: Optional[float] = None
    # Phase gates:
    phase2_max_iter: int = 0
    phase3_seam_warp_enabled: bool = False
    phase3_seam_warp_fired: bool = False
    stitched_fallback_triggered: bool = False
    fallback_reason: Optional[str] = None
    seam_qa_passed: Optional[bool] = None
    seam_reports: list = field(default_factory=list)
    # Per-segment:
    segments: list = field(default_factory=list)
    # Final:
    final_output_path: Optional[str] = None
    finished_at_utc: Optional[str] = None


def _seg_telem(scene_telem: _SceneTelemetry, seg_idx: int) -> _SegmentTelemetry:
    """Fetch or create the per-segment telemetry record for ``seg_idx``."""
    for s in scene_telem.segments:
        if s.seg_idx == seg_idx:
            return s
    rec = _SegmentTelemetry(seg_idx=seg_idx)
    scene_telem.segments.append(rec)
    return rec


def _persist_scene_telemetry(scene_telem: _SceneTelemetry, output_dir: str) -> None:
    """Write the telemetry JSON under ``output_dir``. Swallows any I/O error."""
    try:
        scene_telem.finished_at_utc = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        path = os.path.join(output_dir, "per_segment_telemetry.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(asdict(scene_telem), fh, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as exc:  # intentionally broad — telemetry must not break the pipeline
        print(f"  [per_segment/telemetry] write failed: {exc}")


def _find_gdal_tool(name: str) -> str:
    """Find a GDAL CLI tool, preferring homebrew over ASP's bundled copy.

    ASP bundles GDAL 3.8.1 whose CLI tools shadow homebrew's on PATH.
    Rasterio links against homebrew's GDAL 3.11.1.  Mixing writers
    (ASP CLI) and readers (rasterio) across major versions causes VRT
    chain / overview failures.  This helper ensures CLI tools match
    rasterio's GDAL.
    """
    brew_path = f"/opt/homebrew/bin/{name}"
    if os.path.isfile(brew_path):
        return brew_path
    return name  # fall back to PATH


def _clean_raw_subframe(in_path: str, out_path: str,
                        threshold: int = 5, keep_top_n: int = 2) -> bool:
    """Mask non-content regions of a raw KH sub-frame before mapproject.

    USGS-delivered sub-frames contain the panoramic scan PLUS other
    pixels: film data block markers, registration marks, film rebate,
    and empty regions outside the exposure. These are valid but
    irrelevant content that cam_gen happily mapprojects to the ortho,
    producing doubled-content "ghost" artifacts in the blended mosaic.

    The fix: find the largest connected component(s) of
    ``raw > threshold`` (topology-only, ignores soft gradients), take
    their union bounding box, and write a copy of the raw with
    everything outside that bbox set to a sentinel-nodata value. The
    resulting cleaned raw has the same dimensions (cam_gen's 4-corner
    fit still applies) but mapproject only emits valid pixels for the
    main panoramic region.

    On Bahrain KH-9 seg00 this identifies a 22k × 22k pixel main-content
    rectangle and drops ~300 M pixels of film-margin content including
    two isolated ~450k pixel bright rectangles (data block / registration
    markers) that were producing the doubled-content artifacts on the
    west end of the mosaic.

    Parameters
    ----------
    in_path : str
        Path to the raw sub-frame (post-rotation for Aft cameras).
    out_path : str
        Where to write the cleaned sub-frame.
    threshold : int
        Pixel-value threshold for topology extraction. Above this value
        is considered content for connected-component labelling. The
        final cleaned raw keeps all original pixel values inside the
        bbox regardless of threshold — this is just for finding the
        main-content components.
    keep_top_n : int
        How many of the largest connected components to include in the
        bbox union. Typically 2 for KH-9 PC (main ground content and
        cloud/sky region, which are often split by the threshold).

    Returns
    -------
    bool
        True on success, False if nothing was written.
    """
    try:
        import rasterio
        from scipy.ndimage import label
    except ImportError as e:
        print(f"  [clean_raw] {os.path.basename(in_path)} skipped: {e}")
        return False

    with rasterio.open(in_path) as src:
        data = src.read(1)
        profile = src.profile.copy()

    mask = data > threshold
    if not mask.any():
        print(f"  [clean_raw] {os.path.basename(in_path)}: no pixels > {threshold}")
        return False

    labels, n_cc = label(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    order = np.argsort(sizes)[::-1]
    top = order[:keep_top_n]

    union = np.zeros_like(mask)
    for lbl in top:
        if sizes[lbl] == 0:
            break
        union |= (labels == lbl)
    if not union.any():
        return False

    ys, xs = np.where(union)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    cleaned = data.copy()
    out_bbox = np.ones_like(mask, dtype=bool)
    out_bbox[y0:y1 + 1, x0:x1 + 1] = False
    cleaned[out_bbox] = 0
    profile["nodata"] = 0
    profile["compress"] = profile.get("compress", "deflate")

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(cleaned, 1)

    kept_px = int((cleaned > 0).sum())
    dropped_px = int(mask.sum() - (union & mask).sum())
    print(f"  [clean_raw] {os.path.basename(in_path)}: bbox "
          f"[{x0},{y0}]-[{x1},{y1}] ({x1-x0+1}x{y1-y0+1}); "
          f"kept {kept_px:,} valid, dropped {dropped_px:,} "
          f"isolated > {threshold} pixels")
    return True


def is_aft_camera(entity_id: str, camera_name: str) -> bool:
    """Return True if the entity was captured by an Aft-looking camera.

    Aft cameras capture imagery 180-degree-rotated relative to Forward.
    Both the stitching path (``image_mosaic --rotate``) and the per-segment
    ortho path need to compensate for this.

    Parameters
    ----------
    entity_id : str
        USGS entity identifier, e.g. ``"D3C1213-200346A003"`` (KH-9 Aft).
    camera_name : str
        Camera system name, e.g. ``"KH-9"``, ``"KH-4"``, ``"KH-7"``.
    """
    cam = camera_name.upper().replace("-", "")
    eid = entity_id.upper()
    if cam.startswith("KH4"):
        # KH-4 entity: DS1...-<frame>DA<seq> (Aft) vs DF<seq> (Forward)
        return "DA" in eid and "DF" not in eid
    if cam.startswith("KH9"):
        # KH-9 entity: D3C<mission>-<frame>[AF]<seq>
        parts = eid.split("-")
        if len(parts) == 2:
            for ch in parts[1]:
                if ch.isalpha():
                    return ch == "A"
    return False


# ---------------------------------------------------------------------------
# Sample .tsai templates for OpticalBar cameras
# ---------------------------------------------------------------------------

_OPTICALBAR_TEMPLATE = """\
VERSION_4
OPTICAL_BAR
image_size = {width} {height}
image_center = {cx} {cy}
pitch = {pitch}
f = {focal_length}
scan_time = {scan_time}
forward_tilt = {forward_tilt}
iC = {iC}
iR = {iR}
speed = {speed}
mean_earth_radius = 6371000
mean_surface_elevation = {mean_elev}
motion_compensation_factor = {mcf}
scan_dir = {scan_dir}
"""


def _write_sample_tsai(path, width, height, camera_params, corners=None,
                       mean_elev=0.0, altitude_m=None):
    """Write a sample OpticalBar .tsai file for cam_gen.

    corners, if provided, is a dict with 'nw','ne','se','sw' keys of (lat, lon)
    tuples, used to compute an approximate initial camera position in ECEF.

    altitude_m : float, optional
        Orbital altitude (metres above WGS84 ellipsoid) used to seed the ECEF
        iC position. Defaults to :data:`NOMINAL_ALTITUDE_M` (170 km). Supply a
        per-mission TLE-derived altitude for much tighter cam_gen convergence
        on off-nominal orbits (KH-4B perigee 154 km / apogee 276 km, etc.).
    """
    import math

    # Compute approximate ECEF position from corner center at ~170km altitude
    if corners:
        lats = [corners[k][0] for k in ('nw', 'ne', 'se', 'sw')]
        lons = [corners[k][1] for k in ('nw', 'ne', 'se', 'sw')]
        lat_c = math.radians(sum(lats) / 4)
        lon_c = math.radians(sum(lons) / 4)
    else:
        lat_c, lon_c = 0.0, 0.0

    alt = float(altitude_m) if altitude_m is not None else float(NOMINAL_ALTITUDE_M)
    R_earth = 6371000
    r = R_earth + alt
    iC_x = r * math.cos(lat_c) * math.cos(lon_c)
    iC_y = r * math.cos(lat_c) * math.sin(lon_c)
    iC_z = r * math.sin(lat_c)
    iC_str = f"{iC_x:.4f} {iC_y:.4f} {iC_z:.4f}"
    iR_str = "1 0 0 0 1 0 0 0 1"  # identity — cam_gen refines this

    with open(path, "w") as f:
        f.write(_OPTICALBAR_TEMPLATE.format(
            width=width,
            height=height,
            cx=width / 2.0,
            cy=height / 2.0,
            pitch=camera_params["pixel_pitch"],
            focal_length=camera_params["focal_length"],
            scan_time=camera_params["scan_time"],
            forward_tilt=camera_params.get("forward_tilt", 0.0),
            iC=iC_str,
            iR=iR_str,
            speed=camera_params["speed"],
            mean_elev=mean_elev,
            mcf=camera_params.get("motion_compensation_factor", 1.0),
            scan_dir=camera_params.get("scan_dir", "right"),
        ))


def generate_camera(image_path, camera_params, corners, dem_path=None,
                    altitude_m=None):
    """Generate an OpticalBar camera model (.tsai) from USGS corner coordinates.

    Parameters
    ----------
    image_path : str
        Path to the stitched/georeferenced image.
    camera_params : dict
        Camera intrinsics: focal_length, pixel_pitch, scan_time, speed,
        forward_tilt, scan_dir, motion_compensation_factor.
    corners : dict or list
        Corner coordinates as {'nw': (lat, lon), 'ne': ..., 'se': ..., 'sw': ...}
        or as a flat list [NW_lon, NW_lat, NE_lon, NE_lat, SE_lon, SE_lat, SW_lon, SW_lat].
    dem_path : str, optional
        Path to a DEM for refining the camera model.
    altitude_m : float, optional
        Per-mission orbital altitude seed for cam_gen. Defaults to the 170 km
        nominal. See :func:`_write_sample_tsai`.

    Returns
    -------
    str or None
        Path to generated .tsai camera file, or None if ASP unavailable.
    """
    cam_gen = find_asp_tool("cam_gen")
    if cam_gen is None:
        print("  [camera_model] ASP cam_gen not found, skipping OpticalBar correction")
        return None

    # Parse corners into lon-lat string for cam_gen
    if isinstance(corners, dict):
        corners = {str(k).lower(): v for k, v in corners.items()}
        required = ("nw", "ne", "se", "sw")
        if any(k not in corners for k in required):
            print(f"  [camera_model] Missing corner keys: need {required}, got {sorted(corners.keys())}")
            return None
        ll_str = (f"{corners['nw'][1]} {corners['nw'][0]} "
                  f"{corners['ne'][1]} {corners['ne'][0]} "
                  f"{corners['se'][1]} {corners['se'][0]} "
                  f"{corners['sw'][1]} {corners['sw'][0]}")
    elif isinstance(corners, (list, tuple)) and len(corners) == 8:
        ll_str = " ".join(str(c) for c in corners)
    else:
        print(f"  [camera_model] Invalid corners format: {type(corners)}")
        return None

    # Get image dimensions
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(image_path)
    if ds is None:
        print(f"  [camera_model] Cannot open {image_path}")
        return None
    width, height = ds.RasterXSize, ds.RasterYSize
    ds = None

    # Write sample .tsai file
    output_dir = os.path.dirname(image_path)
    sample_tsai = os.path.join(output_dir, "sample_opticalbar.tsai")
    # Pass corners (lowercase keys) for initial position estimate
    corners_lc = None
    if isinstance(corners, dict):
        corners_lc = {k.lower(): v for k, v in corners.items()}
    _write_sample_tsai(sample_tsai, width, height, camera_params,
                       corners=corners_lc, altitude_m=altitude_m)

    # Run cam_gen
    output_tsai = os.path.splitext(image_path)[0] + ".tsai"
    ok = _run_cam_gen_subprocess(
        cam_gen, sample_tsai, ll_str, image_path, output_tsai, dem_path,
    )
    if not ok:
        return None

    print(f"  [camera_model] Camera model written to {os.path.basename(output_tsai)}")
    # Clean up sample file
    if os.path.isfile(sample_tsai):
        os.remove(sample_tsai)

    return output_tsai


def _run_cam_gen_subprocess(cam_gen, sample_tsai, ll_str, image_path,
                             output_tsai, dem_path, pixel_str=None,
                             timeout_s=None):
    """Invoke ASP cam_gen for an OpticalBar camera.

    Shared between the legacy single-camera ``generate_camera`` path,
    the per-sub-frame ``cam_gen_opticalbar_per_subframe`` helper, and
    the Phase 4 joint-BA seed emitter (which passes ``pixel_str`` so
    cam_gen runs an N-point LSQ instead of a 4-corner solve). All paths
    share the same ISISROOT / refine-camera / error-handling semantics.

    Parameters
    ----------
    cam_gen, sample_tsai, ll_str, image_path, output_tsai, dem_path
        See :func:`generate_camera`.
    pixel_str : str, optional
        Space-separated ``"col0 row0 col1 row1 …"`` ASP passes via
        ``--pixel-values``. When supplied, cam_gen treats
        ``--lon-lat-values`` / ``--pixel-values`` as paired
        correspondences for LSQ rather than 4 assumed corners.
    timeout_s : float, optional
        Subprocess timeout (seconds). Defaults to 120 s for the 4-corner
        path; 300 s when ``pixel_str`` is supplied (Ceres LSQ over
        N > 4 points is slower).

    Returns True on success (``output_tsai`` on disk), False otherwise.
    """
    cmd = [
        cam_gen,
        "--sample-file", sample_tsai,
        "--camera-type", "opticalbar",
        "--lon-lat-values", ll_str,
    ]
    if pixel_str:
        cmd.extend(["--pixel-values", pixel_str])
    cmd.extend([
        image_path,
        "-o", output_tsai,
    ])
    if dem_path and os.path.isfile(dem_path):
        cmd.extend(["--reference-dem", dem_path, "--refine-camera"])

    if timeout_s is None:
        timeout_s = 300.0 if pixel_str else 120.0

    # See mapproject_image() for the rationale — set ISISROOT defensively so
    # subprocess inherits a workable environment regardless of the user's shell.
    env = os.environ.copy()
    if "ISISROOT" not in env:
        asp_root = os.path.dirname(os.path.dirname(cam_gen))
        if os.path.isfile(os.path.join(asp_root, "IsisPreferences")):
            env["ISISROOT"] = asp_root

    print(f"  [camera_model] Running cam_gen for {os.path.basename(image_path)}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, env=env,
        )
        if result.returncode != 0:
            print(f"  [camera_model] cam_gen failed: {result.stderr[:500]}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  [camera_model] cam_gen error: {e}")
        return False

    if not os.path.isfile(output_tsai):
        print(f"  [camera_model] cam_gen did not produce {output_tsai}")
        return False
    return True


# ---------------------------------------------------------------------------
# OpticalBar .tsai parsing and per-sub-frame altitude derivation
# ---------------------------------------------------------------------------

# WGS84 ellipsoid constants (matches pyproj's EPSG:4326 datum).
_WGS84_A = 6378137.0
_WGS84_F_INV = 298.257223563
_WGS84_B = _WGS84_A * (1.0 - 1.0 / _WGS84_F_INV)


def _wgs84_radius_at_lat(lat_rad: float) -> float:
    """Meridional radius of the WGS84 ellipsoid at geodetic latitude (rad).

    Used to convert ECEF camera-centre magnitude into altitude-above-
    ellipsoid without instantiating pyproj. Matches ``_WGS84_A`` at the
    equator and ``_WGS84_B`` at the poles.
    """
    import math
    cos_lat = math.cos(lat_rad)
    sin_lat = math.sin(lat_rad)
    a2_c = (_WGS84_A * cos_lat) ** 2
    b2_s = (_WGS84_B * sin_lat) ** 2
    a4_c = _WGS84_A ** 2 * cos_lat ** 2
    b4_s = _WGS84_B ** 2 * sin_lat ** 2
    num = (_WGS84_A ** 2 * a2_c) + (_WGS84_B ** 2 * b2_s)
    den = a4_c + b4_s
    return (num / den) ** 0.5 if den else _WGS84_A


def _parse_opticalbar_tsai(path: str) -> dict | None:
    """Parse an OpticalBar ``.tsai`` camera file produced by ASP cam_gen.

    Returns a dict with keys ``image_size`` (2,), ``image_center`` (2,),
    ``pitch``, ``f``, ``scan_time``, ``forward_tilt``, ``iC`` (3,),
    ``iR`` (3,3), ``speed``, ``mean_earth_radius``, ``mean_surface_elevation``,
    ``motion_compensation_factor``, ``scan_dir``. Returns ``None`` on
    parse failure.
    """
    import numpy as np
    if not os.path.isfile(path):
        return None
    out: dict = {}
    try:
        with open(path, "r") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or "=" not in line:
                    continue
                key, val = [s.strip() for s in line.split("=", 1)]
                key_lc = key.lower()
                if key_lc in ("image_size", "image_center"):
                    out[key_lc] = tuple(float(x) for x in val.split())
                elif key_lc == "ic":
                    out["iC"] = np.array([float(x) for x in val.split()],
                                          dtype=np.float64)
                elif key_lc == "ir":
                    vals = [float(x) for x in val.split()]
                    if len(vals) != 9:
                        return None
                    out["iR"] = np.array(vals, dtype=np.float64).reshape(3, 3)
                elif key_lc == "scan_dir":
                    out["scan_dir"] = val.strip()
                else:
                    try:
                        out[key_lc] = float(val)
                    except ValueError:
                        out[key_lc] = val
    except OSError:
        return None
    if "iC" not in out or "iR" not in out:
        return None
    return out


def _ecef_to_geodetic(iC):
    """ECEF (X, Y, Z) → (lat_rad, lon_rad, altitude_m) on WGS84.

    Closed-form iterative solution (Bowring 1976); converges in ~3 iters
    at orbital altitudes. Matches pyproj within mm.
    """
    import math
    X, Y, Z = float(iC[0]), float(iC[1]), float(iC[2])
    a, b = _WGS84_A, _WGS84_B
    e2 = 1.0 - (b / a) ** 2
    lon = math.atan2(Y, X)
    p = math.hypot(X, Y)
    # Bowring's parametric latitude initial guess.
    theta = math.atan2(Z * a, p * b)
    sin_theta, cos_theta = math.sin(theta), math.cos(theta)
    ep2 = (a * a - b * b) / (b * b)
    lat = math.atan2(Z + ep2 * b * sin_theta ** 3,
                     p - e2 * a * cos_theta ** 3)
    # One Newton iteration for extra precision.
    for _ in range(3):
        sin_lat = math.sin(lat)
        N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        alt = p / math.cos(lat) - N
        lat_new = math.atan2(Z, p * (1.0 - e2 * N / (N + alt)))
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new
    sin_lat = math.sin(lat)
    N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - N
    return lat, lon, alt


def cam_gen_opticalbar_per_subframe(sub_path, corners, camera_params,
                                     dem_path=None, output_tsai=None,
                                     overwrite=False, altitude_m=None):
    """Run ASP cam_gen on a single sub-frame and return its OpticalBar pose.

    Unlike :func:`generate_camera`, which returns only the .tsai path for
    downstream mapproject, this helper also parses the .tsai and derives
    the altitude above the WGS84 ellipsoid. The caller feeds that altitude
    into the per-segment 14-parameter LM fit so the `f` / `Zs0` gauge
    freedom is broken (observed 30 % `f` collapse on Bahrain KH-9 seg00
    when `Zs0` was pinned at a wrong 170 km nominal).

    Parameters
    ----------
    sub_path : str
        Path to the sub-frame raster.
    corners : dict
        {'NW': (lat, lon), 'NE': ..., 'SE': ..., 'SW': ...} — the
        geographic corners passed to cam_gen as --lon-lat-values.
    camera_params : dict
        Profile camera intrinsics (focal_length, pixel_pitch, scan_time,
        speed, forward_tilt, scan_dir, motion_compensation_factor).
    dem_path : str, optional
        DEM for cam_gen's --refine-camera.
    output_tsai : str, optional
        Where to write the output .tsai. Defaults to
        ``<sub_path_stem>_opticalbar.tsai``.
    overwrite : bool
        If False (default) and ``output_tsai`` exists, skip cam_gen and
        re-parse the cached file.

    Returns
    -------
    dict or None
        ``{tsai_path, parsed (full .tsai dict), iC, iR, altitude_m,
        lat_rad, lon_rad, focal_length, scan_time}`` on success. ``None``
        when ASP is unavailable or cam_gen fails.
    """
    import math
    cam_gen = find_asp_tool("cam_gen")
    if cam_gen is None:
        return None

    # Normalise corner keys to lowercase for cam_gen's lon/lat formatter.
    corners_lc = {str(k).lower(): v for k, v in corners.items()}
    required = ("nw", "ne", "se", "sw")
    if any(k not in corners_lc for k in required):
        return None
    ll_str = (
        f"{corners_lc['nw'][1]} {corners_lc['nw'][0]} "
        f"{corners_lc['ne'][1]} {corners_lc['ne'][0]} "
        f"{corners_lc['se'][1]} {corners_lc['se'][0]} "
        f"{corners_lc['sw'][1]} {corners_lc['sw'][0]}"
    )

    # Image dimensions from rasterio (cheaper than GDAL for a header read).
    import rasterio
    with rasterio.open(sub_path) as src:
        width, height = src.width, src.height

    if output_tsai is None:
        output_tsai = os.path.splitext(sub_path)[0] + "_opticalbar.tsai"
    # Caching: skip the subprocess if the output already exists and the
    # caller didn't ask for an overwrite.
    if not overwrite and os.path.isfile(output_tsai):
        parsed = _parse_opticalbar_tsai(output_tsai)
        if parsed is not None:
            lat_rad, lon_rad, alt_m = _ecef_to_geodetic(parsed["iC"])
            return {
                "tsai_path": output_tsai,
                "parsed": parsed,
                "iC": parsed["iC"],
                "iR": parsed["iR"],
                "altitude_m": float(alt_m),
                "lat_rad": float(lat_rad),
                "lon_rad": float(lon_rad),
                "focal_length": float(parsed.get("f", camera_params["focal_length"])),
                "scan_time": float(parsed.get("scan_time", camera_params["scan_time"])),
            }

    sample_tsai = os.path.splitext(output_tsai)[0] + ".sample.tsai"
    _write_sample_tsai(
        sample_tsai, width, height, camera_params, corners=corners_lc,
        altitude_m=altitude_m,
    )
    try:
        ok = _run_cam_gen_subprocess(
            cam_gen, sample_tsai, ll_str, sub_path, output_tsai, dem_path,
        )
        if not ok:
            return None
    finally:
        if os.path.isfile(sample_tsai):
            os.remove(sample_tsai)

    parsed = _parse_opticalbar_tsai(output_tsai)
    if parsed is None:
        return None

    lat_rad, lon_rad, alt_m = _ecef_to_geodetic(parsed["iC"])
    return {
        "tsai_path": output_tsai,
        "parsed": parsed,
        "iC": parsed["iC"],
        "iR": parsed["iR"],
        "altitude_m": float(alt_m),
        "lat_rad": float(lat_rad),
        "lon_rad": float(lon_rad),
        "focal_length": float(parsed.get("f", camera_params["focal_length"])),
        "scan_time": float(parsed.get("scan_time", camera_params["scan_time"])),
    }


def mapproject_image(image_path, camera_path, dem_path=None, output_path=None,
                     resolution=None, t_srs="EPSG:3857"):
    """Map-project an image using its OpticalBar camera model.

    Removes panoramic distortion and (with DEM) terrain parallax.

    Parameters
    ----------
    image_path : str
        Input image.
    camera_path : str
        .tsai camera file from generate_camera().
    dem_path : str, optional
        DEM for terrain correction. If None, uses WGS84 datum (flat earth,
        still corrects panoramic distortion).
    output_path : str, optional
        Output path. Defaults to image_path with '_mapprojected' suffix.
    resolution : float, optional
        Output resolution in metres/pixel. If None, auto-detected.
    t_srs : str
        Output spatial reference system.

    Returns
    -------
    str or None
        Path to mapprojected image, or None on failure.

    Notes
    -----
    ASP ships ``bin/mapproject`` as a Python wrapper that calls
    ``libexec/mapproject_single --parse-options`` to discover available
    flags before running the binary. In the 3.6.0 build at
    StereoPipeline-3.6.0-2025-12-26-arm64-OSX, ``mapproject_single
    --parse-options`` always exits with code 1 even though it prints the
    expected help text, which causes the wrapper's
    ``run_and_parse_output`` to raise. To sidestep this we call
    ``mapproject_single`` directly. The binary expects positional args
    in the order: ``[options...] <dem_or_datum> <image> <camera> <output>``.
    """
    mapproj = find_asp_tool("mapproject")
    if mapproj is None:
        print("  [camera_model] ASP mapproject not found, skipping")
        return None

    # Locate the mapproject_single binary; bin/mapproject is the shell
    # wrapper, libexec/mapproject_single is the actual executable.
    asp_root = os.path.dirname(os.path.dirname(mapproj))
    mapproj_single = os.path.join(asp_root, "libexec", "mapproject_single")
    if not os.path.isfile(mapproj_single):
        # Older / unusual layouts: fall back to whatever find_asp_tool gave us.
        mapproj_single = mapproj

    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_mapprojected{ext}"

    if os.path.isfile(output_path):
        print(f"  [camera_model] Mapprojected file already exists: {os.path.basename(output_path)}")
        return output_path

    cmd = [mapproj_single]
    if resolution is not None:
        cmd.extend(["--tr", str(resolution)])
    cmd.extend(["--t_srs", t_srs])

    # mapproject_single takes the DEM (or a datum keyword) as the first
    # positional argument, followed by image, camera, output.
    if dem_path and os.path.isfile(dem_path):
        cmd.append(dem_path)
    else:
        cmd.append("WGS84")

    cmd.extend([image_path, camera_path, output_path])

    # ASP mapproject_single instantiates ISIS preferences even for Earth
    # imagery; without ISISROOT pointing at the install dir (where the
    # bundled `IsisPreferences` file lives), libc++abi terminates with
    # "USER ERROR The preference file $ISISROOT/IsisPreferences was not
    # found". The bin/mapproject shell wrapper exports ISISROOT, but we
    # call mapproject_single directly so we have to set it ourselves.
    env = os.environ.copy()
    if "ISISROOT" not in env and os.path.isfile(os.path.join(asp_root, "IsisPreferences")):
        env["ISISROOT"] = asp_root

    print(f"  [camera_model] Running mapproject -> {os.path.basename(output_path)}")
    print(f"  [camera_model] cmd: {' '.join(cmd)}")
    try:
        # Timeout scales with pixel count: native res (0.87m) needs ~30 min
        # per segment, downsampled (3.47m) ~5 min.  Use 2× expected time.
        _timeout = 7200 if (resolution and resolution < 1.5) else 3600
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_timeout, env=env)
        if result.returncode != 0:
            stderr_tail = (result.stderr or "").strip()[-1500:]
            stdout_tail = (result.stdout or "").strip()[-1500:]
            print(f"  [camera_model] mapproject failed (exit {result.returncode})")
            if stderr_tail:
                print(f"  [camera_model] stderr: {stderr_tail}")
            if stdout_tail:
                print(f"  [camera_model] stdout: {stdout_tail}")
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  [camera_model] mapproject error: {e}")
        return None

    if not os.path.isfile(output_path):
        print(f"  [camera_model] mapproject did not produce output")
        return None

    print(f"  [camera_model] Mapprojected image: {os.path.basename(output_path)}")
    return output_path


def opticalbar_precorrect(image_path, camera_params, corners, dem_path=None,
                          resolution=None, t_srs="EPSG:3857"):
    """One-call convenience: generate camera model and mapproject.

    Returns path to the corrected image, or None if ASP unavailable
    (caller should fall back to standard pipeline).
    """
    camera_path = generate_camera(image_path, camera_params, corners, dem_path)
    if camera_path is None:
        return None

    output = mapproject_image(image_path, camera_path, dem_path,
                              resolution=resolution, t_srs=t_srs)
    return output


# ---------------------------------------------------------------------------
# Per-segment (sub-image) processing — 2OC §3.1
# ---------------------------------------------------------------------------

def interpolate_segment_corners(strip_corners, n_segments, segment_idx):
    """Slice a strip's 4 corners into one segment's 4 corners.

    The 2OC paper (Hou et al. 2023, §3.1) processes each USGS sub-image
    (a, b, c, d for KH-4; a..g for KH-9 PC) independently rather than
    stitching first. To do that we need per-segment corner coordinates,
    but USGS only ships 4 corners for the entire scene. Linear
    interpolation along the scan axis is the natural approximation:

        seg_NW(i) = lerp(strip_NW, strip_NE, i / N)
        seg_NE(i) = lerp(strip_NW, strip_NE, (i+1) / N)
        seg_SW(i) = lerp(strip_SW, strip_SE, i / N)
        seg_SE(i) = lerp(strip_SW, strip_SE, (i+1) / N)

    Assumes:
    - The scan axis is horizontal in the stitched image (true for KH-4/KH-9 PC).
    - Sub-frames are equal-width and laid out left-to-right in stitch order.
    - Caller passes ``segment_idx`` in stitch order, **not** USGS delivery
      order. For KH-4 the delivery order (a, b, c, d) is reversed when
      stitched, so segment 0 is sub-frame "d", segment 3 is "a".

    Parameters
    ----------
    strip_corners : dict
        ``{'NW': (lat, lon), 'NE': ..., 'SE': ..., 'SW': ...}`` (or lowercase keys).
    n_segments : int
        Total number of segments in the strip.
    segment_idx : int
        Zero-indexed segment position in stitch order.

    Returns
    -------
    dict
        Per-segment ``{'NW': (lat, lon), 'NE': ..., 'SE': ..., 'SW': ...}`` —
        same key case as the input dict.
    """
    if n_segments <= 0:
        raise ValueError(f"n_segments must be positive, got {n_segments}")
    if not (0 <= segment_idx < n_segments):
        raise ValueError(f"segment_idx {segment_idx} out of range [0, {n_segments})")

    # Detect key case and normalise to lowercase for the math.
    keys_in = list(strip_corners.keys())
    upper_case = any(k.isupper() for k in keys_in)
    cs = {str(k).lower(): tuple(v) for k, v in strip_corners.items()}
    for k in ("nw", "ne", "se", "sw"):
        if k not in cs:
            raise ValueError(f"strip_corners missing key {k!r}; got {sorted(cs.keys())}")

    left = segment_idx / n_segments
    right = (segment_idx + 1) / n_segments

    def _lerp(a, b, t):
        return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))

    seg = {
        "nw": _lerp(cs["nw"], cs["ne"], left),
        "ne": _lerp(cs["nw"], cs["ne"], right),
        "se": _lerp(cs["sw"], cs["se"], right),
        "sw": _lerp(cs["sw"], cs["se"], left),
    }
    if upper_case:
        seg = {k.upper(): v for k, v in seg.items()}
    return seg


def _compute_segment_clip_slabs(all_seg_corners, t_srs: str):
    """Return a list of (west, south, east, north) clip bounds in *t_srs*.

    The strip runs diagonally, so each segment's bounding box overlaps its
    neighbours'.  To eliminate overlap we compute non-overlapping x-coordinate
    slabs: the shared boundary between seg *i* and seg *i+1* is placed at the
    mean x of the two shared corners (NE_i ≡ NW_{i+1}, SE_i ≡ SW_{i+1}).
    The y range uses the full extent of each segment.

    Returns a list of (west, south, east, north) per segment, or None on failure.
    """
    try:
        from pyproj import Transformer
    except ImportError:
        return None

    n = len(all_seg_corners)
    if n == 0:
        return None

    tr = None
    if t_srs.upper() != "EPSG:4326":
        tr = Transformer.from_crs("EPSG:4326", t_srs, always_xy=True)

    def _to_xy(lat, lon):
        if tr is None:
            return (lon, lat)
        return tr.transform(lon, lat)

    # Project all corners to target CRS.
    projected = []  # list of dicts: {nw: (x,y), ne: (x,y), se: (x,y), sw: (x,y)}
    for corners in all_seg_corners:
        lc = {str(k).lower(): v for k, v in corners.items()}
        d = {}
        for k in ("nw", "ne", "se", "sw"):
            d[k] = _to_xy(lc[k][0], lc[k][1])
        projected.append(d)

    # Compute x-slab boundaries from shared corners.
    # Boundary between seg i and seg i+1 is the mean x of the NE/SE of seg i
    # (which equal the NW/SW of seg i+1 by interpolation).
    cuts = []  # n-1 cut x-coordinates
    for i in range(n - 1):
        ne_x = projected[i]["ne"][0]
        se_x = projected[i]["se"][0]
        cuts.append((ne_x + se_x) / 2.0)

    slabs = []
    for i in range(n):
        p = projected[i]
        all_y = [p[k][1] for k in ("nw", "ne", "se", "sw")]
        south, north = min(all_y), max(all_y)
        west = cuts[i - 1] if i > 0 else min(p["nw"][0], p["sw"][0])
        east = cuts[i] if i < n - 1 else max(p["ne"][0], p["se"][0])
        slabs.append((west, south, east, north))

    return slabs


def _validate_segment_seams(seg_orthos: list) -> bool:
    """Check that adjacent segment orthos have consistent content at seams.

    Computes ZNCC on the geographic overlap between consecutive segments.
    A 180-degree-flipped segment will show near-zero or negative correlation
    against its correctly-oriented neighbour.

    Returns True if all seams pass (or there are fewer than 2 segments).
    """
    if len(seg_orthos) < 2:
        return True

    try:
        import rasterio
    except ImportError:
        return True  # can't validate without rasterio

    ok = True
    for i in range(len(seg_orthos) - 1):
        try:
            with rasterio.open(seg_orthos[i]) as sa, rasterio.open(seg_orthos[i + 1]) as sb:
                # Geographic overlap
                ol = max(sa.bounds.left, sb.bounds.left)
                ob = max(sa.bounds.bottom, sb.bounds.bottom)
                or_ = min(sa.bounds.right, sb.bounds.right)
                ot = min(sa.bounds.top, sb.bounds.top)
                if ol >= or_ or ob >= ot:
                    gap_w = ol - or_
                    gap_h = ob - ot
                    print(f"  [per_segment] WARNING: seam {i}–{i+1} NO OVERLAP "
                          f"(gap_w={gap_w:.0f}m gap_h={gap_h:.0f}m)")
                    ok = False
                    continue

                from rasterio.windows import from_bounds
                wa = from_bounds(ol, ob, or_, ot, sa.transform)
                wb = from_bounds(ol, ob, or_, ot, sb.transform)
                pa = sa.read(1, window=wa).astype(np.float64)
                pb = sb.read(1, window=wb).astype(np.float64)

                h = min(pa.shape[0], pb.shape[0])
                w = min(pa.shape[1], pb.shape[1])
                if h < 4 or w < 4:
                    print(f"  [per_segment] seam {i}–{i+1} overlap too small "
                          f"({w}x{h}px), skipped")
                    continue
                pa, pb = pa[:h, :w], pb[:h, :w]

                nodata_a = sa.nodata if sa.nodata is not None else -32768
                nodata_b = sb.nodata if sb.nodata is not None else -32768
                valid = (pa != nodata_a) & (pb != nodata_b)
                if valid.sum() < 100:
                    print(f"  [per_segment] seam {i}–{i+1} no valid overlap "
                          f"pixels (both sides nodata)")
                    continue

                a_v = pa[valid] - pa[valid].mean()
                b_v = pb[valid] - pb[valid].mean()
                denom = np.sqrt((a_v ** 2).sum() * (b_v ** 2).sum())
                ncc = float((a_v * b_v).sum() / denom) if denom > 0 else 0.0

                if ncc < 0.1:
                    print(f"  [per_segment] WARNING: seam {i}–{i+1} NCC={ncc:.3f} "
                          f"(expected >0.1 — possible orientation mismatch)")
                    ok = False
                else:
                    print(f"  [per_segment] seam {i}–{i+1} NCC={ncc:.3f} OK")
        except Exception as e:
            print(f"  [per_segment] seam {i}–{i+1} validation skipped: {e}")

    return ok


def _measure_segment_seams(seg_orthos: list) -> list[dict]:
    """Return per-seam ZNCC and phase-correlation metrics."""
    if len(seg_orthos) < 2:
        return []

    try:
        import cv2
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError:
        return []

    reports = []
    for i in range(len(seg_orthos) - 1):
        try:
            with rasterio.open(seg_orthos[i]) as sa, rasterio.open(seg_orthos[i + 1]) as sb:
                ol = max(sa.bounds.left, sb.bounds.left)
                ob = max(sa.bounds.bottom, sb.bounds.bottom)
                or_ = min(sa.bounds.right, sb.bounds.right)
                ot = min(sa.bounds.top, sb.bounds.top)
                if ol >= or_ or ob >= ot:
                    reports.append({
                        "index": f"{i}-{i+1}",
                        "status": "no_overlap",
                    })
                    continue

                wa = from_bounds(ol, ob, or_, ot, sa.transform)
                wb = from_bounds(ol, ob, or_, ot, sb.transform)
                pa = sa.read(1, window=wa).astype(np.float64)
                pb = sb.read(1, window=wb).astype(np.float64)
                nd_a = sa.nodata if sa.nodata is not None else -32768
                nd_b = sb.nodata if sb.nodata is not None else -32768

            h = min(pa.shape[0], pb.shape[0])
            w = min(pa.shape[1], pb.shape[1])
            if h < 16 or w < 16:
                reports.append({
                    "index": f"{i}-{i+1}",
                    "status": "small_overlap",
                })
                continue

            pa = pa[:h, :w]
            pb = pb[:h, :w]
            valid = (
                np.isfinite(pa) & np.isfinite(pb)
                & (pa != nd_a) & (pb != nd_b)
            )
            if valid.sum() < 100:
                reports.append({
                    "index": f"{i}-{i+1}",
                    "status": "no_valid_overlap",
                })
                continue

            a_v = pa[valid] - pa[valid].mean()
            b_v = pb[valid] - pb[valid].mean()
            denom = np.sqrt((a_v ** 2).sum() * (b_v ** 2).sum())
            raw_zncc = float((a_v * b_v).sum() / denom) if denom > 0 else 0.0

            lo = min(float(pa[valid].min()), float(pb[valid].min()))
            hi = max(float(pa[valid].max()), float(pb[valid].max()))
            pa_u8 = np.zeros_like(pa, dtype=np.uint8)
            pb_u8 = np.zeros_like(pb, dtype=np.uint8)
            if hi > lo:
                scale = 255.0 / (hi - lo)
                pa_u8[valid] = np.clip((pa[valid] - lo) * scale, 0, 255).astype(np.uint8)
                pb_u8[valid] = np.clip((pb[valid] - lo) * scale, 0, 255).astype(np.uint8)
            window = cv2.createHanningWindow((w, h), cv2.CV_32F)
            (dx, dy), response = cv2.phaseCorrelate(
                pa_u8.astype(np.float32),
                pb_u8.astype(np.float32),
                window,
            )

            # Evaluate seam similarity after compensating for the measured
            # phase shift. This avoids rejecting a seam that is geometrically
            # much better but still scores poorly on raw overlap correlation.
            M = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
            pb_shift = cv2.warpAffine(
                pb.astype(np.float32),
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=np.nan,
            )
            valid_b = (np.isfinite(pb) & (pb != nd_b)).astype(np.uint8)
            valid_b_shift = cv2.warpAffine(
                valid_b,
                M,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ) > 0
            aligned_valid = (
                np.isfinite(pa)
                & (pa != nd_a)
                & np.isfinite(pb_shift)
                & valid_b_shift
            )
            if aligned_valid.sum() >= 100:
                a_v = pa[aligned_valid] - pa[aligned_valid].mean()
                b_v = pb_shift[aligned_valid] - pb_shift[aligned_valid].mean()
                denom = np.sqrt((a_v ** 2).sum() * (b_v ** 2).sum())
                zncc = float((a_v * b_v).sum() / denom) if denom > 0 else raw_zncc
            else:
                zncc = raw_zncc

            # Phase 3e: detect low-texture / low-signal overlaps that
            # carry insufficient content to measure seam quality
            # reliably (typical of cloud, ocean, or uniform desert).
            # Both the phase-correlation response and |ZNCC| must be
            # below their signal floors — if either signal is strong
            # the seam is measurable (whether the content matches or
            # not). Callers treat ``low_texture`` like
            # ``no_valid_overlap`` — the seam is skipped in QA rather
            # than counted as a failure. ASP's whole-strip approach
            # sidesteps this by not having seams at all; our per-
            # segment path needs an equivalent escape hatch or
            # otherwise-good geometry fails QA on uniform-content
            # regions.
            _LOW_RESPONSE = 0.003
            _LOW_ZNCC_ABS = 0.15
            is_low_texture = (
                float(response) < _LOW_RESPONSE
                and abs(float(zncc)) < _LOW_ZNCC_ABS
            )

            reports.append({
                "index": f"{i}-{i+1}",
                "status": "low_texture" if is_low_texture else "ok",
                "zncc": float(zncc),
                "raw_zncc": float(raw_zncc),
                "phase_shift_px": float(np.hypot(dx, dy)),
                "response": float(response),
            })
        except Exception as e:
            reports.append({
                "index": f"{i}-{i+1}",
                "status": f"error:{e}",
            })
    return reports


def _seam_report_passes(report: dict | None, seam_shift_px_max: float) -> bool:
    """Return True when a seam report satisfies the production QA gate.

    Phase 3e: ``low_texture`` reports pass the QA gate because the
    overlap's signal is too weak to measure — the seam may be fine,
    it's just unmeasurable. ASP's whole-strip approach doesn't have
    this problem (no seams); we treat low_texture identically.

    Phase 3f: add a sub-pixel-alignment accept rule. When the phase
    shift is sub-2-px (tight geometry) AND ZNCC ≥ 0.3 (real content
    correlation, imperfect but not noise), the seam is demonstrably
    good even if neither the 0.4 ZNCC gate nor the 0.005 response
    gate crosses. Motivated by cross-temporal imagery where real
    terrain change (coastline reclamation, urban buildup, land-use
    change) depresses ZNCC below 0.4 despite correct geometry.
    """
    if report is None:
        return False
    status = report.get("status")
    if status == "low_texture":
        return True
    if status != "ok":
        return False
    phase_shift = float(report.get("phase_shift_px", np.inf))
    if phase_shift > float(seam_shift_px_max):
        return False
    zncc = float(report.get("zncc", -1.0))
    if zncc >= 0.4:
        return True
    if float(report.get("response", float("-inf"))) >= 0.005:
        return True
    # Phase 3f sub-pixel-accept: tight geometry + real (not noise)
    # correlation. Requires BOTH conditions — sub-pixel alone on
    # low-texture would be caught by Phase 3e above; ZNCC 0.3+ alone
    # on a large shift means good content far from alignment, still
    # a failure.
    if phase_shift < 2.0 and zncc >= 0.3:
        return True
    return False


def _seam_report_score(report: dict | None) -> tuple:
    """Rank seam reports so candidate re-fits can be compared cheaply."""
    if not report:
        return (-1, float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    status = str(report.get("status", ""))
    if status == "ok":
        zncc = float(report.get("zncc", float("-inf")))
        shift = float(report.get("phase_shift_px", np.inf))
        return (
            3,
            int(zncc >= 0.4),
            -shift,
            zncc,
            float(report.get("response", float("-inf"))),
        )
    if status == "low_texture":
        # Rank ``low_texture`` below ``ok`` (a measurable seam is
        # preferable to an unmeasurable one) but above the overlap-
        # geometry failures, which are worse outcomes.
        return (2, float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    if status == "small_overlap":
        return (2, float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    if status == "no_valid_overlap":
        return (1, float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    if status == "no_overlap":
        return (0, float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    return (-1, float("-inf"), float("-inf"), float("-inf"), float("-inf"))


def _refine_segment_corners_via_reference(seg_ortho_path: str, reference_path: str,
                                          corners: dict) -> dict | None:
    """Match a segment ortho against the reference to measure offset, return adjusted corners.

    The first-pass cam_gen uses USGS corners (which can be ~10 km off).
    By matching the first-pass ortho against the reference, we measure
    the actual offset and shift the corners to correct it.  A second
    cam_gen + mapproject with the corrected corners produces a more
    accurately positioned ortho — the 2OC §4.4 model-guided approach.

    Returns adjusted corners dict, or None if matching fails.
    """
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        import cv2
    except ImportError:
        return None

    with rasterio.open(seg_ortho_path) as seg:
        seg_crs = seg.crs
        seg_bounds = seg.bounds
        seg_res = seg.res[0]
        seg_nd = seg.nodata if seg.nodata is not None else -32768

    with rasterio.open(reference_path) as ref:
        ref_crs = ref.crs
        ref_bounds = ref.bounds

    # Compute overlap in a common metric CRS (UTM).
    try:
        from pyproj import Transformer, CRS
        center_lon = (seg_bounds.left + seg_bounds.right) / 2
        center_lat = (seg_bounds.bottom + seg_bounds.top) / 2
        if seg_crs.is_projected:
            # Convert to geographic for UTM zone detection
            tr_geo = Transformer.from_crs(seg_crs, "EPSG:4326", always_xy=True)
            center_lon, center_lat = tr_geo.transform(center_lon, center_lat)
        utm_zone = int((center_lon + 180) / 6) + 1
        utm_epsg = f"EPSG:{32600 + utm_zone}" if center_lat >= 0 else f"EPSG:{32700 + utm_zone}"
    except Exception:
        return None

    # Transform both bounds to UTM.
    seg_utm = transform_bounds(seg_crs, utm_epsg, *seg_bounds)
    ref_utm = transform_bounds(ref_crs, utm_epsg, *ref_bounds)

    # Overlap in UTM.
    ol = max(seg_utm[0], ref_utm[0])
    ob = max(seg_utm[1], ref_utm[1])
    or_ = min(seg_utm[2], ref_utm[2])
    ot = min(seg_utm[3], ref_utm[3])

    if or_ - ol < 1000 or ot - ob < 1000:
        return None  # <1km overlap — too small

    # Read both at ~30m resolution for matching.
    match_res = 30.0
    from rasterio.warp import reproject, Resampling

    w = int((or_ - ol) / match_res)
    h = int((ot - ob) / match_res)
    if w < 50 or h < 50:
        return None

    from rasterio.transform import from_bounds as tfm_from_bounds
    out_tfm = tfm_from_bounds(ol, ob, or_, ot, w, h)

    seg_arr = np.empty((h, w), dtype=np.float32)
    ref_arr = np.empty((h, w), dtype=np.float32)

    with rasterio.open(seg_ortho_path) as seg:
        reproject(seg.read(1), seg_arr, src_transform=seg.transform,
                  src_crs=seg.crs, dst_transform=out_tfm, dst_crs=utm_epsg,
                  resampling=Resampling.bilinear, src_nodata=seg_nd,
                  dst_nodata=0)
    with rasterio.open(reference_path) as ref:
        ref_nd = ref.nodata if ref.nodata is not None else 0
        reproject(ref.read(1), ref_arr, src_transform=ref.transform,
                  src_crs=ref.crs, dst_transform=out_tfm, dst_crs=utm_epsg,
                  resampling=Resampling.bilinear, src_nodata=ref_nd,
                  dst_nodata=0)

    # Normalize to uint8 for NCC.
    def _to_u8(arr):
        valid = arr > 0
        if valid.sum() < 100:
            return None
        mn, mx = arr[valid].min(), arr[valid].max()
        if mx - mn < 1:
            return None
        out = np.zeros_like(arr, dtype=np.uint8)
        out[valid] = np.clip((arr[valid] - mn) / (mx - mn) * 255, 0, 255).astype(np.uint8)
        return out

    s_u8 = _to_u8(seg_arr)
    r_u8 = _to_u8(ref_arr)
    if s_u8 is None or r_u8 is None:
        return None

    # NCC template matching: slide a centre crop of reference over seg.
    margin = min(h, w) // 4
    if margin < 10:
        return None
    tmpl = r_u8[margin:-margin, margin:-margin]
    if tmpl.shape[0] < 20 or tmpl.shape[1] < 20:
        return None

    result = cv2.matchTemplate(s_u8, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < 0.15:
        print(f"    [refine] NCC too low ({max_val:.3f}), skipping corner refinement")
        return None

    dx_px = max_loc[0] - margin
    dy_px = max_loc[1] - margin
    dx_m = dx_px * match_res
    dy_m = dy_px * match_res

    if abs(dx_m) < 10 and abs(dy_m) < 10:
        print(f"    [refine] Offset <10m (dx={dx_m:.0f}m dy={dy_m:.0f}m), no adjustment needed")
        return None

    print(f"    [refine] Measured offset: dx={dx_m:.0f}m dy={dy_m:.0f}m (NCC={max_val:.3f})")

    # Convert metric offset to lat/lon shift.
    # dx > 0 means ortho is EAST of reference → shift corners WEST (subtract lon)
    # dy > 0 means ortho is SOUTH of reference → shift corners NORTH (add lat)
    m_per_deg_lat = 111320.0
    import math
    m_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))
    dlon = -dx_m / m_per_deg_lon
    dlat = dy_m / m_per_deg_lat

    # Apply shift to all 4 corners.
    adjusted = {}
    for k, (lat, lon) in corners.items():
        adjusted[k] = (lat + dlat, lon + dlon)

    print(f"    [refine] Adjusted corners by dlon={dlon:.5f}° dlat={dlat:.5f}°")
    return adjusted


def _measure_per_segment_reference_shifts(active_indices: list,
                                          seg_orthos_map: dict,
                                          reference_path: str,
                                          all_seg_corners: list) -> dict:
    """Measure each segment's (dlat, dlon) offset against the reference.

    Returns ``{seg_idx: (dlat, dlon)}`` for segments where
    ``_refine_segment_corners_via_reference`` returned a usable result.
    Segments with low-confidence NCC are simply absent from the dict —
    the caller should take a robust summary (median) over what's present
    and apply it uniformly to ALL segments so their seam alignment stays
    intact (differential per-segment shifts tear overlapping content
    apart and the distance-weighted blend ghosts two copies of the same
    features).
    """
    import math

    ref_shifts: dict = {}
    print(f"  [per_segment] Pass 2: per-segment reference NCC")
    for i in active_indices:
        ortho_path = seg_orthos_map.get(i)
        if not ortho_path:
            continue
        adjusted = _refine_segment_corners_via_reference(
            ortho_path, reference_path, all_seg_corners[i])
        if adjusted is None:
            continue
        old = all_seg_corners[i]
        dlat = adjusted["NW"][0] - old["NW"][0]
        dlon = adjusted["NW"][1] - old["NW"][1]
        ref_shifts[i] = (dlat, dlon)

    if ref_shifts:
        lat_ref = 0.0
        try:
            import rasterio
            from rasterio.warp import transform_bounds
            sample_i = next(iter(ref_shifts.keys()))
            with rasterio.open(seg_orthos_map[sample_i]) as ds:
                b4 = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
                lat_ref = (b4[1] + b4[3]) / 2
        except Exception:
            pass
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(math.radians(lat_ref))
        print(f"  [per_segment] Per-segment offsets (direct NCC):")
        for i, (dlat, dlon) in sorted(ref_shifts.items()):
            print(f"    seg{i:02d}: {dlat * m_per_deg_lat:+7.0f}m N, "
                  f"{dlon * m_per_deg_lon:+7.0f}m E")
        if len(ref_shifts) >= 2:
            ys = [dl * m_per_deg_lat for dl, _ in ref_shifts.values()]
            xs = [dn * m_per_deg_lon for _, dn in ref_shifts.values()]
            print(f"    spread: {max(ys) - min(ys):.0f}m N, "
                  f"{max(xs) - min(xs):.0f}m E")
    else:
        missing = [i for i in active_indices if i not in ref_shifts]
        print(f"  [per_segment] No segment could be NCC-matched against "
              f"reference: {missing}")
    return ref_shifts


def _ortho_sidecar_path(ortho_path: str) -> str:
    return os.path.splitext(ortho_path)[0] + "_corners.json"


def _load_ortho_sidecar(ortho_path: str) -> dict | None:
    import json as _json
    sidecar = _ortho_sidecar_path(ortho_path)
    if not os.path.isfile(sidecar):
        return None
    try:
        with open(sidecar) as fh:
            raw = _json.load(fh)
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


def _save_ortho_corners(
    ortho_path: str,
    base: dict,
    applied: dict,
    metadata: dict | None = None,
) -> None:
    """Record the corners used to build this ortho next to the GeoTIFF.

    *base* is the interpolated-from-strip-corners input (the Pass 1
    baseline for this segment), and *applied* is the actual corners the
    ortho was built with (base + any Pass 2 refinement).  Storing both
    lets the cache survive across runs: Pass 1 compares *base* against
    the freshly interpolated corners to detect upstream changes
    (e.g. edited strip corners), while Pass 2 compares *applied* against
    the newly-intended corners to decide whether to re-mapproject.
    """
    import json as _json
    sidecar = _ortho_sidecar_path(ortho_path)
    try:
        def _norm(c):
            return {str(k).upper(): [float(v[0]), float(v[1])]
                    for k, v in c.items()}
        payload = {"base": _norm(base), "applied": _norm(applied)}
        if metadata:
            payload["metadata"] = metadata
        with open(sidecar, "w") as fh:
            _json.dump(payload, fh)
    except Exception as e:
        print(f"  [per_segment] corner sidecar write skipped: {e}")


def _load_ortho_corners(ortho_path: str) -> tuple[dict, dict] | None:
    """Return (base, applied) corner dicts from the sidecar, or None."""
    raw = _load_ortho_sidecar(ortho_path)
    if raw is None:
        return None
    try:
        def _rd(src):
            return {str(k).upper(): (float(v[0]), float(v[1]))
                    for k, v in src.items()}
        if "base" in raw and "applied" in raw:
            return _rd(raw["base"]), _rd(raw["applied"])
        # Legacy single-corners format: treat as both base and applied.
        only = _rd(raw)
        return only, only
    except Exception:
        return None


def _load_ortho_metadata(ortho_path: str) -> dict:
    raw = _load_ortho_sidecar(ortho_path)
    if raw is None:
        return {}
    meta = raw.get("metadata")
    return meta if isinstance(meta, dict) else {}


def _file_cache_signature(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return {"path": os.path.abspath(path), "missing": True}
    return {
        "path": os.path.abspath(path),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _subframe_cache_signature(
    source_path: str,
    *,
    rotate_180: bool,
    clean_threshold: int = 5,
    clean_keep_top_n: int = 2,
) -> dict:
    return {
        "source": _file_cache_signature(source_path),
        "rotate_180": bool(rotate_180),
        "clean_threshold": int(clean_threshold),
        "clean_keep_top_n": int(clean_keep_top_n),
    }


def _hash_cache_payload(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def _load_array_cache(cache_path: str, cache_key: str) -> np.ndarray | None:
    if not os.path.isfile(cache_path):
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as raw:
            stored_key = str(raw["cache_key"][()])
            if stored_key != cache_key:
                return None
            return raw["data"].astype(np.float64, copy=False)
    except Exception:
        return None


def _save_array_cache(cache_path: str, cache_key: str, data: np.ndarray) -> None:
    tmp_path = cache_path + ".tmp.npz"
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            tmp_path,
            cache_key=np.asarray(cache_key),
            data=np.asarray(data, dtype=np.float64),
        )
        os.replace(tmp_path, cache_path)
    except Exception:
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _corners_match(a: dict, b: dict, tol_deg: float = 1e-7) -> bool:
    a = {str(k).upper(): v for k, v in a.items()}
    b = {str(k).upper(): v for k, v in b.items()}
    for k in ("NW", "NE", "SE", "SW"):
        if k not in a or k not in b:
            return False
        if abs(a[k][0] - b[k][0]) > tol_deg or abs(a[k][1] - b[k][1]) > tol_deg:
            return False
    return True


def _auto_local_utm_crs(strip_corners: dict) -> str:
    """Pick a UTM CRS from the strip centroid."""
    lats = [float(v[0]) for v in strip_corners.values()]
    lons = [float(v[1]) for v in strip_corners.values()]
    lat_c = sum(lats) / max(len(lats), 1)
    lon_c = sum(lons) / max(len(lons), 1)
    zone = int((lon_c + 180.0) / 6.0) + 1
    epsg = 32600 + zone if lat_c >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def _bbox_from_gcps(gcps: np.ndarray,
                    min_pad_m: float = 800.0,
                    pad_frac: float = 0.05):
    """Estimate a stable ortho bbox from the GCP ground coordinates.

    ``pad_frac`` was 0.18 until 2026-04-18; on a 48-km GCP spread that
    added ~9 km per side, inflating the render extent to 65 km and
    causing adjacent-segment overlap. 0.05 keeps a small buffer for
    GCP scatter without dominating the extent. ``min_pad_m = 800 m``
    still protects narrow-band GCP clusters (e.g. Bahrain coastline,
    which can land in a <1 km strip) from over-tightening — the
    ``max(min_pad_m, …)`` rule dominates for spans below ~16 km.
    """
    if gcps is None or gcps.shape[0] == 0:
        return None
    xs = gcps[:, 2].astype(np.float64)
    ys = gcps[:, 3].astype(np.float64)
    x_min = float(xs.min())
    x_max = float(xs.max())
    y_min = float(ys.min())
    y_max = float(ys.max())
    span_x = max(1.0, x_max - x_min)
    span_y = max(1.0, y_max - y_min)
    pad_x = max(min_pad_m, span_x * pad_frac)
    pad_y = max(min_pad_m, span_y * pad_frac)
    return (x_min - pad_x, y_min - pad_y, x_max + pad_x, y_max + pad_y)


def _predicted_segment_bbox(fit_params, pixel_pitch: float,
                            image_width_px: int, image_height_px: int,
                            z_world: float = 0.0,
                            n_grid: int = 8) -> tuple | None:
    """Phase 4: project the sub-frame's corners + edge midpoints through
    the 14-parameter fit to get a predicted ground footprint.

    A panoramic scan is curved in ground coordinates (the bow-tie
    distortion), so the true footprint is not convex on the 4 image
    corners alone. Sample an ``n_grid × n_grid`` grid across the frame
    and take the axis-aligned bounding box of the projected points.
    Infinite/NaN projections (e.g. corners where the collinearity
    inverse is degenerate) are dropped.

    Returns ``(x_min, y_min, x_max, y_max)`` in the same local CRS as
    ``params.Xs0 / Ys0``, or ``None`` if fewer than 4 samples project.
    """
    try:
        from preprocess import kh_panoramic
    except ImportError:
        return None
    cols_1d = np.linspace(0.0, float(image_width_px), n_grid)
    rows_1d = np.linspace(0.0, float(image_height_px), n_grid)
    cc, rr = np.meshgrid(cols_1d, rows_1d)
    cols = cc.ravel()
    rows = rr.ravel()
    try:
        xs, ys = kh_panoramic.raw_to_world(
            params=fit_params,
            cols=cols,
            rows=rows,
            pixel_pitch=float(pixel_pitch),
            image_width_px=int(image_width_px),
            image_height_px=int(image_height_px),
            z_world=float(z_world),
        )
    except Exception:
        return None
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    finite = np.isfinite(xs) & np.isfinite(ys)
    if finite.sum() < 4:
        return None
    xs = xs[finite]
    ys = ys[finite]
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def _union_bbox(a: tuple | None, b: tuple | None) -> tuple | None:
    if a is None:
        return b
    if b is None:
        return a
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def _bbox_with_padding(bbox: tuple | None,
                       min_pad_m: float = 800.0,
                       pad_frac: float = 0.05) -> tuple | None:
    if bbox is None:
        return None
    x_min, y_min, x_max, y_max = bbox
    span_x = max(1.0, x_max - x_min)
    span_y = max(1.0, y_max - y_min)
    pad_x = max(min_pad_m, span_x * pad_frac)
    pad_y = max(min_pad_m, span_y * pad_frac)
    return (x_min - pad_x, y_min - pad_y, x_max + pad_x, y_max + pad_y)


def _resolve_bbox_policy(bbox_policy: str | None) -> str:
    """Return the effective Phase 4 bbox policy.

    Defaults to ``'predicted_union_gcp'`` when nothing is set, so the
    experimental path gets the fix by default. Unknown values fall
    back to ``'gcp_hull'`` (legacy) to avoid crashing on typos.
    """
    raw = (str(bbox_policy).strip().lower() if bbox_policy else "")
    if raw == "gcp_hull":
        return "gcp_hull"
    if raw == "predicted_union_gcp":
        return "predicted_union_gcp"
    if raw:
        print(
            f"  [per_segment/bbox] warning: unknown bbox_policy={bbox_policy!r}; "
            f"valid values are {{'gcp_hull', 'predicted_union_gcp'}}. "
            f"Falling back to 'gcp_hull'."
        )
        return "gcp_hull"
    return "predicted_union_gcp"


def _snap_bbox_to_grid(
    bbox: tuple | None, resolution_m: float, anchor: tuple = (0.0, 0.0),
) -> tuple | None:
    """Expand a bbox outward so its edges lie on a shared pixel grid.

    When adjacent per-segment orthos are rendered with bboxes that aren't
    multiples of the pixel size from a common origin, their pixel grids
    end up offset by sub-pixel amounts. The subsequent blend then has
    each segment land on a slightly different grid, which produces
    visible sub-pixel drift at segment transitions — observable as a
    stair-step at adjacent segment boundaries on Bahrain KH-9 PC.

    Snapping every segment's bbox to a grid anchored at ``anchor`` (0,0
    by default) with pixel size ``resolution_m`` guarantees that every
    segment's pixel edges coincide with every other's, so
    ``_blend_segment_mosaic``'s ``int(round(...))`` offset math is
    exact. Snaps outward (floor for lower edges, ceil for upper) so no
    content is dropped. Bbox width / height grow by at most one pixel
    per side.
    """
    if bbox is None or resolution_m <= 0:
        return bbox
    ax, ay = float(anchor[0]), float(anchor[1])
    w = (bbox[0] - ax) / resolution_m
    s = (bbox[1] - ay) / resolution_m
    e = (bbox[2] - ax) / resolution_m
    n = (bbox[3] - ay) / resolution_m
    return (
        ax + math.floor(w) * resolution_m,
        ay + math.floor(s) * resolution_m,
        ax + math.ceil(e) * resolution_m,
        ay + math.ceil(n) * resolution_m,
    )


def _resolve_render_bbox(bbox_policy: str, gcps: np.ndarray, fit_params,
                         pixel_pitch: float, image_width_px: int,
                         image_height_px: int) -> tuple:
    """Phase 4 bbox resolver used at every ``_do_mapproject`` call site.

    Returns ``(final_bbox, predicted_bbox, gcp_bbox)`` tuple so callers
    can record each candidate in telemetry without re-projecting.
    Falls back to ``gcp_hull`` semantics when either candidate is
    unavailable.
    """
    gcp_bbox = _bbox_from_gcps(gcps)
    if bbox_policy == "gcp_hull" or fit_params is None:
        return gcp_bbox, None, gcp_bbox
    predicted = _predicted_segment_bbox(
        fit_params, pixel_pitch, image_width_px, image_height_px,
    )
    if predicted is None:
        return gcp_bbox, None, gcp_bbox
    padded_predicted = _bbox_with_padding(predicted)
    final = _union_bbox(padded_predicted, gcp_bbox)
    return final, predicted, gcp_bbox


def _auto_output_resolution_m(reference_path: str | None,
                              bbox_xy,
                              image_width_px: int,
                              image_height_px: int) -> float:
    """Estimate a stable output resolution in metres per pixel.

    Prefer the reference raster's existing resolution when available so the
    per-segment orthos land on the same scale as the downstream reference-led
    alignment. If no reference resolution is available, fall back to the
    segment footprint implied by the current ortho bbox.
    """
    if reference_path and os.path.isfile(reference_path):
        try:
            import math
            import rasterio

            with rasterio.open(reference_path) as src:
                px_x = abs(float(src.transform.a))
                px_y = abs(float(src.transform.e))
                if src.crs and src.crs.is_geographic:
                    lat_c = (float(src.bounds.top) + float(src.bounds.bottom)) * 0.5
                    m_per_deg_lat = 111320.0
                    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat_c))
                    ref_res = min(px_x * m_per_deg_lon, px_y * m_per_deg_lat)
                else:
                    ref_res = min(px_x, px_y)
                if np.isfinite(ref_res) and ref_res > 0:
                    return float(ref_res)
        except Exception:
            pass

    if bbox_xy is not None:
        west, south, east, north = bbox_xy
        span_x = max(1.0, float(east) - float(west))
        span_y = max(1.0, float(north) - float(south))
        res_x = span_x / max(int(image_width_px), 1)
        res_y = span_y / max(int(image_height_px), 1)
        footprint_res = max(res_x, res_y)
        if np.isfinite(footprint_res) and footprint_res > 0:
            return float(footprint_res)

    return 4.0


def _gcp_coverage_ok(gcps: np.ndarray, image_width_px: int, image_height_px: int):
    """Return whether a GCP set spans the sub-frame robustly enough."""
    from preprocess import kh_panoramic

    ok, summary = kh_panoramic._gcp_distribution_ok(
        gcps,
        image_width_px,
        image_height_px,
    )
    return ok, summary


def _apply_geotransform_shift(ortho_path: str, dlat_deg: float,
                              dlon_deg: float) -> bool:
    """Translate an ortho's GeoTransform in-place by a lat/lon shift.

    Converts the lat/lon delta to the ortho's native CRS (typically
    EPSG:3857, metres) via pyproj using the ortho's centre latitude to get
    the correct Web Mercator scale factor, then edits the GDAL
    GeoTransform tags.  Pixel content is untouched — the operation is
    O(milliseconds) regardless of ortho size.

    Use this in place of cam_gen + re-mapproject when Pass 2 just needs to
    translate a segment: cam_gen is non-linear on corner shifts (only
    ~31–88% of the intended shift actually shows up in the re-mapprojected
    ortho), while a GeoTransform edit is exact.
    """
    try:
        from osgeo import gdal
        from pyproj import Transformer
        import rasterio
    except ImportError as e:
        print(f"  [shift] skipped: {e}")
        return False

    with rasterio.open(ortho_path) as ds:
        crs = ds.crs
        left, bottom, right, top = ds.bounds
        center_x = (left + right) / 2
        center_y = (bottom + top) / 2

    # Compute the shift in the ortho's native CRS at its centre.
    tr_to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    tr_to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    center_lon, center_lat = tr_to_ll.transform(center_x, center_y)
    target_lon = center_lon + dlon_deg
    target_lat = center_lat + dlat_deg
    x2, y2 = tr_to_crs.transform(target_lon, target_lat)
    dx_crs = x2 - center_x
    dy_crs = y2 - center_y

    gdal.UseExceptions()
    ds = gdal.Open(ortho_path, gdal.GA_Update)
    gt = list(ds.GetGeoTransform())
    gt[0] += dx_crs  # top-left X
    gt[3] += dy_crs  # top-left Y
    ds.SetGeoTransform(tuple(gt))
    ds.FlushCache()
    ds = None
    print(f"  [shift] {os.path.basename(ortho_path)}: "
          f"dlat={dlat_deg:+.5f}° dlon={dlon_deg:+.5f}° "
          f"({dy_crs:+.0f},{dx_crs:+.0f}) CRS metres")
    return True


def _measure_pairwise_seam_shift(path_a: str, path_b: str,
                                 match_res_m: float = 6.0):
    """Template-match two adjacent orthos to find their seam offset.

    Returns ``(dlat_deg, dlon_deg, metric_shift_m, response)`` — the
    lat/lon delta to apply to *path_b* so its content aligns with
    *path_a*'s content. Positive ``dlat`` moves seg_b northward,
    positive ``dlon`` moves it eastward.

    Cam_gen fit independently per sub-frame produces ground-projection
    inconsistencies between adjacent orthos: the same ground feature
    can project to different ortho positions (seen as visible
    "doubling" in the blend). On the Bahrain KH-9 test scene seg00/
    seg01 had the SAME farmland patch at geographic locations 1187 m
    apart — plainly visible side-by-side in the blended mosaic.
    See diagnostics/per_segment_diag.md for the full story.

    **Why template matching, not phase correlation**: cam_gen-produced
    orthos carry large swaths of "valid but value 0" pixels where the
    panoramic camera model extrapolates outside its fitted constraints.
    In the A/B overlap these zero swaths are anti-correlated (A has
    real content where B is zero, and vice-versa). ``phaseCorrelate``
    on the full overlap finds a weak peak near shift 0 (the zeros
    trivially align) and buries the real 1000+ m peak in noise — the
    first implementation of this helper reported 7 m / response 0.043
    on a 1187 m ground-truth shift. Template matching with
    ``cv2.TM_CCOEFF_NORMED`` on a SMALL window of A's non-zero
    content, searched inside a wider window of B, finds the correct
    peak at the right scale (verified 0.932 NCC at 1189 m).

    Uses a UTM grid at ``match_res_m`` for metric consistency.
    """
    try:
        import rasterio
        from rasterio.warp import reproject, Resampling, transform_bounds
        from rasterio.transform import from_bounds as tfm_from_bounds
        from pyproj import Transformer
        import cv2
        import math
    except ImportError as e:
        print(f"    [seam_shift] skipped: {e}")
        return 0.0, 0.0, 0.0, 0.0

    with rasterio.open(path_a) as a:
        a_crs = a.crs
        a_bounds = a.bounds
        a_nd = a.nodata if a.nodata is not None else -32768
    with rasterio.open(path_b) as b:
        b_crs = b.crs
        b_bounds = b.bounds
        b_nd = b.nodata if b.nodata is not None else -32768

    center_x = (a_bounds.left + a_bounds.right + b_bounds.left + b_bounds.right) / 4
    center_y = (a_bounds.bottom + a_bounds.top + b_bounds.bottom + b_bounds.top) / 4
    try:
        if a_crs.is_projected:
            tr_geo = Transformer.from_crs(a_crs, "EPSG:4326", always_xy=True)
            center_lon, center_lat = tr_geo.transform(center_x, center_y)
        else:
            center_lon, center_lat = center_x, center_y
        utm_zone = int((center_lon + 180) / 6) + 1
        utm_epsg = f"EPSG:{32600 + utm_zone}" if center_lat >= 0 else f"EPSG:{32700 + utm_zone}"
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

    a_utm = transform_bounds(a_crs, utm_epsg, *a_bounds)
    b_utm = transform_bounds(b_crs, utm_epsg, *b_bounds)

    # Search window: union of both bounds (so a template from A can be
    # found anywhere in B's extent, not just the narrow overlap). Cam_gen
    # inconsistency can displace content by 1000+ metres — the narrow
    # geographic overlap doesn't hold enough of B's true content to find
    # a match. We render both images over their union and template-match
    # on a crop from A inside B's full rendered area.
    ol_w = max(a_utm[0], b_utm[0])
    ol_s = max(a_utm[1], b_utm[1])
    ol_e = min(a_utm[2], b_utm[2])
    ol_n = min(a_utm[3], b_utm[3])
    if ol_e - ol_w < match_res_m * 20 or ol_n - ol_s < match_res_m * 20:
        return 0.0, 0.0, 0.0, 0.0

    # Render both images over the UNION extent (a superset of the overlap)
    union_w = min(a_utm[0], b_utm[0])
    union_s = min(a_utm[1], b_utm[1])
    union_e = max(a_utm[2], b_utm[2])
    union_n = max(a_utm[3], b_utm[3])
    w = int((union_e - union_w) / match_res_m)
    h = int((union_n - union_s) / match_res_m)
    if w < 100 or h < 100:
        return 0.0, 0.0, 0.0, 0.0
    out_tfm = tfm_from_bounds(union_w, union_s, union_e, union_n, w, h)

    a_arr = np.zeros((h, w), dtype=np.float32)
    b_arr = np.zeros((h, w), dtype=np.float32)

    with rasterio.open(path_a) as a:
        reproject(a.read(1), a_arr, src_transform=a.transform, src_crs=a.crs,
                  dst_transform=out_tfm, dst_crs=utm_epsg,
                  resampling=Resampling.bilinear, src_nodata=a_nd, dst_nodata=0)
    with rasterio.open(path_b) as b:
        reproject(b.read(1), b_arr, src_transform=b.transform, src_crs=b.crs,
                  dst_transform=out_tfm, dst_crs=utm_epsg,
                  resampling=Resampling.bilinear, src_nodata=b_nd, dst_nodata=0)

    # Find MULTIPLE strong-content template windows IN THE OVERLAP from A.
    # Multi-template voting is needed because KH panoramic orthos cover
    # highly repetitive content (farmland, desert, ocean), and a single
    # template can find strong false peaks many kilometres away. We
    # collect shift estimates from several template positions and take
    # the median — robust to both false peaks and local affine drift
    # between cam_gen fits.
    ol_c0 = int((ol_w - union_w) / match_res_m)
    ol_c1 = int((ol_e - union_w) / match_res_m)
    ol_r0 = int((union_n - ol_n) / match_res_m)
    ol_r1 = int((union_n - ol_s) / match_res_m)

    tpl_size_m = 600
    tpl_px = max(64, int(tpl_size_m / match_res_m))

    # Normalise to uint8 using a shared intensity range so templates
    # taken from A match against B on the same brightness scale.
    pooled = np.concatenate([a_arr[a_arr > 0], b_arr[b_arr > 0]])
    if pooled.size < 10000:
        return 0.0, 0.0, 0.0, 0.0
    lo = float(np.percentile(pooled, 2))
    hi = float(np.percentile(pooled, 98))
    if hi - lo < 1:
        return 0.0, 0.0, 0.0, 0.0

    a_u8 = np.zeros_like(a_arr, dtype=np.uint8)
    b_u8 = np.zeros_like(b_arr, dtype=np.uint8)
    va = a_arr > 0
    vb = b_arr > 0
    a_u8[va] = np.clip((a_arr[va] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    b_u8[vb] = np.clip((b_arr[vb] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

    def _score_window(tile):
        tv = tile > 0
        frac = float(tv.mean())
        if frac < 0.5:
            return 0.0
        var = float(tile[tv].std()) if tv.any() else 0.0
        return frac * var

    # Collect candidate template locations: stride-5 grid within the
    # overlap, keep top-K by score. K=8 gives plenty of votes without
    # making the whole step too slow.
    candidates = []
    stride = max(1, tpl_px // 3)
    for rr in range(max(0, ol_r0), min(h - tpl_px, ol_r1 - tpl_px), stride):
        for cc in range(max(0, ol_c0), min(w - tpl_px, ol_c1 - tpl_px), stride):
            tile = a_arr[rr:rr + tpl_px, cc:cc + tpl_px]
            s = _score_window(tile)
            if s > 0:
                candidates.append((s, rr, cc))
    if not candidates:
        return 0.0, 0.0, 0.0, 0.0
    candidates.sort(reverse=True)
    top_k = candidates[:8]

    # For each template, restrict the search window in B to ±2000 m from
    # the template's original position. Cam_gen fit inconsistencies are
    # typically < 1200 m; capping at 2000 m gives headroom while blocking
    # false-peak matches kilometres away on repetitive content.
    max_shift_m = 2000.0
    max_shift_px = int(max_shift_m / match_res_m)

    shifts = []  # list of (dc, dr, max_val) per template
    for _, tr0, tc0 in top_k:
        tpl = a_u8[tr0:tr0 + tpl_px, tc0:tc0 + tpl_px]
        sr0 = max(0, tr0 - max_shift_px)
        sr1 = min(h, tr0 + tpl_px + max_shift_px)
        sc0 = max(0, tc0 - max_shift_px)
        sc1 = min(w, tc0 + tpl_px + max_shift_px)
        search = b_u8[sr0:sr1, sc0:sc1]
        if search.shape[0] < tpl_px or search.shape[1] < tpl_px:
            continue
        # Skip search windows with almost no B content.
        if (b_u8[sr0:sr1, sc0:sc1] > 0).mean() < 0.2:
            continue
        result = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < 0.40:
            continue
        # Shift from template's position in A to best match in B,
        # measured in the union grid.
        dc = (sc0 + max_loc[0]) - tc0
        dr = (sr0 + max_loc[1]) - tr0
        shifts.append((dc, dr, max_val))

    if len(shifts) < 3:
        return 0.0, 0.0, 0.0, 0.0

    # Median shift across votes; response = median NCC peak.
    dcs = sorted(s[0] for s in shifts)
    drs = sorted(s[1] for s in shifts)
    vals = sorted(s[2] for s in shifts)
    mid = len(shifts) // 2
    if len(shifts) % 2:
        median_dc = dcs[mid]
        median_dr = drs[mid]
        median_val = vals[mid]
    else:
        median_dc = (dcs[mid - 1] + dcs[mid]) / 2
        median_dr = (drs[mid - 1] + drs[mid]) / 2
        median_val = (vals[mid - 1] + vals[mid]) / 2

    # Consistency check: reject if votes are wildly scattered — that
    # usually means we're seeing repetitive false peaks.
    dc_spread = dcs[-1] - dcs[0]
    dr_spread = drs[-1] - drs[0]
    if dc_spread > max_shift_px or dr_spread > max_shift_px:
        # Fallback to the MODE (vote cluster) instead of median.
        from collections import Counter
        rounded = [(round(s[0] / 5) * 5, round(s[1] / 5) * 5) for s in shifts]
        cnt = Counter(rounded)
        (mode_dc, mode_dr), mode_count = cnt.most_common(1)[0]
        if mode_count < len(shifts) // 2:
            return 0.0, 0.0, 0.0, 0.0
        median_dc = mode_dc
        median_dr = mode_dr

    dx_m = median_dc * match_res_m
    dy_m = median_dr * match_res_m

    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))

    # In the UTM output grid, row increases downward (north → south) and
    # column increases eastward. median_dc > 0 means B's match is to the
    # RIGHT of A's template — B's content is DISPLACED east of A's
    # content, so to align B with A we shift B WEST by dx_m.
    dlat_deg = (median_dr * match_res_m) / m_per_deg_lat
    dlon_deg = -(median_dc * match_res_m) / m_per_deg_lon
    metric_shift = math.hypot(dx_m, dy_m)
    print(f"    [seam_shift] {len(shifts)} votes, spread "
          f"dc={dc_spread}px dr={dr_spread}px; median "
          f"({median_dc:+.0f},{median_dr:+.0f})px = "
          f"{dx_m:+.0f}m E, {dy_m:+.0f}m S, NCC={median_val:.2f}")
    return dlat_deg, dlon_deg, metric_shift, float(median_val)


def _build_quadrilateral_mask(ortho_path: str, applied_corners: dict,
                              width: int, height: int) -> np.ndarray:
    """Rasterize the interpolated-corner quadrilateral for an ortho.

    Returns a boolean mask (h, w) where True = pixel inside the segment's
    quadrilateral footprint (reliable), False = fringe content extrapolated
    beyond the corners (unreliable, should be masked out).  The applied
    corners are in lat/lon (EPSG:4326) and include any Pass 2 shift; we
    reproject them to the ortho's CRS and rasterize in pixel space.
    """
    try:
        import rasterio
        from rasterio.features import rasterize
        from pyproj import Transformer
    except ImportError:
        return np.ones((height, width), dtype=bool)

    with rasterio.open(ortho_path) as ds:
        ortho_crs = ds.crs
        transform = ds.transform

    tr = Transformer.from_crs("EPSG:4326", ortho_crs, always_xy=True)
    corners_crs = []
    for k in ("NW", "NE", "SE", "SW"):
        lat, lon = applied_corners[str(k).upper()]
        x, y = tr.transform(lon, lat)
        corners_crs.append((x, y))
    # Close the ring
    corners_crs.append(corners_crs[0])

    geojson = {"type": "Polygon", "coordinates": [corners_crs]}
    mask = rasterize(
        [(geojson, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return mask.astype(bool)


def _blend_segment_mosaic(seg_orthos: list, output_path: str,
                          applied_corners_list: list | None = None,
                          blend_mode: str = "argmax",
                          feather_px: int = 200) -> bool:
    """Blend overlapping per-segment orthos into one seamless GeoTIFF.

    ``blend_mode`` controls how overlaps are composited:

    - ``"argmax"`` (default; legacy): each canvas pixel is owned by the
      segment whose distance-to-nearest-edge is largest. Produces a
      clean Voronoi-like split along the midline of the overlap and
      guarantees no "doubled content" ghost when adjacent segments
      project the same ground feature to slightly different positions
      — but the hard boundary is visible as a stair-step when the two
      segments' 14p fits disagree sub-pixel in the overlap region.
    - ``"feather"``: weighted average with per-segment weight clipped
      into ``[0, 1]`` by ``feather_px``. Near a segment's edge (dist
      ≤ ``feather_px``) the weight ramps linearly from 0 → 1; deeper
      inside it is 1. Overlapping segments contribute proportionally,
      smoothing the transition but at the cost of reintroducing ghost
      doubling when fits disagree. Safe when the 14p fits are well-
      anchored in the overlap region (e.g. after §4.4 model-guided
      re-matching populates GCPs across the sub-frame).

    Processed in horizontal chunks to keep memory under ~2 GB.
    """
    import rasterio
    from scipy.ndimage import distance_transform_edt

    if not seg_orthos:
        return False

    # Collect metadata.
    metas = []
    for p in seg_orthos:
        with rasterio.open(p) as ds:
            metas.append({
                "path": p,
                "bounds": ds.bounds,
                "crs": ds.crs,
                "res": ds.res,
                "dtype": ds.dtypes[0],
                "nodata": ds.nodata if ds.nodata is not None else -32768,
                "width": ds.width,
                "height": ds.height,
            })

    res_x, res_y = metas[0]["res"]
    crs = metas[0]["crs"]
    nodata = float(metas[0]["nodata"])
    dtype = metas[0]["dtype"]

    canvas_left = min(m["bounds"].left for m in metas)
    canvas_bottom = min(m["bounds"].bottom for m in metas)
    canvas_right = max(m["bounds"].right for m in metas)
    canvas_top = max(m["bounds"].top for m in metas)

    # Defensive canvas snap: if segments were rendered through
    # ``_do_mapproject`` they already land on the shared grid, but this
    # guarantees the union is pixel-aligned for blend paths that
    # bypass ``_do_mapproject`` (legacy, tests). The snap matches
    # ``_snap_bbox_to_grid`` above.
    canvas_snap = _snap_bbox_to_grid(
        (canvas_left, canvas_bottom, canvas_right, canvas_top), float(res_x),
    )
    if canvas_snap is not None:
        canvas_left, canvas_bottom, canvas_right, canvas_top = canvas_snap

    out_w = int(round((canvas_right - canvas_left) / res_x))
    out_h = int(round((canvas_top - canvas_bottom) / res_y))
    out_transform = rasterio.transform.from_bounds(
        canvas_left, canvas_bottom, canvas_right, canvas_top, out_w, out_h)

    print(f"  [blend] Output canvas: {out_w}x{out_h}px, {len(seg_orthos)} segments")

    # Pre-compute each segment's pixel placement on the canvas, its
    # distance-to-edge weight map, AND an optional quadrilateral-corner
    # mask that excludes cam_gen's fringe extrapolation.  The fringe is
    # the region beyond the interpolated corners where cam_gen is
    # extrapolating the camera model beyond its fitted constraints — two
    # adjacent segments project the same physical sub-frame content into
    # DIFFERENT ground positions in their fringes, which shows as
    # "doubled content" when the blend composites both.  Cropping each
    # segment to its interpolated corner quadrilateral eliminates the
    # fringe: adjacent quadrilaterals share edges by construction, so
    # every ground point is claimed by exactly one segment's interior.
    MAX_DIST = 200  # clamp distance so only edge falloff matters
    seg_info = []
    for i, meta in enumerate(metas):
        b = meta["bounds"]
        col_off = int(round((b.left - canvas_left) / res_x))
        row_off = int(round((canvas_top - b.top) / res_y))
        sw, sh = meta["width"], meta["height"]

        with rasterio.open(meta["path"]) as src:
            data = src.read(1).astype(np.float32)

        valid = data != float(meta["nodata"])

        # Crop to interpolated quadrilateral (if corners provided).
        if applied_corners_list is not None and i < len(applied_corners_list):
            quad_mask = _build_quadrilateral_mask(
                meta["path"], applied_corners_list[i], sw, sh)
            pre_valid = int(valid.sum())
            data[~quad_mask] = float(meta["nodata"])
            valid &= quad_mask
            post_valid = int(valid.sum())
            print(f"  [blend] seg{i}: quadrilateral crop kept "
                  f"{post_valid}/{pre_valid} valid pixels "
                  f"({100 * post_valid / max(pre_valid, 1):.0f}%)")

        # Drop isolated connected components (cam_gen fringe dots).
        # The quadrilateral corners describe the 4 points where cam_gen was
        # fit, but the sub-frame's TRUE valid data is a curved panoramic
        # footprint, not a parallelogram.  Cam_gen extrapolates the
        # OpticalBar model to produce valid-looking pixels INSIDE the
        # quadrilateral but well away from the real data — these appear as
        # small bright islands in the nodata regions between the main
        # content and the quadrilateral edges.  They are far enough from
        # the main content to form separate connected components in the
        # valid mask; we drop anything below 1% of the largest component.
        # On Bahrain KH-9 seg00 this removes ~2.9M "ghost" pixels per
        # segment that would otherwise pollute the blend.
        try:
            from scipy.ndimage import label as cc_label
            labels, n_cc = cc_label(valid)
            if n_cc > 1:
                sizes = np.bincount(labels.ravel())
                sizes[0] = 0  # background label
                main_label = int(sizes.argmax())
                main_size = int(sizes[main_label])
                threshold = max(100, int(main_size * 0.01))
                keep_labels = np.where(sizes >= threshold)[0]
                keep_labels = keep_labels[keep_labels != 0]
                keep_mask = np.isin(labels, keep_labels)
                dropped = int(valid.sum() - keep_mask.sum())
                if dropped > 0:
                    data[~keep_mask] = float(meta["nodata"])
                    valid = keep_mask
                    print(f"  [blend] seg{i}: dropped {dropped:,} fringe pixels "
                          f"({n_cc - len(keep_labels)}/{n_cc} components below "
                          f"{threshold:,} px threshold)")
        except ImportError:
            pass

        dist = distance_transform_edt(valid).astype(np.float32)
        np.minimum(dist, MAX_DIST, out=dist)  # clamp

        seg_info.append({
            "data": data,
            "weight": dist,
            "valid": valid,
            "col_off": col_off,
            "row_off": row_off,
            "w": sw,
            "h": sh,
        })
        print(f"  [blend] seg{i}: {sw}x{sh}px at ({col_off},{row_off}), "
              f"valid={valid.sum()/(sw*sh):.0%}")

    # Write output in horizontal chunks.
    CHUNK = 1024
    profile = {
        "driver": "GTiff", "width": out_w, "height": out_h, "count": 1,
        "dtype": dtype, "crs": crs, "transform": out_transform,
        "nodata": nodata, "compress": "lzw", "predictor": 2,
        "tiled": True, "blockxsize": 256, "blockysize": 256,
        "bigtiff": "if_safer",
    }

    # Compositing: argmax-Voronoi (default) or feathered weighted average.
    # See the docstring for the trade-off. feather_px sets the ramp width
    # for the feather mode (ignored in argmax mode).
    print(f"  [blend] Mode: {blend_mode}  (feather_px={feather_px})")
    with rasterio.open(output_path, "w", **profile) as dst:
        for y0 in range(0, out_h, CHUNK):
            h_chunk = min(CHUNK, out_h - y0)
            if blend_mode == "feather":
                # Weighted average: accumulate weight × data and weight,
                # finalize as numerator / denom. Weights are clipped by
                # feather_px so near-edge pixels contribute less.
                num = np.zeros((h_chunk, out_w), dtype=np.float32)
                den = np.zeros((h_chunk, out_w), dtype=np.float32)
            else:
                best_w = np.full((h_chunk, out_w), -1.0, dtype=np.float32)
                result = np.full((h_chunk, out_w), nodata, dtype=np.float32)

            for si in seg_info:
                seg_top = si["row_off"]
                seg_bot = seg_top + si["h"]
                if seg_bot <= y0 or seg_top >= y0 + h_chunk:
                    continue

                seg_r0 = max(0, y0 - seg_top)
                seg_r1 = min(si["h"], y0 + h_chunk - seg_top)
                chunk_r0 = max(0, seg_top - y0)
                chunk_r1 = chunk_r0 + (seg_r1 - seg_r0)

                c0 = si["col_off"]
                c1 = c0 + si["w"]

                d = si["data"][seg_r0:seg_r1, :]
                w_seg = si["weight"][seg_r0:seg_r1, :]

                if blend_mode == "feather":
                    # Feather ramp: 0 at boundary, 1 at feather_px+ inside.
                    fw = np.clip(
                        w_seg / max(1.0, float(feather_px)), 0.0, 1.0,
                    ).astype(np.float32)
                    contribute = w_seg > 0
                    rv_num = num[chunk_r0:chunk_r1, c0:c1]
                    rv_den = den[chunk_r0:chunk_r1, c0:c1]
                    rv_num[contribute] = rv_num[contribute] + fw[contribute] * d[contribute]
                    rv_den[contribute] = rv_den[contribute] + fw[contribute]
                else:
                    rv = result[chunk_r0:chunk_r1, c0:c1]
                    bv = best_w[chunk_r0:chunk_r1, c0:c1]
                    take = (w_seg > bv) & (w_seg > 0)
                    rv[take] = d[take]
                    bv[take] = w_seg[take]

            if blend_mode == "feather":
                result = np.full((h_chunk, out_w), nodata, dtype=np.float32)
                den_ok = den > 1e-6
                result[den_ok] = num[den_ok] / den[den_ok]

            window = rasterio.windows.Window(0, y0, out_w, h_chunk)
            dst.write(result, 1, window=window)

    # Close pixel-thin seams between adjacent segments only.  The
    # panoramic camera model produces curved valid-data boundaries per
    # segment that don't perfectly tile, leaving 1-3 px diagonal nodata
    # slivers at seams.  maxSearchDist=5 is enough to close those without
    # interpolating across larger gaps.
    #
    # Do NOT raise this value: a prior setting of 100 propagated valid
    # data across ~350 m (100 px × 3.47 m/px) and filled in the area
    # around cam_gen fringe dots on Bahrain KH-9 seg00, producing
    # visible doubled content on the west end of the blended mosaic.
    try:
        from osgeo import gdal
        gdal.UseExceptions()
        ds = gdal.Open(output_path, gdal.GA_Update)
        band = ds.GetRasterBand(1)
        mask = band.GetMaskBand()
        gdal.FillNodata(band, mask, maxSearchDist=5, smoothingIterations=0)
        ds.FlushCache()
        ds = None
        print(f"  [blend] Closed hairline seams (maxSearchDist=5)")
    except Exception as e:
        print(f"  [blend] FillNodata skipped: {e}")

    print(f"  [blend] Mosaic written: {output_path}")
    return True


# ---------------------------------------------------------------------------
# Phase 3 — smoothed displacement-field seam reconciliation
#
# After the 14p fits produce per-segment orthos we measure the residual
# displacement in each adjacent-pair overlap with a tiled RoMa dense match
# (on reprojected float arrays), fit a thin-plate-spline (RBFInterpolator)
# to the surviving control points, and apply an anti-symmetric half-warp
# that moves adjacent segments toward each other only in a feathered band
# around the seam. The resulting mosaic has no visible knife-edge at the
# seam while segment interiors (feather weight ~0) remain at their
# per-segment fit geometry.
# ---------------------------------------------------------------------------

def _phase3_extract_ortho_pair_ties_world(
    prev_ortho_path, curr_ortho_path,
    measurement_res_m=4.0,
    max_matches=3000,
    max_tiles=180,
    ransac_reproj_px=30.0,
    matcher_name="roma",
    matcher_runtime=None,
):
    """RoMa-match two adjacent orthos inside their geographic overlap.

    Returns ``(prev_world_xy, curr_world_xy)`` — two (N, 2) arrays of
    (X, Y) coordinates in the orthos' CRS, for matches that survived
    MAGSAC++ affine + MTE filtering (no Sampson — orthos aren't pinhole).
    Returns ``(None, None)`` when overlap is empty or matching fails.

    Intentionally *not* converting to GCPs (unlike
    :func:`kh_panoramic.extract_ortho_tie_point_gcps`): Phase-3 seam
    smoothing works entirely in the ortho's world coordinate system.
    """
    import rasterio
    from rasterio.windows import from_bounds as win_from_bounds

    from preprocess.experimental.match_ip import (
        _clahe_u8,
        _dedup_spatial,
        apply_geometric_filters,
        normalize_preprocess_matcher,
        run_preprocess_matcher,
    )

    with rasterio.open(prev_ortho_path) as prev, rasterio.open(curr_ortho_path) as curr:
        ol = max(prev.bounds.left, curr.bounds.left)
        ob = max(prev.bounds.bottom, curr.bounds.bottom)
        or_ = min(prev.bounds.right, curr.bounds.right)
        ot = min(prev.bounds.top, curr.bounds.top)
        if ol >= or_ or ob >= ot:
            return None, None
        w_prev = win_from_bounds(ol, ob, or_, ot, prev.transform)
        w_curr = win_from_bounds(ol, ob, or_, ot, curr.transform)
        prev_arr = prev.read(1, window=w_prev).astype(np.float32)
        curr_arr = curr.read(1, window=w_curr).astype(np.float32)
        prev_px = abs(prev.transform[0])

    h = min(prev_arr.shape[0], curr_arr.shape[0])
    w = min(prev_arr.shape[1], curr_arr.shape[1])
    if h < 32 or w < 32:
        return None, None
    prev_arr = prev_arr[:h, :w]
    curr_arr = curr_arr[:h, :w]

    prev_u8 = _clahe_u8(prev_arr)
    curr_u8 = _clahe_u8(curr_arr)

    matcher = normalize_preprocess_matcher(matcher_name)
    pts_a, pts_b, conf = run_preprocess_matcher(
        prev_u8.astype(np.float32),
        curr_u8.astype(np.float32),
        matcher_name=matcher,
        matcher_runtime=matcher_runtime,
        max_matches=max_matches,
        max_tiles=max_tiles,
    )
    if pts_a is None or len(pts_a) < 20:
        return None, None

    pts_a, pts_b, conf, _M = apply_geometric_filters(
        pts_a, pts_b, conf,
        affine_reproj_px=ransac_reproj_px,
        sampson_enabled=False,
        # MTE disabled for cross-modal / ortho-vs-ortho matching: real
        # panoramic distortion produces position-dependent residuals that
        # MTE's local-median + global-scale rejection misinterpret as
        # outliers, dropping valid scan-edge matches and undermining fit
        # coverage. Re-enable if a future dataset shows many land-cover-
        # change false positives that Sampson alone can't catch.
        mte_enabled=False,
        min_inliers=20,
    )
    if pts_a is None:
        return None, None

    pts_a, pts_b, conf = _dedup_spatial(pts_a, pts_b, conf, cell_px=40)

    prev_world_x = ol + pts_a[:, 0] * prev_px
    prev_world_y = ot - pts_a[:, 1] * prev_px
    curr_world_x = ol + pts_b[:, 0] * prev_px
    curr_world_y = ot - pts_b[:, 1] * prev_px

    prev_world = np.column_stack([prev_world_x, prev_world_y]).astype(np.float64)
    curr_world = np.column_stack([curr_world_x, curr_world_y]).astype(np.float64)
    return prev_world, curr_world


def _phase3_fit_seam_tps(pts_src, pts_dst, smoothing=100.0,
                         max_residual_m=30.0, robust_rounds=3):
    """Fit a TPS to a pairwise displacement field.

    ``pts_src`` and ``pts_dst`` are (N, 2) arrays of matched world
    coordinates; displacement = pts_dst - pts_src. Fits two RBF
    thin-plate-splines (one per axis), iteratively dropping > 3σ
    residuals. Returns ``(tps_dx, tps_dy, final_rms_m)`` or
    ``(None, None, inf)`` when the final RMS exceeds ``max_residual_m``
    or fitting fails.
    """
    try:
        from scipy.interpolate import RBFInterpolator
    except ImportError:
        return None, None, float("inf")

    src = np.asarray(pts_src, dtype=np.float64)
    dst = np.asarray(pts_dst, dtype=np.float64)
    if len(src) < 20:
        return None, None, float("inf")
    disp = dst - src

    keep = np.ones(len(src), dtype=bool)
    rms = float("inf")
    tps_dx = tps_dy = None
    for _ in range(max(1, int(robust_rounds))):
        p = src[keep]
        d = disp[keep]
        if len(p) < 20:
            return None, None, float("inf")
        try:
            tps_dx = RBFInterpolator(p, d[:, 0], kernel="thin_plate_spline",
                                     smoothing=smoothing)
            tps_dy = RBFInterpolator(p, d[:, 1], kernel="thin_plate_spline",
                                     smoothing=smoothing)
        except Exception:
            return None, None, float("inf")
        pred = np.column_stack([tps_dx(src), tps_dy(src)])
        res = np.linalg.norm(disp - pred, axis=1)
        rms = float(np.sqrt(np.mean(res[keep] ** 2)))
        thresh = max(3.0 * rms, 1.5)
        new_keep = keep & (res < thresh)
        if int(new_keep.sum()) == int(keep.sum()):
            break
        keep = new_keep

    if rms > float(max_residual_m):
        return None, None, rms
    return tps_dx, tps_dy, rms


def _phase3_compute_seam_warped_array(src_arr, src_transform, nodata,
                                     tps_dx, tps_dy, sign,
                                     feather_px, neighbor_valid_mask,
                                     neighbor_bounds_xy,
                                     self_valid_mask=None):
    """Warp a single ortho array by a half-TPS displacement field.

    The warp tapers from 1 at the *edge of this segment's own valid data*
    (i.e. near the boundary facing the neighbour) to 0 deep in this
    segment's interior. Critically, the warp is also masked to pixels
    where the neighbour has data — so the thin overlap band between the
    two segments is the ONLY region that moves. The rest of the segment
    (including edges on non-neighbour sides) is invariant.

    An earlier revision keyed the feather off the neighbour's distance
    transform directly, which evaluated to 1.0 throughout the full
    geographic overlap region. For wide overlaps that applied the full
    TPS displacement over a large area and the TPS extrapolation at the
    far edge diverged (observed a 54 → 950 px amplification on Bahrain
    KH-9 seg 0-1). The own-edge-distance formulation restricts the warp
    to a narrow band at the seam itself.

    Parameters
    ----------
    src_arr, src_transform, nodata
        The segment ortho to warp, in raster grid.
    tps_dx, tps_dy
        RBFInterpolator callables returning displacement in world metres.
    sign
        Amplitude AND direction of the warp. Pass e.g. ``-0.5 * α_a`` to
        pull segment A toward B by half of the α-weighted displacement,
        or ``+0.5 * α_b`` to pull segment B the other way. Magnitude ≤ 1.
    feather_px
        Band width (in source pixels) over which the warp tapers from 1
        at the segment's own valid edge to 0 in the interior.
    neighbor_valid_mask, neighbor_bounds_xy
        Validity mask of the *other* segment reprojected into this
        ortho's grid. Used to restrict the warp to the overlap region.
    self_valid_mask
        Validity mask of this segment itself (``src_arr != nodata``). If
        None we derive it from src_arr. Used to define this segment's
        own-edge for the feather.
    """
    import cv2

    h, w = src_arr.shape
    cols = np.arange(w, dtype=np.float64)
    rows = np.arange(h, dtype=np.float64)
    cc, rr = np.meshgrid(cols, rows)
    a, b, c, d, e, f = (src_transform.a, src_transform.b, src_transform.c,
                        src_transform.d, src_transform.e, src_transform.f)
    x_world = a * cc + b * rr + c
    y_world = d * cc + e * rr + f

    # Feather: distance from THIS segment's own-valid edge. At the edge of
    # our own valid data (where we meet the seam): dist = 0 → feather = 1.
    # Deep in our interior: dist >> feather_px → feather = 0.
    if self_valid_mask is None:
        if nodata is None:
            self_valid_mask = np.ones_like(src_arr, dtype=bool)
        else:
            self_valid_mask = src_arr != nodata
    # distanceTransform returns distance to nearest zero. So we pass 1
    # where our OWN valid data is, 0 where it isn't: dist is measured
    # from our valid interior back to our own nodata boundary.
    own_valid_u8 = self_valid_mask.astype(np.uint8)
    dist_own_edge = cv2.distanceTransform(own_valid_u8, cv2.DIST_L2, 3)
    feather = 1.0 - np.clip(dist_own_edge / max(1.0, float(feather_px)), 0.0, 1.0)
    # Zero out pixels where the neighbour doesn't reach — those are our
    # non-overlap edges (e.g. the far side of the segment), which must
    # not be warped.
    feather = feather * (neighbor_valid_mask.astype(np.float32))
    # Zero out our own nodata pixels (nothing to warp).
    feather = feather * own_valid_u8.astype(np.float32)
    feather = feather.astype(np.float32)

    # Sample TPS at every pixel (downsample if the ortho is large).
    # Evaluate on a 4× coarser grid and bilinearly upsample.
    coarse_step = 4
    ch = max(2, h // coarse_step)
    cw = max(2, w // coarse_step)
    crs = np.linspace(0, h - 1, ch).astype(np.float64)
    ccs = np.linspace(0, w - 1, cw).astype(np.float64)
    ccc, rrc = np.meshgrid(ccs, crs)
    x_coarse = a * ccc + b * rrc + c
    y_coarse = d * ccc + e * rrc + f
    pts_coarse = np.column_stack([x_coarse.ravel(), y_coarse.ravel()])
    dx_coarse = tps_dx(pts_coarse).reshape(ch, cw)
    dy_coarse = tps_dy(pts_coarse).reshape(ch, cw)
    dx_world_field = cv2.resize(
        dx_coarse.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR,
    )
    dy_world_field = cv2.resize(
        dy_coarse.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR,
    )

    # Convert world-metre displacement → source-pixel displacement.
    # x = a*col + b*row + c   →   dcol = dx/a (when b≈0)
    # Ignore off-diagonal shear (a,e are pixel sizes; b,d are typically 0).
    dcol = (sign * feather * dx_world_field) / float(a)
    drow = (sign * feather * dy_world_field) / float(e)

    # Remap: for each dst pixel, sample from (col + dcol, row + drow).
    map_x = cc.astype(np.float32) + dcol
    map_y = rr.astype(np.float32) + drow
    warped = cv2.remap(
        src_arr.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(nodata) if nodata is not None else 0.0,
    )
    return warped


def _phase3_smooth_seams(seg_orthos, sigmas_px, output_dir, scene_id,
                        matcher_name="roma", matcher_runtime=None,
                        feather_px=400, max_residual_m=30.0,
                        smoothing=100.0):
    """Apply pairwise TPS half-warp to smooth seams.

    For each adjacent pair, measure dense tie points via the Phase 1
    filtered RoMa matcher, fit a TPS, and write two warped ortho copies
    (A moved +α_A·half, B moved −α_B·half) where α_i is weighted by the
    inverse of the per-segment reprojection uncertainty.

    Returns the list of warped ortho paths in the same order as the
    input. If any step fails for a pair, that pair's originals pass
    through unchanged (graceful degradation).
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    if seg_orthos is None or len(seg_orthos) < 2:
        return list(seg_orthos) if seg_orthos else []

    # Start with the originals. Each successful pair-warp overwrites
    # entries for its two segments in this list.
    current_paths = list(seg_orthos)

    for pair_idx in range(len(current_paths) - 1):
        a_path = current_paths[pair_idx]
        b_path = current_paths[pair_idx + 1]
        if a_path is None or b_path is None:
            continue

        pts_prev, pts_curr = _phase3_extract_ortho_pair_ties_world(
            a_path, b_path,
            matcher_name=matcher_name,
            matcher_runtime=matcher_runtime,
        )
        if pts_prev is None or pts_curr is None:
            print(f"  [phase3/seam] pair {pair_idx}-{pair_idx+1}: "
                  f"no ortho ties; skipping warp")
            continue

        # Displacement A → B; TPS maps A-world to the per-match shift.
        tps_dx, tps_dy, rms_m = _phase3_fit_seam_tps(
            pts_prev, pts_curr,
            smoothing=smoothing,
            max_residual_m=max_residual_m,
        )
        if tps_dx is None:
            print(f"  [phase3/seam] pair {pair_idx}-{pair_idx+1}: "
                  f"TPS fit rejected (rms={rms_m:.1f} m)")
            continue

        # Asymmetric α weights from per-segment σ.
        sigma_a = float(sigmas_px.get(pair_idx, 1.0)) if sigmas_px else 1.0
        sigma_b = float(sigmas_px.get(pair_idx + 1, 1.0)) if sigmas_px else 1.0
        sigma_a = max(sigma_a, 0.5)
        sigma_b = max(sigma_b, 0.5)
        alpha_a = (sigma_b ** 2) / (sigma_a ** 2 + sigma_b ** 2)
        alpha_b = 1.0 - alpha_a

        # Read both orthos and compute neighbour validity masks reprojected
        # into each other's grid (for feather weight).
        try:
            with rasterio.open(a_path) as a_src:
                a_arr = a_src.read(1)
                a_transform = a_src.transform
                a_crs = a_src.crs
                a_nd = a_src.nodata if a_src.nodata is not None else 0
                a_w, a_h = a_src.width, a_src.height
                a_profile = a_src.profile
            with rasterio.open(b_path) as b_src:
                b_arr = b_src.read(1)
                b_transform = b_src.transform
                b_crs = b_src.crs
                b_nd = b_src.nodata if b_src.nodata is not None else 0
                b_w, b_h = b_src.width, b_src.height
                b_profile = b_src.profile
        except Exception as e:
            print(f"  [phase3/seam] pair {pair_idx}-{pair_idx+1}: "
                  f"read failed ({e})")
            continue

        # B validity reprojected into A grid:
        b_valid_raw = (b_arr != b_nd).astype(np.uint8)
        b_in_a = np.zeros((a_h, a_w), dtype=np.uint8)
        try:
            reproject(
                b_valid_raw, b_in_a,
                src_transform=b_transform, src_crs=b_crs,
                dst_transform=a_transform, dst_crs=a_crs,
                resampling=Resampling.nearest,
                src_nodata=0, dst_nodata=0,
            )
        except Exception as e:
            print(f"  [phase3/seam] pair {pair_idx}-{pair_idx+1}: "
                  f"B→A reproject failed ({e})")
            continue
        a_valid_raw = (a_arr != a_nd).astype(np.uint8)
        a_in_b = np.zeros((b_h, b_w), dtype=np.uint8)
        try:
            reproject(
                a_valid_raw, a_in_b,
                src_transform=a_transform, src_crs=a_crs,
                dst_transform=b_transform, dst_crs=b_crs,
                resampling=Resampling.nearest,
                src_nodata=0, dst_nodata=0,
            )
        except Exception as e:
            print(f"  [phase3/seam] pair {pair_idx}-{pair_idx+1}: "
                  f"A→B reproject failed ({e})")
            continue

        # Warp A toward B by α_a·half, warp B toward A by α_b·half.
        # TPS returns displacement (B − A) in world coords. cv2.remap with
        # map_x = col + dcol samples from (col + dcol), so the content ends
        # up shifted by −dcol. To move A's content *toward* B (i.e. shift
        # world-X by +α·half·tps), dcol must be −α·half·tps/a: hence the
        # negative sign on A's half-shift and positive on B's.
        try:
            a_warped = _phase3_compute_seam_warped_array(
                a_arr, a_transform, a_nd, tps_dx, tps_dy,
                sign=-0.5 * alpha_a, feather_px=feather_px,
                neighbor_valid_mask=(b_in_a > 0),
                neighbor_bounds_xy=None,
                self_valid_mask=(a_valid_raw > 0),
            )
            b_warped = _phase3_compute_seam_warped_array(
                b_arr, b_transform, b_nd, tps_dx, tps_dy,
                sign=+0.5 * alpha_b, feather_px=feather_px,
                neighbor_valid_mask=(a_in_b > 0),
                neighbor_bounds_xy=None,
                self_valid_mask=(b_valid_raw > 0),
            )
        except Exception as e:
            print(f"  [phase3/seam] pair {pair_idx}-{pair_idx+1}: "
                  f"half-warp failed ({e})")
            continue

        # Restore nodata on pixels that started as nodata in the source.
        a_warped = np.where(a_valid_raw > 0, a_warped, a_nd).astype(a_arr.dtype)
        b_warped = np.where(b_valid_raw > 0, b_warped, b_nd).astype(b_arr.dtype)

        a_out_path = os.path.join(
            output_dir, f"{scene_id}_seg{pair_idx:02d}_ortho_warped.tif",
        )
        b_out_path = os.path.join(
            output_dir, f"{scene_id}_seg{pair_idx+1:02d}_ortho_warped.tif",
        )
        try:
            with rasterio.open(a_out_path, "w", **a_profile) as dst:
                dst.write(a_warped, 1)
            with rasterio.open(b_out_path, "w", **b_profile) as dst:
                dst.write(b_warped, 1)
        except Exception as e:
            print(f"  [phase3/seam] pair {pair_idx}-{pair_idx+1}: "
                  f"write failed ({e})")
            continue

        print(
            f"  [phase3/seam] pair {pair_idx}-{pair_idx+1}: "
            f"TPS rms={rms_m:.1f}m σ_a={sigma_a:.2f} σ_b={sigma_b:.2f} "
            f"α_a={alpha_a:.2f} α_b={alpha_b:.2f} (ties n={len(pts_prev)})"
        )
        current_paths[pair_idx] = a_out_path
        current_paths[pair_idx + 1] = b_out_path

    return current_paths


def opticalbar_per_segment_precorrect(sub_frames, camera_params, strip_corners,
                                      output_dir, dem_path=None, resolution=None,
                                      t_srs="EPSG:3857", scene_id=None,
                                      is_aft=False, reference_path=None,
                                      acq_date=None):
    """Per-segment ortho with per-segment refinement (2OC §3.1 + §4.4).

    Pass 1 runs cam_gen + mapproject on each sub-frame independently using
    linearly-interpolated USGS corners.  Pass 2 measures each ortho's
    offset against the reference basemap and re-mapprojects that segment
    with its own corrected corners (reference-first, pairwise-NCC fallback
    for segments with low reference correlation).  The refined orthos are
    then composited with distance-to-edge weighted blending.

    Results cache across calls via per-ortho corner sidecars
    (``{scene_id}_seg{i:02d}_corners.json``): re-entering with identical
    inputs skips cam_gen + mapproject entirely.  To iterate on blending
    alone, delete ``{scene_id}_per_segment.tif`` and re-run.
    """
    import copy

    n = len(sub_frames)
    if n == 0:
        print("  [per_segment] No sub-frames provided")
        return None

    os.makedirs(output_dir, exist_ok=True)

    if scene_id is None:
        base = os.path.basename(sub_frames[0])
        scene_id, _ = os.path.splitext(base)
        # Strip trailing _<letter> if present (e.g. "..._a" -> "...")
        if len(scene_id) >= 2 and scene_id[-2] == "_" and scene_id[-1].isalpha():
            scene_id = scene_id[:-2]

    # Telemetry record — populated as decisions are taken, persisted on exit.
    scene_telem = _SceneTelemetry(
        scene_id=scene_id,
        started_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        n_subframes_input=n,
        is_aft=bool(is_aft),
    )

    # Aft cameras: reverse frame order (geographic west first).
    if is_aft:
        sub_frames = list(reversed(sub_frames))

    # Pre-compute all segment corners and filter to reference bbox BEFORE
    # any heavy processing (rotation, cam_gen, mapproject).
    all_seg_corners = []
    for i in range(n):
        all_seg_corners.append(interpolate_segment_corners(strip_corners, n, i))

    # Resolve the per-mission TLE altitude (if available). This is the
    # physics-based spacecraft altitude at acquisition — DEM-independent
    # and invariant across the sub-frames in a 0.5 s scan. We seed cam_gen
    # with it (tighter starting point than the 170 km nominal) and later
    # prefer it over cam_gen's refined altitude when both are present.
    strip_tle_altitude = None
    strip_tle_source = None
    try:
        sc_lats = [strip_corners[k][0] for k in ("NW", "NE", "SE", "SW")]
        sc_lons = [strip_corners[k][1] for k in ("NW", "NE", "SE", "SW")]
        ctr_lat = float(sum(sc_lats) / 4.0)
        ctr_lon = float(sum(sc_lons) / 4.0)
    except (KeyError, TypeError):
        ctr_lat = ctr_lon = None
    mref = parse_entity_id(scene_id) if scene_id else None
    if mref is not None and acq_date is not None and ctr_lat is not None:
        alt_res = altitude_m_at(
            mref.mission_id, acq_date, ctr_lat, ctr_lon,
        )
        if alt_res is not None:
            strip_tle_altitude = float(alt_res.altitude_m)
            strip_tle_source = alt_res.source
            scene_telem.strip_tle_altitude_m = strip_tle_altitude
            scene_telem.strip_tle_source = strip_tle_source
            detail = f" (source={alt_res.source}"
            if alt_res.tle_epoch_utc:
                detail += f", tle_epoch={alt_res.tle_epoch_utc}"
            if alt_res.subpoint_distance_km is not None:
                detail += f", d_min={alt_res.subpoint_distance_km:.1f}km"
            detail += ")"
            print(
                f"  [per_segment] {mref.system} mission {mref.mission_id} "
                f"TLE altitude: {strip_tle_altitude:,.0f} m{detail}"
            )

    # Phase 3c catalog-mean altitude: (perigee + apogee) / 2 from
    # ``data/kh_missions.yaml``. Used as a third candidate in the fit-
    # quality tiebreak alongside TLE and cam_gen. Distinct from TLE
    # (which gives the altitude at a specific orbital pass) and from
    # cam_gen (which fits the 4 USGS corners): it's a scalar prior
    # centred in the mission's published altitude range, useful when
    # the other two sources both land in wrong-phase basins.
    strip_catalog_mean_altitude = None
    if mref is not None:
        try:
            cm_alt = catalog_mean_altitude_m(mref.mission_id)
        except Exception as _exc:
            cm_alt = None
            print(f"  [per_segment] catalog_mean lookup failed: {_exc}")
        if cm_alt is not None:
            strip_catalog_mean_altitude = float(cm_alt)
            scene_telem.strip_catalog_mean_altitude_m = strip_catalog_mean_altitude
            print(
                f"  [per_segment] {mref.system} mission {mref.mission_id} "
                f"catalog-mean altitude: {strip_catalog_mean_altitude:,.0f} m "
                f"(from perigee+apogee/2)"
            )

    # One cam_gen run on the STRIP's USGS corners gives us the per-frame
    # altitude. Altitude is invariant across sub-frames within a 0.5 s
    # scan (spacecraft displacement < 1 m), so we derive it once here
    # and inject it into every sub-frame's LM init. Running cam_gen on
    # INTERPOLATED sub-frame corners is unreliable — the corner ordering
    # collides with the 180° rotation we apply to Aft sub-frames and the
    # refined altitude comes back wrong (observed 33–83 km on Bahrain).
    strip_cam_gen_altitude = None
    altitude_tiebreak_pending = False
    if bool(camera_params.get("cam_gen_altitude", False)):
        # cam_gen needs a RAW panoramic image paired with its ground
        # corners. The raw stitched strip (``stitched/{scene_id}_stitched
        # .tif``) is the correct match for ``strip_corners``; it preserves
        # the OpticalBar scan geometry and spans the full footprint.
        # Previous versions of this code used ``{scene_id}_cropped.tif``
        # from the segments directory as a fallback, but that file is the
        # EPSG:3857-projected ortho of the per_segment blend — running
        # cam_gen on an already-orthorectified image leaves the scan
        # geometry undetermined and the solver lands on unphysical 50–55
        # km "altitudes" that look like good 4-corner fits but give
        # garbage mapproject results downstream.
        stitched_dir = os.path.dirname(os.path.dirname(output_dir))
        candidate = os.path.join(
            stitched_dir, "stitched", f"{scene_id}_stitched.tif",
        )
        if not os.path.isfile(candidate):
            # Fall back to the first raw sub-frame. sub_frames[0] has
            # the correct panoramic geometry (it IS a raw scan segment)
            # but its width is only ~1/N of the strip; paired with
            # strip_corners cam_gen will compensate with wrong altitude.
            # This path is a last resort.
            candidate = sub_frames[0]
            print(
                f"  [per_segment] cam_gen: stitched strip not found at "
                f"{candidate}, falling back to sub_frames[0] "
                f"(altitude may be off)"
            )
        cam_gen_info = cam_gen_opticalbar_per_subframe(
            sub_path=candidate,
            corners=strip_corners,
            camera_params=camera_params,
            dem_path=dem_path,
            output_tsai=os.path.join(
                output_dir, f"{scene_id}_strip_opticalbar.tsai",
            ),
            altitude_m=strip_tle_altitude,
        )
        if cam_gen_info is not None:
            alt_m = float(cam_gen_info["altitude_m"])
            scene_telem.strip_cam_gen_altitude_m = alt_m
            delta_km = (
                (alt_m - strip_tle_altitude) / 1000.0
                if strip_tle_altitude is not None else None
            )
            if delta_km is not None:
                scene_telem.cam_gen_altitude_delta_km = delta_km

            # Phase 3b altitude authority gate — 3-zone logic.
            #   a) |cam_gen − TLE| ≤ _TLE_GUARD_TIGHT_KM  → accept cam_gen
            #      directly (no tiebreak: altitudes already agree).
            #   b) _TLE_GUARD_TIGHT_KM < |cam_gen − TLE| ≤ _TLE_GUARD_REJECT_KM
            #      → defer to a Stage-A/B fit-quality tiebreak in the
            #      main loop. Memory records scenes where cam_gen's
            #      refined altitude was more correct than TLE's mean-
            #      motion propagation (TLE can be perigee-biased), so a
            #      simple rejection throws away valid signal on those.
            #   c) |cam_gen − TLE| > _TLE_GUARD_REJECT_KM OR out of
            #      physical range [140, 280] km → reject outright.
            _TLE_GUARD_TIGHT_KM = 5.0
            _TLE_GUARD_REJECT_KM = 30.0
            in_physical_range = (140_000.0 <= alt_m <= 280_000.0)
            extreme_delta = (
                delta_km is not None and abs(delta_km) > _TLE_GUARD_REJECT_KM
            )
            moderate_delta = (
                delta_km is not None
                and abs(delta_km) > _TLE_GUARD_TIGHT_KM
                and abs(delta_km) <= _TLE_GUARD_REJECT_KM
            )
            if not in_physical_range:
                scene_telem.cam_gen_altitude_status = "rejected_out_of_range"
                print(
                    f"  [per_segment] cam_gen altitude {alt_m:,.0f} m outside "
                    f"[140, 280] km — rejecting, falling back to TLE/nominal"
                )
            elif extreme_delta:
                scene_telem.cam_gen_altitude_status = (
                    "rejected_extreme_disagreement"
                )
                print(
                    f"  [per_segment] cam_gen altitude {alt_m:,.0f} m disagrees "
                    f"with TLE {strip_tle_altitude:,.0f} m by {delta_km:+.1f} km "
                    f"(> {_TLE_GUARD_REJECT_KM} km reject gate) — using TLE"
                )
            elif moderate_delta:
                # Defer: leave strip_cam_gen_altitude None for now; the
                # main loop will do the tiebreak once coarse GCPs exist.
                # Keep cam_gen altitude in scene_telem for the tiebreak.
                altitude_tiebreak_pending = True
                scene_telem.cam_gen_altitude_status = "pending_tiebreak"
                print(
                    f"  [per_segment] cam_gen altitude {alt_m:,.0f} m disagrees "
                    f"with TLE {strip_tle_altitude:,.0f} m by {delta_km:+.1f} km "
                    f"(> {_TLE_GUARD_TIGHT_KM} km) — will run Stage-A/B "
                    f"tiebreak on the first successful segment"
                )
            else:
                # Within the tight gate — accept directly.
                strip_cam_gen_altitude = alt_m
                scene_telem.cam_gen_altitude_status = "used"
                print(
                    f"  [per_segment] cam_gen strip altitude: "
                    f"{alt_m:,.0f} m (will override Zs0 for all sub-frames)"
                )
        else:
            scene_telem.cam_gen_altitude_status = "cam_gen_failed"
            print(
                f"  [per_segment] cam_gen unavailable/failed — using TLE/NOMINAL_Z_S0"
            )
    else:
        scene_telem.cam_gen_altitude_status = "not_attempted"

    # ``altitude_tiebreak_pending`` was set above in the moderate-delta
    # branch. The main loop consumes it once coarse GCPs for the first
    # active segment are available: if two viable candidates exist
    # (cam_gen in the 5-30 km disagreement band + TLE), it runs Stage
    # A/B for each and picks the lower-RMS one.

    # Record which altitude source will actually be used by the per-segment
    # fits below. The Stage A/B caller at ``initial.Zs0 = ...`` prefers
    # cam_gen (if accepted) over TLE; falls through to NOMINAL otherwise.
    if strip_cam_gen_altitude is not None:
        scene_telem.altitude_source_used = "cam_gen"
        scene_telem.altitude_used_m = float(strip_cam_gen_altitude)
    elif strip_tle_altitude is not None:
        scene_telem.altitude_source_used = "tle"
        scene_telem.altitude_used_m = float(strip_tle_altitude)
    else:
        scene_telem.altitude_source_used = "nominal"
        scene_telem.altitude_used_m = float(NOMINAL_ALTITUDE_M)

    # Phase 3c: even when cam_gen and TLE already agree tightly (no Phase
    # 3b tiebreak pending), the catalog-mean altitude may be materially
    # different — e.g. both TLE and cam_gen land near perigee while the
    # real orbit-mean is +30 km higher. Triggering the tiebreak in that
    # case costs one extra Stage A/B fit (~15-30 s) and gives us per-
    # candidate RMS in telemetry. We only need to do this when the
    # tiebreak would surface new information, i.e. when catalog_mean
    # differs from the currently-selected altitude by more than the
    # ± 5 km tight-agreement window.
    if (
        not altitude_tiebreak_pending
        and strip_catalog_mean_altitude is not None
        and scene_telem.altitude_used_m is not None
    ):
        cm_delta_km = abs(
            (strip_catalog_mean_altitude - scene_telem.altitude_used_m) / 1000.0
        )
        if cm_delta_km > 5.0:
            altitude_tiebreak_pending = True
            if scene_telem.cam_gen_altitude_status in (None, "used"):
                scene_telem.cam_gen_altitude_status = "pending_tiebreak"
            print(
                f"  [per_segment] catalog-mean altitude "
                f"{strip_catalog_mean_altitude:,.0f} m differs from selected "
                f"{scene_telem.altitude_used_m:,.0f} m by {cm_delta_km:+.1f} km — "
                f"will run Stage-A/B tiebreak on the first successful segment"
            )

    active_indices = list(range(n))
    if reference_path and os.path.isfile(reference_path):
        try:
            import rasterio
            from rasterio.warp import transform_bounds
            with rasterio.open(reference_path) as ref:
                ref_bounds_4326 = transform_bounds(ref.crs, "EPSG:4326", *ref.bounds)
            margin_deg = 0.02  # ~2 km
            ref_west = ref_bounds_4326[0] - margin_deg
            ref_east = ref_bounds_4326[2] + margin_deg
            filtered = []
            for i in active_indices:
                lc = {str(k).lower(): v for k, v in all_seg_corners[i].items()}
                seg_west = min(lc["nw"][1], lc["sw"][1])
                seg_east = max(lc["ne"][1], lc["se"][1])
                if seg_east > ref_west and seg_west < ref_east:
                    filtered.append(i)
            if filtered:
                skipped = len(active_indices) - len(filtered)
                if skipped > 0:
                    print(f"  [per_segment] Skipping {skipped} segments outside reference bbox "
                          f"(processing {len(filtered)}/{n})")
                active_indices = filtered
        except Exception as e:
            print(f"  [per_segment] Reference bbox filter skipped: {e}")

    scene_telem.active_indices = list(active_indices)
    for i in active_indices:
        _seg_telem(scene_telem, i)

    print(f"  [per_segment] Processing {len(active_indices)} sub-frames for {scene_id}"
          f"{' (Aft)' if is_aft else ''}")

    # Rotate only the active Aft sub-frames in parallel (I/O bound).
    working_frames = {}  # index → path (post-rotation, pre-cleaning)
    if is_aft:
        from .stitch import flip_frame_180
        from concurrent.futures import ThreadPoolExecutor
        rot_tasks = []
        for i in active_indices:
            rot_path = os.path.join(output_dir, f"{scene_id}_seg{i:02d}_rot180.tif")
            if not os.path.isfile(rot_path):
                rot_tasks.append((i, sub_frames[i], rot_path))
            working_frames[i] = rot_path
        if rot_tasks:
            print(f"  [per_segment] Rotating {len(rot_tasks)} Aft sub-frames in parallel...")
            with ThreadPoolExecutor(max_workers=min(3, len(rot_tasks))) as pool:
                list(pool.map(lambda t: flip_frame_180(t[1], t[2]), rot_tasks))
    else:
        for i in active_indices:
            working_frames[i] = sub_frames[i]

    # Strip non-content regions (film markers, data blocks, rebate) from
    # each raw sub-frame before cam_gen sees it. Without this, cam_gen
    # happily mapprojects registration marks and data blocks into the
    # ortho fringe — visible as doubled-content "ghosts" in the blended
    # mosaic (see _clean_raw_subframe docstring for detail). Cached
    # cleaned files live next to the rotated ones with a _clean suffix.
    cleaned_frames = {}
    subframe_cache_sigs = {}
    for i in active_indices:
        src_path = working_frames[i]
        stem, ext = os.path.splitext(src_path)
        cleaned_path = f"{stem}_clean{ext}"
        if not os.path.isfile(cleaned_path):
            ok = _clean_raw_subframe(src_path, cleaned_path)
            if not ok:
                print(f"  [per_segment] seg{i:02d} clean failed; using raw")
                cleaned_frames[i] = src_path
                subframe_cache_sigs[i] = _subframe_cache_signature(
                    sub_frames[i],
                    rotate_180=bool(is_aft),
                )
                continue
        cleaned_frames[i] = cleaned_path
        subframe_cache_sigs[i] = _subframe_cache_signature(
            sub_frames[i],
            rotate_180=bool(is_aft),
        )
    working_frames = cleaned_frames

    # Each segment scans 1/N of the full strip's scan period.
    seg_camera_params = copy.deepcopy(camera_params)
    nominal_scan_time = float(seg_camera_params.get("scan_time", 0.0) or 0.0)
    if nominal_scan_time > 0:
        seg_camera_params["scan_time"] = nominal_scan_time / n
    bbox_policy = _resolve_bbox_policy(seg_camera_params.get("bbox_policy"))
    print(f"  [per_segment/bbox] policy={bbox_policy}")
    # Record Phase 10 opt-in as soon as the profile is read so fallback
    # exits before the Phase 3 TPS block still report the request
    # accurately in telemetry.
    scene_telem.phase3_seam_warp_enabled = bool(
        seg_camera_params.get("panoramic_seam_warp", False)
    )

    # --- 2OC-style staged 14-parameter fit in local UTM ---
    try:
        import rasterio
        from preprocess import kh_panoramic
        from preprocess.experimental.match_ip import (
            create_preprocess_matcher_runtime,
            normalize_preprocess_matcher,
            preprocess_matcher_cache_tag,
        )
    except ImportError as e:
        print(f"  [per_segment/14p] infra unavailable ({e})")
        scene_telem.stitched_fallback_triggered = True
        scene_telem.fallback_reason = f"14p_infra_import_failed: {e}"
        _persist_scene_telemetry(scene_telem, output_dir)
        return None

    base_corners = [dict(c) for c in all_seg_corners]
    requested_local_crs = str(seg_camera_params.get("panoramic_local_crs", "") or "").strip()
    local_crs = requested_local_crs or _auto_local_utm_crs(strip_corners)
    coarse_match_res_m = float(seg_camera_params.get("gcp_match_res_m_coarse", 4.0))
    fine_match_res_m = float(seg_camera_params.get("gcp_match_res_m_fine", 2.0))
    search_pad_m = float(seg_camera_params.get("gcp_search_pad_m", 10_000.0))
    fit_rms_px_max = float(seg_camera_params.get("panoramic_fit_rms_px_max", 4.0))
    fit_rms_px_hard_max = float(
        seg_camera_params.get(
            "panoramic_fit_rms_px_hard_max",
            max(fit_rms_px_max * 5.0, 20.0),
        )
    )
    seam_shift_px_max = float(seg_camera_params.get("panoramic_seam_shift_px_max", 2.0))
    mode_tag = "14p_v2"
    preprocess_matcher = normalize_preprocess_matcher(
        seg_camera_params.get("preprocess_matcher", "roma")
    )
    matcher_cache_tag = preprocess_matcher_cache_tag(preprocess_matcher)
    # v4: MTE disabled for cross-modal matching because its local-median +
    # global-scale rejection misclassifies valid scan-edge matches as
    # outliers on panoramic-vs-ortho data. v3 caches were over-filtered.
    cache_version = f"panoramic_match_cache_v4_{matcher_cache_tag}"
    requested_resolution = float(resolution) if resolution is not None else None
    skip_rms_gate = os.environ.get(
        "DECLASS_SKIP_PER_SEGMENT_RMS_GATE", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    persistent_cache_dir = os.path.join(
        os.path.dirname(output_dir),
        ".panoramic_cache",
        scene_id,
    )
    os.makedirs(persistent_cache_dir, exist_ok=True)

    def _merge_gcps(primary: np.ndarray, secondary: np.ndarray | None) -> np.ndarray:
        if secondary is None or secondary.size == 0:
            return primary
        merged = np.vstack([primary, secondary]).astype(np.float64)
        cells = {}
        for row in merged:
            key = (int(row[0] // 80), int(row[1] // 80))
            cells[key] = row
        return np.vstack(list(cells.values())).astype(np.float64)

    def _fit_is_usable(result) -> bool:
        return (
            result is not None
            and np.isfinite(float(getattr(result, "reprojection_rms_px", np.inf)))
            and np.isfinite(float(getattr(result, "reprojection_rms_m", 0.0)))
        )

    def _fit_segment_staged(seg_idx: int, gcps: np.ndarray, initial, sf_w: int, sf_h: int,
                             fix_f: bool = False, zs0_prior_sigma_m: float | None = None):
        """Two-stage LM fit. Default ``fix_f=False`` lets the effective focal
        length float to absorb the 18 % along/cross-track ground-scale
        asymmetry of USGS-delivered KH-9 PC sub-frames (narrow-row GCPs
        drive f to ~1.1 m; ±30 % bounds + 0.5 % prior σ keep it physical).
        Pinning ``fix_f=True`` only makes sense when GCPs span a large
        fraction of image height — rare for RoMa matches on panoramic
        imagery. See ``memory/kh_panoramic_14param_findings.md`` §2.
        ``zs0_prior_sigma_m`` is currently a no-op when ``fix_zs0=True``
        (Zs0 is removed from free params so its prior doesn't apply);
        kept as a kwarg in case a future flow lets Zs0 float.
        """
        kw_prior = ({"zs0_prior_sigma_m": float(zs0_prior_sigma_m)}
                    if zs0_prior_sigma_m is not None else {})
        # Phase 9.1: forward the focal-length policy knobs when set.
        _f_frac_range = seg_camera_params.get("f_frac_range")
        if _f_frac_range is not None:
            kw_prior["f_frac_range"] = float(_f_frac_range)
        _f_prior_frac = seg_camera_params.get("f_prior_frac_sigma")
        if _f_prior_frac is not None:
            kw_prior["f_prior_frac_sigma"] = float(_f_prior_frac)
        stage_a = kh_panoramic.fit_panoramic(
            sub_frame_gcps=gcps,
            initial=initial,
            pixel_pitch=float(seg_camera_params["pixel_pitch"]),
            image_width_px=sf_w,
            image_height_px=sf_h,
            nominal_f=float(seg_camera_params["focal_length"]),
            max_iter=120,
            loss="cauchy",
            fix_zs0=True,
            fix_f=fix_f,
            fix_velocities=True,
            fix_rates=True,
            fix_p=True,
            **kw_prior,
        )
        print(
            f"  [per_segment/14p] seg{seg_idx:02d} Stage A "
            f"RMS={stage_a.reprojection_rms_px:.2f}px "
            f"f={stage_a.params.f:.4f} success={stage_a.success} "
            f"msg={getattr(stage_a, 'message', '')}"
        )
        if not _fit_is_usable(stage_a):
            return stage_a

        stage_b = kh_panoramic.fit_panoramic(
            sub_frame_gcps=gcps,
            initial=stage_a.params,
            pixel_pitch=float(seg_camera_params["pixel_pitch"]),
            image_width_px=sf_w,
            image_height_px=sf_h,
            nominal_f=float(seg_camera_params["focal_length"]),
            max_iter=260,
            loss="cauchy",
            fix_zs0=True,
            fix_f=fix_f,
            fix_velocities=False,
            fix_rates=False,
            fix_p=False,
            **kw_prior,
        )
        print(
            f"  [per_segment/14p] seg{seg_idx:02d} Stage B "
            f"RMS={stage_b.reprojection_rms_px:.2f}px "
            f"f={stage_b.params.f:.4f} success={stage_b.success} "
            f"msg={getattr(stage_b, 'message', '')}"
        )

        # SciPy reports success=False when it hits max evaluations even if the
        # parameters improved materially. For the bootstrap ortho pass we keep
        # the better finite fit rather than blindly reverting to a catastrophic
        # earlier stage.
        if not _fit_is_usable(stage_b):
            return stage_a
        if stage_b.reprojection_rms_px > max(stage_a.reprojection_rms_px * 1.5,
                                             fit_rms_px_max * 2.0):
            return stage_a
        if stage_b.success:
            return stage_b
        if stage_b.reprojection_rms_px < stage_a.reprojection_rms_px * 0.75:
            print(
                f"  [per_segment/14p] seg{seg_idx:02d} Stage B kept despite "
                f"solver flag because RMS improved materially"
            )
            return stage_b
        return stage_a

    def _run_altitude_tiebreak(seg_idx: int, coarse_gcps: np.ndarray,
                               sf_w: int, sf_h: int):
        """Phase 3b/3c: pick the winning altitude by Stage A/B RMS.

        Runs ``_fit_segment_staged`` once per available candidate —
        cam_gen, TLE, and (Phase 3c) catalog mean `(perigee+apogee)/2` —
        and returns a tuple ``(winning_alt_m, winning_source,
        winning_fit, candidates)`` where ``candidates`` is a list of
        dicts suitable for serialising into the telemetry sidecar.
        The caller reuses ``winning_fit`` as the segment's Stage A/B
        result so the tiebreak is not extra net work on the winner.

        Winner selection: lowest RMS. When two candidates are within
        ``_TIEBREAK_HYSTERESIS_PX`` of each other, ties are broken in
        the preference order ``tle > catalog_mean > cam_gen`` — TLE
        is the safest default because its propagation noise is well-
        bounded; catalog_mean is a scalar prior with no per-pass
        signal; cam_gen is the most variable across executions.
        """
        _TIEBREAK_HYSTERESIS_PX = 0.25
        _PREFERENCE_ORDER = ("tle", "catalog_mean", "cam_gen")
        candidates = []
        results = {}
        for source, alt in (
            ("cam_gen", scene_telem.strip_cam_gen_altitude_m),
            ("tle", scene_telem.strip_tle_altitude_m),
            ("catalog_mean", scene_telem.strip_catalog_mean_altitude_m),
        ):
            if alt is None:
                continue
            initial = kh_panoramic.PanoramicParams.from_gcps_nadir(
                sub_frame_gcps=coarse_gcps,
                pixel_pitch=float(seg_camera_params["pixel_pitch"]),
                image_width_px=sf_w,
                image_height_px=sf_h,
                nominal_f=float(seg_camera_params["focal_length"]),
            )
            initial.omega0 = float(
                seg_camera_params.get("forward_tilt", 0.0) or 0.0
            )
            initial.Zs0 = float(alt)
            fit = _fit_segment_staged(
                seg_idx, coarse_gcps, initial, sf_w, sf_h,
                fix_f=False, zs0_prior_sigma_m=None,
            )
            rms = float(getattr(fit, "reprojection_rms_px", float("inf")))
            usable = _fit_is_usable(fit)
            candidates.append({
                "source": source,
                "alt_m": float(alt),
                "rms_px": rms,
                "usable": bool(usable),
                "fitted_f_m": float(getattr(fit.params, "f", 0.0)) if usable else None,
            })
            if usable:
                results[source] = (alt, fit, rms)
        if not results:
            return None, None, None, candidates
        # Find lowest RMS across usable candidates.
        lowest_rms = min(r[2] for r in results.values())
        # Prefer the highest-trust source that is within hysteresis of
        # the lowest RMS. This defaults to TLE on a near-tie and only
        # switches to catalog_mean or cam_gen when one of them is
        # materially (> 0.25 px) better.
        winner_source = None
        for pref in _PREFERENCE_ORDER:
            if pref in results and results[pref][2] <= lowest_rms + _TIEBREAK_HYSTERESIS_PX:
                winner_source = pref
                break
        if winner_source is None:
            # Shouldn't happen but fall back to the actual lowest.
            winner_source = min(results, key=lambda k: results[k][2])
        alt, fit, rms = results[winner_source]
        others = [
            f"{s}={results[s][2]:.2f}px@{results[s][0]/1000:.1f}km"
            for s in results if s != winner_source
        ]
        print(
            f"  [per_segment/altitude] tiebreak: {winner_source} wins "
            f"({rms:.2f}px @ {alt/1000:.1f}km"
            + (f"; runners-up {', '.join(others)}" if others else "")
            + ")"
        )
        return alt, winner_source, fit, candidates

    def _run_phase4_joint_ba(
        active_indices, seg_orthos_map, seg_fit_map, seg_shape_map,
        seg_gcps_map, seg_sub_path_map, seg_base_corners_map,
        seg_camera_params, local_crs, dem_path, output_dir, scene_id,
        scene_telem, do_mapproject, measure_seams, bbox_policy,
        mode_tag, preprocess_matcher, matcher_cache_tag,
    ):
        """Phase 4: joint BA refinement via ASP ``bundle_adjust``.

        1. For each active segment, seed an ASP OpticalBar ``.tsai`` by
           running ``cam_gen`` over the per-segment RoMa-vs-reference
           GCPs (``--pixel-values`` + ``--lon-lat-values`` LSQ). This
           hands the convention bridge to ASP itself — no manual 14p →
           OpticalBar conversion needed.
        2. Concatenate per-segment ``seg_gcps_map`` into one absolute
           GCP file (ASP format) for BA.
        3. Call ``run_strip_bundle_adjustment`` with shared focal
           length and intrinsics limits.
        4. Re-mapproject each refined camera → candidate new orthos.
        5. Compare pre/post aggregate seam quality — keep winner.

        Mutates ``seg_orthos_map`` and ``seg_fit_map`` in place iff BA
        accepted. Records decision in ``scene_telem``.
        """
        from align.experimental.bundle_adjust import (
            _write_absolute_gcp_file,
            _write_cam_gen_controls,
            run_strip_bundle_adjustment,
        )

        phase4_dir = os.path.join(output_dir, "phase4_joint_ba")
        os.makedirs(phase4_dir, exist_ok=True)

        cam_gen = find_asp_tool("cam_gen")
        if cam_gen is None:
            scene_telem.phase4_skipped_reason = "cam_gen_not_found"
            return False

        # Phase 3d's shared f is already applied to every refitted
        # segment's fit.params.f; use the most common value across active
        # segments as the seed focal length for every cam_gen call.
        shared_f_pre = None
        for i in active_indices:
            fit = seg_fit_map.get(i)
            if fit is None:
                scene_telem.phase4_skipped_reason = (
                    "missing_fit_in_seg_fit_map"
                )
                return False
            if shared_f_pre is None:
                shared_f_pre = float(fit.params.f)
        scene_telem.phase4_shared_f_m_before = shared_f_pre

        # Gather inputs and emit per-segment seeds via cam_gen-over-GCPs.
        seed_tsai = []
        ba_frames = []
        gcp_entries = []
        for i in active_indices:
            fit = seg_fit_map[i]
            gcps = seg_gcps_map.get(i)
            if gcps is None or gcps.shape[0] == 0:
                scene_telem.phase4_skipped_reason = "missing_gcps"
                return False
            sub_path = seg_sub_path_map.get(i)
            shape = seg_shape_map.get(i)
            if sub_path is None or shape is None:
                scene_telem.phase4_skipped_reason = "missing_sub_path_or_shape"
                return False
            base_corners_i = seg_base_corners_map.get(i)
            if base_corners_i is None:
                scene_telem.phase4_skipped_reason = "missing_base_corners"
                return False

            # Sample tsai seeds the cam_gen iC from Phase 3d's altitude
            # (fit.params.Zs0 is already the accepted altitude) and the
            # shared focal length (override of the profile default).
            seg_cam_params_f = dict(seg_camera_params)
            seg_cam_params_f["focal_length"] = float(shared_f_pre)

            sample_tsai = os.path.join(
                phase4_dir, f"{scene_id}_seg{i:02d}_sample.tsai",
            )
            corners_lc = {
                k.lower(): v for k, v in base_corners_i.items()
            }
            _write_sample_tsai(
                sample_tsai,
                int(shape[0]), int(shape[1]),
                seg_cam_params_f,
                corners=corners_lc,
                altitude_m=float(fit.params.Zs0),
            )

            lon_lat_str, pixel_str, n_pts = _write_cam_gen_controls(
                gcps, local_crs, max_points=200,
            )
            if n_pts < 10:
                scene_telem.phase4_skipped_reason = (
                    f"seed_cam_gen_too_few_controls:{n_pts}"
                )
                return False

            seed_path = os.path.join(
                phase4_dir, f"{scene_id}_seg{i:02d}_seed.tsai",
            )
            ok = _run_cam_gen_subprocess(
                cam_gen,
                sample_tsai,
                lon_lat_str,
                sub_path,
                seed_path,
                dem_path,
                pixel_str=pixel_str,
            )
            if not ok or not os.path.isfile(seed_path):
                scene_telem.phase4_skipped_reason = (
                    f"seed_cam_gen_failed_seg{i:02d}"
                )
                return False
            print(
                f"  [per_segment/phase4] seg{i:02d} cam_gen seed from "
                f"{n_pts} GCPs (shared f={shared_f_pre:.4f}m, "
                f"Zs0={float(fit.params.Zs0):,.0f}m)"
            )
            seed_tsai.append(seed_path)
            ba_frames.append(sub_path)
            gcp_entries.append((os.path.basename(sub_path), gcps))

        gcp_file = os.path.join(phase4_dir, f"{scene_id}_absolute.gcp")
        try:
            n_gcps = _write_absolute_gcp_file(
                gcp_entries, local_crs, gcp_file,
            )
        except Exception as _gcp_exc:
            scene_telem.phase4_skipped_reason = f"gcp_write_failed:{_gcp_exc}"
            return False
        scene_telem.phase4_gcp_count = int(n_gcps)
        if n_gcps < 10:
            scene_telem.phase4_skipped_reason = (
                f"too_few_gcps:{n_gcps}"
            )
            return False

        # Pre-BA seam snapshot.
        pre_orthos = [seg_orthos_map[i] for i in active_indices]
        pre_seams = measure_seams(pre_orthos) or []
        scene_telem.phase4_pre_seams = [
            {k: v for k, v in r.items() if k != "overlap_bounds"}
            for r in pre_seams
        ]

        # Run ASP bundle_adjust.
        f_nominal = float(seg_camera_params["focal_length"])
        f_frac_limit = float(
            seg_camera_params.get("joint_ba_f_frac_limit", 0.08)
        )
        intrinsics_limits = (
            max(0.01, 1.0 - f_frac_limit),
            1.0 + f_frac_limit,
        )
        print(
            f"  [per_segment/phase4] launching ASP bundle_adjust on "
            f"{len(ba_frames)} segments, {n_gcps} absolute GCPs, "
            f"shared-f bounds ±{f_frac_limit * 100:.1f}%"
        )
        adjusted = run_strip_bundle_adjustment(
            frames=ba_frames,
            camera_params=seg_camera_params,
            corners_list=[],  # unused when initial_tsai_paths supplied
            output_dir=phase4_dir,
            initial_tsai_paths=seed_tsai,
            absolute_gcp_file=gcp_file,
            dem_path=dem_path,
            solve_intrinsics=True,
            shared_intrinsics=True,
            intrinsics_limits=intrinsics_limits,
            reference_terrain_weight=(1000.0 if dem_path else None),
            robust_threshold=2.0,
            camera_weight=0,
        )
        if not adjusted or len(adjusted) != len(ba_frames):
            scene_telem.phase4_skipped_reason = "bundle_adjust_failed"
            return False

        # Map-project each adjusted camera to candidate new orthos.
        # We reuse ASP's own mapproject wrapper because the output must
        # stay compatible with our blend pipeline downstream.
        cand_orthos = {}
        for seg_idx, adj_tsai, sub_path_i in zip(
            active_indices, adjusted, ba_frames
        ):
            sf_w_i, sf_h_i = seg_shape_map[seg_idx]
            gcps_i = seg_gcps_map[seg_idx]
            final_bbox, _pred, _gcp = _resolve_render_bbox(
                bbox_policy=bbox_policy,
                gcps=gcps_i,
                fit_params=seg_fit_map[seg_idx].params,
                pixel_pitch=float(seg_camera_params["pixel_pitch"]),
                image_width_px=int(sf_w_i),
                image_height_px=int(sf_h_i),
            )
            cand_path = os.path.join(
                phase4_dir, f"{scene_id}_seg{seg_idx:02d}_ortho_ba.tif",
            )
            new_ortho = mapproject_image(
                image_path=sub_path_i,
                camera_path=adj_tsai,
                dem_path=dem_path,
                output_path=cand_path,
                resolution=_auto_output_resolution_m(
                    reference_path=reference_path,
                    bbox_xy=final_bbox,
                    image_width_px=int(sf_w_i),
                    image_height_px=int(sf_h_i),
                ),
                t_srs=t_srs,
            )
            if new_ortho is None or not os.path.isfile(new_ortho):
                scene_telem.phase4_skipped_reason = (
                    f"mapproject_failed_seg{seg_idx:02d}"
                )
                return False
            cand_orthos[seg_idx] = new_ortho

        # Post-BA seam measurement.
        post_orthos = [cand_orthos[i] for i in active_indices]
        post_seams = measure_seams(post_orthos) or []
        scene_telem.phase4_post_seams = [
            {k: v for k, v in r.items() if k != "overlap_bounds"}
            for r in post_seams
        ]

        # Aggregate seam quality: count seams that pass production QA
        # (``_seam_report_passes`` — same gate the outer pipeline uses
        # to decide stitched fallback) and break ties on mean ZNCC of
        # "ok" seams. Using the production gate as-is is important:
        # the Apr 18 verification showed BA can raise ZNCC while the
        # phase shift blows out to > 1000 px — an "improvement" the
        # outer pipeline would then reject as a failed QA. Don't ship
        # those swaps.
        def _score(seams):
            n_pass = 0
            znccs = []
            for r in seams:
                if _seam_report_passes(r, seam_shift_px_max):
                    n_pass += 1
                if r.get("status") == "ok":
                    znccs.append(float(r.get("zncc", -1.0)))
            mean_z = sum(znccs) / len(znccs) if znccs else float("-inf")
            return (n_pass, mean_z)

        pre_score = _score(pre_seams)
        post_score = _score(post_seams)
        print(
            f"  [per_segment/phase4] seam score pre={pre_score} "
            f"post={post_score}"
        )
        if post_score <= pre_score:
            scene_telem.phase4_applied = False
            scene_telem.phase4_skipped_reason = (
                f"no_improvement:pre={pre_score} post={post_score}"
            )
            return False

        # Accept: swap orthos + update seg_fit_map with refined f (pose
        # stays 14p — ASP's 7-DoF OpticalBar lacks the rate terms).
        for seg_idx in active_indices:
            seg_orthos_map[seg_idx] = cand_orthos[seg_idx]
            # Parse refined f from the adjusted tsai for telemetry.
            try:
                adj_tsai = adjusted[active_indices.index(seg_idx)]
                with open(adj_tsai) as fh_t:
                    for ln in fh_t:
                        if ln.startswith("f = "):
                            scene_telem.phase4_shared_f_m_after = float(
                                ln.split("=", 1)[1].strip()
                            )
                            break
            except Exception:
                pass
            # Refresh sidecar so the new ortho cache key matches the
            # BA-refined pose + the shared f.
            base_corners_ri = seg_base_corners_map.get(seg_idx)
            if base_corners_ri is not None:
                meta_ri = {
                    "mode": mode_tag,
                    "local_crs": local_crs,
                    "preprocess_matcher": preprocess_matcher,
                    "matcher_cache_tag": matcher_cache_tag,
                    "zs0_m": float(seg_fit_map[seg_idx].params.Zs0),
                    "altitude_source": scene_telem.altitude_source_used,
                    "phase4_joint_ba": True,
                    "phase4_shared_f_m": scene_telem.phase4_shared_f_m_after,
                }
                _save_ortho_corners(
                    cand_orthos[seg_idx],
                    base_corners_ri,
                    base_corners_ri,
                    metadata=meta_ri,
                )
        scene_telem.phase4_applied = True
        print(
            f"  [per_segment/phase4] accepted — seams improved "
            f"{pre_score} → {post_score}; f {shared_f_pre:.4f} → "
            f"{scene_telem.phase4_shared_f_m_after}"
        )
        return True

    def _do_mapproject(params, sub_path: str, bbox, sf_w: int, sf_h: int, out_path: str):
        if bbox is None:
            return None
        if os.path.isfile(out_path):
            os.remove(out_path)
        sc = _ortho_sidecar_path(out_path)
        if os.path.isfile(sc):
            os.remove(sc)
        resolution_m = requested_resolution
        if resolution_m is None:
            resolution_m = _auto_output_resolution_m(
                reference_path=reference_path,
                bbox_xy=bbox,
                image_width_px=sf_w,
                image_height_px=sf_h,
            )
        # Snap bbox outward to a pixel grid anchored at (0, 0) so every
        # segment rendered at the same resolution lands on a shared pixel
        # lattice. Eliminates the sub-pixel drift that shows up as visible
        # stair-step between adjacent segments in the blended mosaic. The
        # snap expands bbox by at most one pixel per edge.
        snapped_bbox = _snap_bbox_to_grid(bbox, float(resolution_m))
        return kh_panoramic.mapproject(
            params=params,
            sub_frame_path=sub_path,
            dem_path=dem_path,
            out_path=out_path,
            pixel_pitch=float(seg_camera_params["pixel_pitch"]),
            image_width_px=sf_w,
            image_height_px=sf_h,
            resolution_m=float(resolution_m),
            bbox_xy=snapped_bbox,
            local_crs=local_crs,
            t_srs=t_srs,
            device=str(matcher_runtime.device),
            chunk_px=2048,
        )

    def _measure_single_seam(prev_ortho_path: str, curr_ortho_path: str) -> dict | None:
        reports = _measure_segment_seams([prev_ortho_path, curr_ortho_path])
        return reports[0] if reports else None

    def _iterative_guided_refit(
        seg_idx: int,
        coarse_gcps: np.ndarray,
        initial_fit,
        sub_path: str,
        sf_w: int,
        sf_h: int,
        first_iter_guided_gcps=None,
        max_iter: int = 3,
        tol_px: float = 0.25,
        prev_ortho_path: str | None = None,
        seam_worsen_px: float = 1.0,
        f_guard_m: float = 0.02,
    ):
        """2OC §4.4 iterative model-guided refit with seam-aware gate.

        Loops: render intermediate ortho → re-match guided GCPs against the
        reference → refit the 14p camera. Keeps best-so-far by reprojection
        RMS AND seam phase-shift against the previous segment (if provided).
        A lower per-GCP RMS can coexist with a worse seam (observed on
        Bahrain seg00: iterated fit found f=1.07 m / 21 px RMS / 54 px seam
        vs f=1.524 m / 12 px RMS / 1.98 px seam for the Stage A/B pose) —
        the seam gate rejects those iterations.

        If ``first_iter_guided_gcps`` is provided, the first loop iteration
        reuses it (this preserves the existing on-disk GCP cache flow).
        Subsequent iterations always re-extract because the intermediate
        ortho changes each iteration.

        Returns ``(best_fit, best_gcps)``.
        """
        best_fit = initial_fit
        best_gcps = coarse_gcps
        best_rms = float(initial_fit.reprojection_rms_px)
        best_seam_px = float("inf")

        if reference_path is None or not os.path.isfile(reference_path):
            return best_fit, best_gcps

        # Baseline seam against previous segment, if we have one. This is
        # the geometric-pose fidelity check — if a candidate iteration's
        # ortho produces a seam worse than this by > seam_worsen_px, reject.
        # Only compute when we'll actually iterate (max_iter > 0) so we
        # don't do an extra render + seam measurement for no benefit.
        if max_iter > 0 and prev_ortho_path is not None and os.path.isfile(prev_ortho_path):
            bbox0 = _bbox_from_gcps(coarse_gcps)
            if bbox0 is not None:
                baseline_ortho_path = os.path.join(
                    output_dir,
                    f"{scene_id}_seg{seg_idx:02d}_ortho_baseline.tif",
                )
                baseline_ortho = _do_mapproject(
                    initial_fit.params, sub_path, bbox0, sf_w, sf_h,
                    baseline_ortho_path,
                )
                try:
                    if baseline_ortho is not None:
                        baseline_seam = _measure_single_seam(
                            prev_ortho_path, baseline_ortho
                        )
                        if baseline_seam and baseline_seam.get("status") == "ok":
                            best_seam_px = float(
                                baseline_seam.get("phase_shift_px", float("inf"))
                            )
                            print(
                                f"  [per_segment/14p] seg{seg_idx:02d} "
                                f"baseline seam shift={best_seam_px:.2f}px "
                                f"against seg{(seg_idx-1):02d} — will reject "
                                f"iterations whose seam worsens by >{seam_worsen_px}px"
                            )
                finally:
                    if os.path.isfile(baseline_ortho_path):
                        os.remove(baseline_ortho_path)
                    sc = _ortho_sidecar_path(baseline_ortho_path)
                    if os.path.isfile(sc):
                        os.remove(sc)

        for it in range(max_iter):
            bbox = _bbox_from_gcps(best_gcps)
            if bbox is None:
                break

            # First iteration: reuse cached intermediate ortho + guided GCPs
            # if the caller supplied them. Iterations > 0 always re-render
            # and re-match because the pose / bbox have moved.
            if it == 0 and first_iter_guided_gcps is not None:
                guided_gcps = first_iter_guided_gcps
            else:
                inter_ortho_path = os.path.join(
                    output_dir,
                    f"{scene_id}_seg{seg_idx:02d}_ortho_iter{it}.tif",
                )
                inter_ortho = _do_mapproject(
                    best_fit.params, sub_path, bbox, sf_w, sf_h, inter_ortho_path,
                )
                try:
                    if inter_ortho is None:
                        break
                    guided_gcps = kh_panoramic.extract_model_guided_gcps(
                        ortho_path=inter_ortho,
                        reference_ortho_path=reference_path,
                        params=best_fit.params,
                        pixel_pitch=float(seg_camera_params["pixel_pitch"]),
                        image_width_px=sf_w,
                        image_height_px=sf_h,
                        local_crs=local_crs,
                        matcher_name=preprocess_matcher,
                        matcher_runtime=matcher_runtime,
                        match_res_m=fine_match_res_m,
                        dem_path=dem_path,
                    )
                finally:
                    # Intermediate ortho is transient.
                    if os.path.isfile(inter_ortho_path):
                        os.remove(inter_ortho_path)
                    sc = _ortho_sidecar_path(inter_ortho_path)
                    if os.path.isfile(sc):
                        os.remove(sc)

            if guided_gcps is None or guided_gcps.shape[0] < 20:
                break

            combined_gcps = _merge_gcps(coarse_gcps, guided_gcps)
            new_fit = _fit_segment_staged(
                seg_idx, combined_gcps, best_fit.params, sf_w, sf_h,
            )
            if not _fit_is_usable(new_fit):
                break
            new_rms = float(new_fit.reprojection_rms_px)

            if new_rms > best_rms + 0.5:
                print(
                    f"  [per_segment/14p] seg{seg_idx:02d} guided iter {it} "
                    f"diverged ({best_rms:.2f} → {new_rms:.2f}px); "
                    f"keeping best-so-far"
                )
                break

            # Gate against the Stage-A/B floor, not just iteration improvement.
            rms_gate = max(fit_rms_px_max, initial_fit.reprojection_rms_px * 1.25)
            if new_rms > rms_gate:
                break

            # Phase 5 focal-drift guard: reject the iteration if its f
            # landed within ``f_guard_m`` of either ±_F_FRAC_RANGE bound
            # AND the Stage-A/B pose was safely inside. The intent is to
            # catch the historical f=1.07 collapse (30 % below nominal,
            # right on the lower bound) without rejecting small drifts
            # that the fit finds legitimately useful.
            try:
                from preprocess.kh_panoramic import _F_FRAC_RANGE as _F_RANGE
            except Exception:
                _F_RANGE = 0.30
            try:
                nominal_f = float(seg_camera_params["focal_length"])
                new_f = float(new_fit.params.f)
                f_bound_lo = nominal_f * (1.0 - _F_RANGE)
                f_bound_hi = nominal_f * (1.0 + _F_RANGE)
                new_margin = min(abs(new_f - f_bound_lo),
                                 abs(new_f - f_bound_hi))
                initial_margin = min(
                    abs(float(initial_fit.params.f) - f_bound_lo),
                    abs(float(initial_fit.params.f) - f_bound_hi),
                )
                if new_margin < f_guard_m < initial_margin:
                    print(
                        f"  [per_segment/14p] seg{seg_idx:02d} guided iter "
                        f"{it} rejected: f={new_f:.4f}m within "
                        f"{f_guard_m:.3f}m of ±{_F_RANGE*100:.0f}% bound "
                        f"(Stage A/B was safely at f="
                        f"{float(initial_fit.params.f):.4f}m)"
                    )
                    break
            except Exception:
                pass

            if prev_ortho_path is not None and np.isfinite(best_seam_px):
                candidate_bbox = _bbox_from_gcps(combined_gcps)
                if candidate_bbox is not None:
                    candidate_path = os.path.join(
                        output_dir,
                        f"{scene_id}_seg{seg_idx:02d}_ortho_iter{it}_seamcheck.tif",
                    )
                    candidate_ortho = _do_mapproject(
                        new_fit.params, sub_path, candidate_bbox,
                        sf_w, sf_h, candidate_path,
                    )
                    try:
                        if candidate_ortho is not None:
                            candidate_seam = _measure_single_seam(
                                prev_ortho_path, candidate_ortho
                            )
                            if candidate_seam and candidate_seam.get("status") == "ok":
                                candidate_shift = float(
                                    candidate_seam.get("phase_shift_px", float("inf"))
                                )
                                if candidate_shift > best_seam_px + seam_worsen_px:
                                    print(
                                        f"  [per_segment/14p] seg{seg_idx:02d} "
                                        f"guided iter {it} rejected: seam shift "
                                        f"{best_seam_px:.2f} → {candidate_shift:.2f}px "
                                        f"(> tolerance {seam_worsen_px}px)"
                                    )
                                    break
                                # Seam OK — update best_seam_px along with fit.
                                if new_rms < best_rms - tol_px:
                                    best_seam_px = candidate_shift
                    finally:
                        if os.path.isfile(candidate_path):
                            os.remove(candidate_path)
                        sc = _ortho_sidecar_path(candidate_path)
                        if os.path.isfile(sc):
                            os.remove(sc)

            if new_rms < best_rms - tol_px:
                print(
                    f"  [per_segment/14p] seg{seg_idx:02d} guided iter {it} "
                    f"accepted RMS {best_rms:.2f} → {new_rms:.2f}px "
                    f"n_gcps={combined_gcps.shape[0]}"
                )
                best_fit = new_fit
                best_gcps = combined_gcps
                best_rms = new_rms
                continue

            # Converged: improvement below tolerance.
            print(
                f"  [per_segment/14p] seg{seg_idx:02d} guided iter {it} "
                f"converged RMS {best_rms:.2f}px (Δ={best_rms - new_rms:+.3f})"
            )
            break

        return best_fit, best_gcps

    def _maybe_regularize_against_previous(
        seg_idx: int,
        prev_idx: int,
        sub_path: str,
        sf_w: int,
        sf_h: int,
        fit_res,
        gcps: np.ndarray,
        ortho_path: str,
    ):
        prev_fit = seg_fit_map.get(prev_idx)
        prev_shape = seg_shape_map.get(prev_idx)
        prev_ortho = seg_orthos_map.get(prev_idx)
        if prev_fit is None or prev_shape is None or prev_ortho is None:
            return fit_res, gcps

        # Phase 6 reference-anchored guard #1: refuse to regularize against
        # a previous segment whose own reference-fit RMS is already poor.
        # Regularizing pulls the current segment toward the previous one's
        # pose; if that pose is bad, we inherit the badness. The threshold
        # is ``fit_rms_px_max * 2.0`` — inside the soft gate the fit is
        # trustworthy, past it we defer to the reference fit unchanged.
        prev_rms_guard = fit_rms_px_max * 2.0
        try:
            prev_rms = float(prev_fit.reprojection_rms_px)
        except Exception:
            prev_rms = float("inf")
        if prev_rms > prev_rms_guard:
            print(
                f"  [per_segment/seam] seg{prev_idx:02d}-{seg_idx:02d}: "
                f"skipping regularization — previous segment RMS "
                f"{prev_rms:.2f}px > {prev_rms_guard:.2f}px gate "
                f"(outlier upstream, would poison current)"
            )
            return fit_res, gcps

        baseline = _measure_single_seam(prev_ortho, ortho_path)
        if _seam_report_passes(baseline, seam_shift_px_max):
            return fit_res, gcps
        if baseline is None:
            print(
                f"  [per_segment/seam] seg{prev_idx:02d}-{seg_idx:02d}: "
                f"baseline seam unavailable; trying raw tie regularization"
            )
        elif baseline.get("status") == "ok":
            if float(baseline.get("phase_shift_px", np.inf)) <= 1.0:
                print(
                    f"  [per_segment/seam] seg{prev_idx:02d}-{seg_idx:02d}: "
                    f"baseline shift={baseline['phase_shift_px']:.2f}px is already "
                    f"subpixel; skipping raw tie regularization"
                )
                return fit_res, gcps
            print(
                f"  [per_segment/seam] seg{prev_idx:02d}-{seg_idx:02d}: "
                f"baseline zncc={baseline['zncc']:.3f} "
                f"shift={baseline['phase_shift_px']:.2f}px; "
                f"trying ortho tie regularization"
            )

            ortho_tie_cache_key = _hash_cache_payload({
                "version": cache_version,
                "kind": "ortho_ties",
                "matcher": preprocess_matcher,
                "matcher_cache_tag": matcher_cache_tag,
                "prev_sub_frame": subframe_cache_sigs[prev_idx],
                "curr_sub_frame": subframe_cache_sigs[seg_idx],
                "prev_params": [round(float(v), 9) for v in prev_fit.params.to_tensor("cpu").tolist()],
                "curr_params": [round(float(v), 9) for v in fit_res.params.to_tensor("cpu").tolist()],
                "curr_image_width_px": int(sf_w),
                "curr_image_height_px": int(sf_h),
                "pixel_pitch": float(seg_camera_params["pixel_pitch"]),
                "local_crs": local_crs,
                "dem": _file_cache_signature(dem_path),
            })
            ortho_tie_cache_path = os.path.join(
                persistent_cache_dir,
                f"{scene_id}_seg{prev_idx:02d}_{seg_idx:02d}_ortho_ties.npz",
            )
            ortho_tie_gcps = _load_array_cache(ortho_tie_cache_path, ortho_tie_cache_key)
            if ortho_tie_gcps is not None:
                print(
                    f"  [per_segment/cache] seg{prev_idx:02d}-{seg_idx:02d} "
                    f"ortho ties cache hit ({ortho_tie_gcps.shape[0]})"
                )
            else:
                ortho_tie_gcps = kh_panoramic.extract_ortho_tie_point_gcps(
                    prev_ortho_path=prev_ortho,
                    curr_ortho_path=ortho_path,
                    curr_params=fit_res.params,
                    curr_pixel_pitch=float(seg_camera_params["pixel_pitch"]),
                    curr_image_width_px=sf_w,
                    curr_image_height_px=sf_h,
                    local_crs=local_crs,
                    dem_path=dem_path,
                    matcher_name=preprocess_matcher,
                    matcher_runtime=matcher_runtime,
                    max_matches=3000,
                    max_tiles=180,
                    ransac_reproj_px=30.0,
                )
                if ortho_tie_gcps is not None and ortho_tie_gcps.shape[0] >= 20:
                    _save_array_cache(ortho_tie_cache_path, ortho_tie_cache_key, ortho_tie_gcps)

            if ortho_tie_gcps is None or ortho_tie_gcps.shape[0] < 20:
                print(
                    f"  [per_segment/seam] seg{prev_idx:02d}-{seg_idx:02d}: "
                    f"ortho tie regularization unavailable"
                )
                return fit_res, gcps

            combined_gcps = _merge_gcps(gcps, ortho_tie_gcps)
            ortho_fit = _fit_segment_staged(seg_idx, combined_gcps, fit_res.params, sf_w, sf_h)
            if not _fit_is_usable(ortho_fit):
                print(
                    f"  [per_segment/seam] seg{seg_idx:02d}: ortho-tie fit unusable; "
                    f"keeping reference fit"
                )
                return fit_res, gcps
            # Phase 6 reference-anchored guard #2: tighten the RMS gate.
            # The old guard allowed up to max(fit_res*1.5, fit_rms_px_max*4)
            # which (for kh9, fit_rms_px_max=4) permits 16 px RMS regardless
            # of how good the reference fit was. Narrowing to
            # ``max(fit_res*1.25, fit_rms_px_max*2)`` still admits ties that
            # reduce a bad seam but rejects the degenerate case where
            # the refit is chasing tie-point geometry at the cost of
            # reference-anchored fidelity.
            rms_ceiling = max(
                fit_res.reprojection_rms_px * 1.25,
                fit_rms_px_max * 2.0,
            )
            if ortho_fit.reprojection_rms_px > rms_ceiling:
                print(
                    f"  [per_segment/seam] seg{seg_idx:02d}: ortho-tie fit RMS "
                    f"{ortho_fit.reprojection_rms_px:.2f}px > reference-anchored "
                    f"ceiling {rms_ceiling:.2f}px; keeping reference fit"
                )
                return fit_res, gcps

            ortho_bbox = _bbox_from_gcps(combined_gcps)
            temp_ortho_path = os.path.join(output_dir, f"{scene_id}_seg{seg_idx:02d}_ortho_seam.tif")
            seam_ortho = _do_mapproject(
                ortho_fit.params,
                sub_path,
                ortho_bbox,
                sf_w,
                sf_h,
                temp_ortho_path,
            )
            if seam_ortho is None:
                return fit_res, gcps

            candidate = _measure_single_seam(prev_ortho, seam_ortho)
            if _seam_report_score(candidate) > _seam_report_score(baseline):
                if candidate and candidate.get("status") == "ok":
                    print(
                        f"  [per_segment/seam] seg{prev_idx:02d}-{seg_idx:02d}: "
                        f"accepted ortho-tie refit zncc={candidate['zncc']:.3f} "
                        f"shift={candidate['phase_shift_px']:.2f}px"
                    )
                else:
                    print(
                        f"  [per_segment/seam] seg{prev_idx:02d}-{seg_idx:02d}: "
                        f"accepted ortho-tie refit status={candidate.get('status') if candidate else 'unknown'}"
                    )
                os.replace(seam_ortho, ortho_path)
                return ortho_fit, combined_gcps

            if candidate and candidate.get("status") == "ok":
                print(
                    f"  [per_segment/seam] seg{prev_idx:02d}-{seg_idx:02d}: "
                    f"rejected ortho-tie refit zncc={candidate['zncc']:.3f} "
                    f"shift={candidate['phase_shift_px']:.2f}px"
                )
            else:
                print(
                    f"  [per_segment/seam] seg{prev_idx:02d}-{seg_idx:02d}: "
                    f"rejected ortho-tie refit status={candidate.get('status') if candidate else 'unknown'}"
                )
            if os.path.isfile(temp_ortho_path):
                os.remove(temp_ortho_path)
            temp_sidecar = _ortho_sidecar_path(temp_ortho_path)
            if os.path.isfile(temp_sidecar):
                os.remove(temp_sidecar)
            return fit_res, gcps
        else:
            print(
                f"  [per_segment/seam] seg{prev_idx:02d}-{seg_idx:02d}: "
                f"baseline {baseline.get('status')}; trying raw tie regularization"
            )

        prev_w, prev_h = prev_shape
        raw_tie_match_res_m = max(coarse_match_res_m, 8.0)
        tie_cache_key = _hash_cache_payload({
            "version": cache_version,
            "kind": "raw_ties",
            "matcher": preprocess_matcher,
            "matcher_cache_tag": matcher_cache_tag,
            "prev_sub_frame": subframe_cache_sigs[prev_idx],
            "curr_sub_frame": subframe_cache_sigs[seg_idx],
            "pixel_pitch": float(seg_camera_params["pixel_pitch"]),
            "focal_m": float(seg_camera_params["focal_length"]),
            "nominal_altitude_m": round(max(abs(float(prev_fit.params.Zs0)), 1_000.0), 3),
            "match_res_m": float(raw_tie_match_res_m),
            "max_matches": 4000,
            "max_tiles": 240,
            "ransac_reproj_px": 120.0,
        })
        tie_cache_path = os.path.join(
            persistent_cache_dir,
            f"{scene_id}_seg{prev_idx:02d}_{seg_idx:02d}_raw_ties.npz",
        )
        tie = _load_array_cache(tie_cache_path, tie_cache_key)
        if tie is not None:
            print(
                f"  [per_segment/cache] seg{prev_idx:02d}-{seg_idx:02d} "
                f"raw ties cache hit ({tie.shape[0]})"
            )
        else:
            tie = kh_panoramic.extract_raw_subframe_tie_points(
                working_frames[prev_idx],
                sub_path,
                pixel_pitch=float(seg_camera_params["pixel_pitch"]),
                focal_m=float(seg_camera_params["focal_length"]),
                nominal_altitude_m=max(abs(float(prev_fit.params.Zs0)), 1_000.0),
                matcher_name=preprocess_matcher,
                matcher_runtime=matcher_runtime,
                match_res_m=raw_tie_match_res_m,
                max_matches=4000,
                max_tiles=240,
                ransac_reproj_px=120.0,
            )
            if tie is not None and tie.shape[0] >= 20:
                _save_array_cache(tie_cache_path, tie_cache_key, tie)
        if tie is None or tie.shape[0] < 20:
            return fit_res, gcps

        cols_a = tie[:, 0]
        rows_a = tie[:, 1]
        cols_b = tie[:, 2]
        rows_b = tie[:, 3]
        prev_world_x, prev_world_y = kh_panoramic.raw_to_world(
            params=prev_fit.params,
            cols=cols_a,
            rows=rows_a,
            pixel_pitch=float(seg_camera_params["pixel_pitch"]),
            image_width_px=prev_w,
            image_height_px=prev_h,
            z_world=0.0,
        )
        curr_world_x, curr_world_y = kh_panoramic.raw_to_world(
            params=fit_res.params,
            cols=cols_b,
            rows=rows_b,
            pixel_pitch=float(seg_camera_params["pixel_pitch"]),
            image_width_px=sf_w,
            image_height_px=sf_h,
            z_world=0.0,
        )
        dx_world_m = prev_world_x - curr_world_x
        dy_world_m = prev_world_y - curr_world_y
        delta_world_m = np.hypot(dx_world_m, dy_world_m)
        finite = np.isfinite(delta_world_m)
        if finite.sum() < 20:
            print(
                f"  [per_segment/seam] seg{prev_idx:02d}-{seg_idx:02d}: "
                f"too few finite raw ties after world projection "
                f"({int(finite.sum())})"
            )
            return fit_res, gcps

        # A bad seam can still have a large *coherent* displacement between
        # neighbouring fits. Filter by consistency around the dominant shift,
        # not by absolute closeness to zero.
        med_dx = float(np.median(dx_world_m[finite]))
        med_dy = float(np.median(dy_world_m[finite]))
        residual_world_m = np.hypot(dx_world_m - med_dx, dy_world_m - med_dy)
        residual_gate_m = max(
            120.0,
            float(np.percentile(residual_world_m[finite], 70)) * 2.5,
        )
        dominant_shift_m = float(np.hypot(med_dx, med_dy))
        max_abs_shift_m = max(
            5_000.0,
            dominant_shift_m * 3.0,
            residual_gate_m * 6.0,
        )
        plausible = (
            finite
            & np.isfinite(residual_world_m)
            & (delta_world_m <= max_abs_shift_m)
            & (residual_world_m <= residual_gate_m)
        )
        if plausible.sum() < 20:
            print(
                f"  [per_segment/seam] seg{prev_idx:02d}-{seg_idx:02d}: "
                f"too few plausible raw ties after dominant-shift filter "
                f"({int(plausible.sum())}; median shift "
                f"{dominant_shift_m:.1f}m, residual gate "
                f"{residual_gate_m:.1f}m, max abs {max_abs_shift_m:.1f}m)"
            )
            return fit_res, gcps

        keep_idx = np.where(plausible)[0]
        keep_idx = keep_idx[np.argsort(residual_world_m[keep_idx])]
        tie_cell_px = max(8, min(256, int(round(min(sf_w, sf_h) / 4.0))))
        tie_cells = {}
        selected = []
        for idx_keep in keep_idx:
            key = (
                int(cols_b[idx_keep] // tie_cell_px),
                int(rows_b[idx_keep] // tie_cell_px),
            )
            if key in tie_cells:
                continue
            tie_cells[key] = True
            selected.append(int(idx_keep))
            if len(selected) >= 240:
                break
        tie = tie[np.asarray(selected, dtype=np.int64)]
        kept_delta = delta_world_m[np.asarray(selected, dtype=np.int64)]
        kept_residual = residual_world_m[np.asarray(selected, dtype=np.int64)]
        print(
            f"  [per_segment/seam] seg{prev_idx:02d}-{seg_idx:02d}: "
            f"kept {tie.shape[0]} seam ties after dominant-shift filter "
            f"(median shift {np.median(kept_delta):.1f}m, "
            f"median residual {np.median(kept_residual):.1f}m, "
            f"max residual {kept_residual.max():.1f}m)"
        )
        if tie.shape[0] < 20:
            return fit_res, gcps

        tie_gcps = kh_panoramic.raw_tie_points_to_gcps(
            params_a=prev_fit.params,
            tie=tie,
            pixel_pitch=float(seg_camera_params["pixel_pitch"]),
            image_width_a_px=prev_w,
            image_height_a_px=prev_h,
            image_width_b_px=sf_w,
            image_height_b_px=sf_h,
        )
        if tie_gcps is None or tie_gcps.shape[0] < 20:
            return fit_res, gcps

        combined_gcps = _merge_gcps(gcps, tie_gcps)
        seam_initial = copy.deepcopy(fit_res.params)
        seam_initial.Xs0 += med_dx
        seam_initial.Ys0 += med_dy
        print(
            f"  [per_segment/seam] seg{prev_idx:02d}-{seg_idx:02d}: "
            f"seeding seam refit with translation "
            f"dx={med_dx:.1f}m dy={med_dy:.1f}m"
        )
        seam_fit = _fit_segment_staged(seg_idx, combined_gcps, seam_initial, sf_w, sf_h)
        if not _fit_is_usable(seam_fit):
            print(
                f"  [per_segment/seam] seg{seg_idx:02d}: raw-tie fit unusable; "
                f"keeping reference fit"
            )
            return fit_res, gcps
        # Phase 6 reference-anchored guard (raw-tie path). Tightened from
        # max(fit_res*1.75, fit_rms_px_max*4) to max(fit_res*1.5,
        # fit_rms_px_max*2.5). Raw ties are noisier than ortho ties
        # (image-domain match error × altitude-dependent projection),
        # so this path keeps a slightly looser ratio than the ortho-tie
        # guard but still rejects refits that significantly worsen the
        # reference fit.
        raw_rms_ceiling = max(
            fit_res.reprojection_rms_px * 1.5,
            fit_rms_px_max * 2.5,
        )
        if seam_fit.reprojection_rms_px > raw_rms_ceiling:
            print(
                f"  [per_segment/seam] seg{seg_idx:02d}: raw-tie fit RMS "
                f"{seam_fit.reprojection_rms_px:.2f}px > reference-anchored "
                f"ceiling {raw_rms_ceiling:.2f}px; keeping reference fit"
            )
            return fit_res, gcps

        seam_bbox = _bbox_from_gcps(combined_gcps)
        temp_ortho_path = os.path.join(output_dir, f"{scene_id}_seg{seg_idx:02d}_ortho_seam.tif")
        seam_ortho = _do_mapproject(
            seam_fit.params,
            sub_path,
            seam_bbox,
            sf_w,
            sf_h,
            temp_ortho_path,
        )
        if seam_ortho is None:
            return fit_res, gcps

        candidate = _measure_single_seam(prev_ortho, seam_ortho)
        if _seam_report_score(candidate) > _seam_report_score(baseline):
            if candidate and candidate.get("status") == "ok":
                print(
                    f"  [per_segment/seam] seg{prev_idx:02d}-{seg_idx:02d}: "
                    f"accepted raw-tie refit zncc={candidate['zncc']:.3f} "
                    f"shift={candidate['phase_shift_px']:.2f}px"
                )
            else:
                print(
                    f"  [per_segment/seam] seg{prev_idx:02d}-{seg_idx:02d}: "
                    f"accepted raw-tie refit status={candidate.get('status') if candidate else 'unknown'}"
                )
            os.replace(seam_ortho, ortho_path)
            return seam_fit, combined_gcps

        if candidate and candidate.get("status") == "ok":
            print(
                f"  [per_segment/seam] seg{prev_idx:02d}-{seg_idx:02d}: "
                f"rejected raw-tie refit zncc={candidate['zncc']:.3f} "
                f"shift={candidate['phase_shift_px']:.2f}px"
            )
        else:
            print(
                f"  [per_segment/seam] seg{prev_idx:02d}-{seg_idx:02d}: "
                f"rejected raw-tie refit status={candidate.get('status') if candidate else 'unknown'}"
            )
        if os.path.isfile(temp_ortho_path):
            os.remove(temp_ortho_path)
        temp_sidecar = _ortho_sidecar_path(temp_ortho_path)
        if os.path.isfile(temp_sidecar):
            os.remove(temp_sidecar)
        return fit_res, gcps

    matcher_runtime = None
    seg_orthos_map: dict = {}
    seg_fit_map: dict = {}
    seg_shape_map: dict = {}
    # Phase 3d: stash the winning GCPs and sub-frame path per segment so
    # the cross-segment shared-f refit has everything it needs without
    # re-extracting. Populated alongside ``seg_fit_map[i] = final_fit``.
    seg_gcps_map: dict = {}
    seg_sub_path_map: dict = {}
    seg_base_corners_map: dict = {}
    failed_segments: list[int] = []
    try:
        matcher_runtime = create_preprocess_matcher_runtime(preprocess_matcher)
        for i in active_indices:
            sub_path = working_frames[i]
            seg_ortho_path = os.path.join(output_dir, f"{scene_id}_seg{i:02d}_ortho.tif")
            stage1_ortho_path = os.path.join(output_dir, f"{scene_id}_seg{i:02d}_ortho_stage1.tif")
            cached = _load_ortho_corners(seg_ortho_path)
            cached_meta = _load_ortho_metadata(seg_ortho_path)
            # Phase 3: the cached ortho is only valid if the altitude
            # authority used to produce it matches today's. Orthos
            # encode the Zs0 implicitly (cam geometry + mapproject),
            # so changing altitude sources without invalidating the
            # cache would leave downstream reads of a stale geographic
            # projection. Reject when |cached.zs0 - planned zs0| > 1km.
            # Cache-altitude check: the cached ortho's Zs0 (as recorded
            # in the sidecar) must match at least one altitude that the
            # current run's Phase 3b/3c tiebreak could plausibly pick.
            # When the tiebreak fires on the first live segment it may
            # select cam_gen, TLE, or catalog_mean; if the cached orthos
            # were written from any one of those candidates in a prior
            # run we can reuse them. Without this widening, partial
            # cache invalidation breaks Phase 3d (only some seg fits
            # populated → spread=0 → no shared-f refit → heterogeneous
            # orthos on disk → seam QA blows up).
            plausible_altitudes_m = [
                float(a)
                for a in (
                    strip_cam_gen_altitude,
                    strip_tle_altitude,
                    strip_catalog_mean_altitude,
                    NOMINAL_ALTITUDE_M,
                )
                if a is not None
            ]
            planned_zs0_m = (
                float(strip_cam_gen_altitude)
                if strip_cam_gen_altitude is not None
                else (
                    float(strip_tle_altitude)
                    if strip_tle_altitude is not None
                    else float(NOMINAL_ALTITUDE_M)
                )
            )
            cached_zs0_m = cached_meta.get("zs0_m")
            altitude_cache_ok = (
                cached_zs0_m is None
                or any(
                    abs(float(cached_zs0_m) - a) <= 1_000.0
                    for a in plausible_altitudes_m
                )
            )
            if (os.path.isfile(seg_ortho_path) and cached is not None
                    and cached_meta.get("mode") == mode_tag
                    and cached_meta.get("local_crs") == local_crs
                    and normalize_preprocess_matcher(
                        cached_meta.get("preprocess_matcher", "roma")
                    ) == preprocess_matcher
                    and altitude_cache_ok):
                cached_base, _ = cached
                if _corners_match(cached_base, base_corners[i]):
                    print(f"  [per_segment/14p] seg{i:02d} cached ({mode_tag})")
                    seg_orthos_map[i] = seg_ortho_path
                    continue
            elif os.path.isfile(seg_ortho_path) and cached_zs0_m is not None \
                    and not altitude_cache_ok:
                print(
                    f"  [per_segment/14p] seg{i:02d} cache invalid: "
                    f"cached Zs0={cached_zs0_m:,.0f}m vs planned "
                    f"{planned_zs0_m:,.0f}m — re-fitting"
                )

            if reference_path is None or not os.path.isfile(reference_path):
                seg_t = _seg_telem(scene_telem, i)
                seg_t.rejected = True
                seg_t.reject_reason = "no_reference_path"
                failed_segments.append(i)
                continue

            with rasterio.open(sub_path) as src_sf:
                sf_w = int(src_sf.width)
                sf_h = int(src_sf.height)

            coarse_cache_key = _hash_cache_payload({
                "version": cache_version,
                "kind": "coarse_gcps",
                "matcher": preprocess_matcher,
                "matcher_cache_tag": matcher_cache_tag,
                "sub_frame": subframe_cache_sigs[i],
                "reference": _file_cache_signature(reference_path),
                "dem": _file_cache_signature(dem_path),
                "local_crs": local_crs,
                "match_res_m": float(coarse_match_res_m),
                "search_pad_m": float(search_pad_m),
                "initial_corners_ll": {
                    str(k).upper(): [float(v[0]), float(v[1])]
                    for k, v in all_seg_corners[i].items()
                },
            })
            coarse_cache_path = os.path.join(
                persistent_cache_dir,
                f"{scene_id}_seg{i:02d}_coarse_gcps.npz",
            )
            coarse_gcps = _load_array_cache(coarse_cache_path, coarse_cache_key)
            if coarse_gcps is not None:
                print(
                    f"  [per_segment/cache] seg{i:02d} coarse GCP cache hit "
                    f"({coarse_gcps.shape[0]})"
                )
            else:
                # Phase 7.1: compute native GSD per-mission from the
                # same altitude authority chosen above (Phase 3). Keeps
                # KH-4 / KH-7 / KH-9 PC on mission-appropriate down-
                # sampling instead of the legacy KH-9-only 0.8 m/px.
                _pp = float(seg_camera_params["pixel_pitch"])
                _ff = float(seg_camera_params["focal_length"])
                _alt = (
                    float(strip_cam_gen_altitude)
                    if strip_cam_gen_altitude is not None
                    else (
                        float(strip_tle_altitude)
                        if strip_tle_altitude is not None
                        else float(NOMINAL_ALTITUDE_M)
                    )
                )
                native_gsd_m = (_alt * _pp / _ff) if _ff > 0 else None
                coarse_gcps = kh_panoramic.extract_reference_gcps(
                    sub_frame_path=sub_path,
                    reference_ortho_path=reference_path,
                    initial_corners_ll=all_seg_corners[i],
                    local_crs=local_crs,
                    matcher_name=preprocess_matcher,
                    matcher_runtime=matcher_runtime,
                    match_res_m=coarse_match_res_m,
                    search_pad_m=search_pad_m,
                    dem_path=dem_path,
                    native_gsd_m=native_gsd_m,
                    mte_enabled=bool(
                        seg_camera_params.get("preprocess_mte_enabled", False)
                    ),
                    mte_radius_px=float(
                        seg_camera_params.get("preprocess_mte_radius_px", 500.0)
                    ),
                )
                if coarse_gcps is not None and coarse_gcps.shape[0] >= 30:
                    _save_array_cache(coarse_cache_path, coarse_cache_key, coarse_gcps)
            if coarse_gcps is None or coarse_gcps.shape[0] < 30:
                print(f"  [per_segment/14p] seg{i:02d}: coarse GCP extraction failed")
                seg_t = _seg_telem(scene_telem, i)
                seg_t.coarse_gcp_count = 0 if coarse_gcps is None else int(coarse_gcps.shape[0])
                seg_t.rejected = True
                seg_t.reject_reason = "coarse_gcp_extraction_failed"
                failed_segments.append(i)
                continue

            coverage_ok, coverage = _gcp_coverage_ok(coarse_gcps, sf_w, sf_h)
            seg_t = _seg_telem(scene_telem, i)
            seg_t.coarse_gcp_count = int(coarse_gcps.shape[0])
            # Coverage is measured in the raw sub-frame's pixel grid
            # (columns 0 = col_px, 1 = row_px per extract_reference_gcps).
            try:
                col_px = coarse_gcps[:, 0].astype(float)
                row_px = coarse_gcps[:, 1].astype(float)
                seg_t.coarse_gcp_coverage_px_w = int(col_px.max() - col_px.min())
                seg_t.coarse_gcp_coverage_px_h = int(row_px.max() - row_px.min())
            except Exception:
                pass
            if not coverage_ok:
                print(f"  [per_segment/14p] seg{i:02d}: reject coarse GCP coverage {coverage}")
                seg_t.rejected = True
                seg_t.reject_reason = f"coarse_gcp_coverage:{coverage}"
                failed_segments.append(i)
                continue

            # Phase 3b: first time through the loop with a moderate
            # cam_gen/TLE disagreement, run the tiebreak using this
            # segment's coarse GCPs. The winning fit is reused as the
            # segment's Stage A/B result, so there's no wasted work on
            # the winner (only the loser's fit is "extra"; ~15-30 s).
            if altitude_tiebreak_pending:
                (winning_alt, winning_source, winning_fit,
                 tiebreak_candidates) = _run_altitude_tiebreak(
                    i, coarse_gcps, sf_w, sf_h,
                )
                scene_telem.altitude_tiebreak_candidates = tiebreak_candidates
                altitude_tiebreak_pending = False
                if winning_alt is None:
                    print(
                        f"  [per_segment/altitude] tiebreak failed — "
                        f"falling back to TLE"
                    )
                    scene_telem.cam_gen_altitude_status = (
                        "rejected_tiebreak_unusable"
                    )
                    strip_cam_gen_altitude = None
                else:
                    # Phase 3c: three sources are possible now.
                    # ``strip_cam_gen_altitude`` is the signal downstream
                    # code reads; when the winner is TLE or catalog_mean
                    # we clear it and set scene_telem.altitude_used_m
                    # to the explicit winning value instead.
                    if winning_source == "cam_gen":
                        strip_cam_gen_altitude = float(winning_alt)
                        scene_telem.cam_gen_altitude_status = (
                            "tiebreak_cam_gen_wins"
                        )
                        scene_telem.altitude_source_used = "cam_gen"
                        scene_telem.altitude_used_m = float(winning_alt)
                    elif winning_source == "catalog_mean":
                        strip_cam_gen_altitude = float(winning_alt)
                        scene_telem.cam_gen_altitude_status = (
                            "tiebreak_catalog_mean_wins"
                        )
                        scene_telem.altitude_source_used = "catalog_mean"
                        scene_telem.altitude_used_m = float(winning_alt)
                    else:  # "tle"
                        strip_cam_gen_altitude = None
                        scene_telem.cam_gen_altitude_status = (
                            "tiebreak_tle_wins"
                        )
                        scene_telem.altitude_source_used = "tle"
                        scene_telem.altitude_used_m = float(
                            winning_alt if winning_alt is not None
                            else (strip_tle_altitude or NOMINAL_ALTITUDE_M)
                        )
                    # Reuse the winning fit as the segment's Stage A/B
                    # result to avoid a pointless third fit.
                    fit_res = winning_fit
                    try:
                        nominal_f = float(seg_camera_params["focal_length"])
                        fitted_f = float(fit_res.params.f)
                        seg_t.stage_ab_rms_px = float(fit_res.reprojection_rms_px)
                        seg_t.stage_ab_fitted_f_m = fitted_f
                        seg_t.stage_ab_f_deviation_pct = (
                            100.0 * (fitted_f - nominal_f) / nominal_f
                        )
                    except Exception:
                        pass
                    # Skip the Stage A/B block below — already done.
                    stage_ab_done = True
                # Regardless, fall through to Stage A/B below only if
                # the tiebreak did not already produce a usable fit.

            initial = kh_panoramic.PanoramicParams.from_gcps_nadir(
                sub_frame_gcps=coarse_gcps,
                pixel_pitch=float(seg_camera_params["pixel_pitch"]),
                image_width_px=sf_w,
                image_height_px=sf_h,
                nominal_f=float(seg_camera_params["focal_length"]),
            )
            initial.omega0 = float(seg_camera_params.get("forward_tilt", 0.0) or 0.0)

            # Override Zs0 with the altitude the ortho was built at —
            # cam_gen's refined altitude when available, else TLE (used as
            # a nominal fallback). The fit MUST use the same altitude as
            # the ortho that produced the GCPs: GCPs encode a pixel-to-
            # ground mapping that's only consistent with one altitude.
            # Forcing a different Zs0 here (e.g. TLE 165 km while ortho
            # was at cam_gen's 216 km) blows up RMS to 100+ px on Bahrain.
            #
            # TLE's role is to SEED cam_gen (above) so its refined output
            # starts from a physics-based altitude instead of 170 km. Once
            # cam_gen has run, its result is authoritative for the ortho
            # chain.
            #
            # Keep ``fix_f=False``: on narrow-row GCPs (typical for KH-9 PC
            # sub-frames where RoMa matches cluster in a ~2 km along-track
            # band), the effective focal length must float to absorb the
            # 18 % along-track/cross-track ground-scale asymmetry of USGS-
            # delivered sub-frames (see memory/kh_panoramic_14param_findings.md
            # §2). Pinning f=nominal over-constrains and leaves Stage B
            # stalled at 37-62 px RMS (~3-4× the physical 15 px noise floor).
            # The ±30 % f bounds + 0.5 % prior sigma keep f from collapsing
            # unphysically while letting the fit reach 17-22 px RMS.
            if strip_cam_gen_altitude is not None:
                initial.Zs0 = float(strip_cam_gen_altitude)
            elif strip_tle_altitude is not None:
                initial.Zs0 = float(strip_tle_altitude)

            if not locals().get("stage_ab_done", False):
                fit_res = _fit_segment_staged(
                    i, coarse_gcps, initial, sf_w, sf_h,
                    fix_f=False,
                    zs0_prior_sigma_m=None,
                )
            # Reset the flag for the next iteration.
            stage_ab_done = False
            if not _fit_is_usable(fit_res):
                print(f"  [per_segment/14p] seg{i:02d}: Stage A/B fit unusable")
                seg_t.rejected = True
                seg_t.reject_reason = "stage_ab_fit_unusable"
                failed_segments.append(i)
                continue

            # Record Stage A/B fit diagnostics before any §4.4 iteration.
            try:
                nominal_f = float(seg_camera_params["focal_length"])
                fitted_f = float(fit_res.params.f)
                seg_t.stage_ab_rms_px = float(fit_res.reprojection_rms_px)
                seg_t.stage_ab_fitted_f_m = fitted_f
                seg_t.stage_ab_f_deviation_pct = (
                    100.0 * (fitted_f - nominal_f) / nominal_f
                )
            except Exception:
                pass

            bbox = _bbox_from_gcps(coarse_gcps)
            if bbox is None:
                seg_t.rejected = True
                seg_t.reject_reason = "bbox_from_coarse_gcps_none"
                failed_segments.append(i)
                continue

            # Phase 5: 2OC §4.4 model-guided re-matching, profile-driven.
            # ``guided_refit_max_iter: 0`` (legacy / production default)
            # skips the loop entirely; ``1+`` re-enters the guided-GCP
            # extract-and-refit path that many times after Stage A/B.
            # The acceptance gates inside ``_iterative_guided_refit``
            # (RMS-not-worsened, seam-not-worsened, f-not-at-bound) are
            # what make iteration safe to re-enable — they were added
            # earlier but kept dormant behind ``_PHASE2_MAX_ITER=0``.
            _PHASE2_MAX_ITER = int(
                seg_camera_params.get("guided_refit_max_iter", 0) or 0
            )
            scene_telem.phase2_max_iter = int(_PHASE2_MAX_ITER)
            seg_t.guided_iter_max = int(_PHASE2_MAX_ITER)
            first_guided = None
            if _PHASE2_MAX_ITER > 0:
                guided_cache_key = _hash_cache_payload({
                    "version": cache_version,
                    "kind": "guided_gcps",
                    "matcher": preprocess_matcher,
                    "matcher_cache_tag": matcher_cache_tag,
                    "sub_frame": subframe_cache_sigs[i],
                    "reference": _file_cache_signature(reference_path),
                    "dem": _file_cache_signature(dem_path),
                    "local_crs": local_crs,
                    "match_res_m": float(fine_match_res_m),
                    "pixel_pitch": float(seg_camera_params["pixel_pitch"]),
                    "image_width_px": int(sf_w),
                    "image_height_px": int(sf_h),
                    "bbox_xy": [round(float(v), 3) for v in bbox],
                    "fit_params": [round(float(v), 9) for v in fit_res.params.to_tensor("cpu").tolist()],
                })
                guided_cache_path = os.path.join(
                    persistent_cache_dir,
                    f"{scene_id}_seg{i:02d}_guided_gcps.npz",
                )
                first_guided = _load_array_cache(guided_cache_path, guided_cache_key)
                if first_guided is not None:
                    print(
                        f"  [per_segment/cache] seg{i:02d} guided GCP cache hit "
                        f"({first_guided.shape[0]})"
                    )
                else:
                    stage1_ortho = _do_mapproject(
                        fit_res.params, sub_path, bbox, sf_w, sf_h, stage1_ortho_path,
                    )
                    if stage1_ortho is None:
                        print(f"  [per_segment/14p] seg{i:02d}: Stage 1 ortho failed")
                        failed_segments.append(i)
                        continue
                    first_guided = kh_panoramic.extract_model_guided_gcps(
                        ortho_path=stage1_ortho,
                        reference_ortho_path=reference_path,
                        params=fit_res.params,
                        pixel_pitch=float(seg_camera_params["pixel_pitch"]),
                        image_width_px=sf_w,
                        image_height_px=sf_h,
                        local_crs=local_crs,
                        matcher_name=preprocess_matcher,
                        matcher_runtime=matcher_runtime,
                        match_res_m=fine_match_res_m,
                        dem_path=dem_path,
                    )
                    if first_guided is not None and first_guided.shape[0] >= 20:
                        _save_array_cache(guided_cache_path, guided_cache_key, first_guided)

            # Pass the previous segment's ortho so the seam-aware gate can
            # reject iterations whose fit degrades geometric alignment with
            # the neighbour (the seg00 f=1.07 regression).
            prev_pos = active_indices.index(i) - 1
            prev_ortho_for_gate = (
                seg_orthos_map.get(active_indices[prev_pos])
                if prev_pos >= 0 else None
            )
            final_fit, final_gcps = _iterative_guided_refit(
                seg_idx=i,
                coarse_gcps=coarse_gcps,
                initial_fit=fit_res,
                sub_path=sub_path,
                sf_w=sf_w,
                sf_h=sf_h,
                first_iter_guided_gcps=first_guided,
                max_iter=_PHASE2_MAX_ITER,
                tol_px=0.25,
                prev_ortho_path=prev_ortho_for_gate,
                f_guard_m=float(
                    seg_camera_params.get("guided_refit_f_guard_m", 0.02) or 0.02
                ),
            )
            # Count an accepted iteration if RMS actually improved vs Stage A/B.
            try:
                if final_fit is not fit_res and (
                    float(final_fit.reprojection_rms_px)
                    < float(fit_res.reprojection_rms_px) - 1e-6
                ):
                    seg_t.guided_iter_accepted = 1
            except Exception:
                pass

            # Phase 4 render bbox: prefer the union of the GCP hull and
            # the predicted sub-frame footprint (forward-projected through
            # the fit). This is what restores meaningful seam overlap on
            # Bahrain — GCPs cluster in a ~2 km along-track band, so the
            # legacy gcp_hull bbox clips the ortho to a narrow Y-strip and
            # adjacent segments' strips don't overlap geographically.
            final_bbox, predicted_bbox, gcp_bbox = _resolve_render_bbox(
                bbox_policy=bbox_policy,
                gcps=final_gcps,
                fit_params=final_fit.params,
                pixel_pitch=float(seg_camera_params["pixel_pitch"]),
                image_width_px=sf_w,
                image_height_px=sf_h,
            )
            # Telemetry — record each candidate so Phase 4 A/B diffs are
            # straightforward.
            try:
                if predicted_bbox is not None:
                    seg_t.predicted_bbox = [float(v) for v in predicted_bbox]
                if gcp_bbox is not None:
                    seg_t.gcp_bbox = [float(v) for v in gcp_bbox]
                if final_bbox is not None:
                    seg_t.final_bbox = [float(v) for v in final_bbox]
                seg_t.bbox_source = (
                    "predicted_union_gcp"
                    if (bbox_policy == "predicted_union_gcp"
                        and predicted_bbox is not None)
                    else "gcp_hull"
                )
            except Exception:
                pass
            final_ortho = _do_mapproject(
                final_fit.params,
                sub_path,
                final_bbox,
                sf_w,
                sf_h,
                seg_ortho_path,
            )
            if final_ortho is None:
                seg_t.rejected = True
                seg_t.reject_reason = "final_mapproject_failed"
                failed_segments.append(i)
                continue

            prev_pos = active_indices.index(i) - 1
            if prev_pos >= 0:
                prev_idx = active_indices[prev_pos]
                pre_reg_rms = float(final_fit.reprojection_rms_px)
                pre_reg_n = int(final_gcps.shape[0])
                final_fit, final_gcps = _maybe_regularize_against_previous(
                    seg_idx=i,
                    prev_idx=prev_idx,
                    sub_path=sub_path,
                    sf_w=sf_w,
                    sf_h=sf_h,
                    fit_res=final_fit,
                    gcps=final_gcps,
                    ortho_path=final_ortho,
                )
                # Did the regularizer actually swap the fit? We don't
                # know here whether the ortho-tie or raw-tie path fired,
                # only that one of them produced a different result. The
                # helper logs that detail to stdout; telemetry records
                # the outcome at this granularity for now.
                if (
                    final_gcps.shape[0] != pre_reg_n
                    or abs(float(final_fit.reprojection_rms_px) - pre_reg_rms) > 1e-6
                ):
                    seg_t.regularizer_fired = "fired"
                else:
                    seg_t.regularizer_fired = "none"
            else:
                seg_t.regularizer_fired = "not_applicable"

            if final_fit.reprojection_rms_px > fit_rms_px_max:
                if final_fit.reprojection_rms_px > fit_rms_px_hard_max and not skip_rms_gate:
                    print(
                        f"  [per_segment/14p] seg{i:02d}: RMS "
                        f"{final_fit.reprojection_rms_px:.2f}px exceeds hard gate "
                        f"{fit_rms_px_hard_max:.2f}px"
                    )
                    seg_t.rejected = True
                    seg_t.reject_reason = (
                        f"final_rms_exceeds_hard_gate:"
                        f"{final_fit.reprojection_rms_px:.2f}>{fit_rms_px_hard_max:.2f}"
                    )
                    failed_segments.append(i)
                    continue
                if skip_rms_gate:
                    print(
                        f"  [per_segment/14p] seg{i:02d}: RMS "
                        f"{final_fit.reprojection_rms_px:.2f}px exceeds gate "
                        f"{fit_rms_px_max:.2f}px but continuing because "
                        f"DECLASS_SKIP_PER_SEGMENT_RMS_GATE is set"
                    )
                else:
                    print(
                        f"  [per_segment/14p] seg{i:02d}: RMS "
                        f"{final_fit.reprojection_rms_px:.2f}px exceeds gate "
                        f"{fit_rms_px_max:.2f}px but is within hard gate "
                        f"{fit_rms_px_hard_max:.2f}px; deferring to seam QA"
                    )

            meta = {
                "mode": mode_tag,
                "local_crs": local_crs,
                "preprocess_matcher": preprocess_matcher,
                "matcher_cache_tag": matcher_cache_tag,
                "fit_rms_px": float(final_fit.reprojection_rms_px),
                "n_gcps": int(final_gcps.shape[0]),
                "zs0_m": float(final_fit.params.Zs0),
                "altitude_source": scene_telem.altitude_source_used,
            }
            _save_ortho_corners(final_ortho, base_corners[i], base_corners[i], metadata=meta)
            seg_orthos_map[i] = final_ortho
            seg_fit_map[i] = final_fit
            # Phase 3d: stash the fit inputs for later cross-segment refit.
            seg_gcps_map[i] = final_gcps
            seg_sub_path_map[i] = sub_path
            seg_base_corners_map[i] = base_corners[i]
            # Record final fit outcome in telemetry.
            try:
                nominal_f = float(seg_camera_params["focal_length"])
                f_final = float(final_fit.params.f)
                seg_t.final_fit_rms_px = float(final_fit.reprojection_rms_px)
                seg_t.final_fitted_f_m = f_final
                seg_t.final_f_deviation_pct = 100.0 * (f_final - nominal_f) / nominal_f
                seg_t.final_gcp_count = int(final_gcps.shape[0])
                seg_t.ortho_path = final_ortho
                # bbox_source + predicted_bbox / gcp_bbox / final_bbox are
                # already populated above by _resolve_render_bbox.
            except Exception:
                pass
            seg_shape_map[i] = (sf_w, sf_h)
            if os.path.isfile(stage1_ortho_path):
                os.remove(stage1_ortho_path)
            print(f"  [per_segment/14p] seg{i:02d} final ortho complete")
    finally:
        if matcher_runtime is not None:
            try:
                matcher_runtime.close()
            except Exception:
                pass

    # Accept partial per-segment success: as long as at least one segment
    # produced a usable ortho we prefer blending those over the whole-strip
    # fallback's single-camera model. Drop the failed segments from the
    # active list and carry on. The geometry for the blend is already
    # per-frame-refined (the 14-param fit on each survivor); gaps from
    # dropped segments are OK — downstream alignment only uses the pixels
    # that are actually present.
    succeeded = [i for i in active_indices if i in seg_orthos_map]
    if not succeeded:
        print(f"  [per_segment] 14p path failed for all active segments "
              f"{failed_segments}; whole-strip fallback")
        scene_telem.stitched_fallback_triggered = True
        scene_telem.fallback_reason = (
            f"all_segments_failed:{sorted(failed_segments)}"
        )
        _persist_scene_telemetry(scene_telem, output_dir)
        return None
    if failed_segments:
        print(
            f"  [per_segment] Dropping failed segments {failed_segments}; "
            f"proceeding with {len(succeeded)}/{len(active_indices)} "
            f"successful segments"
        )
    active_indices = succeeded

    # ------------------------------------------------------------------
    # Phase 3d — cross-segment focal-length consistency refit.
    # ------------------------------------------------------------------
    # After Phase 3c lands the correct scene-level altitude, individual
    # segments may still fit to different f values depending on GCP
    # distribution — one may hit the ±_F_FRAC_RANGE bound while its
    # neighbours land at physically plausible values. Heterogeneous f
    # produces orthos at different ground scales; adjacent-ortho seams
    # then measure huge apparent shifts even when each pose is locally
    # good. Phase 3d picks the "best" (lowest-RMS, off-bound) segment's
    # f and refits every other segment with f pinned there. The refit
    # is accepted per segment only if RMS stays within a multiplier of
    # the original; rejections keep the original fit unchanged.
    _PHASE3D_RMS_REGRESSION_MAX = 1.5  # reject refit if RMS > 1.5× original
    _PHASE3D_F_BOUND_MARGIN_M = 0.005  # exclude fits within this much of bound as "at bound"
    scene_telem.phase3d_enabled = bool(
        seg_camera_params.get("shared_f_refit_enabled", True)
    )
    if scene_telem.phase3d_enabled and len(active_indices) >= 2:
        nominal_f = float(seg_camera_params["focal_length"])
        f_frac_range_val = float(
            seg_camera_params.get("f_frac_range") or 0.30
        )
        f_lo_bound = nominal_f * (1.0 - f_frac_range_val)
        f_hi_bound = nominal_f * (1.0 + f_frac_range_val)

        # Collect (seg_idx, f, rms, at_bound) for every successful seg.
        seg_fits: list = []
        for i in active_indices:
            fit = seg_fit_map.get(i)
            if fit is None:
                continue
            f_i = float(fit.params.f)
            rms_i = float(fit.reprojection_rms_px)
            at_bound = (
                abs(f_i - f_lo_bound) <= _PHASE3D_F_BOUND_MARGIN_M
                or abs(f_i - f_hi_bound) <= _PHASE3D_F_BOUND_MARGIN_M
            )
            seg_fits.append((i, f_i, rms_i, at_bound))

        f_values = [s[1] for s in seg_fits]
        if not f_values:
            scene_telem.phase3d_skipped_reason = "no_usable_fits"
        else:
            f_min = min(f_values)
            f_max = max(f_values)
            f_mean = sum(f_values) / len(f_values)
            spread_frac = (
                (f_max - f_min) / f_mean if f_mean > 0 else 0.0
            )
            min_spread = float(
                seg_camera_params.get("shared_f_refit_min_spread_frac", 0.02)
            )
            off_bound = [s for s in seg_fits if not s[3]]

            if spread_frac < min_spread:
                scene_telem.phase3d_skipped_reason = (
                    f"spread {spread_frac*100:.2f}% < "
                    f"threshold {min_spread*100:.2f}%"
                )
                print(
                    f"  [per_segment/phase3d] segments agree on f within "
                    f"{spread_frac*100:.2f}% — no refit needed"
                )
            elif not off_bound:
                scene_telem.phase3d_skipped_reason = (
                    "all_segments_at_f_bound"
                )
                print(
                    f"  [per_segment/phase3d] all segments' f values sit at "
                    f"the ±{f_frac_range_val*100:.0f}% bound — no signal "
                    f"to share"
                )
            else:
                # Pick the off-bound segment with lowest RMS as the source.
                source_i, shared_f, source_rms, _ = min(
                    off_bound, key=lambda s: s[2]
                )
                scene_telem.phase3d_shared_f_m = float(shared_f)
                scene_telem.phase3d_shared_f_source_seg = int(source_i)
                print(
                    f"  [per_segment/phase3d] shared f = {shared_f:.4f} m "
                    f"(from seg{source_i:02d}, RMS {source_rms:.2f}px); "
                    f"refitting segments whose f differs by > "
                    f"{min_spread*100:.2f}%"
                )

                # Refit each segment with fix_f=True at shared_f.
                accepted_ids: list = []
                for seg_idx, seg_f, seg_rms, seg_at_bound in seg_fits:
                    refit_entry = {
                        "seg_idx": int(seg_idx),
                        "original_f_m": float(seg_f),
                        "original_rms_px": float(seg_rms),
                        "refit_f_m": float(shared_f),
                        "refit_rms_px": None,
                        "accepted": False,
                        "reject_reason": None,
                    }
                    if seg_idx == source_i:
                        refit_entry["accepted"] = True
                        refit_entry["refit_rms_px"] = float(seg_rms)
                        refit_entry["reject_reason"] = "source_segment_no_refit"
                        scene_telem.phase3d_refit_per_segment.append(refit_entry)
                        continue
                    gcps = seg_gcps_map.get(seg_idx)
                    sub_path = seg_sub_path_map.get(seg_idx)
                    shape = seg_shape_map.get(seg_idx)
                    if gcps is None or sub_path is None or shape is None:
                        refit_entry["reject_reason"] = "missing_inputs"
                        scene_telem.phase3d_refit_per_segment.append(refit_entry)
                        continue
                    sf_w_ri, sf_h_ri = shape
                    original_fit = seg_fit_map[seg_idx]
                    # Build initial with Zs0 from the original fit (already
                    # authoritative via Phase 3c) and f pinned to shared.
                    try:
                        initial_ri = kh_panoramic.PanoramicParams(
                            **{k: getattr(original_fit.params, k) for k in (
                                "Xs0", "Ys0", "Zs0",
                                "omega0", "phi0", "kappa0",
                                "Xs1", "Ys1", "Zs1",
                                "omega1", "phi1", "kappa1",
                                "P", "f",
                            )}
                        )
                        initial_ri.f = float(shared_f)
                    except Exception as _exc:
                        refit_entry["reject_reason"] = f"initial_build_failed:{_exc}"
                        scene_telem.phase3d_refit_per_segment.append(refit_entry)
                        continue
                    # Stage-B-like refit: fix f AND Zs0, leave pose free.
                    kw_prior_ri = {}
                    _fprior = seg_camera_params.get("f_prior_frac_sigma")
                    if _fprior is not None:
                        kw_prior_ri["f_prior_frac_sigma"] = float(_fprior)
                    _frange = seg_camera_params.get("f_frac_range")
                    if _frange is not None:
                        kw_prior_ri["f_frac_range"] = float(_frange)
                    try:
                        refit = kh_panoramic.fit_panoramic(
                            sub_frame_gcps=gcps,
                            initial=initial_ri,
                            pixel_pitch=float(seg_camera_params["pixel_pitch"]),
                            image_width_px=int(sf_w_ri),
                            image_height_px=int(sf_h_ri),
                            nominal_f=float(shared_f),
                            max_iter=260,
                            loss="cauchy",
                            fix_zs0=True,
                            fix_f=True,
                            fix_velocities=False,
                            fix_rates=False,
                            fix_p=False,
                            **kw_prior_ri,
                        )
                    except Exception as _exc:
                        refit_entry["reject_reason"] = f"fit_exception:{_exc}"
                        scene_telem.phase3d_refit_per_segment.append(refit_entry)
                        continue
                    if not _fit_is_usable(refit):
                        refit_entry["reject_reason"] = "refit_unusable"
                        scene_telem.phase3d_refit_per_segment.append(refit_entry)
                        continue
                    refit_rms = float(refit.reprojection_rms_px)
                    refit_entry["refit_rms_px"] = refit_rms
                    rms_gate = max(
                        seg_rms * _PHASE3D_RMS_REGRESSION_MAX,
                        fit_rms_px_max * 2.0,
                    )
                    if refit_rms > rms_gate:
                        refit_entry["reject_reason"] = (
                            f"refit_rms_{refit_rms:.2f}_exceeds_gate_"
                            f"{rms_gate:.2f}"
                        )
                        scene_telem.phase3d_refit_per_segment.append(refit_entry)
                        print(
                            f"  [per_segment/phase3d] seg{seg_idx:02d} "
                            f"rejected: RMS {seg_rms:.2f} → {refit_rms:.2f}px "
                            f"(> {rms_gate:.2f}px gate)"
                        )
                        continue
                    # Accept: re-render the ortho with the refit params.
                    ortho_path_ri = seg_orthos_map[seg_idx]
                    final_bbox_ri, predicted_bbox_ri, gcp_bbox_ri = (
                        _resolve_render_bbox(
                            bbox_policy=bbox_policy,
                            gcps=gcps,
                            fit_params=refit.params,
                            pixel_pitch=float(seg_camera_params["pixel_pitch"]),
                            image_width_px=int(sf_w_ri),
                            image_height_px=int(sf_h_ri),
                        )
                    )
                    new_ortho = _do_mapproject(
                        refit.params, sub_path, final_bbox_ri,
                        int(sf_w_ri), int(sf_h_ri), ortho_path_ri,
                    )
                    if new_ortho is None:
                        refit_entry["reject_reason"] = "refit_mapproject_failed"
                        scene_telem.phase3d_refit_per_segment.append(refit_entry)
                        continue
                    seg_fit_map[seg_idx] = refit
                    refit_entry["accepted"] = True
                    accepted_ids.append(seg_idx)
                    # Re-save sidecar so the ortho cache key reflects the new fit.
                    meta_ri = {
                        "mode": mode_tag,
                        "local_crs": local_crs,
                        "preprocess_matcher": preprocess_matcher,
                        "matcher_cache_tag": matcher_cache_tag,
                        "fit_rms_px": refit_rms,
                        "n_gcps": int(gcps.shape[0]),
                        "zs0_m": float(refit.params.Zs0),
                        "altitude_source": scene_telem.altitude_source_used,
                        "phase3d_shared_f_m": float(shared_f),
                    }
                    base_corners_ri = seg_base_corners_map.get(seg_idx)
                    if base_corners_ri is not None:
                        _save_ortho_corners(
                            new_ortho, base_corners_ri, base_corners_ri,
                            metadata=meta_ri,
                        )
                    scene_telem.phase3d_refit_per_segment.append(refit_entry)
                    print(
                        f"  [per_segment/phase3d] seg{seg_idx:02d} "
                        f"refitted: RMS {seg_rms:.2f} → {refit_rms:.2f}px "
                        f"at shared f={shared_f:.4f}m"
                    )
                scene_telem.phase3d_applied = bool(accepted_ids)
                if accepted_ids:
                    print(
                        f"  [per_segment/phase3d] applied shared-f to "
                        f"segs {accepted_ids} (source=seg{source_i:02d})"
                    )
    elif scene_telem.phase3d_enabled:
        scene_telem.phase3d_skipped_reason = "fewer_than_2_active_segments"

    # ------------------------------------------------------------------
    # Phase 4 — joint BA refinement via ASP ``bundle_adjust``.
    # ------------------------------------------------------------------
    # Off-by-default, opt-in via profile. Runs AFTER Phase 3d's shared-f
    # refit and BEFORE seam QA. Seeds ASP's OpticalBar solve from the
    # 14-param fit (position + pose at scan-midpoint + shared f), supplies
    # per-sub-frame absolute GCPs from RoMa-vs-reference, and forces
    # ``--intrinsics-to-share focal_length`` so f cannot gauge-collapse
    # per camera. If ASP produces a better seam set than Phase 3d's
    # orthos (more accepted seams + higher mean ZNCC), the BA orthos
    # replace the Phase 3d ones; otherwise the Phase 3d orthos stay.
    scene_telem.phase4_enabled = bool(
        seg_camera_params.get("joint_ba_refinement", False)
    )
    if scene_telem.phase4_enabled and len(active_indices) >= 2:
        try:
            _phase4_result = _run_phase4_joint_ba(
                active_indices=active_indices,
                seg_orthos_map=seg_orthos_map,
                seg_fit_map=seg_fit_map,
                seg_shape_map=seg_shape_map,
                seg_gcps_map=seg_gcps_map,
                seg_sub_path_map=seg_sub_path_map,
                seg_base_corners_map=seg_base_corners_map,
                seg_camera_params=seg_camera_params,
                local_crs=local_crs,
                dem_path=dem_path,
                output_dir=output_dir,
                scene_id=scene_id,
                scene_telem=scene_telem,
                do_mapproject=_do_mapproject,
                measure_seams=_measure_segment_seams,
                bbox_policy=bbox_policy,
                mode_tag=mode_tag,
                preprocess_matcher=preprocess_matcher,
                matcher_cache_tag=matcher_cache_tag,
            )
            # _run_phase4_joint_ba mutates seg_orthos_map / seg_fit_map
            # in place when the refit is accepted. scene_telem records
            # the decision regardless.
        except Exception as _phase4_exc:
            print(f"  [per_segment/phase4] failed: {_phase4_exc}")
            scene_telem.phase4_applied = False
            scene_telem.phase4_skipped_reason = f"exception:{_phase4_exc}"
    elif scene_telem.phase4_enabled:
        scene_telem.phase4_skipped_reason = "fewer_than_2_active_segments"

    seg_orthos = [seg_orthos_map[i] for i in active_indices]
    seam_reports = _measure_segment_seams(seg_orthos)
    # Record seam reports in telemetry for Phase 3/Phase 5 attribution.
    try:
        scene_telem.seam_reports = [
            {k: v for k, v in r.items() if k != "overlap_bounds"}
            for r in seam_reports
        ]
    except Exception:
        pass
    seam_ok = True
    if seam_reports:
        print("  [per_segment] Seam QA:")
    for report in seam_reports:
        status = report.get("status")
        # "no_overlap" and "no_valid_overlap" are not QA failures: the
        # neighbouring segments' valid data simply doesn't meet (each
        # segment's thin panoramic strip can be positioned so that the
        # y-ranges barely touch). Skip the shift/ZNCC gate in that case
        # and let the winner-take-all blender stitch whatever is there.
        # Phase 3e: "low_texture" is the content analogue — the overlap
        # exists and is spatially well-aligned, but the pixels carry
        # insufficient signal (cloud, ocean, uniform desert) to support
        # a reliable ZNCC/phase-corr measurement. ASP's whole-strip
        # approach doesn't need a seam check; our per-segment path
        # treats a low-texture seam as unmeasurable-not-failed. Each
        # entry still records its numerical metrics in telemetry for
        # manual review.
        if status in (
            "no_overlap", "no_valid_overlap", "overlap_too_small", "low_texture",
        ):
            extra = ""
            if status == "low_texture":
                extra = (
                    f"  [zncc={report.get('zncc', 0):.3f} "
                    f"resp={report.get('response', 0):.3f} "
                    f"shift={report.get('phase_shift_px', 0):.2f}px]"
                )
            print(f"    seam {report['index']}: {status} (skipped){extra}")
            continue
        if status != "ok":
            seam_ok = False
            print(f"    seam {report['index']}: {status}")
            continue
        print(
            f"    seam {report['index']}: zncc={report['zncc']:.3f} "
            f"shift={report['phase_shift_px']:.2f}px "
            f"resp={report['response']:.3f}"
        )
        if not _seam_report_passes(report, seam_shift_px_max):
            seam_ok = False
    scene_telem.seam_qa_passed = bool(seam_ok)
    if not seam_ok:
        print(f"  [per_segment] Seam QA failed; whole-strip fallback")
        scene_telem.stitched_fallback_triggered = True
        scene_telem.fallback_reason = "seam_qa_failed"
        _persist_scene_telemetry(scene_telem, output_dir)
        return None

    valid_orthos = [o for o in seg_orthos if o]

    # Phase 3: smoothed displacement-field seam reconciliation.
    # Gated by profile flag `panoramic_seam_warp: true`; default OFF.
    # A latent bug in the half-warp direction or the TPS field evaluation
    # is amplifying seam offsets rather than reconciling them (observed a
    # 54 px → 950 px regression on Bahrain KH-9 D3C1213 seg 0-1). Leave
    # disabled until the TPS field / sign convention is diagnosed end-to-
    # end against a synthetic ortho pair.
    _PHASE3_ENABLED = bool(seg_camera_params.get("panoramic_seam_warp", False))
    scene_telem.phase3_seam_warp_enabled = bool(_PHASE3_ENABLED)
    phase3_matcher_runtime = None
    if _PHASE3_ENABLED:
        try:
            phase3_matcher_runtime = create_preprocess_matcher_runtime(preprocess_matcher)
        except Exception as e:
            print(f"  [phase3/seam] matcher runtime unavailable ({e}); skipping")
    if phase3_matcher_runtime is not None:
        sigmas_px = {
            idx: float(getattr(seg_fit_map.get(i), "reprojection_rms_px", 1.0))
            for idx, i in enumerate(active_indices)
            if i in seg_fit_map
        }
        phase3_feather_px = int(seg_camera_params.get(
            "panoramic_seam_feather_px", 400,
        ))
        phase3_smoothing = float(seg_camera_params.get(
            "panoramic_seam_tps_smoothing", 100.0,
        ))
        phase3_max_rms_m = float(seg_camera_params.get(
            "panoramic_seam_post_warp_rms_m_max", 30.0,
        ))
        try:
            # Phase 10 reject-if-worsens guard. Measure seam quality on
            # the pre-warp orthos, run TPS smoothing, then compare —
            # reject the warped outputs if any seam's phase shift got
            # larger. ``_phase3_smooth_seams`` rewrites orthos in place
            # at ``..._seam.tif`` alongside originals; we compare the
            # per-pair shift before flipping ``valid_orthos`` over.
            pre_warp_reports = _measure_segment_seams(valid_orthos)
            pre_shift_by_idx = {
                r.get("index"): float(r.get("phase_shift_px", float("inf")))
                for r in (pre_warp_reports or [])
                if r.get("status") == "ok"
            }
            smoothed = _phase3_smooth_seams(
                valid_orthos,
                sigmas_px=sigmas_px,
                output_dir=output_dir,
                scene_id=scene_id,
                matcher_name=preprocess_matcher,
                matcher_runtime=phase3_matcher_runtime,
                feather_px=phase3_feather_px,
                smoothing=phase3_smoothing,
                max_residual_m=phase3_max_rms_m,
            )
            if smoothed and any(p != orig for p, orig in zip(smoothed, valid_orthos)):
                post_reports = _measure_segment_seams(smoothed)
                worsened = []
                for r in (post_reports or []):
                    if r.get("status") != "ok":
                        continue
                    idx = r.get("index")
                    pre_shift = pre_shift_by_idx.get(idx)
                    post_shift = float(r.get("phase_shift_px", float("inf")))
                    if pre_shift is None or not np.isfinite(pre_shift):
                        continue
                    # Allow small numerical noise but reject anything
                    # that materially worsens a measurable seam.
                    if post_shift > pre_shift + 0.5:
                        worsened.append((idx, pre_shift, post_shift))
                if worsened:
                    details = ", ".join(
                        f"{idx}: {pre:.2f}→{post:.2f}px"
                        for idx, pre, post in worsened
                    )
                    print(
                        f"  [phase3/seam] REJECTED warp — seam(s) regressed "
                        f"post-warp: {details}. Keeping unwarped orthos."
                    )
                else:
                    print(
                        f"  [phase3/seam] smoothed {len(smoothed)} segments "
                        f"(feather={phase3_feather_px}px, "
                        f"smoothing={phase3_smoothing}) — no seam regressed"
                    )
                    scene_telem.phase3_seam_warp_fired = True
                    valid_orthos = smoothed
        except Exception as e:
            print(f"  [phase3/seam] smoothing failed ({e}); using unwarped orthos")
        finally:
            try:
                phase3_matcher_runtime.close()
            except Exception:
                pass

    # Diagnostic sidecar: last-wins VRT overlay for side-by-side visual
    # comparison against the distance-weighted blend in QGIS.  A misaligned
    # seam shows as a sharp step in the VRT but as smearing in the blend,
    # so seeing both is the clearest way to read failure modes.
    vrt_path = os.path.join(output_dir, f"{scene_id}_per_segment_vrt.vrt")
    try:
        from osgeo import gdal
        gdal.UseExceptions()
        if os.path.isfile(vrt_path):
            os.remove(vrt_path)
        gdal.BuildVRT(vrt_path, valid_orthos, options=gdal.BuildVRTOptions(
            resolution="highest", separate=False))
        print(f"  [per_segment] Diagnostic VRT written: {vrt_path}")
    except Exception as e:
        print(f"  [per_segment] VRT sidecar skipped: {e}")

    # Composite the orthorectified segments. For the 14p path we trust the
    # mapprojected valid-data mask more than the interpolated USGS corners:
    # the latter can be kilometres off and are only used to seed matching,
    # not to define the final footprint.
    applied_corners_list = None
    tif_path = os.path.join(output_dir, f"{scene_id}_per_segment.tif")
    if os.path.isfile(tif_path):
        tif_mtime = os.path.getmtime(tif_path)
        if any(os.path.getmtime(p) > tif_mtime + 1e-6 for p in valid_orthos):
            os.remove(tif_path)
    if not os.path.isfile(tif_path):
        blend_mode = str(
            seg_camera_params.get("panoramic_blend_mode", "argmax") or "argmax"
        ).strip().lower()
        if blend_mode not in ("argmax", "feather"):
            print(f"  [per_segment] warning: unknown panoramic_blend_mode "
                  f"{blend_mode!r}; falling back to argmax")
            blend_mode = "argmax"
        feather_px = int(
            seg_camera_params.get("panoramic_blend_feather_px", 200)
        )
        ok = _blend_segment_mosaic(
            valid_orthos, tif_path, applied_corners_list,
            blend_mode=blend_mode, feather_px=feather_px,
        )
        if not ok:
            print(f"  [per_segment] Blending failed")
            scene_telem.stitched_fallback_triggered = True
            scene_telem.fallback_reason = "blend_segment_mosaic_failed"
            _persist_scene_telemetry(scene_telem, output_dir)
            return None

    if os.path.isfile(tif_path):
        print(f"  [per_segment] Per-segment mosaic written: {tif_path}")
        scene_telem.final_output_path = tif_path
        _persist_scene_telemetry(scene_telem, output_dir)
        return tif_path

    return None
