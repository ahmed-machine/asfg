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
import json

import numpy as np

from .asp import find_asp_tool
from .mission_altitude import NOMINAL_ALTITUDE_M


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
    """Generate an ASP camera model (.tsai) from USGS corner coordinates.

    Dispatches on ``camera_params['type']``:
      * ``"opticalbar"`` (default) → panoramic bar-scan model (KH-4A/4B, KH-9 PC).
        Writes a sample .tsai with focal length + scan geometry, then runs
        ``cam_gen --camera-type opticalbar``.
      * ``"pinhole"``, ``"frame"`` → pinhole frame model (KH-9 MC).
      * ``"linescan"`` → KH-7 strip / slit-scan. No rigorous public model
        exists; treated as pinhole at the stitched-strip level per
        `memory/cross_kh_system_audit.md`.

    Parameters
    ----------
    image_path : str
        Path to the stitched / georeferenced image.
    camera_params : dict
        Camera intrinsics. ``focal_length`` + ``pixel_pitch`` are required
        for all types; OpticalBar also needs ``scan_time``, ``speed``,
        ``forward_tilt``, ``scan_dir``, ``motion_compensation_factor``.
    corners : dict or list
        Corner coordinates as {'nw': (lat, lon), 'ne': ..., 'se': ..., 'sw': ...}
        or as a flat list [NW_lon, NW_lat, NE_lon, NE_lat, SE_lon, SE_lat, SW_lon, SW_lat].
    dem_path : str, optional
        Path to a DEM for refining the camera model.
    altitude_m : float, optional
        Per-mission orbital altitude seed for OpticalBar cam_gen. Unused for
        pinhole — ASP's pinhole solver derives Z from the 4-corner geometry
        + DEM. Defaults to the 170 km nominal for OpticalBar.

    Returns
    -------
    str or None
        Path to generated .tsai camera file, or None if ASP unavailable.
    """
    cam_gen = find_asp_tool("cam_gen")
    if cam_gen is None:
        print("  [camera_model] ASP cam_gen not found, skipping camera model generation")
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

    output_tsai = os.path.splitext(image_path)[0] + ".tsai"
    camera_type = str(camera_params.get("type", "opticalbar")).strip().lower()

    if camera_type in ("pinhole", "frame", "linescan"):
        # The pinhole 4-corner cam_gen path is currently unreliable for
        # satellite-altitude scenes — the P4P solve on 4 coplanar ground
        # points + known focal length has multiple near-equal branches
        # (observed ASP converging to 30-90 km altitudes on 167 km KH-7,
        # producing 180°-flipped orthos with 0.7 px reprojection error).
        # Neither `--reference-dem --refine-camera` nor `--cam-height
        # --cam-weight` disambiguated reliably in testing. Need either
        # (a) > 4 GCPs from feature matching against the reference, or
        # (b) ASP's `--camera-type linescan` with proper frame index.
        # Until one of those is wired, skip ASP ortho generation for
        # non-panoramic scenes; the pipeline falls back to the
        # 4-corner affine georef which at least has correct orientation.
        print(f"  [camera_model] pinhole/linescan ASP ortho skipped "
              f"(P4P branch ambiguity, see camera_model.py). Using georef-only path.")
        return None

    # OpticalBar path (default).
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(image_path)
    if ds is None:
        print(f"  [camera_model] Cannot open {image_path}")
        return None
    width, height = ds.RasterXSize, ds.RasterYSize
    ds = None

    output_dir = os.path.dirname(image_path)
    sample_tsai = os.path.join(output_dir, "sample_opticalbar.tsai")
    corners_lc = None
    if isinstance(corners, dict):
        corners_lc = {k.lower(): v for k, v in corners.items()}
    _write_sample_tsai(sample_tsai, width, height, camera_params,
                       corners=corners_lc, altitude_m=altitude_m)

    ok = _run_cam_gen_subprocess(
        cam_gen, sample_tsai, ll_str, image_path, output_tsai, dem_path,
    )
    if not ok:
        return None

    print(f"  [camera_model] Camera model written to {os.path.basename(output_tsai)}")
    if os.path.isfile(sample_tsai):
        os.remove(sample_tsai)

    return output_tsai


def _run_cam_gen_subprocess(cam_gen, sample_tsai, ll_str, image_path,
                             output_tsai, dem_path, pixel_str=None,
                             timeout_s=None):
    """Invoke ASP cam_gen for an OpticalBar camera.

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


def _run_cam_gen_pinhole_subprocess(cam_gen, focal_length, pixel_pitch, ll_str,
                                     image_path, output_tsai, dem_path,
                                     altitude_m=None, camera_center_llh=None,
                                     timeout_s=180.0):
    """Invoke ASP cam_gen for a pinhole / frame camera.

    Covers the non-panoramic KH systems:
      * KH-7 GAMBIT-1 (strip / slit-scan) — approximated as pinhole at the
        stitched-strip level; per the cross-KH audit
        (`memory/cross_kh_system_audit.md`) no rigorous public sensor model
        exists for KH-7, so published DEM work falls back to pinhole / RPC.
      * KH-9 MC HEXAGON Mapping Camera — authentic frame camera
        (Itek 12" Petzval); Dehecq et al. 2020 use exactly this workflow.

    Unlike OpticalBar, pinhole cam_gen doesn't need a sample .tsai file —
    focal length + pixel pitch go on the command line and the 4-corner
    `--lon-lat-values` solve determines pose.

    Without an altitude hint the P4P solve on 4 coplanar ground points is
    weakly constrained — observed ASP converging to 30-90 km altitudes
    for what are really 167 km (KH-7) and 170 km (KH-9 MC) orbits, which
    produces 180°-flipped orthos (wrong P4P branch). Supplying
    ``camera_center_llh`` (or ``altitude_m`` which is combined with the
    scene centroid) picks the correct branch.
    """
    cmd = [
        cam_gen,
        "--camera-type", "pinhole",
        "--focal-length", str(float(focal_length)),
        "--pixel-pitch", str(float(pixel_pitch)),
        "--lon-lat-values", ll_str,
    ]
    if camera_center_llh is not None:
        lon, lat, h = camera_center_llh
        cmd.extend(["--camera-center-llh", f"{float(lon)} {float(lat)} {float(h)}"])
    elif altitude_m is not None:
        # Scene centroid from the 4 corners in ll_str: alternate "lon lat lon lat ..."
        parts = ll_str.split()
        try:
            lons = [float(parts[i]) for i in range(0, len(parts), 2)]
            lats = [float(parts[i + 1]) for i in range(0, len(parts), 2)]
            lon_c = sum(lons) / len(lons)
            lat_c = sum(lats) / len(lats)
            cmd.extend([
                "--camera-center-llh",
                f"{lon_c} {lat_c} {float(altitude_m)}",
            ])
        except (ValueError, IndexError):
            pass

    cmd.extend([image_path, "-o", output_tsai])
    if dem_path and os.path.isfile(dem_path):
        cmd.extend(["--reference-dem", dem_path, "--refine-camera"])
    else:
        # cam_gen requires either a DEM or a datum; fall back to WGS84 (sea level).
        cmd.extend(["--datum", "WGS_1984"])

    env = os.environ.copy()
    if "ISISROOT" not in env:
        asp_root = os.path.dirname(os.path.dirname(cam_gen))
        if os.path.isfile(os.path.join(asp_root, "IsisPreferences")):
            env["ISISROOT"] = asp_root

    print(f"  [camera_model] Running pinhole cam_gen for "
          f"{os.path.basename(image_path)} (f={focal_length} m, pitch={pixel_pitch} m)")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, env=env,
        )
        if result.returncode != 0:
            print(f"  [camera_model] pinhole cam_gen failed: {result.stderr[:500]}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  [camera_model] pinhole cam_gen error: {e}")
        return False

    if not os.path.isfile(output_tsai):
        print(f"  [camera_model] pinhole cam_gen did not produce {output_tsai}")
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


def _orthonormalize_rotation(mat: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(mat, dtype=np.float64))
    out = u @ vt
    if np.linalg.det(out) < 0:
        u[:, -1] *= -1.0
        out = u @ vt
    return out


def _rotation_to_quaternion(mat: np.ndarray) -> np.ndarray:
    m = _orthonormalize_rotation(mat)
    trace = float(np.trace(m))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        ], dtype=np.float64)
    else:
        idx = int(np.argmax(np.diag(m)))
        if idx == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = np.array([
                (m[2, 1] - m[1, 2]) / s,
                0.25 * s,
                (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s,
            ], dtype=np.float64)
        elif idx == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q = np.array([
                (m[0, 2] - m[2, 0]) / s,
                (m[0, 1] + m[1, 0]) / s,
                0.25 * s,
                (m[1, 2] + m[2, 1]) / s,
            ], dtype=np.float64)
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q = np.array([
                (m[1, 0] - m[0, 1]) / s,
                (m[0, 2] + m[2, 0]) / s,
                (m[1, 2] + m[2, 1]) / s,
                0.25 * s,
            ], dtype=np.float64)
    return q / max(np.linalg.norm(q), 1e-12)


def _quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / max(np.linalg.norm(q), 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _slerp_rotation(r0: np.ndarray, r1: np.ndarray, t: float) -> np.ndarray:
    q0 = _rotation_to_quaternion(r0)
    q1 = _rotation_to_quaternion(r1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return _quaternion_to_rotation(q)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    s0 = math.cos(theta) - dot * math.sin(theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return _quaternion_to_rotation((s0 * q0) + (s1 * q1))


def interpolate_camera_pose(neighbor_tsai_paths: list[str],
                            output_tsai_path: str,
                            *,
                            alpha: float = 0.5,
                            base_tsai_path: str | None = None) -> str | None:
    """Write an OpticalBar .tsai with iC/iR interpolated from two neighbors."""
    if len(neighbor_tsai_paths) != 2:
        return None
    left = _parse_opticalbar_tsai(neighbor_tsai_paths[0])
    right = _parse_opticalbar_tsai(neighbor_tsai_paths[1])
    if left is None or right is None:
        return None
    base = _parse_opticalbar_tsai(base_tsai_path) if base_tsai_path else None
    static = base or left
    alpha = max(0.0, min(1.0, float(alpha)))
    iC = ((1.0 - alpha) * left["iC"]) + (alpha * right["iC"])
    iR = _slerp_rotation(left["iR"], right["iR"], alpha)

    image_size = static.get("image_size", (0.0, 0.0))
    image_center = static.get(
        "image_center", (float(image_size[0]) / 2.0, float(image_size[1]) / 2.0))
    os.makedirs(os.path.dirname(output_tsai_path) or ".", exist_ok=True)
    with open(output_tsai_path, "w") as f:
        f.write(_OPTICALBAR_TEMPLATE.format(
            width=float(image_size[0]),
            height=float(image_size[1]),
            cx=float(image_center[0]),
            cy=float(image_center[1]),
            pitch=float(static.get("pitch", left.get("pitch", 0.0))),
            focal_length=float(static.get("f", left.get("f", 0.0))),
            scan_time=float(static.get("scan_time", left.get("scan_time", 0.0))),
            forward_tilt=float(static.get("forward_tilt", left.get("forward_tilt", 0.0))),
            iC=" ".join(f"{v:.10g}" for v in iC),
            iR=" ".join(f"{v:.10g}" for v in iR.reshape(-1)),
            speed=float(static.get("speed", left.get("speed", 0.0))),
            mean_elev=float(static.get("mean_surface_elevation",
                                       left.get("mean_surface_elevation", 0.0))),
            mcf=float(static.get("motion_compensation_factor",
                                 left.get("motion_compensation_factor", 1.0))),
            scan_dir=str(static.get("scan_dir", left.get("scan_dir", "right"))),
        ))
    return output_tsai_path


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




def mapproject_image(image_path, camera_path, dem_path=None, output_path=None,
                     resolution=None, t_srs="EPSG:3857"):
    """Map-project an image using its OpticalBar camera model.

    Removes panoramic distortion and (with DEM) terrain parallax.

    Calls ASP's ``bin/mapproject`` shell wrapper (which exports ISISROOT
    + unsets conflicting GDAL/PROJ env vars).
    """
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_mapprojected{ext}"

    if os.path.isfile(output_path):
        print(f"  [camera_model] Mapprojected file already exists: {os.path.basename(output_path)}")
        return output_path

    mapproj = find_asp_tool("mapproject")
    if mapproj is None:
        print("  [camera_model] ASP mapproject not found, skipping")
        return None

    cmd = [mapproj]
    if resolution is not None:
        cmd.extend(["--tr", str(resolution)])
    cmd.extend(["--t_srs", t_srs])

    n_threads = max(1, (os.cpu_count() or 4) - 1)
    cmd.extend(["--threads", str(n_threads), "--cache-size-mb", "4096"])

    if dem_path and os.path.isfile(dem_path):
        cmd.append(dem_path)
    else:
        cmd.append("WGS84")
    cmd.extend([image_path, camera_path, output_path])

    print(f"  [camera_model] Running mapproject -> {os.path.basename(output_path)}")
    print(f"  [camera_model] cmd: {' '.join(cmd)}")
    if resolution and resolution < 1.0:
        _timeout = 18000  # 5 h (KH-7 / KH-9 PC native — should be remote anyway)
    elif resolution and resolution < 2.5:
        _timeout = 14400  # 4 h (KH-4 native / KH-9 PC reference)
    else:
        _timeout = 7200  # 2 h (KH-4 reference res)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_timeout)
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
