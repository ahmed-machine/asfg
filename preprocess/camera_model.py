"""ASP OpticalBar camera model generation and map-projection.

Uses NASA Ames Stereo Pipeline (ASP) cam_gen + mapproject to produce
physically correct orthorectification of panoramic KH satellite imagery,
removing bow-tie distortion, S-shaped IMC residuals, and (with DEM)
terrain parallax.

Falls back gracefully if ASP is not installed.
"""

import os
import subprocess
import tempfile

import numpy as np

from .asp import find_asp_tool


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


def _write_sample_tsai(path, width, height, camera_params, corners=None, mean_elev=0.0):
    """Write a sample OpticalBar .tsai file for cam_gen.

    corners, if provided, is a dict with 'nw','ne','se','sw' keys of (lat, lon)
    tuples, used to compute an approximate initial camera position in ECEF.
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

    alt = 170000  # ~170km orbital altitude for KH
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


def generate_camera(image_path, camera_params, corners, dem_path=None):
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
    _write_sample_tsai(sample_tsai, width, height, camera_params, corners=corners_lc)

    # Run cam_gen
    output_tsai = os.path.splitext(image_path)[0] + ".tsai"
    cmd = [
        cam_gen,
        "--sample-file", sample_tsai,
        "--camera-type", "opticalbar",
        "--lon-lat-values", ll_str,
        image_path,
        "-o", output_tsai,
    ]
    if dem_path and os.path.isfile(dem_path):
        cmd.extend(["--reference-dem", dem_path, "--refine-camera"])

    # See mapproject_image() for the rationale — set ISISROOT defensively so
    # subprocess inherits a workable environment regardless of the user's shell.
    env = os.environ.copy()
    if "ISISROOT" not in env:
        asp_root = os.path.dirname(os.path.dirname(cam_gen))
        if os.path.isfile(os.path.join(asp_root, "IsisPreferences")):
            env["ISISROOT"] = asp_root

    print(f"  [camera_model] Running cam_gen for {os.path.basename(image_path)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
        if result.returncode != 0:
            print(f"  [camera_model] cam_gen failed: {result.stderr[:500]}")
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  [camera_model] cam_gen error: {e}")
        return None

    if not os.path.isfile(output_tsai):
        print(f"  [camera_model] cam_gen did not produce {output_tsai}")
        return None

    print(f"  [camera_model] Camera model written to {os.path.basename(output_tsai)}")
    # Clean up sample file
    if os.path.isfile(sample_tsai):
        os.remove(sample_tsai)

    return output_tsai


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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
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


def opticalbar_per_segment_precorrect(sub_frames, camera_params, strip_corners,
                                      output_dir, dem_path=None, resolution=None,
                                      t_srs="EPSG:3857", scene_id=None):
    """Run cam_gen + mapproject independently on each sub-frame, then build a VRT mosaic.

    This is the 2OC §3.1 approach: instead of stitching N sub-frames first
    and fitting one camera, fit one camera per sub-frame so each segment's
    extrinsics absorb its own offset. The final outputs are mosaic'd via
    ``gdalbuildvrt`` (no re-projection — they share ``t_srs``).

    Parameters
    ----------
    sub_frames : list[str]
        Paths to the sub-frame TIFFs in **stitch order** (left-to-right
        along the scan axis after any KH-4-style reversal).
    camera_params : dict
        Strip-level camera intrinsics. ``scan_time`` is divided by
        ``len(sub_frames)`` for each segment so each segment's OpticalBar
        models the fraction of the strip it actually scanned.
    strip_corners : dict
        Strip-level corners ``{'NW': ..., 'NE': ..., 'SE': ..., 'SW': ...}``.
    output_dir : str
        Directory for per-segment .tsai/.tif outputs and the final .vrt.
    dem_path, resolution, t_srs : forwarded to mapproject_image.
    scene_id : str, optional
        Used in the final VRT filename. Defaults to the basename of
        ``sub_frames[0]`` minus the trailing segment letter.

    Returns
    -------
    str or None
        Path to the per-segment VRT mosaic, or None if any cam_gen /
        mapproject step fails. Per-segment outputs are NOT cleaned up on
        failure so the caller can inspect them.
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

    print(f"  [per_segment] Processing {n} sub-frames for {scene_id}")

    # Each segment scans 1/N of the full strip's scan period.
    seg_camera_params = copy.deepcopy(camera_params)
    nominal_scan_time = float(seg_camera_params.get("scan_time", 0.0) or 0.0)
    if nominal_scan_time > 0:
        seg_camera_params["scan_time"] = nominal_scan_time / n

    seg_orthos = []
    for i, sub_path in enumerate(sub_frames):
        if not os.path.isfile(sub_path):
            print(f"  [per_segment] Missing sub-frame {sub_path}")
            return None

        seg_corners = interpolate_segment_corners(strip_corners, n, i)
        # Normalize for logging only (preserve original case for downstream)
        _seg_lc = {str(k).lower(): v for k, v in seg_corners.items()}
        print(f"  [per_segment] segment {i}/{n}: {os.path.basename(sub_path)}")
        print(f"    corners NW={_seg_lc['nw']} NE={_seg_lc['ne']}")

        # cam_gen writes the .tsai next to the input image.
        cam_path = generate_camera(sub_path, seg_camera_params, seg_corners,
                                   dem_path=dem_path)
        if cam_path is None:
            print(f"  [per_segment] cam_gen failed for segment {i}")
            return None

        seg_ortho_path = os.path.join(
            output_dir, f"{scene_id}_seg{i:02d}_ortho.tif"
        )
        ortho = mapproject_image(
            sub_path, cam_path, dem_path=dem_path,
            output_path=seg_ortho_path, resolution=resolution, t_srs=t_srs,
        )
        if ortho is None:
            print(f"  [per_segment] mapproject failed for segment {i}")
            return None
        seg_orthos.append(ortho)

    # Build the final VRT mosaic. gdalbuildvrt handles overlap-zone
    # priority by default (later inputs win); we pass them in stitch order
    # so the rightmost segment overrides on overlaps.
    vrt_path = os.path.join(output_dir, f"{scene_id}_per_segment.vrt")
    cmd = ["gdalbuildvrt", "-overwrite", vrt_path, *seg_orthos]
    print(f"  [per_segment] gdalbuildvrt -> {os.path.basename(vrt_path)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  [per_segment] gdalbuildvrt failed: {result.stderr[:500]}")
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  [per_segment] gdalbuildvrt error: {e}")
        return None

    if not os.path.isfile(vrt_path):
        print("  [per_segment] gdalbuildvrt did not produce output")
        return None

    print(f"  [per_segment] Per-segment mosaic VRT written: {vrt_path}")
    return vrt_path
