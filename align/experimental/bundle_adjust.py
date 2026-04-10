"""Strip-level bundle adjustment using ASP.

Uses ASP bundle_adjust with OpticalBar camera models to jointly solve
camera poses across a strip of adjacent frames. RoMa matches against
the reference serve as ground control points; inter-frame matches in
overlap zones serve as tie points.

This is an optional high-accuracy mode activated with --bundle-adjust.
"""

import os
import subprocess
import tempfile

import numpy as np

from preprocess.asp import find_asp_tool
from preprocess.camera_model import generate_camera


def _write_gcp_file(gcps, output_path, image_name):
    """Write GCPs in ASP format for bundle_adjust.

    ASP GCP format (one per line):
        lon lat height sigma_lon sigma_lat sigma_height image_name col row sigma_col sigma_row
    """
    with open(output_path, "w") as f:
        for gcp in gcps:
            lon, lat = gcp["lon"], gcp["lat"]
            col, row = gcp["col"], gcp["row"]
            sigma = gcp.get("sigma", 10.0)
            f.write(f"{lon} {lat} 0.0  {sigma} {sigma} 50.0  "
                    f"{image_name} {col} {row} 1.5 1.5\n")


def _read_focal_length(tsai_path):
    """Parse the focal length from an ASP .tsai camera file.

    Returns the fitted `f = ...` value in metres, or None if not found.
    """
    try:
        with open(tsai_path) as fp:
            for line in fp:
                line = line.strip()
                if line.startswith("f ") or line.startswith("f="):
                    # Format: "f = 0.6096" or "f = 0.6096 0"
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        return float(parts[1].strip().split()[0])
    except (OSError, ValueError):
        pass
    return None


def run_strip_bundle_adjustment(frames, camera_params, corners_list,
                                gcps_per_frame=None, dem_path=None,
                                output_dir=None, match_prefix=None,
                                solve_intrinsics=False):
    """Run ASP bundle_adjust on a strip of frames.

    Parameters
    ----------
    frames : list of str
        Paths to frame images in strip order.
    camera_params : dict
        Camera intrinsics (shared across frames).
    corners_list : list of dict
        Per-frame corner coordinates for cam_gen.
    gcps_per_frame : list of list of dict, optional
        Per-frame GCPs from neural matching. Each GCP is
        {"lon": float, "lat": float, "col": float, "row": float, "sigma": float}.
    dem_path : str, optional
        DEM for height constraint.
    output_dir : str, optional
        Output directory for adjusted cameras.
    match_prefix : str, optional
        Path prefix for pre-computed .match files from RoMa.
        When provided, bundle_adjust skips ipfind and uses these
        match files directly via --match-files-prefix.
    solve_intrinsics : bool
        When True, pass --solve-intrinsics --intrinsics-to-float
        focal_length to bundle_adjust. Enables the adaptive focal length
        fit from 2OC (Hou et al. 2023). Fitted focal lengths are logged
        per frame on return.

    Returns
    -------
    list of str or None
        Paths to adjusted camera files, or None on failure.
    """
    ba_tool = find_asp_tool("bundle_adjust")
    if ba_tool is None:
        print("  [BundleAdjust] ASP bundle_adjust not found")
        return None

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="ba_")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Generate camera models for each frame
    camera_files = []
    for i, (frame, corners) in enumerate(zip(frames, corners_list)):
        cam = generate_camera(frame, camera_params, corners, dem_path)
        if cam is None:
            print(f"  [BundleAdjust] cam_gen failed for frame {i}")
            return None
        camera_files.append(cam)

    # Step 2: Write GCP files
    gcp_files = []
    if gcps_per_frame:
        for i, (frame, gcps) in enumerate(zip(frames, gcps_per_frame)):
            if gcps:
                gcp_path = os.path.join(output_dir, f"frame_{i}.gcp")
                _write_gcp_file(gcps, gcp_path, os.path.basename(frame))
                gcp_files.append(gcp_path)

    # Step 3: Run bundle_adjust
    ba_prefix = os.path.join(output_dir, "ba")
    cmd = [ba_tool]
    for f in frames:
        cmd.append(f)
    for c in camera_files:
        cmd.append(c)
    for g in gcp_files:
        cmd.append(g)

    cmd.extend([
        "-t", "opticalbar",
        "--inline-adjustments",
        "--camera-weight", "0",
        "--datum", "WGS84",
        "-o", ba_prefix,
    ])

    if solve_intrinsics:
        # 2OC paper: adaptive focal length fit per strip is worth ~30-45%
        # of reported accuracy gain. ASP floats `focal_length` across the
        # strip while keeping extrinsics the primary free variables.
        cmd.extend([
            "--solve-intrinsics",
            "--intrinsics-to-float", "focal_length",
        ])
        nominal_f = camera_params.get("focal_length")
        if nominal_f:
            print(f"  [BundleAdjust] solve_intrinsics=ON; nominal f={nominal_f:.6f} m")

    if dem_path and os.path.isfile(dem_path):
        cmd.extend(["--heights-from-dem", dem_path])

    # Use pre-computed RoMa match files if available (skips ipfind)
    if match_prefix:
        match_dir = os.path.dirname(match_prefix) or "."
        prefix_base = os.path.basename(match_prefix)
        has_matches = any(
            f.startswith(prefix_base) and f.endswith(".match")
            for f in os.listdir(match_dir)
        ) if os.path.isdir(match_dir) else False
        if has_matches:
            cmd.extend(["--match-files-prefix", match_prefix])
            print(f"  [BundleAdjust] Using pre-computed match files: {match_prefix}")
        else:
            print(f"  [BundleAdjust] No match files at {match_prefix}, "
                  "falling back to ipfind")

    # ASP tools (bundle_adjust, mapproject) require ISISROOT to find their
    # bundled IsisPreferences file. Inject defensively if the user's shell
    # has not set it. See preprocess/camera_model.py:mapproject_image().
    env = os.environ.copy()
    if "ISISROOT" not in env:
        asp_root = os.path.dirname(os.path.dirname(ba_tool))
        if os.path.isfile(os.path.join(asp_root, "IsisPreferences")):
            env["ISISROOT"] = asp_root

    print(f"  [BundleAdjust] Running on {len(frames)} frames...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
        if result.returncode != 0:
            print(f"  [BundleAdjust] Failed: {result.stderr[:500]}")
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  [BundleAdjust] Error: {e}")
        return None

    # Step 4: Collect adjusted camera files
    adjusted = []
    for cam in camera_files:
        # bundle_adjust with --inline-adjustments modifies cameras in place
        if os.path.isfile(cam):
            adjusted.append(cam)
        else:
            print(f"  [BundleAdjust] Missing adjusted camera: {cam}")
            return None

    print(f"  [BundleAdjust] Successfully adjusted {len(adjusted)} cameras")

    # When --solve-intrinsics is active, log the fitted focal lengths so
    # we can compare against 2OC paper Table 6 (per-sub-image f drift
    # ~0.1% from nominal for KH-4B).
    if solve_intrinsics:
        nominal_f = camera_params.get("focal_length", 0.0)
        print("  [BundleAdjust] Fitted focal lengths (solve_intrinsics):")
        for cam, frame in zip(adjusted, frames):
            fitted = _read_focal_length(cam)
            if fitted is not None and nominal_f:
                delta_pct = 100.0 * (fitted - nominal_f) / nominal_f
                print(f"    {os.path.basename(frame)}: f={fitted:.6f} m "
                      f"(Δ={delta_pct:+.3f}% vs nominal)")
            elif fitted is not None:
                print(f"    {os.path.basename(frame)}: f={fitted:.6f} m")

    return adjusted
