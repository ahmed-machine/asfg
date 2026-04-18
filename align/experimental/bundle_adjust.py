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


def _write_absolute_gcp_file(
    per_image_gcps,
    local_crs,
    out_path,
    pixel_sigma=1.5,
    xy_sigma_m=10.0,
    z_sigma_m=20.0,
):
    """Write ASP `bundle_adjust` GCP file from per-image absolute GCPs.

    Each GCP from :func:`preprocess.kh_panoramic.extract_reference_gcps`
    is a RoMa-derived match of a raw sub-frame pixel against a
    georeferenced reference ortho. Per-segment processing means every
    such GCP is visible in exactly ONE sub-frame image, so each output
    row carries a single image observation. ASP accepts multi-image
    observations by appending ``image col row sigma_col sigma_row``
    quintuples, but we don't need that here.

    ASP GCP format (whitespace-separated, one GCP per line)::

        <id> <lat> <lon> <height_above_datum>
             <sigma_east_m> <sigma_north_m> <sigma_up_m>
             <image_name> <col> <row> <sigma_col_px> <sigma_row_px>

    Parameters
    ----------
    per_image_gcps : list[tuple[str, numpy.ndarray]]
        ``(image_name, gcps)`` pairs. Each ``gcps`` is an (N, 5) array
        with columns ``[col, row, X_local, Y_local, Z_local]``.
        ``image_name`` is the basename ASP will match against the
        ``bundle_adjust`` image list.
    local_crs : str
        pyproj CRS string for the GCPs' XY columns (e.g. an EPSG:326xx
        UTM zone). Converted to WGS84 lat/lon before writing.
    out_path : str
        Destination CSV.
    pixel_sigma : float
        Per-observation pixel uncertainty (column/row). RoMa is
        sub-pixel so 1.5 px is conservative.
    xy_sigma_m : float
        Per-GCP horizontal uncertainty (covers DEM lateral error +
        reference-image georef error).
    z_sigma_m : float
        Per-GCP vertical uncertainty.

    Returns
    -------
    int
        Number of GCPs written.
    """
    from pyproj import Transformer

    tr = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)
    n_written = 0
    gcp_id = 0
    with open(out_path, "w") as fh:
        for image_name, gcps in per_image_gcps:
            if gcps is None:
                continue
            arr = np.asarray(gcps)
            if arr.size == 0 or arr.shape[-1] < 5:
                continue
            for row in arr:
                col_px = float(row[0])
                row_px = float(row[1])
                x_local = float(row[2])
                y_local = float(row[3])
                z_local = float(row[4])
                lon, lat = tr.transform(x_local, y_local)
                fh.write(
                    f"{gcp_id} {lat:.8f} {lon:.8f} {z_local:.3f} "
                    f"{xy_sigma_m:.3f} {xy_sigma_m:.3f} {z_sigma_m:.3f} "
                    f"{image_name} {col_px:.3f} {row_px:.3f} "
                    f"{pixel_sigma:.3f} {pixel_sigma:.3f}\n"
                )
                gcp_id += 1
                n_written += 1
    return n_written


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
                                solve_intrinsics=False,
                                initial_tsai_paths=None,
                                absolute_gcp_file=None,
                                shared_intrinsics=False,
                                intrinsics_limits=None,
                                reference_terrain_weight=None,
                                disparity_list=None,
                                robust_threshold=None,
                                camera_weight=10):
    """Run ASP bundle_adjust on a strip of frames.

    Parameters
    ----------
    frames : list of str
        Paths to frame images in strip order.
    camera_params : dict
        Camera intrinsics (shared across frames).
    corners_list : list of dict
        Per-frame corner coordinates for cam_gen. Ignored when
        ``initial_tsai_paths`` supplies pre-built cameras.
    gcps_per_frame : list of list of dict, optional
        DEPRECATED — was the old per-frame GCP dict format. Kept for
        backward compatibility with callers that haven't migrated to
        ``absolute_gcp_file``. Phase 4 prefers the absolute GCP file.
    dem_path : str, optional
        DEM for height or reference-terrain constraint.
    output_dir : str, optional
        Output directory for adjusted cameras.
    match_prefix : str, optional
        Path prefix for pre-computed .match files from RoMa.
        When provided, bundle_adjust skips ipfind and uses these
        match files directly via --match-files-prefix.
    solve_intrinsics : bool
        When True, pass --solve-intrinsics --intrinsics-to-float
        focal_length to bundle_adjust.
    initial_tsai_paths : list of str, optional
        Phase 4: pre-built OpticalBar .tsai seeds (one per frame) to
        use instead of running ``cam_gen`` fresh. Typically produced
        from the per-segment 14-param fit via
        ``preprocess.kh_panoramic.pano_params_to_opticalbar_tsai``.
        When supplied, ``cam_gen`` is skipped entirely.
    absolute_gcp_file : str, optional
        Phase 4: path to an ASP GCP file written by
        ``_write_absolute_gcp_file``. Gives BA absolute-lat/lon
        observations per raw-image pixel, preventing the gauge-collapse
        mode that killed previous inter-frame-tie-only attempts.
    shared_intrinsics : bool
        Phase 4: when True, pass
        ``--intrinsics-to-share "focal_length"``. Only effective with
        ``solve_intrinsics=True``. Forces all cameras to share a single
        fitted focal length — closes the per-camera f-collapse loophole
        recorded in memory/per_segment_ab_results.md.
    intrinsics_limits : tuple of floats, optional
        Phase 4: per-intrinsic fractional bounds, e.g. ``(0.92, 1.08)``
        for ±8 % on focal length. Only effective with
        ``solve_intrinsics=True``. Passed to
        ``--intrinsics-limits "lo hi ..."``.
    reference_terrain_weight : float, optional
        Phase 4: when > 0 AND ``dem_path`` is supplied AND
        ``disparity_list`` is non-empty, pass
        ``--reference-terrain DEM --reference-terrain-weight W`` instead
        of the weaker ``--heights-from-dem``. Dehecq et al. 2020 used
        weight 1000 to break the altitude/focal-length gauge on KH-9.
        If ``disparity_list`` is missing, falls back to
        ``--heights-from-dem`` (ASP refuses ``--reference-terrain``
        without disparity pairs).
    disparity_list : list of str, optional
        Phase 4: paths to ASP stereo disparity ``.tif`` files
        (one per image pair) required by ``--reference-terrain``.
        Typically produced by ``parallel_stereo`` before BA.
    robust_threshold : float, optional
        Phase 4: override ASP's default ``--robust-threshold`` (0.5 px).
        Raise to 2-3 px on cross-temporal imagery where RoMa-filtered
        tie points already have sub-pixel precision but reference-
        content mismatch adds measurement noise.
    camera_weight : float
        Weight on the camera-pose prior. 0 lets extrinsics float
        freely (Dehecq's approach); 10 keeps pose near the seed
        (legacy default; useful when seeds are noisy).

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

    # Step 1: Camera seeds — either caller-supplied (Phase 4 preferred)
    # or freshly generated via cam_gen (legacy path).
    camera_files = []
    if initial_tsai_paths is not None:
        if len(initial_tsai_paths) != len(frames):
            print(
                f"  [BundleAdjust] initial_tsai_paths length "
                f"{len(initial_tsai_paths)} ≠ frames length {len(frames)}"
            )
            return None
        for p in initial_tsai_paths:
            if not os.path.isfile(p):
                print(f"  [BundleAdjust] seed camera missing: {p}")
                return None
            camera_files.append(p)
        print(
            f"  [BundleAdjust] using {len(camera_files)} caller-supplied "
            f"OpticalBar seeds (skipping cam_gen)"
        )
    else:
        for i, (frame, corners) in enumerate(zip(frames, corners_list)):
            cam = generate_camera(frame, camera_params, corners, dem_path)
            if cam is None:
                print(f"  [BundleAdjust] cam_gen failed for frame {i}")
                return None
            camera_files.append(cam)

    # Step 2: GCP inputs.
    gcp_files = []
    if absolute_gcp_file is not None:
        if not os.path.isfile(absolute_gcp_file):
            print(
                f"  [BundleAdjust] absolute_gcp_file not found: "
                f"{absolute_gcp_file}"
            )
            return None
        gcp_files.append(absolute_gcp_file)
    elif gcps_per_frame:
        # Legacy per-frame dict GCPs — not used by Phase 4 but kept
        # for backward compatibility. Requires the dead _write_gcp_file
        # helper which Phase 4 removed; callers hitting this branch
        # should migrate to absolute_gcp_file.
        print(
            "  [BundleAdjust] legacy gcps_per_frame dict format is no "
            "longer supported; pass absolute_gcp_file instead"
        )
        return None

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
        "--camera-weight", str(camera_weight),
        "--datum", "WGS84",
        "-o", ba_prefix,
    ])

    if solve_intrinsics:
        cmd.extend([
            "--solve-intrinsics",
            "--intrinsics-to-float", "focal_length",
        ])
        if shared_intrinsics:
            # Forces all cameras to share the fitted focal length
            # instead of each drifting independently. The key fix for
            # the gauge-collapse failure mode in prior BA attempts.
            cmd.extend(["--intrinsics-to-share", "focal_length"])
        if intrinsics_limits:
            # Format: "lo_f hi_f lo_cx hi_cx lo_cy hi_cy ..."
            limits_str = " ".join(f"{v}" for v in intrinsics_limits)
            cmd.extend(["--intrinsics-limits", limits_str])
        nominal_f = camera_params.get("focal_length")
        if nominal_f:
            print(
                f"  [BundleAdjust] solve_intrinsics=ON; shared={shared_intrinsics} "
                f"limits={intrinsics_limits!r}; nominal f={nominal_f:.6f} m"
            )

    if dem_path and os.path.isfile(dem_path):
        want_reference_terrain = (
            reference_terrain_weight and reference_terrain_weight > 0
            and disparity_list
        )
        if want_reference_terrain:
            # Stronger DEM coupling: penalises reprojection residuals
            # against the DEM, not just heights. Dehecq's gauge-breaker.
            # ASP requires a disparity list paired with --reference-terrain,
            # so we only take this path when the caller has stereo disparities
            # available. Without them ASP hard-errors.
            cmd.extend([
                "--reference-terrain", dem_path,
                "--reference-terrain-weight", str(reference_terrain_weight),
                "--disparity-list", " ".join(disparity_list),
            ])
            print(
                f"  [BundleAdjust] --reference-terrain-weight "
                f"{reference_terrain_weight} with {len(disparity_list)} disparities"
            )
        else:
            cmd.extend([
                "--heights-from-dem", dem_path,
                "--heights-from-dem-uncertainty", "10.0",
            ])
            if reference_terrain_weight and reference_terrain_weight > 0:
                # Caller asked for reference-terrain but didn't supply
                # disparities — fall back to heights-from-dem with a note.
                print(
                    "  [BundleAdjust] reference-terrain requested but no "
                    "disparity_list supplied; falling back to --heights-from-dem"
                )

    if robust_threshold is not None:
        cmd.extend(["--robust-threshold", str(float(robust_threshold))])

    # Reuse cached IP / match files from a previous BA run if available.
    # ipfind is the most expensive phase (~5 min for 3 large frames);
    # skipping it on re-runs saves most of the BA wall clock.
    cached_matches = any(
        f.endswith(".match") for f in os.listdir(output_dir)
    ) if os.path.isdir(output_dir) else False
    if cached_matches:
        cmd.append("--force-reuse-match-files")
        print(f"  [BundleAdjust] Reusing cached match files from {output_dir}")

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
    # ASP's ipfind uses OpenMP; set thread count for parallel IP detection.
    env.setdefault("OMP_NUM_THREADS", str(min(4, os.cpu_count() or 4)))

    print(f"  [BundleAdjust] Running on {len(frames)} frames...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
        if result.returncode != 0:
            print(f"  [BundleAdjust] Failed: {result.stderr[:500]}")
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  [BundleAdjust] Error: {e}")
        return None

    # Step 4: Collect adjusted camera files.
    # Despite --inline-adjustments, ASP writes adjusted cameras to the output
    # prefix (ba/ba-<name>.tsai) rather than modifying the originals in place
    # for OpticalBar cameras.  Prefer the ba-prefixed files.
    adjusted = []
    for cam in camera_files:
        ba_cam = os.path.join(output_dir, "ba-" + os.path.basename(cam))
        if os.path.isfile(ba_cam):
            adjusted.append(ba_cam)
        elif os.path.isfile(cam):
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
