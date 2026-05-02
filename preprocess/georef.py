"""Rough georeferencing, coarse alignment cropping, and Sentinel-2 reference fetch."""

import json
import os
import subprocess
import urllib.request
from collections import defaultdict

import numpy as np

import time

def _run_cmd(cmd, check=True, retries=5):
    """Run a shell command, retrying on failure, raising if *check* is True."""
    # Ensure GDAL does its own native retrying on HTTP errors
    import os
    os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
    os.environ["GDAL_HTTP_RETRY_DELAY"] = "3"
    os.environ["CPL_VSIL_CURL_USE_HEAD"] = "NO"
    os.environ["CURL_CA_BUNDLE"] = "/etc/ssl/cert.pem"
    
    for attempt in range(retries):
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # GDAL sometimes returns 0 even if it fails to open VSI files, but prints Warnings
        has_error = result.returncode != 0 or "Warning 1: HTTP response code" in result.stderr or "Warning 1: Can't open" in result.stderr
        
        if not has_error:
            return result
            
        # If it failed, wait and retry
        if attempt < retries - 1:
            print(f"Warning: Command failed (attempt {attempt+1}/{retries}): {' '.join(cmd)}\nRetrying in 5 seconds...")
            time.sleep(5)
    
    # If we exhausted all retries
    if check and has_error:
        raise RuntimeError(f"Command failed after {retries} attempts: {' '.join(cmd)}\n{result.stderr}")
    return result


def _panoramic_gsd_correction(t_norm):
    """GSD scaling factor for a KH-4 panoramic camera at normalized position t.

    The KH-4 panoramic camera has a 70-degree scan arc. Ground Sampling
    Distance varies from 1.0x at nadir (t=0.5) to ~1.52x at the edges
    (t=0 or t=1) in the along-track direction, and ~1.23x in cross-track.
    This means geographic spacing is compressed at the edges relative to
    the film.  We model this as a cosine-based correction on the scan angle.

    Args:
        t_norm: Normalized position along the panoramic axis [0, 1].

    Returns:
        Fractional correction to apply to geographic position (0 at center,
        positive at edges meaning the geographic coordinate should be pushed
        outward).
    """
    import math
    half_fov_rad = math.radians(35)  # 70° total FOV
    # Angle from nadir for this position
    theta = (t_norm - 0.5) * 2 * half_fov_rad
    # GSD ratio = 1/cos(theta) for a flat-earth approximation
    # The correction needed is the integral of this vs linear spacing
    # Simplified: geographic position shifts outward by ~4% at edges
    return 0.04 * (2 * t_norm - 1) ** 2 * (1 if t_norm > 0.5 else -1)


def _interpolate_corner(c0, c1, t, panoramic_axis=False):
    """Linearly interpolate between two (lat, lon) corners.

    If panoramic_axis is True, applies panoramic GSD correction to
    account for non-uniform pixel spacing in panoramic cameras.
    """
    lat = c0[0] + t * (c1[0] - c0[0])
    lon = c0[1] + t * (c1[1] - c0[1])
    if panoramic_axis:
        corr = _panoramic_gsd_correction(t)
        lat += corr * (c1[0] - c0[0])
        lon += corr * (c1[1] - c0[1])
    return (lat, lon)


def georef_with_corners(input_path: str, output_path: str, corners: dict,
                        panoramic: bool = False):
    """Georeference an image using 4-corner GCPs.

    For panoramic cameras (KH-4), adds intermediate GCPs along the long
    axis with panoramic distortion correction and uses a polynomial warp
    (order 2) instead of affine to better model the non-uniform GSD.

    Args:
        input_path: Ungeoreferenced input TIFF.
        output_path: Output georeferenced TIFF in EPSG:3857.
        corners: Dict with NW, NE, SE, SW keys, each (lat, lon) tuple.
        panoramic: If True, add intermediate GCPs with panoramic correction.
    """
    if os.path.exists(output_path):
        print(f"  [skip] Georef output already exists: {output_path}")
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get image dimensions
    info_result = _run_cmd(["gdalinfo", "-json", input_path])
    info = json.loads(info_result.stdout)
    width = info["size"][0]
    height = info["size"][1]
    print(f"  Image size: {width} x {height}")

    nw_lat, nw_lon = corners["NW"]
    ne_lat, ne_lon = corners["NE"]
    se_lat, se_lon = corners["SE"]
    sw_lat, sw_lon = corners["SW"]

    # Build GCP list
    gcps = [
        ("0", "0", str(nw_lon), str(nw_lat)),
        (str(width), "0", str(ne_lon), str(ne_lat)),
        (str(width), str(height), str(se_lon), str(se_lat)),
        ("0", str(height), str(sw_lon), str(sw_lat)),
    ]

    # For panoramic images, detect the long axis and add intermediate GCPs
    # with panoramic distortion correction
    warp_order = "1"
    aspect_ratio = width / max(height, 1)
    if panoramic and aspect_ratio > 3.0:
        # Long axis is horizontal (NW→NE / SW→SE)
        n_intermediate = 5  # Add 5 points along each edge + midline
        for i in range(1, n_intermediate + 1):
            t = i / (n_intermediate + 1)
            px_x = str(int(round(t * width)))
            # Top edge (NW → NE)
            lat, lon = _interpolate_corner(corners["NW"], corners["NE"], t,
                                           panoramic_axis=True)
            gcps.append((px_x, "0", str(lon), str(lat)))
            # Bottom edge (SW → SE)
            lat, lon = _interpolate_corner(corners["SW"], corners["SE"], t,
                                           panoramic_axis=True)
            gcps.append((px_x, str(height), str(lon), str(lat)))
            # Middle row
            top_lat, top_lon = _interpolate_corner(corners["NW"], corners["NE"], t,
                                                   panoramic_axis=True)
            bot_lat, bot_lon = _interpolate_corner(corners["SW"], corners["SE"], t,
                                                   panoramic_axis=True)
            mid_lat = (top_lat + bot_lat) / 2
            mid_lon = (top_lon + bot_lon) / 2
            gcps.append((px_x, str(height // 2), str(mid_lon), str(mid_lat)))

        warp_order = "2"  # Polynomial order 2 for panoramic distortion
        print(f"  Panoramic mode: {len(gcps)} GCPs, polynomial order {warp_order}")
    elif panoramic and aspect_ratio < 1 / 3.0:
        # Long axis is vertical (NW→SW / NE→SE)
        n_intermediate = 5
        for i in range(1, n_intermediate + 1):
            t = i / (n_intermediate + 1)
            px_y = str(int(round(t * height)))
            # Left edge (NW → SW)
            lat, lon = _interpolate_corner(corners["NW"], corners["SW"], t,
                                           panoramic_axis=True)
            gcps.append(("0", px_y, str(lon), str(lat)))
            # Right edge (NE → SE)
            lat, lon = _interpolate_corner(corners["NE"], corners["SE"], t,
                                           panoramic_axis=True)
            gcps.append((str(width), px_y, str(lon), str(lat)))
            # Middle column
            left_lat, left_lon = _interpolate_corner(corners["NW"], corners["SW"], t,
                                                     panoramic_axis=True)
            right_lat, right_lon = _interpolate_corner(corners["NE"], corners["SE"], t,
                                                       panoramic_axis=True)
            mid_lat = (left_lat + right_lat) / 2
            mid_lon = (left_lon + right_lon) / 2
            gcps.append((str(width // 2), px_y, str(mid_lon), str(mid_lat)))

        warp_order = "2"
        print(f"  Panoramic mode: {len(gcps)} GCPs, polynomial order {warp_order}")

    # Assign GCPs with gdal_translate
    temp_gcp = output_path.replace(".tif", "_gcp.tif")

    cmd_translate = ["gdal_translate", "-a_srs", "EPSG:4326"]
    for px_x, px_y, lon, lat in gcps:
        cmd_translate.extend(["-gcp", px_x, px_y, lon, lat])
    cmd_translate.extend([input_path, temp_gcp])
    _run_cmd(cmd_translate)

    # Warp to EPSG:3857
    cmd_warp = [
        "gdalwarp",
        "-s_srs", "EPSG:4326",
        "-t_srs", "EPSG:3857",
        "-order", warp_order,
        "-r", "lanczos",
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=2",
        "-co", "TILED=YES",
        "-co", "BIGTIFF=IF_SAFER",
        "-dstalpha",
        temp_gcp,
        output_path,
    ]
    try:
        _run_cmd(cmd_warp)
    finally:
        if os.path.exists(temp_gcp):
            os.remove(temp_gcp)

    print(f"  Georeferenced: {output_path}")
    return output_path


def georef_with_bbox(input_path: str, output_path: str,
                     west: float, north: float, east: float, south: float):
    """Georeference an image using a bounding box (ullr).

    Simpler method for single-frame images where a bbox is sufficient.
    """
    if os.path.exists(output_path):
        print(f"  [skip] Georef output already exists: {output_path}")
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    temp_path = output_path.replace(".tif", "_temp_georef.tif")

    # Assign WGS84 coordinates
    _run_cmd([
        "gdal_translate",
        "-a_srs", "EPSG:4326",
        "-a_ullr", str(west), str(north), str(east), str(south),
        input_path,
        temp_path,
    ])

    # Reproject to Web Mercator
    _run_cmd([
        "gdalwarp",
        "-s_srs", "EPSG:4326",
        "-t_srs", "EPSG:3857",
        "-r", "lanczos",
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=2",
        "-co", "TILED=YES",
        "-co", "BIGTIFF=IF_SAFER",
        "-dstalpha",
        temp_path,
        output_path,
    ])

    if os.path.exists(temp_path):
        os.remove(temp_path)

    print(f"  Georeferenced: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Coarse-align and crop: find overlap at low res, shift, then crop
# ---------------------------------------------------------------------------

# The ELoFTR translation estimator and its tunables now live in
# align/scale.py so both the preprocess coarse-align step (this module)
# and the in-pipeline coarse-offset step (align/coarse.py) share a single
# implementation and a single set of thresholds.
from align.scale import (
    eloftr_translation_estimate as _eloftr_translation_estimate,
    _ELOFTR_MIN_AGREEMENT,
    _ELOFTR_MIN_MATCHES,
    _ELOFTR_KP_CONF_MIN,
)


def coarse_align_and_crop(
    target_path: str,
    reference_path: str,
    output_path: str,
    coarse_res: float = 50.0,
    margin_m: float = 10000.0,
    target_bbox_wgs: tuple = None,
    crop: bool = True,
    model_cache=None,
    params=None,
    neighbour_shifts_m: list[tuple[float, float]] | None = None,
    return_details: bool = False,
) -> str | tuple[str, dict] | None:
    """Find the coarse offset between a wide target and a reference, then crop.

    Film frames (especially KH-4 panoramic strips) cover huge areas but only
    a small portion overlaps with the reference. USGS corner coordinates can
    be 20km+ off, so we can't just crop to the reference bbox directly.

    Algorithm:
      1. Read both at low resolution (~50m/px) on a *shared* metric canvas
         (union of the target+reference extents in a common UTM CRS).
      2. Run tiled ELoFTR (MatchAnything-EfficientLoFTR via HF) on the two
         co-gridded rasters; aggregate matches across tiles; estimate the
         translation as the median of per-match (ref - tgt) pixel deltas.
      3. Apply the detected offset to the target's geotransform.
      4. Optionally crop the shifted target to the reference bbox + margin.

    The matcher upgrade (replacing OpenCV ``matchTemplate`` on binary land
    masks) gives the coarse stage a notion of which shoreline it's looking
    at — NCC's "best peak in a binary correlation surface" was easily
    fooled by similar-looking coastlines (e.g., Saudi mainland matching
    Bahrain's north shore). When ELoFTR can't pin position with enough
    self-consistency, this function abstains (returns ``None``); the
    caller in ``process.py`` records ``coarse_align_status="abstained"``
    and ``generate_manifest`` hard-skips the entity on profiles with
    unreliable USGS corners.

    Args:
        target_path: Wide georeferenced target (e.g. ASP ortho mosaic).
        reference_path: Smaller georeferenced reference image.
        output_path: Cropped + shifted output.
        coarse_res: Resolution for coarse matching (metres/pixel).
        margin_m: Padding around reference bbox (metres).
        target_bbox_wgs: Optional (west, south, east, north) in EPSG:4326 from
            USGS corners.  Used to check expected overlap with reference and
            reject spurious template matches.
        crop: When False, returns the shifted target without cropping (used
            by ``_coarse_align_ortho_to_sidecar`` to keep the canonical
            footprint and only emit a sidecar shift).
        model_cache: Optional ``align.models.ModelCache``. Pass one in for
            batch processing so ELoFTR loads once across many calls; the
            function lazy-creates and disposes its own cache when omitted.

    Returns:
        Path to the cropped output (or shifted output when ``crop=False``),
        or ``None`` on abstain / failure.
    """
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from pyproj import CRS, Transformer

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Pre-check: if we know the target's expected footprint (from USGS corners),
    # verify it actually overlaps the reference before doing expensive work.
    if target_bbox_wgs is not None:
        with rasterio.open(reference_path) as src_ref_quick:
            from rasterio.warp import transform_bounds as _tb
            ref_wgs = _tb(src_ref_quick.crs, "EPSG:4326", *src_ref_quick.bounds)
        tw, ts, te, tn = target_bbox_wgs
        rw, rs, re, rn = ref_wgs
        inter_w = max(0, min(te, re) - max(tw, rw))
        inter_h = max(0, min(tn, rn) - max(ts, rs))
        ref_area = max(1e-10, (re - rw) * (rn - rs))
        overlap_frac = (inter_w * inter_h) / ref_area
        if overlap_frac < 0.05:
            print(f"  [coarse_crop] Frame bbox has <5% overlap with reference "
                  f"({overlap_frac*100:.1f}%), skipping coarse crop")
            return None

    with rasterio.open(target_path) as src_tgt, rasterio.open(reference_path) as src_ref:
        # Determine a common metric CRS (UTM from reference center)
        ref_bounds = src_ref.bounds
        ref_cx = (ref_bounds.left + ref_bounds.right) / 2
        ref_cy = (ref_bounds.bottom + ref_bounds.top) / 2

        if src_ref.crs.is_geographic:
            utm_zone = int((ref_cx + 180) / 6) + 1
            hemisphere = "north" if ref_cy >= 0 else "south"
            work_crs = CRS.from_proj4(
                f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84")
        else:
            work_crs = src_ref.crs

        # Transform reference bounds to work CRS
        t_ref = Transformer.from_crs(src_ref.crs, work_crs, always_xy=True)
        ref_left, ref_bottom = t_ref.transform(ref_bounds.left, ref_bounds.bottom)
        ref_right, ref_top = t_ref.transform(ref_bounds.right, ref_bounds.top)

        # Transform target bounds to work CRS
        tgt_bounds = src_tgt.bounds
        t_tgt = Transformer.from_crs(src_tgt.crs, work_crs, always_xy=True)
        tgt_left, tgt_bottom = t_tgt.transform(tgt_bounds.left, tgt_bounds.bottom)
        tgt_right, tgt_top = t_tgt.transform(tgt_bounds.right, tgt_bounds.top)

        # Build a shared canvas (union of both extents) at coarse_res so
        # the ELoFTR matcher sees both rasters on the same grid; per-match
        # (kp_ref - kp_tgt) pixel deltas convert directly to metric shifts
        # without per-raster transform bookkeeping.
        union_left = min(ref_left, tgt_left)
        union_bottom = min(ref_bottom, tgt_bottom)
        union_right = max(ref_right, tgt_right)
        union_top = max(ref_top, tgt_top)

        # Pad the canvas by the stacked-NCC search radius so the ref-content
        # template never spans the full canvas (which would force the
        # ``th >= H or tw >= W`` abstain in ``run_stacked_coarse_align``).
        # When ref's vertical extent dominates the union (KH-4B sub-frame +
        # 1976 KH-9 strip on Bahrain), the template otherwise locks at full
        # canvas height. Padding gives the matchTemplate slide head-room
        # in both axes and accommodates peaks ``radius_m`` away from the
        # USGS-implied anchor without falling off the canvas.
        ncc_radius_m = 0.0
        if params is not None:
            ncc_radius_m = float(getattr(params.camera, "coarse_ncc_search_radius_m", 0.0) or 0.0)
        if ncc_radius_m > 0:
            union_left -= ncc_radius_m
            union_right += ncc_radius_m
            union_bottom -= ncc_radius_m
            union_top += ncc_radius_m

        # Build the coarse arrays via GDAL warp to the common grid
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tgt_coarse = os.path.join(tmpdir, "tgt.tif")
            ref_coarse = os.path.join(tmpdir, "ref.tif")

            shared_te = [
                "-te", str(union_left), str(union_bottom),
                str(union_right), str(union_top),
                "-tr", str(coarse_res), str(coarse_res),
            ]
            _run_cmd([
                "gdalwarp", "-t_srs", work_crs.to_proj4(),
                *shared_te,
                "-r", "average",
                "-co", "COMPRESS=LZW",
                target_path, tgt_coarse,
            ])
            _run_cmd([
                "gdalwarp", "-t_srs", work_crs.to_proj4(),
                *shared_te,
                "-r", "average",
                "-co", "COMPRESS=LZW",
                reference_path, ref_coarse,
            ])

            with rasterio.open(tgt_coarse) as ds_t, rasterio.open(ref_coarse) as ds_r:
                arr_tgt = ds_t.read(1).astype(np.float32)
                arr_ref = ds_r.read(1).astype(np.float32)

        if (np.count_nonzero(arr_tgt > 15) < 100
                or np.count_nonzero(arr_ref > 15) < 100):
            print(f"  [coarse_crop] Insufficient content for matching")
            return None

        # Run tiled ELoFTR on the shared canvas. Returns metric (dx, dy)
        # plus an agreement score derived from the per-match MAD; abstain
        # when matches are too few or too dispersed to support a confident
        # translation.
        eloftr_result = _eloftr_translation_estimate(
            arr_ref, arr_tgt, coarse_res, model_cache=model_cache,
        )

        def _single_shot_ok(res) -> bool:
            """Single-shot ELoFTR cleared all gates: returned a result,
            passed the agreement floor, and within the 50 km sanity
            ceiling. Anything else is treated as 'abstained' for
            fallback purposes."""
            if res is None:
                return False
            _dx, _dy, _n, _ag = res
            if _ag < _ELOFTR_MIN_AGREEMENT:
                return False
            if float(np.hypot(_dx, _dy)) > 50000:
                return False
            return True

        # Stacked NCC -> ELoFTR -> phase-correlate fallback. Engaged
        # whenever single-shot ELoFTR fails ANY of its gates (null
        # result, agreement < floor, magnitude > 50 km) AND the camera
        # profile has unreliable USGS corners (KH-4/KH-7/KH-8). KH-9
        # leaves ``coarse_ncc_search_radius_m`` at 0 so this branch is
        # skipped — the agreement-floor abstain still fires for KH-9
        # via the legacy gates below.
        if (not _single_shot_ok(eloftr_result)
                and params is not None
                and not params.camera.usgs_corners_reliable
                and params.camera.coarse_ncc_search_radius_m > 0):
            if eloftr_result is not None:
                _dx0, _dy0, _n0, _ag0 = eloftr_result
                print(f"  [coarse_crop] single-shot ELoFTR weak "
                      f"(n={_n0}, agreement={_ag0:.2f}); engaging "
                      f"stacked fallback")
            # Forward the rejected single-shot ELoFTR (dx, dy) so the
            # stack's strip-prior tie-breaker can corroborate the prior
            # against this frame's own (weak) measurement. Without this
            # signal an N=1 prior can mis-shift the frame by tens of km
            # when intra-strip USGS errors disagree; with it the prior
            # is only believed when it lines up with the local ELoFTR
            # estimate within the per-axis strip-coherence bounds.
            single_shot_dxdy_m = None
            if eloftr_result is not None:
                _dx0, _dy0, _n0, _ag0 = eloftr_result
                single_shot_dxdy_m = (float(_dx0), float(_dy0))
            from preprocess.coarse_align_ncc_stack import run_stacked_coarse_align
            stacked = run_stacked_coarse_align(
                arr_ref, arr_tgt, coarse_res,
                work_crs=work_crs,
                union_bounds=(union_left, union_bottom, union_right, union_top),
                target_bbox_wgs=target_bbox_wgs,
                params=params,
                model_cache=model_cache,
                target_path=target_path,
                reference_path=reference_path,
                neighbour_shifts_m=neighbour_shifts_m,
                single_shot_dxdy_m=single_shot_dxdy_m,
            )
            if stacked is not None:
                eloftr_result = stacked
        if eloftr_result is None:
            print(f"  [coarse_crop] ELoFTR coarse match abstained "
                  f"(insufficient or dispersed matches)")
            return None
        dx_m, dy_m, n_matches, agreement = eloftr_result

        total_offset = float(np.hypot(dx_m, dy_m))
        print(f"  [coarse_crop] ELoFTR coarse offset: dx={dx_m:+.0f}m, "
              f"dy={dy_m:+.0f}m (total={total_offset:.0f}m, "
              f"matches={n_matches}, agreement={agreement:.2f})")

        # ELoFTR has its own internal "no peak" path (returns None earlier
        # when matches are insufficient by count). Two further gates:
        #
        #   1. Agreement floor: per-match MAD-derived agreement must clear
        #      the ``_ELOFTR_MIN_AGREEMENT`` threshold. agreement≈0 means
        #      161 keypoint pairs may be scattered across competing
        #      geographies (Bahrain coast + Saudi coast), so the median
        #      translation is meaningless. v5 found this exact mode and
        #      wrote a 12.9 km bogus shift to the sidecar.
        #
        #   2. Sanity ceiling on shift magnitude: a >50 km translation
        #      implies the matcher latched onto the wrong geography
        #      entirely.
        if agreement < _ELOFTR_MIN_AGREEMENT:
            print(f"  [coarse_crop] Match dispersion too high "
                  f"(agreement={agreement:.2f} < {_ELOFTR_MIN_AGREEMENT}), "
                  f"abstaining")
            return None
        if total_offset > 50000:
            print(f"  [coarse_crop] Shift too large "
                  f"({total_offset/1000:.0f}km > 50km), abstaining")
            return None

    # Apply the shift to the target by adjusting its geotransform, then crop
    # to the reference bbox + margin
    from osgeo import gdal
    gdal.UseExceptions()

    ds = gdal.Open(target_path)
    gt = list(ds.GetGeoTransform())
    tgt_crs_str = ds.GetProjection()
    ds = None

    # We need to shift in the target's native CRS
    if "3857" in tgt_crs_str:
        # EPSG:3857 — shift directly in metres
        gt[0] += dx_m  # shift origin X
        gt[3] += dy_m  # shift origin Y
    elif "4326" in tgt_crs_str:
        # EPSG:4326 — convert metre shift to degrees (approximate)
        gt[0] += dx_m / 111000.0
        gt[3] += dy_m / 111000.0
    else:
        # Assume metric CRS
        gt[0] += dx_m
        gt[3] += dy_m

    # Write shifted version
    shifted_path = output_path.replace(".tif", "_shifted.tif") if crop else output_path
    ds_in = gdal.Open(target_path)
    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.CreateCopy(shifted_path, ds_in, options=[
        "COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES",
    ])
    ds_out.SetGeoTransform(gt)
    ds_out.FlushCache()
    ds_out = None
    ds_in = None

    if not crop:
        # Shift-only path: the shifted TIFF IS the output; no cropping.
        if not os.path.exists(output_path):
            return None
        if return_details:
            return output_path, {
                "dx_m": float(dx_m),
                "dy_m": float(dy_m),
                "n_matches": int(n_matches),
                "agreement": float(agreement),
                "validated": int(n_matches) > 0,
            }
        return output_path

    # Crop to reference bbox + margin (in reference CRS, typically EPSG:4326)
    with rasterio.open(reference_path) as src_ref:
        rb = src_ref.bounds
        if src_ref.crs.is_geographic:
            margin_deg = margin_m / 111000.0
            crop_w = rb.left - margin_deg
            crop_s = rb.bottom - margin_deg
            crop_e = rb.right + margin_deg
            crop_n = rb.top + margin_deg
            te_srs = "EPSG:4326"
        else:
            crop_w = rb.left - margin_m
            crop_s = rb.bottom - margin_m
            crop_e = rb.right + margin_m
            crop_n = rb.top + margin_m
            te_srs = src_ref.crs.to_string()

    # -overwrite: gdalwarp defaults to refusing to write over an existing
    # file. On cache-warm re-runs the prior cropped ortho is still in
    # place from an earlier run, which caused KH-9 PC e2e_v17 to abort
    # mid-pipeline ("Output dataset ... exists ... Please delete").
    _run_cmd([
        "gdalwarp",
        "-overwrite",
        "-te_srs", te_srs,
        "-te", str(crop_w), str(crop_s), str(crop_e), str(crop_n),
        "-co", "COMPRESS=LZW", "-co", "TILED=YES", "-co", "BIGTIFF=YES",
        shifted_path, output_path,
    ])

    # Clean up shifted intermediate
    if os.path.exists(shifted_path):
        os.remove(shifted_path)

    if not os.path.exists(output_path):
        print(f"  [coarse_crop] Crop produced no output")
        return None

    # Verify content
    with rasterio.open(output_path) as ds:
        data = ds.read(1, out_shape=(ds.height // max(1, ds.height // 200),
                                     ds.width // max(1, ds.width // 200)))
        valid_pct = np.count_nonzero(data > 10) / max(data.size, 1) * 100

    print(f"  [coarse_crop] Cropped to reference bbox + {margin_m/1000:.0f}km margin "
          f"({valid_pct:.0f}% valid content)")
    if return_details:
        return output_path, {
            "dx_m": float(dx_m),
            "dy_m": float(dy_m),
            "n_matches": int(n_matches),
            "agreement": float(agreement),
            "validated": int(n_matches) > 0,
        }
    return output_path


# ---------------------------------------------------------------------------
# Sentinel-2 reference image auto-fetch via Element 84 Earth Search STAC API
# ---------------------------------------------------------------------------

STAC_API = os.environ.get("STAC_API_URL", "https://earth-search.aws.element84.com/v1")
COLLECTION = "sentinel-2-l2a"


def _stac_search_sentinel2(bbox, max_cloud_cover):
    """Search the STAC API and return all features in bbox under the
    cloud-cover threshold, sorted newest-first."""
    west, south, east, north = bbox
    search_body = json.dumps({
        "collections": [COLLECTION],
        "bbox": [west, south, east, north],
        "query": {"eo:cloud_cover": {"lt": max_cloud_cover}},
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        "limit": 100,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{STAC_API}/search",
        data=search_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        results = json.loads(resp.read().decode("utf-8"))
    features = results.get("features", [])
    if not features:
        raise RuntimeError(
            f"No Sentinel-2 images found with <{max_cloud_cover}% cloud cover. "
            f"Try increasing --max-cloud-cover or provide a reference manually."
        )
    return features


def _select_best_per_mgrs_tile(features):
    """Group features by MGRS tile code and pick the lowest-cloud scene
    per tile."""
    tiles = defaultdict(list)
    for f in features:
        grid = f["properties"].get("grid:code", f["id"][:10])
        tiles[grid].append(f)
    selected = []
    for grid, items in sorted(tiles.items()):
        best = min(items, key=lambda x: x["properties"].get("eo:cloud_cover", 99))
        selected.append(best)
        cloud = best["properties"].get("eo:cloud_cover", "?")
        date = best["properties"].get("datetime", "?")[:10]
        print(f"    Tile {grid}: {best['id']} ({date}, {cloud}% cloud)")
    print(f"  Selected {len(selected)} tile(s) for composite")
    return selected


def _extract_band_urls(item, band_keys):
    """Return vsicurl URLs for each band in order, or None if any band
    is missing (caller decides whether to skip or fail)."""
    assets = item.get("assets", {})
    urls = []
    for key in band_keys:
        if key not in assets:
            return None
        urls.append(f"/vsicurl/{assets[key]['href']}")
    return urls


def _build_single_tile_reference(item, bbox, output_path, band_keys, temp_dir):
    """Single-tile path: build VRT of bands, warp to EPSG:3857 at bbox."""
    import uuid
    urls = _extract_band_urls(item, band_keys)
    if urls is None:
        missing = [k for k in band_keys if k not in item.get("assets", {})]
        raise RuntimeError(f"Missing bands {missing} in {item['id']}")
    uid = str(uuid.uuid4())[:8]
    vrt_path = os.path.join(temp_dir, f"sentinel2_ref_{uid}.vrt")
    _run_cmd(["gdalbuildvrt", "-separate", vrt_path] + urls)
    west, south, east, north = bbox
    _run_cmd([
        "gdalwarp",
        "-t_srs", "EPSG:3857",
        "-te_srs", "EPSG:4326",
        "-te", str(west), str(south), str(east), str(north),
        "-r", "bilinear",
        "-co", "COMPRESS=LZW", "-co", "PREDICTOR=2", "-co", "TILED=YES",
        vrt_path, output_path,
    ])
    if os.path.exists(vrt_path):
        os.remove(vrt_path)


def _build_multi_tile_mosaic_reference(selected, bbox, output_path, band_keys, temp_dir):
    """Multi-tile path: per-tile VRT → per-tile warp → mosaic."""
    import uuid
    tile_vrts = []
    for i, item in enumerate(selected):
        urls = _extract_band_urls(item, band_keys)
        if urls is None:
            print(f"    WARNING: missing bands in {item['id']}, skipping tile")
            continue
        uid = str(uuid.uuid4())[:8]
        tile_vrt = os.path.join(temp_dir, f"sentinel2_tile_{i}_{uid}.vrt")
        _run_cmd(["gdalbuildvrt", "-separate", tile_vrt] + urls)
        tile_vrts.append(tile_vrt)

    if not tile_vrts:
        raise RuntimeError("No valid tiles after filtering")

    west, south, east, north = bbox
    tile_warped = []
    for i, tvrt in enumerate(tile_vrts):
        uid = str(uuid.uuid4())[:8]
        tw = os.path.join(temp_dir, f"sentinel2_warped_{i}_{uid}.tif")
        _run_cmd([
            "gdalwarp",
            "-t_srs", "EPSG:3857",
            "-te_srs", "EPSG:4326",
            "-te", str(west), str(south), str(east), str(north),
            "-r", "bilinear",
            "-co", "COMPRESS=LZW", "-co", "PREDICTOR=2", "-co", "TILED=YES",
            tvrt, tw,
        ])
        tile_warped.append(tw)

    uid = str(uuid.uuid4())[:8]
    mosaic_vrt = os.path.join(temp_dir, f"sentinel2_mosaic_{uid}.vrt")
    _run_cmd(["gdalbuildvrt", mosaic_vrt] + tile_warped)
    _run_cmd([
        "gdal_translate",
        "-co", "COMPRESS=LZW", "-co", "PREDICTOR=2", "-co", "TILED=YES",
        mosaic_vrt, output_path,
    ])

    for f in tile_vrts + tile_warped + [mosaic_vrt]:
        if os.path.exists(f):
            os.remove(f)


def fetch_sentinel2_reference(bbox: tuple, output_path: str,
                              max_cloud_cover: int = 10) -> str:
    """Download a recent cloud-free Sentinel-2 composite for an area.

    For large bboxes spanning multiple MGRS tiles, selects the best image
    per tile and mosaics them into a single output covering the full area.

    Returns the path to the EPSG:3857 GeoTIFF.
    """
    if os.path.exists(output_path):
        print(f"  [skip] Reference already exists: {output_path}")
        return output_path

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    west, south, east, north = bbox
    print(f"  Searching Sentinel-2 for bbox: {west:.2f},{south:.2f},{east:.2f},{north:.2f}")

    features = _stac_search_sentinel2(bbox, max_cloud_cover)
    selected = _select_best_per_mgrs_tile(features)

    band_keys = ["red", "green", "blue"]
    temp_dir = os.path.dirname(output_path) or "."
    if len(selected) == 1:
        _build_single_tile_reference(selected[0], bbox, output_path, band_keys, temp_dir)
    else:
        _build_multi_tile_mosaic_reference(selected, bbox, output_path, band_keys, temp_dir)

    print(f"  Sentinel-2 reference: {output_path}")
    return output_path


def _check_primary_covers_target(primary_bounds_wgs, target_bbox, margin_deg):
    """Return the expanded bbox, or ``None`` when primary already covers it."""
    west, south, east, north = target_bbox
    p_west, p_south, p_east, p_north = primary_bounds_wgs
    expanded = (west - margin_deg, south - margin_deg,
                east + margin_deg, north + margin_deg)
    if (p_west <= expanded[0] and p_south <= expanded[1] and
            p_east >= expanded[2] and p_north >= expanded[3]):
        return None
    return expanded


def _prepare_s2_grayscale_at_primary_grid(s2_path, primary_crs, primary_res,
                                          expanded, temp_dir):
    """Convert Sentinel-2 RGB to luminance then warp to primary's CRS + res.

    Returns the warped grayscale path plus an intermediate path to clean
    up afterwards.
    """
    import numpy as np
    from osgeo import gdal
    gdal.UseExceptions()

    s2_gray_path = os.path.join(temp_dir, "sentinel2_gray.tif")
    ds_s2 = gdal.Open(s2_path)
    s2_bands = ds_s2.RasterCount
    s2_w, s2_h = ds_s2.RasterXSize, ds_s2.RasterYSize
    if s2_bands >= 3:
        r = ds_s2.GetRasterBand(1).ReadAsArray().astype(np.float32)
        g = ds_s2.GetRasterBand(2).ReadAsArray().astype(np.float32)
        b = ds_s2.GetRasterBand(3).ReadAsArray().astype(np.float32)
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        gray = ds_s2.GetRasterBand(1).ReadAsArray().astype(np.float32)

    drv = gdal.GetDriverByName("GTiff")
    ds_gray = drv.Create(s2_gray_path, s2_w, s2_h, 1, gdal.GDT_Float32,
                         ["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES"])
    ds_gray.SetGeoTransform(ds_s2.GetGeoTransform())
    ds_gray.SetProjection(ds_s2.GetProjection())
    ds_gray.GetRasterBand(1).WriteArray(gray)
    ds_gray.FlushCache()
    ds_gray = None
    ds_s2 = None

    s2_warped_path = os.path.join(temp_dir, "sentinel2_warped.tif")
    _run_cmd([
        "gdalwarp",
        "-t_srs", str(primary_crs),
        "-tr", str(primary_res[0]), str(primary_res[1]),
        "-te_srs", "EPSG:4326",
        "-te", str(expanded[0]), str(expanded[1]), str(expanded[2]), str(expanded[3]),
        "-r", "bilinear",
        "-co", "COMPRESS=LZW", "-co", "PREDICTOR=2", "-co", "TILED=YES",
        "-co", "BIGTIFF=IF_SAFER",
        s2_gray_path, s2_warped_path,
    ])
    return s2_warped_path, s2_gray_path


def _feathered_composite_chunks(pri_path, s2_path, output_path, feather_px=50,
                                chunk_h=512):
    """Composite primary + S2 into a single uint8 raster, with a feathered
    alpha at the primary/S2 boundary. Processes in row chunks for memory."""
    import numpy as np
    from osgeo import gdal
    gdal.UseExceptions()

    drv = gdal.GetDriverByName("GTiff")
    ds_pri = gdal.Open(pri_path)
    ds_s2 = gdal.Open(s2_path)
    out_w = ds_pri.RasterXSize
    out_h = ds_pri.RasterYSize
    ds_out = drv.Create(output_path, out_w, out_h, 1, gdal.GDT_Byte,
                        ["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES",
                         "BIGTIFF=IF_SAFER"])
    ds_out.SetGeoTransform(ds_pri.GetGeoTransform())
    ds_out.SetProjection(ds_pri.GetProjection())

    for y0 in range(0, out_h, chunk_h):
        y1 = min(y0 + chunk_h, out_h)
        rows = y1 - y0
        pri_data = ds_pri.GetRasterBand(1).ReadAsArray(0, y0, out_w, rows).astype(np.float32)

        s2_w_actual = ds_s2.RasterXSize
        s2_h_actual = ds_s2.RasterYSize
        read_w = min(out_w, s2_w_actual)
        read_h = min(rows, s2_h_actual - y0) if y0 < s2_h_actual else 0
        s2_data = np.zeros((rows, out_w), dtype=np.float32)
        if read_h > 0 and read_w > 0:
            s2_chunk = ds_s2.GetRasterBand(1).ReadAsArray(0, y0, read_w, read_h)
            if s2_chunk is not None:
                s2_data[:read_h, :read_w] = s2_chunk.astype(np.float32)

        # Rescale S2 to primary's 0–255 range; keep nodata as 0.
        s2_valid = s2_data > 0
        if s2_valid.any():
            s2_max = np.percentile(s2_data[s2_valid], 99)
            if s2_max > 0:
                s2_scaled = np.clip(s2_data / s2_max * 255, 0, 255)
                s2_data = np.where(s2_valid, s2_scaled, 0)

        pri_valid = pri_data > 0
        if feather_px > 0 and pri_valid.any() and (~pri_valid).any():
            import cv2
            dist = cv2.distanceTransform(
                pri_valid.astype(np.uint8), cv2.DIST_L2, 3)
            alpha = np.clip(dist / feather_px, 0.0, 1.0)
        else:
            alpha = pri_valid.astype(np.float32)

        result = alpha * pri_data + (1.0 - alpha) * s2_data
        result[~pri_valid & ~s2_valid] = 0
        ds_out.GetRasterBand(1).WriteArray(
            np.clip(result, 0, 255).astype(np.uint8), 0, y0)

    ds_out.FlushCache()
    ds_out = None
    ds_pri = None
    ds_s2 = None


def build_composite_reference(primary_path: str, target_bbox: tuple,
                              output_path: str, margin_deg: float = 0.1,
                              max_cloud_cover: int = 10) -> str:
    """Build a composite reference: primary where available, Sentinel-2 elsewhere.

    Gives the alignment pipeline features to match against beyond the
    primary's footprint (improves edge alignment). Output is a single-band
    grayscale GeoTIFF in the primary's CRS.

    Returns ``primary_path`` when it already covers the target area.
    """
    import rasterio
    from rasterio.warp import transform_bounds

    if os.path.exists(output_path):
        print(f"  [skip] Composite reference already exists: {output_path}")
        return output_path

    with rasterio.open(primary_path) as src:
        primary_crs = src.crs
        primary_res = src.res
        primary_bounds_wgs = transform_bounds(primary_crs, "EPSG:4326", *src.bounds)

    expanded = _check_primary_covers_target(primary_bounds_wgs, target_bbox, margin_deg)
    if expanded is None:
        print(f"  Primary reference fully covers target area, no composite needed")
        return primary_path
    p_west, p_south, p_east, p_north = primary_bounds_wgs
    print(f"  Primary reference covers {p_west:.3f}-{p_east:.3f}E, "
          f"{p_south:.3f}-{p_north:.3f}N")
    print(f"  Target area needs {expanded[0]:.3f}-{expanded[2]:.3f}E, "
          f"{expanded[1]:.3f}-{expanded[3]:.3f}N")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    temp_dir = os.path.dirname(output_path)

    s2_path = os.path.join(temp_dir, "sentinel2_fill.tif")
    try:
        fetch_sentinel2_reference(expanded, s2_path, max_cloud_cover=max_cloud_cover)
    except RuntimeError as e:
        print(f"  WARNING: Could not fetch Sentinel-2: {e}")
        print(f"  Falling back to primary reference only")
        return primary_path

    s2_warped_path, s2_gray_path = _prepare_s2_grayscale_at_primary_grid(
        s2_path, primary_crs, primary_res, expanded, temp_dir,
    )

    # Warp primary to the same extent, nodata outside its bounds.
    primary_expanded_path = os.path.join(temp_dir, "primary_expanded.tif")
    _run_cmd([
        "gdalwarp",
        "-te_srs", "EPSG:4326",
        "-te", str(expanded[0]), str(expanded[1]), str(expanded[2]), str(expanded[3]),
        "-tr", str(primary_res[0]), str(primary_res[1]),
        "-r", "bilinear", "-dstnodata", "0",
        "-co", "COMPRESS=LZW", "-co", "PREDICTOR=2", "-co", "TILED=YES",
        "-co", "BIGTIFF=IF_SAFER",
        primary_path, primary_expanded_path,
    ])

    _feathered_composite_chunks(primary_expanded_path, s2_warped_path, output_path)

    for f in [s2_path, s2_gray_path, s2_warped_path, primary_expanded_path]:
        if os.path.exists(f):
            os.remove(f)

    print(f"  Composite reference: {output_path}")
    return output_path
