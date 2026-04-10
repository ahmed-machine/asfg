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

def coarse_align_and_crop(
    target_path: str,
    reference_path: str,
    output_path: str,
    coarse_res: float = 50.0,
    margin_m: float = 10000.0,
    target_bbox_wgs: tuple = None,
) -> str | None:
    """Find the coarse offset between a wide target and a reference, then crop.

    Film frames (especially KH-4 panoramic strips) cover huge areas but only
    a small portion overlaps with the reference. USGS corner coordinates can
    be 20km+ off, so we can't just crop to the reference bbox directly.

    Algorithm:
      1. Read both at low resolution (~50m/px) in a common metric CRS
      2. Slide the reference land mask over the target to find the best match
      3. Apply the detected offset to the target's geotransform
      4. Crop the shifted target to the reference bbox + margin

    Args:
        target_path: Wide georeferenced target (e.g. ASP ortho mosaic).
        reference_path: Smaller georeferenced reference image.
        output_path: Cropped + shifted output.
        coarse_res: Resolution for coarse matching (metres/pixel).
        margin_m: Padding around reference bbox (metres).
        target_bbox_wgs: Optional (west, south, east, north) in EPSG:4326 from
            USGS corners.  Used to check expected overlap with reference and
            reject spurious template matches.

    Returns:
        Path to the cropped output, or None on failure.
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

        # Compute the union bbox for resampling
        union_left = min(ref_left, tgt_left)
        union_bottom = min(ref_bottom, tgt_bottom)
        union_right = max(ref_right, tgt_right)
        union_top = max(ref_bottom, tgt_top)  # intentional: use ref_bottom as min ref

        # Read target at coarse resolution
        tgt_w = int((tgt_right - tgt_left) / coarse_res)
        tgt_h = int((tgt_top - tgt_bottom) / coarse_res)
        tgt_w = max(tgt_w, 1)
        tgt_h = max(tgt_h, 1)

        # Read reference at coarse resolution
        ref_w = int((ref_right - ref_left) / coarse_res)
        ref_h = int((ref_top - ref_bottom) / coarse_res)
        ref_w = max(ref_w, 1)
        ref_h = max(ref_h, 1)

        # Build the coarse arrays via GDAL warp to the common grid
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tgt_coarse = os.path.join(tmpdir, "tgt.tif")
            ref_coarse = os.path.join(tmpdir, "ref.tif")

            _run_cmd([
                "gdalwarp", "-t_srs", work_crs.to_proj4(),
                "-tr", str(coarse_res), str(coarse_res),
                "-r", "average",
                "-co", "COMPRESS=LZW",
                target_path, tgt_coarse,
            ])
            _run_cmd([
                "gdalwarp", "-t_srs", work_crs.to_proj4(),
                "-tr", str(coarse_res), str(coarse_res),
                "-r", "average",
                "-co", "COMPRESS=LZW",
                reference_path, ref_coarse,
            ])

            with rasterio.open(tgt_coarse) as ds_t, rasterio.open(ref_coarse) as ds_r:
                arr_tgt = ds_t.read(1).astype(np.float32)
                arr_ref = ds_r.read(1).astype(np.float32)
                tgt_transform = ds_t.transform
                ref_transform = ds_r.transform

        # Build binary masks (land = bright pixels)
        tgt_mask = (arr_tgt > 15).astype(np.float32)
        ref_mask = (arr_ref > 15).astype(np.float32)

        if tgt_mask.sum() < 100 or ref_mask.sum() < 100:
            print(f"  [coarse_crop] Insufficient content for matching")
            return None

        # Template match: slide reference mask over target mask.
        # matchTemplate requires the template to be <= the image in BOTH dims.
        # KH-4 panoramic strips are very wide but short — the reference can be
        # taller.  When one dimension exceeds the target, crop the reference
        # symmetrically so the template fits.
        import cv2

        if (ref_mask.shape[0] >= tgt_mask.shape[0] and
                ref_mask.shape[1] >= tgt_mask.shape[1]):
            # Reference is larger in both dimensions — target is fully contained
            print(f"  [coarse_crop] Reference larger than target in both dims, "
                  f"no crop needed")
            import shutil
            shutil.copy2(target_path, output_path)
            return output_path

        # Crop the reference mask to fit within target dims for template matching
        tmpl = ref_mask.copy()
        tmpl_row_offset = 0
        tmpl_col_offset = 0
        if tmpl.shape[0] >= tgt_mask.shape[0]:
            # Reference taller than target — crop to target height (centered)
            excess = tmpl.shape[0] - tgt_mask.shape[0] + 2  # +2 for matchTemplate
            top = excess // 2
            tmpl = tmpl[top:top + tgt_mask.shape[0] - 2, :]
            tmpl_row_offset = top
        if tmpl.shape[1] >= tgt_mask.shape[1]:
            # Reference wider than target — crop to target width (centered)
            excess = tmpl.shape[1] - tgt_mask.shape[1] + 2
            left = excess // 2
            tmpl = tmpl[:, left:left + tgt_mask.shape[1] - 2]
            tmpl_col_offset = left

        if tmpl.shape[0] < 3 or tmpl.shape[1] < 3:
            print(f"  [coarse_crop] Template too small after cropping "
                  f"({tmpl.shape}), skipping")
            return None

        result = cv2.matchTemplate(tgt_mask, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # max_loc is (col, row) of the best match in target pixel coords
        match_col, match_row = max_loc

        # Convert pixel offset to metric offset
        # The offset = where the reference template's origin lands in target coords.
        # Account for any template cropping applied above.
        # Target origin in work CRS:
        tgt_origin_x = tgt_transform.c  # left
        tgt_origin_y = tgt_transform.f  # top
        # Match position in work CRS (adjust for template crop offset):
        match_x = tgt_origin_x + (match_col - tmpl_col_offset) * coarse_res
        match_y = tgt_origin_y - (match_row - tmpl_row_offset) * coarse_res

        # Reference origin in work CRS:
        ref_origin_x = ref_transform.c
        ref_origin_y = ref_transform.f

        # Offset: how much to shift the target so its content aligns with reference
        dx_m = ref_origin_x - match_x
        dy_m = ref_origin_y - match_y

        total_offset = np.sqrt(dx_m**2 + dy_m**2)
        print(f"  [coarse_crop] Coarse offset: dx={dx_m:+.0f}m, dy={dy_m:+.0f}m "
              f"(total={total_offset:.0f}m, corr={max_val:.3f})")

        # Decide whether to trust the template match.  For predominantly-ocean
        # frames (Bahrain, coastal strips) the land mask correlation is low and
        # the match can be completely wrong.  When the match is unreliable, fall
        # back to zero offset (trust USGS corners) and still crop so the
        # alignment pipeline gets a manageable-sized image.
        use_offset = True
        if max_val < 0.15:
            print(f"  [coarse_crop] Correlation too low ({max_val:.3f}), "
                  f"using zero offset (USGS corners)")
            dx_m, dy_m = 0.0, 0.0
            use_offset = False
        elif max_val < 0.40 and total_offset > 25000:
            # Low-confidence match with large shift — almost certainly spurious
            print(f"  [coarse_crop] Low-confidence large shift "
                  f"({total_offset/1000:.0f}km, corr={max_val:.3f}), "
                  f"using zero offset")
            dx_m, dy_m = 0.0, 0.0
            use_offset = False
        elif total_offset > 50000:
            print(f"  [coarse_crop] Shift too large ({total_offset/1000:.0f}km > 50km), "
                  f"using zero offset")
            dx_m, dy_m = 0.0, 0.0
            use_offset = False

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
    shifted_path = output_path.replace(".tif", "_shifted.tif")
    ds_in = gdal.Open(target_path)
    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.CreateCopy(shifted_path, ds_in, options=[
        "COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES",
    ])
    ds_out.SetGeoTransform(gt)
    ds_out.FlushCache()
    ds_out = None
    ds_in = None

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

    _run_cmd([
        "gdalwarp",
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
    return output_path


# ---------------------------------------------------------------------------
# Sentinel-2 reference image auto-fetch via Element 84 Earth Search STAC API
# ---------------------------------------------------------------------------

STAC_API = os.environ.get("STAC_API_URL", "https://earth-search.aws.element84.com/v1")
COLLECTION = "sentinel-2-l2a"


def fetch_sentinel2_reference(bbox: tuple, output_path: str,
                              max_cloud_cover: int = 10) -> str:
    """Download a recent cloud-free Sentinel-2 composite for an area.

    For large bboxes spanning multiple MGRS tiles, selects the best image
    per tile and mosaics them into a single output covering the full area.

    Args:
        bbox: (west, south, east, north) in decimal degrees.
        output_path: Where to write the reference GeoTIFF (EPSG:3857).
        max_cloud_cover: Maximum cloud cover percentage (default 10).

    Returns:
        Path to the reference GeoTIFF.
    """
    if os.path.exists(output_path):
        print(f"  [skip] Reference already exists: {output_path}")
        return output_path

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    west, south, east, north = bbox
    print(f"  Searching Sentinel-2 for bbox: {west:.2f},{south:.2f},{east:.2f},{north:.2f}")

    # Search with high limit to find all tiles covering the bbox
    search_body = json.dumps({
        "collections": [COLLECTION],
        "bbox": [west, south, east, north],
        "query": {
            "eo:cloud_cover": {"lt": max_cloud_cover},
        },
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

    # Group features by MGRS tile and select the best (lowest cloud) per tile
    tiles = defaultdict(list)
    for f in features:
        grid = f["properties"].get("grid:code", f["id"][:10])
        tiles[grid].append(f)

    selected = []
    for grid, items in sorted(tiles.items()):
        best = min(items, key=lambda x: x["properties"].get("eo:cloud_cover", 99))
        selected.append(best)
        item_id = best["id"]
        cloud = best["properties"].get("eo:cloud_cover", "?")
        date = best["properties"].get("datetime", "?")[:10]
        print(f"    Tile {grid}: {item_id} ({date}, {cloud}% cloud)")

    print(f"  Selected {len(selected)} tile(s) for composite")

    # Collect band URLs from all selected tiles
    band_keys = ["red", "green", "blue"]
    temp_dir = os.path.dirname(output_path) or "."

    if len(selected) == 1:
        # Single tile: simple path (original behavior)
        item = selected[0]
        assets = item.get("assets", {})
        band_urls = []
        for key in band_keys:
            if key in assets:
                band_urls.append(assets[key]["href"])
            else:
                raise RuntimeError(f"Missing '{key}' band in {item['id']}")

        # Generate a unique VRT path to avoid race conditions when multiprocessing
        import uuid
        uid = str(uuid.uuid4())[:8]
        vrt_path = os.path.join(temp_dir, f"sentinel2_ref_{uid}.vrt")
        vsicurl_urls = [f"/vsicurl/{url}" for url in band_urls]
        _run_cmd(["gdalbuildvrt", "-separate", vrt_path] + vsicurl_urls)

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
    else:
        # Multi-tile: build per-tile VRTs, then mosaic, then warp
        tile_vrts = []
        for i, item in enumerate(selected):
            assets = item.get("assets", {})
            band_urls = []
            for key in band_keys:
                if key in assets:
                    band_urls.append(assets[key]["href"])
                else:
                    print(f"    WARNING: missing '{key}' in {item['id']}, skipping tile")
                    break
            else:
                import uuid
                uid = str(uuid.uuid4())[:8]
                tile_vrt = os.path.join(temp_dir, f"sentinel2_tile_{i}_{uid}.vrt")
                vsicurl_urls = [f"/vsicurl/{url}" for url in band_urls]
                _run_cmd(["gdalbuildvrt", "-separate", tile_vrt] + vsicurl_urls)
                tile_vrts.append(tile_vrt)

        if not tile_vrts:
            raise RuntimeError("No valid tiles after filtering")

        # Warp each tile to EPSG:3857 clipped to bbox, then mosaic
        tile_warped = []
        for i, tvrt in enumerate(tile_vrts):
            import uuid
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

        # Merge warped tiles into final output
        import uuid
        uid = str(uuid.uuid4())[:8]
        mosaic_vrt = os.path.join(temp_dir, f"sentinel2_mosaic_{uid}.vrt")
        _run_cmd(["gdalbuildvrt", mosaic_vrt] + tile_warped)
        _run_cmd([
            "gdal_translate",
            "-co", "COMPRESS=LZW", "-co", "PREDICTOR=2", "-co", "TILED=YES",
            mosaic_vrt, output_path,
        ])

        # Clean up temp files
        for f in tile_vrts + tile_warped + [mosaic_vrt]:
            if os.path.exists(f):
                os.remove(f)

    print(f"  Sentinel-2 reference: {output_path}")
    return output_path


def build_composite_reference(primary_path: str, target_bbox: tuple,
                              output_path: str, margin_deg: float = 0.1,
                              max_cloud_cover: int = 10) -> str:
    """Build a composite reference: primary where available, Sentinel-2 elsewhere.

    Uses the primary reference (e.g. KH-9 historical image) where it has coverage,
    and fills gaps with Sentinel-2 imagery for areas outside the primary's footprint.
    This gives the alignment pipeline features to match against in areas beyond the
    primary reference, improving edge alignment.

    The output is a single-band grayscale GeoTIFF in the primary's CRS.

    Args:
        primary_path: Path to the primary reference GeoTIFF.
        target_bbox: (west, south, east, north) in decimal degrees — area needing coverage.
        output_path: Where to write the composite GeoTIFF.
        margin_deg: Extra margin beyond target_bbox for Sentinel-2 fetch.
        max_cloud_cover: Max cloud cover % for Sentinel-2 search.

    Returns:
        Path to the composite reference (may be primary_path if it already covers everything).
    """
    import numpy as np
    import rasterio
    from rasterio.warp import transform_bounds
    from osgeo import gdal

    if os.path.exists(output_path):
        print(f"  [skip] Composite reference already exists: {output_path}")
        return output_path

    # Read primary reference bounds
    with rasterio.open(primary_path) as src:
        primary_crs = src.crs
        primary_res = src.res
        primary_bounds_native = src.bounds
        primary_bounds_wgs = transform_bounds(primary_crs, "EPSG:4326", *src.bounds)

    west, south, east, north = target_bbox
    p_west, p_south, p_east, p_north = primary_bounds_wgs

    # Expand target bbox by margin for reference context beyond crop area
    expanded = (west - margin_deg, south - margin_deg,
                east + margin_deg, north + margin_deg)

    # Check if primary already covers the expanded bbox
    if (p_west <= expanded[0] and p_south <= expanded[1] and
            p_east >= expanded[2] and p_north >= expanded[3]):
        print(f"  Primary reference fully covers target area, no composite needed")
        return primary_path

    print(f"  Primary reference covers {p_west:.3f}-{p_east:.3f}E, "
          f"{p_south:.3f}-{p_north:.3f}N")
    print(f"  Target area needs {expanded[0]:.3f}-{expanded[2]:.3f}E, "
          f"{expanded[1]:.3f}-{expanded[3]:.3f}N")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    temp_dir = os.path.dirname(output_path)

    # Fetch Sentinel-2 for the expanded area
    s2_path = os.path.join(temp_dir, "sentinel2_fill.tif")
    try:
        fetch_sentinel2_reference(expanded, s2_path, max_cloud_cover=max_cloud_cover)
    except RuntimeError as e:
        print(f"  WARNING: Could not fetch Sentinel-2: {e}")
        print(f"  Falling back to primary reference only")
        return primary_path

    gdal.UseExceptions()

    # Convert Sentinel-2 RGB to grayscale and warp to primary's CRS/resolution
    s2_gray_path = os.path.join(temp_dir, "sentinel2_gray.tif")
    ds_s2 = gdal.Open(s2_path)
    s2_bands = ds_s2.RasterCount
    s2_w, s2_h = ds_s2.RasterXSize, ds_s2.RasterYSize

    if s2_bands >= 3:
        # Read RGB and convert to luminance
        r = ds_s2.GetRasterBand(1).ReadAsArray().astype(np.float32)
        g = ds_s2.GetRasterBand(2).ReadAsArray().astype(np.float32)
        b = ds_s2.GetRasterBand(3).ReadAsArray().astype(np.float32)
        gray = (0.299 * r + 0.587 * g + 0.114 * b)
    else:
        gray = ds_s2.GetRasterBand(1).ReadAsArray().astype(np.float32)

    # Write grayscale S2 with same geotransform
    drv = gdal.GetDriverByName("GTiff")
    ds_gray = drv.Create(s2_gray_path, s2_w, s2_h, 1, gdal.GDT_Float32,
                         ["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES"])
    ds_gray.SetGeoTransform(ds_s2.GetGeoTransform())
    ds_gray.SetProjection(ds_s2.GetProjection())
    ds_gray.GetRasterBand(1).WriteArray(gray)
    ds_gray.FlushCache()
    ds_gray = None
    ds_s2 = None

    # Warp S2 grayscale to primary's CRS and resolution
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

    # Warp primary to the same expanded extent (filling with nodata outside its bounds)
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

    # Composite: primary where valid, S2 elsewhere, with feathered blend at boundary
    ds_pri = gdal.Open(primary_expanded_path)
    ds_s2w = gdal.Open(s2_warped_path)

    out_w = ds_pri.RasterXSize
    out_h = ds_pri.RasterYSize

    ds_out = drv.Create(output_path, out_w, out_h, 1, gdal.GDT_Byte,
                        ["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES",
                         "BIGTIFF=IF_SAFER"])
    ds_out.SetGeoTransform(ds_pri.GetGeoTransform())
    ds_out.SetProjection(ds_pri.GetProjection())

    # Process in row chunks for memory efficiency
    chunk_h = 512
    feather_px = 50  # feather width at primary/S2 boundary

    for y0 in range(0, out_h, chunk_h):
        y1 = min(y0 + chunk_h, out_h)
        rows = y1 - y0

        pri_data = ds_pri.GetRasterBand(1).ReadAsArray(0, y0, out_w, rows).astype(np.float32)

        # Read S2 — handle size mismatch gracefully
        s2_w_actual = ds_s2w.RasterXSize
        s2_h_actual = ds_s2w.RasterYSize
        read_w = min(out_w, s2_w_actual)
        read_h = min(rows, s2_h_actual - y0) if y0 < s2_h_actual else 0

        s2_data = np.zeros((rows, out_w), dtype=np.float32)
        if read_h > 0 and read_w > 0:
            s2_chunk = ds_s2w.GetRasterBand(1).ReadAsArray(0, y0, read_w, read_h)
            if s2_chunk is not None:
                s2_data[:read_h, :read_w] = s2_chunk.astype(np.float32)

        # Scale S2 to match primary's radiometric range
        # (S2 reflectance values are typically 0-10000, primary is 0-255)
        # Track which S2 pixels have actual data (not MGRS tile nodata)
        s2_valid = s2_data > 0
        if s2_valid.any():
            s2_max = np.percentile(s2_data[s2_valid], 99)
            if s2_max > 0:
                s2_scaled = np.clip(s2_data / s2_max * 255, 0, 255)
                # Only keep scaled values where original was valid
                s2_data = np.where(s2_valid, s2_scaled, 0)

        # Build alpha mask for primary (0 where nodata, 1 where valid)
        pri_valid = pri_data > 0

        # Distance transform for feathered blend at boundary
        if feather_px > 0 and pri_valid.any() and (~pri_valid).any():
            import cv2
            dist = cv2.distanceTransform(
                pri_valid.astype(np.uint8), cv2.DIST_L2, 3)
            alpha = np.clip(dist / feather_px, 0.0, 1.0)
        else:
            alpha = pri_valid.astype(np.float32)

        # Composite: primary where valid (feathered at edge), S2 where no primary
        # Where S2 has nodata (MGRS tile gaps), keep primary or leave as 0
        result = alpha * pri_data + (1.0 - alpha) * s2_data
        neither = ~pri_valid & ~s2_valid
        result[neither] = 0

        ds_out.GetRasterBand(1).WriteArray(
            np.clip(result, 0, 255).astype(np.uint8), 0, y0)

    ds_out.FlushCache()
    ds_out = None
    ds_pri = None
    ds_s2w = None

    # Clean up intermediates
    for f in [s2_path, s2_gray_path, s2_warped_path, primary_expanded_path]:
        if os.path.exists(f):
            os.remove(f)

    print(f"  Composite reference: {output_path}")
    return output_path
