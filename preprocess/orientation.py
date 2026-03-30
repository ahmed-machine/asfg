"""Orientation detection and correction for declassified satellite imagery.

Handles:
- Data-driven orientation detection by matching against a georeferenced reference
- Metadata-based fallback (aspect ratio, corner coordinates)
- Post-georef orientation verification using SIFT + land mask correlation
"""

import json
import os
import subprocess

import numpy as np

from . import run_gdal_cmd as _run_cmd


def swap_corners_180(corners: dict) -> dict:
    """Swap corners for a 180° rotation: NW↔SE, NE↔SW."""
    return {
        "NW": corners["SE"], "NE": corners["SW"],
        "SE": corners["NW"], "SW": corners["NE"],
    }


def rotate_corners_cw90(corners: dict) -> dict:
    """Rotate corner assignments 90° clockwise.

    After CW rotation of the image:
    - Old left edge (NW-SW) becomes top edge → new NW=SW, new NE=NW
    - Old right edge (NE-SE) becomes bottom edge → new SW=SE, new SE=NE
    """
    return {
        "NW": corners["SW"], "NE": corners["NW"],
        "SE": corners["NE"], "SW": corners["SE"],
    }


def rotate_corners_ccw90(corners: dict) -> dict:
    """Rotate corner assignments 90° counter-clockwise.

    After CCW rotation of the image:
    - Old top edge (NW-NE) becomes left edge → new NW=NE, new SW=SE (wait...)
    - Actually: old NE becomes new NW, old SE becomes new NE, etc.
    """
    return {
        "NW": corners["NE"], "NE": corners["SE"],
        "SE": corners["SW"], "SW": corners["NW"],
    }


def get_image_dimensions(image_path: str) -> tuple:
    """Get (width, height) from a TIFF using gdalinfo."""
    result = _run_cmd(["gdalinfo", "-json", image_path])
    info = json.loads(result.stdout)
    return tuple(info["size"])


def _geographic_aspect_ratio(corners: dict) -> float:
    """Compute the geographic aspect ratio (width/height) from corners.

    Uses approximate meter distances at the image latitude.
    """
    import math

    # Average latitude for distance calculation
    avg_lat = (corners["NW"][0] + corners["SE"][0]) / 2
    cos_lat = math.cos(math.radians(avg_lat))

    # Width: average of top and bottom edge lengths in degrees
    top_lon_span = abs(corners["NE"][1] - corners["NW"][1])
    bot_lon_span = abs(corners["SE"][1] - corners["SW"][1])
    avg_lon_span = (top_lon_span + bot_lon_span) / 2

    # Height: average of left and right edge lengths in degrees
    left_lat_span = abs(corners["NW"][0] - corners["SW"][0])
    right_lat_span = abs(corners["NE"][0] - corners["SE"][0])
    avg_lat_span = (left_lat_span + right_lat_span) / 2

    if avg_lat_span == 0:
        return 999.0

    # Convert to approximate meters
    width_m = avg_lon_span * cos_lat * 111000
    height_m = avg_lat_span * 111000

    return width_m / height_m if height_m > 0 else 999.0


def detect_orientation(image_path: str, corners: dict, camera,
                       reference_path: str = None) -> tuple:
    """Detect image orientation from metadata heuristics + optional reference.

    First uses metadata heuristics (portrait detection, aspect mismatch,
    reversed strip). Then, if a reference is available, runs reference-based
    verification via fast mini-georef + land mask correlation. The reference
    result overrides metadata ONLY when the margin is strong (>= 0.10
    difference between best and runner-up correlations).

    Returns (rotation_degrees, corrected_corners) where rotation_degrees
    is 0, 90, 180, or 270.
    """
    # Step 1: metadata-based heuristics (fast, reliable for KH-7)
    w, h = get_image_dimensions(image_path)
    img_aspect = w / h if h > 0 else 1.0
    geo_aspect = _geographic_aspect_ratio(corners)

    nw_lon = corners["NW"][1]
    ne_lon = corners["NE"][1]
    is_reversed = nw_lon > ne_lon
    is_portrait = img_aspect < 1.0

    aspect_mismatch = False
    if geo_aspect > 0 and img_aspect > 0:
        ratio = img_aspect / geo_aspect
        if ratio > 2.0 or ratio < 0.5:
            aspect_mismatch = True

    if is_portrait:
        meta_rotation = 90
        meta_corners = rotate_corners_ccw90(corners)
        print(f"  Orientation (metadata): portrait (aspect={img_aspect:.2f}), needs 90")
    elif aspect_mismatch and img_aspect > 1.0 and geo_aspect < 1.0:
        meta_rotation = 270
        meta_corners = rotate_corners_cw90(corners)
        print(f"  Orientation (metadata): aspect mismatch, needs 270")
    elif is_reversed:
        meta_rotation = 180
        meta_corners = swap_corners_180(corners)
        print(f"  Orientation (metadata): reversed strip, needs 180")
    else:
        meta_rotation = 0
        meta_corners = corners

    # Step 2: reference-based verification (only if reference available)
    if reference_path and os.path.exists(reference_path):
        print(f"  Orientation: verifying against reference...")
        ref_rotation, ref_corners, ref_confidence = \
            detect_orientation_against_reference(
                image_path, corners, reference_path,
            )
        # Only override metadata when the reference result has strong margin
        if ref_confidence >= 10 and ref_rotation != meta_rotation:
            print(f"  Orientation: reference says {ref_rotation} "
                  f"(confidence={ref_confidence}), overriding metadata ({meta_rotation})")
            return (ref_rotation, ref_corners)
        elif ref_confidence >= 10:
            print(f"  Orientation: reference confirms metadata ({meta_rotation}, "
                  f"confidence={ref_confidence})")

    return (meta_rotation, meta_corners)


def _read_band_at_scale(path: str, target_res_m: float = 100.0):
    """Read a georeferenced image band at approximately target_res_m per pixel.

    Returns (array, geotransform, projection) or (None, None, None) on failure.
    """
    from osgeo import gdal, osr
    gdal.UseExceptions()

    ds = gdal.Open(path)
    if ds is None:
        return None, None, None

    gt = ds.GetGeoTransform()
    w, h = ds.RasterXSize, ds.RasterYSize
    proj = ds.GetProjection()

    # Determine pixel size in meters
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    px_size = abs(gt[1])
    if px_size == 0:
        ds = None
        return None, None, None

    # For geographic CRS (e.g. EPSG:4326), convert degrees to approximate meters
    if srs.IsGeographic():
        # At ~26° latitude, 1 degree ≈ 100km lon, 111km lat
        px_size_m = px_size * 111000  # rough approximation
    else:
        px_size_m = px_size

    scale = px_size_m / target_res_m
    if scale <= 0:
        ds = None
        return None, None, None

    out_w = max(1, int(w * scale))
    out_h = max(1, int(h * scale))

    arr = ds.GetRasterBand(1).ReadAsArray(
        buf_xsize=out_w, buf_ysize=out_h
    )

    # Adjust geotransform for the new resolution
    new_gt = (gt[0], gt[1] / scale, gt[2], gt[3], gt[4], gt[5] / scale)

    ds = None
    return arr, new_gt, proj


def _read_reference_crop(reference_path, west, south, east, north, target_h, target_w):
    """Read a crop from a georeferenced reference image at given lat/lon bounds.

    Handles CRS reprojection from EPSG:4326 to whatever the reference uses.
    Returns a 2D numpy array, or None if no valid data.
    """
    from osgeo import gdal, osr

    gdal.UseExceptions()
    ds = gdal.Open(reference_path)
    if ds is None:
        return None

    gt = ds.GetGeoTransform()
    raster_w, raster_h = ds.RasterXSize, ds.RasterYSize
    proj = ds.GetProjection()

    src_srs = osr.SpatialReference()
    src_srs.ImportFromEPSG(4326)
    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    ref_srs = osr.SpatialReference()
    ref_srs.ImportFromWkt(proj)
    ref_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    if src_srs.IsSame(ref_srs):
        x_min, y_min, x_max, y_max = west, south, east, north
    else:
        ct = osr.CoordinateTransformation(src_srs, ref_srs)
        transformed = [
            ct.TransformPoint(west, south),
            ct.TransformPoint(east, north),
            ct.TransformPoint(west, north),
            ct.TransformPoint(east, south),
        ]
        xs = [c[0] for c in transformed]
        ys = [c[1] for c in transformed]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

    col_min = max(0, int((x_min - gt[0]) / gt[1]))
    col_max = min(raster_w, int((x_max - gt[0]) / gt[1]) + 1)
    row_min = max(0, int((y_max - gt[3]) / gt[5]))
    row_max = min(raster_h, int((y_min - gt[3]) / gt[5]) + 1)

    if col_min >= col_max or row_min >= row_max:
        ds = None
        return None

    pixel_w = col_max - col_min
    pixel_h = row_max - row_min
    if pixel_w < 10 or pixel_h < 10:
        ds = None
        return None

    arr = ds.GetRasterBand(1).ReadAsArray(
        xoff=col_min, yoff=row_min,
        win_xsize=pixel_w, win_ysize=pixel_h,
        buf_xsize=target_w, buf_ysize=target_h,
    )
    ds = None

    if arr is None or np.mean(arr > 0) < 0.1:
        return None
    return arr


def detect_orientation_against_reference(image_path, corners, reference_path,
                                         target_res=200.0):
    """Determine image orientation by fast low-res georef + SIFT matching.

    For each candidate rotation (0, 90, 180, 270):
    1. Apply the rotation to the corner coordinates.
    2. Use GDAL to warp the raw image to EPSG:4326 at coarse resolution
       using the rotated corners as GCPs.
    3. Read the reference at the matching bounds.
    4. Run SIFT matching between the warped image and reference.

    The rotation with the most SIFT RANSAC inliers wins.

    Args:
        image_path: Path to the raw/stitched image (not yet georeferenced).
        corners: Dict with NW, NE, SE, SW keys, each (lat, lon).
        reference_path: Path to a georeferenced reference image.
        target_res: Approximate resolution in meters for the low-res georef.

    Returns:
        (rotation_degrees, corrected_corners, n_inliers)
    """
    import cv2
    from osgeo import gdal

    gdal.UseExceptions()

    ds = gdal.Open(image_path)
    if ds is None:
        print("  Orientation ref-match: could not open image")
        return 0, corners, 0

    img_w, img_h = ds.RasterXSize, ds.RasterYSize
    ds = None

    # GCP mappings for testing each rotation in _fast_mini_georef.
    # These map unrotated pixel positions to geographic corners as if the
    # image were captured at orientation R.  For 90/270 the GCP mapping is
    # the INVERSE of the corner rotation applied after actual image rotation.
    #
    # For panoramic cameras (KH-4), the strip is always landscape —
    # only 0° and 180° are valid orientations.
    is_panoramic = img_w > img_h * 2  # clearly landscape strip
    if is_panoramic:
        candidates = [
            (0,   corners),
            (180, swap_corners_180(corners)),
        ]
    else:
        candidates = [
            (0,   corners),
            (180, swap_corners_180(corners)),
            (90,  rotate_corners_ccw90(corners)),
            (270, rotate_corners_cw90(corners)),
        ]

    # Target resolution in degrees (~target_res meters / 111km)
    target_deg = target_res / 111000.0

    # Check if reference covers the target area; if not, try auto-fetching
    all_lats = [corners[k][0] for k in ("NW", "NE", "SE", "SW")]
    all_lons = [corners[k][1] for k in ("NW", "NE", "SE", "SW")]
    full_west, full_east = min(all_lons), max(all_lons)
    full_south, full_north = min(all_lats), max(all_lats)

    # Determine the reference bounds in EPSG:4326 so we can compute overlap
    from osgeo import gdal as _gdal_check, osr as _osr_check
    _gdal_check.UseExceptions()
    _ref_ds = _gdal_check.Open(reference_path)
    if _ref_ds is None:
        print("  Orientation ref-match: cannot open reference")
        return 0, corners, 0
    _ref_gt = _ref_ds.GetGeoTransform()
    _ref_w, _ref_h = _ref_ds.RasterXSize, _ref_ds.RasterYSize
    _ref_proj = _ref_ds.GetProjection()
    _ref_ds = None

    _rx_min = _ref_gt[0]
    _rx_max = _ref_gt[0] + _ref_w * _ref_gt[1]
    _ry_max = _ref_gt[3]
    _ry_min = _ref_gt[3] + _ref_h * _ref_gt[5]

    _ref_srs = _osr_check.SpatialReference()
    _ref_srs.ImportFromWkt(_ref_proj)
    _wgs84 = _osr_check.SpatialReference()
    _wgs84.ImportFromEPSG(4326)
    _ref_srs.SetAxisMappingStrategy(_osr_check.OAMS_TRADITIONAL_GIS_ORDER)
    _wgs84.SetAxisMappingStrategy(_osr_check.OAMS_TRADITIONAL_GIS_ORDER)

    if not _ref_srs.IsSame(_wgs84):
        _ct = _osr_check.CoordinateTransformation(_ref_srs, _wgs84)
        _c1 = _ct.TransformPoint(_rx_min, _ry_min)
        _c2 = _ct.TransformPoint(_rx_max, _ry_max)
        _c3 = _ct.TransformPoint(_rx_min, _ry_max)
        _c4 = _ct.TransformPoint(_rx_max, _ry_min)
        _xs = [_c1[0], _c2[0], _c3[0], _c4[0]]
        _ys = [_c1[1], _c2[1], _c3[1], _c4[1]]
        ref_west, ref_east = min(_xs), max(_xs)
        ref_south, ref_north = min(_ys), max(_ys)
    else:
        ref_west, ref_east = _rx_min, _rx_max
        ref_south, ref_north = _ry_min, _ry_max

    # Compute intersection of target area and reference bounds
    ovl_west = max(full_west, ref_west)
    ovl_east = min(full_east, ref_east)
    ovl_south = max(full_south, ref_south)
    ovl_north = min(full_north, ref_north)

    need_auto_fetch = False
    if ovl_west >= ovl_east or ovl_south >= ovl_north:
        need_auto_fetch = True
    else:
        ovl_area = (ovl_east - ovl_west) * (ovl_north - ovl_south)
        full_area = max(1e-10, (full_east - full_west) * (full_north - full_south))
        ovl_frac = ovl_area / full_area
        # Need substantial overlap for reliable land mask matching
        if ovl_frac < 0.30:
            need_auto_fetch = True
        else:
            print(f"  Orientation ref-match: {ovl_frac:.0%} of image overlaps "
                  f"reference ([{ovl_west:.2f},{ovl_south:.2f}]-"
                  f"[{ovl_east:.2f},{ovl_north:.2f}])")

    if need_auto_fetch:
        # Auto-fetch Sentinel-2 reference for the full strip area
        try:
            from preprocess.georef import fetch_sentinel2_reference
            import hashlib, tempfile
            bbox = (full_west - 0.1, full_south - 0.1,
                    full_east + 0.1, full_north + 0.1)
            bbox_hash = hashlib.md5(str(bbox).encode()).hexdigest()[:8]
            auto_ref = os.path.join(
                tempfile.gettempdir(), f"orient_ref_{bbox_hash}.tif")
            reference_path = fetch_sentinel2_reference(bbox, auto_ref)
            print(f"  Orientation ref-match: using auto-fetched Sentinel-2")

            # Recompute overlap with new reference (might be EPSG:3857)
            _ref_ds2 = _gdal_check.Open(reference_path)
            if _ref_ds2 is None:
                return 0, corners, 0
            _rgt2 = _ref_ds2.GetGeoTransform()
            _rw2, _rh2 = _ref_ds2.RasterXSize, _ref_ds2.RasterYSize
            _rp2 = _ref_ds2.GetProjection()
            _ref_ds2 = None

            _rs2 = _osr_check.SpatialReference()
            _rs2.ImportFromWkt(_rp2)
            _rs2.SetAxisMappingStrategy(_osr_check.OAMS_TRADITIONAL_GIS_ORDER)
            _rx2_min = _rgt2[0]
            _rx2_max = _rgt2[0] + _rw2 * _rgt2[1]
            _ry2_max = _rgt2[3]
            _ry2_min = _rgt2[3] + _rh2 * _rgt2[5]

            if not _rs2.IsSame(_wgs84):
                _ct2 = _osr_check.CoordinateTransformation(_rs2, _wgs84)
                _corners2 = [_ct2.TransformPoint(_rx2_min, _ry2_min),
                             _ct2.TransformPoint(_rx2_max, _ry2_max),
                             _ct2.TransformPoint(_rx2_min, _ry2_max),
                             _ct2.TransformPoint(_rx2_max, _ry2_min)]
                ref_west = min(c[0] for c in _corners2)
                ref_east = max(c[0] for c in _corners2)
                ref_south = min(c[1] for c in _corners2)
                ref_north = max(c[1] for c in _corners2)
            else:
                ref_west, ref_east = _rx2_min, _rx2_max
                ref_south, ref_north = _ry2_min, _ry2_max

            ovl_west = max(full_west, ref_west)
            ovl_east = min(full_east, ref_east)
            ovl_south = max(full_south, ref_south)
            ovl_north = min(full_north, ref_north)
            if ovl_west >= ovl_east or ovl_south >= ovl_north:
                print("  Orientation ref-match: still no overlap after auto-fetch")
                return 0, corners, 0
        except Exception as e:
            print(f"  Orientation ref-match: auto-fetch failed ({e}), "
                  f"falling back to metadata")
            return 0, corners, 0

    best_rotation = 0
    best_score = -2.0

    clip = (ovl_west, ovl_south, ovl_east, ovl_north)

    for rotation, rot_corners in candidates:
        try:
            warped = _fast_mini_georef(
                image_path, rot_corners, img_w, img_h, target_deg,
                clip_bounds=clip,
            )
        except Exception as e:
            print(f"    Rotation {rotation:>3}: georef failed ({e})")
            continue

        if warped is None:
            continue

        warped_arr, warped_west, warped_south, warped_east, warped_north = warped
        wh, ww = warped_arr.shape

        # Read reference at matching bounds
        ref_crop = _read_reference_crop(
            reference_path, warped_west, warped_south, warped_east, warped_north,
            target_h=wh, target_w=ww,
        )
        if ref_crop is None:
            print(f"    Rotation {rotation:>3}: no reference coverage")
            continue

        # Compare land masks via normalized cross-correlation.
        # Only compare in the region where BOTH images have valid (non-zero) data.
        # This works across decades and camera systems because coastlines
        # are invariant.
        valid = (warped_arr > 0) & (ref_crop > 0)
        valid_frac = np.mean(valid)
        if valid_frac < 0.05:
            print(f"    Rotation {rotation:>3}: insufficient overlap "
                  f"(valid={valid_frac:.1%})")
            continue

        import cv2 as _cv2

        # Normalize both to uint8 for consistent Otsu thresholding.
        # Handles both 8-bit historical imagery and 16-bit Sentinel-2.
        def _to_u8(arr, mask):
            vals = arr[mask]
            if vals.size == 0:
                return arr.astype(np.uint8)
            lo, hi = np.percentile(vals, [2, 98])
            if hi <= lo:
                hi = lo + 1
            clipped = np.clip((arr.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255)
            return clipped.astype(np.uint8)

        w_u8 = _to_u8(warped_arr, valid)
        r_u8 = _to_u8(ref_crop, valid)

        w_thresh, _ = _cv2.threshold(w_u8[valid], 0, 1, _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)
        r_thresh, _ = _cv2.threshold(r_u8[valid], 0, 1, _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)

        warped_mask = (w_u8 > w_thresh).astype(np.float32)
        ref_mask = (r_u8 > r_thresh).astype(np.float32)

        # Mask to only the valid overlap region
        w_vals = warped_mask[valid]
        r_vals = ref_mask[valid]

        w_mean = w_vals.mean()
        r_mean = r_vals.mean()

        # Both must have land/water contrast within the overlap
        if w_mean < 0.02 or w_mean > 0.98 or r_mean < 0.02 or r_mean > 0.98:
            print(f"    Rotation {rotation:>3}: no contrast in overlap "
                  f"(w_land={w_mean:.1%}, r_land={r_mean:.1%})")
            continue

        w_centered = w_vals - w_mean
        r_centered = r_vals - r_mean
        w_std = w_centered.std()
        r_std = r_centered.std()

        if w_std < 1e-6 or r_std < 1e-6:
            corr = 0.0
        else:
            corr = float(np.dot(w_centered, r_centered) / (w_std * r_std * len(w_vals)))

        print(f"    Rotation {rotation:>3}: land corr = {corr:.4f} "
              f"(valid={valid_frac:.0%})")

        if corr > best_score:
            best_score = corr
            best_rotation = rotation

    # The winning candidate's corners ARE the correct GCP mapping — return them
    # directly. No physical image rotation needed; GDAL handles the affine warp.
    best_corners = dict(candidates)[best_rotation]
    confidence = max(0, int(best_score * 100))
    return best_rotation, best_corners, confidence


def _fast_mini_georef(image_path, corners, img_w, img_h, target_deg,
                      clip_bounds=None):
    """Warp raw image to EPSG:4326 at very low res using corner GCPs.

    Args:
        clip_bounds: Optional (west, south, east, north) to restrict output.

    Returns (array, west, south, east, north) or None.
    """
    from osgeo import gdal, osr
    import uuid

    gdal.UseExceptions()

    # Build GCPs from corners mapping image pixels to lat/lon
    gcps = [
        gdal.GCP(corners["NW"][1], corners["NW"][0], 0, 0, 0),
        gdal.GCP(corners["NE"][1], corners["NE"][0], 0, img_w, 0),
        gdal.GCP(corners["SE"][1], corners["SE"][0], 0, img_w, img_h),
        gdal.GCP(corners["SW"][1], corners["SW"][0], 0, 0, img_h),
    ]

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    # Create in-memory GCP dataset
    tmp_id = uuid.uuid4().hex[:8]
    vsi_src = f"/vsimem/orient_src_{tmp_id}.vrt"

    try:
        src_ds = gdal.Open(image_path)
        if src_ds is None:
            return None

        tmp_ds = gdal.Translate(vsi_src, src_ds, GCPs=gcps,
                                outputSRS=srs.ExportToWkt())
        src_ds = None
        if tmp_ds is None:
            return None

        # Compute output bounds and size
        all_lats = [corners[k][0] for k in ("NW", "NE", "SE", "SW")]
        all_lons = [corners[k][1] for k in ("NW", "NE", "SE", "SW")]
        west, east = min(all_lons), max(all_lons)
        south, north = min(all_lats), max(all_lats)

        # Optionally clip to a restricted area (e.g. reference overlap)
        if clip_bounds is not None:
            west = max(west, clip_bounds[0])
            south = max(south, clip_bounds[1])
            east = min(east, clip_bounds[2])
            north = min(north, clip_bounds[3])
            if west >= east or south >= north:
                tmp_ds = None
                return None

        out_w = max(50, int((east - west) / target_deg))
        out_h = max(50, int((north - south) / target_deg))
        # Cap to prevent huge outputs
        if out_w > 2000:
            out_h = int(out_h * 2000 / out_w)
            out_w = 2000
        if out_h > 2000:
            out_w = int(out_w * 2000 / out_h)
            out_h = 2000

        warp_opts = gdal.WarpOptions(
            format="MEM",
            width=out_w,
            height=out_h,
            outputBounds=[west, south, east, north],
            dstSRS=srs.ExportToWkt(),
            resampleAlg=gdal.GRA_Bilinear,
            polynomialOrder=1,
        )

        # gdal.Warp with format=MEM requires empty string as destination
        dst_ds = gdal.Warp("", tmp_ds, options=warp_opts)
        tmp_ds = None

        if dst_ds is None:
            return None

        arr = dst_ds.GetRasterBand(1).ReadAsArray()
        dst_ds = None

        if arr is None or np.mean(arr > 0) < 0.05:
            return None

        return arr, west, south, east, north

    finally:
        gdal.Unlink(vsi_src)


def _make_simple_land_mask(arr):
    """Create a simple land mask: nonzero pixels = land."""
    if arr is None:
        return None
    return (arr > 0).astype(np.uint8)


def _compute_overlap_bounds(gt1, shape1, gt2, shape2):
    """Compute the overlapping region between two georeferenced images.

    Returns (x_min, y_min, x_max, y_max) in map coordinates, or None if no overlap.
    """
    # Image 1 bounds
    x1_min = gt1[0]
    x1_max = gt1[0] + gt1[1] * shape1[1]
    y1_max = gt1[3]  # top (y increases downward in pixel coords)
    y1_min = gt1[3] + gt1[5] * shape1[0]

    # Image 2 bounds
    x2_min = gt2[0]
    x2_max = gt2[0] + gt2[1] * shape2[1]
    y2_max = gt2[3]
    y2_min = gt2[3] + gt2[5] * shape2[0]

    # Overlap
    x_min = max(x1_min, x2_min)
    x_max = min(x1_max, x2_max)
    y_min = max(y1_min, y2_min)
    y_max = min(y1_max, y2_max)

    if x_min >= x_max or y_min >= y_max:
        return None

    return (x_min, y_min, x_max, y_max)


def _extract_region(arr, gt, bounds):
    """Extract a subregion from an array given map-coordinate bounds.

    Returns the cropped array.
    """
    x_min, y_min, x_max, y_max = bounds
    h, w = arr.shape

    # Convert map coords to pixel coords
    col_min = max(0, int((x_min - gt[0]) / gt[1]))
    col_max = min(w, int((x_max - gt[0]) / gt[1]))
    row_min = max(0, int((y_max - gt[3]) / gt[5]))  # y_max = top row
    row_max = min(h, int((y_min - gt[3]) / gt[5]))  # y_min = bottom row

    if col_min >= col_max or row_min >= row_max:
        return None

    return arr[row_min:row_max, col_min:col_max]


def _reproject_bounds_to_epsg3857(gt, shape, proj_wkt):
    """Convert image bounds to EPSG:3857 coordinates for overlap comparison.

    Returns (x_min, y_min, x_max, y_max) in EPSG:3857 meters.
    """
    from osgeo import osr

    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj_wkt)

    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(3857)

    h, w = shape

    # Image bounds in native CRS
    x_min = gt[0]
    x_max = gt[0] + gt[1] * w
    y_max = gt[3]
    y_min = gt[3] + gt[5] * h

    if srs.IsSame(target_srs):
        return (x_min, y_min, x_max, y_max)

    # Transform corners
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(srs, target_srs)

    corners = [
        transform.TransformPoint(x_min, y_min),
        transform.TransformPoint(x_max, y_max),
        transform.TransformPoint(x_min, y_max),
        transform.TransformPoint(x_max, y_min),
    ]

    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def _sift_orientation_check(geo_crop, ref_crop, candidate_rotations):
    """Test candidate rotations using SIFT feature matching.

    Returns (best_rotation, best_inliers) or (0, 0) if SIFT fails.
    """
    import cv2

    sift = cv2.SIFT_create(nfeatures=3000)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    ref_u8 = ref_crop.astype(np.uint8)
    if ref_u8.max() == 0:
        return 0, 0
    ref_eq = clahe.apply(ref_u8)
    kp_ref, desc_ref = sift.detectAndCompute(ref_eq, None)
    if desc_ref is None or len(kp_ref) < 10:
        return 0, 0

    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
        dict(checks=50),
    )

    best_rotation = 0
    best_inliers = 0

    for rotation in candidate_rotations:
        if rotation == 0:
            geo_test = geo_crop.copy()
        elif rotation == 180:
            geo_test = geo_crop[::-1, ::-1].copy()
        elif rotation == 90:
            geo_test = np.rot90(geo_crop, k=-1).copy()
        elif rotation == 270:
            geo_test = np.rot90(geo_crop, k=1).copy()
        else:
            continue

        geo_u8 = geo_test.astype(np.uint8)
        if geo_u8.max() == 0:
            continue
        geo_eq = clahe.apply(geo_u8)

        kp_geo, desc_geo = sift.detectAndCompute(geo_eq, None)
        if desc_geo is None or len(kp_geo) < 10:
            print(f"    SIFT rotation {rotation:>3}: too few keypoints")
            continue

        try:
            raw = flann.knnMatch(desc_geo, desc_ref, k=2)
        except cv2.error:
            continue

        good = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.7 * n.distance:
                    good.append(m)

        n_inliers = 0
        if len(good) >= 8:
            pts_geo = np.float32([kp_geo[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, inliers = cv2.estimateAffinePartial2D(
                pts_geo, pts_ref, method=cv2.RANSAC, ransacReprojThreshold=10.0
            )
            if inliers is not None:
                n_inliers = int(inliers.sum())

        print(f"    SIFT rotation {rotation:>3}: {n_inliers} inliers ({len(good)} good matches)")
        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_rotation = rotation

    return best_rotation, best_inliers


def verify_orientation_against_reference(georef_path: str, reference_path: str) -> int:
    """Verify georef orientation using SIFT feature matching + land mask fallback.

    Uses SIFT as the primary method (more discriminative than land masks),
    with land mask correlation as a fallback when SIFT is inconclusive.

    Tests 0, 180, and optionally 90/270 degree rotations.
    Returns the rotation (in degrees) needed to correct the orientation.
    0 means current orientation is correct.
    """
    import cv2
    from osgeo import osr

    # Read both at coarse resolution (~100m/px)
    geo_arr, geo_gt, geo_proj = _read_band_at_scale(georef_path, target_res_m=100.0)
    ref_arr, ref_gt, ref_proj = _read_band_at_scale(reference_path, target_res_m=100.0)

    if geo_arr is None or ref_arr is None:
        print("  Orientation verify: could not read images")
        return 0

    # Determine image bounds -- reproject to EPSG:3857 if different CRSes
    geo_srs = osr.SpatialReference()
    geo_srs.ImportFromWkt(geo_proj)
    ref_srs = osr.SpatialReference()
    ref_srs.ImportFromWkt(ref_proj)

    same_crs = geo_srs.IsSame(ref_srs)

    if same_crs:
        geo_bounds = (
            geo_gt[0],
            geo_gt[3] + geo_gt[5] * geo_arr.shape[0],
            geo_gt[0] + geo_gt[1] * geo_arr.shape[1],
            geo_gt[3],
        )
        ref_bounds = (
            ref_gt[0],
            ref_gt[3] + ref_gt[5] * ref_arr.shape[0],
            ref_gt[0] + ref_gt[1] * ref_arr.shape[1],
            ref_gt[3],
        )
    else:
        geo_bounds = _reproject_bounds_to_epsg3857(geo_gt, geo_arr.shape, geo_proj)
        ref_bounds = _reproject_bounds_to_epsg3857(ref_gt, ref_arr.shape, ref_proj)

    # Check overlap
    x_min = max(geo_bounds[0], ref_bounds[0])
    x_max = min(geo_bounds[2], ref_bounds[2])
    y_min = max(geo_bounds[1], ref_bounds[1])
    y_max = min(geo_bounds[3], ref_bounds[3])

    if x_min >= x_max or y_min >= y_max:
        print("  Orientation verify: no overlap between georef and reference")
        return 0

    # Compute pixel coordinates of overlap in each image
    def _bounds_to_pixels(bounds, arr_shape):
        bx_min, by_min, bx_max, by_max = bounds
        bw = bx_max - bx_min
        bh = by_max - by_min
        if bw <= 0 or bh <= 0:
            return 0, 0, 0, 0
        col_min = max(0, int((x_min - bx_min) / bw * arr_shape[1]))
        col_max = min(arr_shape[1], int((x_max - bx_min) / bw * arr_shape[1]))
        row_min = max(0, int((by_max - y_max) / bh * arr_shape[0]))
        row_max = min(arr_shape[0], int((by_max - y_min) / bh * arr_shape[0]))
        return row_min, row_max, col_min, col_max

    gr1, gr2, gc1, gc2 = _bounds_to_pixels(geo_bounds, geo_arr.shape)
    rr1, rr2, rc1, rc2 = _bounds_to_pixels(ref_bounds, ref_arr.shape)

    geo_crop = geo_arr[gr1:gr2, gc1:gc2]
    ref_crop = ref_arr[rr1:rr2, rc1:rc2]

    if geo_crop.size == 0 or ref_crop.size == 0:
        print("  Orientation verify: overlap too small")
        return 0

    # Resize both to common dimensions
    target_h = min(geo_crop.shape[0], ref_crop.shape[0], 500)
    target_w = min(geo_crop.shape[1], ref_crop.shape[1], 500)
    if target_h < 10 or target_w < 10:
        print("  Orientation verify: overlap too small")
        return 0

    geo_crop = cv2.resize(geo_crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
    ref_crop = cv2.resize(ref_crop, (target_w, target_h), interpolation=cv2.INTER_AREA)

    print(f"  Orientation verify: overlap region {target_w}x{target_h}px")

    # Determine which rotations to test
    geo_h, geo_w = geo_arr.shape
    aspect = geo_w / geo_h if geo_h > 0 else 1.0
    candidate_rotations = [0, 180]
    if aspect < 1.5 or aspect > 8.0:
        candidate_rotations.extend([90, 270])

    # --- Method 1: SIFT feature matching (primary, more discriminative) ---
    sift_best, sift_inliers = _sift_orientation_check(
        geo_crop, ref_crop, candidate_rotations)

    if sift_inliers >= 15:
        if sift_best != 0:
            print(f"  Orientation verify (SIFT): best rotation = {sift_best} "
                  f"({sift_inliers} inliers)")
        else:
            print(f"  Orientation verify (SIFT): current orientation correct "
                  f"({sift_inliers} inliers)")
        return sift_best

    # --- Method 2: Land mask correlation (fallback) ---
    print(f"  SIFT inconclusive ({sift_inliers} inliers), "
          f"falling back to land mask correlation...")

    geo_mask = _make_simple_land_mask(geo_crop)
    ref_mask = _make_simple_land_mask(ref_crop)

    mask_candidates = {
        0: geo_mask,
        180: geo_mask[::-1, ::-1],
    }
    if 90 in candidate_rotations:
        rot90 = np.rot90(geo_mask, k=-1)
        rot270 = np.rot90(geo_mask, k=1)
        if abs(rot90.shape[0] - ref_mask.shape[0]) < max(10, ref_mask.shape[0] * 0.3):
            rh = min(rot90.shape[0], ref_mask.shape[0])
            rw = min(rot90.shape[1], ref_mask.shape[1])
            if rh > 10 and rw > 10:
                mask_candidates[90] = rot90[:rh, :rw]
                mask_candidates[270] = rot270[:rh, :rw]

    best_rotation = 0
    best_corr = -1.0

    for rotation, mask in mask_candidates.items():
        mh = min(mask.shape[0], ref_mask.shape[0])
        mw = min(mask.shape[1], ref_mask.shape[1])
        if mh < 5 or mw < 5:
            continue

        m = mask[:mh, :mw].astype(np.float32)
        r = ref_mask[:mh, :mw].astype(np.float32)

        m_mean = m.mean()
        r_mean = r.mean()
        m_centered = m - m_mean
        r_centered = r - r_mean
        m_std = m_centered.std()
        r_std = r_centered.std()

        if m_std < 1e-6 or r_std < 1e-6:
            corr = 0.0
        else:
            corr = float(np.sum(m_centered * r_centered) / (m_std * r_std * m.size))

        print(f"    Land mask rotation {rotation:>3}: correlation = {corr:.4f}")
        if corr > best_corr:
            best_corr = corr
            best_rotation = rotation

    if best_rotation != 0:
        print(f"  Orientation verify (land mask): best rotation = {best_rotation} "
              f"(corr={best_corr:.4f})")
    else:
        print(f"  Orientation verify (land mask): current orientation correct "
              f"(corr={best_corr:.4f})")

    return best_rotation
