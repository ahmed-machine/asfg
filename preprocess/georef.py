"""Rough georeferencing, coarse alignment cropping, and Sentinel-2 reference fetch."""

import json
import os
import subprocess
import urllib.request
from collections import defaultdict

import numpy as np

from . import run_gdal_cmd as _run_cmd


def georef_with_corners(input_path: str, output_path: str, corners: dict):
    """Georeference an image using 4-corner GCPs.

    Args:
        input_path: Ungeoreferenced input TIFF.
        output_path: Output georeferenced TIFF in EPSG:3857.
        corners: Dict with NW, NE, SE, SW keys, each (lat, lon) tuple.
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

    # Assign GCPs with gdal_translate
    temp_gcp = output_path.replace(".tif", "_gcp.tif")

    cmd_translate = [
        "gdal_translate",
        "-a_srs", "EPSG:4326",
        "-gcp", "0", "0", str(nw_lon), str(nw_lat),
        "-gcp", str(width), "0", str(ne_lon), str(ne_lat),
        "-gcp", str(width), str(height), str(se_lon), str(se_lat),
        "-gcp", "0", str(height), str(sw_lon), str(sw_lat),
        input_path,
        temp_gcp,
    ]
    _run_cmd(cmd_translate)

    # Warp to EPSG:3857
    cmd_warp = [
        "gdalwarp",
        "-s_srs", "EPSG:4326",
        "-t_srs", "EPSG:3857",
        "-order", "1",
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

    Returns:
        Path to the cropped output, or None on failure.
    """
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from pyproj import CRS, Transformer

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

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

        # Template match: slide reference mask over target mask
        import cv2
        if (ref_mask.shape[0] >= tgt_mask.shape[0] or
                ref_mask.shape[1] >= tgt_mask.shape[1]):
            print(f"  [coarse_crop] Reference larger than target, no crop needed")
            # Just pass through
            import shutil
            shutil.copy2(target_path, output_path)
            return output_path

        result = cv2.matchTemplate(tgt_mask, ref_mask, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # max_loc is (col, row) of the best match in target pixel coords
        match_col, match_row = max_loc

        # Convert pixel offset to metric offset
        # The offset = where the reference template's origin lands in target coords
        # Target origin in work CRS:
        tgt_origin_x = tgt_transform.c  # left
        tgt_origin_y = tgt_transform.f  # top
        # Match position in work CRS:
        match_x = tgt_origin_x + match_col * coarse_res
        match_y = tgt_origin_y - match_row * coarse_res  # y decreases downward

        # Reference origin in work CRS:
        ref_origin_x = ref_transform.c
        ref_origin_y = ref_transform.f

        # Offset: how much to shift the target so its content aligns with reference
        dx_m = ref_origin_x - match_x
        dy_m = ref_origin_y - match_y

        total_offset = np.sqrt(dx_m**2 + dy_m**2)
        print(f"  [coarse_crop] Coarse offset: dx={dx_m:+.0f}m, dy={dy_m:+.0f}m "
              f"(total={total_offset:.0f}m, corr={max_val:.3f})")

        if max_val < 0.05:
            print(f"  [coarse_crop] Correlation too low ({max_val:.3f}), skipping crop")
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

        vrt_path = os.path.join(temp_dir, "sentinel2_ref.vrt")
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
                tile_vrt = os.path.join(temp_dir, f"sentinel2_tile_{i}.vrt")
                vsicurl_urls = [f"/vsicurl/{url}" for url in band_urls]
                _run_cmd(["gdalbuildvrt", "-separate", tile_vrt] + vsicurl_urls)
                tile_vrts.append(tile_vrt)

        if not tile_vrts:
            raise RuntimeError("No valid tiles after filtering")

        # Warp each tile to EPSG:3857 clipped to bbox, then mosaic
        tile_warped = []
        for i, tvrt in enumerate(tile_vrts):
            tw = os.path.join(temp_dir, f"sentinel2_warped_{i}.tif")
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
        mosaic_vrt = os.path.join(temp_dir, "sentinel2_mosaic.vrt")
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
