"""Rough georeferencing and Sentinel-2 reference image auto-fetch."""

import json
import os
import subprocess
import urllib.request
from collections import defaultdict

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
