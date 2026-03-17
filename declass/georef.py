"""Rough georeferencing using CSV corner coordinates."""

import json
import os
import subprocess

from .shell import run_gdal_cmd as _run_cmd


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
