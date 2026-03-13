"""Optional Sentinel-2 reference image auto-fetch via Element 84 Earth Search STAC API."""

import json
import os
import subprocess
import urllib.request


STAC_API = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"


def fetch_sentinel2_reference(bbox: tuple, output_path: str,
                              max_cloud_cover: int = 10) -> str:
    """Download a recent cloud-free Sentinel-2 composite for an area.

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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    west, south, east, north = bbox
    print(f"  Searching Sentinel-2 for bbox: {west:.2f},{south:.2f},{east:.2f},{north:.2f}")

    # STAC search for recent, low-cloud images
    search_body = json.dumps({
        "collections": [COLLECTION],
        "bbox": [west, south, east, north],
        "query": {
            "eo:cloud_cover": {"lt": max_cloud_cover},
        },
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        "limit": 5,
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

    # Use the most recent result
    item = features[0]
    item_id = item["id"]
    cloud_cover = item["properties"].get("eo:cloud_cover", "?")
    date = item["properties"].get("datetime", "?")[:10]
    print(f"  Using: {item_id} ({date}, {cloud_cover}% cloud)")

    # Get RGB band URLs (B04=Red, B03=Green, B02=Blue at 10m)
    assets = item.get("assets", {})
    band_keys = ["red", "green", "blue"]  # Element 84 naming
    band_urls = []
    for key in band_keys:
        if key in assets:
            band_urls.append(assets[key]["href"])
        else:
            raise RuntimeError(f"Missing '{key}' band in STAC item {item_id}")

    # Build VRT from the 3 bands, then warp to EPSG:3857 clipped to bbox
    temp_dir = os.path.dirname(output_path)
    vrt_path = os.path.join(temp_dir, "sentinel2_ref.vrt")

    # Use gdalbuildvrt with /vsicurl/ paths
    vsicurl_urls = [f"/vsicurl/{url}" for url in band_urls]

    # Build a separate VRT that stacks the 3 bands
    cmd_vrt = [
        "gdalbuildvrt",
        "-separate",  # Stack as separate bands
        vrt_path,
    ] + vsicurl_urls

    _run_cmd(cmd_vrt)

    # Warp to EPSG:3857, clip to bbox
    _run_cmd([
        "gdalwarp",
        "-s_srs", "EPSG:32639",  # UTM zone varies; let GDAL auto-detect from source
        "-t_srs", "EPSG:3857",
        "-te_srs", "EPSG:4326",
        "-te", str(west), str(south), str(east), str(north),
        "-r", "bilinear",
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=2",
        "-co", "TILED=YES",
        vrt_path,
        output_path,
    ])

    if os.path.exists(vrt_path):
        os.remove(vrt_path)

    print(f"  Sentinel-2 reference: {output_path}")
    return output_path


def _run_cmd(cmd, check=True):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result
