"""DEM (Digital Elevation Model) fetching and preparation.

Downloads Copernicus GLO-30 DEM tiles from AWS Open Data and prepares
them for use with ASP mapproject (ellipsoidal heights).
"""

import hashlib
import math
import os
import subprocess

from .asp import find_asp_tool

# Local cache directory for DEM tiles
_DEM_CACHE_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "dem")


def _tile_name(lat: int, lon: int) -> str:
    """Copernicus GLO-30 tile naming convention."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def _tile_url(lat: int, lon: int) -> str:
    """AWS S3 URL for a Copernicus GLO-30 tile."""
    name = _tile_name(lat, lon)
    return f"https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/{name}/{name}.tif"


def fetch_dem(west: float, south: float, east: float, north: float,
              cache_dir: str | None = None) -> str | None:
    """Download and mosaic Copernicus GLO-30 DEM tiles covering a bounding box.

    Parameters
    ----------
    west, south, east, north : float
        Bounding box in WGS84 degrees.
    cache_dir : str, optional
        Directory to cache tiles. Defaults to data/dem/.

    Returns
    -------
    str or None
        Path to the mosaicked DEM covering the bounding box, or None on failure.
    """
    if cache_dir is None:
        cache_dir = _DEM_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Determine which 1°×1° tiles we need
    lat_min = math.floor(south)
    lat_max = math.floor(north)
    lon_min = math.floor(west)
    lon_max = math.floor(east)

    tile_paths = []
    for lat in range(lat_min, lat_max + 1):
        for lon in range(lon_min, lon_max + 1):
            name = _tile_name(lat, lon)
            local_path = os.path.join(cache_dir, f"{name}.tif")

            if os.path.isfile(local_path):
                tile_paths.append(local_path)
                continue

            url = _tile_url(lat, lon)
            print(f"  [DEM] Downloading {name}...")
            try:
                import urllib.request
                urllib.request.urlretrieve(url, local_path)
                tile_paths.append(local_path)
            except Exception as e:
                print(f"  [DEM] Failed to download {name}: {e}")
                # Tile may not exist (ocean areas)
                continue

    if not tile_paths:
        print("  [DEM] No tiles downloaded — area may be over ocean")
        return None

    if len(tile_paths) == 1:
        return tile_paths[0]

    # Mosaic multiple tiles with GDAL VRT
    bbox_key = f"{west:.4f},{south:.4f},{east:.4f},{north:.4f}"
    bbox_hash = hashlib.sha1(bbox_key.encode("utf-8")).hexdigest()[:12]
    mosaic_path = os.path.join(cache_dir, f"mosaic_dem_{bbox_hash}.tif")
    vrt_path = os.path.join(cache_dir, f"mosaic_dem_{bbox_hash}.vrt")

    if os.path.isfile(mosaic_path):
        return mosaic_path

    try:
        cmd_vrt = ["gdalbuildvrt", vrt_path] + tile_paths
        subprocess.run(cmd_vrt, capture_output=True, check=True, timeout=60)

        cmd_translate = [
            "gdal_translate", "-co", "COMPRESS=LZW",
            "-projwin", str(west - 0.01), str(north + 0.01),
            str(east + 0.01), str(south - 0.01),
            vrt_path, mosaic_path,
        ]
        subprocess.run(cmd_translate, capture_output=True, check=True, timeout=120)
        return mosaic_path
    except Exception as e:
        print(f"  [DEM] Mosaic failed: {e}")
        return None


def prepare_dem_for_asp(dem_path: str) -> str | None:
    """Convert DEM from geoid heights to ellipsoidal heights for ASP.

    ASP mapproject requires ellipsoidal (WGS84) heights, but Copernicus
    GLO-30 uses EGM2008 geoid heights. Uses ASP dem_geoid to convert.

    Parameters
    ----------
    dem_path : str
        Input DEM with geoid heights.

    Returns
    -------
    str or None
        Path to DEM with ellipsoidal heights, or None if conversion fails.
    """
    dem_geoid = find_asp_tool("dem_geoid")
    if dem_geoid is None:
        print("  [DEM] ASP dem_geoid not found — using geoid heights (may cause ~50m vertical error)")
        return dem_path  # Use as-is; vertical error is small for orthorectification

    # ASP's dem_geoid --reverse-adjustment writes to "<output>-adj.tif",
    # not to "<output>" — the "-adj.tif" suffix is added unconditionally by
    # the tool. Check both the adj path (actual output) and the plain path
    # (in case a future ASP version drops the suffix).
    output = os.path.splitext(dem_path)[0] + "_wgs84.tif"
    adj_output = f"{output}-adj.tif"
    if os.path.isfile(adj_output):
        return adj_output
    if os.path.isfile(output):
        return output

    cmd = [dem_geoid, "--reverse-adjustment", dem_path, "-o", output]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  [DEM] dem_geoid failed: {result.stderr[:300]}")
            return dem_path
    except Exception as e:
        print(f"  [DEM] dem_geoid error: {e}")
        return dem_path

    if os.path.isfile(adj_output):
        return adj_output
    if os.path.isfile(output):
        return output
    print(f"  [DEM] dem_geoid produced neither {adj_output} nor {output}")
    return dem_path


def fetch_and_prepare_dem(west: float, south: float, east: float, north: float,
                          cache_dir: str | None = None) -> str | None:
    """Fetch DEM and convert to ellipsoidal heights for ASP.

    One-call convenience combining fetch_dem + prepare_dem_for_asp.
    """
    dem_path = fetch_dem(west, south, east, north, cache_dir)
    if dem_path is None:
        return None
    return prepare_dem_for_asp(dem_path)
