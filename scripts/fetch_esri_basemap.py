#!/usr/bin/env python3
"""Fetch an ESRI World Imagery basemap tile mosaic for an AOI boundary.

Writes a single EPSG:3857 GeoTIFF usable as a modern high-resolution
reference for the declass-process alignment pipeline. Uses the GDAL WMS
minidriver — no third-party dependencies beyond the GDAL Python bindings
already pulled in for the rest of the repo.

Output resolution is determined by ``--zoom``; at zoom 17 the tile pyramid
is roughly 1.2 m/px at the equator (scales with cos(lat) — ~1.1 m/px at
26 °N).

Attribution: ESRI World Imagery is Maxar/Earthstar Geographics sourced.
The service terms permit "evaluation, research, and non-commercial use"
when attribution is preserved. The required attribution string is written
into the output file's ``TIFFTAG_IMAGEDESCRIPTION`` tag. For a commercial
release the reference must be swapped for a licensed Maxar product.

Usage
-----
Populate the target path in ``data/local_paths.yaml`` (add a
``references.bahrain_esri_worldimagery`` entry), then::

    python scripts/fetch_esri_basemap.py \\
        --boundary data/bahrain_boundary.geojson \\
        --output /path/to/references/bahrain_esri_worldimagery.tif \\
        --zoom 17

Pass ``--dry-run`` to print the gdal commands without executing them.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ESRI_ATTRIBUTION = (
    "Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community "
    "(ArcGIS REST World_Imagery MapServer, TOS: evaluation / research / "
    "non-commercial use with attribution preserved)."
)


def _boundary_bbox_wgs84(boundary_path: str) -> tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) for a GeoJSON boundary.

    Handles FeatureCollection / Feature / bare Geometry. Walks nested
    coordinate arrays — no shapely dependency.
    """
    with open(boundary_path) as fh:
        gj = json.load(fh)

    def _walk(obj):
        if isinstance(obj, list):
            # A coordinate pair is [lon, lat] (+ optional elev).
            if (len(obj) >= 2
                    and isinstance(obj[0], (int, float))
                    and isinstance(obj[1], (int, float))):
                yield float(obj[0]), float(obj[1])
                return
            for item in obj:
                yield from _walk(item)
        elif isinstance(obj, dict):
            if "coordinates" in obj:
                yield from _walk(obj["coordinates"])
            if "geometry" in obj:
                yield from _walk(obj["geometry"])
            if "features" in obj:
                yield from _walk(obj["features"])

    lons, lats = [], []
    for lon, lat in _walk(gj):
        lons.append(lon)
        lats.append(lat)
    if not lons:
        raise ValueError(f"No coordinates found in {boundary_path}")
    return (min(lons), min(lats), max(lons), max(lats))


def _wgs84_to_mercator(lon: float, lat: float) -> tuple[float, float]:
    """Project (lon, lat) in degrees to EPSG:3857 (metres).

    Inline formula — avoids adding a pyproj dependency here since this
    script may run in a minimal GDAL-only environment.
    """
    import math

    R = 6_378_137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y


def _write_wms_xml(path: str, zoom: int) -> None:
    """Write a GDAL WMS XML descriptor for the ESRI World Imagery TMS.

    The ArcGIS REST tile endpoint follows the ``{z}/{y}/{x}`` TMS-with-
    top-origin scheme.
    """
    xml = f"""<GDAL_WMS>
    <Service name="TMS">
        <ServerUrl>https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/${{z}}/${{y}}/${{x}}</ServerUrl>
    </Service>
    <DataWindow>
        <UpperLeftX>-20037508.34</UpperLeftX>
        <UpperLeftY>20037508.34</UpperLeftY>
        <LowerRightX>20037508.34</LowerRightX>
        <LowerRightY>-20037508.34</LowerRightY>
        <TileLevel>{int(zoom)}</TileLevel>
        <TileCountX>1</TileCountX>
        <TileCountY>1</TileCountY>
        <YOrigin>top</YOrigin>
    </DataWindow>
    <Projection>EPSG:3857</Projection>
    <BlockSizeX>256</BlockSizeX>
    <BlockSizeY>256</BlockSizeY>
    <BandsCount>3</BandsCount>
    <Cache>
        <Path>{tempfile.gettempdir()}/gdal_wms_esri_worldimagery</Path>
    </Cache>
    <UserAgent>declass-process/scripts/fetch_esri_basemap.py</UserAgent>
</GDAL_WMS>
"""
    Path(path).write_text(xml)


def _find_gdal(name: str) -> str:
    """Prefer homebrew's GDAL CLI over ASP's bundled copy.

    Mirrors the resolution order in ``preprocess/camera_model.py:_find_gdal_tool``
    — ASP's GDAL 3.8 can shadow the homebrew version rasterio links
    against, leading to reader/writer version skew.
    """
    brew = f"/opt/homebrew/bin/{name}"
    if os.path.isfile(brew):
        return brew
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"GDAL CLI '{name}' not found on PATH")


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def fetch(
    boundary_path: str,
    output_path: str,
    *,
    zoom: int = 17,
    pad_km: float = 5.0,
    dry_run: bool = False,
    force: bool = False,
) -> str:
    """Fetch ESRI World Imagery over the boundary AOI and write a GeoTIFF.

    Parameters
    ----------
    boundary_path
        GeoJSON of the AOI. Any nested geometry is walked for coord extrema.
    output_path
        Destination GeoTIFF path.
    zoom
        XYZ tile pyramid level to request. 17 ≈ 1.2 m/px at equator.
    pad_km
        Bbox padding in km beyond the boundary extrema.
    dry_run
        If True, only print the gdal commands.
    force
        If True, re-fetch even if the output file already exists.
    """
    out = Path(output_path)
    if out.exists() and not force:
        print(f"  [fetch_esri_basemap] Output exists: {out}")
        print(f"  [fetch_esri_basemap] Pass --force to re-fetch.")
        return str(out)

    lon_min, lat_min, lon_max, lat_max = _boundary_bbox_wgs84(boundary_path)
    print(f"  [fetch_esri_basemap] Boundary bbox (WGS84): "
          f"lon {lon_min:.4f}..{lon_max:.4f}, lat {lat_min:.4f}..{lat_max:.4f}")

    # Reproject corners to EPSG:3857 for -projwin.
    ul_x, ul_y = _wgs84_to_mercator(lon_min, lat_max)
    lr_x, lr_y = _wgs84_to_mercator(lon_max, lat_min)
    # Pad in metres at this latitude. cos(lat) compensates longitude
    # compression; in mercator both axes are scaled by 1/cos(lat) already,
    # so a metre-pad in mercator is slightly larger than a metre-pad on
    # the ground. Acceptable — pad is a comfort margin, not a measurement.
    pad_m = pad_km * 1000.0
    ul_x -= pad_m
    ul_y += pad_m
    lr_x += pad_m
    lr_y -= pad_m
    print(f"  [fetch_esri_basemap] Request bbox (EPSG:3857, +{pad_km} km pad):")
    print(f"    UL=({ul_x:.1f}, {ul_y:.1f})  LR=({lr_x:.1f}, {lr_y:.1f})")

    gdal_translate = _find_gdal("gdal_translate")

    with tempfile.TemporaryDirectory(prefix="esri_basemap_") as tmpdir:
        wms_xml = os.path.join(tmpdir, "esri_worldimagery.xml")
        _write_wms_xml(wms_xml, zoom=zoom)
        print(f"  [fetch_esri_basemap] WMS XML at {wms_xml}")

        out.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            gdal_translate,
            "-projwin", f"{ul_x}", f"{ul_y}", f"{lr_x}", f"{lr_y}",
            "-projwin_srs", "EPSG:3857",
            "-of", "GTiff",
            "-co", "COMPRESS=LZW",
            "-co", "TILED=YES",
            "-co", "BIGTIFF=IF_SAFER",
            "-co", "PREDICTOR=2",
            "-mo", f"TIFFTAG_IMAGEDESCRIPTION={ESRI_ATTRIBUTION}",
            wms_xml,
            str(out),
        ]
        _run(cmd, dry_run=dry_run)

    if not dry_run and not out.exists():
        raise RuntimeError(f"gdal_translate did not produce {out}")

    if not dry_run:
        sz_mb = out.stat().st_size / (1024 * 1024)
        print(f"  [fetch_esri_basemap] Wrote {out} ({sz_mb:.1f} MB)")
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boundary", required=True,
                    help="Path to GeoJSON boundary of the AOI.")
    ap.add_argument("--output", required=True,
                    help="Destination GeoTIFF path.")
    ap.add_argument("--zoom", type=int, default=17,
                    help="XYZ tile pyramid zoom level (default 17 ~ 1.2 m/px).")
    ap.add_argument("--pad-km", type=float, default=5.0,
                    help="Pad around the boundary bbox in km (default 5).")
    ap.add_argument("--force", action="store_true",
                    help="Re-fetch even if the output file exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the gdal commands without executing.")
    args = ap.parse_args()

    print(f"  [fetch_esri_basemap] {ESRI_ATTRIBUTION}")
    fetch(
        boundary_path=args.boundary,
        output_path=args.output,
        zoom=args.zoom,
        pad_km=args.pad_km,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == "__main__":
    main()
