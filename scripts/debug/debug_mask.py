#!/usr/bin/env python3
"""
Standalone diagnostic for the CoastalObiaMaskProvider land/water mask.

Loads the offset image at 15m resolution (same as the coarse detection step),
runs CoastalObiaMaskProvider.build_masks() with diagnostic dumps enabled,
and saves all intermediate images to diagnostics/mask_debug/.

Usage:
    python3 scripts/debug/debug_mask.py [--res 15] [--source offset|ref|both]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject, transform_bounds

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paths_config import get_target, get_reference

TARGET = get_target("bahrain_1977")
REFERENCE = get_reference("kh9_dzb1212")


def get_utm_crs(lon, lat):
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def load_at_resolution(src_path, overlap, work_crs, res):
    """Load a raster reprojected into work_crs at the given resolution."""
    left, bottom, right, top = overlap
    width = int(round((right - left) / res))
    height = int(round((top - bottom) / res))
    dst_transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)
    dst_array = np.zeros((height, width), dtype=np.float32)
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=work_crs,
            resampling=Resampling.bilinear,
        )
    return dst_array


def compute_overlap(src_path_a, src_path_b, work_crs):
    """Compute overlap bounds in work_crs."""
    with rasterio.open(src_path_a) as a, rasterio.open(src_path_b) as b:
        a_bounds = transform_bounds(a.crs, work_crs, *a.bounds)
        b_bounds = transform_bounds(b.crs, work_crs, *b.bounds)
    overlap = (
        max(a_bounds[0], b_bounds[0]),
        max(a_bounds[1], b_bounds[1]),
        min(a_bounds[2], b_bounds[2]),
        min(a_bounds[3], b_bounds[3]),
    )
    if overlap[0] >= overlap[2] or overlap[1] >= overlap[3]:
        raise ValueError("No overlap between the two rasters")
    return overlap


def main():
    parser = argparse.ArgumentParser(description="Debug mask diagnostics")
    parser.add_argument("--res", type=float, default=15.0,
                        help="Resolution in m/px (default: 15)")
    parser.add_argument("--source", choices=["offset", "ref", "both"], default="offset",
                        help="Which image to run the mask on (default: offset)")
    parser.add_argument("--target", default=TARGET, help="Offset image path")
    parser.add_argument("--reference", default=REFERENCE, help="Reference image path")
    args = parser.parse_args()

    from align.semantic_masking import CoastalObiaMaskProvider

    # Determine work CRS from reference center
    with rasterio.open(args.reference) as ref:
        ref_bounds = ref.bounds
        center_lon = (ref_bounds.left + ref_bounds.right) / 2
        center_lat = (ref_bounds.bottom + ref_bounds.top) / 2
    work_crs = get_utm_crs(center_lon, center_lat)
    print(f"Work CRS: EPSG:{work_crs.to_epsg()}")

    overlap = compute_overlap(args.target, args.reference, work_crs)
    print(f"Overlap bounds: {overlap}")

    sources = []
    if args.source in ("offset", "both"):
        sources.append(("offset", args.target))
    if args.source in ("ref", "both"):
        sources.append(("ref", args.reference))

    provider = CoastalObiaMaskProvider()

    for label, src_path in sources:
        debug_dir = str(PROJECT_ROOT / "diagnostics" / "mask_debug" / label)
        print(f"\n{'='*60}")
        print(f"Running mask on: {label} ({Path(src_path).name})")
        print(f"Resolution: {args.res} m/px")
        print(f"Debug output: {debug_dir}")
        print(f"{'='*60}")

        t0 = time.time()
        arr = load_at_resolution(src_path, overlap, work_crs, args.res)
        print(f"Loaded array: shape={arr.shape}, valid={np.count_nonzero(arr > 0)}/{arr.size} "
              f"({np.mean(arr > 0)*100:.1f}%)")
        print(f"  Load time: {time.time()-t0:.1f}s")

        t1 = time.time()
        bundle = provider.build_masks(arr, _debug_dir=debug_dir)
        elapsed = time.time() - t1
        print(f"\n  Mask computation time: {elapsed:.1f}s")

        land = bundle.land > 0.5
        water = bundle.water > 0.5
        shal = bundle.shallow_water > 0.5
        valid = arr > 0
        print(f"\n  Final mask stats:")
        print(f"    land:          {np.count_nonzero(land):>8} px ({np.mean(land[valid])*100:.1f}% of valid)")
        print(f"    water:         {np.count_nonzero(water):>8} px ({np.mean(water[valid])*100:.1f}% of valid)")
        print(f"    shallow_water: {np.count_nonzero(shal):>8} px ({np.mean(shal[valid])*100:.1f}% of valid)")
        print(f"    stable:        {np.count_nonzero(bundle.stable > 0.5):>8} px")
        print(f"    shoreline:     {np.count_nonzero(bundle.shoreline > 0.5):>8} px")

    print(f"\nDone. Check diagnostics/mask_debug/ for images.")


if __name__ == "__main__":
    main()
