#!/usr/bin/env python3
"""Convert an RGB GeoTIFF reference to single-band panchromatic-style luminance.

Modern basemaps (ESRI World Imagery, Sentinel-2) are RGB; declassified KH
imagery is panchromatic film (single grayscale channel, peaks ~500-650 nm).
RoMa / ELoFTR cross-modal matching is unreliable when one input is RGB color
and the other is panchromatic film (memory/secondary_references_dead_code.md
documents 0 RANSAC survivors on DA026 against the modern composite ref).

Standard luminance weighting (Rec. 601, also used by ITU-R BT.601):
    Y = 0.299 R + 0.587 G + 0.114 B

is a reasonable approximation of panchromatic film response (broadband visible
sensitivity weighted toward green). Better than naive single-band selection
because it preserves green-vegetation contrast that R or B alone would lose.

Block-streamed to handle 7+ GB inputs without OOM. Output is single-band
uint8 with LZW compression.

Usage
-----
    poetry run python scripts/build_esri_pan.py \
        --input ~/.cache/declass-process/bahrain_esri_worldimagery.tif \
        --output ~/.cache/declass-process/bahrain_esri_worldimagery_pan.tif

To register as a reference name, add to data/local_paths.yaml:

    references:
      bahrain_esri_worldimagery_pan: "~/.cache/declass-process/bahrain_esri_worldimagery_pan.tif"
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


# Rec. 601 luminance weights — broadband visible response, vegetation-friendly.
# Adjust if you want a different panchromatic flavor (e.g., 0.5/0.4/0.1 to
# de-emphasize blue water reflectance).
WEIGHTS = (0.299, 0.587, 0.114)


def rgb_to_pan(src_path: str, dst_path: str, block_h: int = 2048,
               block_w: int = 2048) -> None:
    """Stream-convert an RGB raster to single-band luminance.

    Reads in tiles of ``block_h × block_w`` to bound memory use. The output
    matches the input geotransform / CRS / NoData (when first band's NoData
    is set) and uses LZW compression with internal tiling for fast random
    access in subsequent matching.
    """
    with rasterio.open(src_path) as src:
        if src.count < 3:
            raise SystemExit(
                f"{src_path} has only {src.count} band(s); need 3 (RGB).")
        if any(d != "uint8" for d in src.dtypes[:3]):
            print(
                f"WARNING: input dtypes are {src.dtypes[:3]}; "
                f"output will still be uint8 (clipped at 255)."
            )

        nodata = src.nodatavals[0] if src.nodatavals else None

        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            compress="lzw",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="YES",
            photometric="MINISBLACK",
            interleave="band",
        )
        if nodata is not None:
            profile["nodata"] = nodata

        # Drop the alpha-related tags that no longer apply to a 1-band raster
        for key in ("PHOTOMETRIC", "ALPHA"):
            profile.pop(key, None)

        os.makedirs(os.path.dirname(os.path.abspath(dst_path)) or ".", exist_ok=True)

        H, W = src.height, src.width
        n_blocks = ((H + block_h - 1) // block_h) * ((W + block_w - 1) // block_w)
        print(f"Source : {src_path}")
        print(f"  size : {W} x {H} pixels, {src.count} bands ({src.dtypes[0]})")
        print(f"  bounds: {src.bounds}")
        print(f"  CRS  : {src.crs}")
        print(f"Dest   : {dst_path}")
        print(f"Blocks : {n_blocks}")

        with rasterio.open(dst_path, "w", **profile) as dst:
            t0 = time.time()
            done = 0
            for row in range(0, H, block_h):
                for col in range(0, W, block_w):
                    h = min(block_h, H - row)
                    w = min(block_w, W - col)
                    win = Window(col, row, w, h)
                    rgb = src.read([1, 2, 3], window=win).astype(np.float32)
                    # Mask: any band == nodata → output nodata
                    if nodata is not None:
                        nd_mask = (rgb[0] == nodata) | (rgb[1] == nodata) | (rgb[2] == nodata)
                    else:
                        nd_mask = None
                    pan = (
                        WEIGHTS[0] * rgb[0]
                        + WEIGHTS[1] * rgb[1]
                        + WEIGHTS[2] * rgb[2]
                    )
                    pan = np.clip(pan, 0, 255).astype(np.uint8)
                    if nd_mask is not None and nodata is not None:
                        pan[nd_mask] = nodata
                    dst.write(pan, 1, window=win)
                    done += 1
                    if done % 50 == 0 or done == n_blocks:
                        pct = 100.0 * done / n_blocks
                        elapsed = time.time() - t0
                        eta = (elapsed / done) * (n_blocks - done) if done else 0
                        print(
                            f"  block {done}/{n_blocks} ({pct:.1f}%)  "
                            f"elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                            flush=True,
                        )
        # Build overviews for fast pyramid reads (required for 60k×75k tifs)
        print("Building overviews...")
        with rasterio.open(dst_path, "r+") as dst:
            dst.build_overviews([2, 4, 8, 16, 32, 64, 128], rasterio.enums.Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")

    print(f"Done: {dst_path}")
    print(f"Size: {os.path.getsize(dst_path) / 1024 / 1024:.0f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True, type=str)
    ap.add_argument("--output", "-o", required=True, type=str)
    ap.add_argument("--block-h", type=int, default=2048)
    ap.add_argument("--block-w", type=int, default=2048)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing output")
    args = ap.parse_args()

    src = os.path.expanduser(args.input)
    dst = os.path.expanduser(args.output)
    if not os.path.exists(src):
        sys.exit(f"Input not found: {src}")
    if os.path.exists(dst) and not args.force:
        sys.exit(f"Output exists (pass --force to overwrite): {dst}")

    rgb_to_pan(src, dst, block_h=args.block_h, block_w=args.block_w)


if __name__ == "__main__":
    main()
