#!/usr/bin/env python3
"""Detect + inpaint white diagonal film scratches on a reference TIF.

Writes a ``<ref>.scratch_cleaned.tif`` sidecar plus a provenance JSON so
the alignment pipeline (``align/geo.py::read_overlap_region``) picks it
up transparently on subsequent runs. Re-running on an unchanged
reference + params is a no-op.

Optionally writes a binary mask TIF for QGIS inspection. Run it once
per reference; matching cost on subsequent alignment runs is unchanged
(the cleaned sidecar is read instead of the raw reference).

Examples:
    # Clean the 1976 KH-9 reference once
    python3 scripts/clean_reference_scratches.py \\
        ~/Code/openmaps/public/maps/1976-KH9-DZB1212.warped.tif

    # Same plus a mask sidecar for visual QC in QGIS
    python3 scripts/clean_reference_scratches.py \\
        ~/Code/openmaps/public/maps/1976-KH9-DZB1212.warped.tif \\
        --write-mask

    # Tune the detector for a noisy reference (more aggressive)
    python3 scripts/clean_reference_scratches.py REF.tif \\
        --min-length 100 --brightness-percentile 92
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rasterio

from align.film_scratches import (
    ScratchDetectorParams,
    detect_scratches,
    generate_scratch_cleaned_reference,
)
import paths


def _write_mask(ref_path: str, mask_path: str) -> None:
    """Write a binary scratch mask alongside the reference. Useful for
    visual QC: drag the mask into QGIS over the reference and confirm
    the detector is hitting scratches, not real geographic features."""
    print(f"  [scratch_clean] writing mask preview: {mask_path}", flush=True)
    with rasterio.open(ref_path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
    mask = detect_scratches(arr)
    profile.update(dtype="uint8", count=1, nodata=0,
                   compress="LZW", bigtiff="IF_SAFER")
    if profile.get("height", 0) >= 256 and profile.get("width", 0) >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(mask, 1)
    pct = float(np.count_nonzero(mask)) / mask.size * 100
    print(f"  [scratch_clean] mask covers {pct:.2f}% of pixels", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inpaint white diagonal film scratches on a reference TIF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("reference",
                        help="Path to the reference GeoTIFF to clean.")
    parser.add_argument("--write-mask", action="store_true",
                        help="Also write a binary scratch-mask TIF next to the "
                             "reference for visual QC.")
    parser.add_argument("--min-length", type=int, default=200,
                        help="Minimum scratch length in pixels (default: 200).")
    parser.add_argument("--max-width", type=int, default=8,
                        help="Maximum scratch width in pixels (default: 8).")
    parser.add_argument("--brightness-percentile", type=float, default=97.0,
                        help="Top-hat threshold percentile (default: 97).")
    parser.add_argument("--hough-threshold", type=int, default=80,
                        help="Hough line accumulator threshold (default: 80).")
    parser.add_argument("--angle-range", action="append", metavar="LO,HI",
                        default=None,
                        help="Allowed scratch angle range, in degrees from "
                             "horizontal (0=horizontal, ±90=vertical). "
                             "Repeat to add multiple disjoint ranges, e.g. "
                             "'--angle-range -90,-75 --angle-range 75,90' "
                             "for vertical-only. Default covers near-vertical "
                             "(±75–90°) AND diagonal (±10–45°).")
    parser.add_argument("--cluster-mad-k", type=float, default=None,
                        help="Robust outlier rejection: keep only Hough lines "
                             "within k MADs of the dominant-angle cluster. "
                             "Set to 0 to disable. Default: 3.0.")
    parser.add_argument("--cluster-mad-floor", type=float, default=None,
                        help="Minimum cluster tolerance in degrees, "
                             "regardless of MAD. Prevents zero-tolerance "
                             "rejection on tight clusters. Default: 2.0.")
    parser.add_argument("--cluster-min-lines", type=int, default=None,
                        help="Skip clustering when fewer than this many "
                             "candidate lines pass the angle prefilter. "
                             "Default: 3.")
    parser.add_argument("--cluster-max-short-axis", type=int, default=None,
                        help="Reject mask components whose minor axis "
                             "(rotated bounding rect) exceeds this many "
                             "pixels — catches heavy-overlap blobs from "
                             "fragmented Hough segments. Set to 0 to "
                             "disable. Default: 25.")
    args = parser.parse_args()

    ref_path = os.path.abspath(os.path.expanduser(args.reference))
    if not os.path.exists(ref_path):
        print(f"ERROR: reference not found: {ref_path}", file=sys.stderr)
        return 2

    angle_ranges = None
    if args.angle_range:
        angle_ranges = []
        for raw in args.angle_range:
            try:
                lo_str, hi_str = raw.split(",")
                lo = float(lo_str)
                hi = float(hi_str)
            except (ValueError, AttributeError):
                print(f"ERROR: invalid --angle-range {raw!r}; expected "
                      f"'LO,HI' (e.g. '-90,-75')", file=sys.stderr)
                return 2
            if lo >= hi or not (-90.0 <= lo and hi <= 90.0):
                print(f"ERROR: invalid --angle-range {raw!r}; LO < HI in "
                      f"[-90, 90]", file=sys.stderr)
                return 2
            angle_ranges.append((lo, hi))

    params_kwargs = dict(
        min_length_px=args.min_length,
        max_width_px=args.max_width,
        brightness_percentile=args.brightness_percentile,
        hough_threshold=args.hough_threshold,
    )
    if angle_ranges is not None:
        params_kwargs["angle_ranges_deg"] = tuple(angle_ranges)
    if args.cluster_mad_k is not None:
        params_kwargs["cluster_mad_k"] = args.cluster_mad_k
    if args.cluster_mad_floor is not None:
        params_kwargs["cluster_mad_floor_deg"] = args.cluster_mad_floor
    if args.cluster_min_lines is not None:
        params_kwargs["cluster_min_lines"] = args.cluster_min_lines
    if args.cluster_max_short_axis is not None:
        params_kwargs["cluster_max_short_axis_px"] = args.cluster_max_short_axis
    params = ScratchDetectorParams(**params_kwargs)
    cleaned = paths.reference_scratch_cleaned_path(ref_path)
    provenance = paths.reference_scratch_cleaned_provenance_path(ref_path)
    print(f"  [scratch_clean] reference:  {ref_path}")
    print(f"  [scratch_clean] cleaned:    {cleaned}")
    print(f"  [scratch_clean] provenance: {provenance}")
    print(f"  [scratch_clean] params:     {params}")

    out = generate_scratch_cleaned_reference(ref_path, cleaned, provenance,
                                             params=params)
    if out is None:
        print("ERROR: cleaning failed", file=sys.stderr)
        return 1

    if args.write_mask:
        mask_path = os.path.splitext(ref_path)[0] + ".scratch_mask.tif"
        _write_mask(ref_path, mask_path)

    print(f"\nDone. The next alignment run on this reference will read "
          f"the cleaned sidecar transparently.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
