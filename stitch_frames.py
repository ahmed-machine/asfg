#!/usr/bin/env python3
"""
Stitch consecutive satellite frames into a single panoramic strip.

Thin CLI entry point — delegates to declass.stitch for frame ordering,
rotation detection, overlap computation, and VRT-based stitching.

Usage:
    python3 stitch_frames.py frame1.tif frame2.tif [frame3.tif ...] -o output.tif
    python3 stitch_frames.py frame1.tif frame2.tif -o output.tif --preserve-order
"""

import argparse
import os
import sys
import tempfile

from declass.stitch import stitch_frames


def main():
    parser = argparse.ArgumentParser(description="Stitch consecutive satellite frames")
    parser.add_argument("frames", nargs="+", help="Input TIF frames")
    parser.add_argument("-o", "--output", required=True, help="Output TIF path")
    parser.add_argument("--preserve-order", action="store_true",
                        help="Use input order instead of auto-detecting frame sequence")
    args = parser.parse_args()

    if len(args.frames) < 2:
        print("Need at least 2 frames to stitch")
        sys.exit(1)

    for f in args.frames:
        if not os.path.exists(f):
            print(f"ERROR: File not found: {f}")
            sys.exit(1)

    output_dir = tempfile.mkdtemp(prefix="stitch_")
    print(f"Stitching {len(args.frames)} frames...")
    print(f"  Working dir: {output_dir}")

    result = stitch_frames(
        args.frames, args.output, output_dir,
        preserve_order=args.preserve_order,
    )
    print(f"Done: {result}")


if __name__ == "__main__":
    main()
