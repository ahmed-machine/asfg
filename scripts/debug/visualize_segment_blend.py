#!/usr/bin/env python3
"""Diagnose per-segment mosaic alignment.

Loads per-segment orthos and their blended mosaic, measures per-seam
pixel-grid offsets (should be 0 after the grid-snap fix), and writes a
downsampled PNG overlay so we can eyeball where segments transition.

Usage
-----
    python scripts/debug/visualize_segment_blend.py \\
        output/ortho/D3C1213-200346A003_segments/ \\
        --output diagnostics/kh9_blend_grid_snap/seg_blend.png \\
        --downsample 8

The segments folder must contain ``*_seg*_ortho.tif`` plus a mosaic
named ``*_per_segment.tif`` (or ``*_per_segment.vrt``). Segment IDs are
inferred from the file names (``seg00``, ``seg01``, ...).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np


def _load_segments(segdir: Path):
    seg_rx = re.compile(r"_seg(\d+)_ortho\.tif$")
    segs = []
    for p in sorted(segdir.glob("*_seg*_ortho.tif")):
        m = seg_rx.search(p.name)
        if m:
            segs.append((int(m.group(1)), p))
    segs.sort()
    return segs


def _find_mosaic(segdir: Path) -> Path | None:
    for pat in ("*_per_segment.tif", "*_per_segment.vrt"):
        hits = sorted(segdir.glob(pat))
        if hits:
            return hits[-1]
    return None


def _pixel_grid_offset(a_bounds, b_bounds, res_x: float, res_y: float):
    """Sub-pixel offset between two segments on a common grid anchored at (0,0)."""
    frac_col_a = ((a_bounds.left - 0.0) / res_x) % 1.0
    frac_col_b = ((b_bounds.left - 0.0) / res_x) % 1.0
    frac_row_a = ((0.0 - a_bounds.top) / res_y) % 1.0
    frac_row_b = ((0.0 - b_bounds.top) / res_y) % 1.0
    # Signed min-distance on the circular [-0.5, 0.5] axis.
    def _wrap(x):
        x = x % 1.0
        return x - 1.0 if x > 0.5 else x
    return _wrap(frac_col_a - frac_col_b), _wrap(frac_row_a - frac_row_b)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("segdir", type=Path,
                    help="Per-segment output directory (contains *_seg*_ortho.tif)")
    ap.add_argument("--output", type=Path, default=None,
                    help="Destination PNG (default: <segdir>/seg_blend_overlay.png)")
    ap.add_argument("--downsample", type=int, default=8,
                    help="Downsample factor for the preview PNG (default 8)")
    args = ap.parse_args()

    import rasterio

    segdir = args.segdir.resolve()
    if not segdir.exists():
        sys.exit(f"Segments dir not found: {segdir}")
    out_png = args.output or (segdir / "seg_blend_overlay.png")

    segs = _load_segments(segdir)
    if not segs:
        sys.exit(f"No *_seg*_ortho.tif in {segdir}")
    mosaic = _find_mosaic(segdir)
    if mosaic is None:
        sys.exit(f"No *_per_segment.(tif|vrt) in {segdir}")

    print(f"[diag] Found {len(segs)} segments + mosaic {mosaic.name}")

    # Per-seam sub-pixel offset (should be 0.000 after grid-snap fix).
    with rasterio.open(segs[0][1]) as ref_ds:
        res_x, res_y = ref_ds.res
    print(f"[diag] Pixel size (from seg00): {res_x:.6f} × {res_y:.6f} m")

    for i in range(len(segs) - 1):
        (a_id, a_path), (b_id, b_path) = segs[i], segs[i + 1]
        with rasterio.open(a_path) as a, rasterio.open(b_path) as b:
            dx, dy = _pixel_grid_offset(a.bounds, b.bounds, res_x, res_y)
        print(f"[diag] seam {a_id:02d}→{b_id:02d} sub-pixel offset: "
              f"Δcol={dx:+.4f}px  Δrow={dy:+.4f}px  "
              f"({'PASS' if abs(dx) < 1e-3 and abs(dy) < 1e-3 else 'FAIL — grid-snap needed'})")

    # Build the preview PNG: downsampled mosaic with segment outlines.
    print(f"[diag] Rendering preview → {out_png}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        sys.exit("matplotlib not installed — skip preview (install or re-run diagnostic-only)")

    with rasterio.open(mosaic) as ds:
        ds_w, ds_h = ds.width // args.downsample, ds.height // args.downsample
        preview = ds.read(
            1,
            out_shape=(ds_h, ds_w),
            resampling=rasterio.enums.Resampling.average,
        )
        preview_bounds = ds.bounds
        preview_nodata = ds.nodata if ds.nodata is not None else 0
    valid = preview != preview_nodata
    if not valid.any():
        sys.exit("Mosaic is all-nodata; cannot preview")

    lo = float(np.percentile(preview[valid], 2.0))
    hi = float(np.percentile(preview[valid], 98.0))
    norm = np.clip((preview - lo) / max(1e-6, hi - lo), 0, 1)
    norm[~valid] = 0

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(
        norm, cmap="gray", interpolation="nearest",
        extent=(preview_bounds.left, preview_bounds.right,
                preview_bounds.bottom, preview_bounds.top),
    )
    seg_colors = ["#ff4444", "#44ff44", "#4488ff", "#ffcc00", "#cc44ff",
                  "#44ffff", "#ff88cc"]
    for idx, (seg_id, path) in enumerate(segs):
        with rasterio.open(path) as sd:
            b = sd.bounds
        color = seg_colors[idx % len(seg_colors)]
        rect = Rectangle(
            (b.left, b.bottom), b.right - b.left, b.top - b.bottom,
            fill=False, edgecolor=color, linewidth=2, linestyle="--",
            label=f"seg{seg_id:02d}",
        )
        ax.add_patch(rect)
        ax.text(
            (b.left + b.right) / 2, b.top, f" seg{seg_id:02d}",
            color=color, fontsize=10, fontweight="bold",
            va="bottom", ha="center",
        )
    ax.set_aspect("equal")
    ax.set_xlabel("EPSG:3857 X (m)")
    ax.set_ylabel("EPSG:3857 Y (m)")
    ax.set_title(
        f"Per-segment mosaic + segment boundaries\n"
        f"{mosaic.name} ({ds_w * args.downsample}×{ds_h * args.downsample}px → "
        f"preview {ds_w}×{ds_h}px, downsample={args.downsample}×)",
        fontsize=10,
    )
    ax.legend(loc="upper right", framealpha=0.7)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[diag] Preview: {out_png}")


if __name__ == "__main__":
    main()
